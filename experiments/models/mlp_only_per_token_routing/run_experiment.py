#!/usr/bin/env python3
"""
MLP-Only Per-Token Routing: Eliminate cross-attention contamination.

Hypothesis: Applying LoRA adapters ONLY to MLP layers per-token within full-sequence
forward passes eliminates cross-attention contamination while preserving causal context.

Kill criteria:
  K790: MLP-only per-token PPL < per-sequence best PPL on mixed-domain sequences
  K791: MLP-only per-token PPL < segment-isolated PPL (Finding #305 baseline)
  K792: Cross-attention contamination eliminated — per-token-MLP != per-sequence

Prior results (Finding #305):
  per_seq_best:   4.8147
  per_token_full: 4.8147 (null — identical to per-sequence)
  seg_oracle:     4.0542
  seg_router:     4.0420
  base_only:      5.5213

Platform: Apple M5 Pro 48GB, MLX.
"""

import gc
import json
import math
import os
import random
import time
from pathlib import Path
from collections import defaultdict
from itertools import combinations

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse existing adapters and data from prior experiments
ADAPTER_DIR = Path(__file__).parent.parent / "tiny_routing_heads" / "adapters"
DATA_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
SEGMENT_LENGTH = 128
N_SEQUENCES_PER_PAIR = 20
SEED = 42

DOMAINS = ["python", "math", "medical", "legal", "creative"]
N_DOMAINS = len(DOMAINS)

# Module classification
ATTN_KEYS = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"}
MLP_KEYS = {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ===========================================================================
# Model loading utilities
# ===========================================================================
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.bitlinear_layers import BitLinear


def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    w0 = (packed_weights & 3).astype(mx.bfloat16) - 1
    w1 = ((packed_weights >> 2) & 3).astype(mx.bfloat16) - 1
    w2 = ((packed_weights >> 4) & 3).astype(mx.bfloat16) - 1
    w3 = ((packed_weights >> 6) & 3).astype(mx.bfloat16) - 1
    unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
    scale = weight_scale.astype(mx.bfloat16)
    if invert_scale:
        unpacked = unpacked / scale
    else:
        unpacked = unpacked * scale
    return unpacked


def replace_bitlinear_with_linear(model):
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                unpacked_w = unpack_ternary(
                    module.weight, module.out_features,
                    module.weight_scale, module.invert_weight_scales,
                )
                has_bias = module.bias is not None
                linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
                linear.weight = unpacked_w
                if has_bias:
                    linear.bias = module.bias
                updates.append((key, linear))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    log(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


def apply_lora_to_model(model, rank=16, scale=1.0):
    """Apply LoRA to ALL target layers (attn + MLP)."""
    target_keys = ATTN_KEYS | MLP_KEYS
    count = 0
    for layer in model.model.layers:
        lora_updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora_layer = LoRALinear.from_base(module, r=rank, scale=scale, dropout=0.0)
                lora_updates.append((key, lora_layer))
                count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    log(f"  Applied LoRA (r={rank}) to {count} linear layers")
    return model


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def split_adapter_params(adapter_params):
    """Split adapter params into attention-only and MLP-only subsets."""
    attn_params = {}
    mlp_params = {}
    for key, val in adapter_params.items():
        is_attn = any(ak in key for ak in ["self_attn.q_proj", "self_attn.k_proj",
                                             "self_attn.v_proj", "self_attn.o_proj"])
        is_mlp = any(mk in key for mk in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
        if is_attn:
            attn_params[key] = val
        elif is_mlp:
            mlp_params[key] = val
    return attn_params, mlp_params


def apply_adapter_to_model(model, adapter_params):
    model.update(tree_unflatten(list(adapter_params.items())))


def zero_adapter_in_model(model):
    updates = []
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_b" in name:
            updates.append((name, mx.zeros_like(p)))
    if updates:
        model.update(tree_unflatten(updates))


def zero_attn_adapter(model):
    """Zero only attention LoRA parameters, keep MLP LoRA intact."""
    updates = []
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_b" in name and "self_attn" in name:
            updates.append((name, mx.zeros_like(p)))
    if updates:
        model.update(tree_unflatten(updates))


def zero_mlp_adapter(model):
    """Zero only MLP LoRA parameters, keep attention LoRA intact."""
    updates = []
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_b" in name and "mlp" in name:
            updates.append((name, mx.zeros_like(p)))
    if updates:
        model.update(tree_unflatten(updates))


def compose_adapters(adapter_list, weights=None):
    """Merge multiple adapter parameter dicts with given weights (default 1/N)."""
    N = len(adapter_list)
    if weights is None:
        weights = [1.0 / N] * N
    merged = {}
    for key in adapter_list[0].keys():
        merged[key] = sum(adapter_list[i][key] * weights[i] for i in range(N))
    return merged


def load_domain_texts(domain_name, split="valid"):
    fpath = DATA_DIR / domain_name / f"{split}.jsonl"
    if not fpath.exists():
        return []
    texts = []
    with open(fpath) as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
    return texts


def compute_full_seq_ppl(model, tokens):
    """Compute PPL on a full sequence. Returns (total_nll, n_tokens)."""
    if len(tokens) < 2:
        return 0.0, 0
    x = mx.array(tokens[:-1])[None, :]
    y = mx.array(tokens[1:])[None, :]
    logits = model(x)
    loss = nn.losses.cross_entropy(logits, y, reduction="sum")
    mx.eval(loss)
    nll = loss.item()
    n = y.size
    del x, y, logits, loss
    return nll, n


def compute_segment_ppl(model, tokens):
    """Compute PPL on an isolated segment. Returns (total_nll, n_tokens)."""
    return compute_full_seq_ppl(model, tokens)


def compute_per_token_nll(model, tokens):
    """Compute per-token NLL for a full sequence. Returns array of shape (n_tokens,)."""
    if len(tokens) < 2:
        return mx.array([]), 0
    x = mx.array(tokens[:-1])[None, :]
    y = mx.array(tokens[1:])[None, :]
    logits = model(x)
    # Per-token cross entropy (no reduction)
    per_token_ce = nn.losses.cross_entropy(logits, y, reduction="none")  # (1, T)
    mx.eval(per_token_ce)
    result = per_token_ce[0]  # (T,)
    del x, y, logits
    return result, per_token_ce.size


def compute_mlp_only_per_token_ppl(model, tokens, domain_assignments, adapters,
                                    boundary_pos):
    """
    Compute PPL with MLP-only per-token routing on full sequences.

    Strategy:
    1. For each domain adapter, apply ONLY its MLP params (zero attention LoRA)
    2. Compute full-sequence forward pass (attention uses base weights)
    3. Get per-token NLL
    4. Select per-token NLL from the correct domain's adapter

    This is O(N_domains) forward passes but guarantees:
    - Attention uses base weights for all tokens (no contamination)
    - MLP uses domain-specific adapter for evaluated tokens
    - Full causal context preserved

    Args:
        model: the model with LoRA layers applied
        tokens: full sequence token ids
        domain_assignments: list of (start_idx, end_idx, domain_name) for prediction tokens
        adapters: dict of domain -> adapter_params
        boundary_pos: position where domain switches
    Returns:
        (total_nll, n_tokens)
    """
    if len(tokens) < 2:
        return 0.0, 0

    n_pred_tokens = len(tokens) - 1  # prediction targets

    # For each domain, run full-sequence forward with MLP-only adapter
    domain_nlls = {}
    for domain_name in DOMAINS:
        # Apply full adapter first, then zero attention parts
        apply_adapter_to_model(model, adapters[domain_name])
        zero_attn_adapter(model)

        # Forward pass: attention uses base weights, MLP uses this domain's adapter
        per_tok_nll, _ = compute_per_token_nll(model, tokens)
        mx.eval(per_tok_nll)
        domain_nlls[domain_name] = per_tok_nll
        zero_adapter_in_model(model)

    # Select per-token NLL based on domain assignment
    # Prediction token i predicts tokens[i+1], so position i in nll array
    # corresponds to input position i, predicting output position i+1
    # Domain assignment is by INPUT token position:
    #   positions 0..boundary_pos-1 -> domain_a
    #   positions boundary_pos..end -> domain_b
    # For prediction tokens: input positions 0..n_pred_tokens-1

    total_nll = 0.0
    n = 0
    for start_idx, end_idx, domain_name in domain_assignments:
        # Clip to prediction token range
        s = max(0, start_idx)
        e = min(end_idx, n_pred_tokens)
        if s >= e:
            continue
        segment_nll = mx.sum(domain_nlls[domain_name][s:e])
        mx.eval(segment_nll)
        total_nll += segment_nll.item()
        n += (e - s)

    # Cleanup
    for v in domain_nlls.values():
        del v
    del domain_nlls

    return total_nll, n


def compute_full_module_per_token_ppl(model, tokens, domain_assignments, adapters,
                                       boundary_pos):
    """
    Same as MLP-only but with FULL adapter (attn + MLP) per domain.
    This replicates the Finding #305 null result for comparison.
    """
    if len(tokens) < 2:
        return 0.0, 0

    n_pred_tokens = len(tokens) - 1

    domain_nlls = {}
    for domain_name in DOMAINS:
        apply_adapter_to_model(model, adapters[domain_name])
        per_tok_nll, _ = compute_per_token_nll(model, tokens)
        mx.eval(per_tok_nll)
        domain_nlls[domain_name] = per_tok_nll
        zero_adapter_in_model(model)

    total_nll = 0.0
    n = 0
    for start_idx, end_idx, domain_name in domain_assignments:
        s = max(0, start_idx)
        e = min(end_idx, n_pred_tokens)
        if s >= e:
            continue
        segment_nll = mx.sum(domain_nlls[domain_name][s:e])
        mx.eval(segment_nll)
        total_nll += segment_nll.item()
        n += (e - s)

    for v in domain_nlls.values():
        del v
    del domain_nlls

    return total_nll, n


def compute_attn_only_per_token_ppl(model, tokens, domain_assignments, adapters,
                                     boundary_pos):
    """
    Attention-only per-token routing: apply only attention LoRA per domain.
    This is the OPPOSITE of MLP-only — included for completeness and to
    verify that attention-only produces the contamination (control condition).
    """
    if len(tokens) < 2:
        return 0.0, 0

    n_pred_tokens = len(tokens) - 1

    domain_nlls = {}
    for domain_name in DOMAINS:
        apply_adapter_to_model(model, adapters[domain_name])
        zero_mlp_adapter(model)  # Zero MLP, keep attention

        per_tok_nll, _ = compute_per_token_nll(model, tokens)
        mx.eval(per_tok_nll)
        domain_nlls[domain_name] = per_tok_nll
        zero_adapter_in_model(model)

    total_nll = 0.0
    n = 0
    for start_idx, end_idx, domain_name in domain_assignments:
        s = max(0, start_idx)
        e = min(end_idx, n_pred_tokens)
        if s >= e:
            continue
        segment_nll = mx.sum(domain_nlls[domain_name][s:e])
        mx.eval(segment_nll)
        total_nll += segment_nll.item()
        n += (e - s)

    for v in domain_nlls.values():
        del v
    del domain_nlls

    return total_nll, n


# ===========================================================================
# Phase 0: Construct mixed-domain evaluation data (same as Finding #305)
# ===========================================================================
def phase_construct_mixed_data(tokenizer):
    log("\n" + "=" * 70)
    log("[Phase 0] Constructing mixed-domain evaluation sequences")
    log("=" * 70)

    rng = random.Random(SEED)

    domain_texts = {}
    for domain in DOMAINS:
        domain_texts[domain] = load_domain_texts(domain, split="valid")
        log(f"  {domain}: {len(domain_texts[domain])} validation texts")

    domain_pairs = list(combinations(DOMAINS, 2))
    mixed_sequences = []

    for domain_a, domain_b in domain_pairs:
        texts_a = domain_texts[domain_a]
        texts_b = domain_texts[domain_b]
        pair_count = 0

        for _ in range(N_SEQUENCES_PER_PAIR * 3):
            if pair_count >= N_SEQUENCES_PER_PAIR:
                break

            text_a = texts_a[rng.randint(0, len(texts_a) - 1)]
            text_b = texts_b[rng.randint(0, len(texts_b) - 1)]

            toks_a = tokenizer.encode(text_a)
            toks_b = tokenizer.encode(text_b)

            if len(toks_a) < SEGMENT_LENGTH or len(toks_b) < SEGMENT_LENGTH:
                while len(toks_a) < SEGMENT_LENGTH:
                    toks_a = toks_a + toks_a
                while len(toks_b) < SEGMENT_LENGTH:
                    toks_b = toks_b + toks_b

            seg_a = toks_a[:SEGMENT_LENGTH]
            seg_b = toks_b[:SEGMENT_LENGTH]
            combined = seg_a + seg_b

            mixed_sequences.append({
                "tokens": combined,
                "seg_a_tokens": seg_a,
                "seg_b_tokens": seg_b,
                "domain_a": domain_a,
                "domain_b": domain_b,
                "domain_a_idx": DOMAINS.index(domain_a),
                "domain_b_idx": DOMAINS.index(domain_b),
                "boundary_pos": SEGMENT_LENGTH,
                "n_tokens": len(combined),
            })
            pair_count += 1

        log(f"  {domain_a}+{domain_b}: {pair_count} sequences")

    log(f"  Total mixed sequences: {len(mixed_sequences)}")
    log(f"  Avg length: {sum(s['n_tokens'] for s in mixed_sequences)/len(mixed_sequences):.0f} tokens")

    min_per_pair = min(
        sum(1 for s in mixed_sequences if s['domain_a'] == da and s['domain_b'] == db)
        for da, db in domain_pairs
    )
    log(f"  Min sequences per pair: {min_per_pair}")
    k774_pass = min_per_pair >= 10

    return mixed_sequences, k774_pass


# ===========================================================================
# Phase 1: Evaluate all routing strategies
# ===========================================================================
def phase_evaluate(model_id, mixed_sequences):
    """
    Compare routing strategies on mixed-domain sequences.

    Strategies:
    1. base_only: no adapter
    2. per_seq_best: best single adapter for full sequence (oracle)
    3. per_token_full: per-token full-module routing (Finding #305 null)
    4. per_token_mlp_only: per-token MLP-only routing (NEW)
    5. per_token_attn_only: per-token attention-only routing (control)
    6. seg_oracle: segment-isolated with correct adapter (Finding #305)
    7. seg_router: segment-isolated with best-PPL adapter (Finding #305)
    """
    log("\n" + "=" * 70)
    log("[Phase 1] Evaluating routing strategies")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(model_id)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)
    log_memory("model-loaded")

    # Load all adapters
    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(ADAPTER_DIR / domain)
        log(f"  Loaded adapter: {domain}")

    # Pre-split adapters into MLP-only and attn-only
    mlp_adapters = {}
    attn_adapters = {}
    for domain in DOMAINS:
        attn_params, mlp_params = split_adapter_params(adapters[domain])
        mlp_adapters[domain] = mlp_params
        attn_adapters[domain] = attn_params

    # Group by pair
    pair_sequences = defaultdict(list)
    for seq in mixed_sequences:
        pair_key = f"{seq['domain_a']}+{seq['domain_b']}"
        pair_sequences[pair_key].append(seq)

    # Strategy names
    strategy_names = [
        "base_only", "per_seq_best", "per_token_full",
        "per_token_mlp_only", "per_token_attn_only",
        "seg_oracle", "seg_router",
    ]

    # Global NLL accumulators
    global_stats = {s: {"nll": 0.0, "n": 0} for s in strategy_names}
    pair_results = {}

    for pair_key, sequences in pair_sequences.items():
        log(f"\n  === Pair: {pair_key} ({len(sequences)} sequences) ===")

        stats = {s: {"nll": 0.0, "n": 0} for s in strategy_names}

        for seq_idx, seq_data in enumerate(sequences):
            tokens = seq_data["tokens"]
            seg_a_tokens = seq_data["seg_a_tokens"]
            seg_b_tokens = seq_data["seg_b_tokens"]
            domain_a = seq_data["domain_a"]
            domain_b = seq_data["domain_b"]
            boundary = seq_data["boundary_pos"]

            if len(tokens) < 4:
                continue

            # Domain assignments for per-token routing:
            # Input positions 0..boundary-1 -> domain_a
            # Input positions boundary..end -> domain_b
            domain_assignments = [
                (0, boundary, domain_a),
                (boundary, len(tokens) - 1, domain_b),
            ]

            # ---- Strategy 0: Base model only ----
            zero_adapter_in_model(model)
            nll, n = compute_full_seq_ppl(model, tokens)
            stats["base_only"]["nll"] += nll
            stats["base_only"]["n"] += n

            # ---- Strategy 1: Per-sequence best single adapter (oracle) ----
            best_nll = float("inf")
            best_adapter_name = None
            for d_name in DOMAINS:
                apply_adapter_to_model(model, adapters[d_name])
                nll_d, n_d = compute_full_seq_ppl(model, tokens)
                if nll_d < best_nll:
                    best_nll = nll_d
                    best_adapter_name = d_name
                zero_adapter_in_model(model)
            stats["per_seq_best"]["nll"] += best_nll
            stats["per_seq_best"]["n"] += n_d

            # ---- Strategy 2: Per-token full-module (Finding #305 null) ----
            nll_full, n_full = compute_full_module_per_token_ppl(
                model, tokens, domain_assignments, adapters, boundary)
            stats["per_token_full"]["nll"] += nll_full
            stats["per_token_full"]["n"] += n_full

            # ---- Strategy 3: Per-token MLP-only (NEW HYPOTHESIS) ----
            nll_mlp, n_mlp = compute_mlp_only_per_token_ppl(
                model, tokens, domain_assignments, adapters, boundary)
            stats["per_token_mlp_only"]["nll"] += nll_mlp
            stats["per_token_mlp_only"]["n"] += n_mlp

            # ---- Strategy 4: Per-token attn-only (control) ----
            nll_attn, n_attn = compute_attn_only_per_token_ppl(
                model, tokens, domain_assignments, adapters, boundary)
            stats["per_token_attn_only"]["nll"] += nll_attn
            stats["per_token_attn_only"]["n"] += n_attn

            # ---- Strategy 5 & 6: Segment-isolated (from Finding #305) ----
            seg_a_nlls = {}
            seg_b_nlls = {}
            n_a = n_b = None

            for d_name in DOMAINS:
                apply_adapter_to_model(model, adapters[d_name])
                nll_a_d, n_a_d = compute_segment_ppl(model, seg_a_tokens)
                nll_b_d, n_b_d = compute_segment_ppl(model, seg_b_tokens)
                seg_a_nlls[d_name] = nll_a_d
                seg_b_nlls[d_name] = nll_b_d
                if n_a is None:
                    n_a = n_a_d
                    n_b = n_b_d
                zero_adapter_in_model(model)

            # Segment oracle
            oracle_nll = seg_a_nlls[domain_a] + seg_b_nlls[domain_b]
            stats["seg_oracle"]["nll"] += oracle_nll
            stats["seg_oracle"]["n"] += n_a + n_b

            # Segment router (best PPL per segment)
            best_a_name = min(seg_a_nlls, key=seg_a_nlls.get)
            best_b_name = min(seg_b_nlls, key=seg_b_nlls.get)
            router_nll = seg_a_nlls[best_a_name] + seg_b_nlls[best_b_name]
            stats["seg_router"]["nll"] += router_nll
            stats["seg_router"]["n"] += n_a + n_b

            if (seq_idx + 1) % 5 == 0:
                log(f"    Processed {seq_idx+1}/{len(sequences)} sequences")

            gc.collect()
            mx.clear_cache()

        # Compute PPLs for this pair
        pair_result = {}
        for s in strategy_names:
            if stats[s]["n"] > 0:
                pair_result[f"{s}_ppl"] = round(math.exp(stats[s]["nll"] / stats[s]["n"]), 4)
            else:
                pair_result[f"{s}_ppl"] = float("inf")
        pair_result["n_sequences"] = len(sequences)
        pair_result["_raw_stats"] = {s: dict(v) for s, v in stats.items()}
        pair_results[pair_key] = pair_result

        # Accumulate global stats
        for s in strategy_names:
            global_stats[s]["nll"] += stats[s]["nll"]
            global_stats[s]["n"] += stats[s]["n"]

        log(f"    per_seq_best={pair_result['per_seq_best_ppl']:.3f}, "
            f"per_tok_full={pair_result['per_token_full_ppl']:.3f}, "
            f"per_tok_mlp={pair_result['per_token_mlp_only_ppl']:.3f}, "
            f"per_tok_attn={pair_result['per_token_attn_only_ppl']:.3f}, "
            f"seg_oracle={pair_result['seg_oracle_ppl']:.3f}")

    elapsed = time.time() - t0
    log(f"\n  Total evaluation time: {elapsed:.1f}s")
    log_memory("post-eval")

    cleanup(model, tokenizer, adapters, mlp_adapters, attn_adapters)
    return pair_results, global_stats


# ===========================================================================
# Phase 2: Analyze results and evaluate kill criteria
# ===========================================================================
def phase_analyze(pair_results, global_stats):
    log("\n" + "=" * 70)
    log("[Phase 2] Analyzing results")
    log("=" * 70)

    strategy_names = [
        "base_only", "per_seq_best", "per_token_full",
        "per_token_mlp_only", "per_token_attn_only",
        "seg_oracle", "seg_router",
    ]

    avg_ppls = {}
    for s in strategy_names:
        if global_stats[s]["n"] > 0:
            avg_ppls[s] = round(math.exp(global_stats[s]["nll"] / global_stats[s]["n"]), 4)
        else:
            avg_ppls[s] = float("inf")

    log("\n  Global average PPL (NLL-weighted across all pairs):")
    for s in strategy_names:
        marker = ""
        if s == "per_token_mlp_only":
            marker = " <-- NEW"
        log(f"    {s:25s}: {avg_ppls[s]:.4f}{marker}")

    # Kill criteria evaluation
    per_seq_ppl = avg_ppls["per_seq_best"]
    mlp_only_ppl = avg_ppls["per_token_mlp_only"]
    seg_iso_ppl = avg_ppls["seg_router"]  # Use seg_router as the Finding #305 baseline
    full_module_ppl = avg_ppls["per_token_full"]
    attn_only_ppl = avg_ppls["per_token_attn_only"]

    # K790: MLP-only per-token PPL < per-sequence best
    k790_pass = mlp_only_ppl < per_seq_ppl
    k790_delta = (per_seq_ppl - mlp_only_ppl) / per_seq_ppl * 100

    # K791: MLP-only per-token PPL < segment-isolated
    k791_pass = mlp_only_ppl < seg_iso_ppl
    k791_delta = (seg_iso_ppl - mlp_only_ppl) / seg_iso_ppl * 100

    # K792: per-token-MLP != per-sequence (contamination eliminated)
    k792_diff = abs(mlp_only_ppl - per_seq_ppl)
    k792_pass = k792_diff > 0.01

    # Additional analysis: does full-module per-token still == per-sequence?
    full_vs_seq_diff = abs(full_module_ppl - per_seq_ppl)
    full_module_null = full_vs_seq_diff < 0.01

    log(f"\n  === Kill Criteria ===")
    log(f"  K790: MLP-only ({mlp_only_ppl:.4f}) < per-seq ({per_seq_ppl:.4f})? "
        f"Delta {k790_delta:+.2f}% -> {'PASS' if k790_pass else 'FAIL'}")
    log(f"  K791: MLP-only ({mlp_only_ppl:.4f}) < seg-isolated ({seg_iso_ppl:.4f})? "
        f"Delta {k791_delta:+.2f}% -> {'PASS' if k791_pass else 'FAIL'}")
    log(f"  K792: |MLP-only - per-seq| = {k792_diff:.4f} > 0.01? "
        f"-> {'PASS' if k792_pass else 'FAIL'}")

    log(f"\n  === Control checks ===")
    log(f"  Full-module per-token == per-seq? |diff| = {full_vs_seq_diff:.4f} "
        f"({'YES (null confirmed)' if full_module_null else 'NO'})")
    log(f"  Attn-only per-token: {attn_only_ppl:.4f} "
        f"(should be ~= per-seq if contamination is the issue)")

    # Per-pair breakdown
    pair_improvements = {}
    for pair_key, result in pair_results.items():
        ps = result["per_seq_best_ppl"]
        mlp = result["per_token_mlp_only_ppl"]
        seg = result["seg_oracle_ppl"]
        full = result["per_token_full_ppl"]
        attn = result["per_token_attn_only_ppl"]
        pair_improvements[pair_key] = {
            "per_seq_best_ppl": ps,
            "per_token_mlp_only_ppl": mlp,
            "per_token_full_ppl": full,
            "per_token_attn_only_ppl": attn,
            "seg_oracle_ppl": seg,
            "seg_router_ppl": result["seg_router_ppl"],
            "mlp_vs_seq_pct": round((ps - mlp) / ps * 100, 2) if ps > 0 else 0,
            "mlp_vs_seg_pct": round((result["seg_router_ppl"] - mlp) / result["seg_router_ppl"] * 100, 2) if result["seg_router_ppl"] > 0 else 0,
            "full_vs_seq_diff": round(abs(full - ps), 4),
        }

    return {
        "avg_ppls": avg_ppls,
        "k790_mlp_only_ppl": mlp_only_ppl,
        "k790_per_seq_ppl": per_seq_ppl,
        "k790_delta_pct": round(k790_delta, 2),
        "k790_pass": k790_pass,
        "k791_mlp_only_ppl": mlp_only_ppl,
        "k791_seg_iso_ppl": seg_iso_ppl,
        "k791_delta_pct": round(k791_delta, 2),
        "k791_pass": k791_pass,
        "k792_diff": round(k792_diff, 4),
        "k792_pass": k792_pass,
        "full_module_null_confirmed": full_module_null,
        "full_vs_seq_diff": round(full_vs_seq_diff, 4),
        "pair_improvements": pair_improvements,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_start = time.time()
    log("=" * 70)
    log("MLP-Only Per-Token Routing: Eliminate Cross-Attention Contamination")
    log("=" * 70)
    log(f"Model: {MODEL_ID}")
    log(f"Domains: {DOMAINS}")
    log(f"Adapters: {ADAPTER_DIR}")
    log(f"Segment length: {SEGMENT_LENGTH}")
    log(f"Sequences per pair: {N_SEQUENCES_PER_PAIR}")
    log(f"LoRA scale: {LORA_SCALE}")
    log(f"LoRA rank: {LORA_RANK}")
    log_memory("start")

    # Phase 0: Construct data
    log("\n  Loading tokenizer...")
    _, tokenizer = load(MODEL_ID)
    mixed_sequences, k774_pass = phase_construct_mixed_data(tokenizer)
    del tokenizer
    gc.collect()

    # Phase 1: Evaluate all strategies
    pair_results, global_stats = phase_evaluate(MODEL_ID, mixed_sequences)

    # Phase 2: Analyze
    analysis = phase_analyze(pair_results, global_stats)

    # Compile results
    results = {
        "experiment": "mlp_only_per_token_routing",
        "type": "guided-exploration",
        "model": MODEL_ID,
        "n_domains": N_DOMAINS,
        "domains": DOMAINS,
        "n_domain_pairs": len(list(combinations(DOMAINS, 2))),
        "sequences_per_pair": N_SEQUENCES_PER_PAIR,
        "segment_length": SEGMENT_LENGTH,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "seed": SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pair_results": pair_results,
        "summary": analysis["avg_ppls"],
        "K790_mlp_only_ppl": analysis["k790_mlp_only_ppl"],
        "K790_per_seq_ppl": analysis["k790_per_seq_ppl"],
        "K790_delta_pct": analysis["k790_delta_pct"],
        "K790_pass": analysis["k790_pass"],
        "K791_mlp_only_ppl": analysis["k791_mlp_only_ppl"],
        "K791_seg_iso_ppl": analysis["k791_seg_iso_ppl"],
        "K791_delta_pct": analysis["k791_delta_pct"],
        "K791_pass": analysis["k791_pass"],
        "K792_diff": analysis["k792_diff"],
        "K792_pass": analysis["k792_pass"],
        "full_module_null_confirmed": analysis["full_module_null_confirmed"],
        "full_vs_seq_diff": analysis["full_vs_seq_diff"],
        "pair_improvements": analysis["pair_improvements"],
        "finding_305_baselines": {
            "per_seq_best": 4.8147,
            "per_token_full": 4.8147,
            "seg_oracle": 4.0542,
            "seg_router": 4.0420,
            "base_only": 5.5213,
        },
        "total_time_s": round(time.time() - t_start, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n  Results saved to {RESULTS_FILE}")

    # Final summary
    log("\n" + "=" * 70)
    log("FINAL RESULTS")
    log("=" * 70)

    for s_name, ppl in analysis["avg_ppls"].items():
        marker = " <-- NEW" if s_name == "per_token_mlp_only" else ""
        log(f"  {s_name:25s}: PPL {ppl:.4f}{marker}")

    log(f"\n  K790: MLP-only ({analysis['k790_mlp_only_ppl']:.4f}) < per-seq "
        f"({analysis['k790_per_seq_ppl']:.4f})? {analysis['k790_delta_pct']:+.2f}% "
        f"-> {'PASS' if analysis['k790_pass'] else 'FAIL'}")
    log(f"  K791: MLP-only ({analysis['k791_mlp_only_ppl']:.4f}) < seg-isolated "
        f"({analysis['k791_seg_iso_ppl']:.4f})? {analysis['k791_delta_pct']:+.2f}% "
        f"-> {'PASS' if analysis['k791_pass'] else 'FAIL'}")
    log(f"  K792: |MLP-only - per-seq| = {analysis['k792_diff']:.4f} > 0.01? "
        f"-> {'PASS' if analysis['k792_pass'] else 'FAIL'}")
    log(f"  Control: Full-module null confirmed? {analysis['full_module_null_confirmed']}")

    log(f"\n  Total time: {results['total_time_s']:.1f}s")
    log_memory("end")


if __name__ == "__main__":
    main()
