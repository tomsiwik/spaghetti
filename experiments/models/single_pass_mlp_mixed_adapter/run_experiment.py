#!/usr/bin/env python3
"""
Single-Pass MLP Mixed-Adapter Routing: Close Experiment-Proof Gap (Finding #312).

Hypothesis: Single-pass MLP-only mixed-adapter routing produces identical PPL
to multi-pass oracle, proving MLP token-independence holds in practice.

Kill criteria:
  K793: |PPL_single - PPL_multi| / PPL_multi < 1%
  K794: Single-pass MLP PPL < per-sequence best (4.815)
  K795: Per-token adapter assignments identical (trivially true by construction)

Finding #312 (provisional) showed MLP-only per-token = 4.656 (multi-pass oracle).
This experiment verifies the single-pass implementation matches.

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
# Model loading utilities (shared with prior experiment)
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


# ===========================================================================
# Core computation functions
# ===========================================================================

def compute_per_token_nll(model, tokens):
    """Compute per-token NLL for a full sequence. Returns array of shape (T,)."""
    if len(tokens) < 2:
        return mx.array([]), 0
    x = mx.array(tokens[:-1])[None, :]
    y = mx.array(tokens[1:])[None, :]
    logits = model(x)
    per_token_ce = nn.losses.cross_entropy(logits, y, reduction="none")  # (1, T)
    mx.eval(per_token_ce)
    result = per_token_ce[0]  # (T,)
    del x, y, logits
    return result, per_token_ce.size


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


# ===========================================================================
# Multi-pass MLP-only oracle (reproduces Finding #312 methodology)
# ===========================================================================

def compute_multipass_mlp_only_ppl(model, tokens, domain_assignments, adapters, boundary_pos):
    """
    Multi-pass MLP-only oracle: for each domain, run a full forward pass with
    MLP-only adapter (zero attention LoRA), then select per-token NLL from the
    correct domain. This is the Finding #312 methodology.
    """
    if len(tokens) < 2:
        return 0.0, 0, {}

    n_pred_tokens = len(tokens) - 1
    domain_nlls = {}

    for domain_name in DOMAINS:
        apply_adapter_to_model(model, adapters[domain_name])
        zero_attn_adapter(model)
        per_tok_nll, _ = compute_per_token_nll(model, tokens)
        mx.eval(per_tok_nll)
        domain_nlls[domain_name] = per_tok_nll
        zero_adapter_in_model(model)

    # Select per-token NLL based on domain assignment
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

    # Also return per-token NLL arrays for comparison
    per_token_selected = mx.zeros((n_pred_tokens,))
    for start_idx, end_idx, domain_name in domain_assignments:
        s = max(0, start_idx)
        e = min(end_idx, n_pred_tokens)
        if s >= e:
            continue
        per_token_selected = per_token_selected.at[s:e].add(domain_nlls[domain_name][s:e])
    mx.eval(per_token_selected)

    for v in domain_nlls.values():
        del v
    del domain_nlls

    return total_nll, n, per_token_selected


# ===========================================================================
# Single-pass MLP mixed-adapter (THE NEW THING)
# ===========================================================================

class MixedAdapterMLP:
    """
    A wrapper that performs single-pass mixed-adapter MLP computation.

    For each MLP LoRA layer (gate_proj, up_proj, down_proj), it computes:
      base_output = x @ W_base
      lora_A_output = scale * (x @ lora_a_A) @ lora_b_A   (for all tokens)
      lora_B_output = scale * (x @ lora_a_B) @ lora_b_B   (for all tokens)
      output[t] = base_output[t] + lora_A_output[t] if mask_A[t] else lora_B_output[t]

    This applies different adapters to different tokens in a SINGLE forward pass.
    """

    def __init__(self, model, adapter_params_A, adapter_params_B, boundary_pos, scale):
        """
        Pre-extract LoRA parameters for both adapters, organized by layer and module.

        adapter_params_A/B: dict with keys like 'model.layers.0.mlp.gate_proj.lora_a'
        """
        self.scale = scale
        self.boundary_pos = boundary_pos
        self.n_layers = len(model.model.layers)

        # Extract MLP LoRA params per layer for both adapters
        self.layer_params = []
        for l in range(self.n_layers):
            layer_dict = {}
            for module_name in ["gate_proj", "up_proj", "down_proj"]:
                key_a = f"model.layers.{l}.mlp.{module_name}.lora_a"
                key_b = f"model.layers.{l}.mlp.{module_name}.lora_b"
                layer_dict[module_name] = {
                    "lora_a_A": adapter_params_A[key_a],
                    "lora_b_A": adapter_params_A[key_b],
                    "lora_a_B": adapter_params_B[key_a],
                    "lora_b_B": adapter_params_B[key_b],
                }
            self.layer_params.append(layer_dict)

    def compute_mixed_lora_output(self, x, layer_idx, module_name):
        """
        Compute per-token LoRA output with adapter A for tokens < boundary
        and adapter B for tokens >= boundary.

        x: (1, T, d) input to the LoRA layer
        Returns: (1, T, d_out) LoRA delta (to be added to base output)
        """
        params = self.layer_params[layer_idx][module_name]

        # Compute LoRA output for both adapters on ALL tokens
        # x: (1, T, d_in), lora_a: (d_in, r), lora_b: (r, d_out)
        lora_out_A = (x @ params["lora_a_A"]) @ params["lora_b_A"]  # (1, T, d_out)
        lora_out_B = (x @ params["lora_a_B"]) @ params["lora_b_B"]  # (1, T, d_out)

        # Create mask: True for adapter A (tokens < boundary), False for adapter B
        T = x.shape[1]
        # mask shape: (1, T, 1) for broadcasting
        mask_A = mx.arange(T)[None, :, None] < self.boundary_pos  # (1, T, 1)

        # Select per-token: where mask_A, use A; else use B
        mixed = mx.where(mask_A, lora_out_A, lora_out_B)

        # Match LoRALinear: scale in float32, THEN cast to x.dtype
        return (self.scale * mixed).astype(x.dtype)


def single_pass_mixed_adapter_forward(model, tokens, mixed_mlp, boundary_pos):
    """
    Run a single forward pass where:
    - Attention uses BASE weights (zero attention LoRA)
    - MLP uses per-token adapter: adapter A for tokens 0..boundary-1,
      adapter B for tokens boundary..end

    Implementation: We monkey-patch each MLP LoRA layer's forward to use
    the mixed adapter computation, run one forward pass, then restore.

    Returns per-token NLL array.
    """
    if len(tokens) < 2:
        return mx.array([]), 0

    x_in = mx.array(tokens[:-1])[None, :]
    y = mx.array(tokens[1:])[None, :]

    # PRECONDITION: caller must have zeroed all LoRA (both attn and MLP)
    # Manual layer-by-layer forward, intercepting MLP to add per-token LoRA deltas
    h = model.model.embed_tokens(x_in)  # (1, T, d)

    # Use "causal" string mask (same as model's native forward)
    mask = "causal"

    for l in range(len(model.model.layers)):
        layer = model.model.layers[l]

        # --- Attention with base weights (LoRA is zeroed) ---
        r = layer.self_attn(layer.input_layernorm(h), mask, None)
        h_post_attn = h + r

        # --- MLP with per-token mixed adapter ---
        h_normed = layer.post_attention_layernorm(h_post_attn)

        # Base MLP computation (with zeroed LoRA, this is base only)
        # But we need to add the mixed LoRA delta manually

        # Get base MLP LoRA layers
        gate_mod = layer.mlp.gate_proj
        up_mod = layer.mlp.up_proj
        down_mod = layer.mlp.down_proj

        # Base outputs (LoRA is zeroed, so linear(x) + 0)
        gate_base = gate_mod(h_normed)  # includes base weight only (lora_b is zero)
        up_base = up_mod(h_normed)

        # Add mixed LoRA delta for gate and up
        gate_lora_delta = mixed_mlp.compute_mixed_lora_output(h_normed, l, "gate_proj")
        up_lora_delta = mixed_mlp.compute_mixed_lora_output(h_normed, l, "up_proj")

        gate_out = gate_base + gate_lora_delta
        up_out = up_base + up_lora_delta

        # SiLU-gated activation (BitNet uses relu2, which is nn.relu2 = relu(x)^2)
        # Actually BitNet-2B-4T MLP uses: relu2(gate) * up, then sub_norm, then down
        x_mid = nn.relu2(gate_out) * up_out

        # Sub-norm
        x_mid = layer.mlp.ffn_sub_norm(x_mid)

        # Down projection with mixed LoRA
        down_base = down_mod(x_mid)
        down_lora_delta = mixed_mlp.compute_mixed_lora_output(x_mid, l, "down_proj")
        mlp_out = down_base + down_lora_delta

        h = h_post_attn + mlp_out

    # Final layer norm + LM head (BitNet uses tied embeddings)
    h = model.model.norm(h)
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)

    per_token_ce = nn.losses.cross_entropy(logits, y, reduction="none")  # (1, T)
    mx.eval(per_token_ce)
    result = per_token_ce[0]  # (T,)
    del x_in, y, logits, h
    return result, per_token_ce.size


# ===========================================================================
# Data construction (same as prior experiment)
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
    return mixed_sequences


# ===========================================================================
# Phase 1: Compare single-pass vs multi-pass
# ===========================================================================

def phase_evaluate(model_id, mixed_sequences):
    log("\n" + "=" * 70)
    log("[Phase 1] Evaluating single-pass vs multi-pass MLP-only routing")
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

    # Pre-split adapters
    mlp_adapters = {}
    for domain in DOMAINS:
        _, mlp_params = split_adapter_params(adapters[domain])
        mlp_adapters[domain] = mlp_params

    # Group by pair
    pair_sequences = defaultdict(list)
    for seq in mixed_sequences:
        pair_key = f"{seq['domain_a']}+{seq['domain_b']}"
        pair_sequences[pair_key].append(seq)

    # Strategy names
    strategy_names = ["multi_pass_mlp", "single_pass_mlp", "per_seq_best", "base_only"]

    # Global NLL accumulators
    global_stats = {s: {"nll": 0.0, "n": 0} for s in strategy_names}
    pair_results = {}

    # Per-token NLL comparison tracking
    max_per_token_abs_diff = 0.0
    total_tokens_compared = 0
    per_token_diffs_all = []

    # Separate tracking: same-segment (tokens before boundary) vs cross-segment
    same_seg_diffs = []    # tokens 0..boundary-1: same adapter in both regimes
    cross_seg_diffs = []   # tokens boundary..end: different adapter context

    for pair_key, sequences in pair_sequences.items():
        log(f"\n  === Pair: {pair_key} ({len(sequences)} sequences) ===")

        stats = {s: {"nll": 0.0, "n": 0} for s in strategy_names}

        for seq_idx, seq_data in enumerate(sequences):
            tokens = seq_data["tokens"]
            domain_a = seq_data["domain_a"]
            domain_b = seq_data["domain_b"]
            boundary = seq_data["boundary_pos"]

            if len(tokens) < 4:
                continue

            n_pred_tokens = len(tokens) - 1

            # Domain assignments (for prediction tokens)
            domain_assignments = [
                (0, boundary, domain_a),
                (boundary, n_pred_tokens, domain_b),
            ]

            # ---- Strategy 0: Base model only ----
            zero_adapter_in_model(model)
            nll_base, n_base = compute_full_seq_ppl(model, tokens)
            stats["base_only"]["nll"] += nll_base
            stats["base_only"]["n"] += n_base

            # ---- Strategy 1: Per-sequence best (oracle) ----
            best_nll = float("inf")
            for d_name in DOMAINS:
                apply_adapter_to_model(model, adapters[d_name])
                nll_d, n_d = compute_full_seq_ppl(model, tokens)
                if nll_d < best_nll:
                    best_nll = nll_d
                zero_adapter_in_model(model)
            stats["per_seq_best"]["nll"] += best_nll
            stats["per_seq_best"]["n"] += n_d

            # ---- Strategy 2: Multi-pass MLP-only oracle (Finding #312) ----
            nll_multi, n_multi, multi_per_token = compute_multipass_mlp_only_ppl(
                model, tokens, domain_assignments, adapters, boundary)
            stats["multi_pass_mlp"]["nll"] += nll_multi
            stats["multi_pass_mlp"]["n"] += n_multi

            # ---- Strategy 3: SINGLE-PASS MLP mixed-adapter (NEW) ----
            # Build mixed adapter helper
            mixed_mlp = MixedAdapterMLP(
                model, mlp_adapters[domain_a], mlp_adapters[domain_b],
                boundary, LORA_SCALE
            )

            # Zero all LoRA, then run single-pass
            zero_adapter_in_model(model)
            single_per_token, n_single = single_pass_mixed_adapter_forward(
                model, tokens, mixed_mlp, boundary)
            mx.eval(single_per_token)

            # Compute single-pass total NLL using domain assignments
            single_nll = 0.0
            n_single_counted = 0
            for start_idx, end_idx, d_name in domain_assignments:
                s = max(0, start_idx)
                e = min(end_idx, n_pred_tokens)
                if s >= e:
                    continue
                seg_nll = mx.sum(single_per_token[s:e])
                mx.eval(seg_nll)
                single_nll += seg_nll.item()
                n_single_counted += (e - s)

            stats["single_pass_mlp"]["nll"] += single_nll
            stats["single_pass_mlp"]["n"] += n_single_counted

            # ---- Per-token comparison ----
            # Compare single-pass and multi-pass per-token NLL
            diff = mx.abs(single_per_token - multi_per_token)
            mx.eval(diff)
            max_diff = mx.max(diff).item()
            mean_diff = mx.mean(diff).item()

            if max_diff > max_per_token_abs_diff:
                max_per_token_abs_diff = max_diff

            per_token_diffs_all.append(mean_diff)
            total_tokens_compared += n_pred_tokens

            # Same-segment vs cross-segment analysis
            # Tokens 0..boundary-1 get adapter A in both regimes
            # In multi-pass (A pass), ALL tokens get adapter A
            # In single-pass, tokens 0..boundary-1 get adapter A
            # Since causal mask prevents token i from seeing token j>i,
            # tokens 0..boundary-1 only see other A-adapter tokens => should match
            same_seg_diff = diff[:boundary]
            cross_seg_diff = diff[boundary:]
            mx.eval(same_seg_diff, cross_seg_diff)

            if same_seg_diff.size > 0:
                same_seg_diffs.append(mx.mean(same_seg_diff).item())
            if cross_seg_diff.size > 0:
                cross_seg_diffs.append(mx.mean(cross_seg_diff).item())

            del mixed_mlp, single_per_token, multi_per_token, diff
            del same_seg_diff, cross_seg_diff

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
        pair_results[pair_key] = pair_result

        # Accumulate global stats
        for s in strategy_names:
            global_stats[s]["nll"] += stats[s]["nll"]
            global_stats[s]["n"] += stats[s]["n"]

        log(f"    multi_pass_mlp={pair_result['multi_pass_mlp_ppl']:.4f}, "
            f"single_pass_mlp={pair_result['single_pass_mlp_ppl']:.4f}, "
            f"per_seq_best={pair_result['per_seq_best_ppl']:.4f}")

    elapsed = time.time() - t0
    log(f"\n  Total evaluation time: {elapsed:.1f}s")
    log_memory("post-eval")

    mean_per_token_abs_diff = sum(per_token_diffs_all) / len(per_token_diffs_all) if per_token_diffs_all else 0
    mean_same_seg_diff = sum(same_seg_diffs) / len(same_seg_diffs) if same_seg_diffs else 0
    mean_cross_seg_diff = sum(cross_seg_diffs) / len(cross_seg_diffs) if cross_seg_diffs else 0
    max_same_seg_diff = max(same_seg_diffs) if same_seg_diffs else 0
    max_cross_seg_diff = max(cross_seg_diffs) if cross_seg_diffs else 0

    comparison = {
        "max_per_token_abs_diff": max_per_token_abs_diff,
        "mean_per_token_abs_diff": mean_per_token_abs_diff,
        "total_tokens_compared": total_tokens_compared,
        "n_sequences": len(mixed_sequences),
        "same_segment_mean_diff": mean_same_seg_diff,
        "same_segment_max_diff": max_same_seg_diff,
        "cross_segment_mean_diff": mean_cross_seg_diff,
        "cross_segment_max_diff": max_cross_seg_diff,
    }

    cleanup(model, tokenizer, adapters, mlp_adapters)
    return pair_results, global_stats, comparison


# ===========================================================================
# Phase 2: Analyze results and evaluate kill criteria
# ===========================================================================

def phase_analyze(pair_results, global_stats, comparison):
    log("\n" + "=" * 70)
    log("[Phase 2] Analyzing results")
    log("=" * 70)

    strategy_names = ["multi_pass_mlp", "single_pass_mlp", "per_seq_best", "base_only"]

    avg_ppls = {}
    for s in strategy_names:
        if global_stats[s]["n"] > 0:
            avg_ppls[s] = round(math.exp(global_stats[s]["nll"] / global_stats[s]["n"]), 4)
        else:
            avg_ppls[s] = float("inf")

    log("\n  Global average PPL (NLL-weighted across all pairs):")
    for s in strategy_names:
        marker = " <-- NEW" if s == "single_pass_mlp" else ""
        log(f"    {s:25s}: {avg_ppls[s]:.4f}{marker}")

    multi_ppl = avg_ppls["multi_pass_mlp"]
    single_ppl = avg_ppls["single_pass_mlp"]
    per_seq_ppl = avg_ppls["per_seq_best"]

    # K793: |PPL_single - PPL_multi| / PPL_multi < 1%
    if multi_ppl > 0:
        k793_ratio = abs(single_ppl - multi_ppl) / multi_ppl * 100
    else:
        k793_ratio = float("inf")
    k793_pass = k793_ratio < 1.0

    # K794: Single-pass MLP PPL < per-sequence best (4.815)
    per_seq_ref = 4.815
    k794_pass = single_ppl < per_seq_ref

    # K795: Per-token assignments identical (trivially true by construction)
    k795_pass = True

    log(f"\n  === Kill Criteria ===")
    log(f"  K793: |single({single_ppl:.4f}) - multi({multi_ppl:.4f})| / multi = "
        f"{k793_ratio:.4f}% < 1%? -> {'PASS' if k793_pass else 'FAIL'}")
    log(f"  K794: single({single_ppl:.4f}) < per_seq_ref({per_seq_ref})? "
        f"-> {'PASS' if k794_pass else 'FAIL'}")
    log(f"  K795: Assignments identical (by construction)? -> PASS")

    log(f"\n  === Per-Token NLL Comparison (single-pass vs multi-pass) ===")
    log(f"  Overall: max={comparison['max_per_token_abs_diff']:.6f}, "
        f"mean={comparison['mean_per_token_abs_diff']:.6f}")
    log(f"  Same-segment (should be 0.0): max={comparison['same_segment_max_diff']:.6f}, "
        f"mean={comparison['same_segment_mean_diff']:.6f}")
    log(f"  Cross-segment (expected divergence): max={comparison['cross_segment_max_diff']:.6f}, "
        f"mean={comparison['cross_segment_mean_diff']:.6f}")
    log(f"  Total tokens compared: {comparison['total_tokens_compared']}")

    # K793 refined: same-segment tokens should match exactly
    same_seg_exact = comparison['same_segment_max_diff'] < 0.001
    log(f"\n  PROOF VERIFICATION: Same-segment exact match? "
        f"max_diff={comparison['same_segment_max_diff']:.6f} "
        f"-> {'YES (proof confirmed)' if same_seg_exact else 'NO (implementation bug)'}")

    # Per-pair breakdown
    pair_analysis = {}
    for pair_key, result in pair_results.items():
        mp = result["multi_pass_mlp_ppl"]
        sp = result["single_pass_mlp_ppl"]
        ps = result["per_seq_best_ppl"]
        if mp > 0:
            pair_ratio = abs(sp - mp) / mp * 100
        else:
            pair_ratio = float("inf")
        pair_analysis[pair_key] = {
            "multi_pass_ppl": mp,
            "single_pass_ppl": sp,
            "per_seq_best_ppl": ps,
            "single_vs_multi_pct": round(pair_ratio, 4),
        }

    log(f"\n  === Per-Pair Single vs Multi ===")
    for pair_key, pa in pair_analysis.items():
        log(f"    {pair_key:20s}: multi={pa['multi_pass_ppl']:.4f} single={pa['single_pass_ppl']:.4f} "
            f"diff={pa['single_vs_multi_pct']:.4f}%")

    return {
        "avg_ppls": avg_ppls,
        "k793_ratio_pct": round(k793_ratio, 4),
        "k793_pass": k793_pass,
        "k794_single_ppl": single_ppl,
        "k794_ref": per_seq_ref,
        "k794_pass": k794_pass,
        "k795_pass": k795_pass,
        "same_segment_exact_match": same_seg_exact,
        "per_token_comparison": comparison,
        "pair_analysis": pair_analysis,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    t_start = time.time()
    log("=" * 70)
    log("Single-Pass MLP Mixed-Adapter Routing")
    log("Closing experiment-proof gap from Finding #312")
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
    mixed_sequences = phase_construct_mixed_data(tokenizer)
    del tokenizer
    gc.collect()

    # Phase 1: Evaluate
    pair_results, global_stats, comparison = phase_evaluate(MODEL_ID, mixed_sequences)

    # Phase 2: Analyze
    analysis = phase_analyze(pair_results, global_stats, comparison)

    # Compile results
    results = {
        "experiment": "single_pass_mlp_mixed_adapter",
        "type": "verification",
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
        "K793_ratio_pct": analysis["k793_ratio_pct"],
        "K793_pass": analysis["k793_pass"],
        "K794_single_ppl": analysis["k794_single_ppl"],
        "K794_ref": analysis["k794_ref"],
        "K794_pass": analysis["k794_pass"],
        "K795_pass": analysis["k795_pass"],
        "same_segment_exact_match": analysis["same_segment_exact_match"],
        "per_token_comparison": analysis["per_token_comparison"],
        "pair_analysis": analysis["pair_analysis"],
        "finding_312_reference": {
            "multi_pass_mlp_only_ppl": 4.656,
            "per_seq_best_ppl": 4.815,
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
        marker = " <-- NEW (single-pass)" if s_name == "single_pass_mlp" else ""
        marker = " <-- reference (multi-pass)" if s_name == "multi_pass_mlp" else marker
        log(f"  {s_name:25s}: PPL {ppl:.4f}{marker}")

    log(f"\n  K793: |single - multi| / multi = {analysis['k793_ratio_pct']:.4f}% "
        f"< 1%? -> {'PASS' if analysis['k793_pass'] else 'FAIL'}")
    log(f"  K794: single ({analysis['k794_single_ppl']:.4f}) < {analysis['k794_ref']}? "
        f"-> {'PASS' if analysis['k794_pass'] else 'FAIL'}")
    log(f"  K795: Assignments identical (by construction)? -> PASS")

    log(f"\n  Per-token NLL (overall): max={comparison['max_per_token_abs_diff']:.6f}, "
        f"mean={comparison['mean_per_token_abs_diff']:.6f}")
    log(f"  Same-segment (proof check): max={comparison['same_segment_max_diff']:.6f}")
    log(f"  Cross-segment (expected): mean={comparison['cross_segment_mean_diff']:.6f}")

    log(f"\n  Total time: {results['total_time_s']:.1f}s")
    log_memory("end")


if __name__ == "__main__":
    main()
