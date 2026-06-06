#!/usr/bin/env python3
"""
Block-Diagonal Attention + Single-Pass MLP Routing: Exact Match Everywhere.

Hypothesis: Block-diagonal causal attention combined with single-pass MLP-only
routing achieves exact equivalence with segment-isolated evaluation for ALL tokens,
with PPL between segment-isolated (4.042) and per-sequence best (4.815).

Kill criteria:
  K796: Block-diagonal + MLP-only PPL < per-sequence best (4.815)
  K797: Max per-token NLL diff (block-diag-single vs segment-isolated) < 0.01 for ALL tokens
  K798: Block-diagonal PPL within 5% of segment-isolated best (4.042 from Finding #305)

Finding #313: Same-segment tokens match exactly (0.000 diff, 25600 tokens).
This experiment extends that guarantee to ALL tokens via block-diagonal masking.

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
# Model loading utilities (from prior experiment)
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
# Block-diagonal mask construction
# ===========================================================================

def create_block_diagonal_causal_mask(seq_len, boundary_pos):
    """
    Create block-diagonal causal mask.

    M_bd[i,j] = 1 iff (j <= i) AND (segment(i) == segment(j))

    where segment(i) = 0 if i < boundary_pos, 1 if i >= boundary_pos.

    Returns: (seq_len, seq_len) boolean array.
    """
    positions = mx.arange(seq_len)
    # Causal: i >= j
    causal = positions[:, None] >= positions[None, :]
    # Same segment: both < boundary or both >= boundary
    seg_i = (positions >= boundary_pos).astype(mx.int32)
    seg_j = (positions >= boundary_pos).astype(mx.int32)
    same_segment = seg_i[:, None] == seg_j[None, :]
    # Block-diagonal causal
    mask = causal & same_segment
    return mask


# ===========================================================================
# Mixed-adapter MLP computation (from prior experiment)
# ===========================================================================

class MixedAdapterMLP:
    """
    A wrapper that performs single-pass mixed-adapter MLP computation.
    Applies adapter A to tokens < boundary, adapter B to tokens >= boundary.
    """

    def __init__(self, model, adapter_params_A, adapter_params_B, boundary_pos, scale):
        self.scale = scale
        self.boundary_pos = boundary_pos
        self.n_layers = len(model.model.layers)

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
        Returns: (1, T, d_out) LoRA delta
        """
        params = self.layer_params[layer_idx][module_name]

        lora_out_A = (x @ params["lora_a_A"]) @ params["lora_b_A"]
        lora_out_B = (x @ params["lora_a_B"]) @ params["lora_b_B"]

        T = x.shape[1]
        mask_A = mx.arange(T)[None, :, None] < self.boundary_pos

        mixed = mx.where(mask_A, lora_out_A, lora_out_B)

        # Match LoRALinear: scale in float32, THEN cast to x.dtype
        return (self.scale * mixed).astype(x.dtype)


# ===========================================================================
# Block-diagonal single-pass forward (THE NEW THING)
# ===========================================================================

def block_diagonal_single_pass_forward(model, tokens, mixed_mlp, boundary_pos):
    """
    Run a single forward pass with:
    - Block-diagonal causal attention mask (no cross-segment attention)
    - MLP uses per-token adapter: adapter A for tokens < boundary,
      adapter B for tokens >= boundary

    This should produce outputs identical to segment-isolated evaluation.
    """
    if len(tokens) < 2:
        return mx.array([]), 0

    x_in = mx.array(tokens[:-1])[None, :]
    y = mx.array(tokens[1:])[None, :]
    T = x_in.shape[1]

    # Create block-diagonal causal mask
    # mask shape: (T, T) boolean -- applies to prediction token positions
    bd_mask = create_block_diagonal_causal_mask(T, boundary_pos)

    # Manual layer-by-layer forward with block-diagonal mask
    h = model.model.embed_tokens(x_in)  # (1, T, d)

    for l in range(len(model.model.layers)):
        layer = model.model.layers[l]

        # --- Attention with block-diagonal mask and base weights ---
        r = layer.self_attn(layer.input_layernorm(h), bd_mask, None)
        h_post_attn = h + r

        # --- MLP with per-token mixed adapter ---
        h_normed = layer.post_attention_layernorm(h_post_attn)

        # Base MLP computation (LoRA is zeroed)
        gate_base = layer.mlp.gate_proj(h_normed)
        up_base = layer.mlp.up_proj(h_normed)

        # Add mixed LoRA delta
        gate_lora_delta = mixed_mlp.compute_mixed_lora_output(h_normed, l, "gate_proj")
        up_lora_delta = mixed_mlp.compute_mixed_lora_output(h_normed, l, "up_proj")

        gate_out = gate_base + gate_lora_delta
        up_out = up_base + up_lora_delta

        # Activation (BitNet uses relu2 = relu(x)^2)
        x_mid = nn.relu2(gate_out) * up_out

        # Sub-norm
        x_mid = layer.mlp.ffn_sub_norm(x_mid)

        # Down projection with mixed LoRA
        down_base = layer.mlp.down_proj(x_mid)
        down_lora_delta = mixed_mlp.compute_mixed_lora_output(x_mid, l, "down_proj")
        mlp_out = down_base + down_lora_delta

        h = h_post_attn + mlp_out

    # Final norm + LM head
    h = model.model.norm(h)
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)

    per_token_ce = nn.losses.cross_entropy(logits, y, reduction="none")
    mx.eval(per_token_ce)
    result = per_token_ce[0]
    del x_in, y, logits, h, bd_mask
    return result, per_token_ce.size


# ===========================================================================
# Segment-isolated evaluation (Finding #305 methodology)
# ===========================================================================

def compute_segment_isolated_ppl(model, seg_tokens, adapter_params):
    """
    Evaluate a single segment in isolation with its adapter.
    Standard causal mask, adapter applied to all tokens.
    Returns per-token NLL array.
    """
    if len(seg_tokens) < 2:
        return mx.array([]), 0.0, 0

    apply_adapter_to_model(model, adapter_params)
    zero_attn_adapter(model)  # MLP-only

    per_tok_nll, n = compute_per_token_nll(model, seg_tokens)
    mx.eval(per_tok_nll)
    total_nll = mx.sum(per_tok_nll).item()

    zero_adapter_in_model(model)
    return per_tok_nll, total_nll, n


# ===========================================================================
# Multi-pass MLP-only oracle (for comparison)
# ===========================================================================

def compute_multipass_mlp_only_ppl(model, tokens, domain_assignments, adapters, boundary_pos):
    """Multi-pass MLP-only oracle: Finding #312 methodology."""
    if len(tokens) < 2:
        return 0.0, 0, mx.array([])

    n_pred_tokens = len(tokens) - 1
    domain_nlls = {}

    for domain_name in DOMAINS:
        apply_adapter_to_model(model, adapters[domain_name])
        zero_attn_adapter(model)
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
# Data construction
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
# Phase 1: Compare block-diagonal vs segment-isolated vs multi-pass
# ===========================================================================

def phase_evaluate(model_id, mixed_sequences):
    log("\n" + "=" * 70)
    log("[Phase 1] Evaluating block-diagonal single-pass vs segment-isolated vs multi-pass")
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

    # Strategies
    strategy_names = ["block_diag_single", "seg_isolated", "multi_pass_mlp", "per_seq_best", "base_only"]

    global_stats = {s: {"nll": 0.0, "n": 0} for s in strategy_names}
    pair_results = {}

    # Per-token comparison: block-diagonal vs segment-isolated (should be exact)
    max_bd_vs_iso_diff = 0.0
    mean_bd_vs_iso_diffs = []
    # Per-token comparison: block-diagonal vs multi-pass (expected divergence)
    max_bd_vs_multi_diff = 0.0
    mean_bd_vs_multi_diffs = []
    # Segment-level breakdown
    seg_a_bd_vs_iso_diffs = []  # First segment (same in both by Thm 1c)
    seg_b_bd_vs_iso_diffs = []  # Second segment (should also match by Lemma 1)
    seg_a_bd_vs_multi_diffs = []
    seg_b_bd_vs_multi_diffs = []

    total_tokens_compared = 0

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

            n_pred_tokens = len(tokens) - 1

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
            best_n = 0
            for d_name in DOMAINS:
                apply_adapter_to_model(model, adapters[d_name])
                nll_d, n_d = compute_full_seq_ppl(model, tokens)
                if nll_d < best_nll:
                    best_nll = nll_d
                    best_n = n_d
                zero_adapter_in_model(model)
            stats["per_seq_best"]["nll"] += best_nll
            stats["per_seq_best"]["n"] += best_n

            # ---- Strategy 2: Segment-isolated (Finding #305) ----
            # Evaluate each segment independently with its adapter
            iso_a_nll_arr, iso_a_nll, iso_a_n = compute_segment_isolated_ppl(
                model, seg_a_tokens, adapters[domain_a])
            iso_b_nll_arr, iso_b_nll, iso_b_n = compute_segment_isolated_ppl(
                model, seg_b_tokens, adapters[domain_b])
            stats["seg_isolated"]["nll"] += iso_a_nll + iso_b_nll
            stats["seg_isolated"]["n"] += iso_a_n + iso_b_n

            # Build segment-isolated per-token NLL for the full sequence
            # seg_a prediction tokens: positions 0..boundary-2 (predicting tokens 1..boundary-1)
            # seg_b prediction tokens: positions boundary..n_pred_tokens-1
            # But segment-isolated evaluates each segment alone, so:
            # iso_a_nll_arr has SEGMENT_LENGTH-1 entries (predicting tokens 1..127 of seg_a)
            # iso_b_nll_arr has SEGMENT_LENGTH-1 entries (predicting tokens 1..127 of seg_b)
            # In the combined sequence:
            # - prediction positions 0..126 correspond to seg_a tokens 1..127
            # - prediction position 127 corresponds to token 128 (first of seg_b, predicted from seg_a token 127)
            #   This is a CROSS-SEGMENT prediction that segment-isolated cannot handle
            # - prediction positions 128..254 correspond to seg_b tokens 129..255 (or seg_b internal positions 1..127)

            # For block-diagonal: position 127 (predicting token 128) looks at seg_a context only
            # but predicts seg_b's first token -- this is fundamentally the domain boundary token.
            # Segment-isolated has no equivalent since seg_a never sees seg_b tokens and vice versa.
            # For segment B, block-diag starts fresh at position 128 with no preceding context.

            # ---- Strategy 3: Multi-pass MLP-only oracle ----
            nll_multi, n_multi, multi_per_token = compute_multipass_mlp_only_ppl(
                model, tokens, domain_assignments, adapters, boundary)
            stats["multi_pass_mlp"]["nll"] += nll_multi
            stats["multi_pass_mlp"]["n"] += n_multi

            # ---- Strategy 4: BLOCK-DIAGONAL single-pass (NEW) ----
            mixed_mlp = MixedAdapterMLP(
                model, mlp_adapters[domain_a], mlp_adapters[domain_b],
                boundary, LORA_SCALE
            )

            zero_adapter_in_model(model)
            bd_per_token, n_bd = block_diagonal_single_pass_forward(
                model, tokens, mixed_mlp, boundary)
            mx.eval(bd_per_token)

            bd_nll = 0.0
            n_bd_counted = 0
            for start_idx, end_idx, d_name in domain_assignments:
                s = max(0, start_idx)
                e = min(end_idx, n_pred_tokens)
                if s >= e:
                    continue
                seg_nll = mx.sum(bd_per_token[s:e])
                mx.eval(seg_nll)
                bd_nll += seg_nll.item()
                n_bd_counted += (e - s)

            stats["block_diag_single"]["nll"] += bd_nll
            stats["block_diag_single"]["n"] += n_bd_counted

            # ---- Per-token comparison: block-diagonal vs segment-isolated ----
            # Align: seg_a covers prediction positions 0..boundary-2 (SEGMENT_LENGTH-1 = 127 tokens)
            # block-diag covers prediction positions 0..boundary-2 for segment A
            # These should match exactly by Lemma 1.
            # iso_a_nll_arr: (127,) -- predictions for seg_a tokens 1..127
            # bd_per_token[0:127]: predictions for combined tokens 1..127 (all seg_a)
            if iso_a_nll_arr.size > 0 and bd_per_token.size > 0:
                seg_a_pred_len = min(iso_a_nll_arr.size, boundary - 1)
                if seg_a_pred_len > 0:
                    diff_a = mx.abs(bd_per_token[:seg_a_pred_len] - iso_a_nll_arr[:seg_a_pred_len])
                    mx.eval(diff_a)
                    seg_a_bd_vs_iso_diffs.append(mx.mean(diff_a).item())
                    max_a = mx.max(diff_a).item()
                    if max_a > max_bd_vs_iso_diff:
                        max_bd_vs_iso_diff = max_a

            # For segment B: block-diag positions boundary..n_pred-1 vs iso_b positions 0..SEGMENT_LENGTH-2
            # In block-diagonal, token at position boundary sees NO preceding context (block-diagonal
            # resets attention). This is equivalent to evaluating seg_b in isolation.
            # iso_b_nll_arr: (127,) predictions for seg_b tokens 1..127
            # bd_per_token[boundary:boundary+127]: predictions for combined tokens boundary+1..boundary+127
            # But wait: boundary prediction position predicts token at position boundary+1 using
            # token at position boundary. In block-diagonal, token boundary only sees itself.
            # In segment-isolated, token 0 of seg_b only sees itself. These should match.
            if iso_b_nll_arr.size > 0:
                seg_b_bd_start = boundary  # prediction position boundary predicts token boundary+1
                seg_b_pred_len = min(iso_b_nll_arr.size, n_pred_tokens - boundary)
                if seg_b_pred_len > 0:
                    diff_b = mx.abs(bd_per_token[seg_b_bd_start:seg_b_bd_start + seg_b_pred_len]
                                    - iso_b_nll_arr[:seg_b_pred_len])
                    mx.eval(diff_b)
                    seg_b_bd_vs_iso_diffs.append(mx.mean(diff_b).item())
                    max_b = mx.max(diff_b).item()
                    if max_b > max_bd_vs_iso_diff:
                        max_bd_vs_iso_diff = max_b

            # Track overall bd vs iso mean diff
            valid_diffs = []
            if seg_a_bd_vs_iso_diffs:
                valid_diffs.append(seg_a_bd_vs_iso_diffs[-1])
            if seg_b_bd_vs_iso_diffs:
                valid_diffs.append(seg_b_bd_vs_iso_diffs[-1])
            if valid_diffs:
                mean_bd_vs_iso_diffs.append(sum(valid_diffs) / len(valid_diffs))

            # ---- Per-token comparison: block-diagonal vs multi-pass ----
            diff_multi = mx.abs(bd_per_token - multi_per_token)
            mx.eval(diff_multi)
            max_m = mx.max(diff_multi).item()
            if max_m > max_bd_vs_multi_diff:
                max_bd_vs_multi_diff = max_m
            mean_bd_vs_multi_diffs.append(mx.mean(diff_multi).item())

            # Segment-level breakdown for multi comparison
            if boundary - 1 > 0:
                seg_a_diff_multi = diff_multi[:boundary - 1]
                mx.eval(seg_a_diff_multi)
                seg_a_bd_vs_multi_diffs.append(mx.mean(seg_a_diff_multi).item())
            if n_pred_tokens - boundary > 0:
                seg_b_diff_multi = diff_multi[boundary:]
                mx.eval(seg_b_diff_multi)
                seg_b_bd_vs_multi_diffs.append(mx.mean(seg_b_diff_multi).item())

            total_tokens_compared += n_pred_tokens

            del mixed_mlp, bd_per_token, multi_per_token, diff_multi
            if 'diff_a' in dir():
                del diff_a
            if 'diff_b' in dir():
                del diff_b

            if (seq_idx + 1) % 5 == 0:
                log(f"    Processed {seq_idx+1}/{len(sequences)} sequences")

            gc.collect()
            mx.clear_cache()

        # Compute PPLs
        pair_result = {}
        for s in strategy_names:
            if stats[s]["n"] > 0:
                pair_result[f"{s}_ppl"] = round(math.exp(stats[s]["nll"] / stats[s]["n"]), 4)
            else:
                pair_result[f"{s}_ppl"] = float("inf")
        pair_result["n_sequences"] = len(sequences)
        pair_results[pair_key] = pair_result

        for s in strategy_names:
            global_stats[s]["nll"] += stats[s]["nll"]
            global_stats[s]["n"] += stats[s]["n"]

        log(f"    block_diag={pair_result['block_diag_single_ppl']:.4f}, "
            f"seg_iso={pair_result['seg_isolated_ppl']:.4f}, "
            f"multi={pair_result['multi_pass_mlp_ppl']:.4f}, "
            f"per_seq={pair_result['per_seq_best_ppl']:.4f}")

    elapsed = time.time() - t0
    log(f"\n  Total evaluation time: {elapsed:.1f}s")
    log_memory("post-eval")

    comparison = {
        # Block-diagonal vs segment-isolated (should be exact)
        "bd_vs_iso_max_diff": max_bd_vs_iso_diff,
        "bd_vs_iso_mean_diff": (sum(mean_bd_vs_iso_diffs) / len(mean_bd_vs_iso_diffs)
                                 if mean_bd_vs_iso_diffs else 0),
        "seg_a_bd_vs_iso_mean": (sum(seg_a_bd_vs_iso_diffs) / len(seg_a_bd_vs_iso_diffs)
                                  if seg_a_bd_vs_iso_diffs else 0),
        "seg_a_bd_vs_iso_max": max(seg_a_bd_vs_iso_diffs) if seg_a_bd_vs_iso_diffs else 0,
        "seg_b_bd_vs_iso_mean": (sum(seg_b_bd_vs_iso_diffs) / len(seg_b_bd_vs_iso_diffs)
                                  if seg_b_bd_vs_iso_diffs else 0),
        "seg_b_bd_vs_iso_max": max(seg_b_bd_vs_iso_diffs) if seg_b_bd_vs_iso_diffs else 0,
        # Block-diagonal vs multi-pass (expected divergence)
        "bd_vs_multi_max_diff": max_bd_vs_multi_diff,
        "bd_vs_multi_mean_diff": (sum(mean_bd_vs_multi_diffs) / len(mean_bd_vs_multi_diffs)
                                   if mean_bd_vs_multi_diffs else 0),
        "seg_a_bd_vs_multi_mean": (sum(seg_a_bd_vs_multi_diffs) / len(seg_a_bd_vs_multi_diffs)
                                    if seg_a_bd_vs_multi_diffs else 0),
        "seg_b_bd_vs_multi_mean": (sum(seg_b_bd_vs_multi_diffs) / len(seg_b_bd_vs_multi_diffs)
                                    if seg_b_bd_vs_multi_diffs else 0),
        "total_tokens_compared": total_tokens_compared,
        "n_sequences": len(mixed_sequences),
    }

    cleanup(model, tokenizer, adapters, mlp_adapters)
    return pair_results, global_stats, comparison


# ===========================================================================
# Phase 2: Analyze results
# ===========================================================================

def phase_analyze(pair_results, global_stats, comparison):
    log("\n" + "=" * 70)
    log("[Phase 2] Analyzing results")
    log("=" * 70)

    strategy_names = ["block_diag_single", "seg_isolated", "multi_pass_mlp", "per_seq_best", "base_only"]

    avg_ppls = {}
    for s in strategy_names:
        if global_stats[s]["n"] > 0:
            avg_ppls[s] = round(math.exp(global_stats[s]["nll"] / global_stats[s]["n"]), 4)
        else:
            avg_ppls[s] = float("inf")

    log("\n  Global average PPL (NLL-weighted across all pairs):")
    for s in strategy_names:
        marker = ""
        if s == "block_diag_single":
            marker = " <-- NEW (block-diagonal single-pass)"
        elif s == "seg_isolated":
            marker = " <-- reference (segment-isolated)"
        elif s == "multi_pass_mlp":
            marker = " <-- reference (multi-pass)"
        log(f"    {s:25s}: {avg_ppls[s]:.4f}{marker}")

    bd_ppl = avg_ppls["block_diag_single"]
    iso_ppl = avg_ppls["seg_isolated"]
    multi_ppl = avg_ppls["multi_pass_mlp"]
    per_seq_ppl = avg_ppls["per_seq_best"]

    # K796: Block-diagonal PPL < per-sequence best (4.815)
    per_seq_ref = 4.815
    k796_pass = bd_ppl < per_seq_ref

    # K797: Max per-token NLL diff (block-diag vs segment-isolated) < 0.01
    k797_pass = comparison["bd_vs_iso_max_diff"] < 0.01

    # K798: Block-diagonal PPL within 5% of segment-isolated best (4.042)
    seg_iso_ref = 4.042
    if iso_ppl > 0:
        k798_ratio = abs(bd_ppl - iso_ppl) / iso_ppl * 100
    else:
        k798_ratio = float("inf")
    k798_pass = k798_ratio < 5.0

    # PPL ratio block-diag vs multi-pass
    if multi_ppl > 0:
        bd_vs_multi_ppl_ratio = abs(bd_ppl - multi_ppl) / multi_ppl * 100
    else:
        bd_vs_multi_ppl_ratio = float("inf")

    log(f"\n  === Kill Criteria ===")
    log(f"  K796: block_diag({bd_ppl:.4f}) < per_seq_ref({per_seq_ref})? "
        f"-> {'PASS' if k796_pass else 'FAIL'}")
    log(f"  K797: max_per_token_diff(bd vs iso)={comparison['bd_vs_iso_max_diff']:.6f} "
        f"< 0.01? -> {'PASS' if k797_pass else 'FAIL'}")
    log(f"  K798: |bd({bd_ppl:.4f}) - iso({iso_ppl:.4f})| / iso = {k798_ratio:.4f}% "
        f"< 5%? -> {'PASS' if k798_pass else 'FAIL'}")

    log(f"\n  === Per-Token NLL Comparison ===")
    log(f"  Block-diagonal vs Segment-isolated (PROOF CHECK: should be ~0):")
    log(f"    Overall: max={comparison['bd_vs_iso_max_diff']:.6f}, "
        f"mean={comparison['bd_vs_iso_mean_diff']:.6f}")
    log(f"    Segment A: max={comparison['seg_a_bd_vs_iso_max']:.6f}, "
        f"mean={comparison['seg_a_bd_vs_iso_mean']:.6f}")
    log(f"    Segment B: max={comparison['seg_b_bd_vs_iso_max']:.6f}, "
        f"mean={comparison['seg_b_bd_vs_iso_mean']:.6f}")

    log(f"\n  Block-diagonal vs Multi-pass (expected divergence):")
    log(f"    Overall: max={comparison['bd_vs_multi_max_diff']:.6f}, "
        f"mean={comparison['bd_vs_multi_mean_diff']:.6f}")
    log(f"    Segment A: mean={comparison['seg_a_bd_vs_multi_mean']:.6f}")
    log(f"    Segment B: mean={comparison['seg_b_bd_vs_multi_mean']:.6f}")
    log(f"    PPL ratio: {bd_vs_multi_ppl_ratio:.4f}%")

    log(f"\n  Total tokens compared: {comparison['total_tokens_compared']}")

    # Per-pair breakdown
    pair_analysis = {}
    for pair_key, result in pair_results.items():
        bd = result["block_diag_single_ppl"]
        iso = result["seg_isolated_ppl"]
        mp = result["multi_pass_mlp_ppl"]
        ps = result["per_seq_best_ppl"]
        pair_analysis[pair_key] = {
            "block_diag_ppl": bd,
            "seg_isolated_ppl": iso,
            "multi_pass_ppl": mp,
            "per_seq_best_ppl": ps,
            "bd_vs_iso_pct": round(abs(bd - iso) / iso * 100, 4) if iso > 0 else float("inf"),
            "bd_vs_multi_pct": round(abs(bd - mp) / mp * 100, 4) if mp > 0 else float("inf"),
        }

    log(f"\n  === Per-Pair Results ===")
    for pair_key, pa in pair_analysis.items():
        log(f"    {pair_key:20s}: bd={pa['block_diag_ppl']:.4f} iso={pa['seg_isolated_ppl']:.4f} "
            f"multi={pa['multi_pass_ppl']:.4f} | bd_vs_iso={pa['bd_vs_iso_pct']:.4f}%")

    return {
        "avg_ppls": avg_ppls,
        "k796_pass": k796_pass,
        "k796_bd_ppl": bd_ppl,
        "k796_ref": per_seq_ref,
        "k797_pass": k797_pass,
        "k797_max_diff": comparison["bd_vs_iso_max_diff"],
        "k798_pass": k798_pass,
        "k798_ratio_pct": round(k798_ratio, 4),
        "k798_ref": seg_iso_ref,
        "bd_vs_multi_ppl_ratio_pct": round(bd_vs_multi_ppl_ratio, 4),
        "per_token_comparison": comparison,
        "pair_analysis": pair_analysis,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    t_start = time.time()
    log("=" * 70)
    log("Block-Diagonal Attention + Single-Pass MLP Routing")
    log("Extending exact equivalence to ALL tokens via segment isolation")
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
        "experiment": "block_diagonal_single_pass",
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
        "K796_bd_ppl": analysis["k796_bd_ppl"],
        "K796_ref": analysis["k796_ref"],
        "K796_pass": analysis["k796_pass"],
        "K797_max_diff": analysis["k797_max_diff"],
        "K797_pass": analysis["k797_pass"],
        "K798_ratio_pct": analysis["k798_ratio_pct"],
        "K798_ref": analysis["k798_ref"],
        "K798_pass": analysis["k798_pass"],
        "bd_vs_multi_ppl_ratio_pct": analysis["bd_vs_multi_ppl_ratio_pct"],
        "per_token_comparison": analysis["per_token_comparison"],
        "pair_analysis": analysis["pair_analysis"],
        "finding_313_reference": {
            "single_pass_mlp_ppl": 4.684,
            "multi_pass_mlp_ppl": 4.656,
            "per_seq_best_ppl": 4.815,
            "same_segment_max_diff": 0.0,
            "cross_segment_mean_diff": 0.068,
        },
        "finding_305_reference": {
            "seg_isolated_ppl": 4.042,
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
        marker = ""
        if s_name == "block_diag_single":
            marker = " <-- NEW (block-diagonal)"
        elif s_name == "seg_isolated":
            marker = " <-- segment-isolated (Finding #305)"
        elif s_name == "multi_pass_mlp":
            marker = " <-- multi-pass (Finding #313)"
        log(f"  {s_name:25s}: PPL {ppl:.4f}{marker}")

    log(f"\n  K796: block_diag({analysis['k796_bd_ppl']:.4f}) < per_seq({analysis['k796_ref']})? "
        f"-> {'PASS' if analysis['k796_pass'] else 'FAIL'}")
    log(f"  K797: max NLL diff (bd vs iso) = {analysis['k797_max_diff']:.6f} < 0.01? "
        f"-> {'PASS' if analysis['k797_pass'] else 'FAIL'}")
    log(f"  K798: |bd - iso| / iso = {analysis['k798_ratio_pct']:.4f}% < 5%? "
        f"-> {'PASS' if analysis['k798_pass'] else 'FAIL'}")

    log(f"\n  Per-token comparison (bd vs iso, PROOF VERIFICATION):")
    log(f"    Max diff: {comparison['bd_vs_iso_max_diff']:.6f}")
    log(f"    Mean diff: {comparison['bd_vs_iso_mean_diff']:.6f}")
    log(f"    Seg A max: {comparison['seg_a_bd_vs_iso_max']:.6f}")
    log(f"    Seg B max: {comparison['seg_b_bd_vs_iso_max']:.6f}")

    log(f"\n  Block-diagonal vs multi-pass:")
    log(f"    PPL ratio: {analysis['bd_vs_multi_ppl_ratio_pct']:.4f}%")
    log(f"    Max per-token diff: {comparison['bd_vs_multi_max_diff']:.6f}")

    log(f"\n  Total time: {results['total_time_s']:.1f}s")
    log_memory("end")


if __name__ == "__main__":
    main()
