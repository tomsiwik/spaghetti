#!/usr/bin/env python3
"""
KV-Cache Reuse Across Adapter Switches: Segment-Isolated Routing with Context.

Finding #305 showed segment isolation gives +16% PPL over per-sequence routing
but loses cross-segment context. This experiment tests whether KV-cache from
segment A (computed with adapter A) can be reused when processing segment B
(with adapter B) to recover cross-segment context.

Three evaluation strategies compared:
  1. Isolated: each segment evaluated independently (Finding #305 baseline)
  2. KV-reuse: segment A processed with adapter A fills KV-cache, segment B
     processed with adapter B using cached KV from segment A
  3. Full-recompute: segment A+B processed entirely with adapter B from scratch
     (segment B uses adapter B's view of segment A)

Kill criteria:
  K781: KV-reuse PPL within 3% of full-recompute segment routing
  K782: KV-reuse latency < full-recompute latency (speedup > 1.2x)
  K783: Cross-segment context improves PPL vs isolated segments

Platform: Apple M5 Pro 48GB, MLX, $0.
"""

import copy
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
import mlx.optimizers as opt
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
# Model loading utilities (reused from prior experiments)
# ===========================================================================
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.bitlinear_layers import BitLinear
from mlx_lm.models.cache import KVCache, make_prompt_cache


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
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
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


def apply_adapter_to_model(model, adapter_params):
    model.update(tree_unflatten(list(adapter_params.items())))


def zero_adapter_in_model(model):
    updates = []
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_b" in name:
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
# PPL computation utilities with KV-cache support
# ===========================================================================

def compute_ppl_no_cache(model, tokens):
    """Compute PPL on tokens without any cache (isolated evaluation)."""
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


def prefill_cache(model, tokens):
    """Run a forward pass on tokens to fill KV-cache. Returns the cache."""
    cache = make_prompt_cache(model)
    x = mx.array(tokens)[None, :]
    logits = model(x, cache=cache)
    mx.eval(logits, [c.state for c in cache])
    del logits, x
    return cache


def compute_ppl_with_cache(model, tokens, cache):
    """Compute PPL on tokens using a pre-filled KV-cache for context.

    The cache provides context from preceding tokens. We compute NLL
    on the provided tokens using next-token prediction.
    """
    if len(tokens) < 2:
        return 0.0, 0

    # Process all tokens through model with the cache
    # The cache already has the preceding context
    x = mx.array(tokens)[None, :]
    logits = model(x, cache=cache)
    mx.eval(logits)

    # Compute NLL: predict each token from its predecessor
    # logits shape: [1, len(tokens), vocab]
    # We predict tokens[1:] from logits[:-1]
    pred_logits = logits[:, :-1, :]
    targets = mx.array(tokens[1:])[None, :]
    loss = nn.losses.cross_entropy(pred_logits, targets, reduction="sum")
    mx.eval(loss)
    nll = loss.item()
    n = targets.size
    del x, logits, pred_logits, targets, loss
    return nll, n


def compute_ppl_full_sequence(model, seg_a_tokens, seg_b_tokens):
    """Compute PPL on segment B within the full sequence context.

    Process full sequence [seg_a || seg_b] and return NLL only for
    segment B tokens (using the full context).
    """
    full_tokens = seg_a_tokens + seg_b_tokens
    if len(full_tokens) < 2:
        return 0.0, 0

    x = mx.array(full_tokens[:-1])[None, :]
    y_full = mx.array(full_tokens[1:])[None, :]
    logits = model(x)
    mx.eval(logits)

    # Only compute NLL for segment B tokens
    # Segment B starts at position len(seg_a_tokens) in the full sequence
    # In next-token prediction, predicting token at position t uses logits at position t-1
    # So segment B predictions start at logit position len(seg_a_tokens)-1
    # and targets start at position len(seg_a_tokens)
    b_start = len(seg_a_tokens) - 1  # logit index for first seg B prediction
    b_logits = logits[:, b_start:, :]
    b_targets = y_full[:, b_start:]

    loss = nn.losses.cross_entropy(b_logits, b_targets, reduction="sum")
    mx.eval(loss)
    nll = loss.item()
    n = b_targets.size
    del x, y_full, logits, b_logits, b_targets, loss
    return nll, n


# ===========================================================================
# Deep copy cache utility
# ===========================================================================

def deep_copy_cache(cache):
    """Create an independent deep copy of a KV-cache."""
    new_cache = []
    for c in cache:
        new_c = KVCache()
        if c.keys is not None:
            # Copy the actual data up to offset
            new_c.keys = mx.array(c.keys[..., :c.offset, :])
            new_c.values = mx.array(c.values[..., :c.offset, :])
            new_c.offset = c.offset
        new_cache.append(new_c)
    mx.eval([c.state for c in new_cache if c.keys is not None])
    return new_cache


# ===========================================================================
# Phase 0: Construct mixed-domain evaluation data
# ===========================================================================
def phase_construct_mixed_data(tokenizer):
    """Create mixed-domain sequences (identical to Finding #305)."""
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

            mixed_sequences.append({
                "seg_a_tokens": seg_a,
                "seg_b_tokens": seg_b,
                "domain_a": domain_a,
                "domain_b": domain_b,
            })
            pair_count += 1

        log(f"  {domain_a}+{domain_b}: {pair_count} sequences")

    log(f"  Total mixed sequences: {len(mixed_sequences)}")
    return mixed_sequences


# ===========================================================================
# Phase 1: Evaluate all three strategies
# ===========================================================================
def phase_evaluate(model_id, mixed_sequences):
    """Compare isolated, KV-reuse, and full-recompute strategies.

    For each mixed sequence [seg_a | seg_b]:

    Strategy 1 (Isolated): Evaluate seg_b with adapter_b independently.
      No context from seg_a.

    Strategy 2 (KV-reuse): Process seg_a with adapter_a (fills KV-cache).
      Switch to adapter_b. Process seg_b using cached KV from seg_a.
      Seg B sees seg A as processed by the correct adapter.

    Strategy 3 (Full-recompute): Process full [seg_a + seg_b] with adapter_b
      from scratch. Compute PPL only on seg_b tokens.
      Seg B sees seg A as processed by the WRONG adapter (adapter_b on seg_a).

    We measure PPL on segment B only (to isolate the context effect).
    """
    log("\n" + "=" * 70)
    log("[Phase 1] Evaluating strategies: isolated, KV-reuse, full-recompute")
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

    # Group by pair
    pair_sequences = defaultdict(list)
    for seq in mixed_sequences:
        pair_key = f"{seq['domain_a']}+{seq['domain_b']}"
        pair_sequences[pair_key].append(seq)

    # Global accumulators
    global_stats = {
        "isolated_seg_b": {"nll": 0.0, "n": 0},
        "kv_reuse_seg_b": {"nll": 0.0, "n": 0},
        "full_recompute_seg_b": {"nll": 0.0, "n": 0},
        "isolated_both": {"nll": 0.0, "n": 0},  # both segments isolated (Finding #305 baseline)
    }

    # Latency accumulators
    latency_stats = {
        "isolated": [],
        "kv_reuse": [],
        "full_recompute": [],
    }

    pair_results = {}

    for pair_key, sequences in pair_sequences.items():
        log(f"\n  === Pair: {pair_key} ({len(sequences)} sequences) ===")

        pair_stats = {
            "isolated_seg_b": {"nll": 0.0, "n": 0},
            "kv_reuse_seg_b": {"nll": 0.0, "n": 0},
            "full_recompute_seg_b": {"nll": 0.0, "n": 0},
            "isolated_both": {"nll": 0.0, "n": 0},
        }
        pair_latency = {"isolated": [], "kv_reuse": [], "full_recompute": []}

        for seq_idx, seq_data in enumerate(sequences):
            seg_a_tokens = seq_data["seg_a_tokens"]
            seg_b_tokens = seq_data["seg_b_tokens"]
            domain_a = seq_data["domain_a"]
            domain_b = seq_data["domain_b"]

            # ---- Strategy 1: Isolated (baseline from Finding #305) ----
            t_iso_start = time.time()

            # Evaluate seg_a with adapter_a (isolated)
            apply_adapter_to_model(model, adapters[domain_a])
            nll_a_iso, n_a_iso = compute_ppl_no_cache(model, seg_a_tokens)

            # Evaluate seg_b with adapter_b (isolated)
            apply_adapter_to_model(model, adapters[domain_b])
            nll_b_iso, n_b_iso = compute_ppl_no_cache(model, seg_b_tokens)

            t_iso_end = time.time()

            pair_stats["isolated_seg_b"]["nll"] += nll_b_iso
            pair_stats["isolated_seg_b"]["n"] += n_b_iso
            pair_stats["isolated_both"]["nll"] += nll_a_iso + nll_b_iso
            pair_stats["isolated_both"]["n"] += n_a_iso + n_b_iso
            pair_latency["isolated"].append(t_iso_end - t_iso_start)

            zero_adapter_in_model(model)

            # ---- Strategy 2: KV-reuse ----
            t_kv_start = time.time()

            # Step 1: Process seg_a with adapter_a, filling KV-cache
            apply_adapter_to_model(model, adapters[domain_a])
            cache = prefill_cache(model, seg_a_tokens)

            # Step 2: Switch to adapter_b
            apply_adapter_to_model(model, adapters[domain_b])

            # Step 3: Process seg_b using cached KV from seg_a
            nll_b_kv, n_b_kv = compute_ppl_with_cache(model, seg_b_tokens, cache)

            t_kv_end = time.time()

            pair_stats["kv_reuse_seg_b"]["nll"] += nll_b_kv
            pair_stats["kv_reuse_seg_b"]["n"] += n_b_kv
            pair_latency["kv_reuse"].append(t_kv_end - t_kv_start)

            zero_adapter_in_model(model)
            del cache
            gc.collect()
            mx.clear_cache()

            # ---- Strategy 3: Full-recompute ----
            t_fr_start = time.time()

            # Process full [seg_a + seg_b] with adapter_b, compute PPL on seg_b only
            apply_adapter_to_model(model, adapters[domain_b])
            nll_b_fr, n_b_fr = compute_ppl_full_sequence(model, seg_a_tokens, seg_b_tokens)

            t_fr_end = time.time()

            pair_stats["full_recompute_seg_b"]["nll"] += nll_b_fr
            pair_stats["full_recompute_seg_b"]["n"] += n_b_fr
            pair_latency["full_recompute"].append(t_fr_end - t_fr_start)

            zero_adapter_in_model(model)

            if (seq_idx + 1) % 5 == 0:
                log(f"    Processed {seq_idx+1}/{len(sequences)} sequences")

            gc.collect()
            mx.clear_cache()

        # Compute PPLs for this pair
        pair_result = {}
        for strategy, s in pair_stats.items():
            if s["n"] > 0:
                pair_result[f"{strategy}_ppl"] = round(math.exp(s["nll"] / s["n"]), 4)
            else:
                pair_result[f"{strategy}_ppl"] = float("inf")

        pair_result["n_sequences"] = len(sequences)
        pair_result["mean_latency_isolated_s"] = round(
            sum(pair_latency["isolated"]) / max(len(pair_latency["isolated"]), 1), 4
        )
        pair_result["mean_latency_kv_reuse_s"] = round(
            sum(pair_latency["kv_reuse"]) / max(len(pair_latency["kv_reuse"]), 1), 4
        )
        pair_result["mean_latency_full_recompute_s"] = round(
            sum(pair_latency["full_recompute"]) / max(len(pair_latency["full_recompute"]), 1), 4
        )

        pair_result["_raw_stats"] = {s: dict(v) for s, v in pair_stats.items()}

        pair_results[pair_key] = pair_result

        # Accumulate globals
        for strategy in pair_stats:
            global_stats[strategy]["nll"] += pair_stats[strategy]["nll"]
            global_stats[strategy]["n"] += pair_stats[strategy]["n"]
        for strategy in pair_latency:
            latency_stats[strategy].extend(pair_latency[strategy])

        log(f"    Isolated seg_b PPL:      {pair_result['isolated_seg_b_ppl']:.4f}")
        log(f"    KV-reuse seg_b PPL:      {pair_result['kv_reuse_seg_b_ppl']:.4f}")
        log(f"    Full-recompute seg_b PPL: {pair_result['full_recompute_seg_b_ppl']:.4f}")
        log(f"    Latency: iso={pair_result['mean_latency_isolated_s']:.3f}s "
            f"kv={pair_result['mean_latency_kv_reuse_s']:.3f}s "
            f"fr={pair_result['mean_latency_full_recompute_s']:.3f}s")

    elapsed = time.time() - t0
    log(f"\n  Total evaluation time: {elapsed:.1f}s")
    log_memory("post-eval")

    cleanup(model, tokenizer, adapters)
    return pair_results, global_stats, latency_stats


# ===========================================================================
# Phase 2: Analysis and kill criteria
# ===========================================================================
def phase_analyze(pair_results, global_stats, latency_stats):
    """Compute aggregated metrics and evaluate kill criteria."""
    log("\n" + "=" * 70)
    log("[Phase 2] Analyzing results")
    log("=" * 70)

    # Compute global PPLs from NLL accumulators
    strategies = ["isolated_seg_b", "kv_reuse_seg_b", "full_recompute_seg_b", "isolated_both"]
    avg_ppls = {}
    for s in strategies:
        if global_stats[s]["n"] > 0:
            avg_ppls[s] = round(math.exp(global_stats[s]["nll"] / global_stats[s]["n"]), 4)
        else:
            avg_ppls[s] = float("inf")

    log("\n  Average PPL across all pairs (segment B only):")
    for s in strategies:
        log(f"    {s:30s}: {avg_ppls[s]:.4f}")

    iso_ppl = avg_ppls["isolated_seg_b"]
    kv_ppl = avg_ppls["kv_reuse_seg_b"]
    fr_ppl = avg_ppls["full_recompute_seg_b"]

    # K781: KV-reuse vs full-recompute, within 3%
    if fr_ppl > 0 and fr_ppl != float("inf"):
        kv_vs_fr_pct = abs(kv_ppl - fr_ppl) / fr_ppl * 100
    else:
        kv_vs_fr_pct = float("inf")

    k781_pass = kv_vs_fr_pct <= 3.0

    # K782: Latency speedup
    mean_latency_kv = sum(latency_stats["kv_reuse"]) / max(len(latency_stats["kv_reuse"]), 1)
    mean_latency_fr = sum(latency_stats["full_recompute"]) / max(len(latency_stats["full_recompute"]), 1)
    mean_latency_iso = sum(latency_stats["isolated"]) / max(len(latency_stats["isolated"]), 1)

    if mean_latency_kv > 0:
        speedup_vs_fr = mean_latency_fr / mean_latency_kv
    else:
        speedup_vs_fr = float("inf")

    k782_pass = speedup_vs_fr > 1.2

    # K783: Cross-segment context improves PPL vs isolated
    if iso_ppl > 0:
        context_improvement_pct = (iso_ppl - kv_ppl) / iso_ppl * 100
    else:
        context_improvement_pct = 0

    k783_pass = context_improvement_pct > 0  # any improvement = context helps

    log(f"\n  === Kill Criteria ===")
    log(f"  K781: |KV-reuse - full-recompute| / full-recompute = {kv_vs_fr_pct:.2f}% "
        f"(threshold <= 3%) -> {'PASS' if k781_pass else 'FAIL'}")
    log(f"  K782: Speedup vs full-recompute = {speedup_vs_fr:.2f}x "
        f"(threshold > 1.2x) -> {'PASS' if k782_pass else 'FAIL'}")
    log(f"  K783: Context improvement = {context_improvement_pct:+.2f}% "
        f"(threshold > 0%) -> {'PASS' if k783_pass else 'FAIL'}")

    log(f"\n  Latency breakdown:")
    log(f"    Isolated:       {mean_latency_iso:.4f}s")
    log(f"    KV-reuse:       {mean_latency_kv:.4f}s")
    log(f"    Full-recompute: {mean_latency_fr:.4f}s")

    # Compute per-pair analysis
    pair_analysis = {}
    for pair_key, result in pair_results.items():
        iso_b = result["isolated_seg_b_ppl"]
        kv_b = result["kv_reuse_seg_b_ppl"]
        fr_b = result["full_recompute_seg_b_ppl"]

        if fr_b > 0 and fr_b != float("inf"):
            kv_vs_fr = abs(kv_b - fr_b) / fr_b * 100
        else:
            kv_vs_fr = float("inf")

        if iso_b > 0:
            ctx_imp = (iso_b - kv_b) / iso_b * 100
        else:
            ctx_imp = 0

        pair_analysis[pair_key] = {
            "isolated_seg_b_ppl": iso_b,
            "kv_reuse_seg_b_ppl": kv_b,
            "full_recompute_seg_b_ppl": fr_b,
            "kv_vs_full_recompute_pct": round(kv_vs_fr, 2),
            "context_improvement_pct": round(ctx_imp, 2),
            "mean_latency_kv_s": result["mean_latency_kv_reuse_s"],
            "mean_latency_fr_s": result["mean_latency_full_recompute_s"],
        }

    return {
        "avg_ppls": avg_ppls,
        "kv_vs_full_recompute_pct": round(kv_vs_fr_pct, 2),
        "speedup_vs_full_recompute": round(speedup_vs_fr, 2),
        "context_improvement_pct": round(context_improvement_pct, 2),
        "mean_latency_isolated_s": round(mean_latency_iso, 4),
        "mean_latency_kv_reuse_s": round(mean_latency_kv, 4),
        "mean_latency_full_recompute_s": round(mean_latency_fr, 4),
        "pair_analysis": pair_analysis,
        "k781_value": round(kv_vs_fr_pct, 2),
        "k781_pass": k781_pass,
        "k782_value": round(speedup_vs_fr, 2),
        "k782_pass": k782_pass,
        "k783_value": round(context_improvement_pct, 2),
        "k783_pass": k783_pass,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_start = time.time()
    log("=" * 70)
    log("KV-Cache Reuse Across Adapter Switches: Segment-Isolated Routing")
    log("=" * 70)
    log(f"Model: {MODEL_ID}")
    log(f"Domains: {DOMAINS}")
    log(f"Adapters: {ADAPTER_DIR}")
    log(f"Segment length: {SEGMENT_LENGTH}")
    log(f"Sequences per pair: {N_SEQUENCES_PER_PAIR}")
    log_memory("start")

    # Phase 0: Load tokenizer and construct data
    log("\n  Loading tokenizer...")
    _, tokenizer = load(MODEL_ID)
    mixed_sequences = phase_construct_mixed_data(tokenizer)
    del tokenizer
    gc.collect()

    # Phase 1: Evaluate all strategies
    pair_results, global_stats, latency_stats = phase_evaluate(MODEL_ID, mixed_sequences)

    # Phase 2: Analyze
    analysis = phase_analyze(pair_results, global_stats, latency_stats)

    # Compile final results
    results = {
        "experiment": "kv_cache_reuse_segment_routing",
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
        "kv_vs_full_recompute_pct": analysis["kv_vs_full_recompute_pct"],
        "speedup_vs_full_recompute": analysis["speedup_vs_full_recompute"],
        "context_improvement_pct": analysis["context_improvement_pct"],
        "latency": {
            "mean_isolated_s": analysis["mean_latency_isolated_s"],
            "mean_kv_reuse_s": analysis["mean_latency_kv_reuse_s"],
            "mean_full_recompute_s": analysis["mean_latency_full_recompute_s"],
        },
        "pair_analysis": analysis["pair_analysis"],
        "K781_kv_vs_fr_pct": analysis["k781_value"],
        "K781_threshold": 3.0,
        "K781_pass": analysis["k781_pass"],
        "K782_speedup": analysis["k782_value"],
        "K782_threshold": 1.2,
        "K782_pass": analysis["k782_pass"],
        "K783_context_improvement_pct": analysis["k783_value"],
        "K783_threshold": 0.0,
        "K783_pass": analysis["k783_pass"],
        "total_time_s": round(time.time() - t_start, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n  Results saved to {RESULTS_FILE}")

    # Final summary
    log("\n" + "=" * 70)
    log("FINAL RESULTS")
    log("=" * 70)

    for s_name, ppl in analysis["avg_ppls"].items():
        log(f"  {s_name:30s}: PPL {ppl:.4f}")

    log(f"\n  KV-reuse vs full-recompute gap: {analysis['kv_vs_full_recompute_pct']:+.2f}%")
    log(f"  Context improvement (vs isolated): {analysis['context_improvement_pct']:+.2f}%")
    log(f"  Speedup vs full-recompute: {analysis['speedup_vs_full_recompute']:.2f}x")

    log(f"\n  K781: {analysis['k781_value']:.2f}% (<= 3%?) -> "
        f"{'PASS' if analysis['k781_pass'] else 'FAIL'}")
    log(f"  K782: {analysis['k782_value']:.2f}x (> 1.2x?) -> "
        f"{'PASS' if analysis['k782_pass'] else 'FAIL'}")
    log(f"  K783: {analysis['k783_value']:+.2f}% (> 0%?) -> "
        f"{'PASS' if analysis['k783_pass'] else 'FAIL'}")
    log(f"\n  Total time: {results['total_time_s']:.1f}s")
    log_memory("end")


if __name__ == "__main__":
    main()
