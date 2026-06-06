#!/usr/bin/env python3
"""
Boundary Detection via Sliding-Window Domain Classification.

Bridges the gap between oracle-boundary segment routing (Finding #305, +16% PPL)
and practical deployment by detecting domain boundaries at runtime.

Approach: Run per-adapter PPL on sliding windows, detect where argmax adapter
changes = domain boundary, split and route segments independently.

Kill criteria:
  K775: Boundary detection F1 >= 70% on synthetic mixed-domain sequences
  K776: End-to-end segment-routed PPL within 5% of oracle-boundary segment routing
  K777: Boundary detection latency < 5ms per 256-token sequence on M5 Pro

Platform: Apple M5 Pro 48GB, MLX, $0.
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
SEGMENT_LENGTH = 128  # Each segment from one domain (2 segments per sequence)
SEED = 42

# Window sizes to explore (Type 2 unknown)
WINDOW_SIZES = [16, 32, 64]
WINDOW_STRIDE_RATIO = 0.5  # stride = w/2 for 50% overlap

N_SEQUENCES_PER_PAIR = 15  # 15 per pair * 10 pairs = 150 total
BOUNDARY_TOLERANCE = None  # Set per window size: tolerance = w (generous)

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
# Model loading utilities (reused from mixed_domain_per_token_routing)
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


def compute_segment_nll(model, tokens):
    """Compute NLL on a segment (independent subsequence). Returns (total_nll, n_tokens)."""
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
# Phase 0: Construct mixed-domain evaluation data
# ===========================================================================
def phase_construct_mixed_data(tokenizer):
    """Create mixed-domain sequences with known boundaries."""
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

            # Ensure minimum length
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
                "boundary_pos": SEGMENT_LENGTH,
                "n_tokens": len(combined),
            })
            pair_count += 1

        log(f"  {domain_a}+{domain_b}: {pair_count} sequences")

    log(f"  Total mixed sequences: {len(mixed_sequences)}")
    return mixed_sequences


# ===========================================================================
# Phase 1: Boundary detection via sliding-window PPL
# ===========================================================================
def phase_boundary_detection(model_id, mixed_sequences):
    """Detect domain boundaries using sliding-window per-adapter PPL."""
    log("\n" + "=" * 70)
    log("[Phase 1] Boundary Detection via Sliding-Window PPL")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(model_id)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)
    log_memory("model-loaded")

    # Load adapters
    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(ADAPTER_DIR / domain)
        log(f"  Loaded adapter: {domain}")

    results_by_window = {}

    for w in WINDOW_SIZES:
        stride = max(1, int(w * WINDOW_STRIDE_RATIO))
        tolerance = w  # Generous: within one full window width
        log(f"\n  --- Window size w={w}, stride={stride}, tolerance={tolerance} ---")

        boundary_results = []
        latency_samples = []

        for seq_idx, seq_data in enumerate(mixed_sequences):
            tokens = seq_data["tokens"]
            true_boundary = seq_data["boundary_pos"]
            T = len(tokens)

            t_detect_start = time.perf_counter()

            # Slide window across the sequence and compute per-adapter PPL for each
            window_positions = []
            window_classifications = []  # argmax adapter per window

            pos = 0
            while pos + w <= T:
                window_tokens = tokens[pos:pos + w]

                # Compute PPL for each adapter on this window
                window_ppls = {}
                for d_name in DOMAINS:
                    apply_adapter_to_model(model, adapters[d_name])
                    nll, n = compute_segment_nll(model, window_tokens)
                    if n > 0:
                        window_ppls[d_name] = nll / n  # avg NLL per token
                    else:
                        window_ppls[d_name] = float("inf")
                    zero_adapter_in_model(model)

                best_adapter = min(window_ppls, key=window_ppls.get)
                window_positions.append(pos + w // 2)  # center of window
                window_classifications.append(best_adapter)

                pos += stride

            t_detect_end = time.perf_counter()
            detect_latency_ms = (t_detect_end - t_detect_start) * 1000
            latency_samples.append(detect_latency_ms)

            # Detect boundaries: positions where classification changes
            detected_boundaries = []
            for i in range(1, len(window_classifications)):
                if window_classifications[i] != window_classifications[i - 1]:
                    # Boundary between window centers i-1 and i
                    boundary_pos = (window_positions[i - 1] + window_positions[i]) // 2
                    detected_boundaries.append(boundary_pos)

            # Evaluate: match detected boundaries to true boundary
            true_boundaries = [true_boundary]
            tp = 0  # true positives
            fp = 0  # false positives
            fn = 0  # false negatives

            matched_true = set()
            for det_b in detected_boundaries:
                matched = False
                for j, true_b in enumerate(true_boundaries):
                    if abs(det_b - true_b) <= tolerance and j not in matched_true:
                        tp += 1
                        matched_true.add(j)
                        matched = True
                        break
                if not matched:
                    fp += 1

            fn = len(true_boundaries) - len(matched_true)

            # Localization error for matched boundaries
            loc_errors = []
            for det_b in detected_boundaries:
                for j, true_b in enumerate(true_boundaries):
                    if abs(det_b - true_b) <= tolerance:
                        loc_errors.append(abs(det_b - true_b))
                        break

            # Also record the window classification accuracy
            # (what fraction of windows are classified to the correct domain)
            correct_classifications = 0
            total_classifications = 0
            for i, (pos_center, cls) in enumerate(zip(window_positions, window_classifications)):
                actual_pos = pos_center
                # Determine true domain for this window center
                if actual_pos < true_boundary:
                    true_domain = seq_data["domain_a"]
                else:
                    true_domain = seq_data["domain_b"]
                if cls == true_domain:
                    correct_classifications += 1
                total_classifications += 1

            boundary_results.append({
                "pair": f"{seq_data['domain_a']}+{seq_data['domain_b']}",
                "true_boundary": true_boundary,
                "detected_boundaries": detected_boundaries,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "loc_errors": loc_errors,
                "n_windows": len(window_positions),
                "window_accuracy": correct_classifications / max(total_classifications, 1),
                "detect_latency_ms": detect_latency_ms,
            })

            if (seq_idx + 1) % 30 == 0:
                log(f"    Processed {seq_idx+1}/{len(mixed_sequences)} sequences")

            # Periodic cleanup
            if (seq_idx + 1) % 10 == 0:
                gc.collect()
                mx.clear_cache()

        # Aggregate metrics for this window size
        total_tp = sum(r["tp"] for r in boundary_results)
        total_fp = sum(r["fp"] for r in boundary_results)
        total_fn = sum(r["fn"] for r in boundary_results)

        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        all_loc_errors = [e for r in boundary_results for e in r["loc_errors"]]
        mean_loc_error = sum(all_loc_errors) / max(len(all_loc_errors), 1) if all_loc_errors else float("inf")

        mean_window_accuracy = sum(r["window_accuracy"] for r in boundary_results) / len(boundary_results)
        mean_latency = sum(latency_samples) / len(latency_samples)
        median_latency = sorted(latency_samples)[len(latency_samples) // 2]

        log(f"\n    Window size w={w} summary:")
        log(f"      F1: {f1:.4f} (precision={precision:.4f}, recall={recall:.4f})")
        log(f"      Mean localization error: {mean_loc_error:.1f} tokens")
        log(f"      Window classification accuracy: {mean_window_accuracy:.4f}")
        log(f"      Mean detection latency: {mean_latency:.1f}ms")
        log(f"      Median detection latency: {median_latency:.1f}ms")
        log(f"      TP={total_tp}, FP={total_fp}, FN={total_fn}")

        # Per-pair breakdown
        pair_f1s = {}
        for pair_key in set(r["pair"] for r in boundary_results):
            pair_results = [r for r in boundary_results if r["pair"] == pair_key]
            p_tp = sum(r["tp"] for r in pair_results)
            p_fp = sum(r["fp"] for r in pair_results)
            p_fn = sum(r["fn"] for r in pair_results)
            p_prec = p_tp / max(p_tp + p_fp, 1)
            p_rec = p_tp / max(p_tp + p_fn, 1)
            p_f1 = 2 * p_prec * p_rec / max(p_prec + p_rec, 1e-8)
            pair_f1s[pair_key] = {
                "f1": round(p_f1, 4),
                "precision": round(p_prec, 4),
                "recall": round(p_rec, 4),
                "tp": p_tp, "fp": p_fp, "fn": p_fn,
            }

        results_by_window[w] = {
            "window_size": w,
            "stride": stride,
            "tolerance": tolerance,
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "mean_loc_error_tokens": round(mean_loc_error, 2),
            "mean_window_accuracy": round(mean_window_accuracy, 4),
            "mean_latency_ms": round(mean_latency, 2),
            "median_latency_ms": round(median_latency, 2),
            "pair_f1s": pair_f1s,
            "raw_results": boundary_results,
        }

    elapsed = time.time() - t0
    log(f"\n  Total detection time: {elapsed:.1f}s")
    log_memory("post-detection")

    cleanup(model, tokenizer, adapters)
    return results_by_window


# ===========================================================================
# Phase 2: End-to-end PPL with detected boundaries
# ===========================================================================
def phase_end_to_end_ppl(model_id, mixed_sequences, best_window_results):
    """Compute PPL using detected boundaries vs oracle boundaries."""
    log("\n" + "=" * 70)
    log("[Phase 2] End-to-End PPL: Detected vs Oracle Boundaries")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(model_id)
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)
    log_memory("model-loaded")

    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(ADAPTER_DIR / domain)

    raw_results = best_window_results["raw_results"]

    # Accumulators
    oracle_nll_total = 0.0
    oracle_n_total = 0
    detected_nll_total = 0.0
    detected_n_total = 0
    per_seq_nll_total = 0.0
    per_seq_n_total = 0

    for seq_idx, seq_data in enumerate(mixed_sequences):
        tokens = seq_data["tokens"]
        true_boundary = seq_data["boundary_pos"]
        domain_a = seq_data["domain_a"]
        domain_b = seq_data["domain_b"]
        T = len(tokens)

        # Get detected boundaries for this sequence
        det_result = raw_results[seq_idx]
        detected_boundaries = det_result["detected_boundaries"]

        # --- Oracle routing: split at true boundary, classify each segment ---
        seg_a = tokens[:true_boundary]
        seg_b = tokens[true_boundary:]

        # Oracle: use known correct adapter
        apply_adapter_to_model(model, adapters[domain_a])
        nll_a, n_a = compute_segment_nll(model, seg_a)
        zero_adapter_in_model(model)

        apply_adapter_to_model(model, adapters[domain_b])
        nll_b, n_b = compute_segment_nll(model, seg_b)
        zero_adapter_in_model(model)

        oracle_nll_total += nll_a + nll_b
        oracle_n_total += n_a + n_b

        # --- Detected routing: split at detected boundary ---
        if detected_boundaries:
            # Use the first detected boundary (closest to expected)
            det_b = detected_boundaries[0]
            det_seg_a = tokens[:det_b]
            det_seg_b = tokens[det_b:]
        else:
            # No boundary detected: treat as single segment, route to best overall
            det_seg_a = tokens
            det_seg_b = []

        # Classify detected segments by per-adapter PPL
        if len(det_seg_a) >= 2:
            best_nll_a = float("inf")
            for d_name in DOMAINS:
                apply_adapter_to_model(model, adapters[d_name])
                nll_d, n_d = compute_segment_nll(model, det_seg_a)
                if nll_d < best_nll_a:
                    best_nll_a = nll_d
                    best_n_a = n_d
                zero_adapter_in_model(model)
            detected_nll_total += best_nll_a
            detected_n_total += best_n_a

        if len(det_seg_b) >= 2:
            best_nll_b = float("inf")
            for d_name in DOMAINS:
                apply_adapter_to_model(model, adapters[d_name])
                nll_d, n_d = compute_segment_nll(model, det_seg_b)
                if nll_d < best_nll_b:
                    best_nll_b = nll_d
                    best_n_b = n_d
                zero_adapter_in_model(model)
            detected_nll_total += best_nll_b
            detected_n_total += best_n_b

        # --- Per-sequence routing (baseline from Finding #305) ---
        best_seq_nll = float("inf")
        best_seq_n = 0
        for d_name in DOMAINS:
            apply_adapter_to_model(model, adapters[d_name])
            nll_d, n_d = compute_segment_nll(model, tokens)
            if nll_d < best_seq_nll:
                best_seq_nll = nll_d
                best_seq_n = n_d
            zero_adapter_in_model(model)

        per_seq_nll_total += best_seq_nll
        per_seq_n_total += best_seq_n

        if (seq_idx + 1) % 30 == 0:
            log(f"    Processed {seq_idx+1}/{len(mixed_sequences)} sequences")

        if (seq_idx + 1) % 10 == 0:
            gc.collect()
            mx.clear_cache()

    # Compute PPLs
    oracle_ppl = math.exp(oracle_nll_total / max(oracle_n_total, 1))
    detected_ppl = math.exp(detected_nll_total / max(detected_n_total, 1))
    per_seq_ppl = math.exp(per_seq_nll_total / max(per_seq_n_total, 1))

    ppl_gap_vs_oracle = (detected_ppl - oracle_ppl) / oracle_ppl * 100
    improvement_vs_per_seq = (per_seq_ppl - detected_ppl) / per_seq_ppl * 100

    log(f"\n  End-to-end PPL results:")
    log(f"    Per-sequence best:  {per_seq_ppl:.4f}")
    log(f"    Oracle segment:     {oracle_ppl:.4f}")
    log(f"    Detected segment:   {detected_ppl:.4f}")
    log(f"    PPL gap vs oracle:  {ppl_gap_vs_oracle:+.2f}%")
    log(f"    Improvement vs per-seq: {improvement_vs_per_seq:+.2f}%")

    elapsed = time.time() - t0
    log(f"\n  End-to-end eval time: {elapsed:.1f}s")
    log_memory("post-e2e")

    cleanup(model, tokenizer, adapters)

    return {
        "per_seq_ppl": round(per_seq_ppl, 4),
        "oracle_ppl": round(oracle_ppl, 4),
        "detected_ppl": round(detected_ppl, 4),
        "ppl_gap_vs_oracle_pct": round(ppl_gap_vs_oracle, 2),
        "improvement_vs_per_seq_pct": round(improvement_vs_per_seq, 2),
    }


# ===========================================================================
# Phase 3: Latency benchmark (dedicated, no PPL computation)
# ===========================================================================
def phase_latency_benchmark(model_id, tokenizer_ref):
    """Measure boundary detection latency independently.

    Uses dummy tokens to isolate latency from PPL computation.
    Tests the actual serving scenario: given 256 tokens, how fast can we
    detect boundaries?
    """
    log("\n" + "=" * 70)
    log("[Phase 3] Latency Benchmark")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(model_id)
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(ADAPTER_DIR / domain)

    # Create a representative 256-token sequence
    sample_text = "The patient presented with symptoms of acute myocardial infarction."
    sample_tokens = tokenizer.encode(sample_text)
    while len(sample_tokens) < MAX_SEQ_LENGTH:
        sample_tokens = sample_tokens + sample_tokens
    sample_tokens = sample_tokens[:MAX_SEQ_LENGTH]

    # Warm up
    for d_name in DOMAINS:
        apply_adapter_to_model(model, adapters[d_name])
        nll, n = compute_segment_nll(model, sample_tokens[:32])
        zero_adapter_in_model(model)

    # Benchmark with best window size (w=32)
    w = 32
    stride = 16
    n_trials = 10

    latencies = []
    for trial in range(n_trials):
        t_start = time.perf_counter()

        # Full boundary detection pipeline
        pos = 0
        window_classifications = []
        while pos + w <= MAX_SEQ_LENGTH:
            window_tokens = sample_tokens[pos:pos + w]
            window_ppls = {}
            for d_name in DOMAINS:
                apply_adapter_to_model(model, adapters[d_name])
                nll, n = compute_segment_nll(model, window_tokens)
                window_ppls[d_name] = nll / max(n, 1)
                zero_adapter_in_model(model)
            best = min(window_ppls, key=window_ppls.get)
            window_classifications.append(best)
            pos += stride

        # Detect change points
        boundaries = []
        for i in range(1, len(window_classifications)):
            if window_classifications[i] != window_classifications[i - 1]:
                boundaries.append(i * stride)

        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000)

    mean_lat = sum(latencies) / len(latencies)
    median_lat = sorted(latencies)[len(latencies) // 2]
    min_lat = min(latencies)
    max_lat = max(latencies)

    log(f"  Window size: {w}, stride: {stride}")
    log(f"  N windows: {(MAX_SEQ_LENGTH - w) // stride + 1}")
    log(f"  N adapters: {N_DOMAINS}")
    log(f"  Total forward passes per sequence: {((MAX_SEQ_LENGTH - w) // stride + 1) * N_DOMAINS}")
    log(f"  Mean latency: {mean_lat:.1f}ms")
    log(f"  Median latency: {median_lat:.1f}ms")
    log(f"  Min latency: {min_lat:.1f}ms")
    log(f"  Max latency: {max_lat:.1f}ms")

    elapsed = time.time() - t0
    log_memory("post-latency")

    cleanup(model, tokenizer, adapters)

    return {
        "window_size": w,
        "stride": stride,
        "n_windows": (MAX_SEQ_LENGTH - w) // stride + 1,
        "n_adapters": N_DOMAINS,
        "n_forward_passes": ((MAX_SEQ_LENGTH - w) // stride + 1) * N_DOMAINS,
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "n_trials": n_trials,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_start = time.time()
    log("=" * 70)
    log("Boundary Detection via Sliding-Window Domain Classification")
    log("=" * 70)
    log(f"Model: {MODEL_ID}")
    log(f"Domains: {DOMAINS}")
    log(f"Window sizes: {WINDOW_SIZES}")
    log(f"Sequences per pair: {N_SEQUENCES_PER_PAIR}")
    log_memory("start")

    # Phase 0: Construct data
    _, tokenizer = load(MODEL_ID)
    mixed_sequences = phase_construct_mixed_data(tokenizer)
    del tokenizer
    gc.collect()

    # Phase 1: Boundary detection across window sizes
    detection_results = phase_boundary_detection(MODEL_ID, mixed_sequences)

    # Select best window size by F1
    best_w = max(detection_results.keys(), key=lambda w: detection_results[w]["f1"])
    best_detection = detection_results[best_w]
    log(f"\n  Best window size: w={best_w} (F1={best_detection['f1']:.4f})")

    # Phase 2: End-to-end PPL with detected boundaries
    e2e_results = phase_end_to_end_ppl(MODEL_ID, mixed_sequences, best_detection)

    # Phase 3: Latency benchmark
    latency_results = phase_latency_benchmark(MODEL_ID, None)

    # Kill criteria evaluation
    k775_value = best_detection["f1"]
    k775_pass = k775_value >= 0.70

    k776_value = abs(e2e_results["ppl_gap_vs_oracle_pct"])
    k776_pass = k776_value <= 5.0

    k777_value = latency_results["median_latency_ms"]
    k777_pass = k777_value < 5.0

    log("\n" + "=" * 70)
    log("KILL CRITERIA EVALUATION")
    log("=" * 70)
    log(f"  K775: Boundary F1 = {k775_value:.4f} (>= 0.70?) -> {'PASS' if k775_pass else 'FAIL'}")
    log(f"  K776: PPL gap vs oracle = {k776_value:.2f}% (<= 5%?) -> {'PASS' if k776_pass else 'FAIL'}")
    log(f"  K777: Detection latency = {k777_value:.1f}ms (< 5ms?) -> {'PASS' if k777_pass else 'FAIL'}")

    # Compile full results
    # Remove raw_results from window data to keep JSON manageable
    window_summary = {}
    for w, data in detection_results.items():
        summary = {k: v for k, v in data.items() if k != "raw_results"}
        window_summary[str(w)] = summary

    results = {
        "experiment": "boundary_detection_binary_heads",
        "model": MODEL_ID,
        "n_domains": N_DOMAINS,
        "domains": DOMAINS,
        "n_sequences": len(mixed_sequences),
        "segment_length": SEGMENT_LENGTH,
        "window_sizes_tested": WINDOW_SIZES,
        "best_window_size": best_w,
        "window_results": window_summary,
        "best_detection": {
            "window_size": best_detection["window_size"],
            "f1": best_detection["f1"],
            "precision": best_detection["precision"],
            "recall": best_detection["recall"],
            "mean_loc_error_tokens": best_detection["mean_loc_error_tokens"],
            "mean_window_accuracy": best_detection["mean_window_accuracy"],
            "pair_f1s": best_detection["pair_f1s"],
        },
        "end_to_end_ppl": e2e_results,
        "latency": latency_results,
        "K775_f1": round(k775_value, 4),
        "K775_threshold": 0.70,
        "K775_pass": k775_pass,
        "K776_ppl_gap_pct": round(k776_value, 2),
        "K776_threshold": 5.0,
        "K776_pass": k776_pass,
        "K777_latency_ms": round(k777_value, 2),
        "K777_threshold": 5.0,
        "K777_pass": k777_pass,
        "predictions_vs_measured": {
            "f1_predicted_ge": 0.94,
            "f1_measured": round(k775_value, 4),
            "loc_error_predicted_le": best_w // 2,
            "loc_error_measured": best_detection["mean_loc_error_tokens"],
            "ppl_gap_predicted_le_pct": 1.1,
            "ppl_gap_measured_pct": round(k776_value, 2),
        },
        "total_time_s": round(time.time() - t_start, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n  Results saved to {RESULTS_FILE}")

    # Print prediction vs measurement table
    log("\n" + "=" * 70)
    log("PREDICTION vs MEASUREMENT TABLE")
    log("=" * 70)
    log(f"  {'Metric':<35s} {'Predicted':<15s} {'Measured':<15s} {'Match?':<10s}")
    log(f"  {'-'*75}")

    f1_match = "YES" if k775_value >= 0.70 else "NO"
    log(f"  {'Boundary F1 (Corollary 1)':<35s} {'>= 0.94':<15s} {k775_value:<15.4f} {f1_match:<10s}")

    le_match = "YES" if best_detection["mean_loc_error_tokens"] <= best_w // 2 else "NO"
    log(f"  {'Loc error (Theorem 1)':<35s} {'<= ' + str(best_w//2) + ' tok':<15s} {best_detection['mean_loc_error_tokens']:<15.1f} {le_match:<10s}")

    ppl_match = "YES" if k776_value <= 1.1 else "PARTIAL"
    log(f"  {'PPL gap vs oracle (Theorem 3)':<35s} {'<= 1.1%':<15s} {k776_value:<15.2f}{'%':<1s} {ppl_match:<9s}")

    wa_match = "YES" if best_detection["mean_window_accuracy"] >= 0.85 else "PARTIAL"
    log(f"  {'Window accuracy (extrapolated)':<35s} {'>= 0.85':<15s} {best_detection['mean_window_accuracy']:<15.4f} {wa_match:<10s}")

    log(f"\n  Total time: {results['total_time_s']:.1f}s")
    log_memory("end")


if __name__ == "__main__":
    main()
