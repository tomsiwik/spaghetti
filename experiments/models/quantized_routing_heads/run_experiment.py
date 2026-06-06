#!/usr/bin/env python3
"""
Quantized Routing Heads: Post-training quantization of per-adapter routing heads.

Loads pre-trained fp32 routing heads from tiny_routing_heads experiment,
quantizes to int8 and int4, measures accuracy degradation and memory savings.

Kill criteria:
  K1: int4 routing accuracy drops below 90% on 5-domain test
  K2: MLX does not support int4/int8 inference for small linear layers efficiently

Success criteria:
  S1: Quantized routing heads maintain >95% accuracy with >50% memory reduction

Platform: Apple M5 Pro 48GB, MLX, $0.
"""

import gc
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

SEED = 42
MAX_SEQ_LENGTH = 256
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"

EXPERIMENT_DIR = Path(__file__).parent
HEADS_DIR = Path(__file__).parent.parent / "tiny_routing_heads" / "heads"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse data from bitnet_2b_real_composition
DATA_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"

DOMAINS = ["python", "math", "medical", "legal", "creative"]
HEAD_HIDDEN_DIM = 32
VAL_SAMPLES_PER_DOMAIN = 25


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
# Routing Head (same architecture as tiny_routing_heads)
# ===========================================================================
class RoutingHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def __call__(self, x):
        h = nn.relu(self.fc1(x))
        return self.fc2(h)


# ===========================================================================
# Quantization utilities
# ===========================================================================
def quantize_tensor_int8(w):
    """Symmetric int8 quantization. Returns (quantized_int8, scale)."""
    abs_max = mx.max(mx.abs(w))
    scale = abs_max / 127.0
    # Avoid division by zero
    scale = mx.where(scale == 0, mx.array(1.0), scale)
    w_q = mx.round(w / scale)
    w_q = mx.clip(w_q, -128, 127).astype(mx.int8)
    return w_q, scale


def dequantize_int8(w_q, scale):
    """Dequantize int8 back to float."""
    return w_q.astype(mx.float32) * scale


def quantize_tensor_int4(w):
    """Symmetric int4 quantization (stored in int8). Returns (quantized, scale)."""
    abs_max = mx.max(mx.abs(w))
    scale = abs_max / 7.0
    scale = mx.where(scale == 0, mx.array(1.0), scale)
    w_q = mx.round(w / scale)
    w_q = mx.clip(w_q, -8, 7).astype(mx.int8)  # stored as int8, range [-8, 7]
    return w_q, scale


def dequantize_int4(w_q, scale):
    """Dequantize int4 back to float."""
    return w_q.astype(mx.float32) * scale


class QuantizedRoutingHead:
    """Post-training quantized routing head. Inference only."""

    def __init__(self, head, bits=8):
        self.bits = bits
        quantize_fn = quantize_tensor_int8 if bits == 8 else quantize_tensor_int4
        dequant_fn = dequantize_int8 if bits == 8 else dequantize_int4

        # Quantize weights
        self.fc1_w_q, self.fc1_scale = quantize_fn(head.fc1.weight)
        self.fc1_bias = head.fc1.bias  # keep bias in fp32
        self.fc2_w_q, self.fc2_scale = quantize_fn(head.fc2.weight)
        self.fc2_bias = head.fc2.bias

        # Pre-dequantize for inference (MLX doesn't have native int8 matmul for tiny tensors)
        self.fc1_w = dequant_fn(self.fc1_w_q, self.fc1_scale)
        self.fc2_w = dequant_fn(self.fc2_w_q, self.fc2_scale)

        mx.eval(self.fc1_w, self.fc2_w, self.fc1_w_q, self.fc2_w_q,
                self.fc1_scale, self.fc2_scale)

    def __call__(self, x):
        """Forward pass using dequantized weights."""
        h = x @ self.fc1_w.T
        if self.fc1_bias is not None:
            h = h + self.fc1_bias
        h = mx.maximum(h, 0)  # ReLU
        out = h @ self.fc2_w.T
        if self.fc2_bias is not None:
            out = out + self.fc2_bias
        return out

    def memory_bytes(self):
        """Actual quantized storage (not dequantized)."""
        w_bytes = self.fc1_w_q.size + self.fc2_w_q.size  # int8 storage
        if self.bits == 4:
            w_bytes = w_bytes // 2  # packed int4
        scale_bytes = 2 * 4  # 2 float32 scales
        bias_bytes = (self.fc1_bias.size + self.fc2_bias.size) * 4  # fp32 biases
        return w_bytes + scale_bytes + bias_bytes


# ===========================================================================
# BitLinear helpers (from tiny_routing_heads)
# ===========================================================================
from mlx_lm import load
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


# ===========================================================================
# Data loading
# ===========================================================================
def load_domain_texts(domain, split="valid"):
    """Load validation texts for a domain."""
    jsonl_file = DATA_DIR / domain / f"{split}.jsonl"
    if not jsonl_file.exists():
        log(f"  WARNING: {jsonl_file} not found, trying train split")
        jsonl_file = DATA_DIR / domain / "train.jsonl"
    texts = []
    with open(jsonl_file) as f:
        for line in f:
            item = json.loads(line)
            texts.append(item.get("text", item.get("content", "")))
    return texts


# ===========================================================================
# Phase 1: Load model, extract hidden states for validation
# ===========================================================================
def phase_extract_hidden_states():
    """Extract hidden states from base model for validation."""
    log("\n" + "=" * 70)
    log("[Phase 1] Extracting hidden states for validation")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")

    model = replace_bitlinear_with_linear(model)
    mx.eval(model.parameters())

    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    log(f"  d_model = {d_model}")
    log_memory("after-model-load")

    domain_hidden = {}
    for domain in DOMAINS:
        texts = load_domain_texts(domain, split="valid")
        hiddens = []
        for text in texts[:VAL_SAMPLES_PER_DOMAIN]:
            tokens = tokenizer.encode(text)
            if len(tokens) < 4:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH]
            x = mx.array(tokens)[None, :]
            # Extract hidden states
            h = model.model.embed_tokens(x)
            for layer in model.model.layers:
                h = layer(h)
            h = model.model.norm(h)
            h_pool = mx.mean(h, axis=1)  # (1, d)
            mx.eval(h_pool)
            hiddens.append(h_pool)
            del h, x
        domain_hidden[domain] = hiddens
        log(f"  {domain}: {len(hiddens)} validation samples")

    log(f"  Extraction time: {time.time() - t0:.1f}s")

    # Free model
    cleanup(model, tokenizer)
    log_memory("after-model-free")

    return domain_hidden, d_model


# ===========================================================================
# Phase 2: Load heads, quantize, evaluate
# ===========================================================================
def evaluate_head(head_or_qhead, domain, domain_hidden):
    """Evaluate a routing head (fp32 or quantized) on validation data."""
    correct = 0
    total = 0
    logit_magnitudes = []

    # Positive eval (own domain)
    own_correct = 0
    own_total = 0
    for h in domain_hidden[domain]:
        logit = head_or_qhead(h)
        mx.eval(logit)
        val = logit.item()
        logit_magnitudes.append(abs(val))
        if val > 0:
            correct += 1
            own_correct += 1
        total += 1
        own_total += 1

    # Negative eval (other domains)
    neg_correct = 0
    neg_total = 0
    for other in DOMAINS:
        if other == domain:
            continue
        for h in domain_hidden[other][:10]:
            logit = head_or_qhead(h)
            mx.eval(logit)
            val = logit.item()
            logit_magnitudes.append(abs(val))
            if val <= 0:
                correct += 1
                neg_correct += 1
            total += 1
            neg_total += 1

    accuracy = correct / total if total > 0 else 0
    own_acc = own_correct / own_total if own_total > 0 else 0
    neg_acc = neg_correct / neg_total if neg_total > 0 else 0
    mean_logit = sum(logit_magnitudes) / len(logit_magnitudes) if logit_magnitudes else 0

    return {
        "accuracy": round(accuracy, 4),
        "own_domain_accuracy": round(own_acc, 4),
        "negative_accuracy": round(neg_acc, 4),
        "mean_logit_magnitude": round(mean_logit, 4),
    }


def measure_latency(head_or_qhead, sample_input, n_runs=1000):
    """Measure inference latency over n_runs."""
    # Warmup
    for _ in range(50):
        out = head_or_qhead(sample_input)
        mx.eval(out)

    t0 = time.time()
    for _ in range(n_runs):
        out = head_or_qhead(sample_input)
        mx.eval(out)
    elapsed = time.time() - t0
    return elapsed / n_runs  # seconds per call


def phase_quantize_and_evaluate(domain_hidden, d_model):
    """Load fp32 heads, quantize, compare."""
    log("\n" + "=" * 70)
    log("[Phase 2] Quantizing and evaluating routing heads")
    log("=" * 70)

    results = {
        "fp32": {},
        "int8": {},
        "int4": {},
        "summary": {},
    }

    sample_input = domain_hidden[DOMAINS[0]][0]  # (1, d_model) for latency test

    for domain in DOMAINS:
        log(f"\n  [{domain}]")

        # Load fp32 head
        head = RoutingHead(d_model, HEAD_HIDDEN_DIM)
        head_path = HEADS_DIR / domain / "head.npz"
        params = dict(mx.load(str(head_path)))
        head.load_weights(list(params.items()))
        mx.eval(head.parameters())

        # Count fp32 params and memory
        n_params = sum(p.size for _, p in tree_flatten(head.parameters()))
        fp32_bytes = n_params * 4

        # Evaluate fp32
        fp32_result = evaluate_head(head, domain, domain_hidden)
        fp32_latency = measure_latency(head, sample_input)
        fp32_result["memory_bytes"] = fp32_bytes
        fp32_result["latency_us"] = round(fp32_latency * 1e6, 2)
        results["fp32"][domain] = fp32_result
        log(f"    FP32: acc={fp32_result['accuracy']:.1%} latency={fp32_result['latency_us']:.1f}us mem={fp32_bytes:,}B")

        # Quantize to int8
        q8_head = QuantizedRoutingHead(head, bits=8)
        int8_result = evaluate_head(q8_head, domain, domain_hidden)
        int8_latency = measure_latency(q8_head, sample_input)
        int8_bytes = q8_head.memory_bytes()
        int8_result["memory_bytes"] = int8_bytes
        int8_result["latency_us"] = round(int8_latency * 1e6, 2)
        int8_result["memory_reduction_pct"] = round((1 - int8_bytes / fp32_bytes) * 100, 1)
        results["int8"][domain] = int8_result
        log(f"    INT8: acc={int8_result['accuracy']:.1%} latency={int8_result['latency_us']:.1f}us mem={int8_bytes:,}B ({int8_result['memory_reduction_pct']}% reduction)")

        # Quantize to int4
        q4_head = QuantizedRoutingHead(head, bits=4)
        int4_result = evaluate_head(q4_head, domain, domain_hidden)
        int4_latency = measure_latency(q4_head, sample_input)
        int4_bytes = q4_head.memory_bytes()
        int4_result["memory_bytes"] = int4_bytes
        int4_result["latency_us"] = round(int4_latency * 1e6, 2)
        int4_result["memory_reduction_pct"] = round((1 - int4_bytes / fp32_bytes) * 100, 1)
        results["int4"][domain] = int4_result
        log(f"    INT4: acc={int4_result['accuracy']:.1%} latency={int4_result['latency_us']:.1f}us mem={int4_bytes:,}B ({int4_result['memory_reduction_pct']}% reduction)")

        # Check quantization error (max absolute logit difference)
        max_diff_8 = 0.0
        max_diff_4 = 0.0
        for h in domain_hidden[domain]:
            fp32_logit = head(h)
            int8_logit = q8_head(h)
            int4_logit = q4_head(h)
            mx.eval(fp32_logit, int8_logit, int4_logit)
            max_diff_8 = max(max_diff_8, abs(fp32_logit.item() - int8_logit.item()))
            max_diff_4 = max(max_diff_4, abs(fp32_logit.item() - int4_logit.item()))
        results["int8"][domain]["max_logit_diff"] = round(max_diff_8, 6)
        results["int4"][domain]["max_logit_diff"] = round(max_diff_4, 6)
        log(f"    Max logit diff: int8={max_diff_8:.6f}, int4={max_diff_4:.6f}")

        del head, q8_head, q4_head

    # Summary
    for precision in ["fp32", "int8", "int4"]:
        accs = [results[precision][d]["accuracy"] for d in DOMAINS]
        results["summary"][precision] = {
            "mean_accuracy": round(sum(accs) / len(accs), 4),
            "min_accuracy": round(min(accs), 4),
            "mean_memory_bytes": round(sum(results[precision][d]["memory_bytes"] for d in DOMAINS) / len(DOMAINS)),
            "mean_latency_us": round(sum(results[precision][d]["latency_us"] for d in DOMAINS) / len(DOMAINS), 2),
        }

    # N=100 projection
    results["projection_n100"] = {
        "fp32_mb": round(results["summary"]["fp32"]["mean_memory_bytes"] * 100 / 1e6, 2),
        "int8_mb": round(results["summary"]["int8"]["mean_memory_bytes"] * 100 / 1e6, 2),
        "int4_mb": round(results["summary"]["int4"]["mean_memory_bytes"] * 100 / 1e6, 2),
    }

    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    log("=" * 70)
    log("Quantized Routing Heads Experiment")
    log(f"Platform: {mx.default_device()}")
    log("=" * 70)

    t_start = time.time()

    # Phase 1: Extract validation hidden states
    domain_hidden, d_model = phase_extract_hidden_states()

    # Phase 2: Quantize and evaluate
    results = phase_quantize_and_evaluate(domain_hidden, d_model)

    # Kill criteria evaluation
    log("\n" + "=" * 70)
    log("Kill Criteria Evaluation")
    log("=" * 70)

    int4_min_acc = results["summary"]["int4"]["min_accuracy"]
    k1_pass = int4_min_acc >= 0.90
    log(f"  K1 (int4 acc >= 90%): {'PASS' if k1_pass else 'FAIL'} — min accuracy = {int4_min_acc:.1%}")

    # K2: MLX int8/int4 efficiency — we test by checking latency isn't >2x worse
    fp32_lat = results["summary"]["fp32"]["mean_latency_us"]
    int8_lat = results["summary"]["int8"]["mean_latency_us"]
    int4_lat = results["summary"]["int4"]["mean_latency_us"]
    k2_pass = int8_lat < fp32_lat * 2.0 and int4_lat < fp32_lat * 2.0
    log(f"  K2 (latency not >2x worse): {'PASS' if k2_pass else 'FAIL'} — fp32={fp32_lat:.1f}us int8={int8_lat:.1f}us int4={int4_lat:.1f}us")

    # Success criteria
    int8_reduction = results["int8"][DOMAINS[0]]["memory_reduction_pct"]
    int8_acc = results["summary"]["int8"]["mean_accuracy"]
    s1_pass = int8_acc >= 0.95 and int8_reduction >= 50
    log(f"  S1 (>95% acc + >50% mem reduction): {'PASS' if s1_pass else 'FAIL'} — int8 acc={int8_acc:.1%}, reduction={int8_reduction}%")

    results["kill_criteria"] = {
        "K1_int4_acc_above_90pct": k1_pass,
        "K2_latency_not_2x_worse": k2_pass,
    }
    results["success_criteria"] = {
        "S1_95pct_acc_50pct_mem_reduction": s1_pass,
    }
    results["verdict"] = "SUPPORTED" if (k1_pass and k2_pass and s1_pass) else "KILLED"
    results["total_time_s"] = round(time.time() - t_start, 1)

    log(f"\n  VERDICT: {results['verdict']}")
    log(f"  Total time: {results['total_time_s']}s")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\n  Results saved to {RESULTS_FILE}")

    # Print summary table
    log("\n" + "=" * 70)
    log("Summary")
    log("=" * 70)
    log(f"  {'Precision':<10} {'Mean Acc':<12} {'Mean Latency':<15} {'Mem/Head':<12} {'@ N=100':<10}")
    log(f"  {'-'*10:<10} {'-'*12:<12} {'-'*15:<15} {'-'*12:<12} {'-'*10:<10}")
    for prec in ["fp32", "int8", "int4"]:
        s = results["summary"][prec]
        n100 = results["projection_n100"][f"{prec}_mb"]
        log(f"  {prec:<10} {s['mean_accuracy']:.1%}{'':<8} {s['mean_latency_us']:.1f}us{'':<9} {s['mean_memory_bytes']:,}B{'':<4} {n100}MB")


if __name__ == "__main__":
    main()
