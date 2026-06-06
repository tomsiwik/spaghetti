#!/usr/bin/env python3
"""
Output-Averaging vs Parameter-Merging Comparison

Compares two composition strategies on BitNet-2B-4T with existing ternary LoRA
adapters (from bitnet_scale_n50):

1. Pre-merge: W_merged = W_base + (1/k) * sum(B_i @ A_i), single forward pass
2. Output-averaging: run each adapter separately, average logits

Tests at k=5, k=25, k=49 (49 adapters available from N=50 experiment).

Kill criteria:
  K1 (id=270): Output-averaging not better than pre-merge at any tested N -> KILL
  K2 (id=271): k forward passes too slow for interactive serving (> 200ms/tok) -> KILL

Platform: Apple M5 Pro 48GB, MLX 0.31.1, $0.
"""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Paths to existing infrastructure
N50_DIR = EXPERIMENT_DIR.parent / "bitnet_scale_n50"
ADAPTERS_DIR = N50_DIR / "adapters"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 128
VAL_BATCHES = 5  # 5 samples per domain -- balances signal vs runtime at k=49

# Data directories mapping (reuse from N=50 experiment)
N15_DATA = EXPERIMENT_DIR.parent / "bitnet_scale_n15" / "data"
CONV_DATA = EXPERIMENT_DIR.parent / "bitnet_ternary_convergence" / "data"
CAP_DATA = EXPERIMENT_DIR.parent / "capability_expert_taxonomy" / "data"
N25_DATA = EXPERIMENT_DIR.parent / "bitnet_scale_n25" / "data"
N50_DATA = N50_DIR / "data"

# Map domain names to data directories
DATA_DIRS = {
    # From bitnet_ternary_convergence
    "code": CONV_DATA / "code",
    "math": CONV_DATA / "math",
    "legal": CONV_DATA / "legal",
    "creative": CONV_DATA / "creative",
    # From bitnet_ternary_convergence (medical not in n15)
    "medical": CONV_DATA / "medical",
    "sql": N15_DATA / "sql",
    "javascript": N15_DATA / "javascript",
    "physics": N15_DATA / "physics",
    "chemistry": N15_DATA / "chemistry",
    "science": N15_DATA / "science",
    "wikitext": N15_DATA / "wikitext",
    "finance": N15_DATA / "finance",
    "cooking": N15_DATA / "cooking",
    "health": N15_DATA / "health",
    "dialogue": N15_DATA / "dialogue",
    # From capability_expert_taxonomy
    "reasoning": CAP_DATA / "reasoning",
    "instruction": CAP_DATA / "instruction",
    "conciseness": CAP_DATA / "conciseness",
    "safety": CAP_DATA / "safety",
    # From bitnet_scale_n25
    "multilingual": N25_DATA / "multilingual",
    "coding_style": N25_DATA / "coding_style",
    "summarization": N25_DATA / "summarization",
    "debate": N25_DATA / "debate",
    "translation": N25_DATA / "translation",
    "formal_writing": N25_DATA / "formal_writing",
    # From bitnet_scale_n50 (new domains)
    "history": N50_DATA / "history",
    "philosophy": N50_DATA / "philosophy",
    "sports": N50_DATA / "sports",
    "poetry": N50_DATA / "poetry",
    "news": N50_DATA / "news",
    "reviews": N50_DATA / "reviews",
    "qa_pairs": N50_DATA / "qa_pairs",
    "stories": N50_DATA / "stories",
    "science_qa": N50_DATA / "science_qa",
    "recipes": N50_DATA / "recipes",
    "trivia": N50_DATA / "trivia",
    "eli5": N50_DATA / "eli5",
    "movie_plots": N50_DATA / "movie_plots",
    "tweets": N50_DATA / "tweets",
    "abstracts": N50_DATA / "abstracts",
    "contracts": N50_DATA / "contracts",
    "emails": N50_DATA / "emails",
    "bash_code": N50_DATA / "bash_code",
    "math_proofs": N50_DATA / "math_proofs",
    "dialogues_2": N50_DATA / "dialogues_2",
    "product_desc": N50_DATA / "product_desc",
    "bio_text": N50_DATA / "bio_text",
    "travel": N50_DATA / "travel",
    "tech_docs": N50_DATA / "tech_docs",
    "music_text": N50_DATA / "music_text",
}


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
# BitLinear unpacking (from bitnet_2b_real_composition)
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
# TernaryLoRALinear (matches N=50 training)
# ===========================================================================
class TernaryLoRALinear(nn.Module):
    def __init__(self, base_linear, r=16, scale=20.0):
        super().__init__()
        self.linear = base_linear
        self.r = r
        self.scale = scale
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]
        s = 1.0 / math.sqrt(in_features)
        self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, r))
        self.lora_b = mx.zeros((r, out_features))

    def _ste_ternary(self, W):
        alpha = mx.mean(mx.abs(W)) + 1e-10
        W_scaled = W / alpha
        W_q = mx.clip(mx.round(W_scaled), -1.0, 1.0) * alpha
        return W + mx.stop_gradient(W_q - W)

    def __call__(self, x):
        base_out = self.linear(x)
        A = self._ste_ternary(self.lora_a)
        B = self._ste_ternary(self.lora_b)
        lora_out = (x @ A) @ B * self.scale
        return base_out + lora_out


def apply_ternary_lora(model, rank=16, scale=20.0):
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora = TernaryLoRALinear(module, r=rank, scale=scale)
                updates.append((key, lora))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    log(f"  Applied Ternary LoRA (r={rank}) to {count} layers")
    return model


def zero_lora_params(model):
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                in_dims = module.lora_a.shape[0]
                s = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(low=-s, high=s, shape=module.lora_a.shape)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.trainable_parameters())


def apply_adapter_weights(model, adapter_params, scale=1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


# ===========================================================================
# PPL evaluation (standard)
# ===========================================================================
def compute_ppl(model, tokenizer, data_path, max_batches=VAL_BATCHES):
    fpath = data_path / "valid.jsonl"
    if not fpath.exists():
        return float("inf")
    texts = []
    with open(fpath) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    total_loss = 0.0
    total_tokens = 0
    for text in texts[:max_batches]:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += y.size
        del logits, loss, x, y

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


# ===========================================================================
# Phase functions
# ===========================================================================
def phase_load_model():
    """Load BitNet-2B, unpack to bf16, apply ternary LoRA wrappers."""
    log("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Loaded in {time.time() - t0:.1f}s")

    log("  Unpacking ternary weights...")
    model = replace_bitlinear_with_linear(model)

    log("  Applying TernaryLoRALinear wrappers...")
    model = apply_ternary_lora(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Freeze base, enable LoRA (needed for weight setting to work)
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    mx.eval(model.parameters())
    log_memory("model-loaded")
    return model, tokenizer


def phase_discover_adapters():
    """Find all available adapters and matching data directories."""
    log("\n[Phase 1] Discovering adapters and data...")
    available = []
    for name in sorted(ADAPTERS_DIR.iterdir()):
        if (name / "adapter.npz").exists():
            domain = name.name
            if domain in DATA_DIRS:
                data_path = DATA_DIRS[domain]
                valid_file = data_path / "valid.jsonl"
                if valid_file.exists():
                    available.append(domain)

    log(f"  Found {len(available)} adapters with matching validation data")
    return available


def phase_base_ppl(model, tokenizer, eval_domains):
    """Compute base model PPL (LoRA zeroed) on eval domains."""
    log("\n[Phase 2] Base model PPL...")
    zero_lora_params(model)
    base_ppls = {}
    for domain in eval_domains:
        ppl = compute_ppl(model, tokenizer, DATA_DIRS[domain])
        base_ppls[domain] = round(ppl, 4)
    log(f"  Avg base PPL: {sum(base_ppls.values())/len(base_ppls):.2f}")
    return base_ppls


def phase_pre_merge(model, tokenizer, adapter_names, eval_domains):
    """Pre-merge k adapters with 1/k scaling, measure PPL and latency."""
    k = len(adapter_names)
    log(f"\n[Phase Pre-merge k={k}] Merging {k} adapters...")

    # Merge adapter weights with 1/k scaling
    merged = None
    for name in adapter_names:
        params = dict(mx.load(str(ADAPTERS_DIR / name / "adapter.npz")))
        if merged is None:
            merged = {key: v.astype(mx.float32) for key, v in params.items()}
        else:
            for key in merged:
                merged[key] = merged[key] + params[key].astype(mx.float32)
        del params

    merged = {key: (v / k).astype(mx.bfloat16) for key, v in merged.items()}
    mx.eval(merged)

    # Apply merged weights (already evaluated via mx.eval(merged) above)
    apply_adapter_weights(model, merged)
    del merged
    mx.clear_cache()

    # Measure PPL on eval domains
    ppls = {}
    total_time = 0.0
    total_tokens = 0
    for domain in eval_domains:
        fpath = DATA_DIRS[domain] / "valid.jsonl"
        texts = []
        with open(fpath) as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        for text in texts[:VAL_BATCHES]:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH + 1]
            x = mx.array(tokens[:-1])[None, :]
            y = mx.array(tokens[1:])[None, :]

            t0 = time.perf_counter()
            logits = model(x)
            mx.eval(logits)
            t1 = time.perf_counter()
            total_time += (t1 - t0)
            total_tokens += y.size

            loss = nn.losses.cross_entropy(logits, y, reduction="sum")
            mx.eval(loss)
            # Accumulate per-domain
            if domain not in ppls:
                ppls[domain] = {"loss": 0.0, "tokens": 0}
            ppls[domain]["loss"] += loss.item()
            ppls[domain]["tokens"] += y.size
            del logits, loss, x, y

    # Compute per-domain PPL
    result_ppls = {}
    for domain in eval_domains:
        if domain in ppls and ppls[domain]["tokens"] > 0:
            avg_loss = ppls[domain]["loss"] / ppls[domain]["tokens"]
            result_ppls[domain] = round(math.exp(min(avg_loss, 100)), 4)
        else:
            result_ppls[domain] = float("inf")

    ms_per_token = (total_time / total_tokens * 1000) if total_tokens > 0 else 0
    avg_ppl = sum(result_ppls.values()) / len(result_ppls)
    log(f"  Pre-merge k={k}: avg PPL={avg_ppl:.2f}, {ms_per_token:.2f} ms/tok")
    return result_ppls, ms_per_token


def phase_output_avg(model, tokenizer, adapter_names, eval_domains):
    """Output-averaging: run each adapter separately, average logits."""
    k = len(adapter_names)
    log(f"\n[Phase Output-avg k={k}] Averaging {k} adapters' logits...")

    # Pre-load all adapter params
    adapter_params_list = []
    for name in adapter_names:
        params = dict(mx.load(str(ADAPTERS_DIR / name / "adapter.npz")))
        adapter_params_list.append(params)
    mx.eval(*[v for p in adapter_params_list for v in p.values()])

    ppls = {}
    total_time = 0.0
    total_tokens = 0

    for domain in eval_domains:
        fpath = DATA_DIRS[domain] / "valid.jsonl"
        texts = []
        with open(fpath) as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        domain_loss = 0.0
        domain_tokens = 0

        for text in texts[:VAL_BATCHES]:
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:MAX_SEQ_LENGTH + 1]
            x = mx.array(tokens[:-1])[None, :]
            y = mx.array(tokens[1:])[None, :]

            t0 = time.perf_counter()

            # Run each adapter, accumulate logits
            # No need for zero_lora_params -- adapter files contain both
            # lora_a and lora_b, so apply_adapter_weights fully overwrites
            logits_sum = None
            for params in adapter_params_list:
                # Adapter params are already evaluated (loaded from npz).
                # model.update() just swaps references, no eval needed.
                apply_adapter_weights(model, params)

                logits_i = model(x)
                mx.eval(logits_i)

                if logits_sum is None:
                    logits_sum = logits_i
                else:
                    logits_sum = logits_sum + logits_i
                del logits_i

            logits_avg = logits_sum / k
            mx.eval(logits_avg)
            t1 = time.perf_counter()

            total_time += (t1 - t0)
            total_tokens += y.size

            loss = nn.losses.cross_entropy(logits_avg, y, reduction="sum")
            mx.eval(loss)
            domain_loss += loss.item()
            domain_tokens += y.size
            del logits_avg, logits_sum, loss, x, y

        if domain_tokens > 0:
            avg_loss = domain_loss / domain_tokens
            ppls[domain] = round(math.exp(min(avg_loss, 100)), 4)
        else:
            ppls[domain] = float("inf")

    del adapter_params_list
    gc.collect()
    mx.clear_cache()

    ms_per_token = (total_time / total_tokens * 1000) if total_tokens > 0 else 0
    avg_ppl = sum(ppls.values()) / len(ppls)
    log(f"  Output-avg k={k}: avg PPL={avg_ppl:.2f}, {ms_per_token:.2f} ms/tok")
    return ppls, ms_per_token


def phase_latency_benchmark(model, tokenizer, adapter_names):
    """Focused latency benchmark: time pre-merge vs output-avg per token."""
    k = len(adapter_names)
    log(f"\n[Phase Latency k={k}] Benchmarking...")

    # Generate a fixed input for consistent timing
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10  # 100 tokens
    x = mx.array(tokens[:MAX_SEQ_LENGTH])[None, :]
    mx.eval(x)

    # Pre-load adapter params
    adapter_params_list = []
    for name in adapter_names:
        params = dict(mx.load(str(ADAPTERS_DIR / name / "adapter.npz")))
        adapter_params_list.append(params)
    mx.eval(*[v for p in adapter_params_list for v in p.values()])

    # Warmup
    log("  Warming up...")
    for _ in range(3):
        apply_adapter_weights(model, adapter_params_list[0])
        out = model(x)
        mx.eval(out)
        del out

    # Benchmark pre-merge
    log("  Timing pre-merge...")
    merged = None
    for params in adapter_params_list:
        if merged is None:
            merged = {key: v.astype(mx.float32) for key, v in params.items()}
        else:
            for key in merged:
                merged[key] = merged[key] + params[key].astype(mx.float32)
    merged = {key: (v / k).astype(mx.bfloat16) for key, v in merged.items()}
    mx.eval(merged)

    apply_adapter_weights(model, merged)
    del merged

    # Time 10 forward passes for pre-merge
    pre_merge_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        out = model(x)
        mx.eval(out)
        t1 = time.perf_counter()
        pre_merge_times.append(t1 - t0)
        del out

    pre_merge_ms = sum(pre_merge_times) / len(pre_merge_times) * 1000
    pre_merge_ms_per_tok = pre_merge_ms / x.shape[1]

    # Benchmark output-averaging
    log("  Timing output-averaging...")
    oa_times = []
    for _ in range(3):  # Fewer iterations since each involves k forward passes
        t0 = time.perf_counter()
        logits_sum = None
        for params in adapter_params_list:
            apply_adapter_weights(model, params)
            logits_i = model(x)
            mx.eval(logits_i)
            if logits_sum is None:
                logits_sum = logits_i
            else:
                logits_sum = logits_sum + logits_i
            del logits_i
        logits_avg = logits_sum / k
        mx.eval(logits_avg)
        t1 = time.perf_counter()
        oa_times.append(t1 - t0)
        del logits_avg, logits_sum

    oa_ms = sum(oa_times) / len(oa_times) * 1000
    oa_ms_per_tok = oa_ms / x.shape[1]

    del adapter_params_list
    gc.collect()
    mx.clear_cache()

    log(f"  Pre-merge: {pre_merge_ms:.1f}ms total, {pre_merge_ms_per_tok:.2f}ms/tok")
    log(f"  Output-avg: {oa_ms:.1f}ms total, {oa_ms_per_tok:.2f}ms/tok")
    log(f"  Slowdown: {oa_ms/pre_merge_ms:.1f}x")

    return {
        "pre_merge_ms": round(pre_merge_ms, 2),
        "pre_merge_ms_per_tok": round(pre_merge_ms_per_tok, 4),
        "output_avg_ms": round(oa_ms, 2),
        "output_avg_ms_per_tok": round(oa_ms_per_tok, 4),
        "slowdown_factor": round(oa_ms / pre_merge_ms, 2),
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_start = time.time()
    results = {
        "experiment": "output_averaging_vs_param_merge",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    log("=" * 70)
    log("Output-Averaging vs Parameter-Merging Comparison")
    log("=" * 70)

    # Phase 0: Load model
    model, tokenizer = phase_load_model()

    # Phase 1: Discover adapters
    available = phase_discover_adapters()
    results["n_available_adapters"] = len(available)
    log(f"  Available: {len(available)} adapters")

    if len(available) < 5:
        log("FATAL: Need at least 5 adapters")
        results["error"] = "Not enough adapters"
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return

    # Select subsets for k=5, k=25, k=49
    # Use a fixed subset to ensure fair comparison
    k_values = []
    if len(available) >= 5:
        k_values.append(5)
    if len(available) >= 25:
        k_values.append(25)
    if len(available) >= 49:
        k_values.append(min(49, len(available)))

    results["k_values"] = k_values
    log(f"  Testing k={k_values}")

    # Use a representative subset of domains for evaluation
    # Pick 10 diverse domains that definitely have data
    eval_domains_priority = [
        "code", "math", "legal", "creative",
        "finance", "cooking", "eli5", "stories",
    ]
    eval_domains = [d for d in eval_domains_priority if d in available]
    if len(eval_domains) < 5:
        eval_domains = available[:10]
    results["eval_domains"] = eval_domains
    log(f"  Eval domains ({len(eval_domains)}): {eval_domains}")

    # Phase 2: Base PPL
    base_ppls = phase_base_ppl(model, tokenizer, eval_domains)
    results["base_ppls"] = base_ppls
    cleanup()
    log_memory("after-base-ppl")

    # Phase 3-5: Compare strategies at each k
    comparison_results = {}

    for k in k_values:
        log(f"\n{'='*60}")
        log(f"COMPARING AT k={k}")
        log(f"{'='*60}")

        adapter_subset = available[:k]

        # Pre-merge
        pm_ppls, pm_ms_tok = phase_pre_merge(
            model, tokenizer, adapter_subset, eval_domains
        )
        cleanup()
        log_memory(f"after-pre-merge-k{k}")

        # Output-averaging
        oa_ppls, oa_ms_tok = phase_output_avg(
            model, tokenizer, adapter_subset, eval_domains
        )
        cleanup()
        log_memory(f"after-output-avg-k{k}")

        # Latency benchmark
        latency = phase_latency_benchmark(model, tokenizer, adapter_subset)
        cleanup()
        log_memory(f"after-latency-k{k}")

        # Compare
        pm_avg = sum(pm_ppls.values()) / len(pm_ppls)
        oa_avg = sum(oa_ppls.values()) / len(oa_ppls)
        ppl_delta = (oa_avg - pm_avg) / pm_avg * 100  # negative = OA better

        n_oa_wins = sum(1 for d in eval_domains
                        if d in oa_ppls and d in pm_ppls
                        and oa_ppls[d] < pm_ppls[d])

        comparison_results[f"k={k}"] = {
            "k": k,
            "n_adapters_used": len(adapter_subset),
            "pre_merge": {
                "ppls": pm_ppls,
                "avg_ppl": round(pm_avg, 4),
                "ms_per_token": round(pm_ms_tok, 4),
            },
            "output_avg": {
                "ppls": oa_ppls,
                "avg_ppl": round(oa_avg, 4),
                "ms_per_token": round(oa_ms_tok, 4),
            },
            "latency_benchmark": latency,
            "ppl_delta_pct": round(ppl_delta, 2),
            "oa_better_domains": n_oa_wins,
            "oa_wins": oa_avg < pm_avg,
        }

        log(f"\n  Summary k={k}:")
        log(f"    Pre-merge avg PPL:  {pm_avg:.4f}")
        log(f"    Output-avg avg PPL: {oa_avg:.4f}")
        log(f"    Delta: {ppl_delta:+.2f}% ({'OA better' if oa_avg < pm_avg else 'PM better'})")
        log(f"    OA wins {n_oa_wins}/{len(eval_domains)} domains")
        log(f"    Latency: PM={latency['pre_merge_ms_per_tok']:.2f} ms/tok, "
            f"OA={latency['output_avg_ms_per_tok']:.2f} ms/tok "
            f"({latency['slowdown_factor']}x slower)")

    results["comparisons"] = comparison_results

    # ===========================================================================
    # Kill criteria assessment
    # ===========================================================================
    log("\n" + "=" * 70)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 70)

    # K1: Output-averaging not better than pre-merge at any tested N
    oa_wins_any = any(
        comparison_results[f"k={k}"]["oa_wins"] for k in k_values
    )
    k1_pass = oa_wins_any  # PASS = OA wins at least one k
    results["k1_oa_wins_any_k"] = oa_wins_any
    results["k1_pass"] = k1_pass
    log(f"\n  K1: OA better at any k? {'YES -> PASS' if k1_pass else 'NO -> KILL'}")
    for k in k_values:
        cr = comparison_results[f"k={k}"]
        log(f"    k={k}: OA avg PPL={cr['output_avg']['avg_ppl']:.4f} vs "
            f"PM avg PPL={cr['pre_merge']['avg_ppl']:.4f} "
            f"({cr['ppl_delta_pct']:+.2f}%)")

    # K2: k forward passes too slow (> 200ms/token)
    max_oa_ms_tok = max(
        comparison_results[f"k={k}"]["latency_benchmark"]["output_avg_ms_per_tok"]
        for k in k_values
    )
    k2_under_threshold = all(
        comparison_results[f"k={k}"]["latency_benchmark"]["output_avg_ms_per_tok"] <= 200
        for k in k_values
    )
    # K2 is about serving feasibility, test at max k
    k2_max_k = max(k_values)
    k2_ms = comparison_results[f"k={k2_max_k}"]["latency_benchmark"]["output_avg_ms_per_tok"]
    k2_pass = k2_ms <= 200  # PASS means it's fast enough (not killed)
    results["k2_max_oa_ms_per_tok"] = round(max_oa_ms_tok, 4)
    results["k2_pass"] = k2_pass
    log(f"\n  K2: OA ms/tok at k={k2_max_k}: {k2_ms:.2f}ms "
        f"({'<= 200ms -> PASS' if k2_pass else '> 200ms -> KILL'})")
    for k in k_values:
        lat = comparison_results[f"k={k}"]["latency_benchmark"]
        log(f"    k={k}: PM={lat['pre_merge_ms_per_tok']:.2f}ms/tok, "
            f"OA={lat['output_avg_ms_per_tok']:.2f}ms/tok")

    # Overall verdict
    if not k1_pass:
        verdict = "KILLED"
        reason = "K1 FAIL: Output-averaging never beats pre-merge"
    elif not k2_pass:
        verdict = "KILLED"
        reason = f"K2 FAIL: OA too slow at k={k2_max_k} ({k2_ms:.0f}ms/tok > 200ms)"
    else:
        verdict = "SUPPORTED"
        reason = "Both K1 and K2 pass"

    results["verdict"] = verdict
    results["verdict_reason"] = reason
    results["total_time_s"] = round(time.time() - t_start, 1)

    log(f"\n  VERDICT: {verdict}")
    log(f"  Reason: {reason}")
    log(f"  Total time: {results['total_time_s']:.0f}s")

    # Save
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
