#!/usr/bin/env python3
"""
T4.3v2: MLX Adapter Serving — Real Loophole Fixes

Fixes from LOOPHOLE_AUDIT (exp_p1_t4_vllm_adapter_serving):
1. K1240: Swap latency includes first forward pass (not just weight load)
2. K1241: Decode-only throughput via stream_generate.generation_tps (not prefill+decode)
           Compare base (no LoRA) vs LoRA-active model on same prompt
3. K1242: Real TF-IDF+Ridge router (not dict lookup), timing includes vectorize+predict

Kill criteria:
    K1240: Swap + first-forward p50 latency < 100ms (warm swaps after warmup)
    K1241: Decode-only throughput degradation < 15% (LoRA vs base, same prompt)
    K1242: Real TF-IDF routing latency p99 < 5ms at N=5 domains
"""

import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from datasets import load_dataset
from mlx_lm import load, stream_generate
from mlx_lm.tuner.utils import load_adapters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

MODEL_PATH = "mlx-community/gemma-4-e4b-it-4bit"

# Adapters from P4.C1 (real trained, v_proj+o_proj, rank-16, ~46MB each)
P4C1_DIR = EXPERIMENT_DIR.parent / "exp_p4_c1_vproj_soap_adapter"
ADAPTER_SOAP = P4C1_DIR / "soap_adapter"
ADAPTER_LEGAL = P4C1_DIR / "legal_adapter"
ADAPTER_LATEX = P4C1_DIR / "latex_adapter"
ADAPTERS = [ADAPTER_SOAP, ADAPTER_LEGAL, ADAPTER_LATEX]

# Test parameters
N_SWAPS = 6 if IS_SMOKE else 15       # swap trials for K1240 (first is cold, rest warm)
N_DECODE = 50 if IS_SMOKE else 150    # decode tokens for K1241 (longer = more signal)
N_ROUTE_TEST = 30 if IS_SMOKE else 200  # routing latency test queries
N_TRAIN_ROUTE = 50 if IS_SMOKE else 300  # training prompts per domain for router
N_ROUTE_ACCURACY = 20 if IS_SMOKE else 50  # accuracy test per domain
SEED = 42
rng = np.random.default_rng(SEED)

# Short prompt: 1 token prefill → >99% of timing is decode
DECODE_PROMPT = "The"

# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _mmlu(subject: str, n: int, rng_local: np.random.Generator) -> list[str]:
    # Try auxiliary_train first (larger), fall back to test/validation
    for split in ["auxiliary_train", "test", "validation"]:
        try:
            ds = load_dataset("cais/mmlu", subject, split=split)
            break
        except Exception:
            continue
    qs = list(dict.fromkeys(ex["question"] for ex in ds))  # deduplicate
    # Recycle if not enough samples
    while len(qs) < n:
        qs = qs + qs
    idx = rng_local.permutation(len(qs))[:n]
    return [qs[i] for i in idx]

def _gsm8k(n: int, rng_local: np.random.Generator) -> list[str]:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    qs = [ex["question"] for ex in ds]
    idx = rng_local.permutation(len(qs))[:n]
    return [qs[i] for i in idx]

def _code(n: int, rng_local: np.random.Generator) -> list[str]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    prompts = [ex["prompt"] for ex in ds]
    while len(prompts) < n:
        prompts = prompts + prompts
    idx = rng_local.permutation(len(prompts))[:n]
    return [prompts[i] for i in idx]

def _pubmed(n: int, rng_local: np.random.Generator) -> list[str]:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    qs = [ex["question"] for ex in ds]
    idx = rng_local.permutation(len(qs))[:n]
    return [qs[i] for i in idx]

# ─────────────────────────────────────────────────────────────────────────────
# K1242: Real TF-IDF Router
# ─────────────────────────────────────────────────────────────────────────────

def build_router(n_train: int, n_test: int) -> tuple:
    """
    Train TF-IDF+Ridge router on 5 domains. Use disjoint test subjects.
    Returns (vectorizer, classifier, accuracy, test_prompts)
    """
    print(f"[router] Building TF-IDF router ({n_train} train, {n_test} test per domain)...")
    rng_train = np.random.default_rng(SEED)
    rng_test = np.random.default_rng(SEED + 1)

    domains = ["math", "code", "medical", "legal", "finance"]
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    domain_loaders_train = {
        "math": lambda n: _gsm8k(n, rng_train),
        "code": lambda n: _code(n, rng_train),
        "medical": lambda n: _pubmed(n, rng_train),
        "legal": lambda n: _mmlu("international_law", n, rng_train),
        "finance": lambda n: _mmlu("high_school_macroeconomics", n, rng_train),
    }
    domain_loaders_test = {
        "math": lambda n: _gsm8k(n, rng_test),
        "code": lambda n: _code(n, rng_test),
        "medical": lambda n: _pubmed(n, rng_test),
        "legal": lambda n: _mmlu("professional_law", n, rng_test),      # different subject
        "finance": lambda n: _mmlu("econometrics", n, rng_test),         # different subject
    }

    for domain in domains:
        train_texts.extend(domain_loaders_train[domain](n_train))
        train_labels.extend([domain] * n_train)
        test_texts.extend(domain_loaders_test[domain](n_test))
        test_labels.extend([domain] * n_test)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    clf = RidgeClassifier(alpha=1.0)
    X_train = vectorizer.fit_transform(train_texts)
    clf.fit(X_train, train_labels)

    X_test = vectorizer.transform(test_texts)
    preds = clf.predict(X_test)
    accuracy = float(np.mean(np.array(preds) == np.array(test_labels)))
    print(f"[router] Router accuracy: {accuracy:.3f}")
    return vectorizer, clf, accuracy, test_texts

def measure_route_latency(vectorizer, clf, prompts: list[str]) -> dict:
    """Per-query latency: vectorize([prompt]) + clf.predict(). Report p50/p99."""
    latencies_ms = []
    for p in prompts:
        t0 = time.perf_counter()
        X = vectorizer.transform([p])
        _ = clf.predict(X)[0]
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)
    return {
        "n_queries": len(latencies_ms),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "max_ms": float(np.max(latencies_ms)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# K1241: Decode-Only Throughput
# ─────────────────────────────────────────────────────────────────────────────

def decode_tps(model, tokenizer, prompt: str, n_tokens: int) -> float:
    """
    Return decode-only tps from stream_generate's .generation_tps field.
    Uses 1-token prompt so >99% of timing is decode, not prefill.
    """
    last = None
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=n_tokens):
        last = resp
    return float(last.generation_tps) if (last and last.generation_tokens > 0) else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# K1240: Hot-Swap + First Forward Latency
# ─────────────────────────────────────────────────────────────────────────────

def swap_and_first_forward_ms(model, tokenizer, adapter_path: Path, input_ids=None) -> dict:
    """
    True hot-swap latency measured in 3 components:
      1. model.load_weights()     — replace adapter weights in model
      2. mx.eval(parameters())    — force device transfer (materialize on GPU)
      3. model(input_ids) + eval  — raw first forward pass (no stream_generate overhead)
    Returns dict with component timings and total.
    """
    t0 = time.perf_counter()
    model.load_weights(str(adapter_path / "adapters.safetensors"), strict=False)
    t1 = time.perf_counter()
    mx.eval(model.parameters())
    t2 = time.perf_counter()
    # Raw forward pass — captures any graph recompilation without stream_generate overhead
    out = model(input_ids)
    mx.eval(out)
    t3 = time.perf_counter()
    return {
        "load_ms": (t1 - t0) * 1000,
        "eval_ms": (t2 - t1) * 1000,
        "forward_ms": (t3 - t2) * 1000,
        "total_ms": (t3 - t0) * 1000,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    results = {"is_smoke": IS_SMOKE}

    # ── K1242: TF-IDF Router ───────────────────────────────────────────────
    print("\n=== K1242: Real TF-IDF Routing Latency ===")
    vectorizer, clf, router_accuracy, test_prompts = build_router(
        n_train=N_TRAIN_ROUTE, n_test=N_ROUTE_ACCURACY
    )
    # Use math test prompts for latency measurement (neutral domain)
    latency_prompts = _gsm8k(N_ROUTE_TEST, np.random.default_rng(SEED + 99))
    route_lat = measure_route_latency(vectorizer, clf, latency_prompts)

    k1242_pass = route_lat["p99_ms"] < 5.0
    results["k1242"] = {
        **route_lat,
        "accuracy": router_accuracy,
        "n_domains": 5,
        "n_train_per_domain": N_TRAIN_ROUTE,
        "threshold_p99_ms": 5.0,
        "k1242_pass": k1242_pass,
    }
    print(f"[K1242] accuracy={router_accuracy:.3f} p99={route_lat['p99_ms']:.3f}ms | pass={k1242_pass}")

    # ── Load LLM (base model, no LoRA) ────────────────────────────────────
    print(f"\n=== Loading base model (no LoRA): {MODEL_PATH} ===")
    model_base, tokenizer = load(MODEL_PATH)
    mx.eval(model_base.parameters())

    # ── K1241: Base decode throughput ──────────────────────────────────────
    print("\n=== K1241: Base Decode Throughput ===")
    # Warmup base model
    print("[warmup] Warming up base model...")
    _ = decode_tps(model_base, tokenizer, DECODE_PROMPT, n_tokens=20)
    _ = decode_tps(model_base, tokenizer, DECODE_PROMPT, n_tokens=20)

    # Measure base decode tps (3 trials, take median)
    base_tps_trials = [
        decode_tps(model_base, tokenizer, DECODE_PROMPT, n_tokens=N_DECODE)
        for _ in range(3)
    ]
    base_tps = float(np.median(base_tps_trials))
    print(f"[K1241] Base model decode: {base_tps:.1f} tok/s (trials: {[f'{t:.1f}' for t in base_tps_trials]})")

    # Free base model memory
    del model_base
    mx.clear_cache()

    # ── Load LoRA model ────────────────────────────────────────────────────
    print(f"\n=== Loading LoRA model (SOAP adapter): {MODEL_PATH} ===")
    model_lora, _ = load(MODEL_PATH)
    model_lora = load_adapters(model_lora, str(ADAPTER_SOAP))
    mx.eval(model_lora.parameters())

    # ── K1241: LoRA decode throughput ─────────────────────────────────────
    print("[warmup] Warming up LoRA model...")
    _ = decode_tps(model_lora, tokenizer, DECODE_PROMPT, n_tokens=20)
    _ = decode_tps(model_lora, tokenizer, DECODE_PROMPT, n_tokens=20)

    lora_tps_trials = [
        decode_tps(model_lora, tokenizer, DECODE_PROMPT, n_tokens=N_DECODE)
        for _ in range(3)
    ]
    lora_tps = float(np.median(lora_tps_trials))
    degradation = (base_tps - lora_tps) / base_tps if base_tps > 0 else 1.0
    k1241_pass = degradation < 0.15
    print(f"[K1241] LoRA decode: {lora_tps:.1f} tok/s (trials: {[f'{t:.1f}' for t in lora_tps_trials]})")
    print(f"[K1241] Degradation: {degradation*100:.1f}% | pass={k1241_pass}")

    results["k1241"] = {
        "base_tps": base_tps,
        "lora_tps": lora_tps,
        "base_tps_trials": base_tps_trials,
        "lora_tps_trials": lora_tps_trials,
        "degradation_pct": degradation * 100,
        "threshold_pct": 15.0,
        "k1241_pass": k1241_pass,
        "n_decode_tokens": N_DECODE,
        "prompt_tokens": 1,
        "adapter": "soap_v_proj_o_proj_rank16",
    }

    # ── K1240: Swap + First Forward Latency ────────────────────────────────
    print("\n=== K1240: Swap + First Forward Latency ===")
    # Prepare a single-token input for raw forward pass measurement
    swap_input_ids = mx.array([[tokenizer.encode("x")[-1]]])
    swap_trials = []
    adapter_cycle = ADAPTERS * ((N_SWAPS // len(ADAPTERS)) + 1)
    for trial in range(N_SWAPS):
        adapter = adapter_cycle[trial]
        result = swap_and_first_forward_ms(model_lora, tokenizer, adapter, swap_input_ids)
        swap_trials.append(result)
        print(f"[K1240] Trial {trial+1}/{N_SWAPS}: total={result['total_ms']:.1f}ms "
              f"(load={result['load_ms']:.1f} eval={result['eval_ms']:.1f} "
              f"fwd={result['forward_ms']:.1f}) [{adapter.name}]")

    # Extract total latencies
    swap_totals = [t["total_ms"] for t in swap_trials]
    cold_ms = swap_totals[0] if swap_totals else 0.0
    warm_totals = swap_totals[1:] if len(swap_totals) > 1 else swap_totals
    p50_warm = float(np.percentile(warm_totals, 50))
    p99_warm = float(np.percentile(warm_totals, 99))
    p50_all = float(np.percentile(swap_totals, 50))
    k1240_pass = p50_warm < 100.0  # threshold on warm swaps

    # Component-level stats (warm trials only)
    warm_trials = swap_trials[1:] if len(swap_trials) > 1 else swap_trials
    avg_load = float(np.mean([t["load_ms"] for t in warm_trials]))
    avg_eval = float(np.mean([t["eval_ms"] for t in warm_trials]))
    avg_forward = float(np.mean([t["forward_ms"] for t in warm_trials]))

    results["k1240"] = {
        "all_trials": swap_trials,
        "cold_trial_ms": cold_ms,
        "warm_totals_ms": warm_totals,
        "p50_warm_ms": p50_warm,
        "p99_warm_ms": p99_warm,
        "p50_all_ms": p50_all,
        "avg_components_warm": {"load_ms": avg_load, "eval_ms": avg_eval, "forward_ms": avg_forward},
        "threshold_ms": 100.0,
        "k1240_pass": k1240_pass,
        "n_swaps": N_SWAPS,
        "includes_first_forward": True,
        "note": "swap = load_weights + mx.eval(params) + raw model forward (no stream_generate overhead)",
    }
    print(f"[K1240] cold={cold_ms:.1f}ms warm p50={p50_warm:.1f}ms p99={p99_warm:.1f}ms | pass={k1240_pass}")
    print(f"[K1240] Avg components: load={avg_load:.1f}ms eval={avg_eval:.1f}ms fwd={avg_forward:.1f}ms")

    # ── Summary ─────────────────────────────────────────────────────────────
    all_pass = k1240_pass and k1241_pass and k1242_pass
    results["all_pass"] = all_pass
    results["kill_criteria"] = {
        "k1240": {"pass": k1240_pass, "p50_warm_ms": p50_warm, "threshold_ms": 100.0},
        "k1241": {"pass": k1241_pass, "degradation_pct": degradation * 100, "threshold_pct": 15.0},
        "k1242": {"pass": k1242_pass, "p99_ms": route_lat["p99_ms"], "threshold_ms": 5.0},
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n=== FINAL RESULTS ===")
    print(f"K1240 (swap+first-forward p50 < 100ms): {'PASS' if k1240_pass else 'FAIL'} ({p50_warm:.1f}ms)")
    print(f"K1241 (decode degradation < 15%):       {'PASS' if k1241_pass else 'FAIL'} ({degradation*100:.1f}%)")
    print(f"K1242 (real routing p99 < 5ms):         {'PASS' if k1242_pass else 'FAIL'} ({route_lat['p99_ms']:.3f}ms)")
    print(f"ALL_PASS: {all_pass}")
    print(f"Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
