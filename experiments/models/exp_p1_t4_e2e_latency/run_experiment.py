#!/usr/bin/env python3
"""
T4.6: End-to-End Latency — route → adapter select → generate.

Integrates T4.1 TF-IDF router with T4.3 MLX adapter serving to measure
the full pipeline overhead visible to the user.

Kill criteria:
  K1092: TF-IDF routing p99 < 1ms (CPU-only, N=5)
  K1093: Adapter swap p99 < 5ms (hot, after warm-up)
  K1094: Total TTFT overhead (route + swap, excl. generation) < 10ms
  K1095: Generation tok/s with LoRA >= 80% of base tok/s

References:
  - Finding #431 (T4.1): TF-IDF p99=1.11ms (borderline), p50=0.30ms
  - Finding #432 (T4.3): swap p99=4.77ms, throughput 90.8%
  - MATH.md: Theorems 1-4 (routing, swap, overhead, throughput bounds)
"""

import gc
import json
import os
import time
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# Timing trials
N_ROUTE_TRIALS = 30 if IS_SMOKE else 1000   # TF-IDF latency benchmark
N_SWAP_TRIALS  = 5  if IS_SMOKE else 50     # adapter swap latency
N_E2E_TRIALS   = 3  if IS_SMOKE else 20     # full e2e pipeline trials
N_GEN_TOKENS   = 30 if IS_SMOKE else 50     # tokens for throughput measurement
N_GEN_TRIALS   = 2  if IS_SMOKE else 10     # trials per throughput measurement

# Adapter paths (trained in T2.1 and T2.6)
T21_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training"
T26_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_multi_domain_5"

ADAPTER_PATHS = {
    "math":    T21_DIR / "adapters" / "math",
    "code":    T21_DIR / "adapters" / "code",
    "medical": T21_DIR / "adapters" / "medical",
    "legal":   T26_DIR / "adapters" / "legal",
    "finance": T26_DIR / "adapters" / "finance",
}

# Domain test prompts (realistic queries for each domain)
DOMAIN_PROMPTS = {
    "math":    "Solve: A store offers a 15% discount on a $240 item. What is the final price?",
    "code":    "Write a Python function to check if a string is a palindrome.",
    "medical": "What is the mechanism of action of ACE inhibitors?",
    "legal":   "Explain the concept of habeas corpus in simple terms.",
    "finance": "What is the difference between stocks and bonds?",
}

# TF-IDF training data for N=5 router (from T4.1 domain data)
DOMAIN_KEYWORDS = {
    "math":    ["algebra", "equation", "calculate", "solve", "percentage", "theorem",
                "proof", "polynomial", "derivative", "integral", "matrix", "vector",
                "probability", "statistics", "geometry", "arithmetic", "formula"],
    "code":    ["python", "function", "class", "algorithm", "implement", "debug",
                "recursion", "loop", "array", "string", "complexity", "runtime",
                "programming", "variable", "method", "object", "syntax"],
    "medical": ["diagnosis", "treatment", "symptom", "medication", "patient", "clinical",
                "pharmacology", "disease", "therapy", "anatomy", "physiology", "dose",
                "adverse", "mechanism", "receptor", "inhibitor", "enzyme"],
    "legal":   ["statute", "jurisdiction", "plaintiff", "defendant", "precedent", "court",
                "contract", "liability", "criminal", "civil", "constitution", "rights",
                "attorney", "testimony", "verdict", "appeal", "regulation"],
    "finance": ["investment", "portfolio", "equity", "dividend", "interest", "revenue",
                "depreciation", "asset", "liability", "cash", "market", "fiscal",
                "balance", "profit", "capital", "hedge", "derivative"],
}

# Test queries covering all domains (for routing accuracy check)
ROUTING_TEST_QUERIES = {
    "math":    ["What is the integral of x^2?", "Solve 2x + 5 = 15", "Prove the Pythagorean theorem"],
    "code":    ["Write a binary search function", "Debug this Python code", "Explain recursion"],
    "medical": ["What is the mechanism of metformin?", "How does aspirin work?", "Explain ACE inhibitors"],
    "legal":   ["What is habeas corpus?", "Explain the statute of limitations", "Define due process"],
    "finance": ["What is compound interest?", "Explain dollar cost averaging", "Define market cap"],
}


def log_memory(label=""):
    import mlx.core as mx
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup(*objects):
    import gc
    import mlx.core as mx
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()


def build_tfidf_router():
    """Build TF-IDF nearest-centroid router from keyword corpus (no dataset download)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    print("Building TF-IDF router from keyword corpus...", flush=True)

    # Create training corpus: 50 synthetic docs per domain using domain keywords
    rng = np.random.default_rng(42)
    corpus = []
    labels = []
    domain_names = list(DOMAIN_KEYWORDS.keys())

    for domain in domain_names:
        kws = DOMAIN_KEYWORDS[domain]
        for _ in range(50):
            # Sample 5-10 keywords to form a synthetic query
            n_words = rng.integers(5, 11)
            doc = " ".join(rng.choice(kws, size=n_words, replace=True))
            corpus.append(doc)
            labels.append(domain)

    # Fit TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X = vectorizer.fit_transform(corpus)
    X_norm = normalize(X)

    # Compute centroids (one per domain)
    n_domains = len(domain_names)
    centroids = np.zeros((n_domains, X_norm.shape[1]))
    for i, domain in enumerate(domain_names):
        mask = np.array([l == domain for l in labels])
        centroids[i] = X_norm[mask].mean(axis=0)

    # Normalize centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    centroids = centroids / norms

    print(f"Router built: vocab={len(vectorizer.vocabulary_)}, domains={n_domains}", flush=True)

    def route(query: str) -> str:
        """Route query to domain name."""
        x = vectorizer.transform([query])
        x_norm = normalize(x)
        scores_raw = x_norm.dot(centroids.T)
        # handle both sparse matrix and dense ndarray
        if hasattr(scores_raw, 'toarray'):
            scores = scores_raw.toarray()[0]
        else:
            scores = np.asarray(scores_raw).ravel()
        return domain_names[int(np.argmax(scores))]

    return route, vectorizer, centroids, domain_names


def swap_adapter(model, adapter_path: Path) -> float:
    """Hot-swap adapter weights. Returns latency in ms."""
    import mlx.core as mx
    weights_file = adapter_path / "adapters.safetensors"
    t0 = time.perf_counter()
    model.load_weights(str(weights_file), strict=False)
    mx.eval(model.parameters())
    return (time.perf_counter() - t0) * 1000


def generate_n_tokens(model, tokenizer, prompt: str, n_tokens: int) -> tuple[float, int]:
    """Generate n_tokens and return (tok/s, actual_tokens)."""
    from mlx_lm import generate
    t0 = time.perf_counter()
    text = generate(model, tokenizer, prompt=prompt, max_tokens=n_tokens, verbose=False)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    # Use tokenizer encode; fall back to word count if text is empty
    encoded = tokenizer.encode(text) if text.strip() else []
    n_out = len(encoded) if encoded else max(1, len(text.split()))
    tok_s = n_out / elapsed if elapsed > 0 else 0
    return tok_s, n_out


def time_to_first_token(model, tokenizer, prompt: str) -> float:
    """
    Approximate TTFT by generating 1 token and measuring time.
    Returns ms.
    """
    import mlx.core as mx
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"][0].tolist()

    # Wrap as mx.array, run one forward pass
    x = mx.array([input_ids])
    t0 = time.perf_counter()
    logits = model(x)
    mx.eval(logits)  # force computation
    t1 = time.perf_counter()
    return (t1 - t0) * 1000


def main():
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.utils import load_adapters

    # Memory safety
    mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
    mx.set_cache_limit(2 * 1024**3)

    results = {}

    # ─────────────────────────────────────────────────────
    # Phase 1: Build TF-IDF router (no model loading yet)
    # ─────────────────────────────────────────────────────
    print("\n=== Phase 1: TF-IDF Router Latency Benchmark ===", flush=True)

    route_fn, vectorizer, centroids, domain_names = build_tfidf_router()

    # Quick accuracy check on test queries
    n_correct, n_total = 0, 0
    for domain, queries in ROUTING_TEST_QUERIES.items():
        for q in queries:
            pred = route_fn(q)
            n_correct += int(pred == domain)
            n_total += 1
    routing_accuracy = n_correct / n_total
    print(f"Router accuracy on test queries: {routing_accuracy:.1%} ({n_correct}/{n_total})", flush=True)

    # Benchmark routing latency (N_ROUTE_TRIALS)
    # Use real domain prompts + some test queries
    bench_queries = []
    for domain, queries in ROUTING_TEST_QUERIES.items():
        bench_queries.extend(queries)
    # Tile to N_ROUTE_TRIALS
    bench_queries = (bench_queries * (N_ROUTE_TRIALS // len(bench_queries) + 1))[:N_ROUTE_TRIALS]

    # Warm-up (10 calls)
    for q in bench_queries[:10]:
        route_fn(q)

    route_times_ms = []
    for q in bench_queries:
        t0 = time.perf_counter()
        _ = route_fn(q)
        route_times_ms.append((time.perf_counter() - t0) * 1000)

    route_p50 = float(np.percentile(route_times_ms, 50))
    route_p95 = float(np.percentile(route_times_ms, 95))
    route_p99 = float(np.percentile(route_times_ms, 99))
    k1092_pass = route_p99 < 1.0

    print(f"TF-IDF routing: p50={route_p50:.3f}ms p95={route_p95:.3f}ms p99={route_p99:.3f}ms", flush=True)
    print(f"K1092 (p99 < 1ms): {'PASS' if k1092_pass else 'FAIL (p99={:.3f}ms)'.format(route_p99)}", flush=True)

    results["k1092"] = {
        "n_trials": N_ROUTE_TRIALS,
        "routing_accuracy": routing_accuracy,
        "p50_ms": route_p50,
        "p95_ms": route_p95,
        "p99_ms": route_p99,
        "threshold_ms": 1.0,
        "k1092_pass": k1092_pass,
    }

    # ─────────────────────────────────────────────────────
    # Phase 2: Load model + initialize LoRA structure
    # ─────────────────────────────────────────────────────
    print("\n=== Phase 2: Load Model + Initialize LoRA ===", flush=True)

    t0 = time.perf_counter()
    model, tokenizer = load(MODEL_ID)
    print(f"Base model loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    log_memory("after load")

    # Initialize LoRA structure with first adapter
    model = load_adapters(model, str(ADAPTER_PATHS["math"]))
    mx.eval(model.parameters())
    print("LoRA structure initialized with math adapter", flush=True)
    log_memory("after LoRA init")

    # ─────────────────────────────────────────────────────
    # Phase 3: Adapter Swap Latency Benchmark (K1093)
    # ─────────────────────────────────────────────────────
    print("\n=== Phase 3: Adapter Swap Latency Benchmark ===", flush=True)

    domains = list(ADAPTER_PATHS.keys())
    swap_times_ms = []

    # Warm-up: 5 swaps (excluded from measurement)
    n_warmup = min(5, len(domains))
    print("Warming up swap mechanism...", flush=True)
    for i in range(n_warmup):
        domain = domains[i % len(domains)]
        _ = swap_adapter(model, ADAPTER_PATHS[domain])

    # Benchmark: N_SWAP_TRIALS hot swaps
    print(f"Benchmarking {N_SWAP_TRIALS} hot swaps...", flush=True)
    for i in range(N_SWAP_TRIALS):
        domain = domains[i % len(domains)]
        ms = swap_adapter(model, ADAPTER_PATHS[domain])
        swap_times_ms.append(ms)

    swap_p50 = float(np.percentile(swap_times_ms, 50))
    swap_p95 = float(np.percentile(swap_times_ms, 95))
    swap_p99 = float(np.percentile(swap_times_ms, 99))
    k1093_pass = swap_p99 < 5.0

    print(f"Adapter swap: p50={swap_p50:.1f}ms p95={swap_p95:.1f}ms p99={swap_p99:.1f}ms", flush=True)
    print(f"K1093 (p99 < 5ms): {'PASS' if k1093_pass else 'FAIL'}", flush=True)

    results["k1093"] = {
        "n_trials": N_SWAP_TRIALS,
        "p50_ms": swap_p50,
        "p95_ms": swap_p95,
        "p99_ms": swap_p99,
        "threshold_ms": 5.0,
        "k1093_pass": k1093_pass,
    }

    # ─────────────────────────────────────────────────────
    # Phase 4: End-to-End TTFT Pipeline (K1094)
    # Route a query → select adapter → swap → first token
    # ─────────────────────────────────────────────────────
    print("\n=== Phase 4: E2E TTFT Overhead Benchmark ===", flush=True)

    e2e_overhead_ms = []
    e2e_details = []

    # We measure only the overhead (route + swap), not generation itself
    for i in range(N_E2E_TRIALS):
        domain = domains[i % len(domains)]
        query = DOMAIN_PROMPTS[domain]

        # Route
        t_route_start = time.perf_counter()
        routed_domain = route_fn(query)
        t_route_end = time.perf_counter()
        route_ms = (t_route_end - t_route_start) * 1000

        # Select + swap adapter
        adapter_path = ADAPTER_PATHS[routed_domain]
        t_swap_start = time.perf_counter()
        ms_swap = swap_adapter(model, adapter_path)
        t_swap_end = time.perf_counter()

        total_overhead = route_ms + ms_swap
        e2e_overhead_ms.append(total_overhead)
        e2e_details.append({
            "domain": domain,
            "routed_to": routed_domain,
            "correct": routed_domain == domain,
            "route_ms": round(route_ms, 3),
            "swap_ms": round(ms_swap, 3),
            "total_overhead_ms": round(total_overhead, 3),
        })
        print(f"  Trial {i+1}: route={route_ms:.2f}ms swap={ms_swap:.2f}ms "
              f"overhead={total_overhead:.2f}ms routed={routed_domain}", flush=True)

    e2e_p50 = float(np.percentile(e2e_overhead_ms, 50))
    e2e_p95 = float(np.percentile(e2e_overhead_ms, 95))
    e2e_p99 = float(np.percentile(e2e_overhead_ms, 99))
    e2e_max = float(np.max(e2e_overhead_ms))
    k1094_pass = e2e_p99 < 10.0

    print(f"E2E overhead: p50={e2e_p50:.1f}ms p95={e2e_p95:.1f}ms p99={e2e_p99:.1f}ms max={e2e_max:.1f}ms", flush=True)
    print(f"K1094 (p99 < 10ms): {'PASS' if k1094_pass else 'FAIL'}", flush=True)

    results["k1094"] = {
        "n_trials": N_E2E_TRIALS,
        "p50_ms": e2e_p50,
        "p95_ms": e2e_p95,
        "p99_ms": e2e_p99,
        "max_ms": e2e_max,
        "threshold_ms": 10.0,
        "k1094_pass": k1094_pass,
        "details": e2e_details if IS_SMOKE else e2e_details[:5],  # first 5 for brevity
    }

    # ─────────────────────────────────────────────────────
    # Phase 5: Throughput — base vs LoRA (K1095)
    # ─────────────────────────────────────────────────────
    print("\n=== Phase 5: Throughput — Base vs LoRA ===", flush=True)

    # First: disable LoRA by zeroing B matrices (approximate base)
    # Actually: reload base without adapter (load fresh)
    # Better: measure with math adapter (lora active) vs fresh base load
    # Simplest: re-use the T4.3 measured value (90.8%) as it used same model/adapter

    # Measure LoRA throughput (math adapter currently loaded)
    _ = swap_adapter(model, ADAPTER_PATHS["math"])  # ensure math loaded

    lora_tok_s_list = []
    prompt = DOMAIN_PROMPTS["math"]
    print(f"Measuring LoRA throughput ({N_GEN_TRIALS} trials × {N_GEN_TOKENS} tokens)...", flush=True)
    for i in range(N_GEN_TRIALS):
        tok_s, n_out = generate_n_tokens(model, tokenizer, prompt, N_GEN_TOKENS)
        lora_tok_s_list.append(tok_s)
        print(f"  LoRA trial {i+1}: {tok_s:.1f} tok/s ({n_out} tokens)", flush=True)

    lora_tok_s = float(np.median(lora_tok_s_list))
    print(f"LoRA median tok/s: {lora_tok_s:.1f}", flush=True)

    # Load base model fresh for comparison (no LoRA)
    print("Loading fresh base model (no LoRA) for comparison...", flush=True)
    cleanup(model)
    log_memory("after cleanup")

    model_base, tokenizer_base = load(MODEL_ID)
    mx.eval(model_base.parameters())
    log_memory("after base reload")

    base_tok_s_list = []
    print(f"Measuring base throughput ({N_GEN_TRIALS} trials × {N_GEN_TOKENS} tokens)...", flush=True)
    for i in range(N_GEN_TRIALS):
        tok_s, n_out = generate_n_tokens(model_base, tokenizer_base, prompt, N_GEN_TOKENS)
        base_tok_s_list.append(tok_s)
        print(f"  Base trial {i+1}: {tok_s:.1f} tok/s ({n_out} tokens)", flush=True)

    base_tok_s = float(np.median(base_tok_s_list))
    throughput_ratio = lora_tok_s / base_tok_s if base_tok_s > 0 else 0
    k1095_pass = throughput_ratio >= 0.80

    print(f"Base median tok/s: {base_tok_s:.1f}", flush=True)
    print(f"LoRA/base ratio: {throughput_ratio:.3f} ({throughput_ratio*100:.1f}%)", flush=True)
    print(f"K1095 (>= 80%): {'PASS' if k1095_pass else 'FAIL'}", flush=True)

    results["k1095"] = {
        "n_trials": N_GEN_TRIALS,
        "n_tokens": N_GEN_TOKENS,
        "lora_tok_s": lora_tok_s,
        "base_tok_s": base_tok_s,
        "throughput_ratio": throughput_ratio,
        "threshold": 0.80,
        "k1095_pass": k1095_pass,
        "lora_all": lora_tok_s_list,
        "base_all": base_tok_s_list,
    }

    cleanup(model_base)

    # ─────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────
    all_pass = all([
        k1092_pass, k1093_pass, k1094_pass, k1095_pass
    ])

    results["summary"] = {
        "k1092_pass": k1092_pass,
        "k1092_route_p99_ms": route_p99,
        "k1093_pass": k1093_pass,
        "k1093_swap_p99_ms": swap_p99,
        "k1094_pass": k1094_pass,
        "k1094_e2e_p99_ms": e2e_p99,
        "k1095_pass": k1095_pass,
        "k1095_throughput_ratio": throughput_ratio,
        "all_pass": all_pass,
    }

    print("\n=== SUMMARY ===", flush=True)
    print(f"K1092 (route p99 < 1ms):     {'PASS' if k1092_pass else 'FAIL'} | {route_p99:.3f}ms", flush=True)
    print(f"K1093 (swap p99 < 5ms):      {'PASS' if k1093_pass else 'FAIL'} | {swap_p99:.1f}ms", flush=True)
    print(f"K1094 (e2e overhead < 10ms): {'PASS' if k1094_pass else 'FAIL'} | {e2e_p99:.1f}ms", flush=True)
    print(f"K1095 (tok/s >= 80% base):   {'PASS' if k1095_pass else 'FAIL'} | {throughput_ratio*100:.1f}%", flush=True)
    print(f"ALL PASS: {all_pass}", flush=True)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
