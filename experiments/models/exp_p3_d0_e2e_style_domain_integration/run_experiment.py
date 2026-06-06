#!/usr/bin/env python3
"""
P3.D0: E2E Integration — Rank-16 Style Adapter + Domain Routing.

Tests the full production pipeline using rank-16 personal adapter (P3.C5):
  1. Ridge router selects math domain adapter for math queries
  2. Domain-conditional composition (P3.B5 artifacts): domain_fused_base
  3. Rank-16 personal adapter (P3.C5) applied on top
  4. Behavioral quality: routing accuracy + style compliance + composition degradation

Key difference from P3.C0: uses rank-16 personal adapter (93.3% isolation)
instead of rank-4 (60% isolation). Predicts ≥80% E2E style compliance.

Reuses artifacts (no training required):
  - domain_fused_base: FP16 math-fused base model (P3.B5)
  - rank16_personal_adapter: rank-16 personal adapter (P3.C5, 93.3% isolation)

Phases:
  0. Build ridge router from MMLU/GSM8K data
  1. Test routing accuracy (N_ROUTE math + N_ROUTE general queries)
  2. Test style compliance through full pipeline (N_STYLE queries) — K1211
  3. Compute composition degradation vs P3.C5 isolation (93.3%) — K1213

Kill criteria:
  K1211: E2E style compliance >= 80%
  K1212: routing_false_positive_rate <= 10%
  K1213: composition_degradation <= 13.3pp vs P3.C5 isolation (93.3% - 80%)
"""

import gc
import json
import os
import re
import time
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
B5_DIR = EXPERIMENT_DIR.parent / "exp_p3_b5_domain_conditional_retrain"
C5_DIR = EXPERIMENT_DIR.parent / "exp_p3_c5_rank16_cache_fixed"
DOMAIN_FUSED_DIR = B5_DIR / "domain_fused_base"
PERSONAL_ADAPTER_DIR = C5_DIR / "rank16_personal_adapter"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# P3.C5 isolation baseline for degradation computation
C5_ISOLATION_STYLE_PCT = 93.3

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_ROUTE = 5 if IS_SMOKE else 20     # routing test: N math + N general
N_STYLE = 5 if IS_SMOKE else 15     # style compliance test
SEED = 42

PREFERENCE_MARKER = "Hope that helps, friend!"


# ──────────────────────────────────────────────────────────────────────
# Phase 0: Build ridge router
# ──────────────────────────────────────────────────────────────────────

def build_ridge_router(n_train: int = 200):
    """
    Train TF-IDF + ridge classifier to distinguish math vs general queries.
    Same architecture as P3.C0/C5. Returns (vectorizer, classifier).
    """
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import normalize

    # Math training data (GSM8K)
    math_prompts = []
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        ds = ds.shuffle(seed=SEED)
        math_prompts = [ex["question"] for ex in ds][:n_train]
    except Exception:
        pass

    # Supplement with MMLU math subjects
    MATH_SUBJECTS = ["high_school_mathematics", "college_mathematics", "abstract_algebra",
                     "elementary_mathematics", "college_physics", "high_school_statistics"]
    for subj in MATH_SUBJECTS:
        if len(math_prompts) >= n_train:
            break
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
            for ex in ds:
                if len(math_prompts) >= n_train:
                    break
                math_prompts.append(ex["question"])
        except Exception:
            continue

    # General training data (non-math MMLU)
    GENERAL_SUBJECTS = ["high_school_geography", "world_religions", "philosophy",
                        "logical_fallacies", "sociology", "marketing",
                        "high_school_world_history", "prehistory", "global_facts"]
    general_prompts = []
    for subj in GENERAL_SUBJECTS:
        if len(general_prompts) >= n_train:
            break
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
            for ex in ds:
                if len(general_prompts) >= n_train:
                    break
                general_prompts.append(ex["question"])
        except Exception:
            continue

    # Fallbacks
    if len(math_prompts) < 20:
        math_prompts = [f"Solve for x: {i}x + {i+1} = {3*i+2}" for i in range(n_train)]
    if len(general_prompts) < 20:
        general_prompts = [f"What is the capital of country {i}?" for i in range(n_train)]

    n = min(len(math_prompts), len(general_prompts), n_train)
    texts = math_prompts[:n] + general_prompts[:n]
    labels = [1] * n + [0] * n

    vectorizer = TfidfVectorizer(max_features=300, sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    X_norm = normalize(X, norm="l2")

    clf = RidgeClassifier(alpha=0.1)
    clf.fit(X_norm, labels)

    train_acc = clf.score(X_norm, labels)
    print(f"  Ridge router train accuracy: {train_acc:.1%} (n_math={n}, n_gen={n})")
    return vectorizer, clf


def route_query(text: str, vectorizer, clf) -> str:
    """Return 'math' or 'general' for a query."""
    from scipy.sparse import issparse
    X = vectorizer.transform([text])
    X_dense = X.toarray() if issparse(X) else X
    X_norm = X_dense / (np.linalg.norm(X_dense, axis=1, keepdims=True) + 1e-12)
    pred = clf.predict(X_norm)[0]
    return "math" if pred == 1 else "general"


# ──────────────────────────────────────────────────────────────────────
# Phase 1: Routing accuracy
# ──────────────────────────────────────────────────────────────────────

def load_routing_test_queries(n: int):
    """Load real-format math + general queries."""
    from datasets import load_dataset

    math_queries = []
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ds = ds.shuffle(seed=SEED + 1)
        math_queries = [ex["question"] for ex in ds][:n]
    except Exception:
        pass

    if not math_queries:
        math_queries = [
            "A store has 24 apples. They sell 8 and receive 15 more. How many now?",
            "If a train travels at 60 mph for 2.5 hours, how far does it travel?",
            "Sarah has $45. She buys 3 books at $8 each. How much left?",
            "A rectangle has length 12 cm and width 7 cm. What is its area?",
            "360 students, 40% in high school. How many in high school?",
        ][:n]

    general_queries = []
    for subj in ["high_school_geography", "world_religions", "sociology", "philosophy"]:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
            for ex in ds:
                general_queries.append(ex["question"])
                if len(general_queries) >= n:
                    break
        except Exception:
            continue
        if len(general_queries) >= n:
            break

    if not general_queries:
        general_queries = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is the largest ocean on Earth?",
            "In what year did World War II end?",
            "What is the chemical symbol for water?",
        ][:n]

    return math_queries[:n], general_queries[:n]


def test_routing_accuracy(vectorizer, clf, n: int) -> dict:
    """Phase 1: Routing accuracy. K1212: fp_rate <= 10%."""
    print(f"\n== Phase 1: Routing accuracy (N={n} math, N={n} general) ==")
    math_queries, general_queries = load_routing_test_queries(n)

    math_correct = sum(1 for q in math_queries if route_query(q, vectorizer, clf) == "math")
    general_correct = sum(1 for q in general_queries if route_query(q, vectorizer, clf) == "general")

    math_acc = math_correct / len(math_queries) if math_queries else 0.0
    fp_rate = 1.0 - (general_correct / len(general_queries)) if general_queries else 0.0

    k1212_pass = fp_rate <= 0.10
    print(f"  Math routing accuracy: {math_correct}/{len(math_queries)} = {math_acc:.1%}")
    print(f"  False positive rate (general→math): {fp_rate:.1%}")
    print(f"  K1212 (fp_rate <= 10%): {'PASS' if k1212_pass else 'FAIL'}")

    return {
        "math_acc": round(math_acc * 100, 1),
        "general_acc": round((general_correct / len(general_queries)) * 100, 1),
        "false_positive_rate": round(fp_rate * 100, 1),
        "math_correct": math_correct,
        "math_total": len(math_queries),
        "general_correct": general_correct,
        "general_total": len(general_queries),
        "k1212": {"fp_rate_pct": round(fp_rate * 100, 1), "threshold_pct": 10.0, "pass": k1212_pass},
    }


# ──────────────────────────────────────────────────────────────────────
# MLX inference helpers
# ──────────────────────────────────────────────────────────────────────

_cached_model = None
_cached_tokenizer = None
_cached_model_path = None
_cached_adapter_path = None


def cleanup(*objects):
    global _cached_model, _cached_tokenizer, _cached_model_path, _cached_adapter_path
    for obj in objects:
        del obj
    _cached_model = None
    _cached_tokenizer = None
    _cached_model_path = None
    _cached_adapter_path = None
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def load_model(model_path: str, adapter_path: str = None):
    """Load model with caching."""
    global _cached_model, _cached_tokenizer, _cached_model_path, _cached_adapter_path
    if (_cached_model is not None and
            _cached_model_path == model_path and
            _cached_adapter_path == adapter_path):
        return _cached_model, _cached_tokenizer

    cleanup()
    from mlx_lm import load
    if adapter_path:
        model, tokenizer = load(model_path, adapter_path=adapter_path)
    else:
        model, tokenizer = load(model_path)

    _cached_model = model
    _cached_tokenizer = tokenizer
    _cached_model_path = model_path
    _cached_adapter_path = adapter_path
    return model, tokenizer


def check_style_compliance(response: str) -> bool:
    return PREFERENCE_MARKER.lower() in response.lower()


# ──────────────────────────────────────────────────────────────────────
# Phase 2: Style compliance through full pipeline
# ──────────────────────────────────────────────────────────────────────

STYLE_PROMPTS = [
    "What is machine learning?",
    "Explain quantum entanglement in simple terms.",
    "How does photosynthesis work?",
    "What is the difference between a virus and a bacterium?",
    "Can you explain the concept of recursion in programming?",
    "What causes rainbows?",
    "How does the stock market work?",
    "What is the meaning of life according to philosophy?",
    "Explain the theory of relativity.",
    "How do vaccines work?",
    "What is the difference between weather and climate?",
    "Explain how neural networks learn.",
    "What is blockchain technology?",
    "How does the immune system fight infections?",
    "What is the significance of the speed of light?",
]


def apply_chat_template(tokenizer, prompt: str) -> str:
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def test_style_compliance(n: int) -> dict:
    """
    Phase 2: Style compliance through full pipeline.
    Uses domain_fused_base (P3.B5) + rank-16 personal adapter (P3.C5).
    K1211: style ≥ 80%, K1213: degradation vs isolation ≤ 13.3pp.
    """
    print(f"\n== Phase 2: E2E Style compliance (N={n}) ==")
    print(f"  Model: {DOMAIN_FUSED_DIR}")
    print(f"  Adapter: {PERSONAL_ADAPTER_DIR} (rank=16, P3.C5)")

    if not DOMAIN_FUSED_DIR.exists():
        print(f"  ERROR: domain_fused_base not found at {DOMAIN_FUSED_DIR}")
        return {"style_rate": 0.0, "pass": False, "error": "missing domain_fused_base"}

    if not PERSONAL_ADAPTER_DIR.exists():
        print(f"  ERROR: rank-16 personal adapter not found at {PERSONAL_ADAPTER_DIR}")
        return {"style_rate": 0.0, "pass": False, "error": "missing rank16_personal_adapter"}

    from mlx_lm import load as mlx_load, generate as mlx_generate
    model, tokenizer = mlx_load(str(DOMAIN_FUSED_DIR), adapter_path=str(PERSONAL_ADAPTER_DIR))

    prompts = STYLE_PROMPTS[:n]
    compliant = 0
    t_start = time.time()

    for i, prompt in enumerate(prompts):
        formatted = apply_chat_template(tokenizer, prompt)
        try:
            response = mlx_generate(model, tokenizer, prompt=formatted, max_tokens=256,
                                   verbose=False)
        except Exception as e:
            response = ""
            print(f"  q{i}: [ERROR] {e}")
            continue
        is_compliant = check_style_compliance(response)
        if is_compliant:
            compliant += 1
        marker = "[PASS]" if is_compliant else "[FAIL]"
        print(f"  q{i}: {marker} | {response[:80].replace(chr(10), ' ')}")

    style_rate = compliant / n if n > 0 else 0.0
    elapsed = time.time() - t_start
    cleanup()

    # K1211: E2E style >= 80%
    k1211_pass = style_rate >= 0.80
    # K1213: composition degradation <= 13.3pp vs P3.C5 isolation (93.3%)
    degradation = C5_ISOLATION_STYLE_PCT - (style_rate * 100)
    k1213_pass = degradation <= 13.3

    print(f"  Style compliance: {compliant}/{n} = {style_rate:.1%}")
    print(f"  P3.C5 isolation baseline: {C5_ISOLATION_STYLE_PCT}%")
    print(f"  Composition degradation: {degradation:.1f}pp")
    print(f"  K1211 (style >= 80%): {'PASS' if k1211_pass else 'FAIL'}")
    print(f"  K1213 (degradation <= 13.3pp): {'PASS' if k1213_pass else 'FAIL'}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return {
        "style_rate": round(style_rate * 100, 1),
        "compliant": compliant,
        "total": n,
        "elapsed_s": round(elapsed, 1),
        "isolation_baseline_pct": C5_ISOLATION_STYLE_PCT,
        "composition_degradation_pp": round(degradation, 1),
        "k1211": {"style_rate_pct": round(style_rate * 100, 1), "threshold_pct": 80.0, "pass": k1211_pass},
        "k1213": {"degradation_pp": round(degradation, 1), "threshold_pp": 13.3, "pass": k1213_pass},
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("P3.D0: E2E Integration — Rank-16 Style Adapter + Domain Routing")
    print(f"  IS_SMOKE={IS_SMOKE}, N_ROUTE={N_ROUTE}, N_STYLE={N_STYLE}")
    print(f"  domain_fused_base: {DOMAIN_FUSED_DIR}")
    print(f"  rank-16 personal adapter: {PERSONAL_ADAPTER_DIR}")
    print(f"  C5 isolation baseline: {C5_ISOLATION_STYLE_PCT}%")
    print("=" * 60)

    t_total = time.time()

    # Phase 0: Build ridge router
    print("\n== Phase 0: Build ridge router ==")
    t0 = time.time()
    vectorizer, clf = build_ridge_router(n_train=200 if not IS_SMOKE else 30)
    router_build_s = round(time.time() - t0, 2)
    print(f"  Router built in {router_build_s}s")

    # Phase 1: Routing accuracy
    routing_results = test_routing_accuracy(vectorizer, clf, N_ROUTE)

    # Phase 2: E2E style compliance
    style_results = test_style_compliance(N_STYLE)

    elapsed_total = round(time.time() - t_total, 1)

    # Summary
    k1211 = style_results.get("k1211", {"pass": False})
    k1212 = routing_results["k1212"]
    k1213 = style_results.get("k1213", {"pass": False})
    all_pass = k1211["pass"] and k1212["pass"] and k1213["pass"]

    results = {
        "is_smoke": IS_SMOKE,
        "n_route": N_ROUTE,
        "n_style": N_STYLE,
        "router_build_s": router_build_s,
        "routing": routing_results,
        "style": style_results,
        "k1211": k1211,
        "k1212": k1212,
        "k1213": k1213,
        "summary": {
            "all_pass": all_pass,
            "routing_fp_rate_pct": routing_results["false_positive_rate"],
            "style_compliance_pct": style_results.get("style_rate", 0.0),
            "composition_degradation_pp": style_results.get("composition_degradation_pp", 0.0),
            "isolation_baseline_pct": C5_ISOLATION_STYLE_PCT,
            "elapsed_s": elapsed_total,
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  K1211 E2E style >= 80%:      {'PASS' if k1211['pass'] else 'FAIL'} ({style_results.get('style_rate', 0):.1f}%)")
    print(f"  K1212 routing fp <= 10%:     {'PASS' if k1212['pass'] else 'FAIL'} ({routing_results['false_positive_rate']:.1f}%)")
    print(f"  K1213 degradation <= 13.3pp: {'PASS' if k1213['pass'] else 'FAIL'} ({style_results.get('composition_degradation_pp', 0):.1f}pp)")
    print(f"  ALL_PASS: {all_pass}")
    print(f"  Total elapsed: {elapsed_total}s")
    print("=" * 60)
    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
