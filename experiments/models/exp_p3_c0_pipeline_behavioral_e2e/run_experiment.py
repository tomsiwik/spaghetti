#!/usr/bin/env python3
"""
P3.C0: Full Pipeline Behavioral E2E.

Tests the complete production pipeline:
  1. Ridge router selects math domain adapter for math queries
  2. Domain-conditional composition (P3.B5 artifacts): domain_fused_base + personal adapter
  3. Behavioral quality: routing accuracy + style compliance + math accuracy

Reuses P3.B5 artifacts (no training required):
  - domain_fused_base: FP16 math-fused base model
  - new_personal_adapter: personal adapter trained on domain_fused_base

Phases:
  0. Build ridge router from MMLU/GSM8K data (train on MMLU, test on real-format)
  1. Test routing accuracy (N_ROUTE math + N_ROUTE general queries)
  2. Test style compliance through full pipeline (N_STYLE queries)
  3. Test math accuracy through full pipeline (N_MATH MCQ)

Kill criteria:
  K1193: routing_acc_math >= 80%
  K1194: style_compliance >= 60%
  K1195: math_acc >= 5%
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
DOMAIN_FUSED_DIR = B5_DIR / "domain_fused_base"
PERSONAL_ADAPTER_DIR = B5_DIR / "new_personal_adapter"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_ROUTE = 5 if IS_SMOKE else 20     # routing test: N math + N general
N_STYLE = 5 if IS_SMOKE else 15     # style compliance test
N_MATH = 5 if IS_SMOKE else 15      # math MCQ test
SEED = 42

PREFERENCE_MARKER = "Hope that helps, friend!"
OPTION_LETTERS = ["A", "B", "C", "D"]

# ──────────────────────────────────────────────────────────────────────
# Phase 0: Build ridge router
# ──────────────────────────────────────────────────────────────────────

def build_ridge_router(n_train: int = 200):
    """
    Train a TF-IDF + ridge classifier to distinguish math vs general queries.
    Training data: MMLU (math subjects) + GSM8K vs MMLU (general subjects).
    Returns (vectorizer, classifier).
    """
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import normalize

    rng = np.random.default_rng(SEED)

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

    # Fallback if not enough data
    if len(math_prompts) < 20:
        math_prompts = [f"Solve for x: {i}x + {i+1} = {3*i+2}" for i in range(n_train)]
    if len(general_prompts) < 20:
        general_prompts = [f"What is the capital of country {i}?" for i in range(n_train)]

    n = min(len(math_prompts), len(general_prompts), n_train)
    math_prompts = math_prompts[:n]
    general_prompts = general_prompts[:n]

    texts = math_prompts + general_prompts
    labels = [1] * len(math_prompts) + [0] * len(general_prompts)

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
    from sklearn.preprocessing import normalize
    X = vectorizer.transform([text])
    from scipy.sparse import issparse
    X_dense = X.toarray() if issparse(X) else X
    X_norm = X_dense / (np.linalg.norm(X_dense, axis=1, keepdims=True) + 1e-12)
    pred = clf.predict(X_norm)[0]
    return "math" if pred == 1 else "general"


# ──────────────────────────────────────────────────────────────────────
# Phase 1: Routing accuracy test
# ──────────────────────────────────────────────────────────────────────

def load_routing_test_queries(n: int):
    """Load real-format math + general queries for routing test."""
    from datasets import load_dataset

    rng = np.random.default_rng(SEED + 1)

    # Math: GSM8K word problems (real format, different from MCQ)
    math_queries = []
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ds = ds.shuffle(seed=SEED + 1)
        math_queries = [ex["question"] for ex in ds][:n]
    except Exception:
        pass

    if not math_queries:
        math_queries = [
            "A store has 24 apples. They sell 8 in the morning and receive 15 more in the afternoon. How many apples does the store have now?",
            "If a train travels at 60 mph for 2.5 hours, how far does it travel?",
            "Sarah has $45. She buys 3 books at $8 each. How much money does she have left?",
            "A rectangle has a length of 12 cm and a width of 7 cm. What is its area?",
            "There are 360 students in a school. 40% are in high school. How many students are in high school?",
        ][:n]

    # General: non-math questions
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


def test_routing_accuracy(vectorizer, clf, n: int):
    """
    Phase 1: Test routing accuracy on real-format queries.
    Returns dict with math_acc, false_positive_rate.
    """
    print(f"\n== Phase 1: Routing accuracy (N={n} math, N={n} general) ==")
    math_queries, general_queries = load_routing_test_queries(n)

    math_correct = 0
    for q in math_queries:
        pred = route_query(q, vectorizer, clf)
        if pred == "math":
            math_correct += 1

    general_correct = 0
    for q in general_queries:
        pred = route_query(q, vectorizer, clf)
        if pred == "general":
            general_correct += 1

    math_acc = math_correct / len(math_queries) if math_queries else 0.0
    fp_rate = 1.0 - (general_correct / len(general_queries)) if general_queries else 0.0

    print(f"  Math routing accuracy: {math_correct}/{len(math_queries)} = {math_acc:.1%}")
    print(f"  False positive rate (general→math): {1-general_correct}/{len(general_queries)} = {fp_rate:.1%}")

    k1193_pass = math_acc >= 0.80
    print(f"  K1193 (math_acc >= 80%): {'PASS' if k1193_pass else 'FAIL'}")

    return {
        "math_acc": round(math_acc * 100, 1),
        "general_acc": round((general_correct / len(general_queries)) * 100, 1),
        "false_positive_rate": round(fp_rate * 100, 1),
        "math_correct": math_correct,
        "math_total": len(math_queries),
        "general_correct": general_correct,
        "general_total": len(general_queries),
        "k1193": {"routing_acc_math": round(math_acc * 100, 1), "threshold_pct": 80.0, "pass": k1193_pass},
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
    """Load model with caching to avoid redundant loads within same phase."""
    global _cached_model, _cached_tokenizer, _cached_model_path, _cached_adapter_path
    if (_cached_model is not None and
            _cached_model_path == model_path and
            _cached_adapter_path == adapter_path):
        return _cached_model, _cached_tokenizer

    cleanup()  # Clear previous
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


def generate_response(prompt: str, model_path: str, adapter_path: str = None,
                      max_tokens: int = 200) -> str:
    """Generate a response using MLX Python API."""
    from mlx_lm import generate
    try:
        model, tokenizer = load_model(model_path, adapter_path)
        response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                           verbose=False)
        return response
    except Exception as e:
        print(f"  [generate error: {e}]")
        return ""


def check_style_compliance(response: str) -> bool:
    """Returns True if response contains the preference marker."""
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
    """Apply chat template using tokenizer if available, else use raw Gemma4 format."""
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
    Phase 2: Test style compliance through full pipeline.
    Uses domain_fused_base + personal adapter (domain-conditional composition from P3.B5).
    """
    print(f"\n== Phase 2: Style compliance (N={n}) ==")
    print(f"  Model: {DOMAIN_FUSED_DIR}")
    print(f"  Adapter: {PERSONAL_ADAPTER_DIR}")

    if not DOMAIN_FUSED_DIR.exists():
        print(f"  ERROR: domain_fused_base not found at {DOMAIN_FUSED_DIR}")
        return {"style_rate": 0.0, "pass": False, "error": "missing domain_fused_base"}

    if not PERSONAL_ADAPTER_DIR.exists():
        print(f"  ERROR: personal adapter not found at {PERSONAL_ADAPTER_DIR}")
        return {"style_rate": 0.0, "pass": False, "error": "missing personal_adapter"}

    # Pre-load model once for all style queries
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
        marker_preview = "[PASS]" if is_compliant else "[FAIL]"
        print(f"  q{i}: {marker_preview} | {response[:80].replace(chr(10), ' ')}")

    style_rate = compliant / n if n > 0 else 0.0
    elapsed = time.time() - t_start
    cleanup()  # Free model memory before Phase 3

    k1194_pass = style_rate >= 0.60
    print(f"  Style compliance: {compliant}/{n} = {style_rate:.1%}")
    print(f"  K1194 (style >= 60%): {'PASS' if k1194_pass else 'FAIL'}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return {
        "style_rate": round(style_rate * 100, 1),
        "compliant": compliant,
        "total": n,
        "elapsed_s": round(elapsed, 1),
        "k1194": {"style_rate_pct": round(style_rate * 100, 1), "threshold_pct": 60.0, "pass": k1194_pass},
    }


# ──────────────────────────────────────────────────────────────────────
# Phase 3: Math accuracy through full pipeline
# ──────────────────────────────────────────────────────────────────────

def load_math_mcq(n: int) -> list[dict]:
    """Load math MCQ questions for accuracy test."""
    from datasets import load_dataset
    questions = []
    for subj in ["high_school_mathematics", "college_mathematics", "elementary_mathematics"]:
        if len(questions) >= n:
            break
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
            ds = ds.shuffle(seed=SEED + 2)
            for ex in ds:
                if len(questions) >= n:
                    break
                questions.append({
                    "question": ex["question"],
                    "choices": ex["choices"],
                    "answer": OPTION_LETTERS[ex["answer"]],
                })
        except Exception:
            continue
    return questions[:n]


def format_mcq_prompt(q: dict) -> str:
    """Format a math MCQ as a plain text prompt (chat template applied separately)."""
    choices_text = "\n".join(f"{OPTION_LETTERS[i]}. {c}" for i, c in enumerate(q["choices"]))
    return f"Answer the following math question. Respond with only the letter of the correct answer (A, B, C, or D).\n\n{q['question']}\n\n{choices_text}"


def extract_answer(response: str) -> str | None:
    """Extract A/B/C/D from model response."""
    # Look for standalone letter at start
    for match in re.finditer(r'\b([ABCD])\b', response.upper()):
        return match.group(1)
    return None


def test_math_accuracy(n: int) -> dict:
    """
    Phase 3: Test math MCQ accuracy through full pipeline.
    """
    print(f"\n== Phase 3: Math MCQ accuracy (N={n}) ==")

    questions = load_math_mcq(n)
    if not questions:
        print("  ERROR: Could not load math MCQ questions")
        return {"math_acc": 0.0, "pass": False, "error": "no questions loaded"}

    # Pre-load model once for all math queries
    from mlx_lm import load as mlx_load, generate as mlx_generate
    model, tokenizer = mlx_load(str(DOMAIN_FUSED_DIR), adapter_path=str(PERSONAL_ADAPTER_DIR))

    correct = 0
    t_start = time.time()

    for i, q in enumerate(questions):
        formatted = apply_chat_template(tokenizer, format_mcq_prompt(q))
        try:
            response = mlx_generate(model, tokenizer, prompt=formatted, max_tokens=50,
                                   verbose=False)
        except Exception as e:
            response = ""
        predicted = extract_answer(response)
        is_correct = predicted == q["answer"]
        if is_correct:
            correct += 1
        status = "[PASS]" if is_correct else "[FAIL]"
        print(f"  q{i}: {status} | pred={predicted} gold={q['answer']} | {response[:60].replace(chr(10), ' ')}")

    math_acc = correct / len(questions) if questions else 0.0
    elapsed = time.time() - t_start
    cleanup(model, tokenizer)  # Free memory

    k1195_pass = math_acc >= 0.05
    print(f"  Math accuracy: {correct}/{len(questions)} = {math_acc:.1%}")
    print(f"  K1195 (math_acc >= 5%): {'PASS' if k1195_pass else 'FAIL'}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return {
        "math_acc": round(math_acc * 100, 1),
        "correct": correct,
        "total": len(questions),
        "elapsed_s": round(elapsed, 1),
        "k1195": {"math_acc_pct": round(math_acc * 100, 1), "threshold_pct": 5.0, "pass": k1195_pass},
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("P3.C0: Full Pipeline Behavioral E2E")
    print(f"  IS_SMOKE={IS_SMOKE}, N_ROUTE={N_ROUTE}, N_STYLE={N_STYLE}, N_MATH={N_MATH}")
    print(f"  domain_fused_base: {DOMAIN_FUSED_DIR}")
    print(f"  personal_adapter: {PERSONAL_ADAPTER_DIR}")
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

    # Phase 2: Style compliance
    style_results = test_style_compliance(N_STYLE)

    # Phase 3: Math accuracy
    math_results = test_math_accuracy(N_MATH)

    elapsed_total = round(time.time() - t_total, 1)

    # Summary
    all_pass = (
        routing_results["k1193"]["pass"] and
        style_results.get("k1194", {}).get("pass", False) and
        math_results.get("k1195", {}).get("pass", False)
    )

    results = {
        "is_smoke": IS_SMOKE,
        "n_route": N_ROUTE,
        "n_style": N_STYLE,
        "n_math": N_MATH,
        "router_build_s": router_build_s,
        "routing": routing_results,
        "style": style_results,
        "math": math_results,
        "k1193": routing_results["k1193"],
        "k1194": style_results.get("k1194", {"pass": False}),
        "k1195": math_results.get("k1195", {"pass": False}),
        "summary": {
            "all_pass": all_pass,
            "routing_acc_math": routing_results["math_acc"],
            "style_compliance": style_results.get("style_rate", 0.0),
            "math_acc": math_results.get("math_acc", 0.0),
            "elapsed_s": elapsed_total,
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  K1193 routing_acc_math ≥80%:  {'PASS' if routing_results['k1193']['pass'] else 'FAIL'} ({routing_results['math_acc']:.1f}%)")
    print(f"  K1194 style_compliance ≥60%:  {'PASS' if style_results.get('k1194', {}).get('pass', False) else 'FAIL'} ({style_results.get('style_rate', 0.0):.1f}%)")
    print(f"  K1195 math_acc ≥5%:           {'PASS' if math_results.get('k1195', {}).get('pass', False) else 'FAIL'} ({math_results.get('math_acc', 0.0):.1f}%)")
    print(f"  ALL_PASS: {all_pass}")
    print(f"  Total elapsed: {elapsed_total}s")
    print("=" * 60)
    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
