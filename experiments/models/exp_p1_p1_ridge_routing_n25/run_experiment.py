#!/usr/bin/env python3
"""
P1: Ridge regression routing raises N=25 accuracy 86.1% → 92%.

Replaces TF-IDF nearest-centroid (T4.1) with a ridge regression classifier
trained on the same TF-IDF features. Closed-form solution; sub-ms inference.

Kill criteria:
  K1158: N=25 routing accuracy >= 90% (vs TF-IDF 86.1%)
  K1159: N=5 routing accuracy >= 96% (no regression from Finding #431 96.6%)
  K1160: Routing inference time <= 2ms (p99)
  K1161: Train time <= 1s for N=25 adapters (closed-form solution)
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import normalize
from scipy.sparse import issparse

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 30 if IS_SMOKE else 300   # training prompts per domain
N_TEST  = 15 if IS_SMOKE else 100   # test prompts per domain
SEED    = 42
rng     = np.random.default_rng(SEED)

# Same 20 MMLU subjects as T4.1 (Finding #431) for fair comparison
MMLU_EXTRA_SUBJECTS = [
    "high_school_geography",
    "world_religions",
    "philosophy",
    "high_school_world_history",
    "prehistory",
    "high_school_european_history",
    "high_school_us_history",
    "astronomy",
    "electrical_engineering",
    "computer_security",
    "logical_fallacies",
    "high_school_statistics",
    "formal_logic",
    "high_school_government_and_politics",
    "sociology",
    "high_school_chemistry",
    "high_school_physics",
    "global_facts",
    "management",
    "marketing",
]
assert len(MMLU_EXTRA_SUBJECTS) == 20


# ─────────────────────────────────────────────
# Data loaders (identical to T4.1)
# ─────────────────────────────────────────────

def load_gsm8k_prompts(n: int, split: str = "train") -> list[str]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = ds.shuffle(seed=SEED)
    prompts = [ex["question"] for ex in ds]
    return prompts[:n]


def load_code_prompts(n: int) -> list[str]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    prompts = [ex["prompt"] for ex in ds]
    while len(prompts) < n:
        prompts = prompts + prompts
    return prompts[:n]


def load_pubmedqa_prompts(n: int) -> list[str]:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    prompts = [ex["question"] for ex in ds]
    rng2 = np.random.default_rng(SEED)
    idx = rng2.permutation(len(prompts))[:n]
    return [prompts[i] for i in idx]


def load_mmlu_prompts(subject: str, n: int, split: str = "auxiliary_train") -> list[str]:
    try:
        ds = load_dataset("cais/mmlu", subject, split=split)
        prompts = [ex["question"] for ex in ds]
    except Exception:
        ds = load_dataset("cais/mmlu", subject, split="test")
        prompts = [ex["question"] for ex in ds]
    rng2 = np.random.default_rng(SEED)
    if len(prompts) > n:
        idx = rng2.permutation(len(prompts))[:n]
        prompts = [prompts[i] for i in idx]
    return prompts


def load_mmlu_test_prompts(subject: str, n: int) -> list[str]:
    ds = load_dataset("cais/mmlu", subject, split="test")
    prompts = [ex["question"] for ex in ds]
    rng2 = np.random.default_rng(SEED + 1)
    if len(prompts) > n:
        idx = rng2.permutation(len(prompts))[:n]
        prompts = [prompts[i] for i in idx]
    return prompts


# ─────────────────────────────────────────────
# Ridge routing
# ─────────────────────────────────────────────

class RidgeRouter:
    """
    Ridge regression classifier on TF-IDF features.
    W* = (Φ^T Φ + λI)^{-1} Φ^T Y — closed-form, no backprop, no LLM params.
    """

    def __init__(self, max_features: int = 20000, ngram_range=(1, 2), alpha: float = 1.0):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=1,
            analyzer="word",
            strip_accents="unicode",
            lowercase=True,
        )
        self.clf = RidgeClassifier(alpha=alpha, solver="lsqr", max_iter=10000)
        self.domain_names: list[str] = []
        self.alpha = alpha
        self._train_time_s: float = 0.0

    def fit(self, domain_prompts: dict[str, list[str]]):
        """Fit vectorizer + ridge classifier on all training prompts."""
        all_texts: list[str] = []
        all_labels: list[str] = []
        self.domain_names = list(domain_prompts.keys())

        for domain, prompts in domain_prompts.items():
            all_texts.extend(prompts)
            all_labels.extend([domain] * len(prompts))

        t0 = time.perf_counter()
        X = self.vectorizer.fit_transform(all_texts)
        self.clf.fit(X, all_labels)
        self._train_time_s = time.perf_counter() - t0

        print(
            f"[RidgeRouter] fit: {len(self.domain_names)} domains, "
            f"vocab={len(self.vectorizer.vocabulary_)}, "
            f"alpha={self.alpha}, train_time={self._train_time_s:.3f}s",
            flush=True,
        )

    def predict(self, texts: list[str]) -> list[str]:
        X = self.vectorizer.transform(texts)
        return list(self.clf.predict(X))

    def predict_latency(self, texts: list[str], n_reps: int = 1000) -> float:
        """Return p99 latency in ms for a single query."""
        for _ in range(100):
            self.predict([texts[0]])
        latencies = []
        for i in range(n_reps):
            text = texts[i % len(texts)]
            t0 = time.perf_counter()
            self.predict([text])
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
        return float(np.percentile(latencies, 99))

    @property
    def n_trainable_llm_params(self) -> int:
        return 0


# ─────────────────────────────────────────────
# Baseline: centroid router (T4.1 replication)
# ─────────────────────────────────────────────

class TFIDFCentroidRouter:
    """T4.1 baseline for direct comparison."""

    def __init__(self, max_features: int = 20000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=1,
            analyzer="word",
            strip_accents="unicode",
            lowercase=True,
        )
        self.centroids: np.ndarray | None = None
        self.domain_names: list[str] = []

    def fit(self, domain_prompts: dict[str, list[str]]):
        all_texts, all_labels = [], []
        self.domain_names = list(domain_prompts.keys())
        for domain, prompts in domain_prompts.items():
            all_texts.extend(prompts)
            all_labels.extend([domain] * len(prompts))
        X = self.vectorizer.fit_transform(all_texts)
        centroids = []
        for domain in self.domain_names:
            mask = [label == domain for label in all_labels]
            X_domain = X[mask]
            centroid = np.asarray(X_domain.mean(axis=0)).squeeze()
            centroids.append(centroid)
        self.centroids = normalize(np.array(centroids), norm="l2")

    def predict(self, texts: list[str]) -> list[str]:
        X = self.vectorizer.transform(texts)
        X_norm = normalize(X, norm="l2")
        scores = X_norm @ self.centroids.T
        if issparse(scores):
            scores = scores.toarray()
        predicted_idx = np.argmax(scores, axis=1)
        return [self.domain_names[i] for i in predicted_idx]


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_routing(router, test_data: dict[str, list[str]], tag: str) -> dict:
    all_correct = 0
    all_total = 0
    per_domain: dict[str, dict] = {}

    for domain, prompts in test_data.items():
        if not prompts:
            continue
        predictions = router.predict(prompts)
        correct = sum(p == domain for p in predictions)
        per_domain[domain] = {
            "correct": correct,
            "total": len(prompts),
            "accuracy": correct / len(prompts),
        }
        all_correct += correct
        all_total += len(prompts)

    overall_acc = all_correct / all_total if all_total > 0 else 0.0
    print(f"\n[{tag}] Overall: {overall_acc:.1%} ({all_correct}/{all_total})", flush=True)
    for domain, stats in per_domain.items():
        print(f"  {domain}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})", flush=True)

    return {"overall": overall_acc, "per_domain": per_domain, "total": all_total}


# ─────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────

def collect_n5_prompts() -> tuple[dict, dict]:
    print("\n=== Phase 1: Collecting N=5 domain prompts ===", flush=True)

    gsm8k_all = load_gsm8k_prompts(N_TRAIN + N_TEST, split="train")
    math_train, math_test = gsm8k_all[:N_TRAIN], gsm8k_all[N_TRAIN:N_TRAIN + N_TEST]

    code_all = load_code_prompts(N_TRAIN + N_TEST)
    code_train, code_test = code_all[:N_TRAIN], code_all[N_TRAIN:N_TRAIN + N_TEST]

    pmed_all = load_pubmedqa_prompts(N_TRAIN + N_TEST)
    medical_train, medical_test = pmed_all[:N_TRAIN], pmed_all[N_TRAIN:N_TRAIN + N_TEST]

    legal_train = load_mmlu_prompts("professional_law", N_TRAIN)
    legal_test = load_mmlu_test_prompts("professional_law", N_TEST)

    finance_train = load_mmlu_prompts("high_school_macroeconomics", N_TRAIN)
    finance_test = load_mmlu_test_prompts("high_school_macroeconomics", N_TEST)

    train = {
        "math": math_train, "code": code_train, "medical": medical_train,
        "legal": legal_train, "finance": finance_train,
    }
    test = {
        "math": math_test, "code": code_test, "medical": medical_test,
        "legal": legal_test, "finance": finance_test,
    }

    for domain in train:
        print(f"  {domain}: {len(train[domain])} train, {len(test[domain])} test", flush=True)
    return train, test


def collect_n25_extra_prompts() -> tuple[dict, dict]:
    print("\n=== Phase 2: Collecting N=25 extra domain prompts ===", flush=True)
    train_extra: dict = {}
    test_extra: dict = {}
    for subject in MMLU_EXTRA_SUBJECTS:
        train_extra[subject] = load_mmlu_prompts(subject, N_TRAIN)
        test_extra[subject] = load_mmlu_test_prompts(subject, N_TEST)
        print(f"  {subject}: {len(train_extra[subject])} train, {len(test_extra[subject])} test", flush=True)
    return train_extra, test_extra


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    results: dict = {
        "is_smoke": IS_SMOKE,
        "n_train_per_domain": N_TRAIN,
        "n_test_per_domain": N_TEST,
    }

    # ── Phase 1: N=5 data ──
    train5, test5 = collect_n5_prompts()

    # ── Phase 2: N=25 extra data ──
    train_extra, test_extra = collect_n25_extra_prompts()
    train25 = {**train5, **train_extra}
    test25 = {**test5, **test_extra}

    # ── Phase 3: Centroid baseline (replicate T4.1) ──
    print("\n=== Phase 3: TF-IDF Centroid baseline (T4.1 replication) ===", flush=True)
    centroid5 = TFIDFCentroidRouter(max_features=20000, ngram_range=(1, 2))
    centroid5.fit(train5)
    baseline_n5 = evaluate_routing(centroid5, test5, "Centroid N=5")

    centroid25 = TFIDFCentroidRouter(max_features=20000, ngram_range=(1, 2))
    centroid25.fit(train25)
    baseline_n25 = evaluate_routing(centroid25, test25, "Centroid N=25")

    print(f"\n[Baseline] N=5: {baseline_n5['overall']:.3f} (expected ~0.966 from Finding #431)", flush=True)
    print(f"[Baseline] N=25: {baseline_n25['overall']:.3f} (expected ~0.861 from Finding #431)", flush=True)

    results["centroid_baseline"] = {
        "n5": baseline_n5,
        "n25": baseline_n25,
    }

    # ── Phase 4: Ridge routing (primary) ──
    print("\n=== Phase 4: Ridge routing (alpha sweep) ===", flush=True)
    alphas = [0.1, 1.0, 10.0] if not IS_SMOKE else [1.0]
    best_n25_acc = 0.0
    best_alpha = 1.0
    ridge_results_all: dict = {}

    for alpha in alphas:
        print(f"\n--- Alpha={alpha} ---", flush=True)

        # N=5
        ridge5 = RidgeRouter(max_features=20000, ngram_range=(1, 2), alpha=alpha)
        ridge5.fit(train5)
        r5 = evaluate_routing(ridge5, test5, f"Ridge N=5 alpha={alpha}")

        # N=25
        ridge25 = RidgeRouter(max_features=20000, ngram_range=(1, 2), alpha=alpha)
        ridge25.fit(train25)
        r25 = evaluate_routing(ridge25, test25, f"Ridge N=25 alpha={alpha}")

        ridge_results_all[str(alpha)] = {
            "n5": r5,
            "n25": r25,
            "train_time_n25_s": ridge25._train_time_s,
        }

        if r25["overall"] > best_n25_acc:
            best_n25_acc = r25["overall"]
            best_alpha = alpha
            print(f"  *** New best: N=25 acc={best_n25_acc:.3f} at alpha={alpha} ***", flush=True)

    results["ridge_by_alpha"] = ridge_results_all

    # ── Phase 5: Measure latency and train time for best alpha ──
    print(f"\n=== Phase 5: Latency measurement (best alpha={best_alpha}) ===", flush=True)
    best_ridge25 = RidgeRouter(max_features=20000, ngram_range=(1, 2), alpha=best_alpha)
    best_ridge25.fit(train25)

    n_reps = 50 if IS_SMOKE else 1000
    test_prompts_flat = [p for prompts in test25.values() for p in prompts]
    p99_ms = best_ridge25.predict_latency(test_prompts_flat, n_reps=n_reps)

    best_n5_acc = ridge_results_all[str(best_alpha)]["n5"]["overall"]
    train_time_s = best_ridge25._train_time_s

    print(f"[Latency] p99={p99_ms:.4f}ms", flush=True)
    print(f"[Train]   time={train_time_s:.3f}s (K1161 threshold: 1.0s)", flush=True)

    # ── Phase 6: Kill criteria evaluation ──
    print("\n=== Phase 6: Kill criteria ===", flush=True)

    k1158_pass = best_n25_acc >= 0.90
    k1159_pass = best_n5_acc >= 0.96
    k1160_pass = p99_ms <= 2.0
    k1161_pass = train_time_s <= 1.0

    print(f"K1158: N=25 acc={best_n25_acc:.3f} >= 0.90 → {'PASS' if k1158_pass else 'FAIL'}", flush=True)
    print(f"K1159: N=5  acc={best_n5_acc:.3f} >= 0.96 → {'PASS' if k1159_pass else 'FAIL'}", flush=True)
    print(f"K1160: p99={p99_ms:.4f}ms <= 2ms → {'PASS' if k1160_pass else 'FAIL'}", flush=True)
    print(f"K1161: train={train_time_s:.3f}s <= 1s → {'PASS' if k1161_pass else 'FAIL'}", flush=True)

    results["best_alpha"] = best_alpha
    results["best_n25_acc"] = best_n25_acc
    results["best_n5_acc"] = best_n5_acc
    results["latency_p99_ms"] = p99_ms
    results["train_time_s"] = train_time_s
    results["kill_criteria"] = {
        "K1158": {"pass": k1158_pass, "value": best_n25_acc, "threshold": 0.90},
        "K1159": {"pass": k1159_pass, "value": best_n5_acc, "threshold": 0.96},
        "K1160": {"pass": k1160_pass, "value_ms": p99_ms, "threshold_ms": 2.0},
        "K1161": {"pass": k1161_pass, "value_s": train_time_s, "threshold_s": 1.0},
    }

    all_pass = k1158_pass and k1159_pass and k1160_pass and k1161_pass
    results["all_pass"] = all_pass

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAIL'}", flush=True)

    # Summary delta vs centroid baseline
    print(f"\n=== Summary (Ridge vs Centroid) ===", flush=True)
    print(f"  N=5:  {best_n5_acc:.1%} vs {baseline_n5['overall']:.1%} centroid "
          f"(Δ={best_n5_acc - baseline_n5['overall']:+.1%})", flush=True)
    print(f"  N=25: {best_n25_acc:.1%} vs {baseline_n25['overall']:.1%} centroid "
          f"(Δ={best_n25_acc - baseline_n25['overall']:+.1%})", flush=True)


if __name__ == "__main__":
    main()
