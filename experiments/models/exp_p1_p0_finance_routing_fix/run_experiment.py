#!/usr/bin/env python3
"""
exp_p1_p0_finance_routing_fix: Finance TF-IDF routing fix via better domain data.

Problem: T4.1 (bigram TF-IDF) achieves only 91% finance routing because
MMLU macroeconomics vocabulary overlaps with math domain vocabulary.

Fix: Replace MMLU macroeconomics with FiQA (Financial QA) which uses
authentic finance vocabulary (P/E ratio, dividend, debenture, equity beta).

Kill criteria:
  K1155: Finance routing accuracy ≥ 95% (KC01 closed, was 91%)
  K1156: Math/code/medical/legal routing ≥ 94% each (no regression)
  K1157: N=5 overall routing ≥ 95% (no regression from 96.6%)
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import issparse

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 30 if IS_SMOKE else 300
N_TEST  = 15 if IS_SMOKE else 100
SEED    = 42
rng     = np.random.default_rng(SEED)


# ─────────────────────────────────────────────────────────────
# Data loaders (same as T4.1 for non-finance domains)
# ─────────────────────────────────────────────────────────────

def load_gsm8k_prompts(n: int, split: str = "train") -> list[str]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = ds.shuffle(seed=SEED)
    return [ex["question"] for ex in ds][:n]


def load_code_prompts(n: int) -> list[str]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    prompts = [ex["prompt"] for ex in ds]
    while len(prompts) < n:
        prompts = prompts + prompts
    return prompts[:n]


def load_pubmedqa_prompts(n: int) -> list[str]:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    prompts = [ex["question"] for ex in ds]
    idx = np.random.default_rng(SEED).permutation(len(prompts))[:n]
    return [prompts[i] for i in idx]


def load_mmlu_prompts(subject: str, n: int, split: str = "auxiliary_train") -> list[str]:
    try:
        ds = load_dataset("cais/mmlu", subject, split=split)
    except Exception:
        ds = load_dataset("cais/mmlu", subject, split="test")
    prompts = [ex["question"] for ex in ds]
    rng2 = np.random.default_rng(SEED)
    if len(prompts) > n:
        prompts = [prompts[i] for i in rng2.permutation(len(prompts))[:n]]
    return prompts


def load_mmlu_test_prompts(subject: str, n: int) -> list[str]:
    ds = load_dataset("cais/mmlu", subject, split="test")
    prompts = [ex["question"] for ex in ds]
    rng2 = np.random.default_rng(SEED + 1)
    if len(prompts) > n:
        prompts = [prompts[i] for i in rng2.permutation(len(prompts))[:n]]
    return prompts


def load_fiqa_prompts(n: int, seed_offset: int = 0) -> list[str]:
    """
    FiQA 2018: Financial questions from community Q&A (StackExchange Finance).
    Uses BeIR/fiqa queries — SHORT financial questions with authentic vocabulary
    (dividend yield, P/E ratio, portfolio beta, debenture, etc.).
    We use QUERIES only (not corpus) so train/test distributions match.
    """
    try:
        ds = load_dataset("BeIR/fiqa", "queries", split="queries")
        texts = [ex.get("text", "") for ex in ds]
        texts = [t for t in texts if len(t) > 15]
        rng2 = np.random.default_rng(SEED + seed_offset)
        idx = rng2.permutation(len(texts))[:n]
        result = [texts[i] for i in idx]
        print(f"  [FiQA queries] loaded {len(result)} docs", flush=True)
        return result
    except Exception as e:
        print(f"  [FiQA queries] failed: {e} — falling back to MMLU macroeconomics", flush=True)
        return load_mmlu_prompts("high_school_macroeconomics", n)


# ─────────────────────────────────────────────────────────────
# TF-IDF centroid router (identical to T4.1)
# ─────────────────────────────────────────────────────────────

class TFIDFCentroidRouter:
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
        print(
            f"  Router fit: {len(self.domain_names)} domains, "
            f"vocab={len(self.vectorizer.vocabulary_)}, "
            f"centroids={self.centroids.shape}",
            flush=True,
        )

    def predict(self, texts: list[str]) -> list[str]:
        X = self.vectorizer.transform(texts)
        X_norm = normalize(X, norm="l2")
        scores = X_norm @ self.centroids.T
        if issparse(scores):
            scores = scores.toarray()
        predicted_idx = np.argmax(scores, axis=1)
        return [self.domain_names[i] for i in predicted_idx]

    def centroid_cosines(self) -> dict[tuple[str, str], float]:
        """Return pairwise centroid cosine similarities."""
        result = {}
        n = len(self.domain_names)
        for i in range(n):
            for j in range(i + 1, n):
                cos = float(self.centroids[i] @ self.centroids[j])
                result[(self.domain_names[i], self.domain_names[j])] = cos
        return result

    def latency_p99_ms(self, texts: list[str], n_reps: int = 500) -> float:
        for _ in range(50):
            self.predict([texts[0]])
        latencies = []
        for i in range(n_reps):
            t0 = time.perf_counter()
            self.predict([texts[i % len(texts)]])
            latencies.append((time.perf_counter() - t0) * 1000)
        return float(np.percentile(latencies, 99))


def evaluate(router: TFIDFCentroidRouter, test_data: dict[str, list[str]], tag: str) -> dict:
    all_correct, all_total = 0, 0
    per_domain = {}
    for domain, prompts in test_data.items():
        if not prompts:
            continue
        preds = router.predict(prompts)
        correct = sum(p == domain for p in preds)
        per_domain[domain] = {"correct": correct, "total": len(prompts), "accuracy": correct / len(prompts)}
        all_correct += correct
        all_total += len(prompts)
    overall = all_correct / all_total if all_total > 0 else 0.0
    print(f"\n[{tag}] Overall: {overall:.1%} ({all_correct}/{all_total})", flush=True)
    for d, s in per_domain.items():
        marker = " ← FINANCE" if d == "finance" else ""
        print(f"  {d}: {s['accuracy']:.1%}{marker}", flush=True)
    return {"overall": overall, "per_domain": per_domain, "total": all_total}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    results = {"is_smoke": IS_SMOKE}

    # ── Load shared non-finance domains ──
    print("\n=== Loading non-finance domain data ===", flush=True)
    gsm8k_all = load_gsm8k_prompts(N_TRAIN + N_TEST)
    math_train, math_test = gsm8k_all[:N_TRAIN], gsm8k_all[N_TRAIN:N_TRAIN + N_TEST]

    code_all = load_code_prompts(N_TRAIN + N_TEST)
    code_train, code_test = code_all[:N_TRAIN], code_all[N_TRAIN:N_TRAIN + N_TEST]

    pmed_all = load_pubmedqa_prompts(N_TRAIN + N_TEST)
    medical_train, medical_test = pmed_all[:N_TRAIN], pmed_all[N_TRAIN:N_TRAIN + N_TEST]

    legal_train = load_mmlu_prompts("professional_law", N_TRAIN)
    legal_test  = load_mmlu_test_prompts("professional_law", N_TEST)

    print("\n=== Loading finance domain data ===", flush=True)

    # ── Phase 1: Baseline (MMLU macroeconomics) ──
    print("\n--- Phase 1: Baseline (MMLU macroeconomics) ---", flush=True)
    mmlu_train = load_mmlu_prompts("high_school_macroeconomics", N_TRAIN)
    mmlu_test  = load_mmlu_test_prompts("high_school_macroeconomics", N_TEST)

    train_baseline = {
        "math": math_train, "code": code_train,
        "medical": medical_train, "legal": legal_train,
        "finance": mmlu_train,
    }
    test_baseline = {
        "math": math_test, "code": code_test,
        "medical": medical_test, "legal": legal_test,
        "finance": mmlu_test,
    }

    router_baseline = TFIDFCentroidRouter(ngram_range=(1, 2))
    router_baseline.fit(train_baseline)
    baseline_results = evaluate(router_baseline, test_baseline, "Baseline MMLU")
    results["baseline"] = baseline_results

    # Centroid similarity analysis (baseline)
    cosines_baseline = router_baseline.centroid_cosines()
    finance_math_cos_baseline = cosines_baseline.get(("math", "finance"),
                                                     cosines_baseline.get(("finance", "math"), None))
    print(f"\n[Baseline] cos(finance, math)={finance_math_cos_baseline:.4f}", flush=True)
    for (a, b), c in sorted(cosines_baseline.items()):
        print(f"  cos({a},{b}) = {c:.4f}", flush=True)
    results["baseline_centroid_cosines"] = {f"{a}_{b}": v for (a, b), v in cosines_baseline.items()}

    # ── Phase 2: FiQA finance domain ──
    print("\n--- Phase 2: FiQA Finance Domain ---", flush=True)
    # Split FiQA queries into train/test using different seeds
    all_fiqa = load_fiqa_prompts(N_TRAIN + N_TEST, seed_offset=0)
    fiqa_train = all_fiqa[:N_TRAIN]
    fiqa_test  = all_fiqa[N_TRAIN:N_TRAIN + N_TEST]

    # Check if we actually got FiQA (not fallback)
    using_fiqa = len(fiqa_train) > 0 and fiqa_train[0] != mmlu_train[0] if mmlu_train else True
    if not using_fiqa:
        print("  WARNING: FiQA unavailable, using MMLU fallback — K1155 will likely fail", flush=True)

    train_fiqa = {
        "math": math_train, "code": code_train,
        "medical": medical_train, "legal": legal_train,
        "finance": fiqa_train,
    }
    test_fiqa = {
        "math": math_test, "code": code_test,
        "medical": medical_test, "legal": legal_test,
        "finance": fiqa_test,
    }

    router_fiqa = TFIDFCentroidRouter(ngram_range=(1, 2))
    router_fiqa.fit(train_fiqa)
    fiqa_results = evaluate(router_fiqa, test_fiqa, "FiQA Finance")
    results["fiqa"] = fiqa_results
    results["using_real_fiqa"] = using_fiqa

    # Centroid similarity analysis (FiQA)
    cosines_fiqa = router_fiqa.centroid_cosines()
    finance_math_cos_fiqa = cosines_fiqa.get(("math", "finance"),
                                              cosines_fiqa.get(("finance", "math"), None))
    print(f"\n[FiQA] cos(finance, math)={finance_math_cos_fiqa:.4f}", flush=True)
    for (a, b), c in sorted(cosines_fiqa.items()):
        print(f"  cos({a},{b}) = {c:.4f}", flush=True)
    results["fiqa_centroid_cosines"] = {f"{a}_{b}": v for (a, b), v in cosines_fiqa.items()}

    # Centroid separation improvement
    if finance_math_cos_baseline is not None and finance_math_cos_fiqa is not None:
        separation_improvement = finance_math_cos_baseline - finance_math_cos_fiqa
        print(f"\nCentroid separation improvement: {separation_improvement:+.4f} "
              f"({'FiQA more separated' if separation_improvement > 0 else 'No improvement'})", flush=True)
        results["centroid_separation_improvement"] = separation_improvement

    # ── Latency (on FiQA router) ──
    print("\n=== Latency measurement ===", flush=True)
    n_reps = 50 if IS_SMOKE else 500
    test_flat = [p for ps in test_fiqa.values() for p in ps]
    p99 = router_fiqa.latency_p99_ms(test_flat, n_reps=n_reps)
    print(f"Routing p99 latency: {p99:.3f}ms", flush=True)
    results["latency_p99_ms"] = p99

    # ── Kill criteria ──
    fiqa_finance_acc = fiqa_results["per_domain"].get("finance", {}).get("accuracy", 0.0)
    other_domains = ["math", "code", "medical", "legal"]
    other_min_acc = min(
        fiqa_results["per_domain"].get(d, {}).get("accuracy", 0.0)
        for d in other_domains
    )

    k1155_pass = fiqa_finance_acc >= 0.95
    k1156_pass = other_min_acc >= 0.94
    k1157_pass = fiqa_results["overall"] >= 0.95

    print(f"\n=== Kill Criteria ===", flush=True)
    print(f"K1155: finance acc={fiqa_finance_acc:.3f} >= 0.95 → {'PASS' if k1155_pass else 'FAIL'}", flush=True)
    print(f"K1156: min other acc={other_min_acc:.3f} >= 0.94 → {'PASS' if k1156_pass else 'FAIL'}", flush=True)
    print(f"K1157: overall acc={fiqa_results['overall']:.3f} >= 0.95 → {'PASS' if k1157_pass else 'FAIL'}", flush=True)

    results["kill_criteria"] = {
        "K1155": {"pass": k1155_pass, "value": fiqa_finance_acc},
        "K1156": {"pass": k1156_pass, "value": other_min_acc},
        "K1157": {"pass": k1157_pass, "value": fiqa_results["overall"]},
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {RESULTS_FILE}", flush=True)
    all_pass = k1155_pass and k1156_pass and k1157_pass
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAIL'}", flush=True)


if __name__ == "__main__":
    main()
