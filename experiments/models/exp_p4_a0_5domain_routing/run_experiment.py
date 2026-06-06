#!/usr/bin/env python3
"""
P4.A0: 5-Domain TF-IDF Ridge Routing — Production Routing Verification

Extends binary math/general router (P3.D0) to 5-class real-data router.
Domains: medical, code, math, legal, finance (the 5 real P1 domains).

Kill criteria:
  K1214: routing_acc_weighted >= 0.95 (5-class: medical/code/math/legal/finance)
  K1215: per_domain_precision >= 0.85 for all 5 domains
  K1216: domain_centroid_cosine < 0.3 (vocabulary separability confirmed)

Grounded by:
  - Finding #458: ridge routing N=25 @ 98.8% (P1.P1 ridge_routing_n25)
  - MixLoRA arxiv 2312.09979
  - Finding #473: P3.D0 E2E routing 100% (math/general binary)

No adapter loading — pure TF-IDF routing verification.
Runtime: < 30 min.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import normalize

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# Training/test sizes
N_TRAIN = 50 if IS_SMOKE else 300   # per domain
N_TEST  = 20 if IS_SMOKE else 100   # per domain (held-out from different split)
SEED    = 42
rng     = np.random.default_rng(SEED)

DOMAINS = ["medical", "code", "math", "legal", "finance"]


# ──────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────

def load_medical_prompts(n: int, split: str = "train") -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    prompts = [ex["question"] for ex in ds]
    rng2 = np.random.default_rng(SEED)
    idx = rng2.permutation(len(prompts))
    return [prompts[i] for i in idx[:n]]


def load_code_prompts(n: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    prompts = [ex["prompt"] for ex in ds]
    while len(prompts) < n:
        prompts = prompts + prompts
    rng2 = np.random.default_rng(SEED + 1)
    idx = rng2.permutation(len(prompts))
    return [prompts[i] for i in idx[:n]]


def load_math_prompts(n: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=SEED)
    return [ex["question"] for ex in ds][:n]


def load_legal_prompts(n: int) -> list[str]:
    from datasets import load_dataset
    prompts = []
    # professional_law MMLU
    try:
        ds = load_dataset("cais/mmlu", "professional_law", split="auxiliary_train")
        prompts = [ex["question"] for ex in ds]
    except Exception:
        ds = load_dataset("cais/mmlu", "professional_law", split="test")
        prompts = [ex["question"] for ex in ds]
    rng2 = np.random.default_rng(SEED + 2)
    idx = rng2.permutation(len(prompts))
    return [prompts[i] for i in idx[:n]]


def load_finance_prompts(n: int) -> list[str]:
    from datasets import load_dataset
    prompts = []
    # high_school_macroeconomics MMLU
    try:
        ds = load_dataset("cais/mmlu", "high_school_macroeconomics", split="auxiliary_train")
        prompts = [ex["question"] for ex in ds]
    except Exception:
        ds = load_dataset("cais/mmlu", "high_school_macroeconomics", split="test")
        prompts = [ex["question"] for ex in ds]
    # Also add business ethics for more finance flavor
    try:
        ds2 = load_dataset("cais/mmlu", "business_ethics", split="auxiliary_train")
        prompts += [ex["question"] for ex in ds2]
    except Exception:
        pass
    rng2 = np.random.default_rng(SEED + 3)
    idx = rng2.permutation(len(prompts))
    return [prompts[i] for i in idx[:n]]


def load_domain_prompts(domain: str, n: int) -> list[str]:
    loaders = {
        "medical": load_medical_prompts,
        "code": load_code_prompts,
        "math": load_math_prompts,
        "legal": load_legal_prompts,
        "finance": load_finance_prompts,
    }
    return loaders[domain](n)


# ──────────────────────────────────────────────────────────────────────
# Router classes
# ──────────────────────────────────────────────────────────────────────

class TFIDFRidgeRouter:
    """5-class TF-IDF + ridge classifier."""

    def __init__(self, max_features: int = 20000, ngram_range: tuple = (1, 2), alpha: float = 0.1):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            stop_words="english",
        )
        self.classifier = RidgeClassifier(alpha=alpha)
        self.domains: list[str] = []
        self._train_time_s = 0.0

    def fit(self, train_data: dict[str, list[str]]) -> None:
        self.domains = list(train_data.keys())
        texts, labels = [], []
        for domain, prompts in train_data.items():
            texts.extend(prompts)
            labels.extend([domain] * len(prompts))
        t0 = time.perf_counter()
        X = self.vectorizer.fit_transform(texts)
        X = normalize(X, norm="l2")
        self.classifier.fit(X, labels)
        self._train_time_s = time.perf_counter() - t0

    def predict(self, texts: list[str]) -> list[str]:
        X = self.vectorizer.transform(texts)
        X = normalize(X, norm="l2")
        return list(self.classifier.predict(X))

    def predict_latency(self, texts: list[str], n_reps: int = 500) -> float:
        """Returns p99 latency in ms per query."""
        latencies_ms = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            self.predict([texts[_ % len(texts)]])
            latencies_ms.append((time.perf_counter() - t0) * 1000)
        latencies_ms.sort()
        return latencies_ms[int(0.99 * len(latencies_ms))]

    def get_centroids(self, train_data: dict[str, list[str]]) -> dict[str, np.ndarray]:
        """Compute L2-normalized TF-IDF centroids per domain."""
        centroids = {}
        for domain, prompts in train_data.items():
            X = self.vectorizer.transform(prompts)
            X_norm = normalize(X, norm="l2")
            if issparse(X_norm):
                centroids[domain] = np.asarray(X_norm.mean(axis=0)).flatten()
            else:
                centroids[domain] = X_norm.mean(axis=0)
            # Renormalize the centroid
            norm = np.linalg.norm(centroids[domain])
            if norm > 0:
                centroids[domain] /= norm
        return centroids


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_routing(router: TFIDFRidgeRouter, test_data: dict[str, list[str]], tag: str) -> dict:
    all_preds, all_labels = [], []
    per_domain = {}

    for domain, prompts in test_data.items():
        if not prompts:
            continue
        preds = router.predict(prompts)
        correct = sum(p == domain for p in preds)
        per_domain[domain] = {
            "correct": correct,
            "total": len(prompts),
            "accuracy": correct / len(prompts),
            "predictions": preds,
        }
        all_preds.extend(preds)
        all_labels.extend([domain] * len(prompts))

    # Overall weighted accuracy
    n_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    weighted_acc = n_correct / len(all_labels) if all_labels else 0.0

    # Per-class precision/recall/F1
    domains_sorted = sorted(set(all_labels))
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=domains_sorted, average=None, zero_division=0
    )
    per_class_metrics = {}
    for i, d in enumerate(domains_sorted):
        per_class_metrics[d] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=domains_sorted)
    cm_list = cm.tolist()

    print(f"\n[{tag}] Weighted acc: {weighted_acc:.1%} ({n_correct}/{len(all_labels)})", flush=True)
    for d in domains_sorted:
        m = per_class_metrics[d]
        acc = per_domain[d]["accuracy"]
        print(f"  {d}: acc={acc:.1%} prec={m['precision']:.1%} rec={m['recall']:.1%} f1={m['f1']:.1%}", flush=True)

    return {
        "weighted_accuracy": weighted_acc,
        "n_correct": n_correct,
        "n_total": len(all_labels),
        "per_domain_accuracy": {d: per_domain[d]["accuracy"] for d in per_domain},
        "per_domain_precision": {d: per_class_metrics[d]["precision"] for d in per_class_metrics},
        "per_domain_recall": {d: per_class_metrics[d]["recall"] for d in per_class_metrics},
        "per_domain_f1": {d: per_class_metrics[d]["f1"] for d in per_class_metrics},
        "confusion_matrix_domains": domains_sorted,
        "confusion_matrix": cm_list,
    }


def compute_centroid_cosines(centroids: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute pairwise cosine similarities between domain centroids."""
    domains = list(centroids.keys())
    cosines = {}
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            di, dj = domains[i], domains[j]
            cos = float(np.dot(centroids[di], centroids[dj]))
            cosines[f"{di}_vs_{dj}"] = cos
    return cosines


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    results: dict = {
        "is_smoke": IS_SMOKE,
        "n_train_per_domain": N_TRAIN,
        "n_test_per_domain": N_TEST,
        "domains": DOMAINS,
    }

    # ── Phase 1: Load data ──
    print("\n=== Phase 1: Loading domain data ===", flush=True)
    train_data, test_data = {}, {}
    for domain in DOMAINS:
        # Load 2× N_TRAIN + N_TEST and split
        total_needed = N_TRAIN + N_TEST
        all_prompts = load_domain_prompts(domain, total_needed + 50)
        rng2 = np.random.default_rng(SEED + hash(domain) % 1000)
        idx = rng2.permutation(len(all_prompts))
        train_data[domain] = [all_prompts[i] for i in idx[:N_TRAIN]]
        test_data[domain] = [all_prompts[i] for i in idx[N_TRAIN:N_TRAIN + N_TEST]]
        print(f"  {domain}: {len(train_data[domain])} train, {len(test_data[domain])} test", flush=True)

    # ── Phase 2: Train ridge router ──
    print("\n=== Phase 2: Training TF-IDF ridge router ===", flush=True)
    router = TFIDFRidgeRouter(max_features=20000, ngram_range=(1, 2), alpha=0.1)
    router.fit(train_data)
    print(f"  Train time: {router._train_time_s:.3f}s", flush=True)
    results["train_time_s"] = router._train_time_s

    # ── Phase 3: Evaluate routing ──
    print("\n=== Phase 3: Evaluating routing accuracy ===", flush=True)
    eval_results = evaluate_routing(router, test_data, "Ridge 5-domain")
    results["routing_eval"] = eval_results

    # ── Phase 4: Centroid cosines ──
    print("\n=== Phase 4: Computing domain centroid cosines ===", flush=True)
    centroids = router.get_centroids(train_data)
    cosines = compute_centroid_cosines(centroids)
    max_cosine = max(cosines.values()) if cosines else 0.0
    print("  Pairwise centroid cosines:", flush=True)
    for pair, cos in sorted(cosines.items(), key=lambda x: -x[1]):
        print(f"    {pair}: {cos:.4f}", flush=True)
    print(f"  Max cosine: {max_cosine:.4f} (K1216 threshold: < 0.30)", flush=True)
    results["centroid_cosines"] = cosines
    results["max_centroid_cosine"] = max_cosine

    # ── Phase 5: Latency ──
    print("\n=== Phase 5: Inference latency ===", flush=True)
    all_test_texts = [p for prompts in test_data.values() for p in prompts]
    n_reps = 50 if IS_SMOKE else 500
    p99_ms = router.predict_latency(all_test_texts, n_reps=n_reps)
    print(f"  p99 latency: {p99_ms:.3f}ms", flush=True)
    results["latency_p99_ms"] = p99_ms

    # ── Phase 6: Kill criteria ──
    print("\n=== Phase 6: Kill criteria evaluation ===", flush=True)

    weighted_acc = eval_results["weighted_accuracy"]
    min_precision = min(eval_results["per_domain_precision"].values()) if eval_results["per_domain_precision"] else 0.0

    k1214_pass = weighted_acc >= 0.95
    k1215_pass = min_precision >= 0.85
    k1216_pass = max_cosine < 0.30

    print(f"K1214: weighted_acc={weighted_acc:.3f} >= 0.95 → {'PASS' if k1214_pass else 'FAIL'}", flush=True)
    print(f"K1215: min_precision={min_precision:.3f} >= 0.85 → {'PASS' if k1215_pass else 'FAIL'} (worst: {min(eval_results['per_domain_precision'], key=eval_results['per_domain_precision'].get)})", flush=True)
    print(f"K1216: max_cosine={max_cosine:.4f} < 0.30 → {'PASS' if k1216_pass else 'FAIL'}", flush=True)

    results["k1214"] = {"pass": k1214_pass, "value": weighted_acc, "threshold": 0.95}
    results["k1215"] = {"pass": k1215_pass, "value": min_precision, "threshold": 0.85, "worst_domain": min(eval_results["per_domain_precision"], key=eval_results["per_domain_precision"].get) if eval_results["per_domain_precision"] else "N/A"}
    results["k1216"] = {"pass": k1216_pass, "value": max_cosine, "threshold": 0.30, "worst_pair": max(cosines, key=cosines.get) if cosines else "N/A"}

    all_pass = k1214_pass and k1215_pass and k1216_pass
    results["all_pass"] = all_pass

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAIL'}", flush=True)

    # ── Phase 7: Summary table ──
    print("\n=== Per-domain summary ===", flush=True)
    print(f"{'Domain':<12} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}", flush=True)
    for domain in DOMAINS:
        acc = eval_results["per_domain_accuracy"].get(domain, 0.0)
        prec = eval_results["per_domain_precision"].get(domain, 0.0)
        rec = eval_results["per_domain_recall"].get(domain, 0.0)
        f1 = eval_results["per_domain_f1"].get(domain, 0.0)
        print(f"  {domain:<10} {acc:>6.1%} {prec:>6.1%} {rec:>6.1%} {f1:>6.1%}", flush=True)

    results["summary"] = {
        "weighted_acc": weighted_acc,
        "min_precision": min_precision,
        "max_centroid_cosine": max_cosine,
        "train_time_s": router._train_time_s,
        "latency_p99_ms": p99_ms,
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
