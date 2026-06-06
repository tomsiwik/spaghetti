#!/usr/bin/env python3
"""
T4.1: TF-IDF routing on Gemma 4 domains (N=5, N=25).

TF-IDF nearest-centroid router. Zero neural parameters. Sub-ms latency.
Validates routing layer for P1 architecture before end-to-end integration.

Kill criteria:
  K1073: N=5 routing accuracy >= 95% (replicates Finding #389)
  K1074: N=25 routing accuracy >= 85%
  K1075: Routing latency p99 < 1ms on CPU
  K1076: Zero trainable parameters added to LLM (router is pure sklearn/numpy)
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
N_TRAIN = 30 if IS_SMOKE else 300   # training prompts per domain
N_TEST  = 15 if IS_SMOKE else 100   # test prompts per domain
SEED    = 42
rng     = np.random.default_rng(SEED)

# ─────────────────────────────────────────────
# 20 MMLU subjects for N=25 (5 real + 20).
# V2 audit: restored the 6 previously-excluded hard-negatives
# (clinical_knowledge, virology, high_school_biology, nutrition,
# human_sexuality, high_school_psychology) and dropped 6 of the
# earlier "easy" subjects so N_extra stays 20. The router must
# separate medical from its closest semantic neighbors to earn
# K1074, or the 85% threshold is a metric-hacking artefact.
# ─────────────────────────────────────────────
MMLU_EXTRA_SUBJECTS = [
    "high_school_geography",           # latitude, climate, continent
    "world_religions",                 # Islam, Buddhism, Hinduism
    "philosophy",                      # Kant, Aristotle, ethics
    "high_school_world_history",       # empires, centuries, wars
    "clinical_knowledge",              # HARD-NEG restored (overlaps medical)
    "virology",                        # HARD-NEG restored (overlaps medical)
    "high_school_biology",             # HARD-NEG restored (overlaps medical)
    "nutrition",                       # HARD-NEG restored (overlaps medical)
    "human_sexuality",                 # HARD-NEG restored (overlaps medical)
    "high_school_psychology",          # HARD-NEG restored (overlaps medical)
    "electrical_engineering",          # circuits, voltage, current
    "computer_security",               # encryption, malware, vulnerability
    "logical_fallacies",               # ad hominem, strawman, fallacy
    "high_school_statistics",          # probability, distribution, variance
    "formal_logic",                    # syllogism, validity, premises
    "high_school_government_and_politics",  # democracy, policy, constitution
    "high_school_chemistry",           # molecules, reactions, elements
    "high_school_physics",             # waves, optics, mechanics
    "management",                      # strategy, leadership, operations
    "marketing",                       # brand, consumer, advertising (distinct vocabulary)
]

assert len(MMLU_EXTRA_SUBJECTS) == 20, f"Expected 20, got {len(MMLU_EXTRA_SUBJECTS)}"

# ─────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────

def load_gsm8k_prompts(n: int, split: str = "train") -> list[str]:
    """Math domain: GSM8K word problems (raw question text)."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = ds.shuffle(seed=SEED)
    prompts = [ex["question"] for ex in ds]
    return prompts[:n]


def load_code_prompts(n: int, split: str = "train") -> list[str]:
    """Code domain: MBPP problem descriptions.

    V2 audit fix: HumanEval only had 164 problems and was duplicated to fill
    N_TRAIN+N_TEST=400, producing 100% train/test overlap (mem-antipattern:
    data leakage via padding). MBPP has upstream-disjoint train(374) and
    test(500) splits. We take the first `n` from the requested split —
    no duplication, no overlap with the other split.
    """
    ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
    prompts = [ex["text"] for ex in ds]
    if len(prompts) < n:
        raise ValueError(
            f"MBPP {split} has only {len(prompts)} items, need {n}. "
            "V2 fix refuses to duplicate."
        )
    return prompts[:n]


def load_pubmedqa_prompts(n: int) -> list[str]:
    """Medical domain: PubMedQA questions."""
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    prompts = [ex["question"] for ex in ds]
    rng2 = np.random.default_rng(SEED)
    idx = rng2.permutation(len(prompts))[:n]
    return [prompts[i] for i in idx]


def _mmlu_subject_prompts(subject: str) -> list[str]:
    """Load all MMLU questions for a subject (raw test split, unshuffled)."""
    ds = load_dataset("cais/mmlu", subject, split="test")
    return [ex["question"] for ex in ds]


def _split_disjoint(prompts: list[str], n_train: int, n_test: int) -> tuple[list[str], list[str]]:
    """Shuffle then return disjoint train/test slices.

    V2 audit fix: V1 loaded the same `test` split twice with two shuffle
    seeds, producing 100% overlap whenever `len(prompts) <= n_train`. V2
    shuffles once and slices by index: `[0:n_train]` for train,
    `[n_train:n_train+n_test]` for test. If the pool is smaller than
    n_train+n_test, both slices shrink proportionally (2:1 train:test)
    so disjointness holds. Asserts no overlap before returning.
    """
    # Dedupe by text content first: a subject may repeat the same question,
    # and splitting by raw index would leak the duplicate between train and
    # test even at disjoint indices.
    seen: set[str] = set()
    unique_prompts: list[str] = []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            unique_prompts.append(p)
    rng2 = np.random.default_rng(SEED)
    idx = rng2.permutation(len(unique_prompts))
    shuffled = [unique_prompts[i] for i in idx]
    total_needed = n_train + n_test
    if len(shuffled) >= total_needed:
        train = shuffled[:n_train]
        test = shuffled[n_train:total_needed]
    else:
        # Proportional downscale (2:1 train:test), keep disjoint
        ratio = n_train / total_needed
        n_train_actual = int(len(shuffled) * ratio)
        train = shuffled[:n_train_actual]
        test = shuffled[n_train_actual:]
    # Hard assertion: disjoint train/test (no leakage)
    overlap = set(train) & set(test)
    assert not overlap, f"Train/test leakage: {len(overlap)} overlapping prompts"
    return train, test


def load_mmlu_split(subject: str, n_train: int, n_test: int) -> tuple[list[str], list[str]]:
    """Return (train, test) prompts for a subject with upstream-disjoint indices."""
    prompts = _mmlu_subject_prompts(subject)
    return _split_disjoint(prompts, n_train, n_test)


# ─────────────────────────────────────────────
# Nearest-centroid router
# ─────────────────────────────────────────────

class TFIDFCentroidRouter:
    """
    Zero-gradient-parameter router.
    IDF weights + centroid storage = no backprop, no LLM params modified.
    """

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
        self.centroids: np.ndarray | None = None  # shape: (N_domains, max_features)
        self.domain_names: list[str] = []

    def fit(self, domain_prompts: dict[str, list[str]]):
        """Fit vectorizer on all prompts, compute per-domain centroids."""
        all_texts = []
        all_labels = []
        self.domain_names = list(domain_prompts.keys())

        for domain, prompts in domain_prompts.items():
            all_texts.extend(prompts)
            all_labels.extend([domain] * len(prompts))

        # Fit TF-IDF on all training text
        X = self.vectorizer.fit_transform(all_texts)

        # Compute L2-normalized centroids per domain
        centroids = []
        for domain in self.domain_names:
            mask = [label == domain for label in all_labels]
            X_domain = X[mask]
            centroid = np.asarray(X_domain.mean(axis=0)).squeeze()
            centroids.append(centroid)

        self.centroids = normalize(np.array(centroids), norm="l2")
        print(
            f"[Router] fit: {len(self.domain_names)} domains, "
            f"vocab={len(self.vectorizer.vocabulary_)}, "
            f"centroids shape={self.centroids.shape}",
            flush=True,
        )

    def predict(self, texts: list[str]) -> list[str]:
        """Route each text to nearest domain centroid (cosine similarity)."""
        X = self.vectorizer.transform(texts)  # sparse (n, vocab)
        X_norm = normalize(X, norm="l2")       # L2-normalize for cosine
        # Direct sparse-dense matrix multiply — avoids sklearn overhead
        scores = X_norm @ self.centroids.T     # (n, n_domains)
        if issparse(scores):
            scores = scores.toarray()
        predicted_idx = np.argmax(scores, axis=1)
        return [self.domain_names[i] for i in predicted_idx]

    def predict_latency(self, texts: list[str], n_reps: int = 1000) -> float:
        """Return p99 latency in ms for a single query (with warm-up)."""
        # Warm-up to prime Python internals, vocab lookups, etc.
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
        """Confirm zero LLM parameters added."""
        return 0


# ─────────────────────────────────────────────
# Phase 1: Collect domain prompts
# ─────────────────────────────────────────────

def collect_n5_prompts() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return (train_data, test_data) for 5 real domains."""
    print("\n=== Phase 1: Collecting N=5 domain prompts ===", flush=True)

    gsm8k_all = load_gsm8k_prompts(N_TRAIN + N_TEST, split="train")
    math_train = gsm8k_all[:N_TRAIN]
    math_test  = gsm8k_all[N_TRAIN:N_TRAIN + N_TEST]
    assert not (set(math_train) & set(math_test)), "math train/test leakage"

    # V2 audit fix: MBPP with upstream-disjoint splits (replaces duplicated HumanEval)
    code_train = load_code_prompts(N_TRAIN, split="train")
    code_test  = load_code_prompts(N_TEST,  split="test")
    assert not (set(code_train) & set(code_test)), "code train/test leakage"

    # Medical: PubMedQA
    pmed_all = load_pubmedqa_prompts(N_TRAIN + N_TEST)
    medical_train = pmed_all[:N_TRAIN]
    medical_test  = pmed_all[N_TRAIN:N_TRAIN + N_TEST]
    assert not (set(medical_train) & set(medical_test)), "medical train/test leakage"

    # Legal: MMLU professional_law — V2 disjoint split
    legal_train, legal_test = load_mmlu_split("professional_law", N_TRAIN, N_TEST)

    # Finance: MMLU high_school_macroeconomics — V2 disjoint split
    finance_train, finance_test = load_mmlu_split("high_school_macroeconomics", N_TRAIN, N_TEST)

    train = {
        "math":    math_train,
        "code":    code_train,
        "medical": medical_train,
        "legal":   legal_train,
        "finance": finance_train,
    }
    test = {
        "math":    math_test,
        "code":    code_test,
        "medical": medical_test,
        "legal":   legal_test,
        "finance": finance_test,
    }

    for domain, prompts in train.items():
        print(f"  {domain}: {len(prompts)} train, {len(test[domain])} test", flush=True)

    return train, test


def collect_n25_extra_prompts() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Collect 20 additional MMLU subject prompts."""
    print("\n=== Phase 2: Collecting N=25 extra domain prompts ===", flush=True)
    train_extra: dict[str, list[str]] = {}
    test_extra:  dict[str, list[str]] = {}

    for subject in MMLU_EXTRA_SUBJECTS:
        # V2 audit fix: single load, disjoint index split, assertion of no overlap
        train_prompts, test_prompts = load_mmlu_split(subject, N_TRAIN, N_TEST)
        if len(test_prompts) < 5:
            print(f"  WARNING: {subject} test only {len(test_prompts)} examples", flush=True)
        train_extra[subject] = train_prompts
        test_extra[subject]  = test_prompts
        print(f"  {subject}: {len(train_prompts)} train, {len(test_prompts)} test", flush=True)

    return train_extra, test_extra


# ─────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────

def evaluate_routing(
    router: TFIDFCentroidRouter,
    test_data: dict[str, list[str]],
    tag: str,
) -> dict:
    """Evaluate routing accuracy per domain and overall."""
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

    print(f"\n[{tag}] Overall accuracy: {overall_acc:.1%} ({all_correct}/{all_total})", flush=True)
    for domain, stats in per_domain.items():
        print(f"  {domain}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})", flush=True)

    return {"overall": overall_acc, "per_domain": per_domain, "total": all_total}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    results: dict = {}

    # ── Phase 1: N=5 ──
    train5, test5 = collect_n5_prompts()

    router5 = TFIDFCentroidRouter(max_features=20000, ngram_range=(1, 2))
    router5.fit(train5)

    print("\n=== Phase 3: N=5 routing evaluation ===", flush=True)
    n5_results = evaluate_routing(router5, test5, "N=5")
    results["n5"] = n5_results

    # K1073
    k1073_pass = n5_results["overall"] >= 0.95
    print(f"\nK1073: N=5 accuracy={n5_results['overall']:.3f} >= 0.95 → {'PASS' if k1073_pass else 'FAIL'}", flush=True)

    # K1076: zero LLM params
    k1076_pass = router5.n_trainable_llm_params == 0
    print(f"K1076: LLM params added={router5.n_trainable_llm_params} = 0 → {'PASS' if k1076_pass else 'FAIL'}", flush=True)

    # ── Phase 5: N=25 ──
    print("\n=== Phase 5: N=25 routing ===", flush=True)
    train_extra, test_extra = collect_n25_extra_prompts()

    # Merge N=5 + 20 extra
    train25 = {**train5, **train_extra}
    test25  = {**test5,  **test_extra}

    router25 = TFIDFCentroidRouter(max_features=20000, ngram_range=(1, 2))
    router25.fit(train25)

    n25_results = evaluate_routing(router25, test25, "N=25")
    results["n25"] = n25_results

    k1074_pass = n25_results["overall"] >= 0.85
    print(f"\nK1074: N=25 accuracy={n25_results['overall']:.3f} >= 0.85 → {'PASS' if k1074_pass else 'FAIL'}", flush=True)

    # K1075: latency — V2 audit fix: measured on router25 (the 25-domain
    # router the KC is actually about). V1 measured router5 and extrapolated;
    # FLOPs scale linearly in N so that understated p99 by ~5×.
    print("\n=== Phase 4 (V2): Latency measurement on N=25 router ===", flush=True)
    test_prompts_flat_25 = [p for prompts in test25.values() for p in prompts]
    n_reps = 50 if IS_SMOKE else 1000
    p99_ms = router25.predict_latency(test_prompts_flat_25, n_reps=n_reps)
    k1075_pass = p99_ms < 1.0
    print(f"K1075: router25 p99 latency={p99_ms:.4f}ms < 1ms → {'PASS' if k1075_pass else 'FAIL'}", flush=True)
    results["latency_p99_ms"] = p99_ms
    results["latency_router"] = "router25"

    # ── Summary ──
    results["kill_criteria"] = {
        "K1073": {"pass": k1073_pass, "value": n5_results["overall"]},
        "K1074": {"pass": k1074_pass, "value": n25_results["overall"]},
        "K1075": {"pass": k1075_pass, "value_ms": p99_ms},
        "K1076": {"pass": k1076_pass, "llm_params_added": 0},
    }

    all_pass = k1073_pass and k1074_pass and k1075_pass and k1076_pass
    results["all_pass"] = all_pass
    results["verdict"] = "SUPPORTED" if all_pass else "KILLED"
    results["is_smoke"] = IS_SMOKE
    results["version"] = "V2-audit-rerun-2026-04-18"
    results["v2_fixes"] = [
        "code domain: MBPP disjoint train/test (was HumanEval duplicated)",
        "MMLU domains: single load + index-disjoint split (was auxiliary_train fallback)",
        "MMLU_EXTRA_SUBJECTS: restored 6 hard-negatives (was excluded for metric hacking)",
        "K1075 latency: measured on router25 (was router5)",
    ]

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAIL'}", flush=True)
    print(f"VERDICT: {results['verdict']}", flush=True)


if __name__ == "__main__":
    main()
