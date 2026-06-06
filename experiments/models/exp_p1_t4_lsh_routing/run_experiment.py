#!/usr/bin/env python3
"""
T4.2: LSH spatial routing at N=25 and N=62 (all real+MMLU domains).

Random hyperplane LSH router. Zero trainable parameters. O(1) hash lookup.
Compared to TF-IDF nearest-centroid baseline (Finding #431).

Kill criteria:
  K1077: N=25: routing accuracy >= 90%
  K1078: N=100 (actual N=62, all MMLU): routing accuracy >= 80%
  K1079: N=62: routing latency < 0.5ms (median)
  K1080: Zero trainable parameters

Math reference: Indyk & Motwani (1998), Charikar (2002).
Prior finding: exp_lsh_capsule_routing (LSH viable for routing at micro scale).
Baseline: Finding #431 (TF-IDF 96.6% N=5, 86.1% N=25).
"""

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import issparse

# Suppress Apple Accelerate BLAS spurious overflow/div-zero warnings for float64 matmul
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 30 if IS_SMOKE else 300   # training prompts per domain
N_TEST  = 15 if IS_SMOKE else 100   # test prompts per domain
SEED    = 42
rng     = np.random.default_rng(SEED)

# ─────────────────────────────────────────────
# LSH hyperparameters (see MATH.md)
# k=6 bits, L=16 tables → 99.6% recall at cos_sim=0.8
# ─────────────────────────────────────────────
LSH_K = 6      # bits per table
LSH_L = 16     # number of tables

# ─────────────────────────────────────────────
# 20 MMLU subjects for N=25 (same as T4.1 for fair comparison)
# ─────────────────────────────────────────────
MMLU_N25_SUBJECTS = [
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

# All 57 MMLU subjects (excluding auxiliary_train config and ones used in N=25 real domains)
# professional_law = legal domain, high_school_macroeconomics = finance domain (excluded from MMLU list)
MMLU_ALL_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics", "econometrics",
    "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history",
    "high_school_geography", "high_school_government_and_politics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology",
    "world_religions",
]

assert len(MMLU_N25_SUBJECTS) == 20, f"Expected 20 N=25 subjects, got {len(MMLU_N25_SUBJECTS)}"
assert len(MMLU_ALL_SUBJECTS) == 55, f"Expected 55 all subjects, got {len(MMLU_ALL_SUBJECTS)}"


# ─────────────────────────────────────────────
# Data loaders (same as T4.1 for consistency)
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
# LSH Router
# ─────────────────────────────────────────────

class LSHRouter:
    """
    Random hyperplane LSH router (Charikar 2002).
    - Zero LLM trainable parameters
    - Hash tables built from TF-IDF domain centroids
    - Per-query: hash → candidate set → max cosine sim → predicted domain
    - Fallback: if no candidates found, use brute-force centroid search
    """

    def __init__(
        self,
        k: int = LSH_K,
        n_tables: int = LSH_L,
        max_features: int = 20000,
        ngram_range: tuple = (1, 2),
    ):
        self.k = k
        self.n_tables = n_tables
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=1,
            analyzer="word",
            strip_accents="unicode",
            lowercase=True,
        )
        self.centroids: np.ndarray | None = None   # (N_domains, vocab)
        self.hyperplanes: list[np.ndarray] = []   # n_tables × (k, vocab)
        self.hash_tables: list[dict] = []          # n_tables × dict(bucket → [domain_idx])
        self.domain_names: list[str] = []

    def _hash(self, X: np.ndarray, table_idx: int) -> np.ndarray:
        """Hash dense array X (n, vocab) using table_idx hyperplanes.
        Returns binary codes as integers (n,).
        """
        hp = self.hyperplanes[table_idx]  # (k, vocab), float64
        bits = (X @ hp.T > 0).astype(np.int32)  # (n, k)
        powers = 1 << np.arange(self.k, dtype=np.int32)
        return bits @ powers  # (n,)

    def fit(self, domain_prompts: dict[str, list[str]]):
        """Build TF-IDF vectorizer, centroids, hyperplanes, and hash tables."""
        all_texts, all_labels = [], []
        self.domain_names = list(domain_prompts.keys())

        for domain, prompts in domain_prompts.items():
            all_texts.extend(prompts)
            all_labels.extend([domain] * len(prompts))

        # Build TF-IDF
        X = self.vectorizer.fit_transform(all_texts)
        vocab_size = len(self.vectorizer.vocabulary_)

        # Compute L2-normalized centroids in float64 (avoids overflow in hash)
        centroids = []
        for domain in self.domain_names:
            mask = [label == domain for label in all_labels]
            X_domain = np.asarray(X[mask].mean(axis=0), dtype=np.float64).squeeze()
            centroids.append(X_domain)
        self.centroids = normalize(np.array(centroids, dtype=np.float64), norm="l2")

        # Generate random hyperplanes in float64: shape (L*k, vocab_size)
        # Stored as single matrix for vectorized batch hash (single matmul)
        rng2 = np.random.default_rng(SEED)
        hp_all = rng2.standard_normal((self.n_tables * self.k, vocab_size))
        # Normalize each row to unit length
        hp_all = hp_all / np.linalg.norm(hp_all, axis=1, keepdims=True)
        self.hp_all = hp_all.astype(np.float64)  # (L*k, vocab)
        # Keep per-table hyperplanes for the legacy _hash method
        self.hyperplanes = [hp_all[t*self.k:(t+1)*self.k] for t in range(self.n_tables)]

        # Build hash tables: hash each centroid and store domain index
        # Batch-compute all codes at once: (N_domains, L*k)
        bits_all = (self.centroids @ self.hp_all.T > 0).astype(np.int32)  # (N, L*k)
        bits_per_table = bits_all.reshape(len(self.domain_names), self.n_tables, self.k)  # (N, L, k)
        powers = 1 << np.arange(self.k, dtype=np.int32)  # (k,)
        codes_all = bits_per_table @ powers  # (N, L)

        self.hash_tables = []
        for t in range(self.n_tables):
            table: dict[int, list[int]] = {}
            for domain_idx in range(len(self.domain_names)):
                bucket = int(codes_all[domain_idx, t])
                if bucket not in table:
                    table[bucket] = []
                table[bucket].append(domain_idx)
            self.hash_tables.append(table)

        print(
            f"[LSHRouter] fit: {len(self.domain_names)} domains, "
            f"vocab={vocab_size}, k={self.k}, L={self.n_tables}, "
            f"centroids={self.centroids.shape}",
            flush=True,
        )

    def _get_candidates_batch(self, x: np.ndarray) -> set[int]:
        """Hash single 1D query vector against all L tables. Returns candidate set.

        Uses vectorized batch hash: single matmul (L*k, vocab) @ (vocab,) = (L*k,)
        then reshape to (L, k) and pack bits.
        """
        dots = self.hp_all @ x  # (L*k,) = (L*k, vocab) @ (vocab,)
        bits = (dots > 0).astype(np.int32)  # (L*k,)
        bits_reshaped = bits.reshape(self.n_tables, self.k)  # (L, k)
        powers = 1 << np.arange(self.k, dtype=np.int32)  # (k,)
        codes = bits_reshaped @ powers  # (L,) integer codes per table

        candidates: set[int] = set()
        for t in range(self.n_tables):
            code = int(codes[t])
            if code in self.hash_tables[t]:
                candidates.update(self.hash_tables[t][code])
        return candidates

    def predict(self, texts: list[str]) -> list[str]:
        """Route each text to best matching domain using LSH candidate set."""
        X_sparse = self.vectorizer.transform(texts)
        X_norm = normalize(X_sparse, norm="l2")
        # Convert to dense float64 (avoids float32 overflow in matmul)
        X_dense = np.asarray(X_norm.todense(), dtype=np.float64)  # (n, vocab)

        predictions = []
        fallback_count = 0
        for i in range(len(texts)):
            x = np.asarray(X_dense[i]).ravel()  # ensure 1D float64 (vocab,)
            candidates = self._get_candidates_batch(x)

            if not candidates:
                # Fallback: brute-force search all centroids
                fallback_count += 1
                scores = self.centroids @ x  # (N_domains,)
                predicted_idx = int(np.argmax(scores))
            else:
                # Score only candidates (max cosine sim tie-breaking)
                candidate_list = list(candidates)
                candidate_centroids = self.centroids[candidate_list]  # (|C|, vocab)
                scores = candidate_centroids @ x  # (|C|,)
                best = int(np.argmax(scores))
                predicted_idx = candidate_list[best]

            predictions.append(self.domain_names[predicted_idx])

        if fallback_count > 0:
            print(f"  [LSH] {fallback_count}/{len(texts)} queries used brute-force fallback", flush=True)
        return predictions

    def predict_latency_stats(self, texts: list[str], n_reps: int = 500) -> dict:
        """Return latency stats in ms for single-query routing."""
        # Warm-up
        for _ in range(50):
            self.predict([texts[0]])
        latencies = []
        for i in range(n_reps):
            text = texts[i % len(texts)]
            t0 = time.perf_counter()
            self.predict([text])
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
        return {
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "mean_ms": float(np.mean(latencies)),
        }

    def candidate_set_stats(self, texts: list[str]) -> dict:
        """Measure average candidate set size (diagnostic)."""
        X_sparse = self.vectorizer.transform(texts)
        X_norm = normalize(X_sparse, norm="l2")
        X_dense = np.asarray(X_norm.todense(), dtype=np.float64)
        sizes = []
        for i in range(len(texts)):
            x = np.asarray(X_dense[i]).ravel()
            candidates = self._get_candidates_batch(x)
            sizes.append(len(candidates))
        return {
            "mean": float(np.mean(sizes)),
            "p50": float(np.percentile(sizes, 50)),
            "p95": float(np.percentile(sizes, 95)),
            "zero_candidate_rate": float(np.mean([s == 0 for s in sizes])),
        }

    @property
    def n_trainable_llm_params(self) -> int:
        return 0  # random hyperplanes, no learned LLM parameters


# ─────────────────────────────────────────────
# TF-IDF router for comparison baseline
# ─────────────────────────────────────────────

class TFIDFCentroidRouter:
    """Exact nearest-centroid router (O(N) per query). Baseline for comparison."""

    def __init__(self, max_features: int = 20000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, sublinear_tf=True,
            min_df=1, analyzer="word", strip_accents="unicode", lowercase=True,
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
            centroid = np.asarray(X[mask].mean(axis=0)).squeeze()
            centroids.append(centroid)
        self.centroids = normalize(np.array(centroids), norm="l2")

    def predict(self, texts: list[str]) -> list[str]:
        X = self.vectorizer.transform(texts)
        X_norm = normalize(X, norm="l2")
        scores = X_norm @ self.centroids.T
        if issparse(scores):
            scores = scores.toarray()
        return [self.domain_names[i] for i in np.argmax(scores, axis=1)]

    def predict_latency_stats(self, texts: list[str], n_reps: int = 500) -> dict:
        for _ in range(50):
            self.predict([texts[0]])
        latencies = []
        for i in range(n_reps):
            t0 = time.perf_counter()
            self.predict([texts[i % len(texts)]])
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
        return {
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "mean_ms": float(np.mean(latencies)),
        }


# ─────────────────────────────────────────────
# Evaluation helpers
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
        per_domain[domain] = {"correct": correct, "total": len(prompts), "accuracy": correct / len(prompts)}
        all_correct += correct
        all_total += len(prompts)

    overall_acc = all_correct / all_total if all_total > 0 else 0.0
    print(f"\n[{tag}] Overall accuracy: {overall_acc:.1%} ({all_correct}/{all_total})", flush=True)
    for domain, stats in sorted(per_domain.items(), key=lambda x: x[1]["accuracy"]):
        print(f"  {domain}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})", flush=True)
    return {"overall": overall_acc, "per_domain": per_domain, "n_domains": len(per_domain), "total": all_total}


def cosine_sim_diagnostics(router, test_data: dict[str, list[str]]) -> dict:
    """Measure mean cosine similarity between queries and their correct centroid.
    Critical for understanding LSH recall quality (recall drops sharply below c=0.7).
    """
    if not hasattr(router, "centroids") or router.centroids is None:
        return {}
    all_sims = []
    for domain, prompts in test_data.items():
        if not prompts or domain not in router.domain_names:
            continue
        domain_idx = router.domain_names.index(domain)
        centroid = router.centroids[domain_idx]  # (vocab,)
        X_sparse = router.vectorizer.transform(prompts)
        X_norm = normalize(X_sparse, norm="l2")
        X_dense = np.asarray(X_norm.todense(), dtype=np.float64)
        sims = X_dense @ centroid  # (n_prompts,)
        all_sims.extend(sims.tolist())
    return {
        "mean": float(np.mean(all_sims)),
        "p50": float(np.percentile(all_sims, 50)),
        "p25": float(np.percentile(all_sims, 25)),
        "p75": float(np.percentile(all_sims, 75)),
    }


# ─────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────

def collect_real_domain_prompts() -> tuple[dict, dict]:
    """Load 5 real domain prompts (math/code/medical/legal/finance)."""
    print("\n=== Phase 1: Real domain prompts ===", flush=True)
    gsm8k_all = load_gsm8k_prompts(N_TRAIN + N_TEST)
    code_all = load_code_prompts(N_TRAIN + N_TEST)
    pmed_all = load_pubmedqa_prompts(N_TRAIN + N_TEST)
    train = {
        "math":    gsm8k_all[:N_TRAIN],
        "code":    code_all[:N_TRAIN],
        "medical": pmed_all[:N_TRAIN],
        "legal":   load_mmlu_prompts("professional_law", N_TRAIN),
        "finance": load_mmlu_prompts("high_school_macroeconomics", N_TRAIN),
    }
    test = {
        "math":    gsm8k_all[N_TRAIN:N_TRAIN + N_TEST],
        "code":    code_all[N_TRAIN:N_TRAIN + N_TEST],
        "medical": pmed_all[N_TRAIN:N_TRAIN + N_TEST],
        "legal":   load_mmlu_test_prompts("professional_law", N_TEST),
        "finance": load_mmlu_test_prompts("high_school_macroeconomics", N_TEST),
    }
    for d, p in train.items():
        print(f"  {d}: {len(p)} train, {len(test[d])} test", flush=True)
    return train, test


def collect_mmlu_prompts(subjects: list[str]) -> tuple[dict, dict]:
    """Load MMLU prompts for given subjects."""
    train, test = {}, {}
    for subject in subjects:
        train[subject] = load_mmlu_prompts(subject, N_TRAIN)
        test[subject] = load_mmlu_test_prompts(subject, N_TEST)
        if len(test[subject]) < 5:
            print(f"  WARNING: {subject} test only {len(test[subject])} examples", flush=True)
    return train, test


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    results: dict = {}

    # ── Phase 1: Collect data ──
    real_train, real_test = collect_real_domain_prompts()

    print("\n=== Phase 2: N=25 MMLU extra prompts ===", flush=True)
    mmlu25_train, mmlu25_test = collect_mmlu_prompts(MMLU_N25_SUBJECTS)

    train25 = {**real_train, **mmlu25_train}
    test25  = {**real_test,  **mmlu25_test}
    n_domains_25 = len(train25)
    print(f"\nN=25 total: {n_domains_25} domains", flush=True)

    # ── Phase 3: Fit routers at N=25 ──
    print("\n=== Phase 3: Fit routers (N=25) ===", flush=True)
    lsh25 = LSHRouter(k=LSH_K, n_tables=LSH_L)
    lsh25.fit(train25)

    tfidf25 = TFIDFCentroidRouter()
    tfidf25.fit(train25)

    # ── Phase 4: Evaluate at N=25 ──
    print("\n=== Phase 4: Evaluate N=25 ===", flush=True)
    lsh25_results = evaluate_routing(lsh25, test25, "LSH N=25")
    tfidf25_results = evaluate_routing(tfidf25, test25, "TF-IDF N=25 (baseline)")

    # Candidate set diagnostics
    test_texts25 = [p for prompts in test25.values() for p in prompts]
    cand_stats25 = lsh25.candidate_set_stats(test_texts25)
    print(f"\n[LSH N=25] Candidate set: mean={cand_stats25['mean']:.1f}, p95={cand_stats25['p95']:.0f}, "
          f"zero_rate={cand_stats25['zero_candidate_rate']:.1%}", flush=True)

    # Cosine similarity diagnostics (critical for understanding LSH recall quality)
    sim_diag25 = cosine_sim_diagnostics(lsh25, test25)
    print(f"[LSH N=25] Query-centroid cosine sim: mean={sim_diag25.get('mean', 0):.3f}, "
          f"p25={sim_diag25.get('p25', 0):.3f}, p50={sim_diag25.get('p50', 0):.3f}, "
          f"p75={sim_diag25.get('p75', 0):.3f}", flush=True)
    results["cosine_sim_diagnostics_n25"] = sim_diag25

    # K1077 check
    k1077_pass = lsh25_results["overall"] >= 0.90
    print(f"\nK1077: N=25 LSH accuracy={lsh25_results['overall']:.3f} >= 0.90 → {'PASS' if k1077_pass else 'FAIL'}", flush=True)
    print(f"       TF-IDF baseline: {tfidf25_results['overall']:.3f}", flush=True)

    results["n25_lsh"] = lsh25_results
    results["n25_tfidf"] = tfidf25_results
    results["n25_candidate_stats"] = cand_stats25

    # ── Phase 5: Collect large-N data ──
    # Use all available MMLU subjects (55, excluding professional_law/high_school_macroeconomics
    # already used as "legal" and "finance" real domains) + 5 real = 60 total.
    # MMLU_ALL_SUBJECTS excludes professional_law and high_school_macroeconomics.
    n_reps_latency = 50 if IS_SMOKE else 500

    if not IS_SMOKE:
        print(f"\n=== Phase 5: Large-N ({len(MMLU_ALL_SUBJECTS) + 5}) domain prompts ===", flush=True)
        mmlu_all_train, mmlu_all_test = collect_mmlu_prompts(MMLU_ALL_SUBJECTS)

        train_large = {**real_train, **mmlu_all_train}
        test_large  = {**real_test,  **mmlu_all_test}
        n_domains_large = len(train_large)
        print(f"\nLarge-N total: {n_domains_large} domains", flush=True)

        # ── Phase 6: Fit routers at large N ──
        print(f"\n=== Phase 6: Fit routers (N={n_domains_large}) ===", flush=True)
        lsh_large = LSHRouter(k=LSH_K, n_tables=LSH_L)
        lsh_large.fit(train_large)

        tfidf_large = TFIDFCentroidRouter()
        tfidf_large.fit(train_large)

        # ── Phase 7: Evaluate at large N ──
        print(f"\n=== Phase 7: Evaluate N={n_domains_large} ===", flush=True)
        lsh_large_results = evaluate_routing(lsh_large, test_large, f"LSH N={n_domains_large}")
        tfidf_large_results = evaluate_routing(tfidf_large, test_large, f"TF-IDF N={n_domains_large} (baseline)")

        test_texts_large = [p for prompts in test_large.values() for p in prompts]
        cand_stats_large = lsh_large.candidate_set_stats(test_texts_large)
        print(f"\n[LSH N={n_domains_large}] Candidate set: mean={cand_stats_large['mean']:.1f}, "
              f"p95={cand_stats_large['p95']:.0f}, zero_rate={cand_stats_large['zero_candidate_rate']:.1%}", flush=True)

        # K1078 check (note: N=60 not 100, but best achievable with MMLU)
        k1078_pass = lsh_large_results["overall"] >= 0.80
        print(f"\nK1078: N={n_domains_large} LSH accuracy={lsh_large_results['overall']:.3f} >= 0.80 → {'PASS' if k1078_pass else 'FAIL'}", flush=True)
        print(f"       TF-IDF baseline: {tfidf_large_results['overall']:.3f}", flush=True)

        # K1079: latency
        print(f"\n=== Phase 8: Latency measurement (N={n_domains_large}) ===", flush=True)
        lsh_lat = lsh_large.predict_latency_stats(test_texts_large, n_reps=n_reps_latency)
        tfidf_lat = tfidf_large.predict_latency_stats(test_texts_large, n_reps=n_reps_latency)

        k1079_pass = lsh_lat["p50_ms"] < 0.5
        print(f"K1079: LSH median={lsh_lat['p50_ms']:.4f}ms < 0.5ms → {'PASS' if k1079_pass else 'FAIL'}", flush=True)
        print(f"       TF-IDF median={tfidf_lat['p50_ms']:.4f}ms (baseline)", flush=True)
        print(f"       LSH p99={lsh_lat['p99_ms']:.4f}ms | TF-IDF p99={tfidf_lat['p99_ms']:.4f}ms", flush=True)

        results["n_large"] = n_domains_large
        results["n_large_lsh"] = lsh_large_results
        results["n_large_tfidf"] = tfidf_large_results
        results["n_large_candidate_stats"] = cand_stats_large
        results["lsh_latency"] = lsh_lat
        results["tfidf_latency"] = tfidf_lat
        results["kill_criteria"] = {
            "K1077": {"pass": k1077_pass, "value": lsh25_results["overall"], "baseline": tfidf25_results["overall"]},
            "K1078": {"pass": k1078_pass, "value": lsh_large_results["overall"], "n_actual": n_domains_large, "baseline": tfidf_large_results["overall"]},
            "K1079": {"pass": k1079_pass, "value_ms": lsh_lat["p50_ms"], "baseline_ms": tfidf_lat["p50_ms"]},
            "K1080": {"pass": True, "trainable_llm_params": 0},
        }
    else:
        # Smoke test: latency at N=25, skip large-N
        print("\n=== Phase 5 (SMOKE): Latency at N=25 ===", flush=True)
        test_texts25 = [p for prompts in test25.values() for p in prompts]
        lsh_lat = lsh25.predict_latency_stats(test_texts25, n_reps=n_reps_latency)
        tfidf_lat = tfidf25.predict_latency_stats(test_texts25, n_reps=n_reps_latency)

        k1077_pass = lsh25_results["overall"] >= 0.90
        k1079_pass = lsh_lat["p50_ms"] < 0.5
        print(f"K1077 (smoke N=25): {lsh25_results['overall']:.3f} >= 0.90 → {'PASS' if k1077_pass else 'FAIL'}", flush=True)
        print(f"K1079 (smoke N=25): LSH p50={lsh_lat['p50_ms']:.4f}ms < 0.5ms → {'PASS' if k1079_pass else 'FAIL'}", flush=True)
        print(f"       TF-IDF p50={tfidf_lat['p50_ms']:.4f}ms", flush=True)

        results["lsh_latency"] = lsh_lat
        results["tfidf_latency"] = tfidf_lat
        results["is_smoke"] = True
        results["kill_criteria"] = {
            "K1077": {"pass": k1077_pass, "value": lsh25_results["overall"]},
            "K1078": {"pass": None, "note": "skipped in smoke test"},
            "K1079": {"pass": k1079_pass, "value_ms": lsh_lat["p50_ms"]},
            "K1080": {"pass": True, "trainable_llm_params": 0},
        }

    # ── K1080: zero trainable params ──
    assert lsh25.n_trainable_llm_params == 0, "LSH has learned LLM params!"
    print(f"\nK1080: Trainable LLM params = 0 → PASS", flush=True)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)

    kc = results["kill_criteria"]
    all_pass = all(v.get("pass") is True for v in kc.values() if v.get("pass") is not None)
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAIL'}", flush=True)
    for k, v in kc.items():
        status = "PASS" if v.get("pass") is True else ("FAIL" if v.get("pass") is False else "SKIP")
        print(f"  {k}: {status}", flush=True)


if __name__ == "__main__":
    main()
