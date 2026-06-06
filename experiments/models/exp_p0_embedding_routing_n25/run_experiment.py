#!/usr/bin/env python3
"""
P0: Embedding Router at N=25 — Scaling Learned Classifier from N=10

Compare 6 routing methods on 25 domains (5 real + 20 MMLU subjects):
  1. TF-IDF centroid (Finding #431 baseline: 86.1%)
  2. TF-IDF + Ridge
  3. TF-IDF + Logistic Regression
  4. Sentence-embedding centroid (all-MiniLM-L6-v2)
  5. Sentence-embedding + Logistic Regression
  6. Combined TF-IDF + embedding + Logistic Regression

Kill criteria:
  K1473: Overall routing accuracy >= 90% at N=25
  K1474: Embedding-only accuracy >= 85% at N=25
  K1475: Combined TF-IDF+embedding accuracy >= 92% at N=25
  K1476: Worst-domain accuracy >= 70% at N=25
  K1477: Routing latency p99 < 5ms for N=25

Grounded by:
  Finding #525: Combined logistic 89.9% at N=10
  Finding #431: TF-IDF centroid 86.1% at N=25
  Finding #524: TF-IDF degrades sub-linearly with N
"""

import json
import os
import time
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
SEED = 42

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 20 if IS_SMOKE else 200
N_TEST = 10 if IS_SMOKE else 100

# 20 MMLU subjects (same as Finding #431)
MMLU_SUBJECTS = [
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

REAL_DOMAINS = ["math", "code", "medical", "legal", "finance"]
ALL_DOMAINS = REAL_DOMAINS + MMLU_SUBJECTS
assert len(ALL_DOMAINS) == 25, f"Expected 25 domains, got {len(ALL_DOMAINS)}"


def load_routing_texts(n_per_domain: int) -> dict[str, list[str]]:
    """Load routing text samples for all 25 domains."""
    import random
    from huggingface_hub import hf_hub_download
    import pandas as pd

    texts = {}
    rng = random.Random(SEED)

    # Math: GSM8K questions
    print("  Loading math (GSM8K)...", flush=True)
    path = hf_hub_download(
        "openai/gsm8k",
        "main/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    df = df.sample(n=min(n_per_domain, len(df)), random_state=SEED)
    texts["math"] = df["question"].tolist()

    # Code: CodeAlpaca instructions
    print("  Loading code (CodeAlpaca)...", flush=True)
    path = hf_hub_download(
        "sahil2801/CodeAlpaca-20k",
        "code_alpaca_20k.json",
        repo_type="dataset",
    )
    with open(path) as f:
        data = json.load(f)
    rng2 = random.Random(SEED)
    rng2.shuffle(data)
    texts["code"] = [ex["instruction"] for ex in data[:n_per_domain]]

    # Medical: MedMCQA questions
    print("  Loading medical (MedMCQA)...", flush=True)
    path = hf_hub_download(
        "openlifescienceai/medmcqa",
        "data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    df = df.sample(n=min(n_per_domain, len(df)), random_state=SEED)
    texts["medical"] = df["question"].tolist()

    # Legal: MMLU professional_law + jurisprudence + international_law
    print("  Loading legal (MMLU law subjects)...", flush=True)
    legal_subjects = ["professional_law", "jurisprudence", "international_law"]
    legal_texts = []
    for subj in legal_subjects:
        for split in ["auxiliary_train", "test", "validation", "dev"]:
            try:
                path = hf_hub_download(
                    "cais/mmlu",
                    f"{subj}/{split}-00000-of-00001.parquet",
                    repo_type="dataset",
                )
                df = pd.read_parquet(path)
                legal_texts.extend(df["question"].tolist())
            except Exception:
                continue
    rng.shuffle(legal_texts)
    texts["legal"] = legal_texts[:n_per_domain]

    # Finance: MMLU professional_accounting + econometrics
    print("  Loading finance (MMLU accounting/econometrics)...", flush=True)
    finance_subjects = ["professional_accounting", "econometrics"]
    finance_texts = []
    for subj in finance_subjects:
        for split in ["auxiliary_train", "test", "validation", "dev"]:
            try:
                path = hf_hub_download(
                    "cais/mmlu",
                    f"{subj}/{split}-00000-of-00001.parquet",
                    repo_type="dataset",
                )
                df = pd.read_parquet(path)
                finance_texts.extend(df["question"].tolist())
            except Exception:
                continue
    rng.shuffle(finance_texts)
    texts["finance"] = finance_texts[:n_per_domain]

    # 20 MMLU subjects
    for subject in MMLU_SUBJECTS:
        print(f"  Loading {subject}...", flush=True)
        subject_texts = []
        for split in ["auxiliary_train", "test", "validation", "dev"]:
            try:
                path = hf_hub_download(
                    "cais/mmlu",
                    f"{subject}/{split}-00000-of-00001.parquet",
                    repo_type="dataset",
                )
                df = pd.read_parquet(path)
                subject_texts.extend(df["question"].tolist())
            except Exception:
                continue
        rng.shuffle(subject_texts)
        texts[subject] = subject_texts[:n_per_domain]
        if len(texts[subject]) < 50:
            print(f"    WARNING: {subject} only has {len(texts[subject])} texts", flush=True)

    return texts


def split_train_test(texts: dict[str, list[str]], n_train: int, n_test: int):
    """Split texts into train/test sets."""
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for domain in ALL_DOMAINS:
        txts = texts.get(domain, [])
        n_avail = len(txts)
        n_tr = min(n_train, int(n_avail * 0.67))
        n_te = min(n_test, n_avail - n_tr)

        train_texts.extend(txts[:n_tr])
        train_labels.extend([domain] * n_tr)
        test_texts.extend(txts[n_tr : n_tr + n_te])
        test_labels.extend([domain] * n_te)

    return train_texts, train_labels, test_texts, test_labels


def eval_per_domain(preds, test_labels):
    """Compute overall and per-domain accuracy."""
    accuracy = sum(p == t for p, t in zip(preds, test_labels)) / len(test_labels) * 100
    per_domain = {}
    for domain in ALL_DOMAINS:
        d_preds = [p for p, t in zip(preds, test_labels) if t == domain]
        d_true = [t for t in test_labels if t == domain]
        if d_true:
            per_domain[domain] = (
                sum(p == t for p, t in zip(d_preds, d_true)) / len(d_true) * 100
            )
        else:
            per_domain[domain] = 0.0
    return accuracy, per_domain


# ── Routing methods ──


def method_tfidf_centroid(train_texts, train_labels, test_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestCentroid

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vec.fit_transform(train_texts)
    clf = NearestCentroid()
    clf.fit(X_train, train_labels)
    X_test = vec.transform(test_texts)
    return clf.predict(X_test).tolist()


def method_tfidf_ridge(train_texts, train_labels, test_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeClassifier

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vec.fit_transform(train_texts)
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train, train_labels)
    X_test = vec.transform(test_texts)
    return clf.predict(X_test).tolist()


def method_tfidf_logistic(train_texts, train_labels, test_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vec.fit_transform(train_texts)
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED)
    clf.fit(X_train, train_labels)
    X_test = vec.transform(test_texts)
    return clf.predict(X_test).tolist()


def encode_sentences(texts, batch_size=64):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
    )
    return embeddings


def method_embed_centroid(train_emb, train_labels, test_emb):
    labels_arr = np.array(train_labels)
    classes = sorted(set(train_labels))
    centroids = []
    for c in classes:
        mask = labels_arr == c
        centroid = train_emb[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        centroids.append(centroid)
    centroids = np.array(centroids)
    sims = test_emb @ centroids.T
    pred_idx = sims.argmax(axis=1)
    return [classes[i] for i in pred_idx]


def method_embed_logistic(train_emb, train_labels, test_emb):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED)
    clf.fit(train_emb, train_labels)
    return clf.predict(test_emb).tolist()


def method_combined_logistic(train_texts, train_labels, test_texts, train_emb, test_emb):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from scipy.sparse import hstack, csr_matrix

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vec.fit_transform(train_texts)
    X_test_tfidf = vec.transform(test_texts)

    X_train = hstack([X_train_tfidf, csr_matrix(train_emb)])
    X_test = hstack([X_test_tfidf, csr_matrix(test_emb)])

    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED)
    clf.fit(X_train, train_labels)
    return clf.predict(X_test).tolist()


def compute_fisher_ratio(features, labels):
    """Compute multi-class Fisher discriminant ratio."""
    classes = sorted(set(labels))
    if hasattr(features, "toarray"):
        features = features.toarray()
    features = np.array(features)
    labels = np.array(labels)

    grand_mean = features.mean(axis=0)
    S_B = 0.0
    S_W = 0.0
    for c in classes:
        mask = labels == c
        class_data = features[mask]
        class_mean = class_data.mean(axis=0)
        n_c = class_data.shape[0]
        diff = class_mean - grand_mean
        S_B += n_c * np.dot(diff, diff)
        S_W += np.sum(np.var(class_data, axis=0)) * n_c

    return float(S_B / max(S_W, 1e-10))


def measure_latency(predict_fn, test_texts, n_reps=500):
    """Measure p99 single-query routing latency in ms."""
    # Warm-up
    for _ in range(50):
        predict_fn([test_texts[0]])
    latencies = []
    for i in range(n_reps):
        text = test_texts[i % len(test_texts)]
        t0 = time.perf_counter()
        predict_fn([text])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    return float(np.percentile(latencies, 99))


def main():
    t_start = time.time()
    print("=" * 60)
    print("P0: Embedding Router at N=25")
    print("=" * 60, flush=True)

    # ── Phase 1: Load data ──
    print("\n[Phase 1] Loading routing texts for 25 domains...", flush=True)
    total_per_domain = N_TRAIN + N_TEST
    texts = load_routing_texts(total_per_domain)
    train_texts, train_labels, test_texts, test_labels = split_train_test(
        texts, N_TRAIN, N_TEST
    )
    print(
        f"  {len(train_texts)} train, {len(test_texts)} test, "
        f"{len(ALL_DOMAINS)} domains",
        flush=True,
    )

    # ── Phase 2: Compute embeddings ──
    print("\n[Phase 2] Computing sentence embeddings...", flush=True)
    t_embed = time.time()
    all_texts = train_texts + test_texts
    all_emb = encode_sentences(all_texts)
    n_train = len(train_texts)
    train_emb = all_emb[:n_train]
    test_emb = all_emb[n_train:]
    embed_time = time.time() - t_embed
    print(
        f"  Encoded {len(all_texts)} texts in {embed_time:.1f}s "
        f"(dim={train_emb.shape[1]})",
        flush=True,
    )

    # ── Phase 3: Fisher ratio analysis ──
    print("\n[Phase 3] Fisher discriminant analysis...", flush=True)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack, csr_matrix

    vec_fisher = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = vec_fisher.fit_transform(train_texts)
    fisher_tfidf = compute_fisher_ratio(X_tfidf, train_labels)
    fisher_embed = compute_fisher_ratio(train_emb, train_labels)
    X_combined = hstack([X_tfidf, csr_matrix(train_emb)])
    fisher_combined = compute_fisher_ratio(X_combined, train_labels)

    print(f"  Fisher ratio (TF-IDF):     {fisher_tfidf:.4f}", flush=True)
    print(f"  Fisher ratio (embedding):  {fisher_embed:.4f}", flush=True)
    print(f"  Fisher ratio (combined):   {fisher_combined:.4f}", flush=True)

    # ── Phase 4: Run all 6 methods ──
    print("\n[Phase 4] Running 6 routing methods...", flush=True)
    results = {}
    methods = [
        ("tfidf_centroid", lambda: method_tfidf_centroid(train_texts, train_labels, test_texts)),
        ("tfidf_ridge", lambda: method_tfidf_ridge(train_texts, train_labels, test_texts)),
        ("tfidf_logistic", lambda: method_tfidf_logistic(train_texts, train_labels, test_texts)),
        ("embed_centroid", lambda: method_embed_centroid(train_emb, train_labels, test_emb)),
        ("embed_logistic", lambda: method_embed_logistic(train_emb, train_labels, test_emb)),
        (
            "combined_logistic",
            lambda: method_combined_logistic(
                train_texts, train_labels, test_texts, train_emb, test_emb
            ),
        ),
    ]

    total_method_time = 0.0
    for name, fn in methods:
        t_method = time.time()
        preds = fn()
        elapsed = time.time() - t_method
        total_method_time += elapsed

        accuracy, per_domain = eval_per_domain(preds, test_labels)
        min_domain = min(per_domain.values())
        max_domain = max(per_domain.values())
        min_domain_name = min(per_domain, key=per_domain.get)

        results[name] = {
            "accuracy_pct": round(accuracy, 1),
            "per_domain": {d: round(a, 1) for d, a in per_domain.items()},
            "min_domain_pct": round(min_domain, 1),
            "min_domain_name": min_domain_name,
            "max_domain_pct": round(max_domain, 1),
            "time_s": round(elapsed, 3),
        }

        print(f"\n  {name}:", flush=True)
        print(
            f"    Overall: {accuracy:.1f}% (min={min_domain:.1f}% [{min_domain_name}], "
            f"max={max_domain:.1f}%), time={elapsed:.3f}s",
            flush=True,
        )
        for d in ALL_DOMAINS:
            print(f"      {d:40s}: {per_domain[d]:.1f}%", flush=True)

    # ── Phase 5: Latency measurement ──
    print("\n[Phase 5] Latency measurement (combined logistic)...", flush=True)
    # Build combined logistic once for latency test
    from sklearn.feature_extraction.text import TfidfVectorizer as TV2
    from sklearn.linear_model import LogisticRegression as LR2
    from scipy.sparse import hstack as hstack2, csr_matrix as csr2

    vec_lat = TV2(max_features=5000, ngram_range=(1, 2))
    X_lat = vec_lat.fit_transform(train_texts)
    X_lat_combined = hstack2([X_lat, csr2(train_emb)])
    clf_lat = LR2(C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED)
    clf_lat.fit(X_lat_combined, train_labels)

    # Pre-load sentence model for latency
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    def combined_predict_single(texts_in):
        tfidf_feat = vec_lat.transform(texts_in)
        emb_feat = st_model.encode(
            texts_in, batch_size=1, show_progress_bar=False, normalize_embeddings=True
        )
        X = hstack2([tfidf_feat, csr2(emb_feat)])
        return clf_lat.predict(X).tolist()

    p99_ms = measure_latency(combined_predict_single, test_texts)
    print(f"  Combined logistic p99 latency: {p99_ms:.2f}ms", flush=True)

    # ── Phase 6: Embedding space analysis ──
    print("\n[Phase 6] Embedding space analysis...", flush=True)
    centroids = {}
    for domain in ALL_DOMAINS:
        mask = [l == domain for l in train_labels]
        domain_emb = train_emb[np.array(mask)]
        if len(domain_emb) > 0:
            c = domain_emb.mean(axis=0)
            centroids[domain] = c / (np.linalg.norm(c) + 1e-10)

    cosine_matrix = {}
    min_margin = 1.0
    min_pair = ("", "")
    for i, d1 in enumerate(ALL_DOMAINS):
        for d2 in ALL_DOMAINS[i + 1 :]:
            if d1 in centroids and d2 in centroids:
                cos = float(np.dot(centroids[d1], centroids[d2]))
                cosine_matrix[f"{d1}-{d2}"] = round(cos, 4)
                margin = 1.0 - cos
                if margin < min_margin:
                    min_margin = margin
                    min_pair = (d1, d2)

    sorted_pairs = sorted(cosine_matrix.items(), key=lambda x: -x[1])
    print(f"  Min margin: {min_margin:.4f} ({min_pair[0]}-{min_pair[1]})", flush=True)
    print(f"  Top-10 most similar pairs:", flush=True)
    for pair, cos in sorted_pairs[:10]:
        print(f"    {pair}: cos={cos:.4f} (margin={1-cos:.4f})", flush=True)

    # ── Phase 7: Summary ──
    print("\n[Phase 7] Summary", flush=True)
    print("=" * 60, flush=True)

    best_name = max(results, key=lambda k: results[k]["accuracy_pct"])
    best = results[best_name]
    tfidf_centroid = results["tfidf_centroid"]

    total_time = time.time() - t_start

    print(f"  Best method: {best_name} ({best['accuracy_pct']:.1f}%)", flush=True)
    print(
        f"  TF-IDF centroid baseline: {tfidf_centroid['accuracy_pct']:.1f}%", flush=True
    )
    print(
        f"  Improvement: +{best['accuracy_pct'] - tfidf_centroid['accuracy_pct']:.1f}pp",
        flush=True,
    )

    # Kill criteria
    k1_pass = best["accuracy_pct"] >= 90.0
    k2_embed_best = max(
        results["embed_centroid"]["accuracy_pct"],
        results["embed_logistic"]["accuracy_pct"],
    )
    k2_pass = k2_embed_best >= 85.0
    k3_pass = results["combined_logistic"]["accuracy_pct"] >= 92.0
    k4_pass = best["min_domain_pct"] >= 70.0
    k5_pass = p99_ms < 5.0

    print(f"\n  K1473 (overall >= 90%):      {'PASS' if k1_pass else 'FAIL'} ({best['accuracy_pct']:.1f}%)", flush=True)
    print(f"  K1474 (embed >= 85%):        {'PASS' if k2_pass else 'FAIL'} ({k2_embed_best:.1f}%)", flush=True)
    print(f"  K1475 (combined >= 92%):     {'PASS' if k3_pass else 'FAIL'} ({results['combined_logistic']['accuracy_pct']:.1f}%)", flush=True)
    print(f"  K1476 (worst >= 70%):        {'PASS' if k4_pass else 'FAIL'} ({best['min_domain_pct']:.1f}% [{best['min_domain_name']}])", flush=True)
    print(f"  K1477 (latency < 5ms):       {'PASS' if k5_pass else 'FAIL'} ({p99_ms:.2f}ms)", flush=True)
    print(f"\n  Total time: {total_time:.1f}s", flush=True)

    # ── Save results ──
    output = {
        "experiment": "exp_p0_embedding_routing_n25",
        "n_domains": len(ALL_DOMAINS),
        "methods": results,
        "best_method": best_name,
        "best_accuracy_pct": best["accuracy_pct"],
        "tfidf_centroid_baseline_pct": tfidf_centroid["accuracy_pct"],
        "improvement_pp": round(best["accuracy_pct"] - tfidf_centroid["accuracy_pct"], 1),
        "fisher_ratios": {
            "tfidf": round(fisher_tfidf, 4),
            "embedding": round(fisher_embed, 4),
            "combined": round(fisher_combined, 4),
        },
        "embedding_analysis": {
            "min_margin": round(min_margin, 4),
            "min_pair": list(min_pair),
            "top_similar_pairs": sorted_pairs[:10],
        },
        "latency_p99_ms": round(p99_ms, 2),
        "kill_criteria": {
            "K1473_overall_ge90": k1_pass,
            "K1474_embed_ge85": k2_pass,
            "K1475_combined_ge92": k3_pass,
            "K1476_worst_ge70": k4_pass,
            "K1477_latency_lt5ms": k5_pass,
        },
        "timing": {
            "total_s": round(total_time, 1),
            "embedding_encode_s": round(embed_time, 1),
            "all_methods_s": round(total_method_time, 2),
        },
        "config": {
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "n_domains": len(ALL_DOMAINS),
            "domains": ALL_DOMAINS,
            "seed": SEED,
            "smoke": IS_SMOKE,
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
