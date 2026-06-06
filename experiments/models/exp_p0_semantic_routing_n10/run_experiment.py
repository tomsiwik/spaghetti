#!/usr/bin/env python3
"""
P0: Learned Classifier Routing at N=10 — Feature Space Comparison

Compare 6 routing methods on 10 domains from exp_p0_ttlora_n10_scaling:
  1. TF-IDF centroid (nearest-neighbor baseline)
  2. TF-IDF + Ridge (Finding #524 baseline: 79.3%)
  3. TF-IDF + Logistic Regression (cross-entropy loss)
  4. Sentence-embedding centroid (all-MiniLM-L6-v2)
  5. Sentence-embedding + Logistic Regression
  6. Combined TF-IDF + embedding + Logistic Regression

Kill criteria:
  K1443: Best routing method >= 90% at N=10
  K1444: Sentence-embedding routing >= 85% at N=10
  K1445: Best method >= 85% on ALL domains, no domain < 70%
  K1446: Router training + inference < 5 seconds total

Grounded by:
  Finding #524: TF-IDF + Ridge = 79.3% at N=10
  Finding #255: Sentence-embedding = 96% at N=5
  Finding #256: Sentence-embedding collapses at N=24 (33.3%)
  Finding #207: TF-IDF + logistic = 90% at N=5
  arXiv:2402.09997 (LoraRetriever)
"""

import json
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
N10_DIR = EXPERIMENT_DIR.parent / "exp_p0_ttlora_n10_scaling"
SEED = 42

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 20 if IS_SMOKE else 300
N_TEST = 10 if IS_SMOKE else 150

MMLU_DOMAINS = OrderedDict([
    ("science", ["astronomy", "college_biology", "college_chemistry", "college_physics"]),
    ("legal", ["professional_law", "jurisprudence", "international_law"]),
    ("finance", ["professional_accounting", "econometrics", "marketing"]),
    ("history", ["high_school_us_history", "high_school_world_history", "prehistory"]),
    ("psychology", ["professional_psychology", "high_school_psychology"]),
    ("philosophy", ["philosophy", "formal_logic", "logical_fallacies"]),
    ("engineering", ["electrical_engineering", "computer_security", "college_computer_science"]),
])

ALL_DOMAINS = ["math", "code", "medical"] + list(MMLU_DOMAINS.keys())


def load_routing_texts(n_per_domain):
    """Load routing text samples for all 10 domains from HuggingFace."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    texts = {}
    rng = random.Random(SEED + 1)

    # Math: GSM8K questions
    path = hf_hub_download("openai/gsm8k",
        "main/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.sample(n=min(n_per_domain, len(df)), random_state=SEED + 1)
    texts["math"] = df["question"].tolist()

    # Code: CodeAlpaca instructions
    path = hf_hub_download("sahil2801/CodeAlpaca-20k",
        "code_alpaca_20k.json", repo_type="dataset")
    with open(path) as f:
        data = json.load(f)
    rng2 = random.Random(SEED + 1)
    rng2.shuffle(data)
    texts["code"] = [ex["instruction"] for ex in data[:n_per_domain]]

    # Medical: MedMCQA questions
    path = hf_hub_download("openlifescienceai/medmcqa",
        "data/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.sample(n=min(n_per_domain, len(df)), random_state=SEED + 1)
    texts["medical"] = df["question"].tolist()

    # MMLU domains
    for domain, subjects in MMLU_DOMAINS.items():
        domain_texts = []
        for subject in subjects:
            for split in ["test", "validation", "dev"]:
                try:
                    path = hf_hub_download(
                        "cais/mmlu",
                        f"{subject}/{split}-00000-of-00001.parquet",
                        repo_type="dataset"
                    )
                    df = pd.read_parquet(path)
                    domain_texts.extend(df["question"].tolist())
                except Exception:
                    continue
        rng.shuffle(domain_texts)
        texts[domain] = domain_texts[:n_per_domain]
        if len(texts[domain]) < 50:
            print(f"  WARNING: {domain} only has {len(texts[domain])} texts",
                  flush=True)

    return texts


def split_train_test(texts, n_train, n_test):
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
        test_texts.extend(txts[n_tr:n_tr + n_te])
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
            per_domain[domain] = sum(p == t for p, t in zip(d_preds, d_true)) / len(d_true) * 100
        else:
            per_domain[domain] = 0.0
    return accuracy, per_domain


def method_tfidf_centroid(train_texts, train_labels, test_texts):
    """TF-IDF nearest-centroid routing."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestCentroid

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vec.fit_transform(train_texts)
    clf = NearestCentroid()
    clf.fit(X_train, train_labels)
    X_test = vec.transform(test_texts)
    return clf.predict(X_test).tolist()


def method_tfidf_ridge(train_texts, train_labels, test_texts):
    """TF-IDF + RidgeClassifier (Finding #524 baseline)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeClassifier

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vec.fit_transform(train_texts)
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train, train_labels)
    X_test = vec.transform(test_texts)
    return clf.predict(X_test).tolist()


def method_tfidf_logistic(train_texts, train_labels, test_texts):
    """TF-IDF + Logistic Regression (cross-entropy loss)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vec.fit_transform(train_texts)
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                             random_state=SEED)
    clf.fit(X_train, train_labels)
    X_test = vec.transform(test_texts)
    return clf.predict(X_test).tolist()


def encode_sentences(texts, batch_size=64):
    """Encode texts with sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, batch_size=batch_size,
                              show_progress_bar=False, normalize_embeddings=True)
    return embeddings


def method_embed_centroid(train_emb, train_labels, test_emb):
    """Sentence-embedding cosine nearest-centroid routing."""
    labels_arr = np.array(train_labels)
    classes = sorted(set(train_labels))
    centroids = []
    for c in classes:
        mask = labels_arr == c
        centroid = train_emb[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        centroids.append(centroid)
    centroids = np.array(centroids)

    # Cosine similarity: test_emb already L2-normalized from encode
    sims = test_emb @ centroids.T  # (n_test, n_classes)
    pred_idx = sims.argmax(axis=1)
    return [classes[i] for i in pred_idx]


def method_embed_logistic(train_emb, train_labels, test_emb):
    """Sentence-embedding + Logistic Regression."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                             random_state=SEED)
    clf.fit(train_emb, train_labels)
    return clf.predict(test_emb).tolist()


def method_combined_logistic(train_texts, train_labels, test_texts,
                             train_emb, test_emb):
    """Combined TF-IDF + sentence-embedding + Logistic Regression."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from scipy.sparse import hstack, csr_matrix

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vec.fit_transform(train_texts)
    X_test_tfidf = vec.transform(test_texts)

    # Combine: sparse TF-IDF + dense embeddings
    X_train = hstack([X_train_tfidf, csr_matrix(train_emb)])
    X_test = hstack([X_test_tfidf, csr_matrix(test_emb)])

    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                             random_state=SEED)
    clf.fit(X_train, train_labels)
    return clf.predict(X_test).tolist()


def compute_fisher_ratio(features, labels):
    """Compute multi-class Fisher discriminant ratio."""
    classes = sorted(set(labels))
    if hasattr(features, 'toarray'):
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


def main():
    t_start = time.time()
    print("=" * 60)
    print("P0: Learned Classifier Routing at N=10")
    print("=" * 60, flush=True)

    # ── Phase 1: Load data ──
    print("\n[Phase 1] Loading routing texts...", flush=True)
    total_per_domain = N_TRAIN + N_TEST
    texts = load_routing_texts(total_per_domain)
    train_texts, train_labels, test_texts, test_labels = split_train_test(
        texts, N_TRAIN, N_TEST)
    print(f"  {len(train_texts)} train, {len(test_texts)} test, "
          f"{len(ALL_DOMAINS)} domains", flush=True)

    # ── Phase 2: Compute embeddings ──
    print("\n[Phase 2] Computing sentence embeddings...", flush=True)
    t_embed = time.time()
    all_texts = train_texts + test_texts
    all_emb = encode_sentences(all_texts)
    n_train = len(train_texts)
    train_emb = all_emb[:n_train]
    test_emb = all_emb[n_train:]
    embed_time = time.time() - t_embed
    print(f"  Encoded {len(all_texts)} texts in {embed_time:.1f}s "
          f"(dim={train_emb.shape[1]})", flush=True)

    # ── Phase 3: Fisher ratio analysis ──
    print("\n[Phase 3] Fisher discriminant analysis...", flush=True)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec_fisher = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = vec_fisher.fit_transform(train_texts)
    fisher_tfidf = compute_fisher_ratio(X_tfidf, train_labels)
    fisher_embed = compute_fisher_ratio(train_emb, train_labels)

    # Combined features for Fisher ratio
    from scipy.sparse import hstack, csr_matrix
    X_combined = hstack([X_tfidf, csr_matrix(train_emb)])
    fisher_combined = compute_fisher_ratio(X_combined, train_labels)

    print(f"  Fisher ratio (TF-IDF):     {fisher_tfidf:.3f}", flush=True)
    print(f"  Fisher ratio (embedding):  {fisher_embed:.3f}", flush=True)
    print(f"  Fisher ratio (combined):   {fisher_combined:.3f}", flush=True)

    # ── Phase 4: Run all 6 methods ──
    print("\n[Phase 4] Running 6 routing methods...", flush=True)
    results = {}
    methods = [
        ("tfidf_centroid", lambda: method_tfidf_centroid(
            train_texts, train_labels, test_texts)),
        ("tfidf_ridge", lambda: method_tfidf_ridge(
            train_texts, train_labels, test_texts)),
        ("tfidf_logistic", lambda: method_tfidf_logistic(
            train_texts, train_labels, test_texts)),
        ("embed_centroid", lambda: method_embed_centroid(
            train_emb, train_labels, test_emb)),
        ("embed_logistic", lambda: method_embed_logistic(
            train_emb, train_labels, test_emb)),
        ("combined_logistic", lambda: method_combined_logistic(
            train_texts, train_labels, test_texts, train_emb, test_emb)),
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

        results[name] = {
            "accuracy_pct": round(accuracy, 1),
            "per_domain": {d: round(a, 1) for d, a in per_domain.items()},
            "min_domain_pct": round(min_domain, 1),
            "max_domain_pct": round(max_domain, 1),
            "time_s": round(elapsed, 3),
        }

        print(f"\n  {name}:", flush=True)
        print(f"    Overall: {accuracy:.1f}% (min={min_domain:.1f}%, "
              f"max={max_domain:.1f}%), time={elapsed:.3f}s", flush=True)
        for d in ALL_DOMAINS:
            print(f"      {d:15s}: {per_domain[d]:.1f}%", flush=True)

    # ── Phase 5: Inter-centroid analysis (embeddings) ──
    print("\n[Phase 5] Embedding space analysis...", flush=True)
    centroids = {}
    for domain in ALL_DOMAINS:
        mask = [l == domain for l in train_labels]
        domain_emb = train_emb[np.array(mask)]
        centroids[domain] = domain_emb.mean(axis=0)

    cosine_matrix = {}
    min_margin = 1.0
    min_pair = ("", "")
    for i, d1 in enumerate(ALL_DOMAINS):
        for d2 in ALL_DOMAINS[i+1:]:
            cos = float(np.dot(centroids[d1], centroids[d2]))
            cosine_matrix[f"{d1}-{d2}"] = round(cos, 4)
            margin = 1.0 - cos
            if margin < min_margin:
                min_margin = margin
                min_pair = (d1, d2)

    print(f"  Min margin: {min_margin:.4f} ({min_pair[0]}-{min_pair[1]})",
          flush=True)
    print(f"  Top-5 most similar pairs:", flush=True)
    sorted_pairs = sorted(cosine_matrix.items(), key=lambda x: -x[1])
    for pair, cos in sorted_pairs[:5]:
        print(f"    {pair}: cos={cos:.4f} (margin={1-cos:.4f})", flush=True)

    # ── Phase 6: Summary ──
    print("\n[Phase 6] Summary", flush=True)
    print("=" * 60, flush=True)

    best_name = max(results, key=lambda k: results[k]["accuracy_pct"])
    best = results[best_name]
    baseline = results["tfidf_ridge"]

    total_time = time.time() - t_start

    print(f"  Best method: {best_name} ({best['accuracy_pct']:.1f}%)", flush=True)
    print(f"  Baseline (TF-IDF+Ridge): {baseline['accuracy_pct']:.1f}%", flush=True)
    print(f"  Improvement: +{best['accuracy_pct'] - baseline['accuracy_pct']:.1f}pp",
          flush=True)

    # Kill criteria evaluation
    k1_pass = best["accuracy_pct"] >= 90.0
    k2_embed_best = max(results["embed_centroid"]["accuracy_pct"],
                        results["embed_logistic"]["accuracy_pct"])
    k2_pass = k2_embed_best >= 85.0
    k3_pass = best["min_domain_pct"] >= 70.0  # Relaxed: no domain < 70%
    k3_all_85 = best["min_domain_pct"] >= 85.0  # Strict: all >= 85%
    k4_pass = total_method_time < 5.0

    print(f"\n  K1443 (best >= 90%):     {'PASS' if k1_pass else 'FAIL'} "
          f"({best['accuracy_pct']:.1f}%)", flush=True)
    print(f"  K1444 (embed >= 85%):    {'PASS' if k2_pass else 'FAIL'} "
          f"({k2_embed_best:.1f}%)", flush=True)
    print(f"  K1445 (all >= 85%):      {'PASS' if k3_all_85 else 'FAIL'} "
          f"(min={best['min_domain_pct']:.1f}%)", flush=True)
    print(f"    (relaxed: no < 70%):   {'PASS' if k3_pass else 'FAIL'}", flush=True)
    print(f"  K1446 (< 5s):            {'PASS' if k4_pass else 'FAIL'} "
          f"({total_method_time:.2f}s)", flush=True)
    print(f"\n  Total time: {total_time:.1f}s", flush=True)

    # ── Save results ──
    output = {
        "experiment": "exp_p0_semantic_routing_n10",
        "methods": results,
        "best_method": best_name,
        "best_accuracy_pct": best["accuracy_pct"],
        "baseline_accuracy_pct": baseline["accuracy_pct"],
        "improvement_pp": round(best["accuracy_pct"] - baseline["accuracy_pct"], 1),
        "fisher_ratios": {
            "tfidf": round(fisher_tfidf, 3),
            "embedding": round(fisher_embed, 3),
            "combined": round(fisher_combined, 3),
        },
        "embedding_analysis": {
            "min_margin": round(min_margin, 4),
            "min_pair": list(min_pair),
            "top_similar_pairs": sorted_pairs[:5],
        },
        "kill_criteria": {
            "K1443_best_ge90": k1_pass,
            "K1444_embed_ge85": k2_pass,
            "K1445_all_ge85": k3_all_85,
            "K1445_relaxed_none_lt70": k3_pass,
            "K1446_time_lt5s": k4_pass,
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
