#!/usr/bin/env python
"""
TF-IDF routing accuracy on real NLP domains.
Experiment: exp_tfidf_routing_real_text
Kill criteria:
  K950: Nearest-centroid routing accuracy >= 0.80 on ALL 3 domains

Domains:
  math: GSM8K math word problem questions
  code: Python code instruction requests (natural language)
  text: CC News general text articles
"""

import json
import time
import numpy as np
from pathlib import Path
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize

SEED = 42
N_TRAIN = 200  # per domain
N_TEST = 100   # per domain
MAX_FEATURES = 10_000

OUT_DIR = Path("experiments/models/tfidf_routing_real_text")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("TF-IDF Routing: Real NLP Domains")
print("=" * 60)
t0 = time.time()

# ── 1. Load datasets ─────────────────────────────────────────────────────────
print("\n[1] Loading datasets...")

# Math: GSM8K questions
gsm = load_dataset("gsm8k", "main", split="train")
math_pool = [ex["question"] for ex in gsm]

# Code: Python code instructions (natural language coding requests)
code_ds = load_dataset(
    "iamtarun/python_code_instructions_18k_alpaca", split="train"
)
code_pool = [
    ex["instruction"].strip()
    for ex in code_ds
    if ex.get("instruction", "").strip()
]

# Text: CC News general text (streaming to avoid full download)
text_pool = []
cc = load_dataset("cc_news", split="train", streaming=True)
for ex in cc:
    body = ex.get("text", "").strip()
    if len(body) >= 100:
        text_pool.append(body[:600])  # first 600 chars
    if len(text_pool) >= 2000:
        break

print(f"  math pool  : {len(math_pool):,}")
print(f"  code pool  : {len(code_pool):,}")
print(f"  text pool  : {len(text_pool):,}")

# ── 2. Sample train / test (no overlap) ─────────────────────────────────────
print("\n[2] Sampling train/test splits...")
rng = np.random.default_rng(SEED)

def sample_split(pool, n_train, n_test):
    idx = rng.permutation(len(pool))
    train_idx = idx[:n_train]
    test_idx  = idx[n_train: n_train + n_test]
    return [pool[i] for i in train_idx], [pool[i] for i in test_idx]

math_train, math_test = sample_split(math_pool, N_TRAIN, N_TEST)
code_train, code_test = sample_split(code_pool, N_TRAIN, N_TEST)
text_train, text_test = sample_split(text_pool, N_TRAIN, N_TEST)

DOMAIN_NAMES = ["math", "code", "text"]
train_docs   = math_train + code_train + text_train
train_labels = [0] * N_TRAIN + [1] * N_TRAIN + [2] * N_TRAIN
test_docs    = math_test + code_test + text_test
test_labels  = [0] * N_TEST + [1] * N_TEST + [2] * N_TEST

print(f"  Train: {len(train_docs)} docs | Test: {len(test_docs)} docs")

# ── 3. Fit TF-IDF vectorizer ─────────────────────────────────────────────────
print(f"\n[3] Fitting TF-IDF (max_features={MAX_FEATURES}, sublinear_tf=True)...")
vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 2),  # unigrams + bigrams for better domain separation
)
X_train = vectorizer.fit_transform(train_docs)
X_test  = vectorizer.transform(test_docs)
print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")
print(f"  X_train shape: {X_train.shape}")

# ── 4. Centroid cosine analysis ──────────────────────────────────────────────
print("\n[4] Computing centroid cosine similarities...")
train_arr = X_train.toarray()
centroids = np.zeros((3, train_arr.shape[1]))
for label in range(3):
    mask = np.array(train_labels) == label
    centroids[label] = train_arr[mask].mean(axis=0)
centroids_norm = normalize(centroids)

cosines = {}
for i in range(3):
    for j in range(i + 1, 3):
        key = f"{DOMAIN_NAMES[i]}_{DOMAIN_NAMES[j]}"
        cosines[key] = float(centroids_norm[i] @ centroids_norm[j])
        print(f"  cos({DOMAIN_NAMES[i]}, {DOMAIN_NAMES[j]}) = {cosines[key]:.4f}")

# ── 5. Nearest Centroid routing ──────────────────────────────────────────────
print("\n[5] Nearest Centroid routing...")
nc = NearestCentroid()
nc.fit(train_arr, train_labels)

test_arr = X_test.toarray()
y_pred_nc = nc.predict(test_arr)

nc_per_domain = {}
for label, name in enumerate(DOMAIN_NAMES):
    mask = np.array(test_labels) == label
    acc  = accuracy_score(np.array(test_labels)[mask], y_pred_nc[mask])
    nc_per_domain[name] = float(acc)
    print(f"  NC {name}: {acc:.3f}")
nc_overall = float(accuracy_score(test_labels, y_pred_nc))
print(f"  NC overall: {nc_overall:.3f}")

cm_nc = confusion_matrix(test_labels, y_pred_nc).tolist()

# ── 6. Logistic Regression routing ───────────────────────────────────────────
print("\n[6] Logistic Regression routing...")
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, n_jobs=-1)
lr.fit(X_train, train_labels)
y_pred_lr = lr.predict(X_test)

lr_per_domain = {}
for label, name in enumerate(DOMAIN_NAMES):
    mask = np.array(test_labels) == label
    acc  = accuracy_score(np.array(test_labels)[mask], y_pred_lr[mask])
    lr_per_domain[name] = float(acc)
    print(f"  LR {name}: {acc:.3f}")
lr_overall = float(accuracy_score(test_labels, y_pred_lr))
print(f"  LR overall: {lr_overall:.3f}")

cm_lr = confusion_matrix(test_labels, y_pred_lr).tolist()

# ── 7. Top discriminating terms ──────────────────────────────────────────────
print("\n[7] Top discriminating terms per domain...")
feature_names = vectorizer.get_feature_names_out()
top_terms = {}
for label, name in enumerate(DOMAIN_NAMES):
    mask = np.array(train_labels) == label
    domain_mean = train_arr[mask].mean(axis=0)
    other_mean  = train_arr[~mask].mean(axis=0)
    scores = domain_mean - other_mean
    top_idx = np.argsort(scores)[-10:][::-1]
    terms = [feature_names[i] for i in top_idx]
    top_terms[name] = terms
    print(f"  {name}: {', '.join(terms[:5])}")

# ── 8. Kill criteria ─────────────────────────────────────────────────────────
nc_min = min(nc_per_domain.values())
k950_pass = nc_min >= 0.80

print("\n" + "=" * 60)
print("KILL CRITERIA")
print("=" * 60)
print(f"K950 (NC accuracy >= 0.80 on ALL domains):")
for name, acc in nc_per_domain.items():
    status = "OK" if acc >= 0.80 else "FAIL"
    print(f"  {name}: {acc:.3f} [{status}]")
print(f"  Min: {nc_min:.3f} → K950: {'PASS' if k950_pass else 'FAIL'}")

# ── 9. Save results ──────────────────────────────────────────────────────────
runtime = time.time() - t0
results = {
    "experiment": "exp_tfidf_routing_real_text",
    "kill_criteria": {
        "K950": {
            "status": "PASS" if k950_pass else "FAIL",
            "threshold": 0.80,
            "min_domain_accuracy": nc_min,
            "nearest_centroid_per_domain": nc_per_domain,
        }
    },
    "centroid_cosines": cosines,
    "nearest_centroid": {
        "per_domain": nc_per_domain,
        "overall": nc_overall,
        "confusion_matrix": cm_nc,
    },
    "logistic_regression": {
        "per_domain": lr_per_domain,
        "overall": lr_overall,
        "confusion_matrix": cm_lr,
    },
    "top_discriminating_terms": top_terms,
    "config": {
        "n_train_per_domain": N_TRAIN,
        "n_test_per_domain": N_TEST,
        "max_features": MAX_FEATURES,
        "ngram_range": [1, 2],
        "sublinear_tf": True,
        "seed": SEED,
    },
    "runtime_s": round(runtime, 1),
}

out_path = OUT_DIR / "results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nRuntime: {runtime:.1f}s")
print(f"Results: {out_path}")

# Final verdict
print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)
print(f"K950: {'✓ PASS' if k950_pass else '✗ FAIL'} (min domain accuracy = {nc_min:.3f})")
if not k950_pass:
    worst = min(nc_per_domain, key=nc_per_domain.get)
    print(f"  Worst domain: {worst} ({nc_per_domain[worst]:.3f})")
    print(f"  Check confusion matrix for which domain is being confused with {worst}")
