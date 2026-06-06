#!/usr/bin/env python3
"""
Semantic Router at Macro Scale: Real Embeddings on Real Domains
================================================================

Follow-up to `exp_semantic_router` (KILLED at 27.3% top-1 on N=15 synthetic Markov chains).

Hypothesis: parent's information-theoretic ceiling was synthetic-data-specific, not
a property of routing. Real embeddings on real text escape the Markov-chain entropy budget.

Design (mirrors parent's 6-strategy menu, swaps data source):
  - N=25 domains: 5 real (math/code/medical/legal/finance) + 20 MMLU subjects
  - Embeddings: MiniLM-L6-v2 (D=384), L2-normalized
  - Strategies: hash_ring, keyword_freq, cosine_sim, lsh_partition, utterance_1nn, utterance_agg, oracle

Pre-registered KCs (target-gated per F#666):
  K1570 (proxy):   best strategy top-1 >= 37.3% (parent 27.3% + 10pp)
  K1946 (target):  best strategy top-3 >= 85% (behavioral: hierarchical/ensemble routing usable)

See MATH.md for the proof and prediction table.
"""

import hashlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# Apple Accelerate BLAS emits benign 'divide by zero / overflow / invalid' matmul
# warnings on M-series Macs even when the math is well-defined; outputs are correct
# (verified by comparing to sibling exp_p0_embedding_routing_n25 results).
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", message=".*matmul.*")

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
SEED = 42
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN_PER_DOMAIN = 20 if IS_SMOKE else 200
N_TEST_PER_DOMAIN = 10 if IS_SMOKE else 100
N_EXEMPLARS = 10 if IS_SMOKE else 50
LSH_N_PLANES = 32

MMLU_SUBJECTS = [
    "high_school_geography", "world_religions", "philosophy",
    "high_school_world_history", "prehistory", "high_school_european_history",
    "high_school_us_history", "astronomy", "electrical_engineering",
    "computer_security", "logical_fallacies", "high_school_statistics",
    "formal_logic", "high_school_government_and_politics", "sociology",
    "high_school_chemistry", "high_school_physics", "global_facts",
    "management", "marketing",
]
REAL_DOMAINS = ["math", "code", "medical", "legal", "finance"]
ALL_DOMAINS = REAL_DOMAINS + MMLU_SUBJECTS
assert len(ALL_DOMAINS) == 25

# Pre-registered kill criteria (from MATH.md §4 — do NOT edit after running)
KC = {
    "K1570": {
        "text": "Best strategy top-1 domain accuracy at macro (real MiniLM embeddings) beats parent's 27.3% by >=10pp",
        "type": "proxy",
        "threshold": 0.373,
        "compare": "ge",
    },
    "K1946": {
        "text": "Best strategy top-3 domain accuracy at N=25 >= 85% (F#666 target-gate: behavioral hierarchical routing is usable)",
        "type": "target",
        "threshold": 0.85,
        "compare": "ge",
    },
}


# =============================================================================
# Data loading  (identical sources to sibling exp_p0_embedding_routing_n25)
# =============================================================================

def load_routing_texts(n_per_domain):
    """Load per-domain text samples. Mirrors sibling experiment's loader for cache reuse."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    texts = {}
    rng = random.Random(SEED)

    print("  Loading math (GSM8K)...", flush=True)
    path = hf_hub_download("openai/gsm8k",
                           "main/train-00000-of-00001.parquet",
                           repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.sample(n=min(n_per_domain, len(df)), random_state=SEED)
    texts["math"] = df["question"].tolist()

    print("  Loading code (CodeAlpaca)...", flush=True)
    path = hf_hub_download("sahil2801/CodeAlpaca-20k",
                           "code_alpaca_20k.json",
                           repo_type="dataset")
    with open(path) as f:
        data = json.load(f)
    rng2 = random.Random(SEED)
    rng2.shuffle(data)
    texts["code"] = [ex["instruction"] for ex in data[:n_per_domain]]

    print("  Loading medical (MedMCQA)...", flush=True)
    path = hf_hub_download("openlifescienceai/medmcqa",
                           "data/train-00000-of-00001.parquet",
                           repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.sample(n=min(n_per_domain, len(df)), random_state=SEED)
    texts["medical"] = df["question"].tolist()

    print("  Loading legal (MMLU law)...", flush=True)
    legal_subjects = ["professional_law", "jurisprudence", "international_law"]
    legal_texts = []
    for subj in legal_subjects:
        for split in ["auxiliary_train", "test", "validation", "dev"]:
            try:
                p = hf_hub_download("cais/mmlu",
                                    f"{subj}/{split}-00000-of-00001.parquet",
                                    repo_type="dataset")
                df = pd.read_parquet(p)
                legal_texts.extend(df["question"].tolist())
            except Exception:
                continue
    rng.shuffle(legal_texts)
    texts["legal"] = legal_texts[:n_per_domain]

    print("  Loading finance (MMLU accounting/econometrics)...", flush=True)
    finance_subjects = ["professional_accounting", "econometrics"]
    finance_texts = []
    for subj in finance_subjects:
        for split in ["auxiliary_train", "test", "validation", "dev"]:
            try:
                p = hf_hub_download("cais/mmlu",
                                    f"{subj}/{split}-00000-of-00001.parquet",
                                    repo_type="dataset")
                df = pd.read_parquet(p)
                finance_texts.extend(df["question"].tolist())
            except Exception:
                continue
    rng.shuffle(finance_texts)
    texts["finance"] = finance_texts[:n_per_domain]

    for subject in MMLU_SUBJECTS:
        print(f"  Loading {subject}...", flush=True)
        subject_texts = []
        for split in ["auxiliary_train", "test", "validation", "dev"]:
            try:
                p = hf_hub_download("cais/mmlu",
                                    f"{subject}/{split}-00000-of-00001.parquet",
                                    repo_type="dataset")
                df = pd.read_parquet(p)
                subject_texts.extend(df["question"].tolist())
            except Exception:
                continue
        rng.shuffle(subject_texts)
        texts[subject] = subject_texts[:n_per_domain]
        if len(texts[subject]) < 20:
            print(f"    WARNING: {subject} has only {len(texts[subject])} texts", flush=True)

    return texts


def split_train_test(texts, n_train, n_test):
    """Deterministic domain-stratified split. All strategies see identical partitions."""
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    for d in ALL_DOMAINS:
        txts = texts.get(d, [])
        n_avail = len(txts)
        n_tr = min(n_train, int(n_avail * 0.67))
        n_te = min(n_test, n_avail - n_tr)
        train_texts.extend(txts[:n_tr])
        train_labels.extend([d] * n_tr)
        test_texts.extend(txts[n_tr:n_tr + n_te])
        test_labels.extend([d] * n_te)
    return train_texts, train_labels, test_texts, test_labels


# =============================================================================
# Embedding
# =============================================================================

def encode_sentences(texts, batch_size=64):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, batch_size=batch_size,
                       show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


# =============================================================================
# Routing strategies  (same 6 as parent exp_semantic_router, adapted for dense embs)
# =============================================================================

def route_hash_ring(test_texts, domain_names, n_virtual=150, seed=SEED):
    """Content-agnostic consistent hash ring — returns top-1 only (no top-K sensible)."""
    ring_pos, ring_name = [], []
    for name in domain_names:
        for v in range(n_virtual):
            h = hashlib.md5(f"{name}:{v}".encode()).hexdigest()
            ring_pos.append(int(h, 16) % (2**32))
            ring_name.append(name)
    order = np.argsort(ring_pos)
    sorted_pos = np.array(ring_pos)[order]
    sorted_name = [ring_name[i] for i in order]

    preds_top1 = []
    scores = np.zeros((len(test_texts), len(domain_names)), dtype=np.float32)
    dom2idx = {d: i for i, d in enumerate(domain_names)}
    for i, t in enumerate(test_texts):
        pos = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % (2**32)
        idx = int(np.searchsorted(sorted_pos, pos))
        if idx >= len(sorted_pos):
            idx = 0
        chosen = sorted_name[idx]
        preds_top1.append(chosen)
        scores[i, dom2idx[chosen]] = 1.0  # only top-1 defined
    return preds_top1, scores


def route_keyword_freq(train_texts, train_labels, test_texts, domain_names):
    """Character-level frequency L2 — parent's strategy on real text."""
    import string
    chars = string.printable
    ch2idx = {c: i for i, c in enumerate(chars)}
    V = len(chars)

    def freq(txt):
        v = np.zeros(V, dtype=np.float32)
        for c in txt.lower():
            if c in ch2idx:
                v[ch2idx[c]] += 1.0
        s = v.sum()
        return v / s if s > 0 else v

    labels_arr = np.array(train_labels)
    profiles = np.zeros((len(domain_names), V), dtype=np.float32)
    for i, d in enumerate(domain_names):
        mask = labels_arr == d
        mtxt = [train_texts[j] for j in np.where(mask)[0]]
        profiles[i] = np.mean([freq(t) for t in mtxt], axis=0) if mtxt else 0

    test_freqs = np.stack([freq(t) for t in test_texts], axis=0)   # (B, V)
    dists = np.sum((test_freqs[:, None, :] - profiles[None, :, :]) ** 2, axis=2)  # (B, N)
    scores = -dists  # higher = better
    return _topk_from_scores(scores, domain_names)


def route_cosine_centroid(train_emb, train_labels, test_emb, domain_names):
    labels_arr = np.array(train_labels)
    centroids = np.zeros((len(domain_names), train_emb.shape[1]), dtype=np.float32)
    for i, d in enumerate(domain_names):
        mask = labels_arr == d
        if mask.any():
            c = train_emb[mask].mean(axis=0)
            n = np.linalg.norm(c) + 1e-10
            centroids[i] = c / n
    scores = test_emb @ centroids.T    # (B, N), cosine since both normalized
    return _topk_from_scores(scores, domain_names)


def route_lsh(train_emb, train_labels, test_emb, domain_names,
              n_planes=LSH_N_PLANES, rng=None):
    if rng is None:
        rng = np.random.RandomState(SEED)
    D = train_emb.shape[1]
    planes = rng.randn(D, n_planes).astype(np.float32)
    planes /= np.linalg.norm(planes, axis=0, keepdims=True) + 1e-10
    train_codes = (train_emb @ planes > 0).astype(np.float32)   # (Ntr, P)
    test_codes = (test_emb @ planes > 0).astype(np.float32)     # (Nts, P)

    labels_arr = np.array(train_labels)
    code_means = np.zeros((len(domain_names), n_planes), dtype=np.float32)
    for i, d in enumerate(domain_names):
        mask = labels_arr == d
        if mask.any():
            code_means[i] = train_codes[mask].mean(axis=0)

    scores = test_codes @ code_means.T   # (B, N)
    return _topk_from_scores(scores, domain_names)


def route_utterance_1nn(train_emb, train_labels, test_emb, domain_names,
                         n_exemplars=N_EXEMPLARS):
    """1-NN over subsampled exemplars. Top-K = K-NN with domain-dedup from nearest-first order."""
    labels_arr = np.array(train_labels)
    ex_emb, ex_lab = [], []
    for d in domain_names:
        mask = np.where(labels_arr == d)[0][:n_exemplars]
        ex_emb.append(train_emb[mask])
        ex_lab.extend([d] * len(mask))
    ex_emb = np.concatenate(ex_emb, axis=0)   # (total, D)
    ex_lab = np.array(ex_lab)

    sims = test_emb @ ex_emb.T   # (B, total)
    # For each query, walk nearest-first, dedup domains; this gives a domain-ranking
    order = np.argsort(-sims, axis=1)         # (B, total)
    top1, top3, top5, scores = [], [], [], np.full((len(test_emb), len(domain_names)), -np.inf, dtype=np.float32)
    dom2idx = {d: i for i, d in enumerate(domain_names)}
    for b in range(len(test_emb)):
        seen = set()
        ranked_doms = []
        for ex_idx in order[b]:
            d = ex_lab[ex_idx]
            if d in seen:
                continue
            seen.add(d)
            ranked_doms.append(d)
            # Record first-hit similarity as the domain's score
            if np.isneginf(scores[b, dom2idx[d]]):
                scores[b, dom2idx[d]] = sims[b, ex_idx]
            if len(seen) == len(domain_names):
                break
        top1.append(ranked_doms[0])
        top3.append(set(ranked_doms[:3]))
        top5.append(set(ranked_doms[:5]))
    return top1, top3, top5, scores


def route_utterance_agg(train_emb, train_labels, test_emb, domain_names,
                         n_exemplars=N_EXEMPLARS):
    """Mean cosine similarity per domain over exemplars."""
    labels_arr = np.array(train_labels)
    ex_emb, ex_lab = [], []
    for d in domain_names:
        mask = np.where(labels_arr == d)[0][:n_exemplars]
        ex_emb.append(train_emb[mask])
        ex_lab.extend([d] * len(mask))
    ex_emb = np.concatenate(ex_emb, axis=0)
    ex_lab = np.array(ex_lab)

    sims = test_emb @ ex_emb.T   # (B, total)
    scores = np.zeros((len(test_emb), len(domain_names)), dtype=np.float32)
    for i, d in enumerate(domain_names):
        mask = ex_lab == d
        if mask.any():
            scores[:, i] = sims[:, mask].mean(axis=1)
    return _topk_from_scores(scores, domain_names)


def _topk_from_scores(scores, domain_names):
    """Shared top-1 / top-3 / top-5 computation from (B, N) scores matrix."""
    B = scores.shape[0]
    order = np.argsort(-scores, axis=1)       # (B, N) descending
    top1 = [domain_names[int(order[b, 0])] for b in range(B)]
    top3 = [set(domain_names[int(order[b, k])] for k in range(min(3, order.shape[1]))) for b in range(B)]
    top5 = [set(domain_names[int(order[b, k])] for k in range(min(5, order.shape[1]))) for b in range(B)]
    return top1, top3, top5, scores


# =============================================================================
# Evaluation
# =============================================================================

def compute_accuracy(top1, top3, top5, test_labels):
    n = len(test_labels)
    a1 = sum(1 for p, t in zip(top1, test_labels) if p == t) / n
    a3 = sum(1 for s, t in zip(top3, test_labels) if t in s) / n
    a5 = sum(1 for s, t in zip(top5, test_labels) if t in s) / n
    return float(a1), float(a3), float(a5)


def per_domain_accuracy(top1, test_labels, domain_names):
    out = {}
    for d in domain_names:
        idxs = [i for i, t in enumerate(test_labels) if t == d]
        if not idxs:
            out[d] = None
        else:
            out[d] = float(sum(1 for i in idxs if top1[i] == d) / len(idxs))
    return out


# =============================================================================
# Main
# =============================================================================

def run():
    t_start = time.time()
    print("=" * 72, flush=True)
    print("  exp_followup_semantic_router_macro", flush=True)
    print(f"  Config: N_TRAIN={N_TRAIN_PER_DOMAIN}, N_TEST={N_TEST_PER_DOMAIN}, "
          f"N_EX={N_EXEMPLARS}, IS_SMOKE={IS_SMOKE}", flush=True)
    print("=" * 72, flush=True)

    print("\n[1] Loading domain texts...", flush=True)
    texts = load_routing_texts(N_TRAIN_PER_DOMAIN + N_TEST_PER_DOMAIN)
    train_texts, train_labels, test_texts, test_labels = split_train_test(
        texts, N_TRAIN_PER_DOMAIN, N_TEST_PER_DOMAIN)
    print(f"  Train: {len(train_texts)}  Test: {len(test_texts)}", flush=True)

    print("\n[2] Encoding with MiniLM-L6-v2...", flush=True)
    t0 = time.time()
    all_emb = encode_sentences(train_texts + test_texts)
    train_emb = all_emb[:len(train_texts)]
    test_emb = all_emb[len(train_texts):]
    print(f"  Encoded {len(all_emb)} texts in {time.time()-t0:.1f}s; dim={train_emb.shape[1]}", flush=True)

    print("\n[3] Running 6 routing strategies (+ oracle)...", flush=True)
    strategies = {}
    rng = np.random.RandomState(SEED)

    # hash_ring (top-1 only by construction)
    t0 = time.time()
    h_top1, _ = route_hash_ring(test_texts, ALL_DOMAINS)
    # build trivial top-3/top-5 sets containing only top-1 (no information for other ranks)
    h_top3 = [{p} for p in h_top1]
    h_top5 = [{p} for p in h_top1]
    strategies["hash_ring"] = (h_top1, h_top3, h_top5, time.time() - t0)
    print(f"  hash_ring: {time.time()-t0:.2f}s", flush=True)

    # keyword_freq
    t0 = time.time()
    k_top1, k_top3, k_top5, _ = route_keyword_freq(train_texts, train_labels,
                                                    test_texts, ALL_DOMAINS)
    strategies["keyword_freq"] = (k_top1, k_top3, k_top5, time.time() - t0)
    print(f"  keyword_freq: {time.time()-t0:.2f}s", flush=True)

    # cosine_sim
    t0 = time.time()
    c_top1, c_top3, c_top5, _ = route_cosine_centroid(train_emb, train_labels,
                                                       test_emb, ALL_DOMAINS)
    strategies["cosine_sim"] = (c_top1, c_top3, c_top5, time.time() - t0)
    print(f"  cosine_sim: {time.time()-t0:.2f}s", flush=True)

    # lsh_partition
    t0 = time.time()
    l_top1, l_top3, l_top5, _ = route_lsh(train_emb, train_labels, test_emb,
                                           ALL_DOMAINS, rng=rng)
    strategies["lsh_partition"] = (l_top1, l_top3, l_top5, time.time() - t0)
    print(f"  lsh_partition: {time.time()-t0:.2f}s", flush=True)

    # utterance_1nn
    t0 = time.time()
    u1_top1, u1_top3, u1_top5, _ = route_utterance_1nn(train_emb, train_labels,
                                                         test_emb, ALL_DOMAINS)
    strategies["utterance_1nn"] = (u1_top1, u1_top3, u1_top5, time.time() - t0)
    print(f"  utterance_1nn: {time.time()-t0:.2f}s", flush=True)

    # utterance_agg
    t0 = time.time()
    ua_top1, ua_top3, ua_top5, _ = route_utterance_agg(train_emb, train_labels,
                                                         test_emb, ALL_DOMAINS)
    strategies["utterance_agg"] = (ua_top1, ua_top3, ua_top5, time.time() - t0)
    print(f"  utterance_agg: {time.time()-t0:.2f}s", flush=True)

    # oracle
    strategies["oracle"] = (list(test_labels),
                            [{t} for t in test_labels],
                            [{t} for t in test_labels], 0.0)

    # ----------------------------------------------------------------
    # Accuracy table
    # ----------------------------------------------------------------
    print("\n[4] Accuracy results:\n", flush=True)
    print(f"  {'strategy':<16s} {'top-1':>7s} {'top-3':>7s} {'top-5':>7s} {'elapsed':>8s}", flush=True)
    print(f"  {'-'*56}", flush=True)
    per_strategy = {}
    for sname, (top1, top3, top5, elapsed) in strategies.items():
        a1, a3, a5 = compute_accuracy(top1, top3, top5, test_labels)
        per_strategy[sname] = {
            "top1": a1, "top3": a3, "top5": a5, "elapsed_s": float(elapsed),
            "per_domain_top1": per_domain_accuracy(top1, test_labels, ALL_DOMAINS),
        }
        print(f"  {sname:<16s} {a1:>7.4f} {a3:>7.4f} {a5:>7.4f} {elapsed:>7.2f}s", flush=True)

    # ----------------------------------------------------------------
    # KC evaluation (pre-registered, target-gated per F#666)
    # ----------------------------------------------------------------
    semantic_strategies = ["keyword_freq", "cosine_sim", "lsh_partition",
                           "utterance_1nn", "utterance_agg"]
    best_top1_strategy = max(semantic_strategies, key=lambda s: per_strategy[s]["top1"])
    best_top3_strategy = max(semantic_strategies, key=lambda s: per_strategy[s]["top3"])
    best_top1 = per_strategy[best_top1_strategy]["top1"]
    best_top3 = per_strategy[best_top3_strategy]["top3"]

    k1570_pass = best_top1 >= KC["K1570"]["threshold"]
    k1946_pass = best_top3 >= KC["K1946"]["threshold"]

    print("\n[5] Kill Criteria (target-gated per F#666):\n", flush=True)
    print(f"  K1570 (proxy, >= {KC['K1570']['threshold']}):  "
          f"best_top1={best_top1:.4f} [{best_top1_strategy}] -> "
          f"{'PASS' if k1570_pass else 'FAIL'}", flush=True)
    print(f"  K1946 (target, >= {KC['K1946']['threshold']}): "
          f"best_top3={best_top3:.4f} [{best_top3_strategy}] -> "
          f"{'PASS' if k1946_pass else 'FAIL'}", flush=True)

    # F#666 verdict matrix
    if k1570_pass and k1946_pass:
        verdict = "SUPPORTED"
        reason = "K1570 PASS ∧ K1946 PASS: real embeddings escape parent's information bottleneck and enable usable hierarchical routing"
    elif (not k1570_pass) and (not k1946_pass):
        verdict = "KILLED"
        reason = "K1570 FAIL ∧ K1946 FAIL: real embeddings do not escape the ceiling; routing is fundamentally limited"
    elif k1570_pass and not k1946_pass:
        verdict = "PROVISIONAL"
        reason = "K1570 PASS ∧ K1946 FAIL: proxy top-1 passes but top-3 does not — anomalous, investigate top-K calibration"
    else:  # proxy fail, target pass
        verdict = "PROVISIONAL"
        reason = "K1570 FAIL ∧ K1946 PASS: proxy top-1 mis-calibrated; real routing usable via top-K fallback"

    print(f"\n  F#666 verdict: {verdict} — {reason}", flush=True)

    all_pass = k1570_pass and k1946_pass
    output = {
        "config": {
            "n_train_per_domain": N_TRAIN_PER_DOMAIN,
            "n_test_per_domain": N_TEST_PER_DOMAIN,
            "n_exemplars": N_EXEMPLARS,
            "n_domains": len(ALL_DOMAINS),
            "lsh_n_planes": LSH_N_PLANES,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": int(train_emb.shape[1]),
            "seed": SEED,
            "is_smoke": IS_SMOKE,
        },
        "per_strategy": per_strategy,
        "kill_criteria": {
            "K1570": {"text": KC["K1570"]["text"], "threshold": KC["K1570"]["threshold"],
                      "type": "proxy",
                      "measured": best_top1, "best_strategy": best_top1_strategy,
                      "pass": bool(k1570_pass)},
            "K1946": {"text": KC["K1946"]["text"], "threshold": KC["K1946"]["threshold"],
                      "type": "target",
                      "measured": best_top3, "best_strategy": best_top3_strategy,
                      "pass": bool(k1946_pass)},
        },
        "verdict": verdict,
        "verdict_reason": reason,
        "all_pass": bool(all_pass),
        "is_smoke": IS_SMOKE,
        "elapsed_seconds": float(time.time() - t_start),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Wrote {RESULTS_FILE} ({time.time() - t_start:.1f}s total)", flush=True)
    return output


if __name__ == "__main__":
    run()
