#!/usr/bin/env python3
"""
P0.E2E: Combined Logistic Routing → Adapter Generation at N=3

Tests whether combined logistic routing (TF-IDF + embedding) can drive
adapter selection for actual generation without quality degradation.

Kill criteria:
  K1478: GSM8K via routed adapter >= 65%
  K1479: HumanEval via routed adapter >= 50%
  K1480: MedMCQA via routed adapter >= 40%
  K1481: Routing-induced quality loss <= 5pp vs oracle routing

Reuses trained adapters from exp_p0_e2e_benchmark (Finding #508).
Uses hf_hub_download + parquet (Python 3.14 compatible, no dill).
"""

import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# Memory safety
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

# Reuse adapters from Finding #508
ADAPTER_DIR = EXPERIMENT_DIR.parent / "exp_p0_e2e_benchmark" / "adapters"
DOMAINS = ["math", "code", "medical"]

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 5 if IS_SMOKE else 100
N_ROUTE_TRAIN = 20 if IS_SMOKE else 300
N_ROUTE_TEST = 10 if IS_SMOKE else 150
SEED = 42


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ─────────────────────────────────────────────
# Data loading (hf_hub_download, no load_dataset)
# ─────────────────────────────────────────────

def load_gsm8k_test(n, seed=SEED):
    """Load GSM8K test set via parquet."""
    path = hf_hub_download("openai/gsm8k",
        "main/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    return [{"question": r["question"], "answer": r["answer"]} for _, r in df.iterrows()]


def load_humaneval_test(n):
    """Load HumanEval test set via parquet."""
    path = hf_hub_download("openai_humaneval",
        "openai_humaneval/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.head(min(n, len(df)))
    return [{"prompt": r["prompt"], "test": r["test"],
             "entry_point": r["entry_point"]} for _, r in df.iterrows()]


def load_medmcqa_val(n, seed=SEED):
    """Load MedMCQA validation set via parquet."""
    path = hf_hub_download("openlifescienceai/medmcqa",
        "data/validation-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    return [{"question": r["question"], "opa": r["opa"], "opb": r["opb"],
             "opc": r["opc"], "opd": r["opd"], "cop": r["cop"]} for _, r in df.iterrows()]


# ─────────────────────────────────────────────
# Phase 1: Train combined logistic router
# ─────────────────────────────────────────────

def phase_train_router():
    """Train combined TF-IDF + embedding logistic router for N=3 domains."""
    from scipy.sparse import csr_matrix, hstack
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    log("\n[Phase 1] Training combined logistic router")
    t0 = time.time()

    n_total = N_ROUTE_TRAIN + N_ROUTE_TEST
    texts = {}
    rng = np.random.RandomState(SEED + 1)

    # Math: GSM8K train
    path = hf_hub_download("openai/gsm8k",
        "main/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    idx = rng.choice(len(df), min(n_total, len(df)), replace=False)
    texts["math"] = df.iloc[idx]["question"].tolist()

    # Code: CodeAlpaca
    path = hf_hub_download("sahil2801/CodeAlpaca-20k",
        "code_alpaca_20k.json", repo_type="dataset")
    with open(path) as f:
        data = json.load(f)
    rng.shuffle(data)
    texts["code"] = [ex["instruction"] for ex in data[:n_total]]

    # Medical: MedMCQA train
    path = hf_hub_download("openlifescienceai/medmcqa",
        "data/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    idx = rng.choice(len(df), min(n_total, len(df)), replace=False)
    texts["medical"] = df.iloc[idx]["question"].tolist()

    # Split train/test
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    for domain in DOMAINS:
        txts = texts[domain]
        n_tr = min(N_ROUTE_TRAIN, len(txts))
        n_te = min(N_ROUTE_TEST, len(txts) - n_tr)
        train_texts.extend(txts[:n_tr])
        train_labels.extend([domain] * n_tr)
        test_texts.extend(txts[n_tr:n_tr + n_te])
        test_labels.extend([domain] * n_te)

    log(f"  Router: {len(train_texts)} train, {len(test_texts)} test, {len(DOMAINS)} domains")

    # TF-IDF features
    tfidf_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf_vec.fit_transform(train_texts)
    X_test_tfidf = tfidf_vec.transform(test_texts)

    # Sentence embedding features
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    all_texts = train_texts + test_texts
    all_emb = embed_model.encode(all_texts, batch_size=64,
                                  show_progress_bar=False, normalize_embeddings=True)
    train_emb = all_emb[:len(train_texts)]
    test_emb = all_emb[len(train_texts):]

    # Combined features
    X_train = hstack([X_train_tfidf, csr_matrix(train_emb)])
    X_test = hstack([X_test_tfidf, csr_matrix(test_emb)])

    # Train logistic regression
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                             random_state=SEED)
    clf.fit(X_train, train_labels)

    # Evaluate routing accuracy
    preds = clf.predict(X_test)
    overall_acc = sum(p == t for p, t in zip(preds, test_labels)) / len(test_labels) * 100

    per_domain_acc = {}
    for domain in DOMAINS:
        d_preds = [p for p, t in zip(preds, test_labels) if t == domain]
        d_true = [t for t in test_labels if t == domain]
        if d_true:
            per_domain_acc[domain] = sum(p == t for p, t in zip(d_preds, d_true)) / len(d_true) * 100

    elapsed = time.time() - t0
    log(f"  Routing accuracy: {overall_acc:.1f}% ({elapsed:.1f}s)")
    for d, a in per_domain_acc.items():
        log(f"    {d}: {a:.1f}%")

    return {
        "tfidf_vec": tfidf_vec,
        "embed_model": embed_model,
        "clf": clf,
        "accuracy_pct": round(overall_acc, 1),
        "per_domain_acc": {d: round(a, 1) for d, a in per_domain_acc.items()},
        "time_s": round(elapsed, 1),
    }


def route_query(router, query):
    """Route a single query using the combined logistic router."""
    from scipy.sparse import csr_matrix, hstack

    tfidf_feat = router["tfidf_vec"].transform([query])
    embed_feat = router["embed_model"].encode([query], normalize_embeddings=True)
    combined = hstack([tfidf_feat, csr_matrix(embed_feat)])
    return router["clf"].predict(combined)[0]


def route_batch(router, queries):
    """Route a batch of queries (much faster than one-by-one)."""
    from scipy.sparse import csr_matrix, hstack

    tfidf_feat = router["tfidf_vec"].transform(queries)
    embed_feat = router["embed_model"].encode(queries, batch_size=64,
                                               show_progress_bar=False,
                                               normalize_embeddings=True)
    combined = hstack([tfidf_feat, csr_matrix(embed_feat)])
    return router["clf"].predict(combined).tolist()


# ─────────────────────────────────────────────
# Phase 2: Benchmark evaluation functions
# ─────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, dataset, label=""):
    """Evaluate GSM8K accuracy on pre-loaded dataset."""
    from mlx_lm import generate

    log_memory(f"gsm8k-{label}")
    correct = 0

    for i, ex in enumerate(dataset):
        messages = [{"role": "user", "content": f"Solve step by step.\n\n{ex['question']}"}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = generate(model, tokenizer, prompt=formatted,
                           max_tokens=512, verbose=False)

        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt_ans = gt_match.group(1).replace(",", "").strip()

        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match:
            pred_ans = pred_match.group(1).replace(",", "").strip()
            if pred_ans == gt_ans:
                correct += 1
        else:
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt_ans:
                correct += 1

        if (i + 1) % 25 == 0:
            log(f"  GSM8K {label}: {i+1}/{len(dataset)}, acc={correct/(i+1)*100:.1f}%")

    acc = correct / len(dataset) * 100
    log(f"  GSM8K {label}: {correct}/{len(dataset)} = {acc:.1f}%")
    return acc


def eval_humaneval(model, tokenizer, dataset, label=""):
    """Evaluate HumanEval pass@1 on pre-loaded dataset."""
    from mlx_lm import generate

    log_memory(f"humaneval-{label}")
    passed = 0

    for i, ex in enumerate(dataset):
        messages = [{"role": "user", "content": (
            f"Complete the following Python function. "
            f"Respond with ONLY the function body code, no explanation.\n\n"
            f"```python\n{ex['prompt']}\n```"
        )}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = generate(model, tokenizer, prompt=formatted,
                           max_tokens=512, verbose=False)

        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        completion = code_match.group(1) if code_match else response

        full_code = ex["prompt"] + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"

        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                timeout=10, capture_output=True, text=True,
            )
            if result.returncode == 0:
                passed += 1
        except (subprocess.TimeoutExpired, Exception):
            pass

        if (i + 1) % 25 == 0:
            log(f"  HumanEval {label}: {i+1}/{len(dataset)}, pass@1={passed/(i+1)*100:.1f}%")

    acc = passed / len(dataset) * 100
    log(f"  HumanEval {label}: {passed}/{len(dataset)} = {acc:.1f}%")
    return acc


def eval_medmcqa(model, tokenizer, dataset, label=""):
    """Evaluate MedMCQA accuracy on pre-loaded dataset."""
    from mlx_lm import generate

    log_memory(f"medmcqa-{label}")
    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct = 0

    for i, ex in enumerate(dataset):
        question = (
            f"{ex['question']}\n"
            f"(A) {ex['opa']}\n(B) {ex['opb']}\n(C) {ex['opc']}\n(D) {ex['opd']}"
        )
        messages = [{"role": "user", "content":
            f"Answer this medical question. Reply with only the letter.\n\n{question}"}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = generate(model, tokenizer, prompt=formatted,
                           max_tokens=20, verbose=False)

        gt = option_map.get(ex["cop"], "A")
        pred = response.strip().upper()
        pred_letter = None
        for letter in ["A", "B", "C", "D"]:
            if pred.startswith(letter):
                pred_letter = letter
                break
        if pred_letter is None:
            m = re.search(r"\b([ABCD])\b", pred)
            if m:
                pred_letter = m.group(1)

        if pred_letter == gt:
            correct += 1

        if (i + 1) % 25 == 0:
            log(f"  MedMCQA {label}: {i+1}/{len(dataset)}, acc={correct/(i+1)*100:.1f}%")

    acc = correct / len(dataset) * 100
    log(f"  MedMCQA {label}: {correct}/{len(dataset)} = {acc:.1f}%")
    return acc


# ─────────────────────────────────────────────
# Phase 3: Routed evaluation
# ─────────────────────────────────────────────

def eval_routed(router, benchmark_name, dataset, oracle_domain):
    """Evaluate benchmark with router-selected adapters.

    Groups queries by routed domain, loads each adapter once, evaluates.
    """
    from mlx_lm import generate, load

    # Get routing queries from dataset
    if benchmark_name == "gsm8k":
        queries = [ex["question"] for ex in dataset]
    elif benchmark_name == "humaneval":
        queries = [ex["prompt"] for ex in dataset]
    elif benchmark_name == "medmcqa":
        queries = [f"{ex['question']} {ex['opa']} {ex['opb']} {ex['opc']} {ex['opd']}"
                   for ex in dataset]

    # Route all queries (batch for speed)
    t_route = time.time()
    routed_domains = route_batch(router, queries)
    route_time = time.time() - t_route

    # Routing stats
    route_counts = {d: sum(1 for r in routed_domains if r == d) for d in DOMAINS}
    routing_accuracy = sum(d == oracle_domain for d in routed_domains) / len(routed_domains) * 100

    log(f"  {benchmark_name} routing: {routing_accuracy:.1f}% correct "
        f"(oracle={oracle_domain}), {route_time*1000:.0f}ms")
    for d, c in route_counts.items():
        log(f"    routed to {d}: {c}/{len(queries)}")

    # Group queries by routed domain
    domain_groups = {d: [] for d in DOMAINS}
    for i, d in enumerate(routed_domains):
        domain_groups[d].append(i)

    # Evaluate: load each adapter once, run its queries
    routed_correct = 0
    total_queries = 0
    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    for domain in DOMAINS:
        indices = domain_groups[domain]
        if not indices:
            continue

        adapter_path = ADAPTER_DIR / domain
        if adapter_path.exists():
            model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
        else:
            log(f"  WARNING: adapter {domain} not found, using base")
            model, tokenizer = load(MODEL_ID)

        log(f"  Evaluating {len(indices)} queries with {domain} adapter...")

        for idx in indices:
            ex = dataset[idx]

            if benchmark_name == "gsm8k":
                messages = [{"role": "user", "content": f"Solve step by step.\n\n{ex['question']}"}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                response = generate(model, tokenizer, prompt=formatted,
                                   max_tokens=512, verbose=False)

                gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
                if not gt_match:
                    total_queries += 1
                    continue
                gt_ans = gt_match.group(1).replace(",", "").strip()

                pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
                if pred_match:
                    if pred_match.group(1).replace(",", "").strip() == gt_ans:
                        routed_correct += 1
                else:
                    nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
                    if nums and nums[-1] == gt_ans:
                        routed_correct += 1

            elif benchmark_name == "humaneval":
                messages = [{"role": "user", "content": (
                    f"Complete the following Python function. "
                    f"Respond with ONLY the function body code, no explanation.\n\n"
                    f"```python\n{ex['prompt']}\n```"
                )}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                response = generate(model, tokenizer, prompt=formatted,
                                   max_tokens=512, verbose=False)

                code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
                completion = code_match.group(1) if code_match else response
                full_code = ex["prompt"] + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"

                try:
                    result = subprocess.run(
                        [sys.executable, "-c", full_code],
                        timeout=10, capture_output=True, text=True)
                    if result.returncode == 0:
                        routed_correct += 1
                except (subprocess.TimeoutExpired, Exception):
                    pass

            elif benchmark_name == "medmcqa":
                question = (
                    f"{ex['question']}\n"
                    f"(A) {ex['opa']}\n(B) {ex['opb']}\n(C) {ex['opc']}\n(D) {ex['opd']}")
                messages = [{"role": "user", "content":
                    f"Answer this medical question. Reply with only the letter.\n\n{question}"}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                response = generate(model, tokenizer, prompt=formatted,
                                   max_tokens=20, verbose=False)

                gt = option_map.get(ex["cop"], "A")
                pred = response.strip().upper()
                pred_letter = None
                for letter in ["A", "B", "C", "D"]:
                    if pred.startswith(letter):
                        pred_letter = letter
                        break
                if pred_letter is None:
                    m = re.search(r"\b([ABCD])\b", pred)
                    if m:
                        pred_letter = m.group(1)
                if pred_letter == gt:
                    routed_correct += 1

            total_queries += 1

        cleanup(model, tokenizer)

    routed_acc = routed_correct / total_queries * 100 if total_queries > 0 else 0.0
    log(f"  {benchmark_name} routed: {routed_correct}/{total_queries} = {routed_acc:.1f}%")

    return {
        "accuracy_pct": round(routed_acc, 1),
        "routing_accuracy_pct": round(routing_accuracy, 1),
        "route_counts": route_counts,
        "route_time_ms": round(route_time * 1000, 1),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    from mlx_lm import load

    t_start = time.time()
    log("=" * 60)
    log("P0.E2E: Combined Logistic Routing -> Adapter Generation")
    log(f"SMOKE={IS_SMOKE}, N_EVAL={N_EVAL}")
    log(f"Adapters from: {ADAPTER_DIR}")
    log("=" * 60)

    # Verify adapters exist
    for domain in DOMAINS:
        p = ADAPTER_DIR / domain / "adapters.safetensors"
        if not p.exists():
            log(f"FATAL: Missing adapter: {p}")
            sys.exit(1)
    log("  All 3 adapters verified")

    # ── Load datasets once ──
    log("\n[Data] Loading evaluation datasets...")
    gsm8k_data = load_gsm8k_test(N_EVAL)
    humaneval_data = load_humaneval_test(N_EVAL)
    medmcqa_data = load_medmcqa_val(N_EVAL)
    log(f"  GSM8K: {len(gsm8k_data)}, HumanEval: {len(humaneval_data)}, MedMCQA: {len(medmcqa_data)}")

    # ── Phase 1: Train router ──
    router_result = phase_train_router()

    # ── Phase 2: Base model evaluation ──
    log("\n" + "=" * 60)
    log("Phase 2: Base model (no adapter)")
    log("=" * 60)

    model, tokenizer = load(MODEL_ID)
    base_gsm8k = eval_gsm8k(model, tokenizer, gsm8k_data, "base")
    base_humaneval = eval_humaneval(model, tokenizer, humaneval_data, "base")
    base_medmcqa = eval_medmcqa(model, tokenizer, medmcqa_data, "base")
    cleanup(model, tokenizer)

    log(f"\nBase: GSM8K={base_gsm8k:.1f}%, HumanEval={base_humaneval:.1f}%, MedMCQA={base_medmcqa:.1f}%")

    # ── Phase 3: Oracle evaluation (correct adapter always) ──
    log("\n" + "=" * 60)
    log("Phase 3: Oracle routing (always correct adapter)")
    log("=" * 60)

    model, tokenizer = load(MODEL_ID, adapter_path=str(ADAPTER_DIR / "math"))
    oracle_gsm8k = eval_gsm8k(model, tokenizer, gsm8k_data, "oracle-math")
    cleanup(model, tokenizer)

    model, tokenizer = load(MODEL_ID, adapter_path=str(ADAPTER_DIR / "code"))
    oracle_humaneval = eval_humaneval(model, tokenizer, humaneval_data, "oracle-code")
    cleanup(model, tokenizer)

    model, tokenizer = load(MODEL_ID, adapter_path=str(ADAPTER_DIR / "medical"))
    oracle_medmcqa = eval_medmcqa(model, tokenizer, medmcqa_data, "oracle-medical")
    cleanup(model, tokenizer)

    log(f"\nOracle: GSM8K={oracle_gsm8k:.1f}%, HumanEval={oracle_humaneval:.1f}%, MedMCQA={oracle_medmcqa:.1f}%")

    # ── Phase 4: Routed evaluation ──
    log("\n" + "=" * 60)
    log("Phase 4: Combined logistic routed evaluation")
    log("=" * 60)

    routed_gsm8k = eval_routed(router_result, "gsm8k", gsm8k_data, "math")
    routed_humaneval = eval_routed(router_result, "humaneval", humaneval_data, "code")
    routed_medmcqa = eval_routed(router_result, "medmcqa", medmcqa_data, "medical")

    log(f"\nRouted: GSM8K={routed_gsm8k['accuracy_pct']:.1f}%, "
        f"HumanEval={routed_humaneval['accuracy_pct']:.1f}%, "
        f"MedMCQA={routed_medmcqa['accuracy_pct']:.1f}%")

    # ── Phase 5: Compute losses and kill criteria ──
    gsm8k_loss = oracle_gsm8k - routed_gsm8k["accuracy_pct"]
    humaneval_loss = oracle_humaneval - routed_humaneval["accuracy_pct"]
    medmcqa_loss = oracle_medmcqa - routed_medmcqa["accuracy_pct"]
    max_loss = max(gsm8k_loss, humaneval_loss, medmcqa_loss)

    total_time = time.time() - t_start

    results = {
        "experiment": "exp_p0_e2e_combined_routing_n10",
        "smoke": IS_SMOKE,
        "n_eval": N_EVAL,

        # Base
        "base_gsm8k_pct": round(base_gsm8k, 1),
        "base_humaneval_pct": round(base_humaneval, 1),
        "base_medmcqa_pct": round(base_medmcqa, 1),

        # Oracle
        "oracle_gsm8k_pct": round(oracle_gsm8k, 1),
        "oracle_humaneval_pct": round(oracle_humaneval, 1),
        "oracle_medmcqa_pct": round(oracle_medmcqa, 1),

        # Routed
        "routed_gsm8k_pct": routed_gsm8k["accuracy_pct"],
        "routed_humaneval_pct": routed_humaneval["accuracy_pct"],
        "routed_medmcqa_pct": routed_medmcqa["accuracy_pct"],

        # Routing details
        "routed_gsm8k_detail": routed_gsm8k,
        "routed_humaneval_detail": routed_humaneval,
        "routed_medmcqa_detail": routed_medmcqa,

        # Routing accuracy
        "router_overall_accuracy_pct": router_result["accuracy_pct"],
        "router_per_domain_accuracy": router_result["per_domain_acc"],
        "router_train_time_s": router_result["time_s"],

        # Quality losses
        "gsm8k_routing_loss_pp": round(gsm8k_loss, 1),
        "humaneval_routing_loss_pp": round(humaneval_loss, 1),
        "medmcqa_routing_loss_pp": round(medmcqa_loss, 1),
        "max_routing_loss_pp": round(max_loss, 1),

        # Adapter deltas
        "gsm8k_delta_pp": round(oracle_gsm8k - base_gsm8k, 1),
        "humaneval_delta_pp": round(oracle_humaneval - base_humaneval, 1),
        "medmcqa_delta_pp": round(oracle_medmcqa - base_medmcqa, 1),

        # Kill criteria
        "K1478_gsm8k": "PASS" if routed_gsm8k["accuracy_pct"] >= 65 else "FAIL",
        "K1479_humaneval": "PASS" if routed_humaneval["accuracy_pct"] >= 50 else "FAIL",
        "K1480_medmcqa": "PASS" if routed_medmcqa["accuracy_pct"] >= 40 else "FAIL",
        "K1481_routing_loss": "PASS" if max_loss <= 5 else "FAIL",

        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # Summary
    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    log(f"{'Benchmark':<12} {'Base':>8} {'Oracle':>8} {'Routed':>8} {'Loss':>8} {'Delta':>8}")
    log(f"{'GSM8K':<12} {base_gsm8k:>7.1f}% {oracle_gsm8k:>7.1f}% {routed_gsm8k['accuracy_pct']:>7.1f}% {gsm8k_loss:>7.1f}pp {oracle_gsm8k-base_gsm8k:>+7.1f}pp")
    log(f"{'HumanEval':<12} {base_humaneval:>7.1f}% {oracle_humaneval:>7.1f}% {routed_humaneval['accuracy_pct']:>7.1f}% {humaneval_loss:>7.1f}pp {oracle_humaneval-base_humaneval:>+7.1f}pp")
    log(f"{'MedMCQA':<12} {base_medmcqa:>7.1f}% {oracle_medmcqa:>7.1f}% {routed_medmcqa['accuracy_pct']:>7.1f}% {medmcqa_loss:>7.1f}pp {oracle_medmcqa-base_medmcqa:>+7.1f}pp")

    log("\nKILL CRITERIA:")
    log(f"  K1478 GSM8K routed >= 65%:      {results['K1478_gsm8k']} ({routed_gsm8k['accuracy_pct']:.1f}%)")
    log(f"  K1479 HumanEval routed >= 50%:   {results['K1479_humaneval']} ({routed_humaneval['accuracy_pct']:.1f}%)")
    log(f"  K1480 MedMCQA routed >= 40%:     {results['K1480_medmcqa']} ({routed_medmcqa['accuracy_pct']:.1f}%)")
    log(f"  K1481 Max routing loss <= 5pp:   {results['K1481_routing_loss']} ({max_loss:.1f}pp)")
    log(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
