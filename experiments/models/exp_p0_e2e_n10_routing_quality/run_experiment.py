#!/usr/bin/env python3
"""
P0.E2E: N=10 Routing Quality Loss — Imperfect Routing to Generation

Tests quality loss when combined logistic routing at N=10 (~90% accuracy)
drives adapter selection. Misrouted queries fall back to base model.

Kill criteria:
  K1482: GSM8K via 10-way routed adapter >= 70%
  K1483: HumanEval via 10-way routed adapter >= 48%
  K1484: MedMCQA via 10-way routed adapter >= 45%
  K1485: Max quality loss from routing <= 8pp vs oracle

Reuses trained adapters from exp_p0_e2e_benchmark (Finding #508).
Router trained on 10 MMLU domains (7 distractors without adapters).
"""

import gc
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import OrderedDict
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
ADAPTER_DOMAINS = ["math", "code", "medical"]

# 10 domains: 3 with adapters + 7 MMLU distractors (from Finding #525)
MMLU_DOMAINS = OrderedDict([
    ("science", ["astronomy", "college_biology", "college_chemistry", "college_physics"]),
    ("legal", ["professional_law", "jurisprudence", "international_law"]),
    ("finance", ["professional_accounting", "econometrics", "marketing"]),
    ("history", ["high_school_us_history", "high_school_world_history", "prehistory"]),
    ("psychology", ["professional_psychology", "high_school_psychology"]),
    ("philosophy", ["philosophy", "formal_logic", "logical_fallacies"]),
    ("engineering", ["electrical_engineering", "computer_security", "college_computer_science"]),
])
ALL_DOMAINS = ADAPTER_DOMAINS + list(MMLU_DOMAINS.keys())

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
# Data loading
# ─────────────────────────────────────────────

def load_gsm8k_test(n, seed=SEED):
    path = hf_hub_download("openai/gsm8k",
        "main/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    return [{"question": r["question"], "answer": r["answer"]} for _, r in df.iterrows()]


def load_humaneval_test(n):
    path = hf_hub_download("openai_humaneval",
        "openai_humaneval/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.head(min(n, len(df)))
    return [{"prompt": r["prompt"], "test": r["test"],
             "entry_point": r["entry_point"]} for _, r in df.iterrows()]


def load_medmcqa_val(n, seed=SEED):
    path = hf_hub_download("openlifescienceai/medmcqa",
        "data/validation-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    return [{"question": r["question"], "opa": r["opa"], "opb": r["opb"],
             "opc": r["opc"], "opd": r["opd"], "cop": r["cop"]} for _, r in df.iterrows()]


# ─────────────────────────────────────────────
# Phase 1: Train combined logistic router on 10 domains
# ─────────────────────────────────────────────

def load_routing_texts(n_per_domain):
    """Load routing text samples for all 10 domains."""
    texts = {}
    rng = random.Random(SEED + 1)
    np_rng = np.random.RandomState(SEED + 1)

    # Math: GSM8K train
    path = hf_hub_download("openai/gsm8k",
        "main/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    idx = np_rng.choice(len(df), min(n_per_domain, len(df)), replace=False)
    texts["math"] = df.iloc[idx]["question"].tolist()

    # Code: CodeAlpaca
    path = hf_hub_download("sahil2801/CodeAlpaca-20k",
        "code_alpaca_20k.json", repo_type="dataset")
    with open(path) as f:
        data = json.load(f)
    rng.shuffle(data)
    texts["code"] = [ex["instruction"] for ex in data[:n_per_domain]]

    # Medical: MedMCQA train
    path = hf_hub_download("openlifescienceai/medmcqa",
        "data/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    idx = np_rng.choice(len(df), min(n_per_domain, len(df)), replace=False)
    texts["medical"] = df.iloc[idx]["question"].tolist()

    # MMLU domains
    for domain, subjects in MMLU_DOMAINS.items():
        domain_texts = []
        for subject in subjects:
            for split in ["test", "validation", "dev"]:
                try:
                    path = hf_hub_download(
                        "cais/mmlu",
                        f"{subject}/{split}-00000-of-00001.parquet",
                        repo_type="dataset")
                    df = pd.read_parquet(path)
                    domain_texts.extend(df["question"].tolist())
                except Exception:
                    continue
        rng.shuffle(domain_texts)
        texts[domain] = domain_texts[:n_per_domain]
        if len(texts[domain]) < 50:
            log(f"  WARNING: {domain} only has {len(texts[domain])} texts")

    return texts


def phase_train_router():
    """Train combined TF-IDF + embedding logistic router for 10 domains."""
    from scipy.sparse import csr_matrix, hstack
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    log("\n[Phase 1] Training combined logistic router on 10 domains")
    t0 = time.time()

    n_total = N_ROUTE_TRAIN + N_ROUTE_TEST
    texts = load_routing_texts(n_total)

    # Split train/test
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    for domain in ALL_DOMAINS:
        txts = texts.get(domain, [])
        n_tr = min(N_ROUTE_TRAIN, len(txts))
        n_te = min(N_ROUTE_TEST, len(txts) - n_tr)
        train_texts.extend(txts[:n_tr])
        train_labels.extend([domain] * n_tr)
        test_texts.extend(txts[n_tr:n_tr + n_te])
        test_labels.extend([domain] * n_te)

    log(f"  Router: {len(train_texts)} train, {len(test_texts)} test, {len(ALL_DOMAINS)} domains")

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
    for domain in ALL_DOMAINS:
        d_preds = [p for p, t in zip(preds, test_labels) if t == domain]
        d_true = [t for t in test_labels if t == domain]
        if d_true:
            per_domain_acc[domain] = sum(p == t for p, t in zip(d_preds, d_true)) / len(d_true) * 100

    # Adapter-domain routing accuracy (the 3 domains we care about)
    adapter_acc = {d: per_domain_acc.get(d, 0.0) for d in ADAPTER_DOMAINS}

    elapsed = time.time() - t0
    log(f"  Overall routing accuracy: {overall_acc:.1f}% ({elapsed:.1f}s)")
    log(f"  Adapter domain routing:")
    for d in ADAPTER_DOMAINS:
        log(f"    {d}: {adapter_acc[d]:.1f}%")
    log(f"  Distractor domain routing:")
    for d in MMLU_DOMAINS:
        log(f"    {d}: {per_domain_acc.get(d, 0.0):.1f}%")

    return {
        "tfidf_vec": tfidf_vec,
        "embed_model": embed_model,
        "clf": clf,
        "overall_accuracy_pct": round(overall_acc, 1),
        "per_domain_acc": {d: round(a, 1) for d, a in per_domain_acc.items()},
        "adapter_domain_acc": {d: round(a, 1) for d, a in adapter_acc.items()},
        "time_s": round(elapsed, 1),
    }


def route_batch(router, queries):
    """Route a batch of queries through 10-way classifier."""
    from scipy.sparse import csr_matrix, hstack

    tfidf_feat = router["tfidf_vec"].transform(queries)
    embed_feat = router["embed_model"].encode(queries, batch_size=64,
                                               show_progress_bar=False,
                                               normalize_embeddings=True)
    combined = hstack([tfidf_feat, csr_matrix(embed_feat)])
    return router["clf"].predict(combined).tolist()


# ─────────────────────────────────────────────
# Phase 2-3: Benchmark evaluation
# ─────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, dataset, label=""):
    from mlx_lm import generate
    log_memory(f"gsm8k-{label}")
    correct = 0
    for i, ex in enumerate(dataset):
        messages = [{"role": "user", "content": f"Solve step by step.\n\n{ex['question']}"}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted,
                           max_tokens=512, verbose=False)
        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt_ans = gt_match.group(1).replace(",", "").strip()
        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match:
            if pred_match.group(1).replace(",", "").strip() == gt_ans:
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
                passed += 1
        except (subprocess.TimeoutExpired, Exception):
            pass
        if (i + 1) % 25 == 0:
            log(f"  HumanEval {label}: {i+1}/{len(dataset)}, pass@1={passed/(i+1)*100:.1f}%")
    acc = passed / len(dataset) * 100
    log(f"  HumanEval {label}: {passed}/{len(dataset)} = {acc:.1f}%")
    return acc


def eval_medmcqa(model, tokenizer, dataset, label=""):
    from mlx_lm import generate
    log_memory(f"medmcqa-{label}")
    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct = 0
    for i, ex in enumerate(dataset):
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
            correct += 1
        if (i + 1) % 25 == 0:
            log(f"  MedMCQA {label}: {i+1}/{len(dataset)}, acc={correct/(i+1)*100:.1f}%")
    acc = correct / len(dataset) * 100
    log(f"  MedMCQA {label}: {correct}/{len(dataset)} = {acc:.1f}%")
    return acc


# ─────────────────────────────────────────────
# Phase 4: Routed evaluation with base model fallback
# ─────────────────────────────────────────────

def eval_routed(router, benchmark_name, dataset, oracle_domain):
    """Evaluate with 10-way router. Adapter domains get adapter, others get base."""
    from mlx_lm import generate, load

    # Extract routing queries
    if benchmark_name == "gsm8k":
        queries = [ex["question"] for ex in dataset]
    elif benchmark_name == "humaneval":
        queries = [ex["prompt"] for ex in dataset]
    elif benchmark_name == "medmcqa":
        queries = [f"{ex['question']} {ex['opa']} {ex['opb']} {ex['opc']} {ex['opd']}"
                   for ex in dataset]

    # Route all queries through 10-way classifier
    t_route = time.time()
    routed_domains = route_batch(router, queries)
    route_time = time.time() - t_route

    # Routing stats
    route_counts = {}
    for d in ALL_DOMAINS:
        c = sum(1 for r in routed_domains if r == d)
        if c > 0:
            route_counts[d] = c
    routing_accuracy = sum(d == oracle_domain for d in routed_domains) / len(routed_domains) * 100

    log(f"\n  {benchmark_name} routing: {routing_accuracy:.1f}% correct "
        f"(oracle={oracle_domain}), {route_time*1000:.0f}ms")
    for d, c in sorted(route_counts.items(), key=lambda x: -x[1]):
        marker = " <-- CORRECT" if d == oracle_domain else (
            " (has adapter)" if d in ADAPTER_DOMAINS else " (base fallback)")
        log(f"    routed to {d}: {c}/{len(queries)}{marker}")

    # Group queries by routing decision
    # Key insight: queries routed to adapter domains get that adapter,
    # queries routed to non-adapter domains get base model
    adapter_groups = {d: [] for d in ADAPTER_DOMAINS}
    base_group = []  # queries routed to non-adapter domains

    for i, d in enumerate(routed_domains):
        if d in ADAPTER_DOMAINS:
            adapter_groups[d].append(i)
        else:
            base_group.append(i)

    log(f"  Adapter-routed: {sum(len(v) for v in adapter_groups.values())}, "
        f"Base-fallback: {len(base_group)}")

    # Evaluation functions per benchmark
    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    def eval_single(model, tokenizer, idx):
        ex = dataset[idx]
        if benchmark_name == "gsm8k":
            messages = [{"role": "user", "content": f"Solve step by step.\n\n{ex['question']}"}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            response = generate(model, tokenizer, prompt=formatted,
                               max_tokens=512, verbose=False)
            gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
            if not gt_match:
                return False
            gt_ans = gt_match.group(1).replace(",", "").strip()
            pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
            if pred_match:
                return pred_match.group(1).replace(",", "").strip() == gt_ans
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            return bool(nums) and nums[-1] == gt_ans

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
                return result.returncode == 0
            except (subprocess.TimeoutExpired, Exception):
                return False

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
            return pred_letter == gt

    routed_correct = 0
    total_queries = 0

    # Evaluate queries routed to adapter domains
    for domain in ADAPTER_DOMAINS:
        indices = adapter_groups[domain]
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
            if eval_single(model, tokenizer, idx):
                routed_correct += 1
            total_queries += 1
        cleanup(model, tokenizer)

    # Evaluate queries routed to non-adapter domains (base model fallback)
    if base_group:
        log(f"  Evaluating {len(base_group)} queries with base model (fallback)...")
        model, tokenizer = load(MODEL_ID)
        for idx in base_group:
            if eval_single(model, tokenizer, idx):
                routed_correct += 1
            total_queries += 1
        cleanup(model, tokenizer)

    routed_acc = routed_correct / total_queries * 100 if total_queries > 0 else 0.0
    log(f"  {benchmark_name} routed: {routed_correct}/{total_queries} = {routed_acc:.1f}%")

    return {
        "accuracy_pct": round(routed_acc, 1),
        "routing_accuracy_pct": round(routing_accuracy, 1),
        "route_counts": route_counts,
        "base_fallback_count": len(base_group),
        "route_time_ms": round(route_time * 1000, 1),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    from mlx_lm import load

    t_start = time.time()
    log("=" * 60)
    log("P0.E2E: N=10 Routing Quality Loss")
    log(f"SMOKE={IS_SMOKE}, N_EVAL={N_EVAL}")
    log(f"Adapters from: {ADAPTER_DIR}")
    log(f"Domains: {len(ALL_DOMAINS)} ({len(ADAPTER_DOMAINS)} with adapters)")
    log("=" * 60)

    # Verify adapters exist
    for domain in ADAPTER_DOMAINS:
        p = ADAPTER_DIR / domain / "adapters.safetensors"
        if not p.exists():
            log(f"FATAL: Missing adapter: {p}")
            sys.exit(1)
    log("  All 3 adapters verified")

    # Load datasets once
    log("\n[Data] Loading evaluation datasets...")
    gsm8k_data = load_gsm8k_test(N_EVAL)
    humaneval_data = load_humaneval_test(N_EVAL)
    medmcqa_data = load_medmcqa_val(N_EVAL)
    log(f"  GSM8K: {len(gsm8k_data)}, HumanEval: {len(humaneval_data)}, MedMCQA: {len(medmcqa_data)}")

    # Phase 1: Train 10-domain router
    router_result = phase_train_router()

    # Phase 2: Base model evaluation
    log("\n" + "=" * 60)
    log("Phase 2: Base model (no adapter)")
    log("=" * 60)
    model, tokenizer = load(MODEL_ID)
    base_gsm8k = eval_gsm8k(model, tokenizer, gsm8k_data, "base")
    base_humaneval = eval_humaneval(model, tokenizer, humaneval_data, "base")
    base_medmcqa = eval_medmcqa(model, tokenizer, medmcqa_data, "base")
    cleanup(model, tokenizer)
    log(f"\nBase: GSM8K={base_gsm8k:.1f}%, HumanEval={base_humaneval:.1f}%, MedMCQA={base_medmcqa:.1f}%")

    # Phase 3: Oracle evaluation
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

    # Phase 4: 10-way routed evaluation
    log("\n" + "=" * 60)
    log("Phase 4: 10-way combined logistic routed evaluation")
    log("=" * 60)
    routed_gsm8k = eval_routed(router_result, "gsm8k", gsm8k_data, "math")
    routed_humaneval = eval_routed(router_result, "humaneval", humaneval_data, "code")
    routed_medmcqa = eval_routed(router_result, "medmcqa", medmcqa_data, "medical")

    log(f"\nRouted: GSM8K={routed_gsm8k['accuracy_pct']:.1f}%, "
        f"HumanEval={routed_humaneval['accuracy_pct']:.1f}%, "
        f"MedMCQA={routed_medmcqa['accuracy_pct']:.1f}%")

    # Phase 5: Compute losses and kill criteria
    gsm8k_loss = oracle_gsm8k - routed_gsm8k["accuracy_pct"]
    humaneval_loss = oracle_humaneval - routed_humaneval["accuracy_pct"]
    medmcqa_loss = oracle_medmcqa - routed_medmcqa["accuracy_pct"]
    max_loss = max(gsm8k_loss, humaneval_loss, medmcqa_loss)

    # Theorem 1 predictions for comparison
    predicted_gsm8k = 1.00 * oracle_gsm8k + 0.00 * base_gsm8k
    predicted_humaneval = 0.947 * oracle_humaneval + 0.053 * base_humaneval
    predicted_medmcqa = 0.86 * oracle_medmcqa + 0.14 * base_medmcqa

    total_time = time.time() - t_start

    results = {
        "experiment": "exp_p0_e2e_n10_routing_quality",
        "smoke": IS_SMOKE,
        "n_eval": N_EVAL,
        "n_domains": len(ALL_DOMAINS),
        "n_adapter_domains": len(ADAPTER_DOMAINS),

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

        # Router stats
        "router_overall_accuracy_pct": router_result["overall_accuracy_pct"],
        "router_per_domain_accuracy": router_result["per_domain_acc"],
        "router_adapter_domain_acc": router_result["adapter_domain_acc"],
        "router_train_time_s": router_result["time_s"],

        # Quality losses (oracle - routed)
        "gsm8k_routing_loss_pp": round(gsm8k_loss, 1),
        "humaneval_routing_loss_pp": round(humaneval_loss, 1),
        "medmcqa_routing_loss_pp": round(medmcqa_loss, 1),
        "max_routing_loss_pp": round(max_loss, 1),

        # Theorem 1 predictions
        "predicted_gsm8k_pct": round(predicted_gsm8k, 1),
        "predicted_humaneval_pct": round(predicted_humaneval, 1),
        "predicted_medmcqa_pct": round(predicted_medmcqa, 1),

        # Adapter deltas
        "gsm8k_delta_pp": round(oracle_gsm8k - base_gsm8k, 1),
        "humaneval_delta_pp": round(oracle_humaneval - base_humaneval, 1),
        "medmcqa_delta_pp": round(oracle_medmcqa - base_medmcqa, 1),

        # Kill criteria
        "K1482_gsm8k": "PASS" if routed_gsm8k["accuracy_pct"] >= 70 else "FAIL",
        "K1483_humaneval": "PASS" if routed_humaneval["accuracy_pct"] >= 48 else "FAIL",
        "K1484_medmcqa": "PASS" if routed_medmcqa["accuracy_pct"] >= 45 else "FAIL",
        "K1485_max_loss": "PASS" if max_loss <= 8 else "FAIL",

        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # Summary
    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    log(f"{'Benchmark':<12} {'Base':>8} {'Oracle':>8} {'Predict':>8} {'Routed':>8} {'Loss':>8}")
    log(f"{'GSM8K':<12} {base_gsm8k:>7.1f}% {oracle_gsm8k:>7.1f}% {predicted_gsm8k:>7.1f}% {routed_gsm8k['accuracy_pct']:>7.1f}% {gsm8k_loss:>7.1f}pp")
    log(f"{'HumanEval':<12} {base_humaneval:>7.1f}% {oracle_humaneval:>7.1f}% {predicted_humaneval:>7.1f}% {routed_humaneval['accuracy_pct']:>7.1f}% {humaneval_loss:>7.1f}pp")
    log(f"{'MedMCQA':<12} {base_medmcqa:>7.1f}% {oracle_medmcqa:>7.1f}% {predicted_medmcqa:>7.1f}% {routed_medmcqa['accuracy_pct']:>7.1f}% {medmcqa_loss:>7.1f}pp")

    log(f"\nRouting accuracy per benchmark:")
    log(f"  GSM8K→math: {routed_gsm8k['routing_accuracy_pct']:.1f}% "
        f"(base fallback: {routed_gsm8k['base_fallback_count']})")
    log(f"  HumanEval→code: {routed_humaneval['routing_accuracy_pct']:.1f}% "
        f"(base fallback: {routed_humaneval['base_fallback_count']})")
    log(f"  MedMCQA→medical: {routed_medmcqa['routing_accuracy_pct']:.1f}% "
        f"(base fallback: {routed_medmcqa['base_fallback_count']})")

    log("\nKILL CRITERIA:")
    log(f"  K1482 GSM8K routed >= 70%:      {results['K1482_gsm8k']} ({routed_gsm8k['accuracy_pct']:.1f}%)")
    log(f"  K1483 HumanEval routed >= 48%:   {results['K1483_humaneval']} ({routed_humaneval['accuracy_pct']:.1f}%)")
    log(f"  K1484 MedMCQA routed >= 45%:     {results['K1484_medmcqa']} ({routed_medmcqa['accuracy_pct']:.1f}%)")
    log(f"  K1485 Max routing loss <= 8pp:   {results['K1485_max_loss']} ({max_loss:.1f}pp)")
    log(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
