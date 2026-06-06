#!/usr/bin/env python3
"""
P0: End-to-end system benchmark — GSM8K, HumanEval, MedMCQA with routing.

Kill criteria:
  K1328: Math (GSM8K): adapted >= base + 10pp absolute accuracy
  K1329: Code (HumanEval): adapted >= base + 10pp pass@1
  K1330: Medical (MedMCQA): adapted >= base + 10pp accuracy
  K1331: TF-IDF routing correctly selects domain adapter >= 90% of the time
  K1332: E2E latency (route + generate 100 tokens) <= 2 seconds

Grounded by:
  Finding #421: q_proj achieves +22-82pp on these benchmarks
  Finding #504: v_proj+o_proj is the correct projection target
  Finding #502: TF-IDF routing 96% at N=5
  Finding #503: adapter swap 1ms, 0% overhead
  Finding #506: train-eval distribution alignment is the key
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

# Memory safety
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 50 if IS_SMOKE else 2000
N_EVAL = 5 if IS_SMOKE else 100
N_STEPS = 20 if IS_SMOKE else 1000
SEED = 42
LORA_RANK = 8
LORA_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB", flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ─────────────────────────────────────────────
# Phase 1: Dataset preparation
# ─────────────────────────────────────────────

def phase_prepare_data():
    """Download and format training data for all 3 domains."""
    from datasets import load_dataset

    data_dir = EXPERIMENT_DIR / "data"

    # GSM8K
    print("Preparing GSM8K...", flush=True)
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(N_TRAIN, len(ds))))
    math_dir = data_dir / "math"
    math_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for ex in ds:
        records.append(json.dumps({"messages": [
            {"role": "user", "content": f"Solve step by step.\n\n{ex['question']}"},
            {"role": "assistant", "content": ex["answer"]},
        ]}))
    n_val = max(1, len(records) // 10)
    (math_dir / "train.jsonl").write_text("\n".join(records[n_val:]))
    (math_dir / "valid.jsonl").write_text("\n".join(records[:n_val]))
    print(f"  GSM8K: {len(records)-n_val} train, {n_val} val", flush=True)

    # CodeAlpaca
    print("Preparing CodeAlpaca...", flush=True)
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(N_TRAIN, len(ds))))
    code_dir = data_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for ex in ds:
        content = ex["instruction"]
        if ex.get("input", ""):
            content += f"\n\nInput:\n{ex['input']}"
        records.append(json.dumps({"messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": ex["output"]},
        ]}))
    n_val = max(1, len(records) // 10)
    (code_dir / "train.jsonl").write_text("\n".join(records[n_val:]))
    (code_dir / "valid.jsonl").write_text("\n".join(records[:n_val]))
    print(f"  CodeAlpaca: {len(records)-n_val} train, {n_val} val", flush=True)

    # MedMCQA
    print("Preparing MedMCQA...", flush=True)
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(N_TRAIN, len(ds))))
    med_dir = data_dir / "medical"
    med_dir.mkdir(parents=True, exist_ok=True)
    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    records = []
    for ex in ds:
        question = (
            f"{ex['question']}\n"
            f"(A) {ex['opa']}\n(B) {ex['opb']}\n(C) {ex['opc']}\n(D) {ex['opd']}"
        )
        ans_letter = option_map.get(ex["cop"], "A")
        ans_text = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]][ex["cop"]]
        records.append(json.dumps({"messages": [
            {"role": "user", "content": f"Answer this medical question. Reply with only the letter.\n\n{question}"},
            {"role": "assistant", "content": f"{ans_letter}: {ans_text}"},
        ]}))
    n_val = max(1, len(records) // 10)
    (med_dir / "train.jsonl").write_text("\n".join(records[n_val:]))
    (med_dir / "valid.jsonl").write_text("\n".join(records[:n_val]))
    print(f"  MedMCQA: {len(records)-n_val} train, {n_val} val", flush=True)

    return {"domains": ["math", "code", "medical"]}


# ─────────────────────────────────────────────
# Phase 2: Training (subprocess per domain)
# ─────────────────────────────────────────────

def phase_train_adapter(domain: str) -> dict:
    """Train a v_proj+o_proj LoRA adapter for one domain via subprocess."""
    import yaml

    data_dir = EXPERIMENT_DIR / "data" / domain
    adapter_path = EXPERIMENT_DIR / "adapters" / domain
    adapter_path.mkdir(parents=True, exist_ok=True)
    config_path = EXPERIMENT_DIR / f"lora_config_{domain}.yaml"

    config = {
        "model": MODEL_ID,
        "train": True,
        "data": str(data_dir),
        "fine_tune_type": "lora",
        "num_layers": -1,
        "iters": N_STEPS,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "lora_parameters": {
            "rank": LORA_RANK,
            "scale": 8.0,
            "dropout": 0.0,
            "keys": LORA_KEYS,
        },
        "adapter_path": str(adapter_path),
        "save_every": N_STEPS,
        "val_batches": 5,
        "steps_per_report": max(10, N_STEPS // 10),
        "steps_per_eval": N_STEPS,
        "max_seq_length": 512,
        "mask_prompt": True,
        "grad_checkpoint": True,
        "seed": SEED,
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n=== Training {domain} adapter (rank={LORA_RANK}, {N_STEPS} steps, keys={LORA_KEYS}) ===", flush=True)
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, "-m", "mlx_lm", "lora", "-c", str(config_path)],
        capture_output=False,
        text=True,
        cwd=str(EXPERIMENT_DIR),
    )

    elapsed = time.time() - t0
    print(f"  {domain} training: {elapsed:.1f}s (exit={result.returncode})", flush=True)

    # Measure adapter size
    size_mb = 0.0
    if adapter_path.exists():
        size_mb = sum(f.stat().st_size for f in adapter_path.rglob("*") if f.is_file()) / 1e6

    return {
        "train_time_s": round(elapsed, 1),
        "exit_code": result.returncode,
        "adapter_size_mb": round(size_mb, 2),
    }


# ─────────────────────────────────────────────
# Phase 3: Benchmark evaluation
# ─────────────────────────────────────────────

def phase_eval_gsm8k(adapter_path=None) -> float:
    """Evaluate GSM8K accuracy. Returns accuracy 0-100."""
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(N_EVAL, len(ds))))

    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)

    label = f"gsm8k{'(adapter)' if adapter_path else '(base)'}"
    log_memory(f"{label}-loaded")

    correct = 0
    for i, ex in enumerate(ds):
        messages = [{"role": "user", "content": f"Solve step by step.\n\n{ex['question']}"}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=512,  # Fix #421 artifact: 256 too short for CoT
            verbose=False,
        )

        # Extract ground truth
        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt_ans = gt_match.group(1).replace(",", "").strip()

        # Extract prediction: look for #### marker first, then last number
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
            print(f"  GSM8K {label}: {i+1}/{len(ds)}, running acc={correct/(i+1)*100:.1f}%", flush=True)

    acc = correct / len(ds) * 100
    print(f"GSM8K {label}: {correct}/{len(ds)} = {acc:.1f}%", flush=True)

    cleanup(model, tokenizer)
    return acc


def phase_eval_humaneval(adapter_path=None) -> float:
    """Evaluate HumanEval pass@1. Returns accuracy 0-100."""
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("openai_humaneval", split="test")
    ds = ds.select(range(min(N_EVAL, len(ds))))

    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)

    label = f"humaneval{'(adapter)' if adapter_path else '(base)'}"
    log_memory(f"{label}-loaded")

    passed = 0
    for i, ex in enumerate(ds):
        messages = [{"role": "user", "content": (
            f"Complete the following Python function. "
            f"Respond with ONLY the function body code, no explanation.\n\n"
            f"```python\n{ex['prompt']}\n```"
        )}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=512,
            verbose=False,
        )

        # Extract code
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if code_match:
            completion = code_match.group(1)
        else:
            completion = response

        full_code = ex["prompt"] + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"

        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                timeout=10,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                passed += 1
        except (subprocess.TimeoutExpired, Exception):
            pass

        if (i + 1) % 25 == 0:
            print(f"  HumanEval {label}: {i+1}/{len(ds)}, running pass@1={passed/(i+1)*100:.1f}%", flush=True)

    acc = passed / len(ds) * 100
    print(f"HumanEval {label}: {passed}/{len(ds)} = {acc:.1f}%", flush=True)

    cleanup(model, tokenizer)
    return acc


def phase_eval_medmcqa(adapter_path=None) -> float:
    """Evaluate MedMCQA accuracy. Returns accuracy 0-100."""
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    ds = ds.shuffle(seed=SEED).select(range(min(N_EVAL, len(ds))))

    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)

    label = f"medmcqa{'(adapter)' if adapter_path else '(base)'}"
    log_memory(f"{label}-loaded")

    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct = 0

    for i, ex in enumerate(ds):
        question = (
            f"{ex['question']}\n"
            f"(A) {ex['opa']}\n(B) {ex['opb']}\n(C) {ex['opc']}\n(D) {ex['opd']}"
        )
        messages = [{"role": "user", "content": f"Answer this medical question. Reply with only the letter.\n\n{question}"}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=20,
            verbose=False,
        )

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
            print(f"  MedMCQA {label}: {i+1}/{len(ds)}, running acc={correct/(i+1)*100:.1f}%", flush=True)

    acc = correct / len(ds) * 100
    print(f"MedMCQA {label}: {correct}/{len(ds)} = {acc:.1f}%", flush=True)

    cleanup(model, tokenizer)
    return acc


# ─────────────────────────────────────────────
# Phase 4: TF-IDF Routing
# ─────────────────────────────────────────────

def phase_eval_routing() -> dict:
    """Train and evaluate TF-IDF routing at N=3."""
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeClassifier

    # Collect domain-specific text samples for routing training
    n_route_train = 200
    n_route_test = 100

    domain_texts = {}

    # Math: GSM8K questions
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=SEED + 1)  # different seed than training
    texts = [ex["question"] for ex in ds.select(range(min(n_route_train + n_route_test, len(ds))))]
    domain_texts["math"] = texts

    # Code: CodeAlpaca instructions
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    ds = ds.shuffle(seed=SEED + 1)
    texts = [ex["instruction"] for ex in ds.select(range(min(n_route_train + n_route_test, len(ds))))]
    domain_texts["code"] = texts

    # Medical: MedMCQA questions
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    ds = ds.shuffle(seed=SEED + 1)
    texts = [ex["question"] for ex in ds.select(range(min(n_route_train + n_route_test, len(ds))))]
    domain_texts["medical"] = texts

    # Build train/test sets
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    for domain, texts in domain_texts.items():
        train_texts.extend(texts[:n_route_train])
        train_labels.extend([domain] * min(n_route_train, len(texts)))
        test_texts.extend(texts[n_route_train:n_route_train + n_route_test])
        test_labels.extend([domain] * min(n_route_test, len(texts) - n_route_train))

    # Train TF-IDF + Ridge
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train, train_labels)

    X_test = vectorizer.transform(test_texts)
    preds = clf.predict(X_test)

    accuracy = sum(p == t for p, t in zip(preds, test_labels)) / len(test_labels) * 100

    # Per-domain accuracy
    per_domain = {}
    for domain in ["math", "code", "medical"]:
        mask = [t == domain for t in test_labels]
        domain_preds = [p for p, m in zip(preds, mask) if m]
        domain_true = [t for t, m in zip(test_labels, mask) if m]
        per_domain[domain] = sum(p == t for p, t in zip(domain_preds, domain_true)) / len(domain_true) * 100

    print(f"\nRouting accuracy: {accuracy:.1f}%", flush=True)
    for d, a in per_domain.items():
        print(f"  {d}: {a:.1f}%", flush=True)

    return {
        "routing_accuracy_pct": round(accuracy, 1),
        "routing_per_domain": {d: round(a, 1) for d, a in per_domain.items()},
    }


# ─────────────────────────────────────────────
# Phase 5: E2E Latency
# ─────────────────────────────────────────────

def phase_eval_latency() -> dict:
    """Measure end-to-end latency: route + load adapter + generate 100 tokens."""
    from mlx_lm import generate, load
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeClassifier

    # Quick routing setup
    test_prompts = [
        "Solve: what is 25% of 120?",
        "Write a Python function to reverse a string.",
        "What is the mechanism of action of metformin?",
    ]

    # Train minimal router
    train_texts = [
        "solve math problem", "calculate the sum", "what is the equation",
        "write a function", "implement an algorithm", "debug this code",
        "what causes diabetes", "treatment for hypertension", "drug mechanism",
    ]
    train_labels = ["math"] * 3 + ["code"] * 3 + ["medical"] * 3

    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(train_texts)
    clf = RidgeClassifier()
    clf.fit(X, train_labels)

    # Load model once
    model, tokenizer = load(MODEL_ID)

    latencies = []
    for prompt in test_prompts:
        t0 = time.time()

        # Route
        X_q = vectorizer.transform([prompt])
        domain = clf.predict(X_q)[0]
        t_route = time.time() - t0

        # Load adapter (simulate swap)
        adapter_path = EXPERIMENT_DIR / "adapters" / domain
        if adapter_path.exists():
            model_a, tokenizer_a = load(MODEL_ID, adapter_path=str(adapter_path))
        else:
            model_a, tokenizer_a = model, tokenizer
        t_load = time.time() - t0 - t_route

        # Generate
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer_a.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        _ = generate(model_a, tokenizer_a, prompt=formatted, max_tokens=100, verbose=False)
        t_total = time.time() - t0

        latencies.append({
            "prompt": prompt[:50],
            "domain": domain,
            "route_ms": round(t_route * 1000, 1),
            "load_ms": round(t_load * 1000, 1),
            "total_s": round(t_total, 2),
        })
        print(f"  Latency: {domain} -> {t_total:.2f}s (route={t_route*1000:.0f}ms, load={t_load*1000:.0f}ms)", flush=True)

        if model_a is not model:
            cleanup(model_a, tokenizer_a)

    cleanup(model, tokenizer)

    avg_latency = sum(l["total_s"] for l in latencies) / len(latencies)
    return {
        "latencies": latencies,
        "avg_latency_s": round(avg_latency, 2),
    }


# ─────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────

def main():
    t_start = time.time()
    log_memory("start")
    print(f"P0 E2E Benchmark: v_proj+o_proj adapters on GSM8K/HumanEval/MedMCQA", flush=True)
    print(f"SMOKE={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL}, N_STEPS={N_STEPS}", flush=True)
    print(f"LoRA rank={LORA_RANK}, keys={LORA_KEYS}", flush=True)

    # Phase 1: Prepare data
    print("\n" + "="*60, flush=True)
    print("PHASE 1: Prepare datasets", flush=True)
    print("="*60, flush=True)
    phase_prepare_data()

    # Phase 2: Base model evaluation
    print("\n" + "="*60, flush=True)
    print("PHASE 2: Base model evaluation", flush=True)
    print("="*60, flush=True)

    base_gsm8k = phase_eval_gsm8k(adapter_path=None)
    base_humaneval = phase_eval_humaneval(adapter_path=None)
    base_medmcqa = phase_eval_medmcqa(adapter_path=None)

    print(f"\nBase: GSM8K={base_gsm8k:.1f}%, HumanEval={base_humaneval:.1f}%, MedMCQA={base_medmcqa:.1f}%", flush=True)

    # Phase 3: Train adapters
    print("\n" + "="*60, flush=True)
    print("PHASE 3: Train v_proj+o_proj adapters", flush=True)
    print("="*60, flush=True)

    train_results = {}
    for domain in ["math", "code", "medical"]:
        train_results[domain] = phase_train_adapter(domain)
        log_memory(f"after-{domain}-train")

    # Phase 4: Adapted model evaluation
    print("\n" + "="*60, flush=True)
    print("PHASE 4: Adapted model evaluation", flush=True)
    print("="*60, flush=True)

    math_adapter = EXPERIMENT_DIR / "adapters" / "math"
    code_adapter = EXPERIMENT_DIR / "adapters" / "code"
    med_adapter = EXPERIMENT_DIR / "adapters" / "medical"

    adapted_gsm8k = phase_eval_gsm8k(adapter_path=math_adapter)
    adapted_humaneval = phase_eval_humaneval(adapter_path=code_adapter)
    adapted_medmcqa = phase_eval_medmcqa(adapter_path=med_adapter)

    math_delta = adapted_gsm8k - base_gsm8k
    code_delta = adapted_humaneval - base_humaneval
    med_delta = adapted_medmcqa - base_medmcqa

    print(f"\nAdapted: GSM8K={adapted_gsm8k:.1f}% ({math_delta:+.1f}pp), "
          f"HumanEval={adapted_humaneval:.1f}% ({code_delta:+.1f}pp), "
          f"MedMCQA={adapted_medmcqa:.1f}% ({med_delta:+.1f}pp)", flush=True)

    # Phase 5: Routing evaluation
    print("\n" + "="*60, flush=True)
    print("PHASE 5: TF-IDF routing", flush=True)
    print("="*60, flush=True)

    routing_results = phase_eval_routing()

    # Phase 6: E2E latency
    print("\n" + "="*60, flush=True)
    print("PHASE 6: E2E latency", flush=True)
    print("="*60, flush=True)

    latency_results = phase_eval_latency()

    # ── Results ───────────────────────────────
    total_time = time.time() - t_start

    results = {
        "is_smoke": IS_SMOKE,
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "n_steps": N_STEPS,
        "lora_rank": LORA_RANK,
        "lora_keys": LORA_KEYS,

        # Base accuracy
        "base_gsm8k_pct": round(base_gsm8k, 1),
        "base_humaneval_pct": round(base_humaneval, 1),
        "base_medmcqa_pct": round(base_medmcqa, 1),

        # Adapted accuracy
        "adapted_gsm8k_pct": round(adapted_gsm8k, 1),
        "adapted_humaneval_pct": round(adapted_humaneval, 1),
        "adapted_medmcqa_pct": round(adapted_medmcqa, 1),

        # Deltas
        "math_delta_pp": round(math_delta, 1),
        "code_delta_pp": round(code_delta, 1),
        "med_delta_pp": round(med_delta, 1),

        # Training
        "train_results": train_results,

        # Routing
        "routing_accuracy_pct": routing_results["routing_accuracy_pct"],
        "routing_per_domain": routing_results["routing_per_domain"],

        # Latency
        "avg_latency_s": latency_results["avg_latency_s"],
        "latencies": latency_results["latencies"],

        # Kill criteria
        "K1328_math_gsm8k": "PASS" if math_delta >= 10 else "FAIL",
        "K1329_code_humaneval": "PASS" if code_delta >= 10 else "FAIL",
        "K1330_med_medmcqa": "PASS" if med_delta >= 10 else "FAIL",
        "K1331_routing": "PASS" if routing_results["routing_accuracy_pct"] >= 90 else "FAIL",
        "K1332_latency": "PASS" if latency_results["avg_latency_s"] <= 2.0 else "FAIL",

        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # Summary
    print("\n" + "="*60, flush=True)
    print("KILL CRITERIA SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"K1328 Math GSM8K:     {base_gsm8k:.1f}% -> {adapted_gsm8k:.1f}% ({math_delta:+.1f}pp): {results['K1328_math_gsm8k']}", flush=True)
    print(f"K1329 Code HumanEval: {base_humaneval:.1f}% -> {adapted_humaneval:.1f}% ({code_delta:+.1f}pp): {results['K1329_code_humaneval']}", flush=True)
    print(f"K1330 Med MedMCQA:    {base_medmcqa:.1f}% -> {adapted_medmcqa:.1f}% ({med_delta:+.1f}pp): {results['K1330_med_medmcqa']}", flush=True)
    print(f"K1331 Routing:        {routing_results['routing_accuracy_pct']:.1f}%: {results['K1331_routing']}", flush=True)
    print(f"K1332 Latency:        {latency_results['avg_latency_s']:.2f}s: {results['K1332_latency']}", flush=True)
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
