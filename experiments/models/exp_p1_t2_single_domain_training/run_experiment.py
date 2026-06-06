#!/usr/bin/env python3
"""
T2.1: Train single domain adapter on Gemma 4 E4B (math, code, medical)

Kill criteria (canonical KC text from DB — do not edit):
  K1028: Math adapter: GSM8K accuracy improves >= 5pp over base
  K1029: Code adapter: HumanEval improves >= 5pp over base
  K1030: Medical adapter: MedQA improves >= 3pp over base
  K1031: Training cost < 1 GPU-hour per domain on A100/H100
  K1032: Adapter size < 50MB per domain (compressed)

Audit-2026-04-17-rerun fixes applied:
  - metric-swap: MedMCQA → GBaker/MedQA-USMLE-4-options (K1030 now measures canonical KC metric)
  - format-artifact: eval_gsm8k max_tokens 256 → 1024 (prevents Gemma 4 CoT truncation before '#### answer')
"""

import gc
import json
import os
import re
import subprocess
import sys
import textwrap
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
N_EVAL = 5 if IS_SMOKE else 50
N_STEPS = 20 if IS_SMOKE else 1000
SEED = 42


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ─────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────

def prepare_gsm8k_data(data_dir: Path, n_train: int):
    """Prepare GSM8K training JSONL for math adapter."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n_train, len(ds))))

    data_dir.mkdir(parents=True, exist_ok=True)
    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"

    records = []
    for ex in ds:
        msg = {
            "messages": [
                {"role": "user", "content": f"Solve the following math problem step by step.\n\n{ex['question']}"},
                {"role": "assistant", "content": ex["answer"]},
            ]
        }
        records.append(json.dumps(msg))

    # 90/10 train/val split
    n_val = max(1, len(records) // 10)
    train_file.write_text("\n".join(records[n_val:]))
    valid_file.write_text("\n".join(records[:n_val]))
    print(f"GSM8K: {len(records)-n_val} train, {n_val} val", flush=True)


def prepare_code_data(data_dir: Path, n_train: int):
    """Prepare CodeAlpaca training JSONL for code adapter."""
    from datasets import load_dataset

    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n_train, len(ds))))

    data_dir.mkdir(parents=True, exist_ok=True)
    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"

    records = []
    for ex in ds:
        content = ex["instruction"]
        if ex.get("input", ""):
            content += f"\n\nInput:\n{ex['input']}"
        msg = {
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": ex["output"]},
            ]
        }
        records.append(json.dumps(msg))

    n_val = max(1, len(records) // 10)
    train_file.write_text("\n".join(records[n_val:]))
    valid_file.write_text("\n".join(records[:n_val]))
    print(f"CodeAlpaca: {len(records)-n_val} train, {n_val} val", flush=True)


def prepare_medical_data(data_dir: Path, n_train: int):
    """Prepare MedQA (USMLE, 4-option) training JSONL for medical adapter.

    Canonical KC text (K1030): 'Medical adapter: MedQA improves >= 3pp over base'.
    Dataset GBaker/MedQA-USMLE-4-options is USMLE-style with A/B/C/D options.
    """
    from datasets import load_dataset

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n_train, len(ds))))

    data_dir.mkdir(parents=True, exist_ok=True)
    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"

    records = []
    for ex in ds:
        opts = ex["options"]  # dict {'A': ..., 'B': ..., 'C': ..., 'D': ...}
        question = (
            f"{ex['question']}\n"
            f"(A) {opts['A']}\n"
            f"(B) {opts['B']}\n"
            f"(C) {opts['C']}\n"
            f"(D) {opts['D']}"
        )
        answer_letter = ex["answer_idx"]  # 'A'|'B'|'C'|'D'
        answer_text = ex["answer"]
        msg = {
            "messages": [
                {"role": "user", "content": f"Answer this medical multiple choice question. Respond with only the letter (A/B/C/D).\n\n{question}"},
                {"role": "assistant", "content": f"{answer_letter}: {answer_text}"},
            ]
        }
        records.append(json.dumps(msg))

    n_val = max(1, len(records) // 10)
    train_file.write_text("\n".join(records[n_val:]))
    valid_file.write_text("\n".join(records[:n_val]))
    print(f"MedQA: {len(records)-n_val} train, {n_val} val", flush=True)


# ─────────────────────────────────────────────
# LoRA config
# ─────────────────────────────────────────────

def write_lora_config(config_path: Path, data_dir: Path, adapter_path: Path, n_steps: int):
    """Write YAML config for mlx_lm.lora training."""
    config = {
        "model": MODEL_ID,
        "train": True,
        "data": str(data_dir),
        "fine_tune_type": "lora",
        "num_layers": -1,  # all 42 layers
        "iters": n_steps,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "lora_parameters": {
            "rank": 6,
            "scale": 6.0,
            "dropout": 0.0,
            "keys": ["self_attn.q_proj"],
        },
        "adapter_path": str(adapter_path),
        "save_every": n_steps,  # save once at end
        "val_batches": 5,
        "steps_per_report": max(10, n_steps // 10),
        "steps_per_eval": n_steps,  # eval once at end
        "max_seq_length": 512,
        "mask_prompt": True,
        "grad_checkpoint": True,
        "seed": SEED,
    }

    import yaml
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


# ─────────────────────────────────────────────
# Training (subprocess for memory isolation)
# ─────────────────────────────────────────────

def train_adapter(domain: str, data_dir: Path, adapter_path: Path) -> dict:
    """Train a domain LoRA adapter via mlx_lm.lora subprocess."""
    adapter_path.mkdir(parents=True, exist_ok=True)
    config_path = EXPERIMENT_DIR / f"lora_config_{domain}.yaml"
    write_lora_config(config_path, data_dir, adapter_path, N_STEPS)

    print(f"\n=== Training {domain} adapter ({N_STEPS} steps) ===", flush=True)
    t0 = time.time()

    cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", str(config_path)]
    result = subprocess.run(
        cmd,
        capture_output=False,  # show output live
        text=True,
        cwd=str(EXPERIMENT_DIR),
    )

    elapsed = time.time() - t0
    print(f"{domain} training done in {elapsed:.1f}s (exit={result.returncode})", flush=True)

    if result.returncode != 0:
        print(f"WARNING: {domain} training failed (code {result.returncode})", flush=True)

    return {"train_time_s": round(elapsed, 1), "exit_code": result.returncode}


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def eval_gsm8k(adapter_path=None, n_eval=50) -> float:
    """Evaluate GSM8K accuracy. Returns accuracy 0-100."""
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    load_kwargs = {"model_path": MODEL_ID}
    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)

    log_memory(f"gsm8k-eval-loaded{'(adapter)' if adapter_path else ''}")

    correct = 0
    for i, ex in enumerate(ds):
        prompt = f"Solve the following math problem step by step.\n\n{ex['question']}\n\nAnswer:"
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        # max_tokens=1024 to accommodate Gemma 4 CoT reasoning before the '#### <answer>'
        # sentinel; prior 256 truncated CoT → base_gsm8k=0% format-artifact (audit).
        response = generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=1024,
            verbose=False,
        )

        # Extract #### answer
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
            # Try last number in response
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt_ans:
                correct += 1

    acc = correct / len(ds) * 100
    print(f"GSM8K accuracy: {correct}/{len(ds)} = {acc:.1f}%", flush=True)

    cleanup(model, tokenizer)
    return acc


def eval_humaneval(adapter_path=None, n_eval=50) -> float:
    """Evaluate HumanEval pass@1. Returns accuracy 0-100."""
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("openai_humaneval", split="test")
    ds = ds.select(range(min(n_eval, len(ds))))

    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)

    log_memory(f"humaneval-eval-loaded{'(adapter)' if adapter_path else ''}")

    passed = 0
    for i, ex in enumerate(ds):
        prompt = ex["prompt"]
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": f"Complete the following Python function:\n\n```python\n{prompt}\n```\n\nRespond with only the function body, no markdown."}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        response = generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=512,
            verbose=False,
        )

        # Extract code from response
        # Remove markdown code blocks if present
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if code_match:
            completion = code_match.group(1)
        else:
            completion = response

        # Build full function to test
        full_code = prompt + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"

        # Execute with timeout
        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                timeout=10,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                passed += 1
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

    acc = passed / len(ds) * 100
    print(f"HumanEval pass@1: {passed}/{len(ds)} = {acc:.1f}%", flush=True)

    cleanup(model, tokenizer)
    return acc


def eval_medqa(adapter_path=None, n_eval=50) -> float:
    """Evaluate MedQA (USMLE, 4-option) accuracy. Returns accuracy 0-100.

    Canonical KC text (K1030): 'Medical adapter: MedQA improves >= 3pp over base'.
    """
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)

    log_memory(f"medqa-eval-loaded{'(adapter)' if adapter_path else ''}")

    correct = 0

    for ex in ds:
        opts = ex["options"]
        question = (
            f"{ex['question']}\n"
            f"(A) {opts['A']}\n"
            f"(B) {opts['B']}\n"
            f"(C) {opts['C']}\n"
            f"(D) {opts['D']}"
        )
        prompt = f"Answer this medical multiple choice question. Respond with only the letter (A/B/C/D).\n\n{question}"

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        response = generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=20,
            verbose=False,
        )

        gt = ex["answer_idx"]
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

    acc = correct / len(ds) * 100
    print(f"MedQA accuracy: {correct}/{len(ds)} = {acc:.1f}%", flush=True)

    cleanup(model, tokenizer)
    return acc


# ─────────────────────────────────────────────
# Adapter size measurement
# ─────────────────────────────────────────────

def measure_adapter_size(adapter_path: Path) -> float:
    """Return total adapter size in MB."""
    if not adapter_path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in adapter_path.rglob("*") if f.is_file())
    return total / 1e6


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t_start = time.time()
    log_memory("start")

    print(f"T2.1: Single Domain Adapter Training", flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_EVAL={N_EVAL}, N_STEPS={N_STEPS}", flush=True)

    # Dirs
    data_dir = EXPERIMENT_DIR / "data"
    adapters_dir = EXPERIMENT_DIR / "adapters"
    math_adapter = adapters_dir / "math"
    code_adapter = adapters_dir / "code"
    med_adapter = adapters_dir / "medical"

    # ── Phase 1: Prepare datasets ──────────────────────────
    print("\n=== Phase 1: Preparing datasets ===", flush=True)
    prepare_gsm8k_data(data_dir / "math", N_TRAIN)
    prepare_code_data(data_dir / "code", N_TRAIN)
    prepare_medical_data(data_dir / "medical", N_TRAIN)

    # ── Phase 2: Base model evaluation ────────────────────
    print("\n=== Phase 2: Base model evaluation ===", flush=True)
    base_gsm8k = eval_gsm8k(adapter_path=None, n_eval=N_EVAL)
    log_memory("after-base-gsm8k")

    base_humaneval = eval_humaneval(adapter_path=None, n_eval=N_EVAL)
    log_memory("after-base-humaneval")

    base_medqa = eval_medqa(adapter_path=None, n_eval=N_EVAL)
    log_memory("after-base-medqa")

    print(f"\nBase accuracy: GSM8K={base_gsm8k:.1f}%, HumanEval={base_humaneval:.1f}%, MedQA={base_medqa:.1f}%", flush=True)

    # ── Phase 3: Train math adapter ───────────────────────
    print("\n=== Phase 3: Train math adapter ===", flush=True)
    math_train = train_adapter("math", data_dir / "math", math_adapter)
    log_memory("after-math-train")

    # ── Phase 4: Evaluate math adapter ────────────────────
    print("\n=== Phase 4: Eval math adapter ===", flush=True)
    math_gsm8k = eval_gsm8k(adapter_path=math_adapter, n_eval=N_EVAL)
    log_memory("after-math-eval")

    math_delta = math_gsm8k - base_gsm8k
    print(f"Math delta: {math_delta:+.1f}pp (K1028: {'PASS' if math_delta >= 5 else 'FAIL'})", flush=True)

    # ── Phase 5: Train code adapter ───────────────────────
    print("\n=== Phase 5: Train code adapter ===", flush=True)
    code_train = train_adapter("code", data_dir / "code", code_adapter)
    log_memory("after-code-train")

    # ── Phase 6: Evaluate code adapter ────────────────────
    print("\n=== Phase 6: Eval code adapter ===", flush=True)
    code_humaneval = eval_humaneval(adapter_path=code_adapter, n_eval=N_EVAL)
    log_memory("after-code-eval")

    code_delta = code_humaneval - base_humaneval
    print(f"Code delta: {code_delta:+.1f}pp (K1029: {'PASS' if code_delta >= 5 else 'FAIL'})", flush=True)

    # ── Phase 7: Train medical adapter ────────────────────
    print("\n=== Phase 7: Train medical adapter ===", flush=True)
    med_train = train_adapter("medical", data_dir / "medical", med_adapter)
    log_memory("after-medical-train")

    # ── Phase 8: Evaluate medical adapter ─────────────────
    print("\n=== Phase 8: Eval medical adapter ===", flush=True)
    med_medqa = eval_medqa(adapter_path=med_adapter, n_eval=N_EVAL)
    log_memory("after-medical-eval")

    med_delta = med_medqa - base_medqa
    print(f"Medical delta: {med_delta:+.1f}pp (K1030: {'PASS' if med_delta >= 3 else 'FAIL'})", flush=True)

    # ── Phase 9: Measure adapter sizes ────────────────────
    print("\n=== Phase 9: Adapter sizes ===", flush=True)
    math_size_mb = measure_adapter_size(math_adapter)
    code_size_mb = measure_adapter_size(code_adapter)
    med_size_mb = measure_adapter_size(med_adapter)

    print(f"Math adapter: {math_size_mb:.2f} MB", flush=True)
    print(f"Code adapter: {code_size_mb:.2f} MB", flush=True)
    print(f"Medical adapter: {med_size_mb:.2f} MB", flush=True)

    # ── K1031: Training cost ───────────────────────────────
    max_train_time_s = max(math_train["train_time_s"], code_train["train_time_s"], med_train["train_time_s"])
    k1031_pass = max_train_time_s < 3600  # < 1 hour

    # ── Kill criteria summary ──────────────────────────────
    results = {
        "is_smoke": IS_SMOKE,
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "n_steps": N_STEPS,

        # Base accuracy
        "base_gsm8k_pct": round(base_gsm8k, 1),
        "base_humaneval_pct": round(base_humaneval, 1),
        "base_medqa_pct": round(base_medqa, 1),

        # Adapter accuracy
        "math_gsm8k_pct": round(math_gsm8k, 1),
        "code_humaneval_pct": round(code_humaneval, 1),
        "med_medqa_pct": round(med_medqa, 1),

        # Deltas
        "math_delta_pp": round(math_delta, 1),
        "code_delta_pp": round(code_delta, 1),
        "med_delta_pp": round(med_delta, 1),

        # Training
        "math_train_time_s": math_train["train_time_s"],
        "code_train_time_s": code_train["train_time_s"],
        "med_train_time_s": med_train["train_time_s"],

        # Adapter sizes
        "math_adapter_mb": round(math_size_mb, 2),
        "code_adapter_mb": round(code_size_mb, 2),
        "med_adapter_mb": round(med_size_mb, 2),

        # Kill criteria
        "K1028_math_gsm8k": "PASS" if math_delta >= 5 else "FAIL",
        "K1029_code_humaneval": "PASS" if code_delta >= 5 else "FAIL",
        "K1030_med_medqa": "PASS" if med_delta >= 3 else "FAIL",
        "K1031_train_cost": "PASS" if k1031_pass else "FAIL",
        "K1032_adapter_size": "PASS" if max(math_size_mb, code_size_mb, med_size_mb) < 50 else "FAIL",

        "total_time_s": round(time.time() - t_start, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # Summary
    print("\n" + "="*60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"K1028 Math GSM8K  {base_gsm8k:.1f}% → {math_gsm8k:.1f}% ({math_delta:+.1f}pp): {results['K1028_math_gsm8k']}", flush=True)
    print(f"K1029 Code HumanEval {base_humaneval:.1f}% → {code_humaneval:.1f}% ({code_delta:+.1f}pp): {results['K1029_code_humaneval']}", flush=True)
    print(f"K1030 Med MedQA   {base_medqa:.1f}% → {med_medqa:.1f}% ({med_delta:+.1f}pp): {results['K1030_med_medqa']}", flush=True)
    print(f"K1031 Train cost  {max_train_time_s:.0f}s max: {results['K1031_train_cost']}", flush=True)
    print(f"K1032 Adapter size {max(math_size_mb, code_size_mb, med_size_mb):.2f}MB max: {results['K1032_adapter_size']}", flush=True)
    print(f"Total time: {results['total_time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
