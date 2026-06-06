#!/usr/bin/env python3
"""exp_g4_zs_base_transfer_4bit_fp16_full — F#666 retrofit of F#680.

Measures K1815 (median per-domain task-accuracy ratio R_task across
HumanEval/GSM8K/MedQA, 4bit→8bit base, same parent-trained adapter).

K1814 is INHERITED from parent results.json (median R_ppl=0.9459 on disk).

Phased execution (memory safety: phased + clear_cache between bases):
  Phase 0: read parent results.json → K1814 inherited
  Phase 1: load 4-bit base → 3 evals (HumanEval, GSM8K, MedQA) with matching adapter → record acc → unload
  Phase 2: load 8-bit base → 3 evals with same adapters → record acc → unload
  Phase 3: compute R_task per domain → K1815 → write results.json

n=50 per domain (matches parent T2.1).

Eval helpers ported (NOT copy-pasted blindly) from
exp_p1_t2_single_domain_training/run_experiment.py — already audit-2026-04-17
fixed (MedMCQA→MedQA-USMLE-4-options; GSM8K max_tokens 256→1024).
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
import mlx_lm

# Memory safety
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXP_DIR = Path(__file__).parent
RESULTS_FILE = EXP_DIR / "results.json"
PARENT_RESULTS = Path("experiments/models/exp_g4_zs_base_transfer_4bit_fp16/results.json")
ADAPTERS_ROOT = Path("experiments/models/exp_p1_t2_single_domain_training/adapters")

MODEL_4BIT = "mlx-community/gemma-4-e4b-it-4bit"
MODEL_8BIT = "mlx-community/gemma-4-e4b-it-8bit"

SMOKE = os.environ.get("SMOKE_TEST", "").strip() in ("1", "true", "yes")
N_EVAL = 5 if SMOKE else 50
SEED = 42

DOMAIN_ADAPTER = {"code": "code", "math": "math", "medical": "medical"}


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


def eval_gsm8k(model_id, adapter_path):
    """Returns task accuracy 0-100 on GSM8K test split (n=N_EVAL)."""
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(N_EVAL, len(ds))))

    model, tokenizer = load(model_id, adapter_path=str(adapter_path))
    log_memory(f"gsm8k-loaded({model_id.split('-')[-1]})")

    correct = 0
    for ex in ds:
        prompt = f"Solve the following math problem step by step.\n\n{ex['question']}\n\nAnswer:"
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        response = generate(
            model, tokenizer, prompt=formatted, max_tokens=1024, verbose=False
        )

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

    acc = correct / len(ds) * 100
    print(f"GSM8K({model_id.split('-')[-1]}): {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return acc


def eval_humaneval(model_id, adapter_path):
    """Returns pass@1 0-100 on HumanEval (n=N_EVAL)."""
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("openai_humaneval", split="test")
    ds = ds.select(range(min(N_EVAL, len(ds))))

    model, tokenizer = load(model_id, adapter_path=str(adapter_path))
    log_memory(f"humaneval-loaded({model_id.split('-')[-1]})")

    passed = 0
    for ex in ds:
        prompt = ex["prompt"]
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": f"Complete the following Python function:\n\n```python\n{prompt}\n```\n\nRespond with only the function body, no markdown."}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        response = generate(
            model, tokenizer, prompt=formatted, max_tokens=512, verbose=False
        )

        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        completion = code_match.group(1) if code_match else response

        full_code = prompt + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"

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
    print(f"HumanEval({model_id.split('-')[-1]}): {passed}/{len(ds)} = {acc:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return acc


def eval_medqa(model_id, adapter_path):
    """Returns accuracy 0-100 on MedQA-USMLE-4-options test (n=N_EVAL)."""
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(N_EVAL, len(ds))))

    model, tokenizer = load(model_id, adapter_path=str(adapter_path))
    log_memory(f"medqa-loaded({model_id.split('-')[-1]})")

    correct = 0
    for ex in ds:
        opts = ex["options"]
        question = (
            f"{ex['question']}\n"
            f"(A) {opts['A']}\n(B) {opts['B']}\n(C) {opts['C']}\n(D) {opts['D']}"
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
            model, tokenizer, prompt=formatted, max_tokens=20, verbose=False
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
    print(f"MedQA({model_id.split('-')[-1]}): {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return acc


def eval_one_base(model_id):
    """Run all 3 (dataset, matching adapter) evals on one base model. Returns dict."""
    res = {}
    res["humaneval"] = eval_humaneval(model_id, ADAPTERS_ROOT / "code")
    log_memory(f"after-humaneval({model_id.split('-')[-1]})")
    res["gsm8k"] = eval_gsm8k(model_id, ADAPTERS_ROOT / "math")
    log_memory(f"after-gsm8k({model_id.split('-')[-1]})")
    res["medqa"] = eval_medqa(model_id, ADAPTERS_ROOT / "medical")
    log_memory(f"after-medqa({model_id.split('-')[-1]})")
    return res


def main():
    t0 = time.time()
    print(f"[exp_g4_zs_base_transfer_4bit_fp16_full] start (smoke={SMOKE}, n_eval={N_EVAL})", flush=True)

    # Phase 0: K1814 inherited from parent
    parent = json.loads(PARENT_RESULTS.read_text())
    median_r_ppl = parent["median_transfer_ratio"]
    min_r_ppl = parent["min_transfer_ratio"]
    k1814_pass = (median_r_ppl >= 0.95) and (min_r_ppl >= 0.85)
    print(f"K1814 (inherited): median R_ppl={median_r_ppl:.4f}, min={min_r_ppl:.4f} → {'PASS' if k1814_pass else 'FAIL'}", flush=True)

    # Phase 1: 4-bit base
    print("\n=== Phase 1: 4-bit base ===", flush=True)
    acc_4bit = eval_one_base(MODEL_4BIT)
    cleanup()
    log_memory("phase1-end")

    # Phase 2: 8-bit base
    print("\n=== Phase 2: 8-bit base ===", flush=True)
    acc_8bit = eval_one_base(MODEL_8BIT)
    cleanup()
    log_memory("phase2-end")

    # Phase 3: ratios + verdict
    print("\n=== Phase 3: ratios + verdict ===", flush=True)
    domains = {
        "code": ("humaneval", acc_4bit["humaneval"], acc_8bit["humaneval"]),
        "math": ("gsm8k", acc_4bit["gsm8k"], acc_8bit["gsm8k"]),
        "medical": ("medqa", acc_4bit["medqa"], acc_8bit["medqa"]),
    }
    r_task = {}
    for d, (ds, a4, a8) in domains.items():
        if a4 <= 0:
            r_task[d] = 0.0  # cannot compute ratio if base is 0
        else:
            r_task[d] = a8 / a4
        print(f"R_task({d}/{ds}): 4bit={a4:.1f}% → 8bit={a8:.1f}% → R={r_task[d]:.4f}", flush=True)

    median_r_task = sorted(r_task.values())[1]
    min_r_task = min(r_task.values())
    k1815_pass = (median_r_task >= 0.95) and (min_r_task >= 0.85)
    print(f"K1815: median R_task={median_r_task:.4f}, min={min_r_task:.4f} → {'PASS' if k1815_pass else 'FAIL'}", flush=True)

    # F#666 truth table
    if k1814_pass and k1815_pass:
        verdict = "supported"
    elif (not k1814_pass) and k1815_pass:
        verdict = "supported_target_only"  # finding about the proxy
    elif (not k1814_pass) and (not k1815_pass):
        verdict = "killed"
    else:
        verdict = "killed"  # tautological proxy: K1814 PASS + K1815 FAIL

    elapsed = time.time() - t0

    out = {
        "experiment_id": "exp_g4_zs_base_transfer_4bit_fp16_full",
        "verdict": verdict.upper() if verdict != "supported_target_only" else "PROVISIONAL",
        "verdict_internal": verdict,
        "all_pass": k1814_pass and k1815_pass,
        "is_smoke": SMOKE,
        "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
        "n_eval_per_domain": N_EVAL,
        "k1814": {
            "inherited_from": str(PARENT_RESULTS),
            "median_r_ppl": median_r_ppl,
            "min_r_ppl": min_r_ppl,
            "pass": k1814_pass,
        },
        "k1815": {
            "median_r_task": median_r_task,
            "min_r_task": min_r_task,
            "pass": k1815_pass,
            "per_domain": {
                d: {
                    "dataset": ds,
                    "acc_4bit": a4,
                    "acc_8bit": a8,
                    "r_task": r_task[d],
                }
                for d, (ds, a4, a8) in domains.items()
            },
        },
        "elapsed_s": elapsed,
    }
    RESULTS_FILE.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {RESULTS_FILE}", flush=True)
    print(f"Verdict: {out['verdict']} (internal: {verdict}); elapsed {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
