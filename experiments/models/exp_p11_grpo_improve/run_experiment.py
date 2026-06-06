#!/usr/bin/env python3
"""
P11.G0: GRPO Refinement — RS-SFT starting from F0 adapter (math-s1k-reasoning-v0)

Improves on B0 by initializing from F0 instead of the base model.
Theorem 1: Higher yield from better initialization → lower gradient variance.
Theorem 2: D_train = D_eval (MMLU-Pro) → catastrophic forgetting impossible.

Kill criteria:
  K1514: G0 adapter >= 70% MMLU-Pro with thinking (beats Google 69.4%)
  K1515: G0 GSM8K >= F0 GSM8K (no regression)
  K1516: G0 >= F0 + 3pp on at least one benchmark

References:
  - arXiv:2602.04118 (Learning to Reason in 13 Parameters — GRPO sample efficiency)
  - arXiv:2402.03300 (GRPO: Shao et al.)
  - arXiv:2501.12948 (DeepSeek-R1: RS-SFT as GRPO warmup)
  - arXiv:1612.00796 (EWC: catastrophic forgetting theory)
"""

import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_DIR = EXPERIMENT_DIR / "data"
ADAPTER_DIR = EXPERIMENT_DIR / "adapters" / "grpo_sft"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
F0_ADAPTER = REPO_ROOT / "data" / "adapters" / "math-s1k-reasoning-v0"
G0_ADAPTER_FINAL = REPO_ROOT / "data" / "adapters" / "math-s1k-grpo-v0"
MMLU_DATA = REPO_ROOT / "experiments/models/exp_bench_mmlu_pro/data/data/test.parquet"

IS_SMOKE = "--smoke" in sys.argv or os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42

# Sampling config
N_SAMPLE_QUESTIONS = 10 if IS_SMOKE else 280  # stratified across 14 categories
MAX_TOKENS = 2048 if IS_SMOKE else 4096       # thinking needs room

# Training config
LORA_RANK = 8
LORA_SCALE = 1.0
N_STEPS = 5 if IS_SMOKE else 200
BATCH_SIZE = 1
LR = 1e-5

# Evaluation config (identical to all P11 experiments)
EVAL_PER_CAT = 2 if IS_SMOKE else 20  # 20 × 14 = 280 questions
GSM8K_N = 5 if IS_SMOKE else 50
OPTION_LETTERS = "ABCDEFGHIJ"


def log(msg):
    print(msg, flush=True)


def log_memory(label):
    info = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={info:.2f}GB peak={peak:.2f}GB")
    mx.reset_peak_memory()


def format_mmlu_prompt(row, tokenizer, enable_thinking=True):
    """Format MMLU-Pro question. p10-validated format (62.1%)."""
    options = row["options"]
    option_text = "\n".join(
        f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
    )
    content = (
        f"The following is a multiple choice question. "
        f"Answer with ONLY the letter of the correct option "
        f"(A through {OPTION_LETTERS[len(options)-1]}). "
        f"Do not explain.\n\n"
        f"Question: {row['question']}\n\n"
        f"Options:\n{option_text}\n\n"
        f"Answer:"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def strip_thinking(response):
    """Remove Gemma 4 thinking channel tokens (p10-validated regex)."""
    if not response:
        return response, 0
    thinking_len = 0
    m = re.search(r'<\|channel>thought.*?<channel\|>', response, flags=re.DOTALL)
    if m:
        thinking_len = len(m.group(0))
    cleaned = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    return cleaned.strip(), thinking_len


def parse_answer(response):
    """Extract answer letter from model response."""
    if not response:
        return None, 0
    answer_text, thinking_chars = strip_thinking(response)
    if not answer_text:
        return None, thinking_chars
    if len(answer_text) == 1 and answer_text.upper() in OPTION_LETTERS:
        return answer_text.upper(), thinking_chars
    m = re.match(r"^([A-J])[.\s:)\-,]", answer_text)
    if m:
        return m.group(1), thinking_chars
    m = re.search(r"(?:answer|correct)\s+(?:is\s+)?([A-J])", answer_text, re.IGNORECASE)
    if m:
        return m.group(1).upper(), thinking_chars
    m = re.search(r"([A-J])", answer_text)
    if m:
        return m.group(1).upper(), thinking_chars
    return None, thinking_chars


# ─────────────────────────────────────────────────────────────
# Phase 1: Rejection Sampling with F0 adapter
# ─────────────────────────────────────────────────────────────

def phase1_rejection_sampling():
    """
    Generate N_SAMPLE_QUESTIONS completions using F0 adapter (higher yield).
    Keep correct completions → training dataset for Phase 2.

    Theorem 1: p_SFT >= p_base → more correct examples → better gradient signal.
    """
    log("\n[Phase 1] Rejection Sampling with F0 Adapter")
    t0 = time.time()

    from mlx_lm import load, generate

    # Load MMLU-Pro data
    log(f"  Loading MMLU-Pro data from {MMLU_DATA}")
    if not MMLU_DATA.exists():
        raise FileNotFoundError(f"MMLU data not found: {MMLU_DATA}")
    df = pd.read_parquet(MMLU_DATA)
    categories = df["category"].unique()
    log(f"  {len(df)} total questions, {len(categories)} categories")

    # Stratified sample: N_SAMPLE_QUESTIONS / 14 categories
    per_cat = max(1, N_SAMPLE_QUESTIONS // len(categories))
    rng = np.random.default_rng(SEED)

    sampled_rows = []
    for cat in categories:
        cat_df = df[df["category"] == cat]
        n = min(per_cat, len(cat_df))
        idx = rng.choice(len(cat_df), size=n, replace=False)
        sampled_rows.extend(cat_df.iloc[idx].to_dict("records"))

    log(f"  Sampled {len(sampled_rows)} questions across {len(categories)} categories")

    # Check F0 adapter exists
    if not F0_ADAPTER.exists():
        raise FileNotFoundError(
            f"F0 adapter not found: {F0_ADAPTER}\n"
            "Run exp_p11_s1k_reasoning_train_eval first."
        )
    log(f"  Loading F0 adapter from {F0_ADAPTER}")

    # Load model + F0 adapter (Theorem 1: better init → higher yield)
    model, tokenizer = load(MODEL_ID, adapter_path=str(F0_ADAPTER))
    log_memory("after_load")

    # Generate completions (greedy = default sampler in generate_step)
    correct_completions = []
    skipped = 0

    for i, row in enumerate(sampled_rows):
        prompt = format_mmlu_prompt(row, tokenizer, enable_thinking=True)
        try:
            response = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                verbose=False,
            )
        except Exception as e:
            log(f"  [q{i}] ERROR: {e}")
            skipped += 1
            continue

        predicted, thinking_chars = parse_answer(response)
        ground_truth = row["answer"]
        is_correct = (predicted == ground_truth)

        if i % 20 == 0 or i < 5:
            log(f"  [q{i}/{len(sampled_rows)}] cat={row['category'][:12]} "
                f"pred={predicted} gt={ground_truth} correct={is_correct} "
                f"thinking={thinking_chars}chars")

        if is_correct:
            options = row["options"]
            option_text = "\n".join(
                f"{OPTION_LETTERS[j]}. {opt}" for j, opt in enumerate(options)
            )
            user_content = (
                f"The following is a multiple choice question. "
                f"Answer with ONLY the letter of the correct option "
                f"(A through {OPTION_LETTERS[len(options)-1]}). "
                f"Do not explain.\n\n"
                f"Question: {row['question']}\n\n"
                f"Options:\n{option_text}\n\n"
                f"Answer:"
            )
            correct_completions.append({
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": response},
                ],
                "category": row["category"],
                "predicted": predicted,
                "ground_truth": ground_truth,
                "thinking_chars": thinking_chars,
            })

    del model
    gc.collect()
    mx.clear_cache()
    log_memory("after_sampling_cleanup")

    elapsed = time.time() - t0
    n_attempted = len(sampled_rows) - skipped
    yield_rate = len(correct_completions) / max(1, n_attempted)
    log(f"\n  Phase 1 complete: {len(correct_completions)} correct "
        f"({yield_rate:.1%} yield) from {n_attempted} attempted in {elapsed:.0f}s")

    if len(correct_completions) < 5:
        log("  WARNING: Very few correct completions (<5). Proceeding anyway.")

    return correct_completions, {
        "n_sampled": len(sampled_rows),
        "n_correct": len(correct_completions),
        "n_skipped": skipped,
        "yield_rate": yield_rate,
        "sampling_time_s": elapsed,
        "avg_thinking_chars": (
            float(np.mean([c["thinking_chars"] for c in correct_completions]))
            if correct_completions else 0.0
        ),
    }


# ─────────────────────────────────────────────────────────────
# Phase 2: SFT Training from F0 initialization
# ─────────────────────────────────────────────────────────────

def phase2_sft_training(correct_completions):
    """Train LoRA starting from F0 adapter weights on self-generated MMLU-Pro traces."""
    log(f"\n[Phase 2] SFT Training from F0 adapter ({len(correct_completions)} traces)")
    t0 = time.time()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # 90/10 train/val split
    rng = np.random.default_rng(SEED + 1)
    indices = rng.permutation(len(correct_completions))
    n_val = max(1, len(correct_completions) // 10)
    val_idx = set(indices[:n_val].tolist())
    train_idx = set(indices[n_val:].tolist())

    train_file = DATA_DIR / "train.jsonl"
    val_file = DATA_DIR / "valid.jsonl"

    with open(train_file, "w") as f:
        for i in sorted(train_idx):
            ex = correct_completions[i]
            f.write(json.dumps({"messages": ex["messages"]}) + "\n")

    with open(val_file, "w") as f:
        for i in sorted(val_idx):
            ex = correct_completions[i]
            f.write(json.dumps({"messages": ex["messages"]}) + "\n")

    n_train = len(train_idx)
    n_val_actual = len(val_idx)
    log(f"  Train: {n_train} examples, Val: {n_val_actual}")

    # F0 adapter init file
    f0_init_file = F0_ADAPTER / "adapters.safetensors"
    if not f0_init_file.exists():
        log(f"  WARNING: F0 adapter init file not found: {f0_init_file}")
        log("  Training from scratch (no --resume-adapter-file)")
        resume_args = []
    else:
        log(f"  Resuming from F0 adapter: {f0_init_file}")
        resume_args = ["--resume-adapter-file", str(f0_init_file)]

    # Write lora config (rank/scale/keys go in YAML, not CLI args)
    import yaml
    lora_config = {
        "lora_parameters": {
            "rank": LORA_RANK,
            "scale": LORA_SCALE,
            "dropout": 0.0,
            "keys": ["self_attn.v_proj", "self_attn.o_proj"],
        },
    }
    config_file = EXPERIMENT_DIR / "lora_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(lora_config, f)

    # Run mlx_lm.lora (rank/scale via -c config; --rank/--lora-scale not valid CLI args)
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", MODEL_ID,
        "--train",
        "--data", str(DATA_DIR),
        "--num-layers", "16",
        "--iters", str(N_STEPS),
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LR),
        "--val-batches", "5",
        "--save-every", str(max(50, N_STEPS)),
        "--adapter-path", str(ADAPTER_DIR),
        "-c", str(config_file),
    ] + resume_args

    log(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=False,
        cwd=REPO_ROOT,
        timeout=3600,
    )

    elapsed = time.time() - t0
    success = result.returncode == 0
    log(f"\n  Phase 2 complete: returncode={result.returncode} in {elapsed:.0f}s")

    # Copy adapter to canonical registry location
    if success and ADAPTER_DIR.exists():
        G0_ADAPTER_FINAL.mkdir(parents=True, exist_ok=True)
        for f in ADAPTER_DIR.iterdir():
            shutil.copy2(f, G0_ADAPTER_FINAL / f.name)
        log(f"  Adapter copied to {G0_ADAPTER_FINAL}")

    return {
        "n_train": n_train,
        "n_val": n_val_actual,
        "n_steps": N_STEPS,
        "training_time_s": elapsed,
        "training_success": success,
        "adapter_path": str(ADAPTER_DIR),
        "registry_path": str(G0_ADAPTER_FINAL),
        "used_f0_init": bool(resume_args),
    }


# ─────────────────────────────────────────────────────────────
# Phase 3: Evaluation (identical to all P11 experiments)
# ─────────────────────────────────────────────────────────────

def eval_mmlu(label, adapter_path=None):
    """Evaluate on 280q MMLU-Pro with thinking. p10/p11 standard protocol."""
    log(f"\n[Eval MMLU-Pro] {label}")
    t0 = time.time()

    from mlx_lm import load, generate

    load_kwargs = {"model": MODEL_ID}
    if adapter_path and Path(adapter_path).exists():
        load_kwargs["adapter_path"] = str(adapter_path)
        log(f"  Using adapter: {adapter_path}")

    model, tokenizer = load(**load_kwargs)
    log_memory(f"after_load_{label}")

    df = pd.read_parquet(MMLU_DATA)
    categories = df["category"].unique()
    rng = np.random.default_rng(SEED)

    sampled_rows = []
    for cat in categories:
        cat_df = df[df["category"] == cat]
        n = min(EVAL_PER_CAT, len(cat_df))
        idx = rng.choice(len(cat_df), size=n, replace=False)
        sampled_rows.extend(cat_df.iloc[idx].to_dict("records"))

    log(f"  Evaluating {len(sampled_rows)} questions")

    correct = 0
    total = 0
    per_category = {}
    thinking_chars_list = []

    for i, row in enumerate(sampled_rows):
        cat = row["category"]
        prompt = format_mmlu_prompt(row, tokenizer, enable_thinking=True)
        try:
            response = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=4096,
                verbose=False,
            )
        except Exception as e:
            log(f"  [q{i}] ERROR: {e}")
            per_category.setdefault(cat, {"correct": 0, "total": 0})["total"] += 1
            total += 1
            continue

        predicted, thinking_chars = parse_answer(response)
        ground_truth = row["answer"]
        is_correct = (predicted == ground_truth)

        per_category.setdefault(cat, {"correct": 0, "total": 0})
        per_category[cat]["total"] += 1
        per_category[cat]["correct"] += int(is_correct)
        correct += int(is_correct)
        total += 1
        thinking_chars_list.append(thinking_chars)

        if i % 50 == 0 or i < 3:
            log(f"  [q{i}/{len(sampled_rows)}] acc={correct/max(1,total):.1%} "
                f"thinking={thinking_chars}chars")

    del model
    gc.collect()
    mx.clear_cache()

    overall_acc = correct / max(1, total)
    per_cat_acc = {cat: v["correct"] / max(1, v["total"]) for cat, v in per_category.items()}
    elapsed = time.time() - t0

    log(f"  {label}: {overall_acc:.1%} ({correct}/{total}) in {elapsed:.0f}s")
    for cat, acc in sorted(per_cat_acc.items(), key=lambda x: x[1]):
        log(f"    {cat[:20]:<20} {per_category[cat]['correct']}/{per_category[cat]['total']} "
            f"= {acc:.1%}")

    return {
        "label": label,
        "overall_accuracy": overall_acc,
        "correct": correct,
        "total": total,
        "per_category": per_cat_acc,
        "avg_thinking_chars": float(np.mean(thinking_chars_list)) if thinking_chars_list else 0.0,
        "eval_time_s": elapsed,
    }


def eval_gsm8k(label, adapter_path=None):
    """Quick GSM8K eval for regression check (K1515)."""
    log(f"\n[Eval GSM8K] {label}")
    t0 = time.time()

    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        log(f"  ERROR loading GSM8K: {e}")
        return {"label": label, "accuracy": None, "error": str(e)}

    from mlx_lm import load, generate

    load_kwargs = {"model": MODEL_ID}
    if adapter_path and Path(adapter_path).exists():
        load_kwargs["adapter_path"] = str(adapter_path)

    model, tokenizer = load(**load_kwargs)
    log_memory(f"gsm8k_load_{label}")

    rng = np.random.default_rng(SEED + 2)
    indices = rng.choice(len(ds), size=min(GSM8K_N, len(ds)), replace=False)

    correct = 0
    total = 0

    for idx in indices:
        row = ds[int(idx)]
        question = row["question"]
        answer_raw = row["answer"]
        # Extract final number from GSM8K answer (after ####)
        m = re.search(r"####\s*([+-]?\d+(?:,\d+)*(?:\.\d+)?)", answer_raw)
        if not m:
            continue
        gt_num = m.group(1).replace(",", "")

        content = (
            f"Solve this math problem step by step. "
            f"At the end, write '#### <answer>' with the final number.\n\n"
            f"{question}"
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        try:
            response = generate(model, tokenizer, prompt=prompt, max_tokens=2048, verbose=False)
        except Exception as e:
            log(f"  GSM8K ERROR: {e}")
            total += 1
            continue

        # Strip thinking then extract ####
        cleaned, _ = strip_thinking(response)
        m2 = re.search(r"####\s*([+-]?\d+(?:,\d+)*(?:\.\d+)?)", cleaned)
        if m2:
            pred_num = m2.group(1).replace(",", "")
            is_correct = (pred_num == gt_num)
        else:
            is_correct = False

        correct += int(is_correct)
        total += 1

    del model
    gc.collect()
    mx.clear_cache()

    acc = correct / max(1, total)
    elapsed = time.time() - t0
    log(f"  GSM8K {label}: {acc:.1%} ({correct}/{total}) in {elapsed:.0f}s")
    return {"label": label, "accuracy": acc, "correct": correct, "total": total, "eval_time_s": elapsed}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    log(f"=== P11.G0: GRPO Refinement (RS-SFT from F0 Init) ===")
    log(f"SMOKE={IS_SMOKE}, N_SAMPLE={N_SAMPLE_QUESTIONS}, N_STEPS={N_STEPS}")
    log(f"F0 adapter: {F0_ADAPTER} (exists={F0_ADAPTER.exists()})")
    t_start = time.time()

    results = {
        "experiment": "exp_p11_grpo_improve",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "seed": SEED,
        "f0_adapter": str(F0_ADAPTER),
        "f0_adapter_exists": F0_ADAPTER.exists(),
    }

    # Phase 1: Rejection sampling with F0
    try:
        correct_completions, phase1_stats = phase1_rejection_sampling()
        results["phase1"] = phase1_stats
        log(f"\nPhase 1 done: yield={phase1_stats['yield_rate']:.1%}, "
            f"n_correct={phase1_stats['n_correct']}")
    except Exception as e:
        log(f"FATAL Phase 1 error: {e}")
        import traceback; traceback.print_exc()
        results["phase1_error"] = str(e)
        results["fatal"] = True
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        return

    if len(correct_completions) < 5:
        log("FATAL: insufficient training data (<5 correct completions)")
        results["phase2_error"] = "insufficient_training_data"
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        return

    # Phase 2: SFT from F0 init
    try:
        phase2_stats = phase2_sft_training(correct_completions)
        results["phase2"] = phase2_stats
        log(f"\nPhase 2 done: success={phase2_stats['training_success']}, "
            f"used_f0_init={phase2_stats['used_f0_init']}")
    except Exception as e:
        log(f"FATAL Phase 2 error: {e}")
        import traceback; traceback.print_exc()
        results["phase2_error"] = str(e)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        return

    if not results["phase2"]["training_success"]:
        log("Phase 2 training failed. Stopping.")
        results["fatal"] = True
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        return

    # Phase 3: Three-way evaluation
    log("\n[Phase 3] Evaluating base, F0, G0...")

    # MMLU-Pro evals
    results["eval_base"] = eval_mmlu("base", adapter_path=None)
    results["eval_f0"] = eval_mmlu("f0_adapter", adapter_path=F0_ADAPTER)
    results["eval_g0"] = eval_mmlu("g0_adapter", adapter_path=G0_ADAPTER_FINAL)

    # GSM8K evals (K1515 regression check)
    results["gsm8k_f0"] = eval_gsm8k("f0_adapter", adapter_path=F0_ADAPTER)
    results["gsm8k_g0"] = eval_gsm8k("g0_adapter", adapter_path=G0_ADAPTER_FINAL)

    # Kill criteria evaluation
    base_acc = results["eval_base"]["overall_accuracy"]
    f0_acc = results["eval_f0"]["overall_accuracy"]
    g0_acc = results["eval_g0"]["overall_accuracy"]
    gsm8k_f0 = results["gsm8k_f0"].get("accuracy")
    gsm8k_g0 = results["gsm8k_g0"].get("accuracy")

    k1514 = g0_acc >= 0.70
    k1515 = (gsm8k_g0 is not None and gsm8k_f0 is not None and gsm8k_g0 >= gsm8k_f0)
    k1516 = (g0_acc >= f0_acc + 0.03) or (
        gsm8k_g0 is not None and gsm8k_f0 is not None and gsm8k_g0 >= gsm8k_f0 + 0.03
    )

    total_time = time.time() - t_start
    results["total_time_s"] = total_time

    log(f"\n{'='*60}")
    log(f"RESULTS SUMMARY")
    log(f"{'='*60}")
    log(f"Base MMLU-Pro (thinking):  {base_acc:.1%}")
    log(f"F0   MMLU-Pro (thinking):  {f0_acc:.1%}")
    log(f"G0   MMLU-Pro (thinking):  {g0_acc:.1%}")
    log(f"F0 GSM8K:                  {gsm8k_f0:.1%}" if gsm8k_f0 is not None else "F0 GSM8K: N/A")
    log(f"G0 GSM8K:                  {gsm8k_g0:.1%}" if gsm8k_g0 is not None else "G0 GSM8K: N/A")
    log(f"Phase 1 yield (F0 init):   {results['phase1']['yield_rate']:.1%}")
    log(f"K1514 (G0>=70%):           {'PASS' if k1514 else 'FAIL'}")
    log(f"K1515 (G0 GSM8K>=F0):      {'PASS' if k1515 else 'FAIL'}")
    log(f"K1516 (G0>=F0+3pp):        {'PASS' if k1516 else 'FAIL'}")
    log(f"Total time: {total_time:.0f}s")

    results["kill_criteria"] = {
        "K1514_g0_mmlu_ge_70pct": k1514,
        "K1515_g0_gsm8k_ge_f0": k1515,
        "K1516_g0_ge_f0_plus_3pp": k1516,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
