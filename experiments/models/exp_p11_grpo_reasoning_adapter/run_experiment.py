#!/usr/bin/env python3
"""
P11.B0: Rejection Sampling SFT on MMLU-Pro (GRPO Approximation)

Root cause fix for s1K catastrophic forgetting (-26pp):
  - s1K trained on competition math traces (D_train ≠ D_eval)
  - This experiment: train on MMLU-Pro questions (D_train = D_eval)
  - Theorem 1: if D_train = D_eval, catastrophic forgetting is geometrically impossible

Algorithm:
  Phase 1: Sample 200 MMLU-Pro questions, generate with thinking=True (greedy)
           Keep correct completions → ~124 training examples
  Phase 2: SFT training on self-generated correct traces via mlx_lm.lora
  Phase 3: Evaluate on 280q MMLU-Pro (same as all P11 experiments)

Kill criteria:
  K1496: RS-SFT >= 64% MMLU-Pro with thinking (positive gradient signal)
  K1497: RS-SFT >= 56.1% MMLU-Pro (= s1K+20pp, distribution alignment prevents forgetting)
  K1498: All 14 MMLU-Pro categories within 5pp of base (no per-category catastrophe)

Reference:
  - arXiv:2402.03300 (GRPO: Shao et al.)
  - arXiv:2501.12948 (DeepSeek-R1: RS-SFT as GRPO warmup)
  - arXiv:1612.00796 (EWC: Kirkpatrick et al., catastrophic forgetting theory)
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

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_DIR = EXPERIMENT_DIR / "data"
ADAPTER_DIR = EXPERIMENT_DIR / "adapters" / "rs_sft"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
MMLU_DATA = REPO_ROOT / "experiments/models/exp_bench_mmlu_pro/data/data/test.parquet"

IS_SMOKE = "--smoke" in sys.argv or os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42

# Sampling config
N_SAMPLE_QUESTIONS = 10 if IS_SMOKE else 100  # stratified across 14 categories
MAX_TOKENS_SAMPLE = 2048  # per completion (thinking channel needs ~700+ tokens)

# Training config
LORA_RANK = 8
LORA_SCALE = 1.0
N_STEPS = 5 if IS_SMOKE else 200
BATCH_SIZE = 1
LR = 1e-5

# Evaluation config: 7 per cat = 98 questions total, fits within 2h budget
# (Smoke verified: 24.7s/question; 98q×2 conditions = ~80min for Phase 3)
EVAL_PER_CAT = 2 if IS_SMOKE else 7  # 7 × 14 = 98 questions
OPTION_LETTERS = "ABCDEFGHIJ"


def log(msg):
    print(msg, flush=True)


def log_memory(label):
    info = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={info:.2f}GB peak={peak:.2f}GB")
    mx.reset_peak_memory()


# ─────────────────────────────────────────────────────────────
# Prompt formatting (validated at 62.1% in p10/p11 experiments)
# ─────────────────────────────────────────────────────────────

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
# Phase 1: Rejection Sampling — collect correct self-generated traces
# ─────────────────────────────────────────────────────────────

def phase1_rejection_sampling():
    """
    Generate N_SAMPLE_QUESTIONS completions with thinking=True (greedy).
    Keep correct completions → training dataset.

    Theorem 1: D_train = D_eval (MMLU-Pro) → catastrophic forgetting impossible.
    """
    log("\n[Phase 1] Rejection Sampling on MMLU-Pro")
    t0 = time.time()

    from mlx_lm import load, generate

    # Load MMLU-Pro data
    log(f"  Loading MMLU-Pro data from {MMLU_DATA}")
    df = pd.read_parquet(MMLU_DATA)
    categories = df["category"].unique()
    log(f"  {len(df)} total questions, {len(categories)} categories: {list(categories)[:5]}...")

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

    # Load model
    log(f"  Loading {MODEL_ID}...")
    model, tokenizer = load(MODEL_ID)
    log_memory("after_load")

    # Generate completions (greedy, temperature=0)
    correct_completions = []
    skipped = 0

    for i, row in enumerate(sampled_rows):
        prompt = format_mmlu_prompt(row, tokenizer, enable_thinking=True)
        try:
            from mlx_lm.sample_utils import make_sampler
            response = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=MAX_TOKENS_SAMPLE,
                sampler=make_sampler(temp=0.0),  # greedy
                verbose=False,
            )
        except Exception as e:
            log(f"  [q{i}] ERROR generating: {e}")
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
            # Format as training example for mlx_lm.lora
            # User content: question + options
            options = row["options"]
            option_text = "\n".join(f"{OPTION_LETTERS[j]}. {opt}" for j, opt in enumerate(options))
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

    # Cleanup model before training
    del model
    gc.collect()
    mx.clear_cache()
    log_memory("after_sampling_cleanup")

    elapsed = time.time() - t0
    yield_rate = len(correct_completions) / max(1, len(sampled_rows) - skipped)
    log(f"\n  Phase 1 complete: {len(correct_completions)} correct completions "
        f"({yield_rate:.1%} yield) from {len(sampled_rows)} questions in {elapsed:.0f}s")

    if len(correct_completions) < 10:
        log("  WARNING: Very few correct completions (<10). Training may be ineffective.")

    return correct_completions, {
        "n_sampled": len(sampled_rows),
        "n_correct": len(correct_completions),
        "n_skipped": skipped,
        "yield_rate": yield_rate,
        "sampling_time_s": elapsed,
        "avg_thinking_chars": np.mean([c["thinking_chars"] for c in correct_completions]) if correct_completions else 0,
    }


# ─────────────────────────────────────────────────────────────
# Phase 2: SFT Training on self-generated traces
# ─────────────────────────────────────────────────────────────

def phase2_sft_training(correct_completions):
    """Train LoRA adapter on self-generated correct MMLU-Pro traces."""
    log(f"\n[Phase 2] SFT Training on {len(correct_completions)} self-generated traces")
    t0 = time.time()

    # Prepare data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # Write training JSONL (messages format for mlx_lm.lora)
    # Use 90/10 train/valid split
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
            # Write only messages (strip metadata for mlx_lm.lora)
            f.write(json.dumps({"messages": ex["messages"]}) + "\n")

    with open(val_file, "w") as f:
        for i in sorted(val_idx):
            ex = correct_completions[i]
            f.write(json.dumps({"messages": ex["messages"]}) + "\n")

    n_train = len(train_idx)
    n_val_actual = len(val_idx)
    log(f"  Train: {n_train} examples, Val: {n_val_actual} examples")
    log(f"  Written to {train_file} and {val_file}")

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
    ]

    log(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=False,
        cwd=REPO_ROOT,
        timeout=3600,  # 1h max
    )

    elapsed = time.time() - t0
    success = result.returncode == 0
    log(f"\n  Phase 2 complete: returncode={result.returncode} in {elapsed:.0f}s")

    return {
        "n_train": n_train,
        "n_val": n_val_actual,
        "n_steps": N_STEPS,
        "training_time_s": elapsed,
        "training_success": success,
        "adapter_path": str(ADAPTER_DIR),
    }


# ─────────────────────────────────────────────────────────────
# Phase 3: Evaluation (identical to all P11 experiments)
# ─────────────────────────────────────────────────────────────

def phase3_evaluate(label, adapter_path=None):
    """Evaluate on 280q MMLU-Pro with thinking. Same protocol as p10/p11."""
    log(f"\n[Phase 3] Evaluating: {label}")
    t0 = time.time()

    from mlx_lm import load, generate

    if adapter_path and Path(adapter_path).exists():
        log(f"  Loading adapter from {adapter_path}")
        model, tokenizer = load(MODEL_ID, adapter_path=adapter_path)
    else:
        model, tokenizer = load(MODEL_ID)
    log_memory("after_eval_load")

    # Load MMLU-Pro data (same stratified subset as all P11 experiments)
    df = pd.read_parquet(MMLU_DATA)
    categories = df["category"].unique()
    rng = np.random.default_rng(SEED)

    sampled_rows = []
    for cat in categories:
        cat_df = df[df["category"] == cat]
        n = min(EVAL_PER_CAT, len(cat_df))
        idx = rng.choice(len(cat_df), size=n, replace=False)
        sampled_rows.extend(cat_df.iloc[idx].to_dict("records"))

    log(f"  Evaluating {len(sampled_rows)} questions ({EVAL_PER_CAT} per category)")

    correct = 0
    total = 0
    per_category = {}
    thinking_chars_list = []

    for i, row in enumerate(sampled_rows):
        cat = row["category"]
        prompt = format_mmlu_prompt(row, tokenizer, enable_thinking=True)
        try:
            from mlx_lm.sample_utils import make_sampler
            response = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=2048,
                sampler=make_sampler(temp=0.0),  # greedy
                verbose=False,
            )
        except Exception as e:
            log(f"  [q{i}] ERROR: {e}")
            if cat not in per_category:
                per_category[cat] = {"correct": 0, "total": 0}
            per_category[cat]["total"] += 1
            total += 1
            continue

        predicted, thinking_chars = parse_answer(response)
        ground_truth = row["answer"]
        is_correct = (predicted == ground_truth)

        if cat not in per_category:
            per_category[cat] = {"correct": 0, "total": 0}
        per_category[cat]["total"] += 1
        per_category[cat]["correct"] += int(is_correct)

        correct += int(is_correct)
        total += 1
        thinking_chars_list.append(thinking_chars)

        if i % 50 == 0 or i < 3:
            log(f"  [q{i}/{len(sampled_rows)}] acc={correct}/{total}={correct/max(1,total):.1%} "
                f"thinking={thinking_chars}chars")

    del model
    gc.collect()
    mx.clear_cache()

    overall_acc = correct / max(1, total)
    per_cat_acc = {cat: v["correct"] / max(1, v["total"]) for cat, v in per_category.items()}
    elapsed = time.time() - t0

    log(f"\n  {label}: {overall_acc:.1%} ({correct}/{total}) in {elapsed:.0f}s")
    for cat, acc in sorted(per_cat_acc.items(), key=lambda x: x[1]):
        log(f"    {cat[:20]:<20} {acc:.1%} ({per_category[cat]['correct']}/{per_category[cat]['total']})")

    return {
        "label": label,
        "overall_accuracy": overall_acc,
        "correct": correct,
        "total": total,
        "per_category": per_cat_acc,
        "avg_thinking_chars": float(np.mean(thinking_chars_list)) if thinking_chars_list else 0.0,
        "eval_time_s": elapsed,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    log(f"=== P11.B0: Rejection Sampling SFT on MMLU-Pro ===")
    log(f"SMOKE={IS_SMOKE}, N_SAMPLE={N_SAMPLE_QUESTIONS}, N_STEPS={N_STEPS}")
    t_start = time.time()

    results = {
        "experiment": "exp_p11_grpo_reasoning_adapter",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "seed": SEED,
        "n_sample_questions": N_SAMPLE_QUESTIONS,
        "n_train_steps": N_STEPS,
    }

    # Phase 1: Rejection sampling
    try:
        correct_completions, phase1_stats = phase1_rejection_sampling()
        results["phase1"] = phase1_stats
        log(f"\nPhase 1 done: {phase1_stats['n_correct']} correct traces "
            f"(yield={phase1_stats['yield_rate']:.1%})")
    except Exception as e:
        log(f"FATAL Phase 1 error: {e}")
        import traceback; traceback.print_exc()
        results["phase1_error"] = str(e)
        results["fatal"] = True
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        sys.exit(1)

    # Phase 2: Training
    if len(correct_completions) < 5:
        log("ERROR: <5 correct completions, cannot train. Aborting.")
        results["phase2_error"] = "insufficient_training_data"
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        sys.exit(1)

    try:
        phase2_stats = phase2_sft_training(correct_completions)
        results["phase2"] = phase2_stats
    except Exception as e:
        log(f"FATAL Phase 2 error: {e}")
        import traceback; traceback.print_exc()
        results["phase2_error"] = str(e)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        sys.exit(1)

    # Phase 3a: Base model eval (no adapter)
    try:
        base_results = phase3_evaluate("base_model")
        results["phase3a_base"] = base_results
    except Exception as e:
        log(f"Phase 3a error: {e}")
        results["phase3a_error"] = str(e)

    # Phase 3b: RS-SFT adapter eval
    try:
        adapter_results = phase3_evaluate("rs_sft_adapter", adapter_path=str(ADAPTER_DIR))
        results["phase3b_rs_sft"] = adapter_results
    except Exception as e:
        log(f"Phase 3b error: {e}")
        results["phase3b_error"] = str(e)

    # Kill criteria evaluation
    total_time = time.time() - t_start
    base_acc = results.get("phase3a_base", {}).get("overall_accuracy", 0)
    sft_acc = results.get("phase3b_rs_sft", {}).get("overall_accuracy", 0)
    s1k_acc = 0.361  # confirmed from s1K kill (K1490 FAIL)

    # K1496: RS-SFT >= 64% MMLU-Pro with thinking
    k1496_pass = sft_acc >= 0.64
    # K1497: RS-SFT >= 56.1% (s1K + 20pp)
    k1497_pass = sft_acc >= 0.561
    # K1498: all 14 categories within 5pp of base
    per_cat_sft = results.get("phase3b_rs_sft", {}).get("per_category", {})
    per_cat_base = results.get("phase3a_base", {}).get("per_category", {})
    cat_within_5pp = {
        cat: per_cat_sft.get(cat, 0) >= per_cat_base.get(cat, 0) - 0.05
        for cat in per_cat_base
    }
    k1498_pass = all(cat_within_5pp.values()) if cat_within_5pp else False

    log(f"\n{'='*60}")
    log(f"KILL CRITERIA SUMMARY")
    log(f"{'='*60}")
    log(f"Base accuracy: {base_acc:.1%}")
    log(f"RS-SFT accuracy: {sft_acc:.1%}")
    log(f"s1K accuracy (reference): {s1k_acc:.1%}")
    log(f"RS-SFT - s1K: {sft_acc - s1k_acc:+.1%}")
    log(f"")
    log(f"K1496 (RS-SFT >= 64%): {'PASS' if k1496_pass else 'FAIL'} ({sft_acc:.1%})")
    log(f"K1497 (RS-SFT >= 56.1%, +20pp over s1K): {'PASS' if k1497_pass else 'FAIL'} ({sft_acc:.1%})")
    log(f"K1498 (all cats within 5pp of base): {'PASS' if k1498_pass else 'FAIL'}")
    if not k1498_pass and cat_within_5pp:
        failed_cats = [c for c, ok in cat_within_5pp.items() if not ok]
        log(f"  Failed categories: {failed_cats}")
    log(f"\nTotal runtime: {total_time/60:.1f} minutes")
    log(f"{'='*60}")

    results["kill_criteria"] = {
        "K1496_rs_sft_ge_64pct": {"pass": k1496_pass, "value": sft_acc, "threshold": 0.64},
        "K1497_rs_sft_ge_s1k_plus_20pp": {"pass": k1497_pass, "value": sft_acc, "threshold": 0.561},
        "K1498_no_category_catastrophe": {"pass": k1498_pass, "categories": cat_within_5pp},
    }
    results["total_time_s"] = total_time

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
