#!/usr/bin/env python3
"""
exp_followup_budget_forcing_baseline_fix — F#530 identical-config replication.

Reproduces the exact base+thinking evaluation from exp_p10_mcq_adapter_training
that produced Finding #530 (62.1% MMLU-Pro, N=280, seed=42). Diagnoses the
15.4pp gap vs exp_p10_budget_forcing (46.7%, N=210).

Kill criteria (pre-registered in MATH.md; F#666-compliant target-direct):
  K1568: overall_accuracy ∈ Wilson-95% CI of 0.621 at n=280 → [0.564, 0.675]
  K1569: math_accuracy ∈ Wilson-95% CI of 0.85 at n=20 → [0.660, 1.000]
  K1570: thinking chars/question ∈ [1893, 3515]  (±30% of F#530's ~2704)

Grounded by:
  Finding #530 — base+thinking 62.1% MMLU-Pro, 757251 chars / 280 q
  F#666 — target-direct KC is self-gated (not a proxy)
"""

import gc
import json
import os
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
from mlx_lm import generate, load

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_FILE = EXPERIMENT_DIR.parent / "exp_bench_mmlu_pro" / "data" / "test.parquet"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42
OPTION_LETTERS = "ABCDEFGHIJ"
PER_CAT = 2 if IS_SMOKE else 20  # F#530 used 20 per category
MAX_TOKENS = 2048

# Wilson 95% CI around F#530's reference proportions (pre-registered, from MATH.md)
K1568_LO, K1568_HI = 0.564, 0.675   # overall_accuracy, n=280
K1569_LO, K1569_HI = 0.660, 1.000   # math_accuracy, n=20
K1570_LO, K1570_HI = 1893.0, 3515.0 # thinking chars/question


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB peak={peak:.2f}GB")


def load_and_split_data():
    """Replicate F#530 data pipeline EXACTLY.

    From experiments/models/exp_p10_mcq_adapter_training/run_experiment.py:load_and_split_data.
    80/20 stratified split with np.random.RandomState(SEED).permutation per category.
    Return the eval_df (20% hold-out).
    """
    df = pd.read_parquet(DATA_FILE)
    eval_dfs = []
    rng = np.random.RandomState(SEED)
    for cat in sorted(df["category"].unique()):
        cat_df = df[df["category"] == cat].copy()
        idx = rng.permutation(len(cat_df))
        split = int(0.8 * len(cat_df))
        eval_dfs.append(cat_df.iloc[idx[split:]])
    eval_df = pd.concat(eval_dfs, ignore_index=True)
    log(f"  Eval split: {len(eval_df)} questions across {eval_df['category'].nunique()} categories")
    return eval_df


def format_mmlu_prompt(row, tokenizer, enable_thinking=True):
    """Replicate F#530 prompt EXACTLY.

    From exp_p10_mcq_adapter_training:format_mmlu_prompt.
    NOTE: 'Do not explain.' — not 'Think step by step'. This is the prompt F#530 used.
    """
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
    """Remove thinking tokens. Identical to F#530 regex set."""
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
    """Extract answer letter. Identical to F#530 parse order."""
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
    last_letter = None
    for ch in answer_text:
        if ch.upper() in OPTION_LETTERS:
            last_letter = ch.upper()
    if last_letter:
        return last_letter, thinking_chars
    return None, thinking_chars


def evaluate(eval_df, model, tokenizer, per_cat=PER_CAT):
    """Evaluate on F#530 sampling protocol: per-cat .sample(n=per_cat, random_state=SEED)."""
    total_correct = 0
    total_count = 0
    total_thinking_chars = 0
    category_results = {}
    per_question = []
    t0 = time.time()

    categories = sorted(eval_df["category"].unique())
    for cat in categories:
        cat_df = eval_df[eval_df["category"] == cat]
        if per_cat and per_cat < len(cat_df):
            cat_df = cat_df.sample(n=per_cat, random_state=SEED)
        correct = 0
        cat_thinking = 0
        for idx, (_, row) in enumerate(cat_df.iterrows()):
            prompt = format_mmlu_prompt(row, tokenizer, enable_thinking=True)
            q_t0 = time.time()
            try:
                response = generate(
                    model, tokenizer, prompt=prompt,
                    max_tokens=MAX_TOKENS, verbose=False,
                )
            except Exception as e:
                response = f"ERROR: {e}"
            q_time = time.time() - q_t0
            predicted, thinking_chars = parse_answer(response)
            cat_thinking += thinking_chars
            is_correct = predicted == row["answer"]
            if is_correct:
                correct += 1
            per_question.append({
                "category": cat,
                "gold": row["answer"],
                "predicted": predicted,
                "correct": is_correct,
                "thinking_chars": thinking_chars,
                "time_s": round(q_time, 2),
            })

        cat_acc = correct / len(cat_df) if len(cat_df) > 0 else 0
        category_results[cat] = {
            "correct": correct,
            "total": len(cat_df),
            "accuracy": round(cat_acc, 4),
            "thinking_chars": cat_thinking,
        }
        total_correct += correct
        total_count += len(cat_df)
        total_thinking_chars += cat_thinking
        elapsed = time.time() - t0
        log(f"  {cat}: {correct}/{len(cat_df)} = {100*cat_acc:.1f}% "
            f"(cumulative: {100*total_correct/total_count:.1f}%, {elapsed:.0f}s)")

    overall_acc = total_correct / total_count if total_count > 0 else 0
    return {
        "overall_accuracy": round(overall_acc, 4),
        "total_correct": total_correct,
        "total_count": total_count,
        "total_thinking_chars": total_thinking_chars,
        "thinking_chars_per_q": round(total_thinking_chars / max(total_count, 1), 1),
        "categories": category_results,
        "elapsed_s": round(time.time() - t0, 1),
        "per_question": per_question,
    }


def evaluate_kcs(eval_out):
    """Pre-registered KC evaluation (from MATH.md)."""
    overall = eval_out["overall_accuracy"]
    math_acc = eval_out["categories"].get("math", {}).get("accuracy", None)
    chars_per_q = eval_out["thinking_chars_per_q"]

    kcs = {}
    kcs["K1568"] = {
        "description": "overall_accuracy in Wilson-95% CI of 0.621 at n=280 -> [0.564, 0.675]",
        "measured": overall,
        "threshold": [K1568_LO, K1568_HI],
        "pass": (K1568_LO <= overall <= K1568_HI),
        "role": "target",
    }
    if math_acc is not None:
        kcs["K1569"] = {
            "description": "math_accuracy in Wilson-95% CI of 0.85 at n=20 -> [0.660, 1.000]",
            "measured": math_acc,
            "threshold": [K1569_LO, K1569_HI],
            "pass": (K1569_LO <= math_acc <= K1569_HI),
            "role": "secondary_target",
        }
    else:
        kcs["K1569"] = {
            "description": "math_accuracy in Wilson-95% CI of 0.85 at n=20",
            "measured": None,
            "threshold": [K1569_LO, K1569_HI],
            "pass": False,
            "role": "secondary_target",
            "note": "math category not found in eval",
        }
    kcs["K1570"] = {
        "description": "thinking chars/question in +/-30% of 2704 -> [1893, 3515]",
        "measured": chars_per_q,
        "threshold": [K1570_LO, K1570_HI],
        "pass": (K1570_LO <= chars_per_q <= K1570_HI),
        "role": "instrumentation",
    }
    return kcs


def decide_verdict(kcs, is_smoke):
    """F#530-replication verdict matrix (from MATH.md §Kill Criteria)."""
    if is_smoke:
        return "PROVISIONAL", "smoke run (N per category reduced); non-binding."
    k1568 = kcs["K1568"]["pass"]
    overall = kcs["K1568"]["measured"]
    if k1568:
        return "SUPPORTED", (
            f"K1568 PASS: overall={overall:.4f} in [{K1568_LO}, {K1568_HI}]. "
            f"F#530 reproduces; 15.4pp gap vs exp_p10_budget_forcing attributable to config drift."
        )
    direction = "below" if overall < K1568_LO else "above"
    return "KILLED", (
        f"K1568 FAIL: overall={overall:.4f} {direction} [{K1568_LO}, {K1568_HI}]. "
        f"H_repl refuted; 62.1% does not reproduce under identical config on this machine."
    )


def main():
    log("=" * 70)
    log("exp_followup_budget_forcing_baseline_fix — F#530 identical-config replication")
    log(f"SMOKE_TEST={IS_SMOKE}, PER_CAT={PER_CAT}, MAX_TOKENS={MAX_TOKENS}, SEED={SEED}")
    log("=" * 70)

    eval_df = load_and_split_data()

    log("\n[Loading model]")
    log_memory("pre-load")
    model, tokenizer = load(MODEL_ID)
    log_memory("post-load")

    log("\n[Evaluating base+thinking]")
    eval_out = evaluate(eval_df, model, tokenizer, per_cat=PER_CAT)
    log_memory("post-eval")

    log("\n[Evaluating KCs]")
    kcs = evaluate_kcs(eval_out)
    for k, v in kcs.items():
        status = "PASS" if v["pass"] else "FAIL"
        log(f"  {k}: {status} — measured={v['measured']} threshold={v['threshold']}")

    verdict, evidence = decide_verdict(kcs, IS_SMOKE)
    log(f"\nVERDICT: {verdict}")
    log(f"EVIDENCE: {evidence}")

    # Persist
    results = {
        "experiment": "exp_followup_budget_forcing_baseline_fix",
        "smoke": IS_SMOKE,
        "seed": SEED,
        "per_cat": PER_CAT,
        "max_tokens": MAX_TOKENS,
        "model": MODEL_ID,
        "total_questions": eval_out["total_count"],
        "overall_accuracy": eval_out["overall_accuracy"],
        "thinking_chars_per_q": eval_out["thinking_chars_per_q"],
        "categories": eval_out["categories"],
        "kill_criteria": kcs,
        "verdict": verdict,
        "evidence": evidence,
        "all_pass": all(v["pass"] for v in kcs.values()),
        "elapsed_s": eval_out["elapsed_s"],
        "reference_finding_530": {
            "overall_accuracy": 0.621,
            "total_questions": 280,
            "thinking_chars_per_q": 2704,
        },
        "per_question": eval_out["per_question"],
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n[wrote] {RESULTS_FILE}")


if __name__ == "__main__":
    main()
