#!/usr/bin/env python3
"""
exp_g4_cot_vs_direct_mmlu_pro — KC1598

Compare CoT (enable_thinking=True) vs direct (enable_thinking=False) on Gemma 4
E4B 4-bit for reasoning-heavy MMLU-Pro subjects (MATH, Physics), matched N=30
per subject, same sampled questions across both conditions.

Pass if pooled delta >= 8pp (K1598), MATH-only delta >= 8pp (K1598-robust),
wall time <= 45 min (K1598-runtime).

Scaffold adapted from exp_bench_mmlu_pro_thinking/run_experiment.py. Single
load(); chat template swap per phase.
"""

import gc
import json
import os
import re
import time
from pathlib import Path

import mlx.core as mx
import pandas as pd
from mlx_lm import generate, load

try:
    import mlx_lm
    MLX_LM_VERSION = getattr(mlx_lm, "__version__", "unknown")
except Exception:
    MLX_LM_VERSION = "unknown"

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_FILE = EXPERIMENT_DIR.parent / "exp_bench_mmlu_pro" / "data" / "test.parquet"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

REASONING_SUBJECTS = ["math", "physics"]
N_PER_SUBJECT = 3 if IS_SMOKE else 30
RANDOM_STATE = 42
MAX_TOKENS_COT = 2048
MAX_TOKENS_DIRECT = 16
OPTION_LETTERS = "ABCDEFGHIJ"


def log(msg):
    print(msg, flush=True)


def load_matched_data():
    """Same sampled subset for both conditions (matched pairs)."""
    df = pd.read_parquet(DATA_FILE)
    # MMLU-Pro categories are lowercase in the parquet
    df["category_lower"] = df["category"].str.lower()
    sampled = []
    for cat in REASONING_SUBJECTS:
        cat_df = df[df["category_lower"] == cat]
        if len(cat_df) == 0:
            log(f"  WARN: no rows for subject {cat!r}. Available: {sorted(df['category_lower'].unique())}")
            continue
        n = min(N_PER_SUBJECT, len(cat_df))
        sampled.append(cat_df.sample(n=n, random_state=RANDOM_STATE))
    out = pd.concat(sampled, ignore_index=True)
    log(f"  Loaded {len(out)} questions across {out['category'].nunique()} categories")
    return out


def format_prompt(row, tokenizer, *, enable_thinking):
    options = row["options"]
    option_text = "\n".join(
        f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
    )
    if enable_thinking:
        instruction = (
            "The following is a multiple choice question. "
            "Think step by step, then answer with ONLY the letter of the correct option "
            f"(A through {OPTION_LETTERS[len(options)-1]})."
        )
    else:
        instruction = (
            "The following is a multiple choice question. "
            "Answer with ONLY the letter of the correct option "
            f"(A through {OPTION_LETTERS[len(options)-1]}). Do not explain."
        )
    content = (
        f"{instruction}\n\n"
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
    if not response:
        return None, 0
    answer_text, thinking_chars = strip_thinking(response)
    response = answer_text
    if len(response) == 1 and response.upper() in OPTION_LETTERS:
        return response.upper(), thinking_chars
    m = re.match(r"^([A-J])[.\s:)\-,]", response)
    if m:
        return m.group(1), thinking_chars
    m = re.search(r"(?:answer|correct)\s+(?:is\s+)?([A-J])", response, re.IGNORECASE)
    if m:
        return m.group(1).upper(), thinking_chars
    last_letter = None
    for ch in response:
        if ch.upper() in OPTION_LETTERS:
            last_letter = ch.upper()
    if last_letter:
        return last_letter, thinking_chars
    return None, thinking_chars


def evaluate_phase(df, model, tokenizer, *, enable_thinking, label):
    max_tokens = MAX_TOKENS_COT if enable_thinking else MAX_TOKENS_DIRECT
    cats = sorted(df["category_lower"].unique())
    per_cat = {}
    thinking_chars_correct = 0
    thinking_correct_count = 0
    t0 = time.time()

    for cat in cats:
        cat_df = df[df["category_lower"] == cat]
        correct = 0
        errors = 0
        cat_think_chars = 0
        for idx, (_, row) in enumerate(cat_df.iterrows()):
            prompt = format_prompt(row, tokenizer, enable_thinking=enable_thinking)
            try:
                response = generate(
                    model, tokenizer, prompt=prompt, max_tokens=max_tokens
                )
            except Exception as e:
                response = f"ERROR: {e}"
                errors += 1
            predicted, thinking_chars = parse_answer(response)
            cat_think_chars += thinking_chars
            gold = row["answer"]
            if predicted == gold:
                correct += 1
                if enable_thinking:
                    thinking_chars_correct += thinking_chars
                    thinking_correct_count += 1
            if predicted is None:
                errors += 1
            if (idx + 1) % 5 == 0:
                elapsed = time.time() - t0
                log(
                    f"    [{label}] {cat} {idx+1}/{len(cat_df)}: "
                    f"{correct}/{idx+1} ({100*correct/(idx+1):.1f}%) "
                    f"| {elapsed:.0f}s"
                )
        per_cat[cat] = {
            "correct": correct,
            "total": len(cat_df),
            "accuracy": correct / len(cat_df) if len(cat_df) else 0,
            "errors": errors,
            "thinking_chars_total": cat_think_chars,
        }

    total_correct = sum(v["correct"] for v in per_cat.values())
    total = sum(v["total"] for v in per_cat.values())
    mean_think_correct = (
        thinking_chars_correct / thinking_correct_count
        if thinking_correct_count
        else 0
    )
    return {
        "overall_accuracy": total_correct / total if total else 0,
        "total_correct": total_correct,
        "total": total,
        "categories": per_cat,
        "elapsed_s": time.time() - t0,
        "mean_thinking_chars_per_correct": mean_think_correct,
    }


def main():
    log("=" * 70)
    log("exp_g4_cot_vs_direct_mmlu_pro  —  KC1598")
    log(f"  Model: {MODEL_ID}")
    log(f"  mlx_lm version: {MLX_LM_VERSION}")
    log(f"  Subjects: {REASONING_SUBJECTS}")
    log(f"  N/subject: {N_PER_SUBJECT}  (smoke={IS_SMOKE})")
    log(f"  Random state: {RANDOM_STATE}")
    log("=" * 70)

    df = load_matched_data()

    log("\n[Load] Gemma 4 E4B 4-bit")
    t_load_start = time.time()
    model, tokenizer = load(MODEL_ID)
    t_load = time.time() - t_load_start
    log(f"  Loaded in {t_load:.1f}s")

    # Phase 1: direct (fastest first, so a thinking-mode crash doesn't lose baseline)
    log("\n[Phase 1] enable_thinking=False  (direct)")
    direct = evaluate_phase(df, model, tokenizer, enable_thinking=False, label="direct")
    log(
        f"  direct overall: {direct['overall_accuracy']*100:.1f}% "
        f"({direct['total_correct']}/{direct['total']}) in {direct['elapsed_s']:.0f}s"
    )

    mx.clear_cache()
    gc.collect()

    # Phase 2: cot (thinking)
    log("\n[Phase 2] enable_thinking=True   (cot)")
    cot = evaluate_phase(df, model, tokenizer, enable_thinking=True, label="cot")
    log(
        f"  cot    overall: {cot['overall_accuracy']*100:.1f}% "
        f"({cot['total_correct']}/{cot['total']}) in {cot['elapsed_s']:.0f}s"
    )

    # Pooled & per-category deltas
    per_subj_delta = {}
    for cat in sorted(set(direct["categories"]) | set(cot["categories"])):
        d_acc = direct["categories"].get(cat, {}).get("accuracy", 0.0)
        c_acc = cot["categories"].get(cat, {}).get("accuracy", 0.0)
        per_subj_delta[cat] = {
            "direct": d_acc,
            "cot": c_acc,
            "delta_pp": (c_acc - d_acc) * 100,
            "n": cot["categories"].get(cat, {}).get("total", 0),
        }

    pooled_cot_total = sum(v["correct"] for v in cot["categories"].values())
    pooled_direct_total = sum(v["correct"] for v in direct["categories"].values())
    pooled_n = sum(v["total"] for v in cot["categories"].values())
    pooled_delta_pp = ((pooled_cot_total - pooled_direct_total) / pooled_n) * 100 if pooled_n else 0

    math_delta_pp = per_subj_delta.get("math", {}).get("delta_pp", 0.0)
    total_elapsed_s = direct["elapsed_s"] + cot["elapsed_s"] + t_load
    runtime_min = total_elapsed_s / 60

    # Kill criteria
    k_main_pass = pooled_delta_pp >= 8.0
    k_robust_pass = math_delta_pp >= 8.0
    k_runtime_pass = runtime_min <= 45.0
    all_pass = k_main_pass and k_robust_pass and k_runtime_pass

    kill_criteria = {
        "K1598": {
            "pass": bool(k_main_pass),
            "threshold_pp": 8.0,
            "measured_pp": round(pooled_delta_pp, 2),
            "detail": f"Pooled delta (CoT-direct) = {pooled_delta_pp:+.2f}pp on N={pooled_n}",
        },
        "K1598-robust": {
            "pass": bool(k_robust_pass),
            "threshold_pp": 8.0,
            "measured_pp": round(math_delta_pp, 2),
            "detail": f"MATH delta = {math_delta_pp:+.2f}pp",
        },
        "K1598-runtime": {
            "pass": bool(k_runtime_pass),
            "threshold_min": 45.0,
            "measured_min": round(runtime_min, 2),
            "detail": f"Wall time {runtime_min:.2f} min",
        },
    }

    verdict = "supported" if all_pass else ("killed" if not k_main_pass else "supported_with_caveat")
    if IS_SMOKE:
        verdict = "provisional"

    results = {
        "experiment": "exp_g4_cot_vs_direct_mmlu_pro",
        "is_smoke": IS_SMOKE,
        "model": MODEL_ID,
        "mlx_lm_version": MLX_LM_VERSION,
        "subjects": REASONING_SUBJECTS,
        "n_per_subject": N_PER_SUBJECT,
        "random_state": RANDOM_STATE,
        "direct": direct,
        "cot": cot,
        "per_subject_delta": per_subj_delta,
        "pooled_delta_pp": round(pooled_delta_pp, 2),
        "kill_criteria": kill_criteria,
        "all_pass": bool(all_pass),
        "verdict": verdict,
        "elapsed_load_s": round(t_load, 2),
        "elapsed_direct_s": round(direct["elapsed_s"], 2),
        "elapsed_cot_s": round(cot["elapsed_s"], 2),
        "elapsed_total_s": round(total_elapsed_s, 2),
    }

    log("\n" + "=" * 70)
    log("RESULTS")
    log("=" * 70)
    for cat, d in per_subj_delta.items():
        log(f"  {cat:10s}: direct={d['direct']*100:5.1f}% -> cot={d['cot']*100:5.1f}% | delta={d['delta_pp']:+.2f}pp")
    log(f"  POOLED delta = {pooled_delta_pp:+.2f}pp  (N_pool={pooled_n})")
    log(f"  Wall time    = {runtime_min:.2f} min")
    for kid, kv in kill_criteria.items():
        status = "PASS" if kv["pass"] else "FAIL"
        log(f"  {kid}: {status} — {kv['detail']}")
    log(f"  VERDICT: {verdict}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
