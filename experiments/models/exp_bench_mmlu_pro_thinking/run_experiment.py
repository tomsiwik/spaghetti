#!/usr/bin/env python3
"""
BENCH: MMLU-Pro Baseline + Pierre Adapted
Google target: 69.4%

MMLU-Pro evaluation WITH THINKING MODE enabled.
Finding #517: 42.3% without thinking, Google reports 69.4% with thinking.
Two phases: (1) base + thinking, (2) base + MCQ adapter + thinking.
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

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_FILE = EXPERIMENT_DIR.parent / "exp_bench_mmlu_pro" / "data" / "test.parquet"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# Best single adapter: math from T2.1 (82% delta on MMLU math, q_proj LoRA r=6)
MATH_ADAPTER = Path(__file__).parent.parent / "exp_p1_t2_single_domain_training" / "adapters" / "math"

LIMIT_PER_CAT = 3 if IS_SMOKE else 100
OPTION_LETTERS = "ABCDEFGHIJ"


def log(msg):
    print(msg, flush=True)


def load_data(limit_per_cat):
    """Load MMLU-Pro test set from parquet, sample per category."""
    df = pd.read_parquet(DATA_FILE)
    if limit_per_cat:
        sampled = []
        for cat in sorted(df["category"].unique()):
            cat_df = df[df["category"] == cat]
            n = min(limit_per_cat, len(cat_df))
            sampled.append(cat_df.sample(n=n, random_state=42))
        df = pd.concat(sampled, ignore_index=True)
    log(f"  Loaded {len(df)} questions across {df['category'].nunique()} categories")
    return df


def format_prompt(row, tokenizer):
    """Format MMLU-Pro question with chat template."""
    options = row["options"]
    option_text = "\n".join(
        f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
    )
    content = (
        f"The following is a multiple choice question. "
        f"Think step by step, then answer with ONLY the letter of the correct option "
        f"(A through {OPTION_LETTERS[len(options)-1]}).\n\n"
        f"Question: {row['question']}\n\n"
        f"Options:\n{option_text}\n\n"
        f"Answer:"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def strip_thinking(response):
    """Remove thinking tokens from Gemma 4 response.

    Returns (answer_text, thinking_len) where thinking_len is char count of thinking.
    """
    if not response:
        return response, 0
    thinking_len = 0
    # Gemma 4 thinking: <|channel>thought\n...<channel|>
    m = re.search(r'<\|channel>thought.*?<channel\|>', response, flags=re.DOTALL)
    if m:
        thinking_len = len(m.group(0))
    cleaned = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    return cleaned.strip(), thinking_len


def parse_answer(response):
    """Extract answer letter from model response (after stripping thinking).

    Returns (letter, thinking_chars).
    """
    if not response:
        return None, 0
    answer_text, thinking_chars = strip_thinking(response)
    response = answer_text
    # Direct single letter
    if len(response) == 1 and response.upper() in OPTION_LETTERS:
        return response.upper(), thinking_chars
    # Starts with letter (possibly followed by punctuation)
    m = re.match(r"^([A-J])[.\s:)\-,]", response)
    if m:
        return m.group(1), thinking_chars
    # "The answer is X" pattern
    m = re.search(r"(?:answer|correct)\s+(?:is\s+)?([A-J])", response, re.IGNORECASE)
    if m:
        return m.group(1).upper(), thinking_chars
    # Last letter in answer (thinking may mention options — use final choice)
    last_letter = None
    for ch in response:
        if ch.upper() in OPTION_LETTERS:
            last_letter = ch.upper()
    if last_letter:
        return last_letter, thinking_chars
    return None, thinking_chars


def evaluate(df, model, tokenizer, label="eval"):
    """Evaluate all questions using in-process generation."""
    total_correct = 0
    total_count = 0
    total_errors = 0
    total_thinking_chars = 0
    total_response_chars = 0
    category_results = {}
    t0 = time.time()

    categories = sorted(df["category"].unique())
    for cat in categories:
        cat_df = df[df["category"] == cat]
        correct = 0
        errors = 0
        cat_thinking = 0
        cat_response = 0
        for idx, (_, row) in enumerate(cat_df.iterrows()):
            prompt = format_prompt(row, tokenizer)
            try:
                response = generate(model, tokenizer, prompt=prompt, max_tokens=2048)
            except Exception as e:
                response = f"ERROR: {e}"
                errors += 1

            predicted, thinking_chars = parse_answer(response)
            cat_thinking += thinking_chars
            cat_response += len(response) if response else 0
            gold = row["answer"]
            if predicted == gold:
                correct += 1
            if predicted is None:
                errors += 1

            if (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (total_count + idx + 1) / elapsed if elapsed > 0 else 0
                log(f"    [{label}] {cat} {idx+1}/{len(cat_df)}: "
                    f"{correct}/{idx+1} ({100*correct/(idx+1):.1f}%) | "
                    f"{rate:.1f} q/s | {elapsed:.0f}s")

        cat_acc = correct / len(cat_df) if len(cat_df) > 0 else 0
        category_results[cat] = {
            "correct": correct,
            "total": len(cat_df),
            "accuracy": round(cat_acc, 4),
            "errors": errors,
        }
        total_correct += correct
        total_count += len(cat_df)
        total_errors += errors
        total_thinking_chars += cat_thinking
        total_response_chars += cat_response
        elapsed = time.time() - t0
        log(f"  [{label}] {cat}: {correct}/{len(cat_df)} = {100*cat_acc:.1f}% "
            f"(cumulative: {100*total_correct/total_count:.1f}%, {elapsed:.0f}s)")

    overall_acc = total_correct / total_count if total_count > 0 else 0
    elapsed = time.time() - t0
    answer_chars = max(total_response_chars - total_thinking_chars, 1)
    thinking_ratio = total_thinking_chars / answer_chars if answer_chars > 0 else 0
    return {
        "overall_accuracy": round(overall_acc, 4),
        "total_correct": total_correct,
        "total_count": total_count,
        "total_errors": total_errors,
        "total_thinking_chars": total_thinking_chars,
        "total_response_chars": total_response_chars,
        "thinking_ratio": round(thinking_ratio, 1),
        "categories": category_results,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    log("=" * 70)
    log("BENCH: MMLU-Pro -- Base E4B vs Pierre Adapted (math adapter)")
    log(f"SMOKE_TEST={IS_SMOKE}, LIMIT_PER_CAT={LIMIT_PER_CAT}")
    log(f"Math adapter: {MATH_ADAPTER}")
    log(f"Adapter exists: {MATH_ADAPTER.exists()}")
    log("=" * 70)

    df = load_data(LIMIT_PER_CAT)
    results = {
        "experiment": "exp_bench_mmlu_pro_thinking",
        "smoke": IS_SMOKE,
        "limit_per_cat": LIMIT_PER_CAT,
        "total_questions": len(df),
        "model": MODEL_ID,
        "thinking_enabled": True,
    }

    # Phase 1: Base model
    log("\n[Phase 1] Base Gemma 4 E4B (4-bit)")
    model, tokenizer = load(MODEL_ID)
    base_eval = evaluate(df, model, tokenizer, label="base")
    results["base"] = base_eval
    log(f"\n  Phase 1 result: {base_eval['overall_accuracy']*100:.1f}% "
        f"({base_eval['total_correct']}/{base_eval['total_count']}) in {base_eval['elapsed_s']:.0f}s")

    # Free memory before Phase 2
    del model
    gc.collect()
    mx.metal.clear_cache()

    # Phase 2: Pierre adapted (math adapter)
    log("\n[Phase 2] Pierre Adapted (math adapter)")
    if not MATH_ADAPTER.exists():
        log(f"  SKIP: Math adapter not found at {MATH_ADAPTER}")
        results["adapted"] = {"status": "skipped", "reason": "adapter not found"}
    else:
        model, tokenizer = load(MODEL_ID, adapter_path=str(MATH_ADAPTER))
        adapted_eval = evaluate(df, model, tokenizer, label="adapted")
        results["adapted"] = adapted_eval
        log(f"\n  Phase 2 result: {adapted_eval['overall_accuracy']*100:.1f}% "
            f"({adapted_eval['total_correct']}/{adapted_eval['total_count']}) in {adapted_eval['elapsed_s']:.0f}s")
        del model
        gc.collect()
        mx.metal.clear_cache()

    # Summary
    base_acc = results["base"]["overall_accuracy"]
    adapted_acc = results.get("adapted", {}).get("overall_accuracy")
    delta = (adapted_acc - base_acc) if adapted_acc is not None else None

    total_time = results["base"]["elapsed_s"] + results.get("adapted", {}).get("elapsed_s", 0)
    summary = {
        "base_accuracy": base_acc,
        "adapted_accuracy": adapted_acc,
        "delta_pp": round(delta * 100, 1) if delta is not None else None,
        "total_elapsed_s": round(total_time, 1),
        "total_elapsed_h": round(total_time / 3600, 2),
    }

    # Kill criteria
    google_target = 0.694
    k1_pass = abs(base_acc - google_target) <= 0.05
    k2_pass = delta is not None and delta >= 0.02
    k3_pass = total_time < 6 * 3600

    kill = {
        "K1411": {"pass": k1_pass, "detail": f"Base={base_acc:.4f}, target={google_target}, gap={abs(base_acc-google_target)*100:.1f}pp"},
        "K1412": {"pass": k2_pass, "detail": f"Delta={delta*100:.1f}pp" if delta else "Delta=N/A"},
        "K1413": {"pass": k3_pass, "detail": f"Elapsed={total_time/3600:.2f}h"},
    }
    summary["kill_criteria"] = kill
    results["summary"] = summary

    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)
    log(f"  Base: {base_acc*100:.1f}%")
    log(f"  Adapted: {adapted_acc*100:.1f}%" if adapted_acc else "  Adapted: N/A")
    log(f"  Delta: {summary['delta_pp']}pp" if delta else "  Delta: N/A")
    log(f"  Time: {summary['total_elapsed_h']}h")
    log(f"  K1411 (base within 5pp of 69.4%): {'PASS' if k1_pass else 'FAIL'}")
    log(f"  K1412 (adapted >= base + 2pp): {'PASS' if k2_pass else 'FAIL'}")
    log(f"  K1413 (< 6h): {'PASS' if k3_pass else 'FAIL'}")

    # Per-category comparison
    if adapted_acc is not None:
        log("\n  Per-category (base -> adapted):")
        for cat in sorted(results["base"]["categories"]):
            b = results["base"]["categories"][cat]["accuracy"]
            a = results["adapted"]["categories"][cat]["accuracy"]
            d = (a - b) * 100
            log(f"    {cat:20s}: {b*100:5.1f}% -> {a*100:5.1f}% ({d:+.1f}pp)")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
