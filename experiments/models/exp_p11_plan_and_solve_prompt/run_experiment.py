#!/usr/bin/env python3
"""
P11.Z0: Plan-and-Solve Prompting on MMLU-Pro

Tests 3 prompt variants (with thinking enabled) against 62.1% baseline:
  P0: direct-answer (current baseline)
  P1: Plan-and-Solve (arXiv:2305.04091)
  P2: PS+ (Plan-and-Solve with self-check)

Kill criteria:
  K1529: Best prompt + thinking >= 64% MMLU-Pro (>= 2pp over 62.1% baseline)
  K1530: PS+ >= PS accuracy (self-check adds value)
  K1531: Best prompt output token count <= 2x direct-answer count
"""

import gc
import json
import os
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MMLU_PATH = REPO_ROOT / "experiments/models/exp_bench_mmlu_pro/data/test.parquet"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
SEED = 42
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
EVAL_PER_CAT = 2 if IS_SMOKE else 20
OPTION_LETTERS = "ABCDEFGHIJ"


def log(msg):
    print(msg, flush=True)


PROMPT_TEMPLATES = {
    "P0_direct": (
        "The following is a multiple choice question. "
        "Answer with ONLY the letter of the correct option "
        "(A through {max_letter}). "
        "Do not explain.\n\n"
        "Question: {question}\n\n"
        "Options:\n{options}\n\n"
        "Answer:"
    ),
    "P1_ps": (
        "The following is a multiple choice question. "
        "Let's first understand the problem and devise a plan to solve it. "
        "Then, let's carry out the plan and solve the problem step by step "
        "to get the correct answer.\n\n"
        "Question: {question}\n\n"
        "Options:\n{options}\n\n"
        "Answer:"
    ),
    "P2_ps_plus": (
        "The following is a multiple choice question. "
        "Let's first understand the problem and devise a plan to solve it. "
        "Then, let's carry out the plan, paying attention to not miss any "
        "calculations, and double-check the answer.\n\n"
        "Question: {question}\n\n"
        "Options:\n{options}\n\n"
        "Answer:"
    ),
}


def load_data():
    import pandas as pd
    df = pd.read_parquet(MMLU_PATH)
    sampled = []
    for cat in sorted(df["category"].unique()):
        cat_df = df[df["category"] == cat]
        n = min(EVAL_PER_CAT, len(cat_df))
        sampled.append(cat_df.sample(n=n, random_state=SEED))
    import pandas as pd
    result = pd.concat(sampled, ignore_index=True)
    log(f"  Loaded {len(result)} questions across {result['category'].nunique()} categories")
    return result


def format_prompt(row, tokenizer, template_name):
    """Format MMLU-Pro question with given prompt template, thinking enabled."""
    options = row["options"]
    option_text = "\n".join(
        f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
    )
    max_letter = OPTION_LETTERS[len(options) - 1]
    content = PROMPT_TEMPLATES[template_name].format(
        max_letter=max_letter,
        question=row["question"],
        options=option_text,
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def parse_answer(response):
    """Extract answer letter from model response (handles thinking traces)."""
    if not response:
        return None, 0
    # Count total output tokens (rough estimate via len)
    token_count = len(response.split())

    # Strip <think>...</think> block if present
    clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if not clean:
        clean = response

    # Direct single letter
    if len(clean) == 1 and clean.upper() in OPTION_LETTERS:
        return clean.upper(), token_count

    # Starts with letter (possibly followed by punctuation)
    m = re.match(r"^([A-J])[.\s:)\-,]", clean)
    if m:
        return m.group(1), token_count

    # "The answer is X" pattern
    m = re.search(r"(?:answer|correct)\s+(?:is\s+)?([A-J])\b", clean, re.IGNORECASE)
    if m:
        return m.group(1).upper(), token_count

    # Last occurrence of a standalone letter (final answer after reasoning)
    matches = list(re.finditer(r"\b([A-J])\b", clean))
    if matches:
        return matches[-1].group(1), token_count

    # First letter found anywhere
    for ch in clean:
        if ch.upper() in OPTION_LETTERS:
            return ch.upper(), token_count

    return None, token_count


def evaluate_prompt(df, model, tokenizer, template_name, label):
    """Evaluate one prompt variant across all categories."""
    total_correct = 0
    total_count = 0
    total_errors = 0
    total_tokens = 0
    category_results = {}
    t0 = time.time()

    categories = sorted(df["category"].unique())
    for cat in categories:
        cat_df = df[df["category"] == cat]
        correct = 0
        errors = 0
        cat_tokens = 0

        for idx, (_, row) in enumerate(cat_df.iterrows()):
            prompt = format_prompt(row, tokenizer, template_name)
            try:
                response = generate(
                    model, tokenizer, prompt=prompt,
                    max_tokens=2048,  # allow full thinking trace
                    verbose=False,
                )
            except Exception as e:
                response = f"ERROR: {e}"
                errors += 1

            predicted, tok_count = parse_answer(response)
            gold = row["answer"]
            if predicted == gold:
                correct += 1
            if predicted is None:
                errors += 1
            cat_tokens += tok_count

            if (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (total_count + idx + 1) / elapsed if elapsed > 0 else 0
                log(
                    f"    [{label}/{cat}] {idx+1}/{len(cat_df)}: "
                    f"{correct}/{idx+1} ({100*correct/(idx+1):.1f}%) | "
                    f"{rate:.1f} q/s"
                )

        cat_acc = correct / len(cat_df) if len(cat_df) > 0 else 0
        category_results[cat] = {
            "correct": correct,
            "total": len(cat_df),
            "accuracy": round(cat_acc, 4),
            "errors": errors,
            "avg_tokens": round(cat_tokens / len(cat_df), 1) if len(cat_df) > 0 else 0,
        }
        total_correct += correct
        total_count += len(cat_df)
        total_errors += errors
        total_tokens += cat_tokens
        elapsed = time.time() - t0
        log(
            f"  [{label}] {cat}: {correct}/{len(cat_df)} = {100*cat_acc:.1f}% "
            f"(cumul: {100*total_correct/total_count:.1f}%, {elapsed:.0f}s)"
        )

    overall_acc = total_correct / total_count if total_count > 0 else 0
    elapsed = time.time() - t0
    avg_tokens = total_tokens / total_count if total_count > 0 else 0
    return {
        "template": template_name,
        "overall_accuracy": round(overall_acc, 4),
        "total_correct": total_correct,
        "total_count": total_count,
        "total_errors": total_errors,
        "avg_output_tokens": round(avg_tokens, 1),
        "categories": category_results,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    log("=" * 70)
    log("P11.Z0: Plan-and-Solve Prompting on MMLU-Pro")
    log(f"SMOKE_TEST={IS_SMOKE}, EVAL_PER_CAT={EVAL_PER_CAT}")
    log(f"Model: {MODEL_ID}")
    log("=" * 70)

    import pandas as pd
    df = load_data()

    results = {
        "experiment": "exp_p11_plan_and_solve_prompt",
        "smoke": IS_SMOKE,
        "eval_per_cat": EVAL_PER_CAT,
        "total_questions": len(df),
        "model": MODEL_ID,
        "thinking_enabled": True,
    }

    log("\n[Load] Loading model (single load, reuse across all prompts)")
    model, tokenizer = load(MODEL_ID)

    prompt_results = {}
    for name in PROMPT_TEMPLATES:
        log(f"\n{'='*70}")
        log(f"[Eval] Prompt: {name}")
        log("=" * 70)
        res = evaluate_prompt(df, model, tokenizer, name, label=name)
        prompt_results[name] = res
        acc = res["overall_accuracy"]
        log(f"\n  [{name}] RESULT: {acc*100:.1f}% ({res['total_correct']}/{res['total_count']}) "
            f"avg_tokens={res['avg_output_tokens']:.0f} in {res['elapsed_s']:.0f}s")

    del model
    gc.collect()
    mx.metal.clear_cache()

    results["prompts"] = prompt_results

    # Kill criteria evaluation
    baseline = 0.621  # Finding #530
    p0_acc = prompt_results["P0_direct"]["overall_accuracy"]
    p1_acc = prompt_results["P1_ps"]["overall_accuracy"]
    p2_acc = prompt_results["P2_ps_plus"]["overall_accuracy"]
    best_acc = max(p0_acc, p1_acc, p2_acc)
    best_name = max(prompt_results, key=lambda k: prompt_results[k]["overall_accuracy"])

    p0_tokens = prompt_results["P0_direct"]["avg_output_tokens"]
    best_tokens = prompt_results[best_name]["avg_output_tokens"]
    token_ratio = best_tokens / p0_tokens if p0_tokens > 0 else float("inf")

    k1529_pass = best_acc >= (baseline + 0.02)  # >= 64.1%
    k1530_pass = p2_acc >= p1_acc               # PS+ >= PS
    k1531_pass = token_ratio <= 2.0             # token count <= 2x

    kill = {
        "K1529": {
            "pass": k1529_pass,
            "detail": f"best={best_name} {best_acc*100:.1f}% vs baseline {baseline*100:.1f}% (need +2pp)",
        },
        "K1530": {
            "pass": k1530_pass,
            "detail": f"PS+={p2_acc*100:.1f}% vs PS={p1_acc*100:.1f}%",
        },
        "K1531": {
            "pass": k1531_pass,
            "detail": f"best_tokens={best_tokens:.0f}, P0_tokens={p0_tokens:.0f}, ratio={token_ratio:.2f}x",
        },
    }

    summary = {
        "P0_direct_acc": p0_acc,
        "P1_ps_acc": p1_acc,
        "P2_ps_plus_acc": p2_acc,
        "best_prompt": best_name,
        "best_acc": best_acc,
        "baseline": baseline,
        "delta_vs_baseline_pp": round((best_acc - baseline) * 100, 1),
        "token_ratio_best_vs_direct": round(token_ratio, 2),
        "kill_criteria": kill,
    }
    results["summary"] = summary

    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)
    log(f"  Baseline (Finding #530): {baseline*100:.1f}%")
    log(f"  P0_direct:               {p0_acc*100:.1f}%  (avg_tokens={prompt_results['P0_direct']['avg_output_tokens']:.0f})")
    log(f"  P1_ps:                   {p1_acc*100:.1f}%  (avg_tokens={prompt_results['P1_ps']['avg_output_tokens']:.0f})")
    log(f"  P2_ps_plus:              {p2_acc*100:.1f}%  (avg_tokens={prompt_results['P2_ps_plus']['avg_output_tokens']:.0f})")
    log(f"  Best: {best_name} = {best_acc*100:.1f}% (+{(best_acc-baseline)*100:.1f}pp vs baseline)")
    log(f"  Token ratio (best/P0): {token_ratio:.2f}x")
    log(f"  K1529 (best >= 64.1%): {'PASS' if k1529_pass else 'FAIL'}")
    log(f"  K1530 (PS+ >= PS):     {'PASS' if k1530_pass else 'FAIL'}")
    log(f"  K1531 (tokens <= 2x):  {'PASS' if k1531_pass else 'FAIL'}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
