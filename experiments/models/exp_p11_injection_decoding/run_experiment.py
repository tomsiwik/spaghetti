"""
P11.Z1: 'Well, Keep Thinking' Adaptive Injection Decoding (Zero Training)

Kill criteria:
  K1532: Injection decoding + PS prompt >= 65% MMLU-Pro
  K1533: Wait injection improves >= 1pp over no injection
  K1534: Injection does NOT cause degenerate loops (< 5% of responses loop)

Papers:
  arXiv:2501.12599 — s1: "Wait" budget forcing for extended reasoning
  arXiv:2503.10167 — Well, Keep Thinking: adaptive injection decoding
  arXiv:2205.01068 — Plan-and-Solve prompting
"""

import json
import re
import sys
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
SEED = 42

# N per category for eval (4 categories × 25 = 100 questions per condition)
EVAL_CATEGORIES = ["math", "biology", "physics", "chemistry"]
N_PER_CAT = 25

IS_SMOKE = "--smoke" in sys.argv
if IS_SMOKE:
    N_PER_CAT = 3
    EVAL_CATEGORIES = ["math", "biology"]

OPTION_LETTERS = "ABCDEFGHIJ"

# Injection parameters
MIN_THINKING_CHARS = 1500  # Threshold: below this, inject "Wait" (raised from 500: Gemma 4 mean=1641, 500 never triggers)
MAX_INJECTIONS = 2          # Max injection attempts per question
MAX_TOKENS_INITIAL = 4096   # Initial generation budget
MAX_TOKENS_AFTER_INJECT = 2048  # Additional tokens after injection

# Plan-and-Solve prompt prefix (arXiv:2205.01068)
PS_PREFIX = (
    "Let's first plan what steps are needed to solve this problem, "
    "then work through it carefully step by step.\n\n"
)


def log(msg):
    print(msg, flush=True)


def log_memory(label):
    import mlx.core as mx
    info = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={info:.2f}GB peak={peak:.2f}GB")
    mx.reset_peak_memory()


def format_question(row, tokenizer, use_ps_prefix=False):
    """Format MMLU-Pro question with optional Plan-and-Solve prefix."""
    options = row["options"]
    option_text = "\n".join(
        f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
    )
    prefix = PS_PREFIX if use_ps_prefix else ""
    content = (
        f"{prefix}"
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
        enable_thinking=True,
    )


def extract_thinking(text):
    """Extract thinking content and its character count from response."""
    m = re.search(r'<\|channel>thought(.*?)<channel\|>', text, flags=re.DOTALL)
    if m:
        return m.group(1), len(m.group(1))
    return "", 0


def strip_thinking(text):
    """Strip thinking block, return (answer_text, thinking_char_count)."""
    thinking_content, thinking_chars = extract_thinking(text)
    cleaned = re.sub(r'<\|channel>thought.*?<channel\|>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    return cleaned.strip(), len(thinking_content)


def parse_answer(response):
    """Extract answer letter from response. Returns (letter_or_None, thinking_chars)."""
    if not response:
        return None, 0
    answer_text, thinking_chars = strip_thinking(response)
    if not answer_text:
        return None, thinking_chars
    # Direct single letter
    if len(answer_text) == 1 and answer_text.upper() in OPTION_LETTERS:
        return answer_text.upper(), thinking_chars
    # Starts with letter followed by delimiter
    m = re.match(r"^([A-J])[.\s:)\-,]", answer_text)
    if m:
        return m.group(1), thinking_chars
    # "The answer is X" pattern
    m = re.search(r"(?:answer|correct)\s+(?:is\s+)?([A-J])", answer_text, re.IGNORECASE)
    if m:
        return m.group(1).upper(), thinking_chars
    # Last letter in text
    m = re.search(r"([A-J])", answer_text)
    if m:
        return m.group(1).upper(), thinking_chars
    return None, thinking_chars


def generate_standard(model, tokenizer, prompt, max_tokens=MAX_TOKENS_INITIAL):
    """Standard generate without injection."""
    from mlx_lm import generate
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def generate_with_wait_injection(model, tokenizer, formatted_prompt,
                                  min_thinking=MIN_THINKING_CHARS,
                                  max_injections=MAX_INJECTIONS):
    """
    Generate with 'Wait' injection when thinking terminates prematurely.

    Strategy (prefix-continuation):
    1. Generate normally; collect full response
    2. Check thinking character count
    3. If thinking < min_thinking, reconstruct prompt with:
       [formatted_prompt] + [thinking_start + existing_thinking + "\\nWait\\n"]
       and re-generate for continuation
    4. Cap at max_injections to prevent loops

    Returns: (full_response, injection_count, thinking_chars_final)
    """
    from mlx_lm import generate
    import mlx.core as mx

    mx.random.seed(SEED)
    response = generate(model, tokenizer, prompt=formatted_prompt,
                        max_tokens=MAX_TOKENS_INITIAL, verbose=False)

    _, thinking_chars = extract_thinking(response)
    injection_count = 0

    for attempt in range(max_injections):
        if thinking_chars >= min_thinking:
            break

        # Check for degenerate (no thinking block at all)
        if "<|channel>thought" not in response:
            break

        # Extract partial thinking content (before <channel|>)
        m = re.search(r'(<\|channel>thought)(.*?)(<channel\|>)?', response, re.DOTALL)
        if not m:
            break

        thinking_start_token = m.group(1)
        partial_thinking = m.group(2)

        # Reconstruct prefix: formatted_prompt + thinking_start + partial + "Wait\n"
        # This causes the model to continue from the partial thinking context
        injection_prefix = (
            formatted_prompt
            + thinking_start_token
            + partial_thinking
            + "\nWait, I should think about this more carefully.\n"
        )

        # Generate continuation from injected prefix
        continuation = generate(
            model, tokenizer,
            prompt=injection_prefix,
            max_tokens=MAX_TOKENS_AFTER_INJECT,
            verbose=False
        )

        # Reconstruct: drop the partial from response, replace with injected version
        prefix_without_channel = thinking_start_token + partial_thinking + \
            "\nWait, I should think about this more carefully.\n"
        response = prefix_without_channel + continuation

        _, thinking_chars = extract_thinking(response)
        injection_count += 1

        log(f"    injection {attempt+1}: thinking now {thinking_chars} chars")

    return response, injection_count, thinking_chars


def is_degenerate(response, loop_threshold=3):
    """Check if response has degenerate loop (same phrase repeated > loop_threshold times)."""
    # Simple heuristic: check if any 30-char substring repeats > loop_threshold times
    if len(response) < 100:
        return False
    chunks = [response[i:i+30] for i in range(0, len(response)-30, 30)]
    from collections import Counter
    counts = Counter(chunks)
    return any(c > loop_threshold for c in counts.values())


def run_condition(model, tokenizer, questions, condition_name, use_ps=False, use_injection=False):
    """Run a single condition on a list of questions."""
    correct = 0
    total = 0
    total_thinking = 0
    injection_counts = []
    degenerate_count = 0
    per_cat = {}

    log(f"\n  [{condition_name}] n={len(questions)}, ps={use_ps}, injection={use_injection}")

    for i, (cat, row) in enumerate(questions):
        prompt = format_question(row, tokenizer, use_ps_prefix=use_ps)

        if use_injection:
            response, inj_count, t_chars = generate_with_wait_injection(model, tokenizer, prompt)
            injection_counts.append(inj_count)
        else:
            response = generate_standard(model, tokenizer, prompt)
            _, t_chars = extract_thinking(response)
            injection_counts.append(0)

        degen = is_degenerate(response)
        if degen:
            degenerate_count += 1

        predicted, _ = parse_answer(response)
        expected = OPTION_LETTERS[row["answer"]] if isinstance(row["answer"], int) else row["answer"]

        is_correct = predicted == expected if predicted else False
        if is_correct:
            correct += 1
        total += 1
        total_thinking += t_chars

        if cat not in per_cat:
            per_cat[cat] = {"correct": 0, "total": 0}
        per_cat[cat]["total"] += 1
        if is_correct:
            per_cat[cat]["correct"] += 1

        if (i + 1) % 10 == 0:
            log(f"    {i+1}/{len(questions)}: acc={correct/total:.3f} thinking={total_thinking//max(total,1)}c/q")

    acc = correct / total if total else 0
    avg_thinking = total_thinking / total if total else 0
    avg_injections = sum(injection_counts) / len(injection_counts) if injection_counts else 0
    degen_rate = degenerate_count / total if total else 0

    result = {
        "condition": condition_name,
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": total,
        "avg_thinking_chars": round(avg_thinking, 0),
        "avg_injections": round(avg_injections, 3),
        "degenerate_rate": round(degen_rate, 4),
        "per_category": {
            cat: round(v["correct"] / v["total"], 4)
            for cat, v in per_cat.items() if v["total"] > 0
        },
    }
    log(f"    RESULT: acc={acc:.3f} ({correct}/{total}), think={avg_thinking:.0f}c, "
        f"inj={avg_injections:.2f}, degen={degen_rate:.3f}")
    return result


def main():
    import mlx.core as mx
    from mlx_lm import load

    log("=== P11.Z1: Injection Decoding for Extended Thinking ===")
    log(f"Model: {MODEL_ID}")
    log(f"N per category: {N_PER_CAT}, Categories: {EVAL_CATEGORIES}")
    log(f"Smoke: {IS_SMOKE}")

    # Load data
    import pandas as pd
    mmlu_path = REPO_ROOT / "experiments/models/exp_bench_mmlu_pro/data/test.parquet"
    if not mmlu_path.exists():
        mmlu_path = REPO_ROOT / "experiments/models/exp_bench_mmlu_pro_thinking/data/test.parquet"
    if not mmlu_path.exists():
        log("ERROR: MMLU-Pro data not found")
        sys.exit(1)

    df = pd.read_parquet(mmlu_path)
    log(f"Loaded MMLU-Pro: {len(df)} total questions")

    # Select questions: same seed for all conditions
    import numpy as np
    rng = np.random.RandomState(SEED)

    questions = []
    for cat in EVAL_CATEGORIES:
        cat_df = df[df["category"] == cat].reset_index(drop=True)
        if len(cat_df) < N_PER_CAT:
            log(f"  WARNING: {cat} has only {len(cat_df)} questions (need {N_PER_CAT})")
            n = len(cat_df)
        else:
            n = N_PER_CAT
        idxs = rng.choice(len(cat_df), n, replace=False)
        questions.extend([(cat, cat_df.iloc[i]) for i in idxs])

    log(f"Selected {len(questions)} questions across {len(EVAL_CATEGORIES)} categories")

    # Load model once (shared across all conditions)
    log("\n[Loading model]")
    mx.random.seed(SEED)
    model, tokenizer = load(MODEL_ID)
    log_memory("after load")

    t0 = time.time()
    results = {}

    # ─────────────────────────────────────────────
    # Condition 1: Base (no PS, no injection) — validates Finding #536
    # ─────────────────────────────────────────────
    log("\n[Phase 1] Base + thinking (K1533 baseline)")
    r1 = run_condition(model, tokenizer, questions, "base", use_ps=False, use_injection=False)
    results["base"] = r1
    log_memory("after base")

    # ─────────────────────────────────────────────
    # Condition 2: Plan-and-Solve prompt (no injection)
    # ─────────────────────────────────────────────
    log("\n[Phase 2] PS prompt + thinking")
    r2 = run_condition(model, tokenizer, questions, "ps_only", use_ps=True, use_injection=False)
    results["ps_only"] = r2
    log_memory("after ps_only")

    # ─────────────────────────────────────────────
    # Condition 3: Wait injection (no PS)
    # ─────────────────────────────────────────────
    log("\n[Phase 3] Wait injection + thinking (K1533)")
    r3 = run_condition(model, tokenizer, questions, "injection_only", use_ps=False, use_injection=True)
    results["injection_only"] = r3
    log_memory("after injection_only")

    # ─────────────────────────────────────────────
    # Condition 4: PS + Wait injection (K1532)
    # ─────────────────────────────────────────────
    log("\n[Phase 4] PS + Wait injection (K1532)")
    r4 = run_condition(model, tokenizer, questions, "ps_injection", use_ps=True, use_injection=True)
    results["ps_injection"] = r4
    log_memory("after ps_injection")

    elapsed = time.time() - t0

    # ─────────────────────────────────────────────
    # Kill criteria evaluation
    # ─────────────────────────────────────────────
    base_acc = results["base"]["accuracy"]
    injection_acc = results["injection_only"]["accuracy"]
    ps_injection_acc = results["ps_injection"]["accuracy"]
    max_degen = max(r["degenerate_rate"] for r in results.values())

    k1532 = ps_injection_acc >= 0.65
    k1533 = (injection_acc - base_acc) >= 0.01
    k1534 = max_degen < 0.05

    log("\n=== Kill Criteria ===")
    log(f"K1532 (ps_injection >= 65%): {'PASS' if k1532 else 'FAIL'} — {ps_injection_acc:.3f}")
    log(f"K1533 (injection improves >= 1pp): {'PASS' if k1533 else 'FAIL'} — delta={injection_acc-base_acc:+.3f}")
    log(f"K1534 (degen < 5%): {'PASS' if k1534 else 'FAIL'} — max_degen={max_degen:.3f}")

    log("\n=== Condition Summary ===")
    for cname, r in results.items():
        log(f"  {cname:20s}: acc={r['accuracy']:.3f} think={r['avg_thinking_chars']:.0f}c "
            f"inj={r['avg_injections']:.2f} degen={r['degenerate_rate']:.3f}")

    output = {
        "conditions": results,
        "kill_criteria": {
            "K1532": {"result": "pass" if k1532 else "fail",
                      "value": ps_injection_acc, "threshold": 0.65},
            "K1533": {"result": "pass" if k1533 else "fail",
                      "value": round(injection_acc - base_acc, 4), "threshold": 0.01},
            "K1534": {"result": "pass" if k1534 else "fail",
                      "value": round(max_degen, 4), "threshold": 0.05},
        },
        "meta": {
            "model": MODEL_ID,
            "n_questions": len(questions),
            "categories": EVAL_CATEGORIES,
            "n_per_cat": N_PER_CAT,
            "min_thinking_threshold": MIN_THINKING_CHARS,
            "max_injections": MAX_INJECTIONS,
            "elapsed_seconds": round(elapsed, 1),
            "smoke": IS_SMOKE,
        },
    }

    out_path = EXPERIMENT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nResults saved to {out_path}")
    log(f"Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
