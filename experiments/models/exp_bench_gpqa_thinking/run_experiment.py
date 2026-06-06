#!/usr/bin/env python3
"""
BENCH: GPQA Diamond WITH Thinking Mode
Finding #518: 31.8% without thinking. Google target: 58.6% with thinking.

Single phase: base model with thinking enabled.
"""

import gc
import json
import os
import random
import re
import time
from pathlib import Path

import mlx.core as mx
import pandas as pd
from mlx_lm import generate, load

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_FILE = EXPERIMENT_DIR.parent / "exp_bench_gpqa_diamond" / "data" / "gpqa_diamond.csv"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

OPTION_LETTERS = "ABCD"


def log(msg):
    print(msg, flush=True)


def load_data():
    """Load GPQA Diamond from local CSV, shuffle answers for each question."""
    df = pd.read_csv(DATA_FILE)
    questions = []
    rng = random.Random(42)

    for _, row in df.iterrows():
        correct = row["Correct Answer"]
        incorrect = [
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        correct = str(correct).strip()
        incorrect = [str(a).strip() for a in incorrect]

        options = [correct] + incorrect
        rng.shuffle(options)
        correct_idx = options.index(correct)

        questions.append({
            "question": str(row["Question"]).strip(),
            "options": options,
            "correct_letter": OPTION_LETTERS[correct_idx],
            "domain": str(row.get("High-level domain", "Unknown")).strip(),
            "subdomain": str(row.get("Subdomain", "Unknown")).strip(),
        })

    log(f"  Loaded {len(questions)} GPQA Diamond questions")
    domains = {}
    for q in questions:
        domains[q["domain"]] = domains.get(q["domain"], 0) + 1
    log(f"  Domains: {domains}")
    return questions


def format_prompt(q, tokenizer):
    """Format GPQA question with thinking-enabled chat template."""
    option_text = "\n".join(
        f"({OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(q["options"])
    )
    content = (
        f"What is the correct answer to this question:\n"
        f"{q['question']}\n\n"
        f"Choices:\n{option_text}\n\n"
        f"Think carefully step by step, then answer with ONLY the letter "
        f"(A, B, C, or D)."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def strip_thinking(response):
    """Remove thinking tokens from Gemma 4 response.

    Returns (answer_text, thinking_len).
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
    if not response:
        return None, thinking_chars
    # Direct single letter
    if len(response) == 1 and response.upper() in OPTION_LETTERS:
        return response.upper(), thinking_chars
    # "(A)" pattern
    m = re.match(r"^\(?([A-D])\)?", response)
    if m:
        return m.group(1), thinking_chars
    # Starts with letter followed by punctuation
    m = re.match(r"^([A-D])[.\s:)\-,]", response)
    if m:
        return m.group(1), thinking_chars
    # "The answer is X" pattern
    m = re.search(r"(?:answer|correct)\s+(?:is\s+)?(?:\()?([A-D])", response, re.IGNORECASE)
    if m:
        return m.group(1).upper(), thinking_chars
    # Last letter A-D (thinking may discuss options — final choice matters)
    last_letter = None
    for ch in response:
        if ch.upper() in OPTION_LETTERS:
            last_letter = ch.upper()
    if last_letter:
        return last_letter, thinking_chars
    return None, thinking_chars


def evaluate(questions, model, tokenizer, label="eval", limit=None):
    """Evaluate all questions using in-process generation with thinking."""
    if limit:
        questions = questions[:limit]

    total_correct = 0
    total_count = 0
    total_thinking_chars = 0
    total_response_chars = 0
    domain_results = {}
    t0 = time.time()

    for idx, q in enumerate(questions):
        prompt = format_prompt(q, tokenizer)
        try:
            response = generate(model, tokenizer, prompt=prompt, max_tokens=4096)
        except Exception as e:
            response = f"ERROR: {e}"

        predicted, thinking_chars = parse_answer(response)
        total_thinking_chars += thinking_chars
        total_response_chars += len(response) if response else 0

        gold = q["correct_letter"]
        is_correct = predicted == gold
        if is_correct:
            total_correct += 1
        total_count += 1

        # Track per-domain
        domain = q["domain"]
        if domain not in domain_results:
            domain_results[domain] = {"correct": 0, "total": 0}
        domain_results[domain]["total"] += 1
        if is_correct:
            domain_results[domain]["correct"] += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == len(questions):
            elapsed = time.time() - t0
            rate = total_count / elapsed if elapsed > 0 else 0
            acc = 100 * total_correct / total_count
            log(f"    [{label}] {idx+1}/{len(questions)}: "
                f"{total_correct}/{total_count} ({acc:.1f}%) | "
                f"{rate:.2f} q/s | {elapsed:.0f}s")

    elapsed = time.time() - t0
    overall_acc = total_correct / total_count if total_count > 0 else 0

    for d in domain_results:
        dr = domain_results[d]
        dr["accuracy"] = round(dr["correct"] / dr["total"], 4) if dr["total"] > 0 else 0

    answer_chars = max(total_response_chars - total_thinking_chars, 1)
    thinking_ratio = total_thinking_chars / answer_chars if answer_chars > 0 else 0

    return {
        "overall_accuracy": round(overall_acc, 4),
        "total_correct": total_correct,
        "total_count": total_count,
        "domains": domain_results,
        "elapsed_s": round(elapsed, 1),
        "total_thinking_chars": total_thinking_chars,
        "total_response_chars": total_response_chars,
        "thinking_ratio": round(thinking_ratio, 1),
    }


def main():
    log("=" * 70)
    log("BENCH: GPQA Diamond -- WITH THINKING MODE")
    log(f"SMOKE_TEST={IS_SMOKE}")
    log(f"Model: {MODEL_ID}")
    log(f"Prior non-thinking result: 31.8% (Finding #518)")
    log(f"Google target: 58.6%")
    log("=" * 70)

    questions = load_data()
    limit = 10 if IS_SMOKE else None

    results = {
        "experiment": "exp_bench_gpqa_thinking",
        "smoke": IS_SMOKE,
        "total_questions": len(questions),
        "model": MODEL_ID,
        "thinking_enabled": True,
        "prior_non_thinking_accuracy": 0.318,
    }

    # Single phase: Base model with thinking
    log("\n[Phase 1] Base Gemma 4 E4B (4-bit) + Thinking Mode")
    model, tokenizer = load(MODEL_ID)
    eval_result = evaluate(questions, model, tokenizer, label="thinking", limit=limit)
    results["thinking"] = eval_result
    log(f"\n  Result: {eval_result['overall_accuracy']*100:.1f}% "
        f"({eval_result['total_correct']}/{eval_result['total_count']}) "
        f"in {eval_result['elapsed_s']:.0f}s")
    log(f"  Thinking ratio: {eval_result['thinking_ratio']}x answer chars")

    del model
    gc.collect()
    mx.metal.clear_cache()

    # Summary
    acc = results["thinking"]["overall_accuracy"]
    non_thinking = results["prior_non_thinking_accuracy"]
    boost = acc - non_thinking
    elapsed = results["thinking"]["elapsed_s"]

    google_target = 0.586
    k1458_pass = acc >= 0.50
    k1459_pass = boost >= 0.15
    k1460_pass = elapsed < 4 * 3600

    summary = {
        "thinking_accuracy": acc,
        "non_thinking_accuracy": non_thinking,
        "thinking_boost_pp": round(boost * 100, 1),
        "google_target": google_target,
        "gap_to_google_pp": round((acc - google_target) * 100, 1),
        "elapsed_s": round(elapsed, 1),
        "elapsed_h": round(elapsed / 3600, 2),
        "kill_criteria": {
            "K1458": {
                "pass": k1458_pass,
                "detail": f"Accuracy={acc*100:.1f}%, threshold=50%",
            },
            "K1459": {
                "pass": k1459_pass,
                "detail": f"Boost={boost*100:.1f}pp over 31.8%, threshold=15pp",
            },
            "K1460": {
                "pass": k1460_pass,
                "detail": f"Elapsed={elapsed/3600:.2f}h, threshold=4h",
            },
        },
    }
    results["summary"] = summary

    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)
    log(f"  Thinking: {acc*100:.1f}%")
    log(f"  Non-thinking (prior): {non_thinking*100:.1f}%")
    log(f"  Boost: {summary['thinking_boost_pp']}pp")
    log(f"  Google target: {google_target*100:.1f}%")
    log(f"  Gap to Google: {summary['gap_to_google_pp']}pp")
    log(f"  Time: {summary['elapsed_h']}h")
    log(f"  K1458 (>= 50%): {'PASS' if k1458_pass else 'FAIL'}")
    log(f"  K1459 (>= 15pp boost): {'PASS' if k1459_pass else 'FAIL'}")
    log(f"  K1460 (< 4h): {'PASS' if k1460_pass else 'FAIL'}")

    log("\n  Per-domain results:")
    for domain in sorted(results["thinking"]["domains"]):
        d = results["thinking"]["domains"][domain]
        log(f"    {domain:20s}: {d['correct']}/{d['total']} = {d['accuracy']*100:.1f}%")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
