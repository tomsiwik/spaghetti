#!/usr/bin/env python3
"""
BENCH: GPQA Diamond Baseline + Pierre Adapted
Google target: 58.6%

Direct GPQA Diamond evaluation using mlx_lm in-process (no server).
Two phases: (1) base model, (2) base + math adapter.
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
DATA_FILE = EXPERIMENT_DIR / "data" / "gpqa_diamond.csv"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# Best single adapter: math from T2.1
MATH_ADAPTER = Path(__file__).parent.parent / "exp_p1_t2_single_domain_training" / "adapters" / "math"

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
        # Clean whitespace
        correct = str(correct).strip()
        incorrect = [str(a).strip() for a in incorrect]

        # Shuffle options, track correct index
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
    """Format GPQA question with chat template."""
    option_text = "\n".join(
        f"({OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(q["options"])
    )
    content = (
        f"What is the correct answer to this question:\n"
        f"{q['question']}\n\n"
        f"Choices:\n{option_text}\n\n"
        f"Answer with ONLY the letter (A, B, C, or D). Do not explain."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def parse_answer(response):
    """Extract answer letter from model response."""
    if not response:
        return None
    response = response.strip()
    # Direct single letter
    if len(response) == 1 and response.upper() in OPTION_LETTERS:
        return response.upper()
    # "(A)" pattern
    m = re.match(r"^\(?([A-D])\)?", response)
    if m:
        return m.group(1)
    # Starts with letter followed by punctuation
    m = re.match(r"^([A-D])[.\s:)\-,]", response)
    if m:
        return m.group(1)
    # "The answer is X" pattern
    m = re.search(r"(?:answer|correct)\s+(?:is\s+)?(?:\()?([A-D])", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # First letter A-D found
    for ch in response:
        if ch.upper() in OPTION_LETTERS:
            return ch.upper()
    return None


def evaluate(questions, model, tokenizer, label="eval", limit=None):
    """Evaluate all questions using in-process generation."""
    if limit:
        questions = questions[:limit]

    total_correct = 0
    total_count = 0
    domain_results = {}
    t0 = time.time()

    for idx, q in enumerate(questions):
        prompt = format_prompt(q, tokenizer)
        try:
            response = generate(model, tokenizer, prompt=prompt, max_tokens=16)
        except Exception as e:
            response = f"ERROR: {e}"

        predicted = parse_answer(response)
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

        if (idx + 1) % 20 == 0 or (idx + 1) == len(questions):
            elapsed = time.time() - t0
            rate = total_count / elapsed if elapsed > 0 else 0
            acc = 100 * total_correct / total_count
            log(f"    [{label}] {idx+1}/{len(questions)}: "
                f"{total_correct}/{total_count} ({acc:.1f}%) | "
                f"{rate:.1f} q/s | {elapsed:.0f}s")

    elapsed = time.time() - t0
    overall_acc = total_correct / total_count if total_count > 0 else 0

    # Finalize domain results
    for d in domain_results:
        dr = domain_results[d]
        dr["accuracy"] = round(dr["correct"] / dr["total"], 4) if dr["total"] > 0 else 0

    return {
        "overall_accuracy": round(overall_acc, 4),
        "total_correct": total_correct,
        "total_count": total_count,
        "domains": domain_results,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    log("=" * 70)
    log("BENCH: GPQA Diamond -- Base E4B vs Pierre Adapted")
    log(f"SMOKE_TEST={IS_SMOKE}")
    log(f"Math adapter: {MATH_ADAPTER}")
    log(f"Adapter exists: {MATH_ADAPTER.exists()}")
    log("=" * 70)

    questions = load_data()
    limit = 10 if IS_SMOKE else None

    results = {
        "experiment": "exp_bench_gpqa_diamond",
        "smoke": IS_SMOKE,
        "total_questions": len(questions),
        "model": MODEL_ID,
        "thinking_enabled": False,
    }

    # Phase 1: Base model
    log("\n[Phase 1] Base Gemma 4 E4B (4-bit)")
    model, tokenizer = load(MODEL_ID)
    base_eval = evaluate(questions, model, tokenizer, label="base", limit=limit)
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
        adapted_eval = evaluate(questions, model, tokenizer, label="adapted", limit=limit)
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

    # Kill criteria (from experiment DB)
    google_target = 0.586
    k1_pass = abs(base_acc - google_target) <= 0.10  # within 10pp
    k2_pass = delta is not None and delta > 0  # adapter improves
    k3_pass = total_time < 2 * 3600  # < 2h

    summary = {
        "base_accuracy": base_acc,
        "adapted_accuracy": adapted_acc,
        "delta_pp": round(delta * 100, 1) if delta is not None else None,
        "google_target": google_target,
        "gap_to_google_pp": round((base_acc - google_target) * 100, 1),
        "total_elapsed_s": round(total_time, 1),
        "total_elapsed_h": round(total_time / 3600, 2),
        "kill_criteria": {
            "K1": {"pass": k1_pass, "detail": f"Base={base_acc:.4f}, target={google_target}, gap={abs(base_acc-google_target)*100:.1f}pp (threshold 10pp)"},
            "K2": {"pass": k2_pass, "detail": f"Delta={delta*100:.1f}pp" if delta is not None else "Delta=N/A"},
            "K3": {"pass": k3_pass, "detail": f"Elapsed={total_time/3600:.2f}h"},
        },
    }
    results["summary"] = summary

    log("\n" + "=" * 70)
    log("RESULTS SUMMARY")
    log("=" * 70)
    log(f"  Base: {base_acc*100:.1f}%")
    if adapted_acc is not None:
        log(f"  Adapted: {adapted_acc*100:.1f}%")
        log(f"  Delta: {summary['delta_pp']}pp")
    else:
        log(f"  Adapted: N/A")
    log(f"  Google target: {google_target*100:.1f}%")
    log(f"  Gap to Google: {summary['gap_to_google_pp']}pp")
    log(f"  Time: {summary['total_elapsed_h']}h")
    log(f"  K1 (base within 10pp of 58.6%): {'PASS' if k1_pass else 'FAIL'}")
    log(f"  K2 (adapter improves): {'PASS' if k2_pass else 'FAIL'}")
    log(f"  K3 (< 2h): {'PASS' if k3_pass else 'FAIL'}")

    # Per-domain comparison
    log("\n  Per-domain results:")
    for domain in sorted(results["base"]["domains"]):
        b = results["base"]["domains"][domain]
        log(f"    {domain:20s}: {b['correct']}/{b['total']} = {b['accuracy']*100:.1f}%")
        if adapted_acc is not None and domain in results["adapted"]["domains"]:
            a = results["adapted"]["domains"][domain]
            d = (a["accuracy"] - b["accuracy"]) * 100
            log(f"    {' ':20s}  adapted: {a['correct']}/{a['total']} = {a['accuracy']*100:.1f}% ({d:+.1f}pp)")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
