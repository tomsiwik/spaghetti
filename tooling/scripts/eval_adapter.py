#!/usr/bin/env python3
"""
Adapter Evaluation Harness

Evaluates a trained adapter on standard benchmarks and writes evals.json.
Used by Ralph after every adapter training experiment.

Usage:
  uv run python scripts/eval_adapter.py <adapter_path> [--thinking] [--quick]

  --thinking    Enable thinking mode (for reasoning/thinking adapters)
  --quick       Reduced eval (n=20 per benchmark, for smoke testing)

Benchmarks:
  - GSM8K (math reasoning)
  - MMLU-Pro (knowledge + reasoning, 14 categories)

Output: <adapter_path>/evals.json
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import mlx.core as mx


MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
SEED = 42


def log(msg):
    print(msg, flush=True)


def eval_gsm8k(model, tokenizer, thinking=False, n=100):
    """Evaluate GSM8K. Returns dict with score and details."""
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))

    correct = 0
    total = 0
    t0 = time.time()

    for ex in ds:
        prompt = f"Solve the following math problem step by step.\n\n{ex['question']}"
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking,
        )

        max_tok = 4096 if thinking else 512
        response = generate(model, tokenizer, prompt=formatted,
                          max_tokens=max_tok, verbose=False)

        # Strip thinking
        if thinking:
            response = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL)
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt_ans = gt_match.group(1).replace(",", "").strip()
        total += 1

        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match:
            if pred_match.group(1).replace(",", "").strip() == gt_ans:
                correct += 1
        else:
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt_ans:
                correct += 1

    acc = correct / total * 100 if total > 0 else 0
    return {
        "score": round(acc, 1),
        "correct": correct,
        "total": total,
        "thinking": thinking,
        "elapsed_s": round(time.time() - t0, 1),
    }


def eval_mmlu_pro(model, tokenizer, thinking=False, n_per_cat=20):
    """Evaluate MMLU-Pro subset. Returns dict with score and per-category."""
    from mlx_lm import generate

    # Try to load from parquet (from our benchmark experiment)
    data_path = Path(__file__).parent.parent.parent / "experiments/models/exp_bench_mmlu_pro/data/test.parquet"
    if not data_path.exists():
        log(f"  MMLU-Pro data not found at {data_path}, skipping")
        return {"score": None, "note": "data not found"}

    import pandas as pd
    df = pd.read_parquet(data_path)

    OPTION_LETTERS = "ABCDEFGHIJ"
    sampled = []
    for cat in sorted(df["category"].unique()):
        cat_df = df[df["category"] == cat]
        n = min(n_per_cat, len(cat_df))
        sampled.append(cat_df.sample(n=n, random_state=SEED))
    df = pd.concat(sampled, ignore_index=True)

    correct = 0
    total = 0
    t0 = time.time()

    for _, row in df.iterrows():
        options = row["options"]
        option_text = "\n".join(f"{OPTION_LETTERS[i]}. {opt}" for i, opt in enumerate(options))
        content = (
            f"The following is a multiple choice question. "
            f"Think step by step, then answer with ONLY the letter of the correct option "
            f"(A through {OPTION_LETTERS[len(options)-1]}).\n\n"
            f"Question: {row['question']}\n\n"
            f"Options:\n{option_text}\n\nAnswer:"
        )
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking,
        )

        max_tok = 4096 if thinking else 32
        response = generate(model, tokenizer, prompt=formatted,
                          max_tokens=max_tok, verbose=False)

        # Strip thinking
        if thinking:
            response = re.sub(r'<\|channel>thought.*?<channel\|>', '', response, flags=re.DOTALL)
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

        # Parse answer letter
        response = response.strip()
        predicted = None
        if len(response) == 1 and response.upper() in OPTION_LETTERS:
            predicted = response.upper()
        else:
            m = re.match(r"^([A-J])[.\s:)\-,]", response)
            if m:
                predicted = m.group(1)
            else:
                m = re.search(r"(?:answer|correct)\s+(?:is\s+)?([A-J])", response, re.IGNORECASE)
                if m:
                    predicted = m.group(1).upper()
                else:
                    for ch in response:
                        if ch.upper() in OPTION_LETTERS:
                            predicted = ch.upper()
                            break

        total += 1
        if predicted == row["answer"]:
            correct += 1

    acc = correct / total * 100 if total > 0 else 0
    return {
        "score": round(acc, 1),
        "correct": correct,
        "total": total,
        "thinking": thinking,
        "n_per_category": n_per_cat,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate adapter on standard benchmarks")
    parser.add_argument("adapter_path", help="Path to adapter directory (with adapters.safetensors)")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument("--quick", action="store_true", help="Quick eval (n=20)")
    parser.add_argument("--base-only", action="store_true", help="Eval base model without adapter")
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    n_gsm8k = 20 if args.quick else 100
    n_mmlu_cat = 5 if args.quick else 20

    log("=" * 60)
    log(f"Adapter Eval: {adapter_path.name}")
    log(f"Thinking: {args.thinking}, Quick: {args.quick}")
    log("=" * 60)

    from mlx_lm import load

    if args.base_only:
        log("\nLoading base model (no adapter)...")
        model, tokenizer = load(MODEL_ID)
    else:
        log(f"\nLoading model + adapter from {adapter_path}...")
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))

    results = {
        "adapter": str(adapter_path),
        "base_model": MODEL_ID,
        "thinking": args.thinking,
        "quick": args.quick,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # GSM8K
    log(f"\n[1/2] GSM8K (n={n_gsm8k}, thinking={args.thinking})")
    gsm8k = eval_gsm8k(model, tokenizer, thinking=args.thinking, n=n_gsm8k)
    results["gsm8k"] = gsm8k
    log(f"  GSM8K: {gsm8k['score']}% ({gsm8k['correct']}/{gsm8k['total']}) in {gsm8k['elapsed_s']}s")

    # MMLU-Pro
    log(f"\n[2/2] MMLU-Pro (n_per_cat={n_mmlu_cat}, thinking={args.thinking})")
    mmlu = eval_mmlu_pro(model, tokenizer, thinking=args.thinking, n_per_cat=n_mmlu_cat)
    results["mmlu_pro"] = mmlu
    if mmlu["score"] is not None:
        log(f"  MMLU-Pro: {mmlu['score']}% ({mmlu['correct']}/{mmlu['total']}) in {mmlu['elapsed_s']}s")

    # Save
    evals_path = adapter_path / "evals.json" if not args.base_only else Path("evals_base.json")
    evals_path.write_text(json.dumps(results, indent=2))
    log(f"\nSaved to {evals_path}")

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log(f"  GSM8K:    {gsm8k['score']}%")
    log(f"  MMLU-Pro: {mmlu['score']}%")
    log(f"  Thinking: {args.thinking}")
    log("=" * 60)


if __name__ == "__main__":
    main()
