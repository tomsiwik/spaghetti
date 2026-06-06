#!/usr/bin/env python3
"""Serverless evaluation client for SOLE experiments via Groq API.

Runs locally on your Mac, sends inference requests to Groq's LPU cloud.
No GPU needed locally. OpenAI-compatible API.

Setup:
    1. Set GROQ_API_KEY in .env
    2. uv run --extra serverless python tools/serverless_eval.py math500

Usage:
    uv run --extra serverless python tools/serverless_eval.py math500 [--max-examples 200]
    uv run --extra serverless python tools/serverless_eval.py mmlu [--domains math,python]
    uv run --extra serverless python tools/serverless_eval.py ping
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from dotenv import load_dotenv

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent.parent / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = os.environ.get("EVAL_MODEL", "qwen/qwen3-32b")

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "results"

MATH500_URL = (
    "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
    "main/ch03/01_main-chapter-code/math500_test.json"
)

SEED = 42


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_headers() -> dict:
    if not GROQ_API_KEY:
        print("Set GROQ_API_KEY in .env")
        sys.exit(1)
    return {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }


# ── MATH-500 Answer Parsing ──────────────────────────────────────────────────

RE_BOXED = re.compile(r"\\boxed\{", re.DOTALL)
RE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def get_last_boxed(text: str) -> str:
    matches = list(RE_BOXED.finditer(text))
    if not matches:
        return ""
    start = matches[-1].end()
    depth, pos = 1, start
    while pos < len(text) and depth > 0:
        if text[pos] == '{': depth += 1
        elif text[pos] == '}': depth -= 1
        pos += 1
    return text[start:pos - 1] if depth == 0 else text[start:]


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    boxed = get_last_boxed(text.strip())
    if boxed:
        return boxed.strip().strip("$ ")
    numbers = RE_NUMBER.findall(text)
    return numbers[-1] if numbers else text.strip()


def normalize_text(text: str) -> str:
    text = text.strip()
    for s in ["\\$", "$", "\\left", "\\right", "\\,", "\\text{", "\\mathrm{",
              "\\(", "\\)", "\\{", "\\}"]:
        text = text.replace(s, "")
    text = text.replace("\\ ", " ").replace("\\dfrac", "\\frac")
    # Strip trailing braces from \text{...} removal
    text = text.rstrip("}")
    return " ".join(text.split())


def grade_answer(predicted: str, ground_truth: str) -> bool:
    pred, gt = normalize_text(predicted), normalize_text(ground_truth)
    if pred == gt:
        return True
    try:
        if "/" in pred and "/" not in gt:
            p = pred.split("/")
            if len(p) == 2:
                return abs(float(p[0]) / float(p[1]) - float(gt)) < 1e-6
        elif "/" in gt and "/" not in pred:
            p = gt.split("/")
            if len(p) == 2:
                return abs(float(p[0]) / float(p[1]) - float(pred)) < 1e-6
        return abs(float(pred) - float(gt)) < 1e-6
    except (ValueError, ZeroDivisionError):
        return False


# ── Dataset Loading ──────────────────────────────────────────────────────────

def load_math500(max_examples: int = 500) -> list[dict]:
    local_path = REPO_ROOT / "micro" / "models" / "reasoning_expert_distillation" / "math500_test.json"
    if local_path.exists():
        with open(local_path) as f:
            return json.load(f)[:max_examples]
    log("Downloading MATH-500...")
    r = httpx.get(MATH500_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w") as f:
        json.dump(data, f, indent=2)
    return data[:max_examples]


# ── Groq Chat Completion ────────────────────────────────────────────────────

def chat_complete(messages: list[dict], max_tokens: int = 2048,
                  temperature: float = 0.0, retries: int = 3) -> str:
    """Single chat completion via Groq API with retry on rate limit."""
    headers = get_headers()
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(retries):
        try:
            r = httpx.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers=headers, json=payload, timeout=120.0
            )
            if r.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                log(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"] or ""
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            log(f"  API error after {retries} retries: {e}")
            return ""
    return ""


def chat_complete_batch(problems: list[dict], system_prompt: str,
                        max_tokens: int = 2048, max_workers: int = 5) -> list[str]:
    """Parallel chat completions. Groq rate limits ~30 req/min on free tier."""
    results = [""] * len(problems)

    def do_one(i: int) -> tuple[int, str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problems[i]["problem"]},
        ]
        return i, chat_complete(messages, max_tokens=max_tokens)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(do_one, i): i for i in range(len(problems))}
        done = 0
        for future in as_completed(futures):
            idx, text = future.result()
            results[idx] = text
            done += 1
            if done % 20 == 0:
                log(f"    {done}/{len(problems)} complete")

    return results


# ── MATH-500 Evaluation ─────────────────────────────────────────────────────

def eval_math500(args):
    get_headers()
    math500 = load_math500(args.max_examples)
    log(f"MATH-500: {len(math500)} problems, model={MODEL}")

    system_prompt = (
        "You are a helpful math assistant. Solve the problem step by step "
        "and write your final answer as \\boxed{ANSWER}."
    )

    log(f"Generating {len(math500)} answers (parallel={args.workers})...")
    t0 = time.time()
    responses = chat_complete_batch(
        math500, system_prompt,
        max_tokens=args.max_tokens, max_workers=args.workers
    )
    elapsed = time.time() - t0

    correct = 0
    results = []
    for i, (response, example) in enumerate(zip(responses, math500)):
        predicted = extract_final_answer(response)
        is_correct = grade_answer(predicted, example["answer"])
        correct += int(is_correct)
        results.append({
            "idx": i,
            "ground_truth": example["answer"],
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / len(math500) if math500 else 0
    log(f"\nMATH-500: {correct}/{len(math500)} = {100*accuracy:.1f}% ({elapsed:.0f}s)")
    log(f"Throughput: {len(math500)/elapsed:.1f} problems/sec")

    output = {
        "experiment": "math500_serverless",
        "model": MODEL,
        "engine": "groq",
        "correct": correct,
        "total": len(math500),
        "accuracy": round(accuracy, 4),
        "accuracy_pct": round(100 * accuracy, 2),
        "elapsed_s": round(elapsed, 1),
        "per_example": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = OUTPUT_DIR / "math500_serverless.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"Results saved to {out_path}")
    return output


# ── MMLU Evaluation ──────────────────────────────────────────────────────────

def eval_mmlu_subject(subject: str, max_per_subset: int = 100) -> dict:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", subject, split="test")
    if max_per_subset:
        ds = ds.select(range(min(len(ds), max_per_subset)))

    letters = ('A', 'B', 'C', 'D')
    letter_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    correct = 0
    total = 0

    for ex in ds:
        prompt = f"Multiple Choice question: {ex['question']}\n"
        prompt += "".join(f"- {c}={l}\n" for l, c in zip(letters, ex["choices"]))
        prompt += "\nRespond only with the letter of the correct answer."

        response = chat_complete(
            [{"role": "user", "content": prompt}],
            max_tokens=5, temperature=0.0
        )
        pred = response.strip().upper()
        if pred and pred[0] in "ABCD":
            pred = pred[0]
        else:
            pred = "X"

        gold = letter_map.get(ex["answer"], str(ex["answer"]).strip().upper())
        correct += int(pred == gold)
        total += 1

    return {"subject": subject, "correct": correct, "total": total,
            "accuracy": correct / max(1, total)}


def eval_mmlu(args):
    get_headers()

    domain_to_mmlu = {
        "math": ["high_school_mathematics", "college_mathematics", "elementary_mathematics"],
        "medical": ["professional_medicine", "clinical_knowledge", "college_medicine"],
        "python": ["high_school_computer_science", "college_computer_science", "machine_learning"],
        "physics": ["high_school_physics", "college_physics"],
        "legal": ["professional_law", "international_law"],
    }

    subjects = set()
    for domain in (args.domains or domain_to_mmlu.keys()):
        subjects.update(domain_to_mmlu.get(domain, [domain]))
    subjects = sorted(subjects)

    log(f"MMLU: {len(subjects)} subjects, model={MODEL}")
    results = {}
    for subject in subjects:
        log(f"  {subject}...")
        r = eval_mmlu_subject(subject, args.max_per_subset)
        results[subject] = r
        log(f"    {r['correct']}/{r['total']} = {100*r['accuracy']:.1f}%")

    total_correct = sum(r["correct"] for r in results.values())
    total_count = sum(r["total"] for r in results.values())
    avg_acc = total_correct / max(1, total_count)
    log(f"\nMMLU aggregate: {total_correct}/{total_count} = {100*avg_acc:.1f}%")

    output = {
        "experiment": "mmlu_serverless",
        "model": MODEL,
        "engine": "groq",
        "subjects": results,
        "aggregate": {"correct": total_correct, "total": total_count, "accuracy": round(avg_acc, 4)},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = OUTPUT_DIR / "mmlu_serverless.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"Results saved to {out_path}")
    return output


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Serverless eval for SOLE via Groq")
    sub = parser.add_subparsers(dest="command")

    p_math = sub.add_parser("math500", help="MATH-500 evaluation")
    p_math.add_argument("--max-examples", type=int, default=500)
    p_math.add_argument("--max-tokens", type=int, default=2048)
    p_math.add_argument("--workers", type=int, default=5)

    p_mmlu = sub.add_parser("mmlu", help="MMLU evaluation")
    p_mmlu.add_argument("--domains", nargs="*", default=None)
    p_mmlu.add_argument("--max-per-subset", type=int, default=100)

    sub.add_parser("ping", help="Test API connectivity")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "ping":
        log(f"Pinging Groq with model={MODEL}...")
        resp = chat_complete(
            [{"role": "user", "content": "Say 'ok'"}],
            max_tokens=5
        )
        log(f"Response: {resp}")
        log("API is live.")
        return

    {"math500": eval_math500, "mmlu": eval_mmlu}[args.command](args)


if __name__ == "__main__":
    main()
