#!/usr/bin/env python3
"""Evaluate reasoning LoRA adapter on MATH-500 — batched HF inference.

Uses HuggingFace generate() with batch_size > 1 for better GPU utilization
than sequential generation. Function-scoped per our GPU_CODING_GUIDELINES.md.

Base Qwen2.5-7B MATH-500 accuracy: 57.0% (published, verified).

Usage:
    python eval_reasoning_batched.py
    SMOKE_TEST=1 python eval_reasoning_batched.py
    python eval_reasoning_batched.py --batch-size 8
"""

import gc
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
HF_CACHE = os.environ.get("HF_CACHE", "/workspace/hf_cache")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "micro" / "models" / "reasoning_expert_distillation"
ADAPTER_DIR = OUTPUT_DIR / "reasoning_adapter"

BASE_ACCURACY_PCT = 57.0
MAX_EVAL = 10 if SMOKE_TEST else 500
MAX_NEW_TOKENS = 256 if SMOKE_TEST else 2048
BATCH_SIZE = 2 if SMOKE_TEST else 8  # Batched generation for better GPU util
SEED = 42

MATH500_URL = (
    "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
    "main/ch03/01_main-chapter-code/math500_test.json"
)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Answer parsing ───────────────────────────────────────────────────────────

RE_BOXED = re.compile(r"\\boxed\s*\{", re.DOTALL)
RE_NUMBER = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?)")


def get_last_boxed(text):
    matches = list(RE_BOXED.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, pos = 1, start
    while pos < len(text) and depth > 0:
        if text[pos] == "{": depth += 1
        elif text[pos] == "}": depth -= 1
        pos += 1
    return text[start:pos - 1] if depth == 0 else None


def extract_answer(text):
    if not text:
        return ""
    boxed = get_last_boxed(text.strip())
    if boxed:
        return boxed.strip().strip("$ ")
    numbers = RE_NUMBER.findall(text)
    return numbers[-1] if numbers else text.strip()


def normalize(text):
    if not text:
        return ""
    for s in ["\\left", "\\right", "\\,", "\\$", "$", "\\text{", "\\mathrm{"]:
        text = text.replace(s, "")
    text = text.replace("\\ ", " ").replace("\\dfrac", "\\frac")
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    return text.replace("{", "").replace("}", "").strip()


def grade(predicted, ground_truth):
    p, g = normalize(predicted), normalize(ground_truth)
    if p == g:
        return True
    try:
        def to_f(s):
            if "/" in s:
                parts = s.strip("()").split("/")
                return float(parts[0]) / float(parts[1])
            return float(s)
        return abs(to_f(p) - to_f(g)) < 1e-6
    except (ValueError, ZeroDivisionError):
        return False


# ── Dataset ──────────────────────────────────────────────────────────────────

def load_math500():
    local = OUTPUT_DIR / "math500_test.json"
    if local.exists():
        with open(local) as f:
            return json.load(f)[:MAX_EVAL]
    import requests
    log("Downloading MATH-500...")
    r = requests.get(MATH500_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    local.parent.mkdir(parents=True, exist_ok=True)
    with open(local, "w") as f:
        json.dump(data, f, indent=2)
    return data[:MAX_EVAL]


# ── Batched eval (function-scoped per GPU_CODING_GUIDELINES.md) ──────────────

def eval_reasoning_adapter(problems, adapter_path, batch_size=BATCH_SIZE):
    """Function-scoped: all GPU resources freed when this returns."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log(f"Loading model + adapter ({adapter_path})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for batched generation

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto",
        cache_dir=HF_CACHE,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    if torch.cuda.is_available():
        log(f"  GPU memory after load: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    # Build prompts
    prompts = []
    for ex in problems:
        messages = [
            {"role": "system", "content": (
                "You are a helpful math assistant. Solve the problem step by step "
                "and write your final answer as \\boxed{ANSWER}."
            )},
            {"role": "user", "content": ex["problem"]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    log(f"Generating {len(prompts)} answers (batch_size={batch_size})...")
    gc.disable()  # Nanochat pattern: no GC pauses during compute
    t0 = time.time()
    all_texts = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"].shape[1]
            text = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            all_texts.append(text)

        # Cleanup per batch
        del outputs, inputs
        torch.cuda.empty_cache()

        done = min(i + batch_size, len(prompts))
        if done % 25 == 0 or done == len(prompts):
            elapsed = time.time() - t0
            eta = elapsed / done * (len(prompts) - done) if done > 0 else 0
            log(f"    [{done}/{len(prompts)}] elapsed: {elapsed:.0f}s, eta: {eta:.0f}s")

    gc.enable()
    gc.collect()
    elapsed = time.time() - t0

    # Grade
    correct = 0
    results = []
    for i, (text, ex) in enumerate(zip(all_texts, problems)):
        predicted = extract_answer(text)
        ok = grade(predicted, ex["answer"])
        correct += int(ok)
        results.append({
            "idx": i, "ground_truth": ex["answer"],
            "predicted": predicted, "correct": ok,
        })

    accuracy = correct / len(problems) if problems else 0
    log(f"Result: {correct}/{len(problems)} = {100*accuracy:.1f}% ({elapsed:.0f}s)")

    # Cleanup — function scope handles the rest
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "correct": correct, "total": len(problems),
        "accuracy_pct": round(100 * accuracy, 2),
        "elapsed_s": round(elapsed, 1),
        "batch_size": batch_size,
        "per_example": results,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    log("=" * 72)
    log("REASONING ADAPTER EVAL — BATCHED HF INFERENCE")
    log(f"  Adapter:       {ADAPTER_DIR}")
    log(f"  Batch size:    {args.batch_size}")
    log(f"  Base accuracy: {BASE_ACCURACY_PCT}% (published)")
    log(f"  Smoke test:    {SMOKE_TEST}")
    log("=" * 72)

    if not (ADAPTER_DIR / "adapter_config.json").exists():
        log(f"ERROR: Adapter not found at {ADAPTER_DIR}")
        sys.exit(1)

    problems = load_math500()
    log(f"MATH-500: {len(problems)} problems")

    result = eval_reasoning_adapter(problems, str(ADAPTER_DIR), args.batch_size)

    # Kill criteria
    reasoning_acc = result["accuracy_pct"]
    improvement = reasoning_acc - BASE_ACCURACY_PCT
    k1_pass = improvement > 10

    log(f"\n{'=' * 72}")
    log(f"  Base:        {BASE_ACCURACY_PCT}%")
    log(f"  Reasoning:   {reasoning_acc}%")
    log(f"  Delta:       {improvement:+.1f}pp")
    log(f"  K1 (>10pp):  {'PASS' if k1_pass else 'KILL'}")
    log(f"{'=' * 72}")

    combined = {
        "experiment": "reasoning_expert_distillation",
        "base_accuracy_pct": BASE_ACCURACY_PCT,
        "reasoning": result,
        "improvement_pp": round(improvement, 2),
        "kill_criteria": {"K1": {"threshold": 10, "actual": round(improvement, 2), "pass": k1_pass}},
        "verdict": "PASS" if k1_pass else "KILLED",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out = OUTPUT_DIR / "math500_results.json"
    with open(out, "w") as f:
        json.dump(combined, f, indent=2)
    log(f"Results saved to {out}")


if __name__ == "__main__":
    main()
