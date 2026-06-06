#!/usr/bin/env python3
"""Evaluate reasoning LoRA adapter on MATH-500 using vLLM batched inference.

Uses vLLM's offline LLM class with native LoRA support for ~100% GPU utilization.
Only evaluates the reasoning adapter — base model accuracy uses published numbers.

Qwen2.5-7B base MATH-500 accuracy: 57.0% (from previous RunPod eval, matches published).

Kill criteria:
  K1: Reasoning LoRA improves MATH-500 accuracy >10pp over base (>67%)

Usage:
    python eval_reasoning_vllm.py
    SMOKE_TEST=1 python eval_reasoning_vllm.py
"""

import gc
import json
import os
import re
import sys
import time
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
HF_CACHE = os.environ.get("HF_CACHE", "/workspace/hf_cache")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "micro" / "models" / "reasoning_expert_distillation"
ADAPTER_DIR = OUTPUT_DIR / "reasoning_adapter"

MATH500_URL = (
    "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
    "main/ch03/01_main-chapter-code/math500_test.json"
)

# Published baseline (no need to re-evaluate)
BASE_ACCURACY_PCT = 57.0

MAX_EVAL_EXAMPLES = 10 if SMOKE_TEST else 500
MAX_NEW_TOKENS = 256 if SMOKE_TEST else 2048
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── MATH-500 Answer Parsing ──────────────────────────────────────────────────

RE_BOXED = re.compile(r"\\boxed\s*\{", re.DOTALL)
RE_NUMBER = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?)")


def get_last_boxed(text: str) -> str | None:
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


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    boxed = get_last_boxed(text.strip())
    if boxed:
        return boxed.strip().strip("$ ")
    numbers = RE_NUMBER.findall(text)
    return numbers[-1] if numbers else text.strip()


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    for s in ["\\left", "\\right", "\\,", "\\$", "$", "\\text{", "\\mathrm{"]:
        text = text.replace(s, "")
    text = text.replace("\\ ", " ").replace("\\dfrac", "\\frac")
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    return text.replace("{", "").replace("}", "").strip()


def grade_answer(predicted: str, ground_truth: str) -> bool:
    pred, gt = normalize_text(predicted), normalize_text(ground_truth)
    if pred == gt:
        return True
    try:
        def to_float(s):
            if "/" in s:
                parts = s.strip("()").split("/")
                return float(parts[0]) / float(parts[1]) if len(parts) == 2 else float(s)
            return float(s)
        return abs(to_float(pred) - to_float(gt)) < 1e-6
    except (ValueError, ZeroDivisionError):
        return False


# ── Dataset ──────────────────────────────────────────────────────────────────

def load_math500(max_examples: int) -> list[dict]:
    local_path = OUTPUT_DIR / "math500_test.json"
    if local_path.exists():
        with open(local_path) as f:
            return json.load(f)[:max_examples]
    import requests
    log(f"Downloading MATH-500...")
    r = requests.get(MATH500_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w") as f:
        json.dump(data, f, indent=2)
    return data[:max_examples]


# ── vLLM Batched Eval ────────────────────────────────────────────────────────

def eval_with_vllm(problems: list[dict], adapter_path: str) -> dict:
    """Evaluate reasoning adapter using vLLM batched inference.

    Function-scoped: all GPU resources freed when this function returns.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    log(f"Loading vLLM engine: {BASE_MODEL} + LoRA from {adapter_path}")
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=16,
        dtype="half",
        max_model_len=2048,  # Reduced from 4096 — math answers rarely exceed 2K tokens
        gpu_memory_utilization=0.80,  # Leave headroom for LoRA + CUDA graphs
        seed=SEED,
        enforce_eager=True,  # Skip CUDA graph capture — saves ~5GB VRAM
    )

    sampling = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
    )

    lora_req = LoRARequest("reasoning", 1, adapter_path)

    # Build prompts using vLLM's tokenizer
    tokenizer = llm.get_tokenizer()
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

    log(f"Generating {len(prompts)} answers in batch...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=lora_req)
    elapsed = time.time() - t0

    # Grade
    correct = 0
    results = []
    for i, (output, example) in enumerate(zip(outputs, problems)):
        text = output.outputs[0].text
        predicted = extract_final_answer(text)
        is_correct = grade_answer(predicted, example["answer"])
        correct += int(is_correct)
        results.append({
            "idx": i,
            "ground_truth": example["answer"],
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / len(problems) if problems else 0
    log(f"Reasoning adapter: {correct}/{len(problems)} = {100*accuracy:.1f}% ({elapsed:.0f}s)")
    log(f"Throughput: {len(problems)/elapsed:.1f} problems/sec")

    # Cleanup — function scope handles the rest
    del llm
    gc.collect()

    return {
        "condition": "reasoning_only",
        "correct": correct,
        "total": len(problems),
        "accuracy": accuracy,
        "accuracy_pct": round(100 * accuracy, 2),
        "elapsed_s": round(elapsed, 1),
        "per_example": results,
        "engine": "vllm",
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log("=" * 72)
    log("REASONING ADAPTER EVAL — vLLM BATCHED INFERENCE")
    log(f"  Base model:     {BASE_MODEL}")
    log(f"  Adapter:        {ADAPTER_DIR}")
    log(f"  Base accuracy:  {BASE_ACCURACY_PCT}% (published)")
    log(f"  Smoke test:     {SMOKE_TEST}")
    log("=" * 72)

    if not (ADAPTER_DIR / "adapter_config.json").exists():
        log(f"ERROR: Adapter not found at {ADAPTER_DIR}")
        sys.exit(1)

    problems = load_math500(MAX_EVAL_EXAMPLES)
    log(f"MATH-500: {len(problems)} problems")

    # Single function-scoped GPU phase
    result = eval_with_vllm(problems, str(ADAPTER_DIR))

    # Kill criteria (CPU only from here)
    reasoning_acc = result["accuracy_pct"]
    improvement_pp = reasoning_acc - BASE_ACCURACY_PCT
    k1_pass = improvement_pp > 10

    log(f"\n{'=' * 72}")
    log(f"KILL CRITERIA")
    log(f"  Base (published):    {BASE_ACCURACY_PCT}%")
    log(f"  Reasoning adapter:   {reasoning_acc}%")
    log(f"  Improvement:         {improvement_pp:+.1f}pp")
    log(f"  K1 (>10pp):          {'PASS' if k1_pass else 'KILL'}")
    log(f"{'=' * 72}")

    verdict = "PASS" if k1_pass else f"KILLED (K1: {improvement_pp:+.1f}pp)"
    log(f"  Verdict: {verdict}")

    # Save results
    combined = {
        "experiment": "reasoning_expert_distillation",
        "base_model": BASE_MODEL,
        "adapter_path": str(ADAPTER_DIR),
        "base_accuracy_pct": BASE_ACCURACY_PCT,
        "base_source": "published + prior RunPod eval",
        "reasoning": result,
        "improvement_pp": round(improvement_pp, 2),
        "kill_criteria": {
            "K1_reasoning_gt_10pp": {
                "threshold": 10,
                "actual": round(improvement_pp, 2),
                "pass": k1_pass,
            },
        },
        "verdict": verdict,
        "smoke_test": SMOKE_TEST,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = OUTPUT_DIR / "math500_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    log(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
