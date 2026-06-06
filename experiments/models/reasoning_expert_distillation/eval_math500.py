#!/usr/bin/env python3
"""Evaluate reasoning LoRA expert on MATH-500 benchmark.

Uses vLLM for efficient batched inference with native LoRA support.
Falls back to HuggingFace generate() with explicit memory management if vLLM unavailable.

Tests four conditions:
  1. Base model (Qwen2.5-7B, no adapter)
  2. Reasoning adapter only (base + reasoning LoRA)
  3. Domain expert only (base + math domain LoRA from pilot50)
  4. Composed: reasoning + domain expert (base + both LoRAs pre-merged)

Kill criteria:
  K1: Reasoning LoRA does not improve MATH-500 accuracy >10pp over base
  K2: Reasoning + domain composed model does not outperform either alone

Usage (on RunPod):
    cd /workspace/llm
    python experiments/models/reasoning_expert_distillation/eval_math500.py
"""

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
HF_CACHE = os.environ.get("HF_CACHE", "/workspace/hf_cache")
REPO_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "micro" / "models" / "reasoning_expert_distillation"
REASONING_ADAPTER = OUTPUT_DIR / "reasoning_adapter"

MATH500_URL = (
    "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
    "main/ch03/01_main-chapter-code/math500_test.json"
)

PILOT50_MATH_ADAPTER = REPO_ROOT / "data" / "adapters" / "math"
PILOT50_FALLBACK_ADAPTERS = [
    REPO_ROOT / "data" / "adapters" / "abstract-math",
    REPO_ROOT / "data" / "adapters" / "statistics",
]

MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.0
MAX_EVAL_EXAMPLES = 500
SEED = 42


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── MATH-500 Answer Parsing ──────────────────────────────────────────────────

RE_BOXED = re.compile(r"\\boxed\{", re.DOTALL)
RE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def get_last_boxed(text: str) -> str:
    matches = list(RE_BOXED.finditer(text))
    if not matches:
        return ""
    start = matches[-1].end()
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
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
    text = text.replace("\\$", "").replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\ ", " ")
    text = text.replace("\\text{", "").replace("\\mathrm{", "")
    text = text.replace("\\dfrac", "\\frac")
    text = " ".join(text.split())
    return text


def grade_answer(predicted: str, ground_truth: str) -> bool:
    pred_norm = normalize_text(predicted)
    gt_norm = normalize_text(ground_truth)
    if pred_norm == gt_norm:
        return True
    try:
        if "/" in pred_norm and "/" not in gt_norm:
            parts = pred_norm.split("/")
            if len(parts) == 2:
                return abs(float(parts[0]) / float(parts[1]) - float(gt_norm)) < 1e-6
        elif "/" in gt_norm and "/" not in pred_norm:
            parts = gt_norm.split("/")
            if len(parts) == 2:
                return abs(float(parts[0]) / float(parts[1]) - float(pred_norm)) < 1e-6
        return abs(float(pred_norm) - float(gt_norm)) < 1e-6
    except (ValueError, ZeroDivisionError):
        return False


# ── Dataset Loading ──────────────────────────────────────────────────────────

def load_math500(max_examples: int = MAX_EVAL_EXAMPLES) -> list[dict]:
    import requests
    local_path = OUTPUT_DIR / "math500_test.json"
    if local_path.exists():
        with open(local_path) as f:
            data = json.load(f)
    else:
        ref_path = REPO_ROOT / "references" / "reasoning-from-scratch" / \
            "ch03" / "01_main-chapter-code" / "math500_test.json"
        if ref_path.exists():
            with open(ref_path) as f:
                data = json.load(f)
        else:
            log(f"Downloading MATH-500 from {MATH500_URL}")
            r = requests.get(MATH500_URL, timeout=30)
            r.raise_for_status()
            data = r.json()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "w") as f:
                json.dump(data, f, indent=2)
    return data[:max_examples]


# ── vLLM-based Generation (primary) ─────────────────────────────────────────
# PagedAttention handles memory automatically — no leaks, no fragmentation.
# Native LoRA support via enable_lora + LoRARequest.
# Batched inference is 10-50x faster than sequential HF generate().

def try_vllm_available() -> bool:
    try:
        import vllm
        return True
    except ImportError:
        return False


def build_prompts(problems: list[dict], tokenizer_or_None=None) -> list[str]:
    """Build chat-formatted prompts for all problems."""
    prompts = []
    for ex in problems:
        messages = [
            {"role": "system", "content": (
                "You are a helpful math assistant. Solve the problem step by step "
                "and write your final answer as \\boxed{ANSWER}."
            )},
            {"role": "user", "content": ex["problem"]},
        ]
        if tokenizer_or_None is not None:
            prompt = tokenizer_or_None.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Qwen2.5 chat format fallback
            prompt = (
                f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        prompts.append(prompt)
    return prompts


def evaluate_vllm(model_path: str, problems: list[dict],
                  adapter_path: str = None, adapter_name: str = "default",
                  max_new_tokens: int = MAX_NEW_TOKENS,
                  verbose: bool = False) -> dict:
    """Evaluate using vLLM offline batch inference.

    vLLM handles:
    - PagedAttention: no memory leaks between requests
    - Batched generation: all problems processed efficiently
    - Native LoRA: adapters loaded without model reload
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    log(f"  vLLM: Loading {model_path}" + (f" + {adapter_path}" if adapter_path else ""))

    # Build engine with LoRA support if adapter provided
    engine_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        dtype="half",  # fp16 for A5000
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        seed=SEED,
    )
    if adapter_path:
        engine_kwargs["enable_lora"] = True
        engine_kwargs["max_lora_rank"] = 16

    llm = LLM(**engine_kwargs)
    tokenizer = llm.get_tokenizer()

    prompts = build_prompts(problems, tokenizer)
    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,  # greedy
    )

    lora_req = None
    if adapter_path:
        lora_req = LoRARequest(adapter_name, 1, adapter_path)

    log(f"  vLLM: Generating {len(prompts)} answers...")
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
            "problem": example["problem"][:100] + "..." if len(example["problem"]) > 100 else example["problem"],
            "ground_truth": example["answer"],
            "predicted": predicted,
            "correct": is_correct,
        })
        if verbose and (i + 1) % 50 == 0:
            log(f"    [{i+1}/{len(problems)}] accuracy: {100*correct/(i+1):.1f}%")

    accuracy = correct / len(problems) if problems else 0.0
    log(f"  {adapter_name}: {correct}/{len(problems)} = {100*accuracy:.1f}% ({elapsed:.0f}s)")

    # Cleanup vLLM engine
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "condition": adapter_name,
        "correct": correct,
        "total": len(problems),
        "accuracy": accuracy,
        "accuracy_pct": round(100 * accuracy, 2),
        "elapsed_s": elapsed,
        "per_example": results,
        "engine": "vllm",
    }


# ── HF-based Generation (fallback) ──────────────────────────────────────────
# Used when vLLM is not available. Includes explicit memory management
# to prevent the leaks that caused OOM on the old instance.

def evaluate_hf(model, tokenizer, problems: list[dict],
                condition_name: str, max_new_tokens: int = MAX_NEW_TOKENS,
                verbose: bool = False) -> dict:
    """Evaluate using HuggingFace generate() with explicit memory management.

    Key fixes vs original:
    - del outputs + empty_cache every example (prevents KV cache accumulation)
    - Periodic full GC every 50 examples
    - Input tensors explicitly deleted
    """
    log(f"\n  Evaluating (HF): {condition_name} ({len(problems)} examples)")
    t0 = time.time()
    correct = 0
    results = []
    prompts = build_prompts(problems, tokenizer)

    for i, (prompt, example) in enumerate(zip(prompts, problems)):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)

        # Explicit cleanup — prevents KV cache memory accumulation
        del outputs, inputs, generated
        torch.cuda.empty_cache()

        predicted = extract_final_answer(text)
        is_correct = grade_answer(predicted, example["answer"])
        correct += int(is_correct)
        results.append({
            "idx": i,
            "problem": example["problem"][:100] + "...",
            "ground_truth": example["answer"],
            "predicted": predicted,
            "correct": is_correct,
        })

        if verbose and (i + 1) % 10 == 0:
            log(f"    [{i+1}/{len(problems)}] accuracy: {100*correct/(i+1):.1f}%")

        # Periodic deep cleanup every 50 examples
        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    accuracy = correct / len(problems) if problems else 0.0
    log(f"  {condition_name}: {correct}/{len(problems)} = {100*accuracy:.1f}% ({elapsed:.0f}s)")

    return {
        "condition": condition_name,
        "correct": correct,
        "total": len(problems),
        "accuracy": accuracy,
        "accuracy_pct": round(100 * accuracy, 2),
        "elapsed_s": elapsed,
        "per_example": results,
        "engine": "hf",
    }


# ── Adapter merge for composed condition ─────────────────────────────────────

def merge_adapters_to_model(base_model_path: str, adapter_paths: list[str],
                            output_path: str) -> str:
    """Merge LoRA adapters into base model weights, save merged model.

    For vLLM: we need a merged model on disk (vLLM LoRA only supports 1 adapter).
    For composed condition (reasoning + domain), merge both into base and save.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from safetensors.torch import load_file

    log(f"Merging {len(adapter_paths)} adapters into base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, device_map="cpu",
        trust_remote_code=True,
    )

    for adapter_path in adapter_paths:
        log(f"  Merging: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    log(f"  Merged model saved to {output_path}")

    del model
    gc.collect()
    return str(output_path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning expert on MATH-500")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--reasoning-adapter", default=str(REASONING_ADAPTER))
    parser.add_argument("--domain-adapter", default=None)
    parser.add_argument("--max-examples", type=int, default=MAX_EVAL_EXAMPLES)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--conditions", nargs="*",
                        default=["base", "reasoning", "domain", "composed"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "math500_results.json"))
    parser.add_argument("--force-hf", action="store_true", help="Force HF backend even if vLLM available")
    args = parser.parse_args()

    use_vllm = try_vllm_available() and not args.force_hf

    # Find domain adapter
    domain_adapter = args.domain_adapter
    if domain_adapter is None:
        for path in [PILOT50_MATH_ADAPTER] + PILOT50_FALLBACK_ADAPTERS:
            if path.exists() and (path / "adapter_config.json").exists():
                domain_adapter = str(path)
                break

    if domain_adapter is None and ("domain" in args.conditions or "composed" in args.conditions):
        log("WARNING: No domain adapter found. Skipping domain/composed conditions.")
        args.conditions = [c for c in args.conditions if c not in ("domain", "composed")]

    math500 = load_math500(args.max_examples)
    log(f"MATH-500: {len(math500)} problems loaded")
    log(f"Engine: {'vLLM (batched, PagedAttention)' if use_vllm else 'HuggingFace (sequential)'}")

    all_results = {
        "experiment": "reasoning_expert_distillation_eval",
        "base_model": args.base_model,
        "reasoning_adapter": args.reasoning_adapter,
        "domain_adapter": domain_adapter,
        "max_examples": args.max_examples,
        "engine": "vllm" if use_vllm else "hf",
        "conditions": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if use_vllm:
        # ── vLLM path: each condition gets its own engine (clean memory) ──

        if "base" in args.conditions:
            result = evaluate_vllm(
                args.base_model, math500, adapter_name="base",
                max_new_tokens=args.max_new_tokens, verbose=args.verbose
            )
            all_results["conditions"]["base"] = result

        if "reasoning" in args.conditions:
            result = evaluate_vllm(
                args.base_model, math500,
                adapter_path=args.reasoning_adapter, adapter_name="reasoning_only",
                max_new_tokens=args.max_new_tokens, verbose=args.verbose
            )
            all_results["conditions"]["reasoning_only"] = result

        if "domain" in args.conditions and domain_adapter:
            result = evaluate_vllm(
                args.base_model, math500,
                adapter_path=domain_adapter, adapter_name="domain_only",
                max_new_tokens=args.max_new_tokens, verbose=args.verbose
            )
            all_results["conditions"]["domain_only"] = result

        if "composed" in args.conditions and domain_adapter:
            # vLLM supports only 1 LoRA at a time, so merge both into a temp model
            merged_path = str(OUTPUT_DIR / "merged_reasoning_domain")
            merge_adapters_to_model(
                args.base_model, [args.reasoning_adapter, domain_adapter], merged_path
            )
            result = evaluate_vllm(
                merged_path, math500, adapter_name="reasoning_plus_domain",
                max_new_tokens=args.max_new_tokens, verbose=args.verbose
            )
            all_results["conditions"]["composed"] = result

    else:
        # ── HF fallback path with explicit memory management ──
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        log("Loading base model...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, cache_dir=HF_CACHE, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.float16, device_map="auto",
            cache_dir=HF_CACHE, trust_remote_code=True,
        )
        base_model.eval()

        if "base" in args.conditions:
            result = evaluate_hf(
                base_model, tokenizer, math500, "base",
                max_new_tokens=args.max_new_tokens, verbose=args.verbose
            )
            all_results["conditions"]["base"] = result

        if "reasoning" in args.conditions:
            reasoning_model = PeftModel.from_pretrained(base_model, args.reasoning_adapter)
            reasoning_model.eval()
            result = evaluate_hf(
                reasoning_model, tokenizer, math500, "reasoning_only",
                max_new_tokens=args.max_new_tokens, verbose=args.verbose
            )
            all_results["conditions"]["reasoning_only"] = result
            del reasoning_model; gc.collect(); torch.cuda.empty_cache()

        if "domain" in args.conditions and domain_adapter:
            domain_model = PeftModel.from_pretrained(base_model, domain_adapter)
            domain_model.eval()
            result = evaluate_hf(
                domain_model, tokenizer, math500, "domain_only",
                max_new_tokens=args.max_new_tokens, verbose=args.verbose
            )
            all_results["conditions"]["domain_only"] = result
            del domain_model; gc.collect(); torch.cuda.empty_cache()

        if "composed" in args.conditions and domain_adapter:
            # Merge both adapters into base
            composed = PeftModel.from_pretrained(base_model, args.reasoning_adapter)
            composed = composed.merge_and_unload()
            composed = PeftModel.from_pretrained(composed, domain_adapter)
            composed = composed.merge_and_unload()
            composed.eval()
            result = evaluate_hf(
                composed, tokenizer, math500, "reasoning_plus_domain",
                max_new_tokens=args.max_new_tokens, verbose=args.verbose
            )
            all_results["conditions"]["composed"] = result
            del composed; gc.collect(); torch.cuda.empty_cache()

        del base_model; gc.collect(); torch.cuda.empty_cache()

    # ── Kill Criteria Assessment ─────────────────────────────────────────
    log("\n" + "=" * 72)
    log("KILL CRITERIA ASSESSMENT")
    log("=" * 72)

    base_acc = all_results["conditions"].get("base", {}).get("accuracy_pct", 0)
    reasoning_acc = all_results["conditions"].get("reasoning_only", {}).get("accuracy_pct", 0)
    domain_acc = all_results["conditions"].get("domain_only", {}).get("accuracy_pct", 0)
    composed_acc = all_results["conditions"].get("composed", {}).get("accuracy_pct", 0)

    log(f"\n  Accuracy Summary:")
    log(f"    Base:                {base_acc:.1f}%")
    log(f"    Reasoning only:      {reasoning_acc:.1f}%")
    if domain_adapter:
        log(f"    Domain only:         {domain_acc:.1f}%")
        log(f"    Reasoning + Domain:  {composed_acc:.1f}%")

    improvement_pp = reasoning_acc - base_acc
    k1_pass = improvement_pp > 10
    log(f"\n  K1: Reasoning improvement > 10pp over base")
    log(f"      Improvement: {improvement_pp:+.1f}pp")
    log(f"      Verdict: {'PASS' if k1_pass else 'KILL'}")

    best_single = max(reasoning_acc, domain_acc) if domain_adapter else reasoning_acc
    k2_pass = composed_acc > best_single if domain_adapter else None
    if domain_adapter:
        log(f"\n  K2: Composed outperforms best single adapter")
        log(f"      Best single: {best_single:.1f}%")
        log(f"      Composed: {composed_acc:.1f}%")
        log(f"      Verdict: {'PASS' if k2_pass else 'KILL'}")

    if not k1_pass:
        verdict = "KILLED (K1: reasoning distillation failed)"
    elif k2_pass is False:
        verdict = "KILLED (K2: composition does not improve reasoning)"
    elif k2_pass is True:
        verdict = "PASS (reasoning is composable capability)"
    else:
        verdict = "PARTIAL (K1 pass, K2 not tested)"

    all_results["kill_criteria"] = {
        "K1_reasoning_gt_10pp": {"threshold": 10, "actual": round(improvement_pp, 2), "pass": k1_pass},
        "K2_composed_gt_best_single": {"best_single": round(best_single, 2), "composed": round(composed_acc, 2), "pass": k2_pass},
    }
    all_results["verdict"] = verdict
    log(f"\n  Overall: {verdict}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
