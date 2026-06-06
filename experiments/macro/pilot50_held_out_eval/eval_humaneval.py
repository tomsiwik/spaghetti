#!/usr/bin/env python3
"""
HumanEval pass@1 evaluation for code-domain pilot experts.

Uses the openai/openai_humaneval dataset (164 problems).
Generates completions, executes tests in sandbox, reports pass@1.

Usage:
    python eval_humaneval.py --adapter python --out results/humaneval_python.json
    python eval_humaneval.py --all --out results/humaneval_all.json
"""

import argparse
import json
import time
import os
import sys
import signal
import tempfile
import subprocess
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Code adapters to evaluate
CODE_ADAPTERS = ["python", "javascript", "rust", "go", "cpp", "java", "typescript", "bash", "swift", "sql"]

# HumanEval is Python-only, so we evaluate ALL code adapters on Python tasks
# to see if code distillation generalizes to Python problem-solving


def load_base_model(model_path):
    """Load quantized base model."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def load_adapter(base_model, adapter_path):
    """Load LoRA adapter."""
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model


class GenerationTimeout(Exception):
    pass


def generate_completion(model, tokenizer, prompt, max_new_tokens=512, temperature=0.0, timeout_s=120):
    """Generate code completion for a HumanEval prompt with timeout."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    def _alarm_handler(signum, frame):
        raise GenerationTimeout(f"Generation exceeded {timeout_s}s")

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    try:
        with torch.no_grad():
            if temperature == 0.0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
    except GenerationTimeout:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return "pass  # generation timed out"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    # Decode only the new tokens
    completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Stop at common termination patterns
    for stop in ["\ndef ", "\nclass ", "\n#", "\nif __name__", "\n```", "\nprint("]:
        if stop in completion:
            completion = completion[:completion.index(stop)]

    return completion


def execute_test(prompt, completion, test, entry_point, timeout=10):
    """Execute HumanEval test case and return pass/fail."""
    # Build the full program
    full_code = prompt + completion + "\n\n" + test + f"\n\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = result.returncode == 0
        error = result.stderr if not passed else None
    except subprocess.TimeoutExpired:
        passed = False
        error = "TIMEOUT"
    except Exception as e:
        passed = False
        error = str(e)
    finally:
        os.unlink(tmp_path)

    return passed, error


def evaluate_humaneval(model, tokenizer, max_problems=None, verbose=True, checkpoint_path=None):
    """Evaluate model on HumanEval dataset with incremental checkpointing."""
    ds = load_dataset("openai/openai_humaneval", split="test")
    if max_problems:
        ds = ds.select(range(min(len(ds), max_problems)))

    # Resume from checkpoint if available
    results = []
    start_idx = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        results = ckpt.get("per_problem", [])
        start_idx = len(results)
        print(f"  Resuming from checkpoint: {start_idx}/{len(ds)} done")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    for i in range(start_idx, len(ds)):
        ex = ds[i]
        prompt = ex["prompt"]
        test = ex["test"]
        entry_point = ex["entry_point"]
        task_id = ex["task_id"]

        # Generate completion with timeout
        t0 = time.time()
        try:
            completion = generate_completion(model, tokenizer, prompt)
        except Exception as e:
            print(f"  WARN: generation failed for {task_id}: {e}")
            completion = "pass  # generation error"
        gen_time = time.time() - t0

        # Execute test
        success, error = execute_test(prompt, completion, test, entry_point)

        results.append({
            "task_id": task_id,
            "passed": success,
            "error": error[:200] if error else None,
            "gen_time_s": round(gen_time, 1),
        })

        passed += int(success)
        total += 1

        if verbose and (total % 10 == 0 or total == len(ds)):
            print(f"  [{total}/{len(ds)}] pass@1={passed/total:.3f} (last gen: {gen_time:.1f}s)")

        # Save checkpoint every 20 problems
        if checkpoint_path and total % 20 == 0:
            _save_checkpoint(checkpoint_path, results, passed, total)

    return {
        "pass_at_1": passed / max(1, total),
        "passed": passed,
        "total": total,
        "per_problem": results,
    }


def _save_checkpoint(path, results, passed, total):
    """Save incremental checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"per_problem": results, "passed": passed, "total": total}, f)
    print(f"    [checkpoint saved: {total} problems]")


def main():
    parser = argparse.ArgumentParser(description="HumanEval eval for code experts")
    parser.add_argument("--base-model", default="/workspace/models/Qwen2.5-7B")
    parser.add_argument("--adapter-dir", default="/workspace/llm/adapters")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Single adapter to evaluate")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all code adapters")
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Max HumanEval problems (for quick testing)")
    parser.add_argument("--out", type=str, default="results/humaneval_held_out.json")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--base-cache", type=str, default=None)
    args = parser.parse_args()

    if args.all:
        adapters = CODE_ADAPTERS
    elif args.adapter:
        adapters = [args.adapter]
    else:
        # Just eval base + python adapter by default
        adapters = ["python"]

    print(f"Evaluating {len(adapters)} code adapters on HumanEval")

    # Load base model
    print(f"\nLoading base model: {args.base_model}")
    base_model, tokenizer = load_base_model(args.base_model)

    results = {
        "experiment": "pilot50_held_out_humaneval",
        "base_model": args.base_model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_results": None,
        "adapter_results": {},
        "comparisons": {},
    }

    # Evaluate base model
    ckpt_dir = os.path.dirname(args.out) or "."
    os.makedirs(ckpt_dir, exist_ok=True)
    base_ckpt = os.path.join(ckpt_dir, "humaneval_base_ckpt.json")

    if args.base_cache and os.path.exists(args.base_cache):
        with open(args.base_cache) as f:
            results["base_results"] = json.load(f).get("base_results")
        print(f"Loaded cached base: pass@1={results['base_results']['pass_at_1']:.3f}")
    elif not args.skip_base:
        print("\n=== Evaluating BASE model on HumanEval ===")
        results["base_results"] = evaluate_humaneval(
            base_model, tokenizer, max_problems=args.max_problems,
            checkpoint_path=base_ckpt,
        )
        print(f"Base pass@1: {results['base_results']['pass_at_1']:.3f}")

        # Cache base results
        with open(args.out.replace(".json", "_base_cache.json"), "w") as f:
            json.dump({"base_results": results["base_results"]}, f, indent=2)

    # Evaluate each adapter
    for adapter_name in adapters:
        adapter_path = os.path.join(args.adapter_dir, adapter_name)
        if not os.path.exists(adapter_path):
            print(f"\n  SKIP {adapter_name}: not found")
            continue

        print(f"\n=== Evaluating ADAPTER: {adapter_name} on HumanEval ===")
        try:
            adapted_model = load_adapter(base_model, adapter_path)
            adapter_ckpt = os.path.join(ckpt_dir, f"humaneval_{adapter_name}_ckpt.json")
            r = evaluate_humaneval(
                adapted_model, tokenizer, max_problems=args.max_problems,
                checkpoint_path=adapter_ckpt,
            )
            results["adapter_results"][adapter_name] = r
            print(f"  {adapter_name} pass@1: {r['pass_at_1']:.3f}")

            # Compare with base
            if results["base_results"]:
                base_p1 = results["base_results"]["pass_at_1"]
                adapter_p1 = r["pass_at_1"]
                results["comparisons"][adapter_name] = {
                    "base_pass_at_1": base_p1,
                    "adapter_pass_at_1": adapter_p1,
                    "delta": adapter_p1 - base_p1,
                    "delta_pct": (adapter_p1 - base_p1) * 100,
                    "adapter_wins": adapter_p1 > base_p1,
                }

            del adapted_model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate
    if results["comparisons"]:
        wins = sum(1 for c in results["comparisons"].values() if c["adapter_wins"])
        total = len(results["comparisons"])
        avg_delta = sum(c["delta_pct"] for c in results["comparisons"].values()) / total

        results["aggregate"] = {
            "adapters_evaluated": total,
            "adapters_that_beat_base": wins,
            "win_rate_pct": wins / total * 100,
            "avg_delta_pct": avg_delta,
            "kill_criteria": {
                "python_below_base": results["comparisons"].get("python", {}).get("adapter_wins") == False,
            }
        }
        print(f"\n=== AGGREGATE ===")
        print(f"Code adapters beating base: {wins}/{total}")
        print(f"Average delta: {avg_delta:+.2f}pp")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
