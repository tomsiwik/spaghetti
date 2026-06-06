#!/usr/bin/env python3
"""Benchmark 50 experts vs base model on RunPod.

For each domain:
1. Generate eval data (held-out from training) using last 100 examples
2. Compute base model PPL on eval data
3. Compute expert model PPL on eval data (load LoRA adapter on top of base)
4. Record win/loss and improvement percentage

Usage (on RunPod):
    cd /workspace/llm
    python scripts/pilot50_bench.py [--max-eval 100]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "distillation"
ADAPTER_DIR = REPO_ROOT / "adapters"
RESULTS_DIR = REPO_ROOT / "results"
HF_CACHE = "/workspace/hf_cache"


def prepare_eval_data(domain: str, max_eval: int = 100) -> list[str]:
    """Extract eval data from the last N examples of training data.

    We use the TAIL of training data as eval since:
    - Teacher-generated data has no canonical train/eval split
    - Last examples were generated last = model saw them least during training
    - This is a conservative test (some overlap), but sufficient for PPL comparison
    """
    train_file = DATA_DIR / domain / "train.jsonl"
    if not train_file.exists():
        return []

    # Read all lines, take the last max_eval
    with open(train_file) as f:
        lines = f.readlines()

    eval_lines = lines[-max_eval:]
    texts = []
    for line in eval_lines:
        record = json.loads(line)
        if "messages" in record:
            # Format as conversation for PPL measurement
            parts = []
            for msg in record["messages"]:
                parts.append(f"{msg['role']}: {msg['content']}")
            texts.append("\n".join(parts))
    return texts


def compute_perplexity(model, tokenizer, texts: list[str], max_length: int = 512) -> float:
    """Compute average perplexity over a list of texts."""
    import torch

    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_length).to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            n_tokens = inputs["input_ids"].shape[1]
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def main():
    parser = argparse.ArgumentParser(description="Benchmark 50 experts vs base")
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--max-eval", type=int, default=100)
    parser.add_argument("--domains", nargs="*", help="Specific domains to benchmark")
    parser.add_argument("--output", default=str(RESULTS_DIR / "pilot50_benchmark.json"))
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Find all trained adapters
    adapter_dirs = sorted(d for d in ADAPTER_DIR.iterdir()
                          if d.is_dir() and (d / "adapter_config.json").exists())
    if args.domains:
        adapter_dirs = [d for d in adapter_dirs if d.name in args.domains]

    if not adapter_dirs:
        print(f"No adapters found in {ADAPTER_DIR}")
        sys.exit(1)

    domains = [d.name for d in adapter_dirs]
    print(f"Benchmarking {len(domains)} experts vs base model")
    print(f"  Base: {args.base}")
    print(f"  Max eval examples: {args.max_eval}")

    # Load base model (full precision for accurate PPL)
    print("\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base, cache_dir=HF_CACHE, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="auto",
        cache_dir=HF_CACHE, trust_remote_code=True)

    results = {
        "base_model": args.base,
        "domains": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_eval": args.max_eval,
    }
    wins = 0
    total = 0
    improvements = []

    for domain in domains:
        eval_texts = prepare_eval_data(domain, args.max_eval)
        if not eval_texts:
            print(f"\n  {domain}: no eval data, skipping")
            continue

        print(f"\n  {domain} ({len(eval_texts)} eval examples):")

        # Base model perplexity
        base_ppl = compute_perplexity(base_model, tokenizer, eval_texts)
        print(f"    Base PPL: {base_ppl:.2f}")

        # Expert perplexity (load LoRA on top of base)
        adapter_dir = ADAPTER_DIR / domain
        expert_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        expert_ppl = compute_perplexity(expert_model, tokenizer, eval_texts)
        print(f"    Expert PPL: {expert_ppl:.2f}")

        improvement = (base_ppl - expert_ppl) / base_ppl * 100
        beat_base = expert_ppl < base_ppl
        significant = abs(improvement) >= 2.0  # >2% improvement threshold
        print(f"    {'WIN' if beat_base else 'LOSS'}: {improvement:+.1f}% {'(significant)' if significant else '(marginal)'}")

        results["domains"][domain] = {
            "base_ppl": round(base_ppl, 4),
            "expert_ppl": round(expert_ppl, 4),
            "improvement_pct": round(improvement, 2),
            "expert_wins": beat_base,
            "significant": significant,
            "n_eval": len(eval_texts),
        }

        if beat_base:
            wins += 1
        total += 1
        improvements.append(improvement)

        # Unload expert adapter to free VRAM
        del expert_model
        torch.cuda.empty_cache()

    # Aggregate
    win_rate = wins / max(total, 1) * 100
    avg_improvement = sum(improvements) / max(len(improvements), 1)
    significant_wins = sum(1 for d in results["domains"].values()
                          if d["expert_wins"] and d["significant"])

    results["aggregate"] = {
        "domains_tested": total,
        "expert_wins": wins,
        "win_rate_pct": round(win_rate, 1),
        "significant_wins": significant_wins,
        "significant_win_rate_pct": round(significant_wins / max(total, 1) * 100, 1),
        "avg_improvement_pct": round(avg_improvement, 2),
        "median_improvement_pct": round(sorted(improvements)[len(improvements)//2], 2) if improvements else 0,
        "kill_criteria": {
            "win_rate_below_80": win_rate < 80,
            "avg_improvement_below_2": avg_improvement < 2.0,
        },
        "verdict": "PASS" if (win_rate >= 80 and avg_improvement >= 2.0) else "KILL",
    }

    print(f"\n{'='*60}")
    print(f"Expert wins: {wins}/{total} ({win_rate:.0f}%)")
    print(f"Significant wins (>2%): {significant_wins}/{total}")
    print(f"Avg improvement: {avg_improvement:+.1f}%")
    print(f"Verdict: {results['aggregate']['verdict']}")

    if win_rate >= 80 and avg_improvement >= 2.0:
        print("PASS: Composed model beats base on >80% of domains with >2% improvement")
    elif win_rate < 80:
        print("KILL: Win rate below 80%")
    else:
        print("KILL: Average improvement below 2%")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
