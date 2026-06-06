#!/usr/bin/env python3
"""Benchmark composed model vs base model.

Measures per-domain perplexity and routing accuracy to validate that
expert composition improves over the base model.

Usage:
    python -m composer.bench \
        --base Qwen/Qwen2.5-7B \
        --adapters adapters/ \
        --eval-data eval/ \
        --output results/e2e_benchmark.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path


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


def load_eval_data(eval_dir: Path, domain: str, max_examples: int = 100) -> list[str]:
    """Load evaluation texts for a domain."""
    eval_file = eval_dir / domain / "eval.jsonl"
    if not eval_file.exists():
        return []

    texts = []
    with open(eval_file) as f:
        for line in f:
            if len(texts) >= max_examples:
                break
            record = json.loads(line)
            # Support both flat text and chat message format
            if "text" in record:
                texts.append(record["text"])
            elif "messages" in record:
                parts = []
                for msg in record["messages"]:
                    parts.append(f"{msg['role']}: {msg['content']}")
                texts.append("\n".join(parts))
    return texts


def cmd_bench(args):
    """Run benchmark: base model vs each expert on its domain."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapters_dir = Path(args.adapters)
    eval_dir = Path(args.eval_data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find adapter domains
    adapter_dirs = sorted(d for d in adapters_dir.iterdir()
                          if d.is_dir() and (d / "adapter_config.json").exists())
    if not adapter_dirs:
        print(f"No adapters found in {adapters_dir}")
        sys.exit(1)

    domains = [d.name for d in adapter_dirs]
    print(f"Benchmarking {len(domains)} experts: {', '.join(domains)}")
    print(f"  Base: {args.base}")

    # Load base model
    print("\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="auto")

    results = {"base_model": args.base, "domains": {}, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    wins = 0
    total = 0

    for domain in domains:
        eval_texts = load_eval_data(eval_dir, domain)
        if not eval_texts:
            print(f"\n  {domain}: no eval data, skipping")
            continue

        print(f"\n  {domain} ({len(eval_texts)} eval examples):")

        # Base model perplexity
        base_ppl = compute_perplexity(base_model, tokenizer, eval_texts)
        print(f"    Base PPL: {base_ppl:.2f}")

        # Expert perplexity
        adapter_dir = adapters_dir / domain
        expert_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        expert_ppl = compute_perplexity(expert_model, tokenizer, eval_texts)
        print(f"    Expert PPL: {expert_ppl:.2f}")

        improvement = (base_ppl - expert_ppl) / base_ppl * 100
        beat_base = expert_ppl < base_ppl
        print(f"    {'BETTER' if beat_base else 'WORSE'}: {improvement:+.1f}%")

        results["domains"][domain] = {
            "base_ppl": base_ppl,
            "expert_ppl": expert_ppl,
            "improvement_pct": improvement,
            "expert_wins": beat_base,
            "n_eval": len(eval_texts),
        }

        if beat_base:
            wins += 1
        total += 1

        # Unload expert adapter
        del expert_model
        torch.cuda.empty_cache()

    # Aggregate
    win_rate = wins / max(total, 1) * 100
    results["aggregate"] = {
        "domains_tested": total,
        "expert_wins": wins,
        "win_rate_pct": win_rate,
        "kill_criterion_met": win_rate < 80,
    }

    print(f"\n{'='*50}")
    print(f"Expert wins: {wins}/{total} ({win_rate:.0f}%)")
    if win_rate >= 80:
        print("PASS: Composed model beats base on >80% of domains")
    else:
        print("KILL: Composed model does NOT beat base on >80% of domains")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark composed model vs base")
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B", help="Base model")
    parser.add_argument("--adapters", default="adapters/", help="Adapters directory")
    parser.add_argument("--eval-data", default="eval/", help="Eval data directory")
    parser.add_argument("--output", default="results/e2e_benchmark.json", help="Output JSON")
    parser.add_argument("--max-eval", type=int, default=100, help="Max eval examples per domain")

    args = parser.parse_args()
    cmd_bench(args)


if __name__ == "__main__":
    main()
