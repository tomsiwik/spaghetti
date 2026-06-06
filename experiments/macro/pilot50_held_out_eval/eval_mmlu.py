#!/usr/bin/env python3
"""
Held-out MMLU evaluation for pilot 50 experts.

Evaluates base model and base+adapter on MMLU test split using log-prob scoring.
Each adapter is evaluated on its mapped MMLU categories.

Usage:
    python eval_mmlu.py --adapter python --out results/mmlu_python.json
    python eval_mmlu.py --all --out results/mmlu_all.json
"""

import argparse
import json
import time
import os
import sys
import torch

# ── Monkey-patch set_submodule for PyTorch builds missing it ──────────────────
if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: "torch.nn.Module") -> None:
        atoms = target.split(".")
        mod = self
        for item in atoms[:-1]:
            mod = getattr(mod, item)
        setattr(mod, atoms[-1], module)
    torch.nn.Module.set_submodule = _set_submodule
    print(f"[patch] set_submodule monkey-patched onto torch {torch.__version__}")
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Domain-to-MMLU mapping ──────────────────────────────────────────────
# Maps each pilot adapter to the most relevant MMLU test categories.
# Only adapters with clear MMLU counterparts are included.
# This is the core mapping that determines which adapters get evaluated.

DOMAIN_TO_MMLU = {
    # Science
    "physics": ["high_school_physics", "college_physics", "conceptual_physics"],
    "chemistry": ["high_school_chemistry", "college_chemistry"],
    "biology": ["high_school_biology", "college_biology"],
    "math": ["high_school_mathematics", "college_mathematics", "elementary_mathematics", "abstract_algebra"],
    "statistics": ["high_school_statistics", "econometrics"],
    "astronomy": ["astronomy"],
    "genetics": ["medical_genetics"],
    "ecology": ["high_school_biology"],  # closest match
    "neuroscience": ["high_school_psychology", "professional_psychology"],
    # Professional
    "legal": ["professional_law", "international_law", "jurisprudence"],
    "medical": ["professional_medicine", "clinical_knowledge", "college_medicine", "anatomy"],
    "finance": ["high_school_macroeconomics", "high_school_microeconomics"],
    "accounting": ["professional_accounting"],
    "marketing": ["marketing"],
    "cybersecurity": ["computer_security", "security_studies"],
    # Reasoning
    "logic-puzzles": ["formal_logic", "logical_fallacies"],
    "ethics": ["business_ethics", "moral_disputes", "moral_scenarios"],
    "abstract-math": ["abstract_algebra", "college_mathematics"],
    # Programming (mapped to CS MMLU subsets)
    "python": ["high_school_computer_science", "college_computer_science", "machine_learning"],
    "cpp": ["high_school_computer_science", "college_computer_science"],
    "java": ["high_school_computer_science", "college_computer_science"],
    "javascript": ["high_school_computer_science"],
    "rust": ["college_computer_science"],
}

# Adapters with NO good MMLU mapping (evaluated via HumanEval or skipped)
NO_MMLU_MAPPING = [
    "bash", "go", "sql", "swift", "typescript",  # code but no MMLU match
    "creative-fiction", "technical-writing", "academic-writing", "journalism",
    "copywriting", "poetry", "screenplay", "speechwriting", "grant-writing",
    "documentation",  # writing domains
    "debate", "game-theory", "systems-thinking", "critical-analysis",
    "causal-reasoning", "analogical-reasoning", "spatial-reasoning",  # reasoning w/o match
    "hr", "project-management", "data-engineering", "devops", "geology",  # misc
]


def format_mmlu_prompt(example):
    """Format MMLU example as multiple-choice prompt.

    Nanochat insight (references/nanochat/tasks/common.py:112-131):
    - Letter AFTER choice helps small models bind letter to answer
    - No whitespace before letter ensures tokenizer produces same token
      for "A" in prompt and expected answer (critical for log-prob scoring)
    - "=" delimiter without spaces for consistent tokenization
    """
    question = example['question']
    choices = example['choices']
    letters = ('A', 'B', 'C', 'D')
    prompt = f"Multiple Choice question: {question}\n"
    prompt += "".join(f"- {choice}={letter}\n" for letter, choice in zip(letters, choices))
    prompt += "\nRespond only with the letter of the correct answer."
    return prompt


def evaluate_logprob(model, tokenizer, dataset, device, verbose=True):
    """Evaluate model on dataset using log-prob scoring of A/B/C/D.

    Nanochat insight: the prompt format uses "choice=A" so the answer letter
    appears WITHOUT a leading space. We must match the exact same tokenization
    for the answer token. For Qwen2.5, "A" and " A" are different tokens.
    """
    correct = 0
    total = 0
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    # Get ALL possible token IDs for each letter (bare, with space, lowercase)
    # Nanochat: ensure the token we score matches what appears in the prompt
    choice_token_ids = {}
    for letter in "ABCD":
        ids = set()
        for variant in [letter, f" {letter}", f"={letter}"]:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                ids.add(encoded[-1])  # Last token is the letter
        choice_token_ids[letter] = list(ids)

    for ex in dataset:
        prompt = format_mmlu_prompt(ex)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]  # last token logits
            log_probs = torch.log_softmax(logits, dim=-1)

        # Score each choice: take max log-prob across all token variants
        scores = {}
        for letter in "ABCD":
            token_scores = [log_probs[tid].item() for tid in choice_token_ids[letter]]
            scores[letter] = max(token_scores) if token_scores else float('-inf')

        pred = max(scores, key=scores.get)
        gold = letter_map[ex["answer"]] if isinstance(ex["answer"], int) else str(ex["answer"]).strip().upper()

        correct += int(pred == gold)
        total += 1

        if verbose and total % 50 == 0:
            print(f"  [{total}] acc={correct/total:.3f}")

    return {"correct": correct, "total": total, "accuracy": correct / max(1, total)}


def load_base_model(model_path, device):
    """Load quantized base model.

    Nanochat pattern: explicit dtype detection. bf16 on SM 80+ (A5000/4090/A100/H100),
    fp16 on older hardware. bf16 has better numerical properties for eval accuracy.
    """
    # Nanochat: explicit precision, no autocast
    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    model.eval()
    return model, tokenizer


def load_adapter(base_model, adapter_path):
    """Load LoRA adapter on top of base model."""
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model


def evaluate_domain(model, tokenizer, domain, device, max_per_subset=None):
    """Evaluate model on all MMLU subsets mapped to a domain."""
    subsets = DOMAIN_TO_MMLU.get(domain, [])
    if not subsets:
        return None

    results = {}
    domain_correct = 0
    domain_total = 0

    for subset in subsets:
        try:
            ds = load_dataset("cais/mmlu", subset, split="test")
            if max_per_subset:
                ds = ds.select(range(min(len(ds), max_per_subset)))
            r = evaluate_logprob(model, tokenizer, ds, device, verbose=False)
            results[subset] = r
            domain_correct += r["correct"]
            domain_total += r["total"]
            print(f"    {subset}: {r['accuracy']:.3f} ({r['correct']}/{r['total']})")
        except Exception as e:
            print(f"    {subset}: ERROR - {e}")
            results[subset] = {"error": str(e)}

    agg_acc = domain_correct / max(1, domain_total)
    return {
        "subsets": results,
        "aggregate": {
            "correct": domain_correct,
            "total": domain_total,
            "accuracy": agg_acc,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="MMLU held-out eval for pilot 50 experts")
    parser.add_argument("--base-model", default="/workspace/models/Qwen2.5-7B",
                        help="Path to base model")
    parser.add_argument("--adapter-dir", default="/workspace/llm/adapters",
                        help="Directory containing adapter subdirectories")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Single adapter to evaluate (or --all)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all adapters with MMLU mappings")
    parser.add_argument("--max-per-subset", type=int, default=None,
                        help="Max examples per MMLU subset (for quick testing)")
    parser.add_argument("--out", type=str, default="results/mmlu_held_out.json",
                        help="Output JSON file")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip base model eval (use cached results)")
    parser.add_argument("--base-cache", type=str, default=None,
                        help="Path to cached base results JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Determine which adapters to evaluate
    if args.all:
        adapters = sorted(DOMAIN_TO_MMLU.keys())
    elif args.adapter:
        adapters = [args.adapter]
    else:
        print("Specify --adapter <name> or --all")
        sys.exit(1)

    print(f"Evaluating {len(adapters)} adapters: {adapters}")

    # Load base model once
    print(f"\nLoading base model: {args.base_model}")
    base_model, tokenizer = load_base_model(args.base_model, device)

    # Evaluate base model on all needed subsets
    all_subsets = set()
    for adapter in adapters:
        all_subsets.update(DOMAIN_TO_MMLU.get(adapter, []))
    all_subsets = sorted(all_subsets)
    print(f"MMLU subsets needed: {len(all_subsets)}")

    base_results = {}
    if args.base_cache and os.path.exists(args.base_cache):
        with open(args.base_cache) as f:
            base_results = json.load(f).get("base_results", {})
        print(f"Loaded {len(base_results)} cached base results")

    if not args.skip_base:
        print("\n=== Evaluating BASE model ===")
        for subset in all_subsets:
            if subset in base_results:
                print(f"  {subset}: cached {base_results[subset]['accuracy']:.3f}")
                continue
            try:
                ds = load_dataset("cais/mmlu", subset, split="test")
                if args.max_per_subset:
                    ds = ds.select(range(min(len(ds), args.max_per_subset)))
                r = evaluate_logprob(base_model, tokenizer, ds, device, verbose=False)
                base_results[subset] = r
                print(f"  {subset}: {r['accuracy']:.3f} ({r['correct']}/{r['total']})")
            except Exception as e:
                print(f"  {subset}: ERROR - {e}")
                base_results[subset] = {"error": str(e)}

        # Save intermediate base results
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out.replace(".json", "_base_cache.json"), "w") as f:
            json.dump({"base_results": base_results}, f, indent=2)
        print("Base results cached.")

    # Evaluate each adapter
    results = {
        "experiment": "pilot50_held_out_mmlu",
        "base_model": args.base_model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_results": base_results,
        "adapter_results": {},
        "comparisons": {},
    }

    for adapter_name in adapters:
        adapter_path = os.path.join(args.adapter_dir, adapter_name)
        if not os.path.exists(adapter_path):
            print(f"\n  SKIP {adapter_name}: adapter not found at {adapter_path}")
            continue

        print(f"\n=== Evaluating ADAPTER: {adapter_name} ===")
        try:
            # Load adapter
            adapted_model = load_adapter(base_model, adapter_path)

            # Evaluate on mapped subsets
            subsets = DOMAIN_TO_MMLU.get(adapter_name, [])
            adapter_subset_results = {}

            for subset in subsets:
                try:
                    ds = load_dataset("cais/mmlu", subset, split="test")
                    if args.max_per_subset:
                        ds = ds.select(range(min(len(ds), args.max_per_subset)))
                    r = evaluate_logprob(adapted_model, tokenizer, ds, device, verbose=False)
                    adapter_subset_results[subset] = r
                    print(f"  {subset}: {r['accuracy']:.3f} ({r['correct']}/{r['total']})")
                except Exception as e:
                    print(f"  {subset}: ERROR - {e}")
                    adapter_subset_results[subset] = {"error": str(e)}

            results["adapter_results"][adapter_name] = adapter_subset_results

            # Compute comparison
            comparison = {}
            for subset in subsets:
                base_acc = base_results.get(subset, {}).get("accuracy")
                adapter_acc = adapter_subset_results.get(subset, {}).get("accuracy")
                if base_acc is not None and adapter_acc is not None:
                    comparison[subset] = {
                        "base_acc": base_acc,
                        "adapter_acc": adapter_acc,
                        "delta": adapter_acc - base_acc,
                        "delta_pct": (adapter_acc - base_acc) * 100,
                        "adapter_wins": adapter_acc > base_acc,
                    }
            results["comparisons"][adapter_name] = comparison

            # Print summary for this adapter
            wins = sum(1 for c in comparison.values() if c["adapter_wins"])
            total = len(comparison)
            avg_delta = sum(c["delta_pct"] for c in comparison.values()) / max(1, total)
            print(f"  Summary: {wins}/{total} subsets improved, avg delta: {avg_delta:+.1f}pp")

            # Unload adapter to free memory
            del adapted_model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR loading adapter {adapter_name}: {e}")
            import traceback
            traceback.print_exc()
            results["adapter_results"][adapter_name] = {"error": str(e)}

    # Aggregate across all adapters
    all_comparisons = []
    for adapter_name, comp in results["comparisons"].items():
        for subset, c in comp.items():
            all_comparisons.append(c)

    if all_comparisons:
        total_wins = sum(1 for c in all_comparisons if c["adapter_wins"])
        total_subsets = len(all_comparisons)
        avg_delta_all = sum(c["delta_pct"] for c in all_comparisons) / total_subsets

        # Per-adapter wins
        adapter_wins = 0
        adapter_total = 0
        for adapter_name, comp in results["comparisons"].items():
            if comp:
                avg_d = sum(c["delta_pct"] for c in comp.values()) / len(comp)
                if avg_d > 0:
                    adapter_wins += 1
                adapter_total += 1

        results["aggregate"] = {
            "total_subset_comparisons": total_subsets,
            "subset_wins": total_wins,
            "subset_win_rate_pct": total_wins / total_subsets * 100,
            "avg_delta_pct": avg_delta_all,
            "adapter_count": adapter_total,
            "adapters_with_positive_delta": adapter_wins,
            "adapter_win_rate_pct": adapter_wins / max(1, adapter_total) * 100,
            "kill_criteria": {
                "win_rate_below_80": adapter_wins / max(1, adapter_total) * 100 < 80,
                "avg_improvement_below_2": avg_delta_all < 2.0,
            }
        }

        print(f"\n=== AGGREGATE ===")
        print(f"Adapters evaluated: {adapter_total}")
        print(f"Adapters with positive delta: {adapter_wins}/{adapter_total} "
              f"({adapter_wins/max(1,adapter_total)*100:.0f}%)")
        print(f"Average delta: {avg_delta_all:+.2f}pp")
        print(f"Kill criteria met: {results['aggregate']['kill_criteria']}")

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
