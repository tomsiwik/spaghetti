#!/usr/bin/env python3
"""Small-N composition held-out eval: sweep N=2,3,5,10,25,50 on MMLU.

Tests whether N=50 MMLU regression (-3.67pp) is caused by dilution (1/N scaling)
or fundamental composition failure. If small N shows positive delta, SOLE needs
selective composition (top-k) rather than compose-all.

Uses existing pilot50 adapters. No training needed.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ADAPTER_DIR = Path("/workspace/llm/adapters")
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
RESULTS_DIR = Path("/workspace/llm/results/small_n_held_out_eval")
N_VALUES = [2, 3, 5, 10, 25, 50]
SEED = 42

# MMLU subjects for held-out eval (not used in training)
HELD_OUT_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def load_base_model():
    """Load quantized base model."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def get_adapter_quality_ranking():
    """Rank adapters by individual domain quality (PPL on training data).

    Falls back to alphabetical if no benchmark data available.
    """
    benchmark_path = Path("/workspace/llm/results/pilot50_benchmark.json")
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            data = json.load(f)
        # Sort by PPL improvement (best first)
        ranked = []
        for name, info in data.get("per_adapter", {}).items():
            ppl_improvement = info.get("ppl_improvement_pct", 0)
            ranked.append((name, ppl_improvement))
        ranked.sort(key=lambda x: x[1], reverse=True)
        if ranked:
            return [name for name, _ in ranked]

    # Fallback: alphabetical
    adapters = sorted(
        [d.name for d in ADAPTER_DIR.iterdir() if d.is_dir() and (d / "adapter_config.json").exists()]
    )
    return adapters


def format_mmlu_prompt(example):
    """Format MMLU example as multiple choice prompt."""
    question = example["question"]
    choices = example["choices"]
    prompt = f"{question}\n"
    for i, choice in enumerate(choices):
        letter = "ABCD"[i]
        prompt += f"{letter}. {choice}\n"
    prompt += "Answer:"
    return prompt


def evaluate_mmlu_logprob(model, tokenizer, subjects, max_per_subject=100):
    """Evaluate model on MMLU subjects using log-prob scoring."""
    from datasets import load_dataset

    results = {}
    total_correct = 0
    total_count = 0

    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  Skip {subject}: {e}")
            continue

        if len(ds) > max_per_subject:
            ds = ds.select(range(max_per_subject))

        correct = 0
        count = 0

        # Get token IDs for A, B, C, D
        choice_tokens = {}
        for letter in "ABCD":
            ids = tokenizer.encode(letter, add_special_tokens=False)
            choice_tokens[letter] = ids[0]
            ids_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
            if ids_space:
                choice_tokens[f" {letter}"] = ids_space[-1]

        for ex in ds:
            prompt = format_mmlu_prompt(ex)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(
                model.device
            )

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]
                log_probs = torch.log_softmax(logits, dim=-1)

            scores = {}
            for letter in "ABCD":
                tid = choice_tokens[letter]
                tid_space = choice_tokens.get(f" {letter}", tid)
                scores[letter] = max(log_probs[tid].item(), log_probs[tid_space].item())

            pred = max(scores, key=scores.get)
            gold = "ABCD"[ex["answer"]]
            correct += int(pred == gold)
            count += 1

        acc = correct / max(1, count)
        results[subject] = {"correct": correct, "total": count, "accuracy": round(acc, 4)}
        total_correct += correct
        total_count += count
        print(f"  {subject}: {acc:.1%} ({correct}/{count})")

    overall_acc = total_correct / max(1, total_count) if total_count > 0 else 0.0
    return {
        "per_subject": results,
        "overall": {
            "correct": total_correct,
            "total": total_count,
            "accuracy": round(overall_acc, 4),
        },
    }


def compose_adapters(base_model, adapter_names):
    """Compose N adapters via PEFT weighted addition."""
    from peft import PeftModel

    adapter_paths = [str(ADAPTER_DIR / name) for name in adapter_names]

    # Load first adapter
    model = PeftModel.from_pretrained(base_model, adapter_paths[0], adapter_name=adapter_names[0])

    # Load remaining
    for name, path in zip(adapter_names[1:], adapter_paths[1:]):
        model.load_adapter(path, adapter_name=name)

    # Merge with equal weights
    model.add_weighted_adapter(
        adapters=list(adapter_names),
        weights=[1.0 / len(adapter_names)] * len(adapter_names),
        adapter_name="composed",
        combination_type="linear",
    )
    model.set_adapter("composed")
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-subject", type=int, default=100)
    parser.add_argument("--subjects", type=int, default=20, help="Number of MMLU subjects to eval")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    # Select subset of subjects for speed
    rng = np.random.RandomState(SEED)
    subjects = list(HELD_OUT_SUBJECTS)
    rng.shuffle(subjects)
    subjects = subjects[: args.subjects]
    print(f"Evaluating {len(subjects)} MMLU subjects: {subjects[:5]}...")

    # Get adapter ranking
    ranked_adapters = get_adapter_quality_ranking()
    print(f"Found {len(ranked_adapters)} adapters, top-5: {ranked_adapters[:5]}")

    # Phase 1: Base model eval
    print("\n=== Phase 1: Base model evaluation ===")
    t0 = time.time()
    base_model, tokenizer = load_base_model()
    base_results = evaluate_mmlu_logprob(base_model, tokenizer, subjects, args.max_per_subject)
    base_acc = base_results["overall"]["accuracy"]
    print(f"Base accuracy: {base_acc:.4f} ({time.time() - t0:.0f}s)")

    all_results = {"base": base_results, "compositions": {}, "config": {
        "subjects": subjects,
        "max_per_subject": args.max_per_subject,
        "n_values": N_VALUES,
        "seed": SEED,
        "base_model": BASE_MODEL,
        "adapter_ranking": ranked_adapters[:50],
    }}

    # Phase 2: Composition eval at each N
    # Free base model before composition loop — each iteration reloads fresh
    del base_model
    torch.cuda.empty_cache()

    for n in N_VALUES:
        if n > len(ranked_adapters):
            print(f"\nSkipping N={n}: only {len(ranked_adapters)} adapters available")
            continue

        print(f"\n=== Phase 2: N={n} composition ===")
        t1 = time.time()

        # Reload fresh base model (PeftModel modifies base in-place)
        fresh_model, _ = load_base_model()

        # Select top-N adapters by quality
        selected = ranked_adapters[:n]
        print(f"Selected adapters: {selected[:5]}{'...' if n > 5 else ''}")

        # Compose
        composed_model = compose_adapters(fresh_model, selected)
        comp_results = evaluate_mmlu_logprob(
            composed_model, tokenizer, subjects, args.max_per_subject
        )
        comp_acc = comp_results["overall"]["accuracy"]
        delta_pp = (comp_acc - base_acc) * 100  # percentage points

        print(f"N={n}: accuracy={comp_acc:.4f}, delta={delta_pp:+.2f}pp ({time.time() - t1:.0f}s)")

        all_results["compositions"][str(n)] = {
            "n_experts": n,
            "adapters_used": selected,
            "eval_results": comp_results,
            "delta_vs_base_pp": round(delta_pp, 4),
        }

        # Cleanup
        del composed_model, fresh_model
        torch.cuda.empty_cache()

    # Phase 3: Analysis
    print("\n=== Results Summary ===")
    print(f"Base: {base_acc:.4f}")
    for n_str, res in sorted(all_results["compositions"].items(), key=lambda x: int(x[0])):
        n = int(n_str)
        delta = res["delta_vs_base_pp"]
        acc = res["eval_results"]["overall"]["accuracy"]
        print(f"N={n:3d}: {acc:.4f} ({delta:+.2f}pp)")

    # Kill criteria assessment
    deltas = {int(k): v["delta_vs_base_pp"] for k, v in all_results["compositions"].items()}
    k1_pass = True  # Even N=2 doesn't regress >2pp
    k2_pass = True  # Quality monotonically decreases with N

    if 2 in deltas and deltas[2] < -2.0:
        k1_pass = False
        print("\nK1 KILLED: N=2 composition shows >2pp MMLU regression")

    # Check monotonic decrease
    sorted_ns = sorted(deltas.keys())
    for i in range(1, len(sorted_ns)):
        if deltas[sorted_ns[i]] > deltas[sorted_ns[i - 1]] + 1.0:  # Allow 1pp noise
            k2_pass = False
            print(f"\nK2 KILLED: Non-monotonic at N={sorted_ns[i]}")

    verdict = "PASS" if k1_pass and k2_pass else "FAIL"
    all_results["kill_criteria"] = {
        "k1_n2_regression_below_2pp": k1_pass,
        "k2_monotonic_decrease": k2_pass,
        "verdict": verdict,
    }

    print(f"\nVerdict: {verdict}")
    print(f"  K1 (N=2 safe): {'PASS' if k1_pass else 'KILLED'}")
    print(f"  K2 (monotonic): {'PASS' if k2_pass else 'KILLED'}")

    # Save
    out_path = RESULTS_DIR / "small_n_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
