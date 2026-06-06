#!/usr/bin/env python3
"""Selective composition MMLU eval: compose only relevant experts per question.

Tests whether targeted top-k expert selection (by domain relevance) avoids
the -3.67pp dilution seen with all-50 composition. Uses MMLU subject names
to match against adapter domain names.

Strategy: For each MMLU subject, find the k most relevant adapters (by name
similarity and semantic category) and compose only those. Compare against
base, all-50, and random-k baselines.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: "torch.nn.Module") -> None:
        atoms = target.split(".")
        mod = self
        for item in atoms[:-1]:
            mod = getattr(mod, item)
        setattr(mod, atoms[-1], module)
    torch.nn.Module.set_submodule = _set_submodule

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ADAPTER_DIR = Path("/workspace/llm/adapters")
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
RESULTS_DIR = Path("/workspace/llm/results/selective_composition_mmlu")
SEED = 42

# Domain-to-MMLU mapping: which adapters are relevant for which MMLU subjects
# Keys are adapter names, values are lists of MMLU subjects they should help
DOMAIN_MMLU_MAP = {
    # Science adapters → science MMLU subjects
    "physics": ["college_physics", "high_school_physics", "conceptual_physics", "astronomy"],
    "chemistry": ["college_chemistry", "high_school_chemistry"],
    "biology": ["college_biology", "high_school_biology", "anatomy", "virology", "medical_genetics"],
    "mathematics": ["college_mathematics", "high_school_mathematics", "elementary_mathematics",
                     "abstract_algebra", "high_school_statistics", "econometrics"],
    "computer-science": ["college_computer_science", "high_school_computer_science",
                         "machine_learning", "computer_security"],
    "medicine": ["college_medicine", "professional_medicine", "clinical_knowledge",
                 "anatomy", "nutrition", "human_aging"],
    # Professional adapters
    "law": ["professional_law", "international_law", "jurisprudence"],
    "economics": ["high_school_macroeconomics", "high_school_microeconomics", "econometrics"],
    "psychology": ["professional_psychology", "high_school_psychology", "human_sexuality"],
    "business": ["business_ethics", "management", "marketing", "professional_accounting"],
    "political-science": ["high_school_government_and_politics", "us_foreign_policy",
                          "security_studies", "public_relations"],
    "history": ["high_school_us_history", "high_school_world_history",
                "high_school_european_history", "prehistory"],
    "philosophy": ["philosophy", "formal_logic", "logical_fallacies", "moral_disputes",
                   "moral_scenarios"],
    "sociology": ["sociology", "human_sexuality", "global_facts"],
}

HELD_OUT_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
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


def format_mmlu_prompt(example):
    question = example["question"]
    choices = example["choices"]
    prompt = f"{question}\n"
    for i, choice in enumerate(choices):
        letter = "ABCD"[i]
        prompt += f"{letter}. {choice}\n"
    prompt += "Answer:"
    return prompt


def get_available_adapters():
    """Get list of available adapter names."""
    return sorted(
        d.name for d in ADAPTER_DIR.iterdir()
        if d.is_dir() and (d / "adapter_config.json").exists()
    )


def build_subject_to_adapters(available_adapters):
    """Build mapping from MMLU subject → relevant adapter names."""
    # Invert the DOMAIN_MMLU_MAP
    subject_map = {}
    for adapter_domain, subjects in DOMAIN_MMLU_MAP.items():
        for subject in subjects:
            if subject not in subject_map:
                subject_map[subject] = []
            # Find matching adapters (fuzzy match on domain name)
            for adapter in available_adapters:
                adapter_lower = adapter.lower().replace("-", " ").replace("_", " ")
                domain_lower = adapter_domain.lower().replace("-", " ")
                if domain_lower in adapter_lower or adapter_lower in domain_lower:
                    if adapter not in subject_map[subject]:
                        subject_map[subject].append(adapter)

    # Also try direct name matching for adapters not in the map
    for subject in HELD_OUT_SUBJECTS:
        if subject not in subject_map:
            subject_map[subject] = []
        subject_words = set(subject.lower().replace("_", " ").split())
        for adapter in available_adapters:
            adapter_words = set(adapter.lower().replace("-", " ").replace("_", " ").split())
            overlap = subject_words & adapter_words
            if len(overlap) >= 1 and adapter not in subject_map[subject]:
                subject_map[subject].append(adapter)

    return subject_map


def evaluate_mmlu_with_model(model, tokenizer, subjects, max_per_subject=50):
    """Evaluate on MMLU subjects."""
    from datasets import load_dataset

    choice_tokens = {}
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        choice_tokens[letter] = ids[0]
        ids_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if ids_space:
            choice_tokens[f" {letter}"] = ids_space[-1]

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

        for ex in ds:
            prompt = format_mmlu_prompt(ex)
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(model.device)

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
        "overall": {"correct": total_correct, "total": total_count, "accuracy": round(overall_acc, 4)},
    }


def compose_adapters(base_model, adapter_names):
    """Compose N adapters via PEFT weighted addition."""
    from peft import PeftModel

    adapter_paths = [str(ADAPTER_DIR / name) for name in adapter_names]
    model = PeftModel.from_pretrained(base_model, adapter_paths[0], adapter_name=adapter_names[0])
    for name, path in zip(adapter_names[1:], adapter_paths[1:]):
        model.load_adapter(path, adapter_name=name)
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
    parser.add_argument("--max-per-subject", type=int, default=50)
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument("--k-values", type=str, default="1,3,5", help="Comma-separated k values for top-k")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)
    k_values = [int(k) for k in args.k_values.split(",")]

    # Select subjects
    subjects = list(HELD_OUT_SUBJECTS)
    rng.shuffle(subjects)
    subjects = subjects[:args.subjects]
    print(f"Evaluating {len(subjects)} MMLU subjects, k_values={k_values}")

    available_adapters = get_available_adapters()
    subject_to_adapters = build_subject_to_adapters(available_adapters)
    print(f"Available adapters: {len(available_adapters)}")

    # Show mapping
    for subject in subjects[:5]:
        relevant = subject_to_adapters.get(subject, [])
        print(f"  {subject} → {relevant[:5]}")

    # Phase 1: Base model
    print("\n=== Phase 1: Base model ===")
    t0 = time.time()
    base_model, tokenizer = load_base_model()
    # Disable GC during heavy GPU inference (nanochat pattern: ~500ms/step saved)
    gc.disable()
    gc.collect()
    try:
        base_results = evaluate_mmlu_with_model(base_model, tokenizer, subjects, args.max_per_subject)
    finally:
        gc.enable()
        gc.collect()
    base_acc = base_results["overall"]["accuracy"]
    print(f"Base accuracy: {base_acc:.4f} ({time.time() - t0:.0f}s)")

    all_results = {
        "base": base_results,
        "selective": {},
        "random_baseline": {},
        "config": {
            "subjects": subjects,
            "max_per_subject": args.max_per_subject,
            "k_values": k_values,
            "seed": SEED,
            "subject_to_adapters": {s: subject_to_adapters.get(s, []) for s in subjects},
        },
    }

    # Phase 2: Selective composition per subject
    from peft import PeftModel
    from datasets import load_dataset

    choice_tokens = {}
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        choice_tokens[letter] = ids[0]
        ids_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if ids_space:
            choice_tokens[f" {letter}"] = ids_space[-1]

    # Disable GC during heavy GPU inference phases 2-3 (nanochat pattern: ~500ms/step saved)
    gc.disable()
    gc.collect()
    try:
        for k in k_values:
            print(f"\n=== Phase 2: Selective top-{k} composition ===")
            t1 = time.time()

            total_correct = 0
            total_count = 0
            per_subject = {}

            for subject in subjects:
                relevant = subject_to_adapters.get(subject, [])[:k]

                if not relevant:
                    # No relevant adapter — use base model
                    try:
                        ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
                    except Exception:
                        continue
                    if len(ds) > args.max_per_subject:
                        ds = ds.select(range(args.max_per_subject))

                    correct = 0
                    for ex in ds:
                        prompt = format_mmlu_prompt(ex)
                        inputs = tokenizer(
                            prompt, return_tensors="pt", truncation=True, max_length=512
                        ).to(base_model.device)
                        with torch.no_grad():
                            outputs = base_model(**inputs)
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

                    acc = correct / max(1, len(ds))
                    per_subject[subject] = {
                        "correct": correct, "total": len(ds), "accuracy": round(acc, 4),
                        "adapters_used": [], "note": "no_relevant_adapter",
                    }
                    total_correct += correct
                    total_count += len(ds)
                    print(f"  {subject}: {acc:.1%} (base, no relevant adapter)")
                    continue

                # Compose relevant adapters (reload fresh base — PeftModel modifies in-place)
                fresh_model, _ = load_base_model()
                try:
                    composed = compose_adapters(fresh_model, relevant)
                except Exception as e:
                    print(f"  {subject}: FAILED to compose {relevant}: {e}")
                    del fresh_model
                    torch.cuda.empty_cache()
                    continue

                try:
                    ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
                except Exception:
                    del composed, fresh_model
                    torch.cuda.empty_cache()
                    continue
                if len(ds) > args.max_per_subject:
                    ds = ds.select(range(args.max_per_subject))

                correct = 0
                for ex in ds:
                    prompt = format_mmlu_prompt(ex)
                    inputs = tokenizer(
                        prompt, return_tensors="pt", truncation=True, max_length=512
                    ).to(composed.device)
                    with torch.no_grad():
                        outputs = composed(**inputs)
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

                acc = correct / max(1, len(ds))
                delta = (acc - base_results["per_subject"].get(subject, {}).get("accuracy", base_acc)) * 100
                per_subject[subject] = {
                    "correct": correct, "total": len(ds), "accuracy": round(acc, 4),
                    "adapters_used": relevant, "delta_vs_base_pp": round(delta, 4),
                }
                total_correct += correct
                total_count += len(ds)
                print(f"  {subject}: {acc:.1%} ({delta:+.2f}pp) adapters={relevant}")

                del composed, fresh_model
                torch.cuda.empty_cache()

            overall_acc = total_correct / max(1, total_count) if total_count > 0 else 0.0
            delta_pp = (overall_acc - base_acc) * 100
            all_results["selective"][str(k)] = {
                "k": k,
                "per_subject": per_subject,
                "overall": {"correct": total_correct, "total": total_count, "accuracy": round(overall_acc, 4)},
                "delta_vs_base_pp": round(delta_pp, 4),
            }
            print(f"Top-{k} selective: {overall_acc:.4f} ({delta_pp:+.2f}pp) [{time.time() - t1:.0f}s]")

        # Phase 3: Random-k baseline (for comparison)
        for k in [3]:
            print(f"\n=== Phase 3: Random-{k} baseline ===")
            t2 = time.time()

            total_correct = 0
            total_count = 0
            per_subject = {}

            for subject in subjects:
                # Pick k random adapters
                random_adapters = list(rng.choice(available_adapters, size=min(k, len(available_adapters)), replace=False))

                fresh_model, _ = load_base_model()
                try:
                    composed = compose_adapters(fresh_model, random_adapters)
                except Exception as e:
                    print(f"  {subject}: FAILED: {e}")
                    del fresh_model
                    torch.cuda.empty_cache()
                    continue

                try:
                    ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
                except Exception:
                    del composed, fresh_model
                    torch.cuda.empty_cache()
                    continue
                if len(ds) > args.max_per_subject:
                    ds = ds.select(range(args.max_per_subject))

                correct = 0
                for ex in ds:
                    prompt = format_mmlu_prompt(ex)
                    inputs = tokenizer(
                        prompt, return_tensors="pt", truncation=True, max_length=512
                    ).to(composed.device)
                    with torch.no_grad():
                        outputs = composed(**inputs)
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

                acc = correct / max(1, len(ds))
                per_subject[subject] = {
                    "correct": correct, "total": len(ds), "accuracy": round(acc, 4),
                    "adapters_used": random_adapters,
                }
                total_correct += correct
                total_count += len(ds)

                del composed, fresh_model
                torch.cuda.empty_cache()

            overall_acc = total_correct / max(1, total_count) if total_count > 0 else 0.0
            delta_pp = (overall_acc - base_acc) * 100
            all_results["random_baseline"][str(k)] = {
                "k": k,
                "per_subject": per_subject,
                "overall": {"correct": total_correct, "total": total_count, "accuracy": round(overall_acc, 4)},
                "delta_vs_base_pp": round(delta_pp, 4),
            }
            print(f"Random-{k}: {overall_acc:.4f} ({delta_pp:+.2f}pp) [{time.time() - t2:.0f}s]")
    finally:
        gc.enable()
        gc.collect()

    # Summary
    print("\n=== Summary ===")
    print(f"Base: {base_acc:.4f}")
    for k_str, res in sorted(all_results["selective"].items(), key=lambda x: int(x[0])):
        print(f"Selective top-{k_str}: {res['overall']['accuracy']:.4f} ({res['delta_vs_base_pp']:+.2f}pp)")
    for k_str, res in all_results["random_baseline"].items():
        print(f"Random-{k_str}: {res['overall']['accuracy']:.4f} ({res['delta_vs_base_pp']:+.2f}pp)")

    # Kill criteria
    selective_deltas = {int(k): v["delta_vs_base_pp"] for k, v in all_results["selective"].items()}
    k1_selective_beats_all50 = any(d > -3.67 for d in selective_deltas.values())
    k2_selective_positive = any(d > 0 for d in selective_deltas.values())

    all_results["kill_criteria"] = {
        "k1_selective_beats_all50": k1_selective_beats_all50,
        "k2_selective_positive": k2_selective_positive,
        "verdict": "PASS" if k2_selective_positive else "FAIL",
    }

    print(f"\nK1 (selective > all-50): {'PASS' if k1_selective_beats_all50 else 'FAIL'}")
    print(f"K2 (selective > base): {'PASS' if k2_selective_positive else 'FAIL'}")

    # Save
    out_path = RESULTS_DIR / "selective_composition_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
