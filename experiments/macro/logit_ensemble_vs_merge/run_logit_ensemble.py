#!/usr/bin/env python3
"""Logit-space ensembling vs weight-space merging diagnostic.

Key question: is the -3.67pp MMLU regression from WEIGHT interference or
DISTILLATION quality? If logit ensembling (run each adapter independently,
average logits) beats weight merging, the issue is weight-space interference.
If both regress equally, the issue is distillation quality.

Compares at N=5,10,25,50:
  (a) Weight-merge: add LoRA weights, run once
  (b) Logit-ensemble: run each adapter separately, average output logits
  (c) Base: no adapter

Depends: exp_distillation_pilot_50 (supported). No training needed.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ADAPTER_DIR = Path("/workspace/llm/adapters")
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
RESULTS_DIR = Path("/workspace/llm/results/logit_ensemble_vs_merge")
N_VALUES = [5, 10, 25, 50]
SEED = 42
N_SUBJECTS = 15  # Fewer subjects to keep runtime reasonable with ensembling

HELD_OUT_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering",
]


def load_base_model():
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


def get_adapter_ranking():
    benchmark_path = Path("/workspace/llm/results/pilot50_benchmark.json")
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            data = json.load(f)
        ranked = []
        for name, info in data.get("per_adapter", {}).items():
            ppl_improvement = info.get("ppl_improvement_pct", 0)
            ranked.append((name, ppl_improvement))
        ranked.sort(key=lambda x: x[1], reverse=True)
        if ranked:
            return [name for name, _ in ranked]
    adapters = sorted(
        [d.name for d in ADAPTER_DIR.iterdir()
         if d.is_dir() and (d / "adapter_config.json").exists()]
    )
    return adapters


def format_mmlu_prompt(example):
    question = example["question"]
    choices = example["choices"]
    prompt = f"{question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{'ABCD'[i]}. {choice}\n"
    prompt += "Answer:"
    return prompt


def get_choice_token_ids(tokenizer):
    choice_tokens = {}
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        choice_tokens[letter] = ids[0]
        ids_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if ids_space:
            choice_tokens[f" {letter}"] = ids_space[-1]
    return choice_tokens


def evaluate_logprob(model, tokenizer, subjects, max_per_subject=100):
    """Evaluate using next-token logprob scoring. Returns per-example logits too."""
    choice_tokens = get_choice_token_ids(tokenizer)
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
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
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
    return {"per_subject": results, "overall": {"correct": total_correct, "total": total_count, "accuracy": round(overall_acc, 4)}}


def evaluate_ensemble_logprob(base_model, tokenizer, adapter_names, subjects, max_per_subject=100):
    """Evaluate via logit-space ensembling: run each adapter, average logits."""
    choice_tokens = get_choice_token_ids(tokenizer)
    results = {}
    total_correct = 0
    total_count = 0

    # Pre-load all examples
    all_examples = {}
    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        except Exception:
            continue
        if len(ds) > max_per_subject:
            ds = ds.select(range(max_per_subject))
        all_examples[subject] = list(ds)

    # For each adapter, collect logits for all examples
    n_adapters = len(adapter_names)
    # Store accumulated logits per (subject, example_idx)
    accumulated_logits = {}  # (subject, idx) -> tensor

    for ai, adapter_name in enumerate(adapter_names):
        print(f"  Ensemble adapter {ai + 1}/{n_adapters}: {adapter_name}")
        adapter_path = str(ADAPTER_DIR / adapter_name)
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        for subject, examples in all_examples.items():
            for idx, ex in enumerate(examples):
                prompt = format_mmlu_prompt(ex)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits[0, -1].cpu()

                key = (subject, idx)
                if key not in accumulated_logits:
                    accumulated_logits[key] = logits
                else:
                    accumulated_logits[key] = accumulated_logits[key] + logits

        # Unload adapter
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Score using averaged logits
    for subject, examples in all_examples.items():
        correct = 0
        count = 0
        for idx, ex in enumerate(examples):
            key = (subject, idx)
            avg_logits = accumulated_logits[key] / n_adapters
            log_probs = torch.log_softmax(avg_logits, dim=-1)
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
    return {"per_subject": results, "overall": {"correct": total_correct, "total": total_count, "accuracy": round(overall_acc, 4)}}


def compose_adapters(base_model, adapter_names):
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


MAX_RUNTIME = int(os.environ.get("MAX_RUNTIME", 43200))  # 12h default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-subject", type=int, default=50, help="Reduce for speed with ensembling")
    args = parser.parse_args()

    if os.environ.get("SMOKE_TEST") == "1":
        args.max_per_subject = 5

    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    subjects = list(HELD_OUT_SUBJECTS)
    ranked_adapters = get_adapter_ranking()
    print(f"Found {len(ranked_adapters)} adapters")
    print(f"Evaluating {len(subjects)} MMLU subjects")

    # Phase 1: Base model
    print("\n=== Base model ===")
    t0 = time.time()
    base_model, tokenizer = load_base_model()
    base_results = evaluate_logprob(base_model, tokenizer, subjects, args.max_per_subject)
    base_acc = base_results["overall"]["accuracy"]
    print(f"Base: {base_acc:.4f} ({time.time() - t0:.0f}s)")

    all_results = {
        "base": base_results,
        "weight_merge": {},
        "logit_ensemble": {},
        "config": {
            "subjects": subjects,
            "max_per_subject": args.max_per_subject,
            "n_values": N_VALUES,
            "seed": SEED,
        },
    }

    # Free base model — each composition/ensemble reloads fresh
    del base_model
    torch.cuda.empty_cache()

    for n in N_VALUES:
        if n > len(ranked_adapters):
            continue
        if time.time() - start_time > MAX_RUNTIME:
            print(f"MAX_RUNTIME ({MAX_RUNTIME}s) exceeded, stopping early")
            break
        selected = ranked_adapters[:n]

        # Weight-space merge (reload fresh base each time — PeftModel modifies in-place)
        print(f"\n=== N={n} Weight Merge ===")
        t1 = time.time()
        fresh_model, _ = load_base_model()
        merged_model = compose_adapters(fresh_model, selected)
        merge_results = evaluate_logprob(merged_model, tokenizer, subjects, args.max_per_subject)
        merge_acc = merge_results["overall"]["accuracy"]
        merge_delta = (merge_acc - base_acc) * 100
        print(f"Weight merge N={n}: {merge_acc:.4f} ({merge_delta:+.2f}pp, {time.time() - t1:.0f}s)")
        del merged_model, fresh_model
        gc.collect()
        torch.cuda.empty_cache()

        all_results["weight_merge"][str(n)] = {
            "n_experts": n,
            "accuracy": merge_acc,
            "delta_vs_base_pp": round(merge_delta, 4),
            "elapsed_s": round(time.time() - t1, 1),
        }

        # Logit-space ensemble (reload fresh base — each adapter wrapping modifies it)
        print(f"\n=== N={n} Logit Ensemble ===")
        t2 = time.time()
        fresh_model2, _ = load_base_model()
        ensemble_results = evaluate_ensemble_logprob(
            fresh_model2, tokenizer, selected, subjects, args.max_per_subject
        )
        del fresh_model2
        gc.collect()
        torch.cuda.empty_cache()
        ens_acc = ensemble_results["overall"]["accuracy"]
        ens_delta = (ens_acc - base_acc) * 100
        print(f"Logit ensemble N={n}: {ens_acc:.4f} ({ens_delta:+.2f}pp, {time.time() - t2:.0f}s)")

        all_results["logit_ensemble"][str(n)] = {
            "n_experts": n,
            "accuracy": ens_acc,
            "delta_vs_base_pp": round(ens_delta, 4),
            "elapsed_s": round(time.time() - t2, 1),
        }

    # Summary
    print("\n=== Summary ===")
    print(f"{'N':>5}  {'Merge':>10}  {'Ensemble':>10}  {'Gap':>8}")
    for n_str in sorted(all_results["weight_merge"].keys(), key=int):
        m = all_results["weight_merge"][n_str]
        e = all_results["logit_ensemble"].get(n_str)
        if e:
            gap = e["delta_vs_base_pp"] - m["delta_vs_base_pp"]
            print(f"{n_str:>5}  {m['delta_vs_base_pp']:>+9.2f}pp  {e['delta_vs_base_pp']:>+9.2f}pp  {gap:>+7.2f}pp")

    # Diagnosis
    gaps = []
    for n_str in all_results["weight_merge"]:
        e = all_results["logit_ensemble"].get(n_str)
        if e:
            gaps.append(e["delta_vs_base_pp"] - all_results["weight_merge"][n_str]["delta_vs_base_pp"])

    avg_gap = np.mean(gaps) if gaps else 0.0
    diagnosis = "WEIGHT_INTERFERENCE" if avg_gap > 2.0 else "DISTILLATION_QUALITY" if avg_gap < 0.5 else "MIXED"
    all_results["diagnosis"] = {
        "avg_ensemble_vs_merge_gap_pp": round(avg_gap, 4),
        "verdict": diagnosis,
        "interpretation": {
            "WEIGHT_INTERFERENCE": "Ensembling beats merging → weight-space interference causes regression. Fix: selective/weighted composition.",
            "DISTILLATION_QUALITY": "Both regress similarly → distillation quality is the issue. Fix: improve training data/process.",
            "MIXED": "Partial improvement from ensembling → both factors contribute.",
        }[diagnosis],
    }
    print(f"\nDiagnosis: {diagnosis} (avg gap: {avg_gap:+.2f}pp)")
    print(f"  {all_results['diagnosis']['interpretation']}")

    out_path = RESULTS_DIR / "logit_ensemble_vs_merge.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
