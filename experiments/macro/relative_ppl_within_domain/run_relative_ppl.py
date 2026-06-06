#!/usr/bin/env python3
"""Within-domain PPL ranking: does answer-only PPL rank adapters correctly WITHIN a domain?

Motivation: Answer-only PPL failed cross-domain at macro (r=-0.63) but this may be
because different domains have incomparable PPL scales. Clone-and-compete only needs
within-domain ranking. If that's preserved, PPL is still usable for evolution.

Kill criteria:
  K1: within-domain PPL ranking Kendall tau < 0.5 vs accuracy ranking
  K2: ranking flips occur for >30% of adapter pairs
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ADAPTER_DIR = Path("/workspace/llm/adapters")
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
RESULTS_DIR = Path("/workspace/llm/results/relative_ppl_within_domain")
SEED = 42
N_EXPERTS = 15


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
    """Format MMLU example as multiple choice prompt."""
    question = example["question"]
    choices = example["choices"]
    prompt = f"{question}\n"
    for i, choice in enumerate(choices):
        letter = "ABCD"[i]
        prompt += f"{letter}. {choice}\n"
    prompt += "Answer:"
    return prompt


def evaluate_adapter_on_subject(model, tokenizer, dataset, choice_tokens):
    """Evaluate adapter on one MMLU subject. Returns accuracy and answer-only PPL."""
    correct = 0
    total_nll = 0.0
    count = 0

    for ex in dataset:
        prompt = format_mmlu_prompt(ex)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]
            log_probs = torch.log_softmax(logits, dim=-1)

        # Accuracy
        scores = {}
        for letter in "ABCD":
            tid = choice_tokens[letter]
            tid_space = choice_tokens.get(f" {letter}", tid)
            scores[letter] = max(log_probs[tid].item(), log_probs[tid_space].item())

        pred = max(scores, key=scores.get)
        gold = "ABCD"[ex["answer"]]
        correct += int(pred == gold)

        # Answer-only PPL (NLL of correct answer token)
        gold_letter = "ABCD"[ex["answer"]]
        answer_ids = tokenizer.encode(gold_letter, add_special_tokens=False)
        if answer_ids:
            total_nll -= log_probs[answer_ids[0]].item()
            count += 1

    accuracy = correct / max(1, len(dataset))
    ppl = np.exp(total_nll / max(1, count))
    return accuracy, ppl


def get_adapter_list():
    """Get available adapters."""
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
    return sorted(
        d.name for d in ADAPTER_DIR.iterdir()
        if d.is_dir() and (d / "adapter_config.json").exists()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-experts", type=int, default=N_EXPERTS)
    parser.add_argument("--max-per-subject", type=int, default=100)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 10 diverse MMLU subjects
    subjects = [
        "abstract_algebra", "anatomy", "college_computer_science",
        "college_physics", "econometrics", "high_school_biology",
        "high_school_us_history", "machine_learning",
        "professional_medicine", "world_religions",
    ]

    from datasets import load_dataset

    print(f"Loading {len(subjects)} MMLU subjects...")
    subject_datasets = {}
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
            if len(ds) > args.max_per_subject:
                ds = ds.select(range(args.max_per_subject))
            subject_datasets[subj] = ds
            print(f"  {subj}: {len(ds)} examples")
        except Exception as e:
            print(f"  Skip {subj}: {e}")

    adapters = get_adapter_list()[:args.n_experts]
    print(f"Testing {len(adapters)} adapters across {len(subject_datasets)} subjects")

    # Precompute choice tokens
    base_model, tokenizer = load_base_model()
    choice_tokens = {}
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        choice_tokens[letter] = ids[0]
        ids_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if ids_space:
            choice_tokens[f" {letter}"] = ids_space[-1]

    # Evaluate base model
    print("\n=== Base model ===")
    base_scores = {}
    for subj, ds in subject_datasets.items():
        acc, ppl = evaluate_adapter_on_subject(base_model, tokenizer, ds, choice_tokens)
        base_scores[subj] = {"accuracy": round(acc, 4), "answer_ppl": round(ppl, 4)}
        print(f"  {subj}: acc={acc:.3f}, ppl={ppl:.2f}")

    del base_model
    torch.cuda.empty_cache()

    # Evaluate each adapter
    from peft import PeftModel

    # adapter_name -> subject -> {accuracy, ppl}
    adapter_scores = {}

    for adapter_idx, adapter_name in enumerate(adapters):
        print(f"\n=== Adapter {adapter_idx+1}/{len(adapters)}: {adapter_name} ===")
        t0 = time.time()

        fresh_model, _ = load_base_model()
        adapter_path = str(ADAPTER_DIR / adapter_name)
        try:
            peft_model = PeftModel.from_pretrained(
                fresh_model, adapter_path, adapter_name=adapter_name
            )
            peft_model.set_adapter(adapter_name)
            peft_model.eval()
        except Exception as e:
            print(f"  Failed: {e}")
            del fresh_model
            torch.cuda.empty_cache()
            continue

        adapter_scores[adapter_name] = {}
        for subj, ds in subject_datasets.items():
            acc, ppl = evaluate_adapter_on_subject(peft_model, tokenizer, ds, choice_tokens)
            adapter_scores[adapter_name][subj] = {
                "accuracy": round(acc, 4),
                "answer_ppl": round(ppl, 4),
            }
            print(f"  {subj}: acc={acc:.3f}, ppl={ppl:.2f}")

        print(f"  [{time.time() - t0:.0f}s]")
        del peft_model, fresh_model
        torch.cuda.empty_cache()

    # Phase 3: Within-domain ranking analysis
    print("\n=== Within-Domain Ranking Analysis ===")
    from scipy.stats import kendalltau

    adapter_names = list(adapter_scores.keys())
    within_domain_results = {}
    all_taus = []
    all_flip_rates = []

    for subj in subject_datasets:
        accs = [adapter_scores[n][subj]["accuracy"] for n in adapter_names if subj in adapter_scores[n]]
        ppls = [adapter_scores[n][subj]["answer_ppl"] for n in adapter_names if subj in adapter_scores[n]]

        if len(accs) < 3:
            continue

        acc_arr = np.array(accs)
        ppl_arr = np.array(ppls)

        # Rankings (higher accuracy = better, lower PPL = better)
        acc_rank = np.argsort(np.argsort(-acc_arr))
        ppl_rank = np.argsort(np.argsort(ppl_arr))  # lower PPL = rank 0

        tau, p_value = kendalltau(acc_rank, ppl_rank)
        all_taus.append(tau)

        # Pairwise flip rate: fraction of pairs where PPL and accuracy disagree on ordering
        n = len(accs)
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                acc_order = np.sign(accs[i] - accs[j])
                ppl_order = np.sign(ppls[j] - ppls[i])  # lower PPL = better
                if acc_order == ppl_order:
                    concordant += 1
                elif acc_order != 0 and ppl_order != 0:
                    discordant += 1

        total_pairs = concordant + discordant
        flip_rate = discordant / max(1, total_pairs)
        all_flip_rates.append(flip_rate)

        within_domain_results[subj] = {
            "kendall_tau": round(tau, 4),
            "p_value": round(p_value, 4),
            "flip_rate": round(flip_rate, 4),
            "n_adapters": len(accs),
            "concordant_pairs": concordant,
            "discordant_pairs": discordant,
        }
        print(f"  {subj}: tau={tau:.3f} (p={p_value:.3f}), flip_rate={flip_rate:.3f}")

    # Kill criteria
    mean_tau = np.mean(all_taus) if all_taus else 0.0
    mean_flip_rate = np.mean(all_flip_rates) if all_flip_rates else 1.0

    k1_fail = mean_tau < 0.5
    k2_fail = mean_flip_rate > 0.3

    results = {
        "config": {
            "subjects": list(subject_datasets.keys()),
            "n_adapters": len(adapter_names),
            "max_per_subject": args.max_per_subject,
            "seed": SEED,
        },
        "base_scores": base_scores,
        "adapter_scores": adapter_scores,
        "within_domain_rankings": within_domain_results,
        "kill_criteria": {
            "K1_mean_within_domain_tau": round(mean_tau, 4),
            "K1_threshold": 0.5,
            "K1_fail": bool(k1_fail),
            "K2_mean_flip_rate": round(mean_flip_rate, 4),
            "K2_threshold": 0.3,
            "K2_fail": bool(k2_fail),
        },
    }

    print(f"\n=== Kill Criteria ===")
    print(f"K1: mean within-domain tau = {mean_tau:.3f} (threshold 0.5) -> {'KILL' if k1_fail else 'PASS'}")
    print(f"K2: mean flip rate = {mean_flip_rate:.3f} (threshold 0.3) -> {'KILL' if k2_fail else 'PASS'}")

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
