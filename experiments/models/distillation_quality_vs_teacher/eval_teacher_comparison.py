#!/usr/bin/env python3
"""Evaluate and compare adapters trained from 8B vs 70B teacher data.

Metrics:
1. Training loss (from train_meta.json)
2. Answer-conditioned PPL on held-out test prompts
3. Per-domain quality comparison

Generates held-out test prompts using the 70B teacher (gold standard answers),
then measures how well each adapter's distribution matches them.

Usage (on RunPod, submitted via gpu_queue.py):
    python experiments/models/distillation_quality_vs_teacher/eval_teacher_comparison.py
"""

import json
import os
import sys
import time
import gc
from pathlib import Path

import torch
import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent.parent
ADAPTER_DIR_70B = REPO_ROOT / "data" / "adapters"       # existing pilot50 adapters (70B teacher)
ADAPTER_DIR_8B = REPO_ROOT / "adapters_8b"     # new 8B teacher adapters
RESULTS_DIR = REPO_ROOT / "micro" / "models" / "distillation_quality_vs_teacher"
HF_CACHE = "/workspace/hf_cache"

SELECTED_DOMAINS = [
    "python", "sql", "bash", "physics", "accounting",
    "ethics", "creative-fiction", "causal-reasoning", "legal", "game-theory",
]

# Domain categorization for mixed strategy analysis
CODE_FACTUAL = {"python", "sql", "bash", "physics", "accounting"}
REASONING_NUANCED = {"ethics", "creative-fiction", "causal-reasoning", "legal", "game-theory"}

# Number of test examples per domain for answer-conditioned PPL
N_TEST = 50


def load_test_data(domain, data_dir, n_test=N_TEST):
    """Load last N examples from training data as held-out test set.

    Since we generated 1000 examples, use examples 951-1000 as test.
    Both 8B and 70B adapters trained on examples 1-1000, but we use
    70B-generated data as the test reference (gold answers).
    """
    # Use 70B data as reference for test prompts + gold answers
    data_70b = REPO_ROOT / "data" / "distillation" / domain / "train.jsonl"
    data_8b = data_dir / domain / "train.jsonl"

    if not data_70b.exists():
        print(f"  {domain}: no 70B test data at {data_70b}")
        return []

    # Load last n_test examples from 70B data as test
    with open(data_70b) as f:
        lines = f.readlines()

    # Use last N as test (not seen in first 950 of training)
    test_lines = lines[-n_test:]
    test_data = []
    for line in test_lines:
        record = json.loads(line)
        msgs = record["messages"]
        test_data.append({
            "instruction": msgs[0]["content"],
            "response": msgs[1]["content"],  # 70B gold answer
        })
    return test_data


def compute_answer_conditioned_ppl(model, tokenizer, test_data, device="cuda"):
    """Compute answer-conditioned PPL (proven metric, r=0.811).

    For each (instruction, response) pair:
    1. Tokenize full sequence [instruction + response]
    2. Compute NTP loss on response tokens only
    3. Return mean PPL across test set
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_valid = 0

    with torch.no_grad():
        for item in test_data:
            # Format as chat messages
            messages = [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["response"]},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)

            # Also get instruction-only to find answer start
            inst_messages = [
                {"role": "user", "content": item["instruction"]},
            ]
            inst_text = tokenizer.apply_chat_template(
                inst_messages, tokenize=False, add_generation_prompt=True)

            # Tokenize
            full_ids = tokenizer.encode(full_text, return_tensors="pt",
                                        truncation=True, max_length=1024).to(device)
            inst_ids = tokenizer.encode(inst_text, return_tensors="pt",
                                        truncation=True, max_length=1024).to(device)

            if full_ids.shape[1] <= inst_ids.shape[1]:
                continue  # Skip if response is empty after tokenization

            # Forward pass
            outputs = model(full_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

            # Compute loss on answer tokens only
            answer_start = inst_ids.shape[1] - 1  # -1 because logits are shifted
            shift_logits = logits[:, answer_start:-1, :].contiguous()
            shift_labels = full_ids[:, answer_start+1:].contiguous()

            n_answer_tokens = shift_labels.shape[1]
            if n_answer_tokens == 0:
                continue

            loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                          shift_labels.view(-1))

            total_loss += loss.item()
            total_tokens += n_answer_tokens
            n_valid += 1

    if total_tokens == 0:
        return float('inf'), 0

    avg_nll = total_loss / total_tokens
    ppl = np.exp(avg_nll)
    return ppl, n_valid


def evaluate_adapter(base_model_name, adapter_path, test_data, domain, teacher_label):
    """Load an adapter and evaluate on test data."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"\n  Evaluating {teacher_label} adapter for {domain}...")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )

    # Load adapter
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    # Compute answer-conditioned PPL
    ppl, n_valid = compute_answer_conditioned_ppl(model, tokenizer, test_data)

    # Also get training loss from metadata
    meta_file = adapter_path / "train_meta.json"
    train_loss = None
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        train_loss = meta.get("train_loss")

    print(f"    {teacher_label} {domain}: PPL={ppl:.2f}, n_valid={n_valid}, train_loss={train_loss}")

    # Free GPU
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "domain": domain,
        "teacher": teacher_label,
        "answer_ppl": ppl,
        "n_test": n_valid,
        "train_loss": train_loss,
    }


def evaluate_base(base_model_name, test_data, domain):
    """Evaluate base model (no adapter) as baseline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\n  Evaluating base model for {domain}...")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )

    ppl, n_valid = compute_answer_conditioned_ppl(model, tokenizer, test_data)
    print(f"    base {domain}: PPL={ppl:.2f}, n_valid={n_valid}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "domain": domain,
        "teacher": "base",
        "answer_ppl": ppl,
        "n_test": n_valid,
        "train_loss": None,
    }


def main():
    base_model = "Qwen/Qwen2.5-7B"
    data_8b_dir = REPO_ROOT / "data" / "distillation_8b"
    results = []

    print(f"Teacher Size Comparison: Evaluation")
    print(f"  Base model: {base_model}")
    print(f"  70B adapters: {ADAPTER_DIR_70B}")
    print(f"  8B adapters: {ADAPTER_DIR_8B}")
    print(f"  Test examples/domain: {N_TEST}")
    print()

    # Check adapter availability
    available_domains = []
    for domain in SELECTED_DOMAINS:
        has_70b = (ADAPTER_DIR_70B / domain / "adapter_config.json").exists()
        has_8b = (ADAPTER_DIR_8B / domain / "adapter_config.json").exists()
        has_data = (REPO_ROOT / "data" / "distillation" / domain / "train.jsonl").exists()
        status = f"70B:{'Y' if has_70b else 'N'} 8B:{'Y' if has_8b else 'N'} data:{'Y' if has_data else 'N'}"
        print(f"  {domain}: {status}")
        if has_70b and has_8b and has_data:
            available_domains.append(domain)

    if not available_domains:
        print("\nERROR: No domains have both 8B and 70B adapters + test data")
        sys.exit(1)

    print(f"\n{len(available_domains)} domains ready for comparison")

    # Evaluate base model on first domain to get baseline
    # (base is same for all domains, but PPL varies by test data)
    for domain in available_domains:
        test_data = load_test_data(domain, data_8b_dir)
        if not test_data:
            print(f"  {domain}: insufficient test data, skipping")
            continue

        # Evaluate base model
        base_result = evaluate_base(base_model, test_data, domain)
        results.append(base_result)

        # Evaluate 70B adapter
        result_70b = evaluate_adapter(
            base_model, ADAPTER_DIR_70B / domain, test_data, domain, "70B")
        results.append(result_70b)

        # Evaluate 8B adapter
        result_8b = evaluate_adapter(
            base_model, ADAPTER_DIR_8B / domain, test_data, domain, "8B")
        results.append(result_8b)

    # Save raw results
    results_file = RESULTS_DIR / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Analyze results
    print(f"\n{'='*80}")
    print(f"RESULTS: Teacher Size vs Expert Quality")
    print(f"{'='*80}")

    header = f"{'Domain':<20} {'Category':<12} {'Base PPL':>10} {'70B PPL':>10} {'8B PPL':>10} {'70B Imp%':>10} {'8B Imp%':>10} {'70B-8B Gap%':>12}"
    print(header)
    print("-" * len(header))

    domain_results = {}
    for domain in available_domains:
        base_r = next((r for r in results if r["domain"] == domain and r["teacher"] == "base"), None)
        r70b = next((r for r in results if r["domain"] == domain and r["teacher"] == "70B"), None)
        r8b = next((r for r in results if r["domain"] == domain and r["teacher"] == "8B"), None)

        if not all([base_r, r70b, r8b]):
            continue

        base_ppl = base_r["answer_ppl"]
        ppl_70b = r70b["answer_ppl"]
        ppl_8b = r8b["answer_ppl"]

        # Improvement = (base - expert) / base * 100 (positive = expert better)
        imp_70b = (base_ppl - ppl_70b) / base_ppl * 100 if base_ppl > 0 else 0
        imp_8b = (base_ppl - ppl_8b) / base_ppl * 100 if base_ppl > 0 else 0

        # Gap = (8B - 70B) / 70B * 100 (positive = 70B better)
        gap = (ppl_8b - ppl_70b) / ppl_70b * 100 if ppl_70b > 0 else 0

        category = "code/fact" if domain in CODE_FACTUAL else "reason/nuan"

        print(f"{domain:<20} {category:<12} {base_ppl:>10.2f} {ppl_70b:>10.2f} {ppl_8b:>10.2f} {imp_70b:>+10.1f} {imp_8b:>+10.1f} {gap:>+12.1f}")

        domain_results[domain] = {
            "category": category,
            "base_ppl": base_ppl,
            "ppl_70b": ppl_70b,
            "ppl_8b": ppl_8b,
            "imp_70b": imp_70b,
            "imp_8b": imp_8b,
            "gap": gap,
        }

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY BY CATEGORY")
    print(f"{'='*80}")

    for cat_name, cat_domains in [("Code/Factual", CODE_FACTUAL), ("Reasoning/Nuanced", REASONING_NUANCED)]:
        cat_results = [v for k, v in domain_results.items() if k in cat_domains]
        if not cat_results:
            continue
        avg_gap = np.mean([r["gap"] for r in cat_results])
        avg_imp_70b = np.mean([r["imp_70b"] for r in cat_results])
        avg_imp_8b = np.mean([r["imp_8b"] for r in cat_results])
        print(f"\n{cat_name} (N={len(cat_results)}):")
        print(f"  Avg 70B improvement over base: {avg_imp_70b:+.1f}%")
        print(f"  Avg 8B improvement over base:  {avg_imp_8b:+.1f}%")
        print(f"  Avg 70B-8B gap:                {avg_gap:+.1f}% (positive = 70B better)")

    # Overall
    if domain_results:
        all_gaps = [v["gap"] for v in domain_results.values()]
        avg_gap = np.mean(all_gaps)
        print(f"\nOVERALL:")
        print(f"  Average 70B-8B gap: {avg_gap:+.1f}%")
        print(f"  Median 70B-8B gap:  {np.median(all_gaps):+.1f}%")

        # Kill criteria assessment
        print(f"\n{'='*80}")
        print("KILL CRITERIA ASSESSMENT")
        print(f"{'='*80}")

        # K1: 70B teacher produces <5% better experts than 8B
        print(f"\nK1: 70B produces <5% better experts than 8B?")
        print(f"  Average gap: {avg_gap:+.1f}%")
        if abs(avg_gap) < 5:
            print(f"  RESULT: KILLED -- teacher size doesn't matter (gap < 5%)")
        else:
            print(f"  RESULT: SURVIVES -- 70B is meaningfully better (gap >= 5%)")

        # K2: Mixed strategy does not beat uniform 70B
        cf_gaps = [v["gap"] for k, v in domain_results.items() if k in CODE_FACTUAL]
        rn_gaps = [v["gap"] for k, v in domain_results.items() if k in REASONING_NUANCED]
        if cf_gaps and rn_gaps:
            avg_cf_gap = np.mean(cf_gaps)
            avg_rn_gap = np.mean(rn_gaps)

            # Mixed strategy: use 8B for code/factual, 70B for reasoning
            # Mixed PPL = avg(8B PPL for code/fact) + avg(70B PPL for reason)
            # vs Uniform 70B = avg(70B PPL for all)
            mixed_ppls = []
            uniform_70b_ppls = []
            for k, v in domain_results.items():
                if k in CODE_FACTUAL:
                    mixed_ppls.append(v["ppl_8b"])  # Use 8B for code/factual
                else:
                    mixed_ppls.append(v["ppl_70b"])  # Use 70B for reasoning
                uniform_70b_ppls.append(v["ppl_70b"])

            avg_mixed = np.mean(mixed_ppls)
            avg_uniform = np.mean(uniform_70b_ppls)
            mixed_vs_uniform = (avg_mixed - avg_uniform) / avg_uniform * 100

            print(f"\nK2: Mixed strategy beats uniform 70B?")
            print(f"  Code/Factual avg gap (70B-8B): {avg_cf_gap:+.1f}%")
            print(f"  Reasoning/Nuanced avg gap (70B-8B): {avg_rn_gap:+.1f}%")
            print(f"  Mixed avg PPL: {avg_mixed:.2f}")
            print(f"  Uniform 70B avg PPL: {avg_uniform:.2f}")
            print(f"  Mixed vs Uniform 70B: {mixed_vs_uniform:+.1f}%")

            if mixed_vs_uniform <= 0:
                print(f"  RESULT: SURVIVES -- mixed strategy matches/beats uniform 70B")
            else:
                print(f"  RESULT: KILLED -- mixed strategy worse than uniform 70B")

            # Cost analysis
            cost_mixed = len(CODE_FACTUAL & set(domain_results.keys())) * 0.02 + \
                         len(REASONING_NUANCED & set(domain_results.keys())) * 0.19
            cost_uniform = len(domain_results) * 0.19
            savings = (1 - cost_mixed / cost_uniform) * 100
            print(f"\n  Cost analysis (data generation only):")
            print(f"    Uniform 70B: ${cost_uniform:.2f} for {len(domain_results)} domains")
            print(f"    Mixed:       ${cost_mixed:.2f} for {len(domain_results)} domains")
            print(f"    Savings:     {savings:.0f}%")

    # Save summary
    summary = {
        "domain_results": domain_results,
        "kill_criteria": {
            "K1_gap_pct": avg_gap if domain_results else None,
            "K1_threshold": 5.0,
            "K1_killed": abs(avg_gap) < 5 if domain_results else None,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_file = RESULTS_DIR / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
