#!/usr/bin/env python3
"""Macro validation: answer-conditioned PPL predicts task accuracy at scale.

Validates the micro finding (r=0.811 on synthetic tasks) on real Qwen2.5-7B
with pilot50 LoRA adapters and MMLU evaluation.

For each adapter with MMLU mapping:
  1. Compute answer-only PPL (log-prob of correct answer letter after "Answer:")
  2. Compute full-sequence PPL (average over all tokens)
  3. Compute MMLU accuracy (log-prob scoring of A/B/C/D)
  4. Compare base model vs base+adapter for PPL improvement and accuracy improvement

Kill criteria:
  K1: Pearson r(answer_ppl_improvement, accuracy_improvement) < 0.5 across adapters
  K2: AUC for distinguishing improved vs degraded adapters < 0.7
  K3: Shadow ranking disagrees with accuracy ranking on >40% of pairwise comparisons

Supports SMOKE_TEST=1 env var (3 adapters, 10 questions each).
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"
RESULTS_DIR = Path("/workspace/llm/results/answer_conditioned_shadow")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR = Path("/workspace/llm/adapters")
BASE_MODEL = "Qwen/Qwen2.5-7B"

# Domain-to-MMLU mapping (same as eval_mmlu.py)
DOMAIN_TO_MMLU = {
    "physics": ["high_school_physics", "college_physics", "conceptual_physics"],
    "chemistry": ["high_school_chemistry", "college_chemistry"],
    "biology": ["high_school_biology", "college_biology"],
    "math": ["high_school_mathematics", "college_mathematics", "elementary_mathematics", "abstract_algebra"],
    "statistics": ["high_school_statistics", "econometrics"],
    "astronomy": ["astronomy"],
    "genetics": ["medical_genetics"],
    "ecology": ["high_school_biology"],
    "neuroscience": ["high_school_psychology", "professional_psychology"],
    "legal": ["professional_law", "international_law", "jurisprudence"],
    "medical": ["professional_medicine", "clinical_knowledge", "college_medicine", "anatomy"],
    "finance": ["high_school_macroeconomics", "high_school_microeconomics"],
    "accounting": ["professional_accounting"],
    "marketing": ["marketing"],
    "cybersecurity": ["computer_security", "security_studies"],
    "logic-puzzles": ["formal_logic", "logical_fallacies"],
    "ethics": ["business_ethics", "moral_disputes", "moral_scenarios"],
    "abstract-math": ["abstract_algebra", "college_mathematics"],
    "python": ["high_school_computer_science", "college_computer_science", "machine_learning"],
    "cpp": ["high_school_computer_science", "college_computer_science"],
    "java": ["high_school_computer_science", "college_computer_science"],
    "javascript": ["high_school_computer_science"],
    "rust": ["college_computer_science"],
}


def format_mmlu_prompt(example):
    """Format MMLU example as multiple-choice prompt."""
    return (
        f"{example['question']}\n"
        f"A. {example['choices'][0]}\n"
        f"B. {example['choices'][1]}\n"
        f"C. {example['choices'][2]}\n"
        f"D. {example['choices'][3]}\n"
        "Answer:"
    )


def get_choice_token_ids(tokenizer):
    """Get token IDs for answer letters A, B, C, D."""
    ids = {}
    for letter in "ABCD":
        # Without space prefix
        toks = tokenizer.encode(letter, add_special_tokens=False)
        ids[letter] = toks[0]
        # With space prefix (common in subword tokenizers)
        toks_sp = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(toks_sp) > 0:
            ids[f"sp_{letter}"] = toks_sp[-1]
    return ids


def evaluate_adapter(model, tokenizer, dataset, choice_ids, device, max_examples=None):
    """Evaluate one adapter configuration on MMLU examples.

    Returns per-example results with:
      - correct: bool
      - answer_log_prob: log P(correct_answer | prompt)
      - full_seq_ppl: exp(mean NLL over all tokens)
      - answer_ppl: exp(-log P(correct_answer | prompt))
    """
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    results = []

    examples = list(dataset)
    if max_examples and len(examples) > max_examples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(examples), size=max_examples, replace=False)
        examples = [examples[i] for i in sorted(indices)]

    for ex in examples:
        prompt = format_mmlu_prompt(ex)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # (1, seq_len, vocab)

        # --- Answer-only PPL ---
        # Last token position predicts the answer
        last_logits = logits[0, -1]  # (vocab,)
        log_probs_last = torch.log_softmax(last_logits, dim=-1)

        gold_letter = letter_map[ex["answer"]] if isinstance(ex["answer"], int) else str(ex["answer"]).strip().upper()
        gold_tid = choice_ids[gold_letter]
        gold_tid_sp = choice_ids.get(f"sp_{gold_letter}", gold_tid)
        answer_log_prob = max(log_probs_last[gold_tid].item(), log_probs_last[gold_tid_sp].item())
        answer_ppl = math.exp(-answer_log_prob)

        # --- Accuracy ---
        scores = {}
        for letter in "ABCD":
            tid = choice_ids[letter]
            tid_sp = choice_ids.get(f"sp_{letter}", tid)
            scores[letter] = max(log_probs_last[tid].item(), log_probs_last[tid_sp].item())
        pred = max(scores, key=scores.get)
        correct = pred == gold_letter

        # --- Full-sequence PPL ---
        # Shift logits for next-token prediction
        shift_logits = logits[0, :-1]  # (seq_len-1, vocab)
        shift_labels = input_ids[0, 1:]  # (seq_len-1,)
        log_probs_full = torch.log_softmax(shift_logits, dim=-1)
        token_nlls = -log_probs_full[range(len(shift_labels)), shift_labels].float()
        full_seq_ppl = math.exp(token_nlls.mean().item())

        # --- Prompt-only PPL (all tokens before "Answer:") ---
        # Find where "Answer:" starts in the token sequence
        prompt_text = prompt.rsplit("Answer:", 1)[0]
        prompt_only_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        n_prompt_tokens = len(prompt_only_ids)
        if n_prompt_tokens > 1 and n_prompt_tokens < len(shift_labels):
            prompt_nlls = token_nlls[:n_prompt_tokens - 1]
            prompt_ppl = math.exp(prompt_nlls.mean().item()) if len(prompt_nlls) > 0 else float("inf")
        else:
            prompt_ppl = full_seq_ppl

        results.append({
            "correct": correct,
            "answer_log_prob": answer_log_prob,
            "answer_ppl": answer_ppl,
            "full_seq_ppl": full_seq_ppl,
            "prompt_ppl": prompt_ppl,
            "pred": pred,
            "gold": gold_letter,
        })

    return results


def load_mmlu_for_domain(domain):
    """Load MMLU test data for a domain's categories."""
    categories = DOMAIN_TO_MMLU.get(domain, [])
    all_examples = []
    for cat in categories:
        try:
            ds = load_dataset("cais/mmlu", cat, split="test", trust_remote_code=True)
            all_examples.extend(list(ds))
        except Exception as e:
            print(f"  Warning: failed to load MMLU/{cat}: {e}")
    return all_examples


def aggregate_results(per_example_results):
    """Compute aggregate metrics from per-example results."""
    if not per_example_results:
        return {"accuracy": 0.0, "mean_answer_ppl": float("inf"),
                "mean_full_ppl": float("inf"), "mean_prompt_ppl": float("inf"),
                "mean_answer_log_prob": float("-inf"), "n": 0}

    accuracy = sum(r["correct"] for r in per_example_results) / len(per_example_results)
    mean_answer_ppl = np.mean([r["answer_ppl"] for r in per_example_results])
    mean_full_ppl = np.mean([r["full_seq_ppl"] for r in per_example_results])
    mean_prompt_ppl = np.mean([r["prompt_ppl"] for r in per_example_results])
    mean_answer_log_prob = np.mean([r["answer_log_prob"] for r in per_example_results])

    return {
        "accuracy": float(accuracy),
        "mean_answer_ppl": float(mean_answer_ppl),
        "mean_full_ppl": float(mean_full_ppl),
        "mean_prompt_ppl": float(mean_prompt_ppl),
        "mean_answer_log_prob": float(mean_answer_log_prob),
        "n": len(per_example_results),
    }


def safe_pearson(x, y):
    """Pearson correlation, returns 0.0 if degenerate."""
    if len(x) < 3:
        return 0.0
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def pairwise_agreement(ranking_a, ranking_b):
    """Fraction of pairwise comparisons where two rankings agree."""
    n = len(ranking_a)
    if n < 2:
        return 1.0
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            pos_a_i = ranking_a.index(ranking_a[i])
            pos_a_j = ranking_a.index(ranking_a[j])
            pos_b_i = ranking_b.index(ranking_a[i])
            pos_b_j = ranking_b.index(ranking_a[j])
            if (pos_a_i < pos_a_j) == (pos_b_i < pos_b_j):
                agree += 1
            total += 1
    return agree / total if total > 0 else 1.0


def compute_auc(labels, scores):
    """Compute AUC for binary classification. labels=1 means improved."""
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = 0
    fp = 0
    tp_total = sum(labels)
    fp_total = len(labels) - tp_total
    if tp_total == 0 or fp_total == 0:
        return 0.5  # undefined
    auc = 0.0
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    return auc / (tp_total * fp_total)


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  Answer-Conditioned Shadow Scoring — Macro Validation")
    print(f"  Device: {device}, Smoke: {IS_SMOKE}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 70)

    # Find adapters with MMLU mappings
    available_adapters = []
    for domain in sorted(DOMAIN_TO_MMLU.keys()):
        adapter_path = ADAPTER_DIR / domain
        if adapter_path.exists() and (adapter_path / "adapter_model.safetensors").exists():
            available_adapters.append(domain)
        else:
            print(f"  Skip {domain}: no adapter at {adapter_path}")

    if IS_SMOKE:
        available_adapters = available_adapters[:3]
        max_examples_per_domain = 10
    else:
        max_examples_per_domain = 100  # balance coverage vs compute

    print(f"\n  Evaluating {len(available_adapters)} adapters: {available_adapters}")

    # Load base model
    print(f"\n[1] Loading base model: {BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    choice_ids = get_choice_token_ids(tokenizer)
    print(f"  Model loaded. Choice token IDs: { {k: v for k, v in choice_ids.items() if not k.startswith('sp_')} }")

    # Evaluate each adapter
    all_adapter_results = {}

    for i, domain in enumerate(available_adapters):
        print(f"\n[{i+2}] Evaluating {domain} ({i+1}/{len(available_adapters)})")

        # Load MMLU data for this domain
        mmlu_data = load_mmlu_for_domain(domain)
        if not mmlu_data:
            print(f"  No MMLU data for {domain}, skipping")
            continue
        print(f"  Loaded {len(mmlu_data)} MMLU examples for {domain}")

        # Evaluate BASE model on this domain's MMLU
        print(f"  Evaluating BASE on {domain}...")
        base_results = evaluate_adapter(
            base_model, tokenizer, mmlu_data, choice_ids, device,
            max_examples=max_examples_per_domain,
        )
        base_agg = aggregate_results(base_results)
        print(f"    BASE: acc={base_agg['accuracy']:.3f}, ans_ppl={base_agg['mean_answer_ppl']:.2f}, "
              f"full_ppl={base_agg['mean_full_ppl']:.2f}")

        # Load adapter and evaluate
        adapter_path = ADAPTER_DIR / domain
        print(f"  Loading adapter: {adapter_path}")
        try:
            expert_model = PeftModel.from_pretrained(base_model, str(adapter_path))
            expert_model.eval()
        except Exception as e:
            print(f"  Failed to load adapter {domain}: {e}")
            continue

        print(f"  Evaluating {domain} expert...")
        expert_results = evaluate_adapter(
            expert_model, tokenizer, mmlu_data, choice_ids, device,
            max_examples=max_examples_per_domain,
        )
        expert_agg = aggregate_results(expert_results)
        print(f"    EXPERT: acc={expert_agg['accuracy']:.3f}, ans_ppl={expert_agg['mean_answer_ppl']:.2f}, "
              f"full_ppl={expert_agg['mean_full_ppl']:.2f}")

        # Compute improvements
        acc_imp = expert_agg["accuracy"] - base_agg["accuracy"]
        ans_ppl_imp = (base_agg["mean_answer_ppl"] - expert_agg["mean_answer_ppl"]) / max(base_agg["mean_answer_ppl"], 1e-10)
        full_ppl_imp = (base_agg["mean_full_ppl"] - expert_agg["mean_full_ppl"]) / max(base_agg["mean_full_ppl"], 1e-10)
        prompt_ppl_imp = (base_agg["mean_prompt_ppl"] - expert_agg["mean_prompt_ppl"]) / max(base_agg["mean_prompt_ppl"], 1e-10)

        print(f"    DELTA: acc={acc_imp:+.3f}, ans_ppl={ans_ppl_imp:+.4f}, full_ppl={full_ppl_imp:+.4f}")

        all_adapter_results[domain] = {
            "base": base_agg,
            "expert": expert_agg,
            "acc_improvement": float(acc_imp),
            "answer_ppl_improvement": float(ans_ppl_imp),
            "full_ppl_improvement": float(full_ppl_imp),
            "prompt_ppl_improvement": float(prompt_ppl_imp),
            "n_examples": base_agg["n"],
        }

        # Unload adapter to free memory for next one
        del expert_model
        torch.cuda.empty_cache()

    # ── Correlation Analysis ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  CORRELATION ANALYSIS")
    print(f"{'='*70}")

    domains_eval = sorted(all_adapter_results.keys())
    acc_imps = [all_adapter_results[d]["acc_improvement"] for d in domains_eval]
    ans_ppl_imps = [all_adapter_results[d]["answer_ppl_improvement"] for d in domains_eval]
    full_ppl_imps = [all_adapter_results[d]["full_ppl_improvement"] for d in domains_eval]
    prompt_ppl_imps = [all_adapter_results[d]["prompt_ppl_improvement"] for d in domains_eval]

    r_answer = safe_pearson(ans_ppl_imps, acc_imps)
    r_full = safe_pearson(full_ppl_imps, acc_imps)
    r_prompt = safe_pearson(prompt_ppl_imps, acc_imps)

    print(f"  N adapters evaluated: {len(domains_eval)}")
    print(f"  Pearson r(answer_ppl_imp, acc_imp) = {r_answer:.4f}")
    print(f"  Pearson r(full_ppl_imp, acc_imp)   = {r_full:.4f}")
    print(f"  Pearson r(prompt_ppl_imp, acc_imp) = {r_prompt:.4f}")

    # Rankings
    acc_ranking = sorted(domains_eval, key=lambda d: all_adapter_results[d]["acc_improvement"], reverse=True)
    ans_ranking = sorted(domains_eval, key=lambda d: all_adapter_results[d]["answer_ppl_improvement"], reverse=True)
    full_ranking = sorted(domains_eval, key=lambda d: all_adapter_results[d]["full_ppl_improvement"], reverse=True)

    # Pairwise agreement (shadow scoring ranking vs accuracy ranking)
    shadow_agreement = pairwise_agreement(acc_ranking, ans_ranking)
    full_agreement = pairwise_agreement(acc_ranking, full_ranking)
    print(f"\n  Answer PPL ranking agreement with accuracy: {shadow_agreement:.2%}")
    print(f"  Full PPL ranking agreement with accuracy:   {full_agreement:.2%}")

    # AUC for distinguishing improved vs degraded
    improved_labels = [1 if imp > 0 else 0 for imp in acc_imps]
    if sum(improved_labels) > 0 and sum(improved_labels) < len(improved_labels):
        auc_answer = compute_auc(improved_labels, ans_ppl_imps)
        auc_full = compute_auc(improved_labels, full_ppl_imps)
    else:
        auc_answer = 0.5
        auc_full = 0.5
        print(f"  Warning: all adapters {'improved' if sum(improved_labels) == len(improved_labels) else 'degraded'} — AUC undefined")

    print(f"\n  AUC(answer_ppl -> improved): {auc_answer:.4f}")
    print(f"  AUC(full_ppl -> improved):   {auc_full:.4f}")

    # ── Kill Criteria ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  KILL CRITERIA ASSESSMENT")
    print(f"{'='*70}")

    k1_pass = r_answer >= 0.5
    k2_pass = auc_answer >= 0.7
    k3_pass = (1 - shadow_agreement) <= 0.4  # disagree on <=40%

    print(f"  K1: Pearson r(ans_ppl, acc) >= 0.5?  r={r_answer:.4f}  {'PASS' if k1_pass else 'KILL'}")
    print(f"  K2: AUC >= 0.7?                      AUC={auc_answer:.4f}  {'PASS' if k2_pass else 'KILL'}")
    print(f"  K3: Shadow disagree <= 40%?           disagree={1-shadow_agreement:.2%}  {'PASS' if k3_pass else 'KILL'}")

    overall = k1_pass and k2_pass and k3_pass
    print(f"  Overall: {'SURVIVES' if overall else 'KILLED'}")
    print(f"  (Reference: micro r=0.811, macro r={r_answer:.4f})")

    # Per-adapter detail table
    print(f"\n{'='*70}")
    print(f"  {'Domain':<16s} {'AccImp':>8s} {'AnsPPLImp':>10s} {'FullPPLImp':>11s} {'Improved':>9s}")
    print(f"  {'-'*16} {'-'*8} {'-'*10} {'-'*11} {'-'*9}")
    for d in acc_ranking:
        r = all_adapter_results[d]
        imp = "YES" if r["acc_improvement"] > 0 else "no"
        print(f"  {d:<16s} {r['acc_improvement']:>+8.3f} {r['answer_ppl_improvement']:>+10.4f} "
              f"{r['full_ppl_improvement']:>+11.4f} {imp:>9s}")

    # ── Save results ──────────────────────────────────────────────────
    elapsed = time.time() - t0
    output = {
        "experiment": "exp_answer_conditioned_shadow_scoring",
        "model": BASE_MODEL,
        "n_adapters": len(domains_eval),
        "adapters_evaluated": domains_eval,
        "max_examples_per_domain": max_examples_per_domain,
        "is_smoke": IS_SMOKE,
        "correlations": {
            "pearson_r_answer": r_answer,
            "pearson_r_full": r_full,
            "pearson_r_prompt": r_prompt,
            "shadow_ranking_agreement": shadow_agreement,
            "full_ranking_agreement": full_agreement,
            "auc_answer": auc_answer,
            "auc_full": auc_full,
        },
        "kill_criteria": {
            "k1_pearson_r": {"value": r_answer, "threshold": 0.5, "pass": k1_pass},
            "k2_auc": {"value": auc_answer, "threshold": 0.7, "pass": k2_pass},
            "k3_shadow_disagree": {"value": 1 - shadow_agreement, "threshold": 0.4, "pass": k3_pass},
            "overall": "SURVIVES" if overall else "KILLED",
        },
        "per_adapter": all_adapter_results,
        "rankings": {
            "by_accuracy": acc_ranking,
            "by_answer_ppl": ans_ranking,
            "by_full_ppl": full_ranking,
        },
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
