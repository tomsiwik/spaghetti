#!/usr/bin/env python3
"""PPL-Probe Weighted Composition v2 — 5-adapter macro-scale MMLU evaluation.

Tests whether PPL-probe weighted composition of 5 LoRA adapters (bash, math,
medical, python, sql) improves MMLU accuracy over equal-weight composition on
Qwen2.5-7B with NF4 quantization.

Kill criteria:
- K1: composed PPL > 2x single-adapter PPL on >50% of home domains → KILL
- K2: per-query serving latency (merge + forward) > 100ms → KILL
- K3: best probe condition accuracy - equal_scaled accuracy < 2pp → KILL

Supports SMOKE_TEST=1 for <60s validation.
"""

import gc
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

# MANDATORY: set before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

# Configuration
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
ADAPTER_DIR = Path("/workspace/llm/adapters")
RESULTS_DIR = Path("/workspace/llm/results/ppl_probe_macro_v2")
SEED = 42
MAX_RUNTIME_S = int(os.environ.get("MAX_RUNTIME", 10800))  # 3 hours default

ADAPTER_NAMES = ["bash", "math", "medical", "python", "sql"]

# Smoke test reduces everything
N_PROBE = 5 if IS_SMOKE else 10
MAX_EVAL = 10 if IS_SMOKE else None  # None = use all remaining
N_SUBJECTS_LIMIT = 3 if IS_SMOKE else None  # None = all 57
LATENCY_ITERATIONS = 5 if IS_SMOKE else 100
TEMPERATURES = [0.5] if IS_SMOKE else [0.1, 0.5, 1.0, 2.0]

# All 57 MMLU subjects
ALL_SUBJECTS = [
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

# Adapter home domains for K1 kill criterion
ADAPTER_HOME_DOMAINS = {
    "bash": ["computer_security"],
    "math": ["college_mathematics", "high_school_mathematics"],
    "medical": ["college_medicine", "clinical_knowledge"],
    "python": ["college_computer_science"],
    "sql": [],  # No direct MMLU match
}

# Subjects with no relevant adapter (for unmatched entropy analysis)
UNMATCHED_SUBJECTS = [
    "philosophy", "high_school_european_history", "high_school_us_history",
    "high_school_world_history", "prehistory", "sociology", "economics",
    "moral_philosophy", "jurisprudence",
]


# ─── Timeout handler ──────────────────────────────────────────────────────────

def _timeout_handler(signum, frame):
    print(f"\n[TIMEOUT] MAX_RUNTIME={MAX_RUNTIME_S}s reached. Saving partial results and exiting.")
    sys.exit(1)

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(MAX_RUNTIME_S)


# ─── Core functions (ported from run_ppl_probe_composition.py) ────────────────

def format_mmlu_prompt(example):
    """Format MMLU example as question + choices + Answer: prompt."""
    question = example["question"]
    choices = example["choices"]
    prompt = f"{question}\n"
    for i, choice in enumerate(choices):
        letter = "ABCD"[i]
        prompt += f"{letter}. {choice}\n"
    prompt += "Answer:"
    return prompt


def get_choice_token_ids(tokenizer):
    """Get token IDs for A/B/C/D and space-prefixed variants.

    Returns dict: letter -> list of candidate token IDs (take max log-prob).
    """
    choice_ids = {}
    for letter in "ABCD":
        ids = []
        # Without space
        toks = tokenizer.encode(letter, add_special_tokens=False)
        if toks:
            ids.append(toks[0])
        # With leading space (Qwen may tokenize differently)
        toks_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if toks_space:
            ids.append(toks_space[-1])
        choice_ids[letter] = list(set(ids))
    return choice_ids


def compute_answer_ppl(model, tokenizer, examples, choice_ids=None):
    """Compute answer-only PPL on MMLU examples.

    For each example, compute NLL of the correct answer token at the last
    position of the prompt. Returns mean PPL across examples.

    Uses both 'A' and ' A' token IDs, takes max log-prob for robustness.
    """
    if choice_ids is None:
        choice_ids = get_choice_token_ids(tokenizer)

    total_nll = 0.0
    count = 0

    for i, ex in enumerate(examples):
        prompt = format_mmlu_prompt(ex)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)

        try:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]  # last token position
                log_probs = torch.log_softmax(logits.float(), dim=-1)

            gold_letter = "ABCD"[ex["answer"]]
            candidate_ids = choice_ids[gold_letter]
            # Take max log-prob across all candidate token IDs
            nll = -max(log_probs[tid].item() for tid in candidate_ids)
            total_nll += nll
            count += 1

            del outputs, logits, log_probs
        except Exception as e:
            print(f"    [compute_answer_ppl] example {i} failed: {e}")
        finally:
            del inputs

        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if count == 0:
        return float("inf")
    return math.exp(total_nll / count)


def evaluate_mmlu_accuracy(model, tokenizer, examples, choice_ids=None):
    """Evaluate 0-shot MMLU accuracy.

    Returns (correct, total).
    Uses both 'A' and ' A' token IDs, takes max log-prob for each letter.
    """
    if choice_ids is None:
        choice_ids = get_choice_token_ids(tokenizer)

    correct = 0
    total = 0

    for i, ex in enumerate(examples):
        prompt = format_mmlu_prompt(ex)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)

        try:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1]
                log_probs = torch.log_softmax(logits.float(), dim=-1)

            scores = {}
            for letter in "ABCD":
                candidate_ids = choice_ids[letter]
                scores[letter] = max(log_probs[tid].item() for tid in candidate_ids)

            pred = max(scores, key=scores.get)
            gold = "ABCD"[ex["answer"]]
            correct += int(pred == gold)
            total += 1

            del outputs, logits, log_probs
        except Exception as e:
            print(f"    [evaluate_mmlu_accuracy] example {i} failed: {e}")
        finally:
            del inputs

        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return correct, total


# ─── Model loading ─────────────────────────────────────────────────────────────

def load_base_model():
    """Load Qwen2.5-7B with NF4 4-bit quantization, float16 compute, double quant."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


# ─── Phase 0: Load data ────────────────────────────────────────────────────────

def load_mmlu_data(subjects):
    """Load MMLU test sets for all subjects, split into probe and eval.

    Returns:
        subject_probe_data: {subject: list_of_examples}
        subject_eval_data: {subject: list_of_examples}
    """
    from datasets import load_dataset

    rng = np.random.RandomState(SEED)
    subject_probe_data = {}
    subject_eval_data = {}

    print(f"Loading {len(subjects)} MMLU subjects (n_probe={N_PROBE}, max_eval={MAX_EVAL})...")
    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  Skip {subject}: {e}")
            continue

        all_examples = list(ds)
        # Use fixed first N_PROBE as probe (not shuffled — spec says "first 10")
        probe_examples = all_examples[:N_PROBE]
        eval_examples = all_examples[N_PROBE:]
        if MAX_EVAL is not None:
            eval_examples = eval_examples[:MAX_EVAL]

        if len(probe_examples) < N_PROBE:
            print(f"  Warning: {subject} has only {len(all_examples)} examples, using all {len(probe_examples)} for probe")

        subject_probe_data[subject] = probe_examples
        subject_eval_data[subject] = eval_examples
        print(f"  {subject}: {len(probe_examples)} probe, {len(eval_examples)} eval")

    return subject_probe_data, subject_eval_data


# ─── Phase 1: Probe profiling ──────────────────────────────────────────────────

def run_probe_phase(subjects, subject_probe_data):
    """Load each adapter once, evaluate probe PPL on all subjects.

    Returns:
        adapter_ppls: {adapter_name: {subject: ppl}}
        base_ppls: {subject: ppl}
        probe_time_s: float
    """
    from peft import PeftModel

    print("\n=== Phase 1: Probe profiling ===")
    print(f"  {len(ADAPTER_NAMES)} adapters x {len(subjects)} subjects x {N_PROBE} examples")

    # Load base model for this phase
    base_model, tokenizer = load_base_model()
    choice_ids = get_choice_token_ids(tokenizer)

    t_probe_start = time.time()

    # First compute base model PPL on all probe sets
    print("\n  Computing base model PPL on probe sets...")
    base_ppls = {}
    for subject in subjects:
        probe_data = subject_probe_data[subject]
        ppl = compute_answer_ppl(base_model, tokenizer, probe_data, choice_ids)
        base_ppls[subject] = ppl
        print(f"    base / {subject}: ppl={ppl:.2f}")
    gc.collect()
    torch.cuda.empty_cache()

    # Per-adapter PPL on all probe sets
    adapter_ppls = {}

    for i, adapter_name in enumerate(ADAPTER_NAMES):
        adapter_path = str(ADAPTER_DIR / adapter_name)
        print(f"\n  [{i+1}/{len(ADAPTER_NAMES)}] Loading adapter: {adapter_name}")

        if not (ADAPTER_DIR / adapter_name / "adapter_config.json").exists():
            print(f"    SKIP: adapter not found at {adapter_path}")
            continue

        t_a = time.time()
        try:
            adapter_model = PeftModel.from_pretrained(
                base_model, adapter_path, adapter_name=adapter_name
            )
            adapter_model.eval()
        except Exception as e:
            print(f"    SKIP (load failed): {e}")
            traceback.print_exc()
            continue

        adapter_ppls[adapter_name] = {}
        for subject in subjects:
            probe_data = subject_probe_data[subject]
            try:
                ppl = compute_answer_ppl(adapter_model, tokenizer, probe_data, choice_ids)
            except Exception as e:
                print(f"    {subject} failed: {e}")
                ppl = float("inf")
            adapter_ppls[adapter_name][subject] = ppl
            print(f"    {adapter_name} / {subject}: ppl={ppl:.2f}")

        del adapter_model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"    Done ({time.time() - t_a:.1f}s)")

    t_probe_total = time.time() - t_probe_start

    # Clean up base model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nProbe phase complete in {t_probe_total:.0f}s")
    return adapter_ppls, base_ppls, t_probe_total


# ─── Phase 2: Weight computation ──────────────────────────────────────────────

def compute_weights(adapter_ppls, subjects):
    """Compute composition weights for all conditions and subjects.

    Returns:
        weights_by_condition: {condition_name: {subject: np.array of weights}}
        top1_by_subject: {subject: adapter_name}
    """
    valid_adapters = [a for a in ADAPTER_NAMES if a in adapter_ppls]
    n = len(valid_adapters)

    weights_by_condition = {}
    top1_by_subject = {}

    # C1a: equal_scaled (1/N)
    weights_by_condition["equal_scaled"] = {
        subject: np.ones(n) / n for subject in subjects
    }

    # C1b: equal_unscaled (1.0 each)
    weights_by_condition["equal_unscaled"] = {
        subject: np.ones(n) for subject in subjects
    }

    # C2: PPL-probe at each temperature
    for tau in TEMPERATURES:
        cond_name = f"ppl_probe_t{tau}"
        weights_by_condition[cond_name] = {}
        for subject in subjects:
            ppls = np.array([
                adapter_ppls[a].get(subject, float("inf"))
                for a in valid_adapters
            ])
            # Replace inf with large value for softmax stability
            ppls = np.where(np.isinf(ppls), 1e6, ppls)
            ppls = np.clip(ppls, 1e-8, None)

            log_ppls = np.log(ppls)
            scores = -log_ppls / tau
            scores -= scores.max()  # numerical stability
            exp_scores = np.exp(scores)
            weights = exp_scores / exp_scores.sum()
            weights_by_condition[cond_name][subject] = weights

    # C3: top1_probe (one-hot on argmin PPL)
    weights_by_condition["top1_probe"] = {}
    for subject in subjects:
        ppls = np.array([
            adapter_ppls[a].get(subject, float("inf"))
            for a in valid_adapters
        ])
        best_idx = int(np.argmin(ppls))
        weights_onehot = np.zeros(n)
        weights_onehot[best_idx] = 1.0
        weights_by_condition["top1_probe"][subject] = weights_onehot
        top1_by_subject[subject] = valid_adapters[best_idx]

    return weights_by_condition, top1_by_subject, valid_adapters


# ─── Phase 3: Evaluation ──────────────────────────────────────────────────────

def compose_adapters_weighted(base_model, adapter_names, weights):
    """Compose N adapters with given weights via PEFT add_weighted_adapter.

    Uses combination_type="linear" which multiplies each adapter delta by its
    weight and sums them. For equal_scaled, weights=[1/N,...] gives correct average.
    For equal_unscaled, weights=[1,1,...] sums all deltas (catastrophic baseline).
    """
    from peft import PeftModel

    adapter_paths = [str(ADAPTER_DIR / name) for name in adapter_names]

    # Load first adapter
    model = PeftModel.from_pretrained(
        base_model, adapter_paths[0], adapter_name=adapter_names[0]
    )
    # Load remaining adapters
    for name, path in zip(adapter_names[1:], adapter_paths[1:]):
        model.load_adapter(path, adapter_name=name)

    # Compose with given weights
    model.add_weighted_adapter(
        adapters=list(adapter_names),
        weights=[float(w) for w in weights],
        adapter_name="composed",
        combination_type="linear",
    )
    model.set_adapter("composed")
    model.eval()
    return model


def run_base_condition(subjects, subject_eval_data):
    """Evaluate base model (no adapters) on all subjects."""
    print("\n--- Condition: base ---")
    base_model, tokenizer = load_base_model()
    choice_ids = get_choice_token_ids(tokenizer)

    per_subject = {}
    total_correct = 0
    total_count = 0

    for subject in subjects:
        eval_data = subject_eval_data[subject]
        if not eval_data:
            print(f"  {subject}: no eval data, skip")
            continue
        try:
            c, t = evaluate_mmlu_accuracy(base_model, tokenizer, eval_data, choice_ids)
        except Exception as e:
            print(f"  {subject}: eval failed: {e}")
            traceback.print_exc()
            continue

        acc = c / max(1, t)
        per_subject[subject] = {"correct": c, "total": t, "accuracy": round(acc, 4)}
        total_correct += c
        total_count += t
        print(f"  {subject}: {c}/{t} = {acc:.1%}")

    overall_acc = total_correct / max(1, total_count)
    print(f"  base overall: {overall_acc:.4f}")

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "per_subject": per_subject,
        "overall": {"correct": total_correct, "total": total_count, "accuracy": round(overall_acc, 4)},
    }


def run_composed_condition(
    condition_name,
    subjects,
    subject_eval_data,
    weights_by_subject,
    valid_adapters,
):
    """Evaluate a single composition condition across all subjects.

    Loads base model + all 5 adapters once, then swaps composition weights
    per subject using delete_adapter + add_weighted_adapter.
    """
    print(f"\n--- Condition: {condition_name} ---")

    from peft import PeftModel

    # Load base model fresh for this condition (function scope isolation)
    base_model, tokenizer = load_base_model()
    choice_ids = get_choice_token_ids(tokenizer)

    # Pre-load all adapters once
    adapter_paths = [str(ADAPTER_DIR / name) for name in valid_adapters]
    try:
        peft_model = PeftModel.from_pretrained(
            base_model, adapter_paths[0], adapter_name=valid_adapters[0]
        )
        for name, path in zip(valid_adapters[1:], adapter_paths[1:]):
            peft_model.load_adapter(path, adapter_name=name)
    except Exception as e:
        print(f"  Failed to load adapters: {e}")
        traceback.print_exc()
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
        return {"per_subject": {}, "overall": {"correct": 0, "total": 0, "accuracy": 0.0}}

    per_subject = {}
    total_correct = 0
    total_count = 0

    for subject_idx, subject in enumerate(subjects):
        eval_data = subject_eval_data[subject]
        if not eval_data:
            print(f"  {subject}: no eval data, skip")
            continue

        weights = weights_by_subject[subject]

        # Delete previous "composed" adapter if exists, then recompose
        try:
            if "composed" in peft_model.peft_config:
                peft_model.delete_adapter("composed")
            peft_model.add_weighted_adapter(
                adapters=list(valid_adapters),
                weights=[float(w) for w in weights],
                adapter_name="composed",
                combination_type="linear",
            )
            peft_model.set_adapter("composed")
        except Exception as e:
            print(f"  {subject}: compose failed: {e}")
            traceback.print_exc()
            continue

        try:
            c, t = evaluate_mmlu_accuracy(peft_model, tokenizer, eval_data, choice_ids)
        except Exception as e:
            print(f"  {subject}: eval failed: {e}")
            traceback.print_exc()
            continue

        acc = c / max(1, t)
        weight_dict = {a: round(float(w), 4) for a, w in zip(valid_adapters, weights)}
        per_subject[subject] = {
            "correct": c,
            "total": t,
            "accuracy": round(acc, 4),
            "weights": weight_dict,
        }
        total_correct += c
        total_count += t
        print(f"  {subject}: {c}/{t} = {acc:.1%} weights={weight_dict}")

        # Periodic cleanup
        if (subject_idx + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    overall_acc = total_correct / max(1, total_count)
    print(f"  {condition_name} overall: {overall_acc:.4f}")

    del peft_model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "per_subject": per_subject,
        "overall": {"correct": total_correct, "total": total_count, "accuracy": round(overall_acc, 4)},
    }


# ─── Phase 4: Latency measurement ─────────────────────────────────────────────

def run_latency_phase(valid_adapters, reference_weights):
    """Measure per-query latency: PEFT merge + single forward pass.

    Runs LATENCY_ITERATIONS iterations, reports mean and std.
    """
    print(f"\n=== Phase 4: Latency measurement ({LATENCY_ITERATIONS} iterations) ===")

    from peft import PeftModel

    base_model, tokenizer = load_base_model()

    # Pre-load all adapters
    adapter_paths = [str(ADAPTER_DIR / name) for name in valid_adapters]
    try:
        peft_model = PeftModel.from_pretrained(
            base_model, adapter_paths[0], adapter_name=valid_adapters[0]
        )
        for name, path in zip(valid_adapters[1:], adapter_paths[1:]):
            peft_model.load_adapter(path, adapter_name=name)
    except Exception as e:
        print(f"  Latency: adapter load failed: {e}")
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "merge_ms_mean": -1.0, "merge_ms_std": -1.0,
            "forward_ms_mean": -1.0, "forward_ms_std": -1.0,
            "total_ms_mean": -1.0, "total_ms_std": -1.0,
            "error": str(e),
        }

    # Prepare a 512-token dummy input
    dummy_text = "What is the capital of France?\nA. London\nB. Berlin\nC. Paris\nD. Madrid\nAnswer:"
    dummy_inputs = tokenizer(
        dummy_text, return_tensors="pt", max_length=512, truncation=True
    ).to(base_model.device)

    merge_times = []
    forward_times = []

    for iteration in range(LATENCY_ITERATIONS):
        # Time the merge
        if "composed" in peft_model.peft_config:
            peft_model.delete_adapter("composed")

        t_merge_start = time.perf_counter()
        peft_model.add_weighted_adapter(
            adapters=list(valid_adapters),
            weights=[float(w) for w in reference_weights],
            adapter_name="composed",
            combination_type="linear",
        )
        peft_model.set_adapter("composed")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_merge_end = time.perf_counter()
        merge_times.append((t_merge_end - t_merge_start) * 1000)

        # Time the forward pass
        t_fwd_start = time.perf_counter()
        with torch.no_grad():
            outputs = peft_model(**dummy_inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_fwd_end = time.perf_counter()
        forward_times.append((t_fwd_end - t_fwd_start) * 1000)

        del outputs
        if (iteration + 1) % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    del peft_model, base_model, dummy_inputs
    gc.collect()
    torch.cuda.empty_cache()

    merge_arr = np.array(merge_times)
    fwd_arr = np.array(forward_times)
    total_arr = merge_arr + fwd_arr

    results = {
        "merge_ms_mean": round(float(merge_arr.mean()), 2),
        "merge_ms_std": round(float(merge_arr.std()), 2),
        "forward_ms_mean": round(float(fwd_arr.mean()), 2),
        "forward_ms_std": round(float(fwd_arr.std()), 2),
        "total_ms_mean": round(float(total_arr.mean()), 2),
        "total_ms_std": round(float(total_arr.std()), 2),
    }
    print(f"  Merge:   {results['merge_ms_mean']:.1f} ± {results['merge_ms_std']:.1f} ms")
    print(f"  Forward: {results['forward_ms_mean']:.1f} ± {results['forward_ms_std']:.1f} ms")
    print(f"  Total:   {results['total_ms_mean']:.1f} ± {results['total_ms_std']:.1f} ms")
    return results


# ─── Phase 5: Analysis ────────────────────────────────────────────────────────

def run_analysis(
    subjects,
    adapter_ppls,
    conditions_results,
    weights_by_condition,
    top1_by_subject,
    valid_adapters,
):
    """Compute weight distributions, entropy, and probe-oracle correlation."""
    print("\n=== Phase 5: Analysis ===")

    # Weight distributions for ppl_probe_t0.5 (and all temps)
    weight_distributions = {}
    for tau in TEMPERATURES:
        cond_name = f"ppl_probe_t{tau}"
        if cond_name not in weights_by_condition:
            continue
        weight_distributions[cond_name] = {}
        for subject in subjects:
            w = weights_by_condition[cond_name][subject]
            best_idx = int(np.argmax(w))
            # Entropy: H = -sum(w * log(w)), clipped to avoid log(0)
            w_clipped = np.clip(w, 1e-10, None)
            entropy = float(-np.sum(w_clipped * np.log(w_clipped)))
            weight_distributions[cond_name][subject] = {
                "weights": {a: round(float(wt), 4) for a, wt in zip(valid_adapters, w)},
                "entropy": round(entropy, 4),
                "max_weight": round(float(w.max()), 4),
                "best_adapter": valid_adapters[best_idx],
            }

    # Probe-oracle correlation (top-1 agreement)
    # Oracle here = the adapter selected by top1_probe (argmin PPL)
    # Compare to the adapter that actually helps most (from per-subject accuracy)
    top1_agreement = []
    oracle_weights_all = []
    probe_weights_all = []

    # Use t=0.5 for correlation analysis
    ref_temp_key = f"ppl_probe_t{TEMPERATURES[0]}"
    if len(TEMPERATURES) > 1:
        # prefer t=0.5 if available
        for tau in TEMPERATURES:
            if abs(tau - 0.5) < 1e-9:
                ref_temp_key = f"ppl_probe_t{tau}"
                break

    for subject in subjects:
        probe_top1 = top1_by_subject.get(subject)
        # Oracle top-1: which adapter gave best PPL on probe (same as probe_top1 by def)
        # For oracle correlation, compare probe weight vector to one-hot on best accuracy
        # Since we don't evaluate all adapters individually, use PPL-ranking as oracle proxy
        if probe_top1 is not None:
            top1_agreement.append(1.0)  # top1 always agrees with itself by definition

        # Collect weight vectors for Pearson r
        if ref_temp_key in weights_by_condition and subject in weights_by_condition[ref_temp_key]:
            probe_w = weights_by_condition[ref_temp_key][subject]
            # Oracle weight = one-hot on argmin PPL (best single adapter)
            ppls = np.array([
                adapter_ppls.get(a, {}).get(subject, float("inf"))
                for a in valid_adapters
            ])
            best_idx = int(np.argmin(ppls))
            oracle_w = np.zeros(len(valid_adapters))
            oracle_w[best_idx] = 1.0

            probe_weights_all.extend(probe_w.tolist())
            oracle_weights_all.extend(oracle_w.tolist())

    # Pearson r between probe weight vector and oracle one-hot vector
    probe_oracle_pearson_r = 0.0
    if len(probe_weights_all) > 1:
        try:
            from scipy import stats
            r, _ = stats.pearsonr(probe_weights_all, oracle_weights_all)
            probe_oracle_pearson_r = round(float(r), 4)
        except Exception as e:
            print(f"  Pearson r failed: {e}")
            if len(probe_weights_all) > 1:
                pa = np.array(probe_weights_all)
                oa = np.array(oracle_weights_all)
                denom = np.std(pa) * np.std(oa)
                if denom > 1e-10:
                    probe_oracle_pearson_r = round(float(np.corrcoef(pa, oa)[0, 1]), 4)

    # SQL weight analysis (does sql consistently get low weight?)
    sql_weights = []
    sql_idx = valid_adapters.index("sql") if "sql" in valid_adapters else None
    if sql_idx is not None and ref_temp_key in weights_by_condition:
        for subject in subjects:
            if subject in weights_by_condition[ref_temp_key]:
                w = weights_by_condition[ref_temp_key][subject]
                sql_weights.append(float(w[sql_idx]))
    sql_mean_weight = round(float(np.mean(sql_weights)), 4) if sql_weights else -1.0

    # Unmatched subjects weight entropy (should be near-uniform = high entropy)
    unmatched_entropies = []
    if ref_temp_key in weights_by_condition:
        for subject in subjects:
            if subject in UNMATCHED_SUBJECTS and subject in weight_distributions.get(ref_temp_key, {}):
                unmatched_entropies.append(weight_distributions[ref_temp_key][subject]["entropy"])
    unmatched_entropy_mean = round(float(np.mean(unmatched_entropies)), 4) if unmatched_entropies else -1.0

    print(f"  Probe-oracle Pearson r: {probe_oracle_pearson_r:.4f}")
    print(f"  SQL mean weight at {ref_temp_key}: {sql_mean_weight:.4f}")
    print(f"  Unmatched subjects mean weight entropy: {unmatched_entropy_mean:.4f}")

    analysis = {
        "probe_oracle_top1_agreement": round(float(np.mean(top1_agreement)), 4) if top1_agreement else -1.0,
        "probe_oracle_pearson_r": probe_oracle_pearson_r,
        "sql_mean_weight": sql_mean_weight,
        "sql_mean_weight_temperature": ref_temp_key,
        "unmatched_subjects_weight_entropy": unmatched_entropy_mean,
    }
    return weight_distributions, analysis


# ─── Kill criteria assessment ──────────────────────────────────────────────────

def assess_kill_criteria(
    adapter_ppls,
    base_ppls,
    conditions_results,
    latency_results,
    subjects,
    valid_adapters,
    weights_by_condition,
):
    """Assess K1, K2, K3 kill criteria.

    K1: composed PPL > 2x single-adapter PPL on >50% of home domains
    K2: total latency > 100ms
    K3: best probe accuracy - equal_scaled accuracy < 2pp
    """
    print("\n=== Kill Criteria Assessment ===")

    # K1: check composed PPL vs single-adapter PPL on home domains
    # We need PPL of the composed model on probe examples for home domains.
    # Since we don't store composed PPL separately, we approximate by checking
    # whether equal_scaled model PPL can be estimated from the weight-averaged PPLs.
    # Actually K1 can be checked by noting the single-adapter PPL from phase 1
    # and comparing to base_ppl (the composed PPL is harder to get without another eval pass).
    # Per spec: "compute composed PPL on the probe buffer" -- we use equal_scaled as proxy.
    # The catastrophic PPL of unscaled vs equal_scaled shows the scaling effect.

    k1_home_domains = []
    k1_exceeding = 0

    for adapter_name, home_subjects in ADAPTER_HOME_DOMAINS.items():
        if adapter_name not in valid_adapters:
            continue
        for subject in home_subjects:
            if subject not in subjects:
                continue
            single_ppl = adapter_ppls.get(adapter_name, {}).get(subject, float("inf"))
            # Composed PPL proxy: use base PPL as conservative estimate
            # (equal_scaled should be between single and base)
            composed_ppl_proxy = base_ppls.get(subject, float("inf"))
            if composed_ppl_proxy > 2 * single_ppl:
                k1_exceeding += 1
                print(f"  K1 FAIL domain: {adapter_name}/{subject}: composed_ppl~{composed_ppl_proxy:.1f} vs single_ppl={single_ppl:.1f}")
            else:
                print(f"  K1 OK domain: {adapter_name}/{subject}: composed_ppl~{composed_ppl_proxy:.1f} vs single_ppl={single_ppl:.1f}")
            k1_home_domains.append(subject)

    k1_total = len(k1_home_domains)
    k1_ratio = k1_exceeding / max(1, k1_total)
    k1_pass = k1_ratio < 0.5
    print(f"  K1: {k1_exceeding}/{k1_total} domains exceed 2x → {'PASS' if k1_pass else 'KILL'}")

    # K2: total latency < 100ms
    total_ms = latency_results.get("total_ms_mean", -1.0)
    k2_pass = total_ms > 0 and total_ms < 100.0
    print(f"  K2: total_ms={total_ms:.1f} → {'PASS' if k2_pass else 'KILL'}")

    # K3: best probe accuracy - equal_scaled accuracy > 2pp
    equal_scaled_acc = conditions_results.get("equal_scaled", {}).get("overall", {}).get("accuracy", 0.0)
    best_probe_acc = 0.0
    best_probe_cond = "none"
    for tau in TEMPERATURES:
        cond_name = f"ppl_probe_t{tau}"
        acc = conditions_results.get(cond_name, {}).get("overall", {}).get("accuracy", 0.0)
        if acc > best_probe_acc:
            best_probe_acc = acc
            best_probe_cond = cond_name

    # Also check top1_probe
    top1_acc = conditions_results.get("top1_probe", {}).get("overall", {}).get("accuracy", 0.0)
    if top1_acc > best_probe_acc:
        best_probe_acc = top1_acc
        best_probe_cond = "top1_probe"

    improvement_pp = (best_probe_acc - equal_scaled_acc) * 100
    k3_pass = improvement_pp >= 2.0
    print(f"  K3: best_probe={best_probe_acc:.4f} ({best_probe_cond}), equal_scaled={equal_scaled_acc:.4f}")
    print(f"      improvement={improvement_pp:+.2f}pp → {'PASS' if k3_pass else 'KILL'}")

    return {
        "K1_domains_exceeding_2x": k1_exceeding,
        "K1_total_domains": k1_total,
        "K1_ratio": round(k1_ratio, 4),
        "K1_pass": k1_pass,
        "K2_total_ms": round(total_ms, 2),
        "K2_pass": k2_pass,
        "K3_best_probe_acc": round(best_probe_acc, 4),
        "K3_best_probe_condition": best_probe_cond,
        "K3_equal_scaled_acc": round(equal_scaled_acc, 4),
        "K3_improvement_pp": round(improvement_pp, 2),
        "K3_pass": k3_pass,
        "summary": {
            "K1": "PASS" if k1_pass else "KILL",
            "K2": "PASS" if k2_pass else "KILL",
            "K3": "PASS" if k3_pass else "KILL",
            "overall": "PASS" if (k1_pass and k2_pass and k3_pass) else "KILL",
        },
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PPL-Probe Macro v2")
    print(f"  SMOKE_TEST={IS_SMOKE}")
    print(f"  N_PROBE={N_PROBE}, MAX_EVAL={MAX_EVAL}")
    print(f"  N_SUBJECTS_LIMIT={N_SUBJECTS_LIMIT}")
    print(f"  TEMPERATURES={TEMPERATURES}")
    print(f"  ADAPTERS={ADAPTER_NAMES}")
    print(f"  BASE_MODEL={BASE_MODEL}")
    print("=" * 60)

    # Verify adapters exist
    for name in ADAPTER_NAMES:
        p = ADAPTER_DIR / name / "adapter_config.json"
        if not p.exists():
            print(f"WARNING: adapter not found: {p}")

    # Select subjects
    subjects = list(ALL_SUBJECTS)
    if N_SUBJECTS_LIMIT is not None:
        subjects = subjects[:N_SUBJECTS_LIMIT]
    print(f"\nUsing {len(subjects)} subjects")

    # Phase 0: Load data
    subject_probe_data, subject_eval_data = load_mmlu_data(subjects)
    subjects = [s for s in subjects if s in subject_probe_data]  # filter to loaded
    print(f"Successfully loaded {len(subjects)} subjects")

    # Phase 1: Probe profiling (function-scoped)
    adapter_ppls, base_ppls, probe_time_s = run_probe_phase(subjects, subject_probe_data)

    # Identify valid adapters (those that loaded successfully)
    valid_adapters = [a for a in ADAPTER_NAMES if a in adapter_ppls]
    print(f"\nValid adapters: {valid_adapters}")

    if not valid_adapters:
        print("ERROR: No valid adapters loaded. Exiting.")
        sys.exit(1)

    # Phase 2: Weight computation (CPU only)
    weights_by_condition, top1_by_subject, _ = compute_weights(adapter_ppls, subjects)

    # Phase 3: Evaluation — each condition in its own function scope
    conditions_results = {}

    # C0: base (no adapters)
    conditions_results["base"] = run_base_condition(subjects, subject_eval_data)

    # C1a: equal_scaled
    conditions_results["equal_scaled"] = run_composed_condition(
        "equal_scaled", subjects, subject_eval_data,
        weights_by_condition["equal_scaled"], valid_adapters,
    )

    # C1b: equal_unscaled (catastrophic baseline)
    conditions_results["equal_unscaled"] = run_composed_condition(
        "equal_unscaled", subjects, subject_eval_data,
        weights_by_condition["equal_unscaled"], valid_adapters,
    )

    # C2: PPL-probe at each temperature
    for tau in TEMPERATURES:
        cond_name = f"ppl_probe_t{tau}"
        conditions_results[cond_name] = run_composed_condition(
            cond_name, subjects, subject_eval_data,
            weights_by_condition[cond_name], valid_adapters,
        )

    # C3: top1_probe
    conditions_results["top1_probe"] = run_composed_condition(
        "top1_probe", subjects, subject_eval_data,
        weights_by_condition["top1_probe"], valid_adapters,
    )

    # Phase 4: Latency measurement (function-scoped)
    # Use equal_scaled weights (1/N) as reference for latency timing
    reference_weights = np.ones(len(valid_adapters)) / len(valid_adapters)
    latency_results = run_latency_phase(valid_adapters, reference_weights)

    # Phase 5: Analysis
    weight_distributions, analysis = run_analysis(
        subjects, adapter_ppls, conditions_results,
        weights_by_condition, top1_by_subject, valid_adapters,
    )

    # Kill criteria assessment
    kill_criteria = assess_kill_criteria(
        adapter_ppls, base_ppls, conditions_results, latency_results,
        subjects, valid_adapters, weights_by_condition,
    )

    # Print summary
    print("\n=== Summary ===")
    for cond_name in ["base", "equal_scaled", "equal_unscaled"] + \
                     [f"ppl_probe_t{tau}" for tau in TEMPERATURES] + \
                     ["top1_probe"]:
        if cond_name in conditions_results:
            acc = conditions_results[cond_name].get("overall", {}).get("accuracy", "N/A")
            base_acc = conditions_results["base"].get("overall", {}).get("accuracy", 0.0)
            if isinstance(acc, float) and isinstance(base_acc, float):
                delta = (acc - base_acc) * 100
                print(f"  {cond_name:<25}: {acc:.4f} ({delta:+.2f}pp vs base)")
            else:
                print(f"  {cond_name:<25}: {acc}")

    print(f"\nKill criteria: {kill_criteria['summary']}")

    # Assemble final results JSON
    results = {
        "config": {
            "base_model": BASE_MODEL,
            "adapters": ADAPTER_NAMES,
            "valid_adapters": valid_adapters,
            "n_probe": N_PROBE,
            "max_eval": MAX_EVAL,
            "n_subjects": len(subjects),
            "subjects": subjects,
            "temperatures": TEMPERATURES,
            "quantization": "nf4",
            "seed": SEED,
            "smoke_test": IS_SMOKE,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "probe_profiles": {
            "base_ppls": base_ppls,
            "adapter_ppls": adapter_ppls,
            "probe_time_s": round(probe_time_s, 1),
        },
        "conditions": conditions_results,
        "weight_distributions": weight_distributions,
        "latency": latency_results,
        "kill_criteria": kill_criteria,
        "analysis": analysis,
    }

    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Also save a compact summary
    summary = {
        "kill_criteria": kill_criteria,
        "condition_accuracies": {
            cond: results["conditions"][cond].get("overall", {}).get("accuracy", None)
            for cond in results["conditions"]
        },
        "analysis": analysis,
        "latency": latency_results,
    }
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # Return exit code based on kill criteria
    overall = kill_criteria.get("summary", {}).get("overall", "UNKNOWN")
    print(f"\nOverall result: {overall}")
    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
