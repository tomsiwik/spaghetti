#!/usr/bin/env python3
"""PPL-probe weighted composition at macro scale.

Tests whether PPL-probe weighting (proven at micro, r=0.990 oracle correlation)
improves MMLU accuracy over equal-weight pre-merge at Qwen2.5-7B scale.

Method:
1. For each MMLU subject, compute answer-only PPL on n_probe calibration examples
   for each adapter individually
2. Softmax-weight adapters by inverse PPL (better PPL → higher weight)
3. Compose via weighted addition
4. Evaluate on held-out MMLU questions

Conditions:
- base: no adapters
- equal_weight: uniform 1/N composition of all adapters
- ppl_probe_all: PPL-probe weighted composition of all adapters
- ppl_probe_topk: PPL-probe weighted, zero-out bottom adapters (k=5,10)
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
RESULTS_DIR = Path("/workspace/llm/results/ppl_probe_macro_composition")
SEED = 42


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


def get_available_adapters():
    """Get list of available adapter names."""
    return sorted(
        d.name for d in ADAPTER_DIR.iterdir()
        if d.is_dir() and (d / "adapter_config.json").exists()
    )


def format_mmlu_prompt(example):
    question = example["question"]
    choices = example["choices"]
    prompt = f"{question}\n"
    for i, choice in enumerate(choices):
        letter = "ABCD"[i]
        prompt += f"{letter}. {choice}\n"
    prompt += "Answer:"
    return prompt


def compute_answer_ppl(model, tokenizer, examples):
    """Compute answer-only PPL on a set of MMLU examples.

    For each example, compute the NLL of the correct answer token given the
    question+choices prompt. Returns mean PPL across examples.
    """
    choice_tokens = {}
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        choice_tokens[letter] = ids[0]

    total_nll = 0.0
    count = 0

    for ex in examples:
        prompt = format_mmlu_prompt(ex)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]
            log_probs = torch.log_softmax(logits, dim=-1)

        gold_letter = "ABCD"[ex["answer"]]
        gold_tid = choice_tokens[gold_letter]
        nll = -log_probs[gold_tid].item()
        total_nll += nll
        count += 1

    if count == 0:
        return float("inf")
    return math.exp(total_nll / count)


def compose_adapters_weighted(base_model, adapter_names, weights):
    """Compose N adapters with specified weights via PEFT."""
    from peft import PeftModel

    adapter_paths = [str(ADAPTER_DIR / name) for name in adapter_names]
    model = PeftModel.from_pretrained(
        base_model, adapter_paths[0], adapter_name=adapter_names[0]
    )
    for name, path in zip(adapter_names[1:], adapter_paths[1:]):
        model.load_adapter(path, adapter_name=name)
    model.add_weighted_adapter(
        adapters=list(adapter_names),
        weights=list(weights),
        adapter_name="composed",
        combination_type="linear",
    )
    model.set_adapter("composed")
    model.eval()
    return model


def evaluate_mmlu_accuracy(model, tokenizer, examples):
    """Evaluate MMLU accuracy on examples."""
    choice_tokens = {}
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        choice_tokens[letter] = ids[0]
        ids_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if ids_space:
            choice_tokens[f" {letter}"] = ids_space[-1]

    correct = 0
    total = 0

    for ex in examples:
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
        total += 1

    return correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-probe", type=int, default=10, help="Calibration examples per subject for PPL probe")
    parser.add_argument("--max-eval", type=int, default=50, help="Max eval examples per subject")
    parser.add_argument("--n-subjects", type=int, default=20, help="Number of MMLU subjects to test")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for weight computation")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    # MMLU subjects to evaluate
    all_subjects = [
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
    rng.shuffle(all_subjects)
    subjects = all_subjects[:args.n_subjects]
    print(f"Testing {len(subjects)} subjects, n_probe={args.n_probe}, max_eval={args.max_eval}")

    # Get adapters
    available_adapters = get_available_adapters()
    n_adapters = len(available_adapters)
    print(f"Available adapters: {n_adapters}")

    # Load base model
    print("\n=== Loading base model ===")
    t0 = time.time()
    base_model, tokenizer = load_base_model()
    print(f"Model loaded in {time.time() - t0:.0f}s")

    from peft import PeftModel
    from datasets import load_dataset

    all_results = {
        "config": {
            "n_probe": args.n_probe,
            "max_eval": args.max_eval,
            "n_subjects": len(subjects),
            "subjects": subjects,
            "n_adapters": n_adapters,
            "temperature": args.temperature,
            "seed": SEED,
        },
        "base": {},
        "equal_weight": {},
        "ppl_probe_all": {},
        "ppl_probe_top10": {},
        "ppl_probe_top5": {},
        "probe_profiles": {},
    }

    # Phase 1: Compute PPL-probe scores for each adapter on each subject
    print("\n=== Phase 1: PPL-probe profiling ===")
    # We need to evaluate each adapter individually on the probe set
    # To be efficient, we load each adapter once and evaluate on all subjects' probe sets

    subject_probe_data = {}
    subject_eval_data = {}
    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  Skip {subject}: {e}")
            continue

        # Split: first n_probe for probing, rest for eval
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        probe_indices = indices[:args.n_probe]
        eval_indices = indices[args.n_probe:args.n_probe + args.max_eval]

        subject_probe_data[subject] = ds.select(probe_indices)
        subject_eval_data[subject] = ds.select(eval_indices) if eval_indices else ds.select([])
        print(f"  {subject}: {len(probe_indices)} probe, {len(eval_indices)} eval")

    subjects = list(subject_probe_data.keys())  # filter to loaded subjects

    # Compute per-adapter PPL on probe sets
    adapter_ppls = {}  # adapter_name → {subject → ppl}
    t_probe_start = time.time()

    for i, adapter_name in enumerate(available_adapters):
        adapter_path = str(ADAPTER_DIR / adapter_name)
        print(f"\n  [{i+1}/{n_adapters}] Loading adapter: {adapter_name}")
        t_a = time.time()

        try:
            adapter_model = PeftModel.from_pretrained(base_model, adapter_path)
            adapter_model.eval()
        except Exception as e:
            print(f"    SKIP (load failed): {e}")
            continue

        adapter_ppls[adapter_name] = {}
        for subject in subjects:
            probe_data = subject_probe_data[subject]
            ppl = compute_answer_ppl(adapter_model, tokenizer, probe_data)
            adapter_ppls[adapter_name][subject] = ppl

        # Unload adapter
        del adapter_model
        torch.cuda.empty_cache()

        dt = time.time() - t_a
        print(f"    Done ({dt:.1f}s), sample PPLs: {list(adapter_ppls[adapter_name].values())[:3]}")

    t_probe_total = time.time() - t_probe_start
    print(f"\nProbe profiling done in {t_probe_total:.0f}s")

    # Also compute base model PPL on probe sets
    base_ppls = {}
    for subject in subjects:
        base_ppls[subject] = compute_answer_ppl(base_model, tokenizer, subject_probe_data[subject])
    all_results["probe_profiles"] = {
        "adapter_ppls": adapter_ppls,
        "base_ppls": base_ppls,
        "probe_time_s": t_probe_total,
    }

    # Phase 2: Compute weights and evaluate
    print("\n=== Phase 2: Evaluate conditions ===")

    # 2a: Base model
    print("\n--- Condition: base ---")
    base_total_correct = 0
    base_total_count = 0
    base_per_subject = {}
    for subject in subjects:
        c, t = evaluate_mmlu_accuracy(base_model, tokenizer, subject_eval_data[subject])
        base_per_subject[subject] = {"correct": c, "total": t, "accuracy": round(c / max(1, t), 4)}
        base_total_correct += c
        base_total_count += t
        print(f"  {subject}: {c}/{t} = {c/max(1,t):.1%}")

    base_acc = base_total_correct / max(1, base_total_count)
    all_results["base"] = {
        "per_subject": base_per_subject,
        "overall": {"correct": base_total_correct, "total": base_total_count, "accuracy": round(base_acc, 4)},
    }
    print(f"Base overall: {base_acc:.4f}")

    # Composition conditions
    conditions = {
        "equal_weight": None,  # uniform 1/N
        "ppl_probe_all": "all",  # PPL-probe, all adapters
        "ppl_probe_top10": 10,  # PPL-probe, top-10
        "ppl_probe_top5": 5,  # PPL-probe, top-5
    }

    for cond_name, cond_param in conditions.items():
        print(f"\n--- Condition: {cond_name} ---")
        cond_total_correct = 0
        cond_total_count = 0
        cond_per_subject = {}

        for subject in subjects:
            # Compute weights for this subject
            adapters_with_ppl = []
            for adapter_name in available_adapters:
                if adapter_name in adapter_ppls and subject in adapter_ppls[adapter_name]:
                    ppl = adapter_ppls[adapter_name][subject]
                    if ppl < float("inf"):
                        adapters_with_ppl.append((adapter_name, ppl))

            if not adapters_with_ppl:
                print(f"  {subject}: no valid adapters, skip")
                continue

            adapter_names = [a for a, _ in adapters_with_ppl]
            ppls = np.array([p for _, p in adapters_with_ppl])

            if cond_param is None:
                # Equal weight
                weights = np.ones(len(adapter_names)) / len(adapter_names)
                selected_adapters = adapter_names
            else:
                # PPL-probe: lower PPL → higher weight via softmax(-log(ppl)/T)
                log_ppls = np.log(ppls + 1e-8)
                scores = -log_ppls / args.temperature
                scores -= scores.max()  # numerical stability
                exp_scores = np.exp(scores)
                weights = exp_scores / exp_scores.sum()

                if isinstance(cond_param, int):
                    # Top-k: zero out all but top-k weights, renormalize
                    k = min(cond_param, len(weights))
                    top_k_idx = np.argsort(weights)[-k:]
                    mask = np.zeros_like(weights)
                    mask[top_k_idx] = 1.0
                    weights = weights * mask
                    weights = weights / weights.sum()
                    # Only include adapters with non-zero weight
                    selected_adapters = [adapter_names[j] for j in top_k_idx]
                    weights_selected = weights[top_k_idx]
                else:
                    selected_adapters = adapter_names
                    weights_selected = weights

            if cond_param is not None and isinstance(cond_param, int):
                final_adapters = selected_adapters
                final_weights = weights_selected
            else:
                final_adapters = selected_adapters if cond_param is None else adapter_names
                final_weights = weights

            # Compose and evaluate
            try:
                composed_model = compose_adapters_weighted(base_model, final_adapters, final_weights)
                c, t = evaluate_mmlu_accuracy(composed_model, tokenizer, subject_eval_data[subject])

                # Unload composed adapter
                del composed_model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  {subject}: compose failed: {e}")
                continue

            acc = c / max(1, t)
            cond_per_subject[subject] = {
                "correct": c, "total": t, "accuracy": round(acc, 4),
                "n_adapters": len(final_adapters),
                "top_weights": sorted(
                    [(a, round(float(w), 4)) for a, w in zip(final_adapters, final_weights)],
                    key=lambda x: -x[1]
                )[:5],
            }
            cond_total_correct += c
            cond_total_count += t
            print(f"  {subject}: {c}/{t} = {acc:.1%} (n_adapt={len(final_adapters)})")

        cond_acc = cond_total_correct / max(1, cond_total_count)
        all_results[cond_name] = {
            "per_subject": cond_per_subject,
            "overall": {"correct": cond_total_correct, "total": cond_total_count, "accuracy": round(cond_acc, 4)},
        }
        print(f"{cond_name} overall: {cond_acc:.4f}")

    # Phase 3: Summary
    print("\n=== Summary ===")
    for cond_name in ["base", "equal_weight", "ppl_probe_all", "ppl_probe_top10", "ppl_probe_top5"]:
        acc = all_results[cond_name].get("overall", {}).get("accuracy", "N/A")
        print(f"  {cond_name}: {acc}")

    # Kill criteria assessment
    base_acc = all_results["base"]["overall"]["accuracy"]
    equal_acc = all_results["equal_weight"]["overall"]["accuracy"]
    probe_all_acc = all_results["ppl_probe_all"]["overall"]["accuracy"]
    best_probe_acc = max(
        all_results[k]["overall"]["accuracy"]
        for k in ["ppl_probe_all", "ppl_probe_top10", "ppl_probe_top5"]
        if all_results[k].get("overall")
    )

    improvement_pp = (best_probe_acc - equal_acc) * 100
    probe_time = all_results["probe_profiles"]["probe_time_s"]

    print(f"\n=== Kill Criteria ===")
    print(f"K1: improvement = {improvement_pp:+.1f}pp (threshold >2pp) → {'PASS' if improvement_pp > 2 else 'KILL'}")
    print(f"K2: probe_time = {probe_time:.0f}s (informational)")
    print(f"K3: probe-oracle correlation (need per-subject analysis for this)")

    # Compute probe-oracle correlation
    # Oracle = which adapter gives best PPL per subject
    # Probe = which adapter gets highest probe weight
    if adapter_ppls:
        probe_oracle_corr = []
        for subject in subjects:
            oracle_best = None
            oracle_best_ppl = float("inf")
            for adapter_name, ppls_dict in adapter_ppls.items():
                if subject in ppls_dict and ppls_dict[subject] < oracle_best_ppl:
                    oracle_best_ppl = ppls_dict[subject]
                    oracle_best = adapter_name
            if oracle_best:
                probe_oracle_corr.append(oracle_best)

        print(f"Probe top-1 adapters per subject: {probe_oracle_corr[:5]}...")

    all_results["kill_criteria"] = {
        "K1_improvement_pp": round(improvement_pp, 2),
        "K1_pass": improvement_pp > 2,
        "K2_probe_time_s": round(probe_time, 1),
        "base_acc": round(base_acc, 4),
        "equal_acc": round(equal_acc, 4),
        "probe_all_acc": round(probe_all_acc, 4),
        "best_probe_acc": round(best_probe_acc, 4),
    }

    # Save results
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
