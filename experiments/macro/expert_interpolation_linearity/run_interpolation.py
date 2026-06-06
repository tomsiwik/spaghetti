#!/usr/bin/env python3
"""Expert interpolation linearity: test if W + alpha*dA + (1-alpha)*dB produces smooth quality gradient.

Validates SOLE's core assumption that LoRA composition is functionally linear
(not just mathematically linear in weight space).

Uses existing pilot50 adapters. No training needed.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ADAPTER_DIR = Path("/workspace/llm/adapters")
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
RESULTS_DIR = Path("/workspace/llm/results/expert_interpolation_linearity")
SEED = 42
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
N_PAIRS = 5


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


def get_adapter_metadata():
    """Get adapter names and their domain clusters."""
    adapters = sorted(
        [d.name for d in ADAPTER_DIR.iterdir()
         if d.is_dir() and (d / "adapter_config.json").exists()]
    )

    # Try to load domain info from benchmark
    benchmark_path = Path("/workspace/llm/results/pilot50_benchmark.json")
    domain_info = {}
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            data = json.load(f)
        for name, info in data.get("per_adapter", {}).items():
            domain_info[name] = info.get("domain", "unknown")

    return adapters, domain_info


def compute_adapter_cosine(name_a, name_b):
    """Compute cosine similarity between two adapters."""
    from safetensors.torch import load_file

    def load_vec(name):
        path = ADAPTER_DIR / name / "adapter_model.safetensors"
        if not path.exists():
            path_bin = ADAPTER_DIR / name / "adapter_model.bin"
            weights = torch.load(str(path_bin), map_location="cpu", weights_only=True)
        else:
            weights = load_file(str(path))
        parts = [weights[k].float().numpy().flatten() for k in sorted(weights.keys())]
        return np.concatenate(parts)

    va, vb = load_vec(name_a), load_vec(name_b)
    cos = abs(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
    return float(cos)


def select_pairs(adapters, domain_info, n_pairs):
    """Select pairs: mix of similar (within-cluster) and dissimilar (cross-cluster)."""
    rng = np.random.RandomState(SEED)

    # Compute pairwise cosines for a subset to find similar/dissimilar
    n = min(len(adapters), 20)  # Check first 20
    subset = adapters[:n]
    cosines = {}
    for i in range(n):
        for j in range(i + 1, n):
            cos = compute_adapter_cosine(subset[i], subset[j])
            cosines[(subset[i], subset[j])] = cos

    sorted_pairs = sorted(cosines.items(), key=lambda x: x[1], reverse=True)

    pairs = []
    # Pick top similar pairs
    n_similar = n_pairs // 2
    for (a, b), cos in sorted_pairs[:n_similar]:
        pairs.append({"a": a, "b": b, "cosine": round(cos, 6), "type": "similar"})

    # Pick most dissimilar pairs
    n_dissimilar = n_pairs - n_similar
    for (a, b), cos in sorted_pairs[-n_dissimilar:]:
        pairs.append({"a": a, "b": b, "cosine": round(cos, 6), "type": "dissimilar"})

    return pairs


def eval_mmlu_accuracy(model, tokenizer, subjects, max_per_subject=50):
    """Quick MMLU eval on selected subjects."""
    from datasets import load_dataset

    choice_tokens = {}
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        choice_tokens[letter] = ids[0]
        ids_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if ids_space:
            choice_tokens[f" {letter}"] = ids_space[-1]

    total_correct, total_count = 0, 0
    per_subject = {}

    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
        except Exception:
            continue
        if len(ds) > max_per_subject:
            ds = ds.select(range(max_per_subject))

        correct = 0
        for ex in ds:
            prompt = f"{ex['question']}\n"
            for i, c in enumerate(ex["choices"]):
                prompt += f"{'ABCD'[i]}. {c}\n"
            prompt += "Answer:"

            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                out = model(**inputs)
                logits = out.logits[0, -1]
                lp = torch.log_softmax(logits, dim=-1)

            scores = {}
            for letter in "ABCD":
                tid = choice_tokens[letter]
                tid_s = choice_tokens.get(f" {letter}", tid)
                scores[letter] = max(lp[tid].item(), lp[tid_s].item())

            pred = max(scores, key=scores.get)
            gold = "ABCD"[ex["answer"]]
            correct += int(pred == gold)

        per_subject[subj] = {"correct": correct, "total": len(ds),
                             "accuracy": round(correct / max(1, len(ds)), 4)}
        total_correct += correct
        total_count += len(ds)

    return {
        "overall_accuracy": round(total_correct / max(1, total_count), 4),
        "per_subject": per_subject,
        "total_correct": total_correct,
        "total_count": total_count,
    }


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Setup
    adapters, domain_info = get_adapter_metadata()
    print(f"Found {len(adapters)} adapters")

    # Select pairs
    pairs = select_pairs(adapters, domain_info, N_PAIRS)
    print(f"\nSelected {len(pairs)} pairs:")
    for p in pairs:
        print(f"  {p['a']} <-> {p['b']} (cos={p['cosine']:.6f}, {p['type']})")

    # Load model
    print("\nLoading base model...")
    base_model, tokenizer = load_base_model()

    # Select eval subjects (small set for speed)
    eval_subjects = ["abstract_algebra", "college_physics", "machine_learning",
                     "medical_genetics", "philosophy", "computer_security",
                     "high_school_mathematics", "college_biology"]

    all_results = {"pairs": [], "config": {
        "alphas": ALPHAS,
        "n_pairs": N_PAIRS,
        "eval_subjects": eval_subjects,
        "seed": SEED,
    }}

    for pair_idx, pair in enumerate(pairs):
        name_a, name_b = pair["a"], pair["b"]
        print(f"\n=== Pair {pair_idx + 1}/{len(pairs)}: {name_a} <-> {name_b} ===")

        pair_results = {
            "a": name_a,
            "b": name_b,
            "cosine": pair["cosine"],
            "type": pair["type"],
            "alpha_sweep": [],
        }

        from peft import PeftModel

        for alpha in ALPHAS:
            t1 = time.time()
            print(f"  alpha={alpha:.2f}...", end=" ", flush=True)

            # Compose: W + alpha * dA + (1-alpha) * dB
            model = PeftModel.from_pretrained(
                base_model, str(ADAPTER_DIR / name_a), adapter_name=name_a)
            model.load_adapter(str(ADAPTER_DIR / name_b), adapter_name=name_b)

            weights = [alpha, 1.0 - alpha]
            model.add_weighted_adapter(
                adapters=[name_a, name_b],
                weights=weights,
                adapter_name="interpolated",
                combination_type="linear",
            )
            model.set_adapter("interpolated")
            model.eval()

            # Evaluate
            result = eval_mmlu_accuracy(model, tokenizer, eval_subjects, max_per_subject=50)
            acc = result["overall_accuracy"]
            print(f"acc={acc:.4f} ({time.time() - t1:.0f}s)")

            pair_results["alpha_sweep"].append({
                "alpha": alpha,
                "weight_a": round(alpha, 2),
                "weight_b": round(1.0 - alpha, 2),
                "accuracy": acc,
                "per_subject": result["per_subject"],
            })

            del model
            torch.cuda.empty_cache()

        # Analyze monotonicity and smoothness
        accs = [r["accuracy"] for r in pair_results["alpha_sweep"]]
        acc_a = accs[-1]  # alpha=1.0 (pure A)
        acc_b = accs[0]  # alpha=0.0 (pure B)
        acc_mid = accs[3]  # alpha=0.5

        # Check if midpoint is below both endpoints
        midpoint_dip = acc_mid < min(acc_a, acc_b)
        max_acc = max(accs)
        min_acc = min(accs)
        range_pp = (max_acc - min_acc) * 100

        # Smoothness: max step-to-step change
        steps = [abs(accs[i + 1] - accs[i]) * 100 for i in range(len(accs) - 1)]
        max_step = max(steps) if steps else 0

        pair_results["analysis"] = {
            "acc_pure_a": acc_a,
            "acc_pure_b": acc_b,
            "acc_midpoint": acc_mid,
            "midpoint_dip": midpoint_dip,
            "range_pp": round(range_pp, 2),
            "max_step_pp": round(max_step, 2),
            "smooth": max_step < 5.0,  # <5pp step = smooth
        }

        all_results["pairs"].append(pair_results)
        print(f"  Midpoint dip: {midpoint_dip}, range: {range_pp:.2f}pp, "
              f"max step: {max_step:.2f}pp")

    # Kill criteria
    n_dips = sum(1 for p in all_results["pairs"] if p["analysis"]["midpoint_dip"])
    n_rough = sum(1 for p in all_results["pairs"] if not p["analysis"]["smooth"])

    k1_pass = n_dips <= len(pairs) // 2  # Majority don't dip
    k2_pass = n_dips == 0 or not all(
        p["analysis"]["midpoint_dip"] for p in all_results["pairs"]
    )  # Not ALL pairs dip

    verdict = "PASS" if k1_pass and k2_pass else "FAIL"
    all_results["kill_criteria"] = {
        "k1_majority_no_dip": k1_pass,
        "k2_not_all_dip": k2_pass,
        "n_midpoint_dips": n_dips,
        "n_rough": n_rough,
        "verdict": verdict,
    }

    print(f"\n=== Summary ===")
    print(f"Midpoint dips: {n_dips}/{len(pairs)}")
    print(f"Non-smooth: {n_rough}/{len(pairs)}")
    print(f"Verdict: {verdict}")
    print(f"Total time: {time.time() - t0:.0f}s")

    out_path = RESULTS_DIR / "interpolation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
