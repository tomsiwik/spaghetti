#!/usr/bin/env python3
"""
P0: Composition-under-benchmark — routed multi-adapter vs solo on GSM8K/HumanEval/MedMCQA.

Kill criteria:
  K1408: Routed composition accuracy within 5pp of solo for each of 3 domains
  K1409: TF-IDF router correct-domain rate >= 90% on benchmark text
  K1410: Base-only accuracy matches e2e benchmark baselines (±3pp)

Grounded by:
  Finding #508: Solo adapters +19-56pp on benchmarks (baselines: GSM8K=73%, HumanEval=63%, MedMCQA=50%)
  Finding #505: Composition preserves behavioral quality (PPL max 2.1% degradation at N=5)
  Finding #502: TF-IDF routing 96-100% at N=5
"""

import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# Memory safety
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
E2E_DIR = EXPERIMENT_DIR.parent / "exp_p0_e2e_benchmark"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 5 if IS_SMOKE else 100
SEED = 42
DOMAINS = ["math", "code", "medical"]

# Solo baselines from Finding #508
SOLO_BASELINES = {
    "base_gsm8k": 17.0,
    "base_humaneval": 18.0,
    "base_medmcqa": 31.0,
    "solo_gsm8k": 73.0,
    "solo_humaneval": 63.0,
    "solo_medmcqa": 50.0,
}


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB", flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ─────────────────────────────────────────────
# Pre-merge: fuse all 3 adapters into base
# ─────────────────────────────────────────────

def create_combined_adapter():
    """Create a combined adapter by concatenating all 3 LoRA adapters along rank dimension.

    For each layer, concatenate A matrices along rank dim and B matrices along rank dim:
      A_combined = concat([A_1, A_2, A_3], axis=1)  → (d_in, 3*r)
      B_combined = concat([B_1, B_2, B_3], axis=0)  → (3*r, d_out)

    This gives: x @ A_combined @ B_combined = Σ_i (x @ A_i @ B_i)
    With scale=N*alpha and rank=N*r, the scaling factor is preserved:
      (N*alpha) / (N*r) = alpha/r = original scale factor
    """
    combined_dir = EXPERIMENT_DIR / "combined_adapter"
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Load all 3 adapter weight dicts
    adapter_weights = []
    for domain in DOMAINS:
        weights_path = E2E_DIR / "adapters" / domain / "adapters.safetensors"
        w = mx.load(str(weights_path))
        adapter_weights.append(w)
        print(f"  Loaded {domain} adapter: {len(w)} keys", flush=True)

    # Read config from first adapter for reference
    with open(E2E_DIR / "adapters" / DOMAINS[0] / "adapter_config.json") as f:
        base_config = json.load(f)

    original_rank = base_config["lora_parameters"]["rank"]  # 8
    original_scale = base_config["lora_parameters"]["scale"]  # 8.0
    n_adapters = len(DOMAINS)
    combined_rank = original_rank * n_adapters  # 24

    # Concatenate weights
    combined = {}
    keys = sorted(adapter_weights[0].keys())
    a_keys = [k for k in keys if k.endswith(".lora_a")]

    for a_key in a_keys:
        b_key = a_key.replace(".lora_a", ".lora_b")

        # Concatenate A along rank dim (axis=1): (d_in, r) → (d_in, 3r)
        a_list = [w[a_key] for w in adapter_weights]
        combined[a_key] = mx.concatenate(a_list, axis=1)

        # Concatenate B along rank dim (axis=0): (r, d_out) → (3r, d_out)
        b_list = [w[b_key] for w in adapter_weights]
        combined[b_key] = mx.concatenate(b_list, axis=0)

    mx.eval(combined)

    # Save combined weights
    mx.save_safetensors(str(combined_dir / "adapters.safetensors"), combined)

    # Write adapter config with combined rank
    # scale=N*alpha ensures (N*alpha)/(N*r) = alpha/r = 1.0 (same as original)
    combined_config = dict(base_config)
    combined_config["lora_parameters"] = dict(base_config["lora_parameters"])
    combined_config["lora_parameters"]["rank"] = combined_rank
    combined_config["lora_parameters"]["scale"] = original_scale * n_adapters
    combined_config["adapter_path"] = str(combined_dir)

    with open(combined_dir / "adapter_config.json", "w") as f:
        json.dump(combined_config, f, indent=4)

    print(f"  Combined adapter: rank={combined_rank}, scale={original_scale * n_adapters}, "
          f"{len(combined)} weights", flush=True)

    # Free individual adapters
    del adapter_weights, combined
    gc.collect()
    mx.clear_cache()

    return combined_dir


def load_model_with_merged_adapters():
    """Load base model with all 3 adapters pre-merged via concatenated LoRA."""
    from mlx_lm import load

    print("Creating combined adapter...", flush=True)
    combined_dir = create_combined_adapter()

    print("Loading model with combined adapter...", flush=True)
    model, tokenizer = load(MODEL_ID, adapter_path=str(combined_dir))
    log_memory("merged-loaded")

    return model, tokenizer


# ─────────────────────────────────────────────
# Benchmark evaluation (reused from e2e)
# ─────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, label="") -> float:
    """Evaluate GSM8K accuracy. Returns accuracy 0-100."""
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(N_EVAL, len(ds))))

    correct = 0
    for i, ex in enumerate(ds):
        messages = [{"role": "user", "content": f"Solve step by step.\n\n{ex['question']}"}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = generate(model, tokenizer, prompt=formatted, max_tokens=512, verbose=False)

        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt_ans = gt_match.group(1).replace(",", "").strip()

        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match:
            pred_ans = pred_match.group(1).replace(",", "").strip()
            if pred_ans == gt_ans:
                correct += 1
        else:
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt_ans:
                correct += 1

        if (i + 1) % 25 == 0:
            print(f"  GSM8K {label}: {i+1}/{len(ds)}, running acc={correct/(i+1)*100:.1f}%", flush=True)

    acc = correct / len(ds) * 100
    print(f"GSM8K {label}: {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    return acc


def eval_humaneval(model, tokenizer, label="") -> float:
    """Evaluate HumanEval pass@1. Returns accuracy 0-100."""
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("openai_humaneval", split="test")
    ds = ds.select(range(min(N_EVAL, len(ds))))

    passed = 0
    for i, ex in enumerate(ds):
        messages = [{"role": "user", "content": (
            f"Complete the following Python function. "
            f"Respond with ONLY the function body code, no explanation.\n\n"
            f"```python\n{ex['prompt']}\n```"
        )}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = generate(model, tokenizer, prompt=formatted, max_tokens=512, verbose=False)

        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        completion = code_match.group(1) if code_match else response
        full_code = ex["prompt"] + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"

        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                timeout=10, capture_output=True, text=True,
            )
            if result.returncode == 0:
                passed += 1
        except (subprocess.TimeoutExpired, Exception):
            pass

        if (i + 1) % 25 == 0:
            print(f"  HumanEval {label}: {i+1}/{len(ds)}, running pass@1={passed/(i+1)*100:.1f}%", flush=True)

    acc = passed / len(ds) * 100
    print(f"HumanEval {label}: {passed}/{len(ds)} = {acc:.1f}%", flush=True)
    return acc


def eval_medmcqa(model, tokenizer, label="") -> float:
    """Evaluate MedMCQA accuracy. Returns accuracy 0-100."""
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    ds = ds.shuffle(seed=SEED).select(range(min(N_EVAL, len(ds))))
    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct = 0

    for i, ex in enumerate(ds):
        question = (
            f"{ex['question']}\n"
            f"(A) {ex['opa']}\n(B) {ex['opb']}\n(C) {ex['opc']}\n(D) {ex['opd']}"
        )
        messages = [{"role": "user", "content": f"Answer this medical question. Reply with only the letter.\n\n{question}"}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = generate(model, tokenizer, prompt=formatted, max_tokens=20, verbose=False)

        gt = option_map.get(ex["cop"], "A")
        pred = response.strip().upper()
        pred_letter = None
        for letter in ["A", "B", "C", "D"]:
            if pred.startswith(letter):
                pred_letter = letter
                break
        if pred_letter is None:
            m = re.search(r"\b([ABCD])\b", pred)
            if m:
                pred_letter = m.group(1)

        if pred_letter == gt:
            correct += 1

        if (i + 1) % 25 == 0:
            print(f"  MedMCQA {label}: {i+1}/{len(ds)}, running acc={correct/(i+1)*100:.1f}%", flush=True)

    acc = correct / len(ds) * 100
    print(f"MedMCQA {label}: {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    return acc


# ─────────────────────────────────────────────
# Routing evaluation on actual benchmark text
# ─────────────────────────────────────────────

def eval_routing_on_benchmarks() -> dict:
    """Train TF-IDF router and evaluate on actual benchmark questions."""
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeClassifier

    # Router training data (disjoint from benchmark eval data)
    n_route_train = 200
    domain_train = {}

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=SEED + 1)
    domain_train["math"] = [ex["question"] for ex in ds.select(range(min(n_route_train, len(ds))))]

    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    ds = ds.shuffle(seed=SEED + 1)
    domain_train["code"] = [ex["instruction"] for ex in ds.select(range(min(n_route_train, len(ds))))]

    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    ds = ds.shuffle(seed=SEED + 1)
    domain_train["medical"] = [ex["question"] for ex in ds.select(range(min(n_route_train, len(ds))))]

    # Build router
    train_texts, train_labels = [], []
    for domain, texts in domain_train.items():
        train_texts.extend(texts)
        train_labels.extend([domain] * len(texts))

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train, train_labels)

    # Test on ACTUAL BENCHMARK QUESTIONS (the same ones used for accuracy eval)
    n_bench = 50 if IS_SMOKE else 100

    ds_gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    ds_gsm8k = ds_gsm8k.shuffle(seed=SEED).select(range(min(n_bench, len(ds_gsm8k))))
    bench_math = [ex["question"] for ex in ds_gsm8k]

    ds_he = load_dataset("openai_humaneval", split="test")
    ds_he = ds_he.select(range(min(n_bench, len(ds_he))))
    bench_code = [ex["prompt"] for ex in ds_he]

    ds_med = load_dataset("openlifescienceai/medmcqa", split="validation")
    ds_med = ds_med.shuffle(seed=SEED).select(range(min(n_bench, len(ds_med))))
    bench_med = [ex["question"] for ex in ds_med]

    test_texts = bench_math + bench_code + bench_med
    test_labels = (["math"] * len(bench_math) +
                   ["code"] * len(bench_code) +
                   ["medical"] * len(bench_med))

    X_test = vectorizer.transform(test_texts)
    preds = clf.predict(X_test)

    overall_acc = sum(p == t for p, t in zip(preds, test_labels)) / len(test_labels) * 100

    per_domain = {}
    for domain in DOMAINS:
        mask = [t == domain for t in test_labels]
        d_preds = [p for p, m in zip(preds, mask) if m]
        d_true = [t for t, m in zip(test_labels, mask) if m]
        per_domain[domain] = sum(p == t for p, t in zip(d_preds, d_true)) / len(d_true) * 100

    print(f"\nRouting accuracy on benchmark text: {overall_acc:.1f}%", flush=True)
    for d, a in per_domain.items():
        print(f"  {d}: {a:.1f}%", flush=True)

    return {
        "routing_overall_pct": round(overall_acc, 1),
        "routing_per_domain": {d: round(a, 1) for d, a in per_domain.items()},
        "vectorizer": vectorizer,
        "classifier": clf,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t_start = time.time()
    log_memory("start")
    print(f"P0 Composition Benchmark: pre-merged 3-adapter vs solo", flush=True)
    print(f"SMOKE={IS_SMOKE}, N_EVAL={N_EVAL}, SEED={SEED}", flush=True)
    print(f"Solo baselines (Finding #508): {SOLO_BASELINES}", flush=True)

    # ── Phase 1: Base model replication ──────────────
    print("\n" + "=" * 60, flush=True)
    print("PHASE 1: Base model accuracy replication", flush=True)
    print("=" * 60, flush=True)

    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    log_memory("base-loaded")

    base_gsm8k = eval_gsm8k(model, tokenizer, label="base")
    base_humaneval = eval_humaneval(model, tokenizer, label="base")
    base_medmcqa = eval_medmcqa(model, tokenizer, label="base")
    cleanup(model, tokenizer)

    print(f"\nBase replication: GSM8K={base_gsm8k:.1f}% (ref=17%), "
          f"HumanEval={base_humaneval:.1f}% (ref=18%), MedMCQA={base_medmcqa:.1f}% (ref=31%)", flush=True)

    # ── Phase 2: Pre-merged composition evaluation ───
    print("\n" + "=" * 60, flush=True)
    print("PHASE 2: Pre-merged 3-adapter composition", flush=True)
    print("=" * 60, flush=True)

    model_merged, tokenizer_merged = load_model_with_merged_adapters()

    merged_gsm8k = eval_gsm8k(model_merged, tokenizer_merged, label="merged")
    merged_humaneval = eval_humaneval(model_merged, tokenizer_merged, label="merged")
    merged_medmcqa = eval_medmcqa(model_merged, tokenizer_merged, label="merged")
    cleanup(model_merged, tokenizer_merged)

    print(f"\nMerged: GSM8K={merged_gsm8k:.1f}%, HumanEval={merged_humaneval:.1f}%, "
          f"MedMCQA={merged_medmcqa:.1f}%", flush=True)

    # ── Phase 3: Routing on benchmark text ───────────
    print("\n" + "=" * 60, flush=True)
    print("PHASE 3: Routing accuracy on benchmark text", flush=True)
    print("=" * 60, flush=True)

    routing = eval_routing_on_benchmarks()

    # ── Results ──────────────────────────────────────
    total_time = time.time() - t_start

    # Deltas vs solo baselines
    gsm8k_delta = merged_gsm8k - SOLO_BASELINES["solo_gsm8k"]
    humaneval_delta = merged_humaneval - SOLO_BASELINES["solo_humaneval"]
    medmcqa_delta = merged_medmcqa - SOLO_BASELINES["solo_medmcqa"]

    # Base replication deltas
    base_gsm8k_delta = base_gsm8k - SOLO_BASELINES["base_gsm8k"]
    base_humaneval_delta = base_humaneval - SOLO_BASELINES["base_humaneval"]
    base_medmcqa_delta = base_medmcqa - SOLO_BASELINES["base_medmcqa"]

    # Kill criteria evaluation
    k1408_gsm8k = abs(gsm8k_delta) <= 5
    k1408_humaneval = abs(humaneval_delta) <= 5
    k1408_medmcqa = abs(medmcqa_delta) <= 5
    k1408_pass = k1408_gsm8k and k1408_humaneval and k1408_medmcqa

    k1409_pass = routing["routing_overall_pct"] >= 90
    k1410_pass = (abs(base_gsm8k_delta) <= 3 and
                  abs(base_humaneval_delta) <= 3 and
                  abs(base_medmcqa_delta) <= 3)

    results = {
        "is_smoke": IS_SMOKE,
        "n_eval": N_EVAL,
        "seed": SEED,

        # Base replication
        "base_gsm8k_pct": round(base_gsm8k, 1),
        "base_humaneval_pct": round(base_humaneval, 1),
        "base_medmcqa_pct": round(base_medmcqa, 1),
        "base_gsm8k_delta_vs_508": round(base_gsm8k_delta, 1),
        "base_humaneval_delta_vs_508": round(base_humaneval_delta, 1),
        "base_medmcqa_delta_vs_508": round(base_medmcqa_delta, 1),

        # Pre-merged composition
        "merged_gsm8k_pct": round(merged_gsm8k, 1),
        "merged_humaneval_pct": round(merged_humaneval, 1),
        "merged_medmcqa_pct": round(merged_medmcqa, 1),
        "merged_gsm8k_delta_vs_solo": round(gsm8k_delta, 1),
        "merged_humaneval_delta_vs_solo": round(humaneval_delta, 1),
        "merged_medmcqa_delta_vs_solo": round(medmcqa_delta, 1),

        # Solo baselines (reference from Finding #508)
        "solo_baselines": SOLO_BASELINES,

        # Routing on benchmark text
        "routing_overall_pct": routing["routing_overall_pct"],
        "routing_per_domain": routing["routing_per_domain"],

        # Kill criteria
        "K1408_composition_within_5pp": "PASS" if k1408_pass else "FAIL",
        "K1408_gsm8k": "PASS" if k1408_gsm8k else "FAIL",
        "K1408_humaneval": "PASS" if k1408_humaneval else "FAIL",
        "K1408_medmcqa": "PASS" if k1408_medmcqa else "FAIL",
        "K1409_routing_on_benchmarks": "PASS" if k1409_pass else "FAIL",
        "K1410_base_replication": "PASS" if k1410_pass else "FAIL",

        "total_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("PREDICTION VS MEASUREMENT", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Metric':<30} {'Solo(#508)':<12} {'Merged':<12} {'Delta':<10} {'Predicted':<15} {'Kill':<6}", flush=True)
    print("-" * 85, flush=True)
    print(f"{'GSM8K':<30} {SOLO_BASELINES['solo_gsm8k']:<12.1f} {merged_gsm8k:<12.1f} {gsm8k_delta:<+10.1f} {'68-73%':<15} {'PASS' if k1408_gsm8k else 'FAIL':<6}", flush=True)
    print(f"{'HumanEval':<30} {SOLO_BASELINES['solo_humaneval']:<12.1f} {merged_humaneval:<12.1f} {humaneval_delta:<+10.1f} {'58-63%':<15} {'PASS' if k1408_humaneval else 'FAIL':<6}", flush=True)
    print(f"{'MedMCQA':<30} {SOLO_BASELINES['solo_medmcqa']:<12.1f} {merged_medmcqa:<12.1f} {medmcqa_delta:<+10.1f} {'45-50%':<15} {'PASS' if k1408_medmcqa else 'FAIL':<6}", flush=True)
    print(f"{'Routing (benchmark text)':<30} {'':<12} {routing['routing_overall_pct']:<12.1f} {'':<10} {'>=95%':<15} {'PASS' if k1409_pass else 'FAIL':<6}", flush=True)
    print(f"{'Base GSM8K replication':<30} {SOLO_BASELINES['base_gsm8k']:<12.1f} {base_gsm8k:<12.1f} {base_gsm8k_delta:<+10.1f} {'±3pp':<15} {'PASS' if abs(base_gsm8k_delta) <= 3 else 'FAIL':<6}", flush=True)
    print(f"{'Base HumanEval replication':<30} {SOLO_BASELINES['base_humaneval']:<12.1f} {base_humaneval:<12.1f} {base_humaneval_delta:<+10.1f} {'±3pp':<15} {'PASS' if abs(base_humaneval_delta) <= 3 else 'FAIL':<6}", flush=True)
    print(f"{'Base MedMCQA replication':<30} {SOLO_BASELINES['base_medmcqa']:<12.1f} {base_medmcqa:<12.1f} {base_medmcqa_delta:<+10.1f} {'±3pp':<15} {'PASS' if abs(base_medmcqa_delta) <= 3 else 'FAIL':<6}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"K1408 Composition ≤5pp: {results['K1408_composition_within_5pp']}", flush=True)
    print(f"K1409 Routing ≥90%:     {results['K1409_routing_on_benchmarks']}", flush=True)
    print(f"K1410 Base ±3pp:        {results['K1410_base_replication']}", flush=True)
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
