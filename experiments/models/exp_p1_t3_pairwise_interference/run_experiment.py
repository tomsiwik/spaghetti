#!/usr/bin/env python3
"""
T3.1: Pairwise interference = 0 for all 10 domain pairs (N=5)

Loads 5 domain adapters (math/code/medical/legal/finance), computes all 10
pairwise Frobenius cosine similarities, then evaluates a block-diagonal merged
adapter on all 5 domains to test K1051 (≥90% of single) and K1052 (above base).

Kill criteria:
  K1050: max |cos(ΔW_i, ΔW_j)|_F < 1e-5 for all 10 pairs  → predicted FAIL (needs Grassmannian)
  K1051: Composed quality ≥ 90% of best single adapter on each domain → predicted PASS
  K1052: No domain degrades below base under composition → predicted PASS

Adapter paths:
  math/code/medical: exp_p1_t2_single_domain_training/adapters/{domain}/
  legal/finance:     exp_p1_t2_multi_domain_5/adapters/{domain}/
"""

import gc
import json
import os
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import mlx.core as mx
import numpy as np
import safetensors

# Memory safety — leave 8GB for OS + base model
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MERGED_ADAPTER_DIR = EXPERIMENT_DIR / "merged_adapter"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
N_LAYERS = 42
RANK = 6
N_DOMAINS = 5
MERGED_RANK = N_DOMAINS * RANK  # 30

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 3 if IS_SMOKE else 25
SEED = 42
OPTION_LETTERS = ["A", "B", "C", "D"]

# Adapter paths for all 5 domains
T21_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training"
T26_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_multi_domain_5"

ADAPTER_PATHS = {
    "math":    T21_DIR / "adapters" / "math",
    "code":    T21_DIR / "adapters" / "code",
    "medical": T21_DIR / "adapters" / "medical",
    "legal":   T26_DIR / "adapters" / "legal",
    "finance": T26_DIR / "adapters" / "finance",
}

# Single-adapter baselines (from T2.1 PAPER.md and T2.6 PAPER.md)
SINGLE_BASELINES = {
    "math":    {"base": 0.0,  "adapter": 82.0},
    "code":    {"base": 20.0, "adapter": 66.0},
    "medical": {"base": 26.0, "adapter": 48.0},
    "legal":   {"base": 4.0,  "adapter": 54.0},
    "finance": {"base": 4.0,  "adapter": 60.0},
}

DOMAINS = list(ADAPTER_PATHS.keys())


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()


# ─────────────────────────────────────────────
# Phase 1: Pairwise cosine similarity (K1050)
# ─────────────────────────────────────────────

def load_adapter_tensors(adapter_path: Path) -> dict:
    """Load adapter safetensors as mlx arrays."""
    tensors = {}
    with safetensors.safe_open(str(adapter_path / "adapters.safetensors"), framework="numpy") as f:
        for k in f.keys():
            tensors[k] = mx.array(f.get_tensor(k))
    return tensors


def compute_layer_inner_products(t_i: dict, t_j: dict) -> tuple[float, float, float]:
    """Compute ⟨ΔW_i, ΔW_j⟩_F, ‖ΔW_i‖²_F, ‖ΔW_j‖²_F via trace trick."""
    inner_sum = 0.0
    norm_i_sq = 0.0
    norm_j_sq = 0.0

    for layer in range(N_LAYERS):
        key_a = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_a"
        key_b = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_b"
        if key_a not in t_i:
            continue

        A_i = t_i[key_a]   # (d_in, r)
        B_i = t_i[key_b]   # (r, d_out)
        A_j = t_j[key_a]
        B_j = t_j[key_b]

        # ⟨ΔW_i, ΔW_j⟩_F = trace((A_i^T A_j)(B_j B_i^T))
        AtA_ij = A_i.T @ A_j          # (r, r)
        BBt_ji = B_j @ B_i.T          # (r, r)
        inner_sum += float(mx.trace(AtA_ij @ BBt_ji).item())

        # ‖ΔW_i‖²_F = trace((A_i^T A_i)(B_i B_i^T))
        AtA_ii = A_i.T @ A_i
        BBt_ii = B_i @ B_i.T
        norm_i_sq += float(mx.trace(AtA_ii @ BBt_ii).item())

        AtA_jj = A_j.T @ A_j
        BBt_jj = B_j @ B_j.T
        norm_j_sq += float(mx.trace(AtA_jj @ BBt_jj).item())

    return inner_sum, norm_i_sq, norm_j_sq


def compute_all_pairwise_cosines() -> dict:
    """Compute 10 pairwise cosines for 5 adapters. Returns pair -> cos value."""
    print("\n=== Phase 1: Pairwise cosine similarity (K1050) ===", flush=True)

    # Load all adapter tensors
    all_tensors = {}
    for domain, path in ADAPTER_PATHS.items():
        print(f"  Loading {domain} adapter...", flush=True)
        all_tensors[domain] = load_adapter_tensors(path)

    pair_cosines = {}
    max_cos = 0.0

    for d_i, d_j in combinations(DOMAINS, 2):
        t_i = all_tensors[d_i]
        t_j = all_tensors[d_j]

        inner, norm_i_sq, norm_j_sq = compute_layer_inner_products(t_i, t_j)
        cos = abs(inner) / (norm_i_sq**0.5 * norm_j_sq**0.5 + 1e-12)
        pair_cosines[f"{d_i}_{d_j}"] = round(cos, 8)

        print(f"  cos({d_i}, {d_j}) = {cos:.8f}", flush=True)
        if cos > max_cos:
            max_cos = cos

    print(f"\n  Max |cos| across 10 pairs = {max_cos:.8f}", flush=True)
    k1050_pass = max_cos < 1e-5
    print(f"  K1050 (< 1e-5): {'PASS' if k1050_pass else 'FAIL'} (expected FAIL)", flush=True)

    # Free all adapter tensors
    for tensors in all_tensors.values():
        for arr in tensors.values():
            del arr
    del all_tensors
    mx.clear_cache()

    return {"pair_cosines": pair_cosines, "max_cos": round(max_cos, 8), "k1050_pass": k1050_pass}


# ─────────────────────────────────────────────
# Phase 2: Build block-diagonal merged adapter
# ─────────────────────────────────────────────

def build_merged_adapter():
    """
    Create block-diagonal merged adapter: ΔW_merged = Σ A_i B_i (full scale, all 5 domains).

    Structure for each layer:
      merged_a[:, i*r:(i+1)*r] = A_i   →  shape (d_in, 30)
      merged_b[i*r:(i+1)*r, :]  = B_i   →  shape (30, d_out)

    Proof: merged_a @ merged_b = Σ_i A_i B_i (exact, no cross terms — see MATH.md Theorem 2)
    """
    print("\n=== Phase 2: Building block-diagonal merged adapter ===", flush=True)
    MERGED_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    from safetensors.numpy import save_file

    merged_tensors = {}

    for layer in range(N_LAYERS):
        key_a = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_a"
        key_b = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_b"

        a_blocks = []
        b_blocks = []

        for domain, path in ADAPTER_PATHS.items():
            with safetensors.safe_open(str(path / "adapters.safetensors"), framework="numpy") as f:
                if key_a in f.keys():
                    A = f.get_tensor(key_a)   # (d_in, r)
                    B = f.get_tensor(key_b)   # (r, d_out)
                    a_blocks.append(A)
                    b_blocks.append(B)
                else:
                    print(f"  WARNING: layer {layer} missing in {domain}", flush=True)

        if not a_blocks:
            continue

        # Block-diagonal: concat along rank dim
        merged_a = np.concatenate(a_blocks, axis=1).astype(np.float32)  # (d_in, N*r)
        merged_b = np.concatenate(b_blocks, axis=0).astype(np.float32)  # (N*r, d_out)

        merged_tensors[key_a] = merged_a
        merged_tensors[key_b] = merged_b

    save_file(merged_tensors, str(MERGED_ADAPTER_DIR / "adapters.safetensors"))
    print(f"  Saved merged adapter: {len(merged_tensors)} tensors", flush=True)

    # Write adapter_config.json  — rank=30 (5×6), scale=6.0 (same as single)
    config = {
        "adapter_path": str(MERGED_ADAPTER_DIR),
        "batch_size": 2,
        "fine_tune_type": "lora",
        "lora_parameters": {
            "rank": MERGED_RANK,
            "scale": 6.0,
            "dropout": 0.0,
            "keys": ["self_attn.q_proj"],
        },
        "model": MODEL_ID,
        "num_layers": -1,
    }
    import json as _json
    (MERGED_ADAPTER_DIR / "adapter_config.json").write_text(_json.dumps(config, indent=2))

    # Size
    size_mb = (MERGED_ADAPTER_DIR / "adapters.safetensors").stat().st_size / 1e6
    print(f"  Merged adapter size: {size_mb:.1f} MB", flush=True)
    return size_mb


# ─────────────────────────────────────────────
# Phase 3: Evaluate merged adapter on all domains
# ─────────────────────────────────────────────

def eval_gsm8k(adapter_path, n_eval):
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    log_memory("gsm8k-loaded")

    correct = 0
    for ex in ds:
        prompt = f"Solve the following math problem step by step.\n\n{ex['question']}\n\nAnswer:"
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = prompt

        response = generate(model, tokenizer, prompt=formatted, max_tokens=256, verbose=False)

        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt = gt_match.group(1).replace(",", "").strip()

        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match:
            pred = pred_match.group(1).replace(",", "").strip()
            if pred == gt:
                correct += 1
        else:
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt:
                correct += 1

    acc = correct / len(ds) * 100
    print(f"  GSM8K (merged): {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return acc


def eval_humaneval(adapter_path, n_eval):
    import subprocess as sp
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("openai_humaneval", split="test")
    ds = ds.select(range(min(n_eval, len(ds))))

    model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    log_memory("humaneval-loaded")

    passed = 0
    for ex in ds:
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": f"Complete the following Python function:\n\n```python\n{ex['prompt']}\n```\n\nRespond with only the function body, no markdown."}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = ex["prompt"]

        response = generate(model, tokenizer, prompt=formatted, max_tokens=512, verbose=False)

        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        completion = code_match.group(1) if code_match else response
        full_code = ex["prompt"] + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"

        try:
            result = sp.run([sys.executable, "-c", full_code], timeout=10, capture_output=True, text=True)
            if result.returncode == 0:
                passed += 1
        except Exception:
            pass

    acc = passed / len(ds) * 100
    print(f"  HumanEval (merged): {passed}/{len(ds)} = {acc:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return acc


def eval_medmcqa(adapter_path, n_eval):
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    log_memory("medmcqa-loaded")

    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct = 0
    for ex in ds:
        question = (
            f"{ex['question']}\n"
            f"(A) {ex['opa']}\n(B) {ex['opb']}\n(C) {ex['opc']}\n(D) {ex['opd']}"
        )
        prompt = f"Answer this medical multiple choice question. Respond with only the letter (A/B/C/D).\n\n{question}"
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = prompt

        response = generate(model, tokenizer, prompt=formatted, max_tokens=20, verbose=False)

        gt = option_map.get(ex["cop"], "A")
        pred = response.strip().upper()
        pred_letter = next((l for l in OPTION_LETTERS if pred.startswith(l)), None)
        if pred_letter is None:
            m = re.search(r"\b([ABCD])\b", pred)
            if m:
                pred_letter = m.group(1)

        if pred_letter == gt:
            correct += 1

    acc = correct / len(ds) * 100
    print(f"  MedMCQA (merged): {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return acc


def eval_mmlu(subject, adapter_path, n_eval, label):
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("cais/mmlu", subject, split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    log_memory(f"{label}-loaded")

    correct = 0
    for ex in ds:
        formatted_q = (
            f"{ex['question']}\n"
            + "\n".join(f"({OPTION_LETTERS[i]}) {ex['choices'][i]}" for i in range(len(ex["choices"])))
        )
        prompt = "Answer this multiple choice question. Respond with only the letter (A/B/C/D).\n\n" + formatted_q
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = prompt

        response = generate(model, tokenizer, prompt=formatted, max_tokens=20, verbose=False)

        gt = OPTION_LETTERS[ex["answer"]]
        pred = response.strip().upper()
        pred_letter = next((l for l in OPTION_LETTERS if pred.startswith(l)), None)
        if pred_letter is None:
            m = re.search(r"\b([ABCD])\b", pred)
            if m:
                pred_letter = m.group(1)

        if pred_letter == gt:
            correct += 1

    acc = correct / len(ds) * 100
    print(f"  {label} (merged): {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return acc


def evaluate_merged_adapter(merged_path: Path) -> dict:
    print("\n=== Phase 3: Evaluate merged adapter (all 5 domains) ===", flush=True)
    log_memory("pre-eval")

    domain_accs = {}

    print("\n--- Math (GSM8K) ---", flush=True)
    domain_accs["math"] = eval_gsm8k(merged_path, N_EVAL)

    print("\n--- Code (HumanEval) ---", flush=True)
    domain_accs["code"] = eval_humaneval(merged_path, N_EVAL)

    print("\n--- Medical (MedMCQA) ---", flush=True)
    domain_accs["medical"] = eval_medmcqa(merged_path, N_EVAL)

    print("\n--- Legal (MMLU professional_law) ---", flush=True)
    domain_accs["legal"] = eval_mmlu("professional_law", merged_path, N_EVAL, "legal")

    print("\n--- Finance (MMLU high_school_macroeconomics) ---", flush=True)
    domain_accs["finance"] = eval_mmlu("high_school_macroeconomics", merged_path, N_EVAL, "finance")

    return domain_accs


# ─────────────────────────────────────────────
# Phase 4: Kill criteria computation
# ─────────────────────────────────────────────

def compute_kill_criteria(cos_results: dict, domain_accs: dict) -> dict:
    print("\n=== Phase 4: Kill criteria ===", flush=True)
    criteria = {}

    # K1050: max |cos| < 1e-5
    max_cos = cos_results["max_cos"]
    k1050_pass = max_cos < 1e-5
    criteria["K1050"] = {
        "pass": k1050_pass,
        "max_cos": max_cos,
        "threshold": 1e-5,
        "detail": cos_results["pair_cosines"],
    }
    print(f"K1050: max |cos| = {max_cos:.2e} (threshold 1e-5): {'PASS' if k1050_pass else 'FAIL'}", flush=True)

    # K1051: composed >= 90% of single for each domain
    k1051_pass = True
    k1051_detail = {}
    for domain in DOMAINS:
        single = SINGLE_BASELINES[domain]["adapter"]
        composed = domain_accs.get(domain, 0.0)
        ratio = composed / single if single > 0 else 0.0
        passes = ratio >= 0.90
        k1051_detail[domain] = {
            "single_pct": single,
            "composed_pct": round(composed, 1),
            "ratio": round(ratio, 3),
            "pass": passes,
        }
        if not passes:
            k1051_pass = False
        print(f"  K1051 {domain}: composed={composed:.1f}% / single={single:.1f}% = {ratio:.3f} {'✓' if passes else '✗'}", flush=True)

    criteria["K1051"] = {"pass": k1051_pass, "detail": k1051_detail}
    print(f"K1051 overall: {'PASS' if k1051_pass else 'FAIL'}", flush=True)

    # K1052: no domain below base
    k1052_pass = True
    k1052_detail = {}
    for domain in DOMAINS:
        base = SINGLE_BASELINES[domain]["base"]
        composed = domain_accs.get(domain, 0.0)
        passes = composed >= base
        k1052_detail[domain] = {
            "base_pct": base,
            "composed_pct": round(composed, 1),
            "pass": passes,
        }
        if not passes:
            k1052_pass = False
        print(f"  K1052 {domain}: composed={composed:.1f}% vs base={base:.1f}% {'✓' if passes else '✗'}", flush=True)

    criteria["K1052"] = {"pass": k1052_pass, "detail": k1052_detail}
    print(f"K1052 overall: {'PASS' if k1052_pass else 'FAIL'}", flush=True)

    return criteria


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t_start = time.time()
    print("T3.1: Pairwise Interference Experiment", flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_EVAL={N_EVAL}", flush=True)
    log_memory("start")

    # Verify adapter paths
    for domain, path in ADAPTER_PATHS.items():
        adapter_file = path / "adapters.safetensors"
        if not adapter_file.exists():
            print(f"ERROR: Missing adapter: {adapter_file}", flush=True)
            sys.exit(1)
    print("All 5 adapters found.", flush=True)

    results = {"is_smoke": IS_SMOKE, "n_eval": N_EVAL}

    # Phase 1: Cosine similarity
    cos_results = compute_all_pairwise_cosines()
    results["cosine"] = cos_results

    # Phase 2: Build merged adapter
    merged_size_mb = build_merged_adapter()
    results["merged_adapter_size_mb"] = round(merged_size_mb, 1)

    # Phase 3: Evaluate merged adapter
    if not IS_SMOKE:
        domain_accs = evaluate_merged_adapter(MERGED_ADAPTER_DIR)
    else:
        # Smoke: use single-adapter numbers as proxy
        print("SMOKE MODE: skipping inference eval, using single-adapter baselines", flush=True)
        domain_accs = {d: SINGLE_BASELINES[d]["adapter"] for d in DOMAINS}

    results["domain_accuracies"] = {d: round(v, 1) for d, v in domain_accs.items()}

    # Phase 4: Kill criteria
    kill_criteria = compute_kill_criteria(cos_results, domain_accs)
    results["kill_criteria"] = kill_criteria

    # Summary
    elapsed = time.time() - t_start
    results["total_time_s"] = round(elapsed, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"K1050 max |cos| < 1e-5:          {'PASS' if kill_criteria['K1050']['pass'] else 'FAIL'} (max={cos_results['max_cos']:.2e})", flush=True)
    print(f"K1051 composed ≥ 90% single:     {'PASS' if kill_criteria['K1051']['pass'] else 'FAIL'}", flush=True)
    print(f"K1052 no domain below base:       {'PASS' if kill_criteria['K1052']['pass'] else 'FAIL'}", flush=True)
    print(f"Total time: {elapsed:.0f}s ({elapsed/3600:.2f}h)", flush=True)


if __name__ == "__main__":
    main()
