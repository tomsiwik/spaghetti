#!/usr/bin/env python3
"""
P0: TT-LoRA Composition Under Benchmarks — Does Compressed Pre-Merge Survive?

Tests whether TT-LoRA's ~20x smaller perturbation makes pre-merge composition
safe, unlike standard LoRA which destroys benchmarks (Finding #510: 0% GSM8K).

Kill criteria:
  K1447: Pre-merged 3 TT-LoRA GSM8K >= 60% (solo=68%, std pre-merge=0%)
  K1448: Pre-merged 3 TT-LoRA HumanEval >= 45% (solo=55%, std pre-merge=0%)
  K1449: Pre-merged 3 TT-LoRA MedMCQA >= 25% (solo=21%, std pre-merge=20%)
  K1450: Per-query routed TT-LoRA within 5pp of solo on all 3

Grounded by:
  Finding #510: Standard LoRA pre-merge destroys benchmarks
  Finding #516: TT-LoRA r=6 retains 84% quality solo
  arXiv:2504.21190 (TT-LoRA)
"""

import gc
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Memory safety
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
E2E_DIR = EXPERIMENT_DIR.parent / "exp_p0_ttlora_e2e_benchmark"
LORA_E2E_DIR = EXPERIMENT_DIR.parent / "exp_p0_e2e_benchmark"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_EVAL = 5 if IS_SMOKE else 100
SEED = 42
DOMAINS = ["math", "code", "medical"]

TT_RANK = 6
TT_ALPHA = 1.0
PROJ_NAMES = ["v_proj", "o_proj"]

# Solo baselines from exp_p0_ttlora_e2e_benchmark
TT_SOLO = {"gsm8k": 68.0, "humaneval": 55.0, "medmcqa": 21.0}
# Standard LoRA pre-merge from Finding #510
STD_PREMERGE = {"gsm8k": 0.0, "humaneval": 0.0, "medmcqa": 20.0}
# Base model
BASE = {"gsm8k": 17.0, "humaneval": 18.0, "medmcqa": 31.0}


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB",
          flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ──────────────────────────────────────────────────
# TT-LoRA Module (from exp_p0_ttlora_e2e_benchmark)
# ──────────────────────────────────────────────────

class TTLoRAWrapper(nn.Module):
    def __init__(self, base_layer, in_features, out_features, tt_shape,
                 tt_rank=6, alpha=1.0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.tt_shape = tt_shape

        self._validate_split(tt_shape, in_features, out_features)

        d = len(tt_shape)
        ranks = [1] + [tt_rank] * (d - 1) + [1]
        self._n_cores = d
        for k in range(d):
            shape = (ranks[k], tt_shape[k], ranks[k + 1])
            if k == d - 1:
                core = mx.zeros(shape)
            else:
                std = 1.0 / math.sqrt(tt_shape[k] * ranks[k])
                core = mx.random.normal(shape) * std
            setattr(self, f"core_{k}", core)
        self._cached_delta_w = None

    def _validate_split(self, tt_shape, in_features, out_features):
        prod = 1
        for i, s in enumerate(tt_shape):
            prod *= s
            if prod == in_features:
                rest = 1
                for j in range(i + 1, len(tt_shape)):
                    rest *= tt_shape[j]
                assert rest == out_features
                return
        raise ValueError(f"Cannot split {tt_shape} into {in_features} x {out_features}")

    @property
    def tt_cores(self):
        return [getattr(self, f"core_{k}") for k in range(self._n_cores)]

    def reconstruct_delta_w(self):
        cores = self.tt_cores
        result = cores[0].squeeze(0)
        for k in range(1, len(cores)):
            core = cores[k]
            r_k, s_k, r_next = core.shape
            result = result @ core.reshape(r_k, s_k * r_next)
            leading = result.shape[0]
            result = result.reshape(leading * s_k, r_next)
        result = result.squeeze(-1)
        return result.reshape(self.in_features, self.out_features).T

    def cache_delta_w(self):
        self._cached_delta_w = self.reconstruct_delta_w()
        mx.eval(self._cached_delta_w)

    def __call__(self, x):
        base_out = self.base_layer(x)
        dw = (self._cached_delta_w if self._cached_delta_w is not None
              else self.reconstruct_delta_w())
        return base_out + self.alpha * (x @ dw.T)

    def num_params(self):
        return sum(c.size for c in self.tt_cores)


def factorize(n, max_factor=10):
    factors = []
    while n % 8 == 0 and n > 8:
        factors.append(8)
        n //= 8
    for f in range(max_factor, 1, -1):
        while n % f == 0 and n > 1:
            factors.append(f)
            n //= f
    if n > 1:
        factors.append(n)
    factors.sort()
    return factors


def compute_tt_shape(in_features, out_features):
    return factorize(in_features) + factorize(out_features)


# ──────────────────────────────────────────────────
# Model Setup
# ──────────────────────────────────────────────────

def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    return model.layers


def detect_proj_dims(base_layer):
    if hasattr(base_layer, 'scales'):
        out_features = base_layer.weight.shape[0]
        in_features = base_layer.weight.shape[1] * 32 // base_layer.bits
        return in_features, out_features
    if hasattr(base_layer, 'weight'):
        return base_layer.weight.shape[1], base_layer.weight.shape[0]
    raise ValueError(f"Cannot detect dimensions for {type(base_layer)}")


def inject_ttlora(model, proj_names, tt_rank, alpha):
    layers = get_layers(model)
    total_params = 0
    for layer in layers:
        for name in proj_names:
            base = getattr(layer.self_attn, name)
            in_f, out_f = detect_proj_dims(base)
            tt_shape = compute_tt_shape(in_f, out_f)
            wrapper = TTLoRAWrapper(base, in_f, out_f, tt_shape, tt_rank, alpha)
            setattr(layer.self_attn, name, wrapper)
            total_params += wrapper.num_params()
    return total_params


def load_ttlora_cores(model, adapter_dir, proj_names):
    from safetensors.numpy import load_file
    filepath = Path(adapter_dir) / "tt_cores.safetensors"
    weights = load_file(str(filepath))
    layers = get_layers(model)
    for key, arr in weights.items():
        parts = key.split(".")
        layer_idx = int(parts[1])
        pname = parts[3]
        core_name = parts[4]
        proj = getattr(layers[layer_idx].self_attn, pname)
        core = mx.array(arr.astype(np.float32))
        setattr(proj, core_name, core)
        proj._cached_delta_w = None


def cache_all_delta_w(model, proj_names):
    for layer in get_layers(model):
        for pname in proj_names:
            proj = getattr(layer.self_attn, pname)
            if isinstance(proj, TTLoRAWrapper):
                proj.cache_delta_w()


# ──────────────────────────────────────────────────
# Pre-merge: sum reconstructed ΔW from all adapters
# ──────────────────────────────────────────────────

def premerge_ttlora_adapters(model, adapter_dirs, proj_names):
    """Load all adapters, reconstruct ΔW for each, sum them, cache the sum.

    This is the TT-LoRA equivalent of the concatenated LoRA pre-merge.
    Instead of concatenating cores (which changes the TT structure),
    we reconstruct each ΔW and sum in weight space.
    """
    from safetensors.numpy import load_file

    layers = get_layers(model)
    n_adapters = len(adapter_dirs)
    norms_per_adapter = []

    # For each adapter, load cores -> reconstruct ΔW -> accumulate
    for adapter_idx, adapter_dir in enumerate(adapter_dirs):
        filepath = Path(adapter_dir) / "tt_cores.safetensors"
        weights = load_file(str(filepath))

        adapter_norm_sq = 0.0

        for layer_idx, layer in enumerate(layers):
            for pname in proj_names:
                proj = getattr(layer.self_attn, pname)
                if not isinstance(proj, TTLoRAWrapper):
                    continue

                # Load this adapter's cores into temporary storage
                for k in range(proj._n_cores):
                    key = f"layers.{layer_idx}.self_attn.{pname}.core_{k}"
                    if key in weights:
                        core = mx.array(weights[key].astype(np.float32))
                        setattr(proj, f"core_{k}", core)

                # Reconstruct this adapter's ΔW
                dw = proj.reconstruct_delta_w()
                mx.eval(dw)
                adapter_norm_sq += (dw * dw).sum().item()

                # Accumulate into cached sum
                if proj._cached_delta_w is None:
                    proj._cached_delta_w = dw
                else:
                    proj._cached_delta_w = proj._cached_delta_w + dw
                mx.eval(proj._cached_delta_w)

        norm = adapter_norm_sq ** 0.5
        norms_per_adapter.append(norm)
        print(f"  Adapter {adapter_idx} ({Path(adapter_dir).name}): "
              f"||DW||_F = {norm:.4f}", flush=True)

    # Final eval of all cached sums
    for layer in layers:
        for pname in proj_names:
            proj = getattr(layer.self_attn, pname)
            if isinstance(proj, TTLoRAWrapper) and proj._cached_delta_w is not None:
                mx.eval(proj._cached_delta_w)

    return norms_per_adapter


def compute_standard_lora_norms(adapter_dirs):
    """Compute Frobenius norms of standard LoRA adapters for comparison."""
    norms = []
    for adapter_dir in adapter_dirs:
        weights_path = Path(adapter_dir) / "adapters.safetensors"
        if not weights_path.exists():
            norms.append(None)
            continue
        w = mx.load(str(weights_path))
        a_keys = [k for k in sorted(w.keys()) if k.endswith(".lora_a")]
        total_norm_sq = 0.0
        for a_key in a_keys:
            b_key = a_key.replace(".lora_a", ".lora_b")
            dw = w[a_key] @ w[b_key]  # (in, rank) @ (rank, out) — norm is shape-invariant
            mx.eval(dw)
            total_norm_sq += (dw * dw).sum().item()
            del dw
        norms.append(total_norm_sq ** 0.5)
        del w
        gc.collect()
        mx.clear_cache()
    return norms


# ──────────────────────────────────────────────────
# Dataset Loaders (via huggingface_hub, not datasets)
# ──────────────────────────────────────────────────

def load_gsm8k_test(n_eval, seed=SEED):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download("openai/gsm8k",
        "main/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.sample(n=min(n_eval, len(df)), random_state=seed)
    return df.to_dict("records")


def load_humaneval_test(n_eval):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download("openai_humaneval",
        "openai_humaneval/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    return df.head(min(n_eval, len(df))).to_dict("records")


def load_medmcqa_val(n_eval, seed=SEED):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download("openlifescienceai/medmcqa",
        "data/validation-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.sample(n=min(n_eval, len(df)), random_state=seed)
    return df.to_dict("records")


# ──────────────────────────────────────────────────
# Benchmark Evaluation
# ──────────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, n_eval, label=""):
    from mlx_lm import generate
    ds = load_gsm8k_test(n_eval)
    correct = 0
    for i, ex in enumerate(ds):
        messages = [{"role": "user", "content": f"Solve step by step.\n\n{ex['question']}"}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted,
                          max_tokens=512, verbose=False)
        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt = gt_match.group(1).replace(",", "").strip()
        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match:
            if pred_match.group(1).replace(",", "").strip() == gt:
                correct += 1
        else:
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt:
                correct += 1
        if (i + 1) % 25 == 0:
            print(f"    GSM8K {label}: {i+1}/{len(ds)}, acc={correct/(i+1)*100:.1f}%",
                  flush=True)
    acc = correct / len(ds) * 100
    print(f"  GSM8K {label}: {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    return acc


def eval_humaneval(model, tokenizer, n_eval, label=""):
    from mlx_lm import generate
    ds = load_humaneval_test(n_eval)
    passed = 0
    for i, ex in enumerate(ds):
        messages = [{"role": "user", "content": (
            f"Complete the following Python function. "
            f"Respond with ONLY the function body code, no explanation.\n\n"
            f"```python\n{ex['prompt']}\n```"
        )}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted,
                          max_tokens=512, verbose=False)
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        completion = code_match.group(1) if code_match else response
        full_code = (ex["prompt"] + completion + "\n\n" +
                    ex["test"] + f"\n\ncheck({ex['entry_point']})\n")
        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                timeout=10, capture_output=True, text=True)
            if result.returncode == 0:
                passed += 1
        except (subprocess.TimeoutExpired, Exception):
            pass
        if (i + 1) % 25 == 0:
            print(f"    HumanEval {label}: {i+1}/{len(ds)}, pass@1={passed/(i+1)*100:.1f}%",
                  flush=True)
    acc = passed / len(ds) * 100
    print(f"  HumanEval {label}: {passed}/{len(ds)} = {acc:.1f}%", flush=True)
    return acc


def eval_medmcqa(model, tokenizer, n_eval, label=""):
    from mlx_lm import generate
    ds = load_medmcqa_val(n_eval)
    option_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct = 0
    for i, ex in enumerate(ds):
        question = (
            f"{ex['question']}\n"
            f"(A) {ex['opa']}\n(B) {ex['opb']}\n(C) {ex['opc']}\n(D) {ex['opd']}")
        messages = [{"role": "user", "content":
            f"Answer this medical question. Reply with only the letter.\n\n{question}"}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=formatted,
                          max_tokens=20, verbose=False)
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
            print(f"    MedMCQA {label}: {i+1}/{len(ds)}, acc={correct/(i+1)*100:.1f}%",
                  flush=True)
    acc = correct / len(ds) * 100
    print(f"  MedMCQA {label}: {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    return acc


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 60)
    print("P0: TT-LoRA Composition Under Benchmarks")
    print(f"SMOKE={IS_SMOKE}, N_EVAL={N_EVAL}, SEED={SEED}")
    print(f"TT solo baselines: {TT_SOLO}")
    print(f"Std LoRA pre-merge: {STD_PREMERGE}")
    print("=" * 60, flush=True)
    log_memory("start")

    results = {
        "experiment": "exp_p0_ttlora_composition_benchmark",
        "is_smoke": IS_SMOKE,
        "n_eval": N_EVAL,
        "seed": SEED,
        "tt_solo_baselines": TT_SOLO,
        "std_premerge_baselines": STD_PREMERGE,
        "base_baselines": BASE,
    }

    tt_adapter_dirs = [E2E_DIR / "adapters" / d for d in DOMAINS]
    lora_adapter_dirs = [LORA_E2E_DIR / "adapters" / d for d in DOMAINS]

    # ── Phase 0: Perturbation norm analysis ──────
    print("\n" + "=" * 60)
    print("PHASE 0: Perturbation norm analysis (Theorem 1 verification)")
    print("=" * 60, flush=True)

    # Compute standard LoRA norms
    print("Computing standard LoRA perturbation norms...", flush=True)
    lora_norms = compute_standard_lora_norms(lora_adapter_dirs)
    for d, norm in zip(DOMAINS, lora_norms):
        print(f"  Standard LoRA {d}: ||DW||_F = {norm:.4f}" if norm else
              f"  Standard LoRA {d}: not available", flush=True)

    # Load model and inject TT-LoRA for norm computation
    from mlx_lm import load
    print("\nLoading model for TT-LoRA norm analysis...", flush=True)
    model, tokenizer = load(MODEL_ID)
    inject_ttlora(model, PROJ_NAMES, TT_RANK, TT_ALPHA)
    log_memory("model-loaded")

    # Compute TT-LoRA norms one at a time
    tt_norms = []
    for domain, adapter_dir in zip(DOMAINS, tt_adapter_dirs):
        load_ttlora_cores(model, adapter_dir, PROJ_NAMES)
        norm_sq = 0.0
        for layer in get_layers(model):
            for pname in PROJ_NAMES:
                proj = getattr(layer.self_attn, pname)
                if isinstance(proj, TTLoRAWrapper):
                    dw = proj.reconstruct_delta_w()
                    mx.eval(dw)
                    norm_sq += (dw * dw).sum().item()
                    proj._cached_delta_w = None
        norm = norm_sq ** 0.5
        tt_norms.append(norm)
        print(f"  TT-LoRA {domain}: ||DW||_F = {norm:.4f}", flush=True)

    # Norm ratios
    norm_ratios = []
    for d, tt_n, lora_n in zip(DOMAINS, tt_norms, lora_norms):
        if lora_n and lora_n > 0:
            ratio = tt_n / lora_n
            norm_ratios.append(ratio)
            print(f"  Norm ratio {d}: {ratio:.4f} (predicted ~0.21)", flush=True)

    avg_ratio = sum(norm_ratios) / len(norm_ratios) if norm_ratios else None
    print(f"\n  Average norm ratio: {avg_ratio:.4f} (predicted ~0.21)" if avg_ratio
          else "  Cannot compute norm ratio", flush=True)

    results["tt_norms"] = {d: round(n, 4) for d, n in zip(DOMAINS, tt_norms)}
    results["lora_norms"] = {d: round(n, 4) if n else None for d, n in zip(DOMAINS, lora_norms)}
    results["norm_ratios"] = {d: round(r, 4) for d, r in zip(DOMAINS, norm_ratios)}
    results["avg_norm_ratio"] = round(avg_ratio, 4) if avg_ratio else None
    results["predicted_norm_ratio"] = 0.21

    # ── Phase 1: Pre-merged TT-LoRA benchmarks ──
    print("\n" + "=" * 60)
    print("PHASE 1: Pre-merged 3 TT-LoRA adapters (sum of DW)")
    print("=" * 60, flush=True)

    # Clear any cached state, then pre-merge all 3
    for layer in get_layers(model):
        for pname in PROJ_NAMES:
            proj = getattr(layer.self_attn, pname)
            if isinstance(proj, TTLoRAWrapper):
                proj._cached_delta_w = None

    print("Pre-merging all 3 TT-LoRA adapters...", flush=True)
    premerge_norms = premerge_ttlora_adapters(
        model, tt_adapter_dirs, PROJ_NAMES)
    results["premerge_norms"] = [round(n, 4) for n in premerge_norms]

    model.eval()

    print("\n--- GSM8K (pre-merged) ---", flush=True)
    merged_gsm8k = eval_gsm8k(model, tokenizer, N_EVAL, "merged")

    print("\n--- HumanEval (pre-merged) ---", flush=True)
    merged_humaneval = eval_humaneval(model, tokenizer, N_EVAL, "merged")

    print("\n--- MedMCQA (pre-merged) ---", flush=True)
    merged_medmcqa = eval_medmcqa(model, tokenizer, N_EVAL, "merged")

    results["merged_gsm8k_pct"] = round(merged_gsm8k, 1)
    results["merged_humaneval_pct"] = round(merged_humaneval, 1)
    results["merged_medmcqa_pct"] = round(merged_medmcqa, 1)

    # Checkpoint
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"Checkpoint saved after pre-merge phase.", flush=True)

    # ── Phase 2: Per-query routed TT-LoRA ────────
    print("\n" + "=" * 60)
    print("PHASE 2: Per-query routed TT-LoRA (one adapter per query)")
    print("=" * 60, flush=True)

    # GSM8K with math adapter
    print("\n--- GSM8K (math adapter) ---", flush=True)
    for layer in get_layers(model):
        for pname in PROJ_NAMES:
            proj = getattr(layer.self_attn, pname)
            if isinstance(proj, TTLoRAWrapper):
                proj._cached_delta_w = None
    load_ttlora_cores(model, E2E_DIR / "adapters" / "math", PROJ_NAMES)
    cache_all_delta_w(model, PROJ_NAMES)
    routed_gsm8k = eval_gsm8k(model, tokenizer, N_EVAL, "routed-math")

    # HumanEval with code adapter
    print("\n--- HumanEval (code adapter) ---", flush=True)
    for layer in get_layers(model):
        for pname in PROJ_NAMES:
            proj = getattr(layer.self_attn, pname)
            if isinstance(proj, TTLoRAWrapper):
                proj._cached_delta_w = None
    load_ttlora_cores(model, E2E_DIR / "adapters" / "code", PROJ_NAMES)
    cache_all_delta_w(model, PROJ_NAMES)
    routed_humaneval = eval_humaneval(model, tokenizer, N_EVAL, "routed-code")

    # MedMCQA with medical adapter
    print("\n--- MedMCQA (medical adapter) ---", flush=True)
    for layer in get_layers(model):
        for pname in PROJ_NAMES:
            proj = getattr(layer.self_attn, pname)
            if isinstance(proj, TTLoRAWrapper):
                proj._cached_delta_w = None
    load_ttlora_cores(model, E2E_DIR / "adapters" / "medical", PROJ_NAMES)
    cache_all_delta_w(model, PROJ_NAMES)
    routed_medmcqa = eval_medmcqa(model, tokenizer, N_EVAL, "routed-medical")

    results["routed_gsm8k_pct"] = round(routed_gsm8k, 1)
    results["routed_humaneval_pct"] = round(routed_humaneval, 1)
    results["routed_medmcqa_pct"] = round(routed_medmcqa, 1)

    cleanup(model, tokenizer)

    # ── Kill criteria evaluation ─────────────────
    total_time = time.time() - t_start

    # K1447: Pre-merged GSM8K >= 60%
    k1447_pass = merged_gsm8k >= 60.0
    # K1448: Pre-merged HumanEval >= 45%
    k1448_pass = merged_humaneval >= 45.0
    # K1449: Pre-merged MedMCQA >= 25%
    k1449_pass = merged_medmcqa >= 25.0
    # K1450: Routed within 5pp of solo on all 3
    k1450_gsm8k = abs(routed_gsm8k - TT_SOLO["gsm8k"]) <= 5
    k1450_humaneval = abs(routed_humaneval - TT_SOLO["humaneval"]) <= 5
    k1450_medmcqa = abs(routed_medmcqa - TT_SOLO["medmcqa"]) <= 5
    k1450_pass = k1450_gsm8k and k1450_humaneval and k1450_medmcqa

    results["K1447_premerge_gsm8k"] = "PASS" if k1447_pass else "FAIL"
    results["K1448_premerge_humaneval"] = "PASS" if k1448_pass else "FAIL"
    results["K1449_premerge_medmcqa"] = "PASS" if k1449_pass else "FAIL"
    results["K1450_routed_within_5pp"] = "PASS" if k1450_pass else "FAIL"
    results["K1450_detail"] = {
        "gsm8k": "PASS" if k1450_gsm8k else "FAIL",
        "humaneval": "PASS" if k1450_humaneval else "FAIL",
        "medmcqa": "PASS" if k1450_medmcqa else "FAIL",
    }
    results["total_time_s"] = round(total_time, 1)

    # ── Summary ──────────────────────────────────
    print("\n" + "=" * 60)
    print("PREDICTION VS MEASUREMENT")
    print("=" * 60)
    hdr = f"{'Metric':<25} {'Solo':<8} {'Merged':<8} {'Routed':<8} {'StdMerge':<8} {'Predicted':<12} {'Kill':<6}"
    print(hdr)
    print("-" * len(hdr))

    print(f"{'GSM8K':<25} {TT_SOLO['gsm8k']:<8.1f} {merged_gsm8k:<8.1f} "
          f"{routed_gsm8k:<8.1f} {STD_PREMERGE['gsm8k']:<8.1f} {'58-65%':<12} "
          f"{'PASS' if k1447_pass else 'FAIL':<6}")
    print(f"{'HumanEval':<25} {TT_SOLO['humaneval']:<8.1f} {merged_humaneval:<8.1f} "
          f"{routed_humaneval:<8.1f} {STD_PREMERGE['humaneval']:<8.1f} {'47-55%':<12} "
          f"{'PASS' if k1448_pass else 'FAIL':<6}")
    print(f"{'MedMCQA':<25} {TT_SOLO['medmcqa']:<8.1f} {merged_medmcqa:<8.1f} "
          f"{routed_medmcqa:<8.1f} {STD_PREMERGE['medmcqa']:<8.1f} {'18-21%':<12} "
          f"{'PASS' if k1449_pass else 'FAIL':<6}")
    print(f"{'Routed vs Solo (5pp)':<25} {'—':<8} {'—':<8} {'all':<8} {'—':<8} {'<1pp':<12} "
          f"{'PASS' if k1450_pass else 'FAIL':<6}")

    if avg_ratio is not None:
        print(f"\n{'Norm ratio (TT/LoRA)':<25} {avg_ratio:.4f} (predicted ~0.21)")
        print(f"{'Interference ratio':<25} {avg_ratio**2:.4f}x std LoRA (predicted ~0.044x)")

    print(f"\n{'='*60}")
    print(f"K1447 Pre-merge GSM8K >=60%:   {results['K1447_premerge_gsm8k']} ({merged_gsm8k:.1f}%)")
    print(f"K1448 Pre-merge HumanEval >=45%: {results['K1448_premerge_humaneval']} ({merged_humaneval:.1f}%)")
    print(f"K1449 Pre-merge MedMCQA >=25%: {results['K1449_premerge_medmcqa']} ({merged_medmcqa:.1f}%)")
    print(f"K1450 Routed <=5pp of solo:    {results['K1450_routed_within_5pp']}")
    print(f"  GSM8K:     {routed_gsm8k:.1f}% vs {TT_SOLO['gsm8k']}% (delta {routed_gsm8k - TT_SOLO['gsm8k']:+.1f}pp)")
    print(f"  HumanEval: {routed_humaneval:.1f}% vs {TT_SOLO['humaneval']}% (delta {routed_humaneval - TT_SOLO['humaneval']:+.1f}pp)")
    print(f"  MedMCQA:   {routed_medmcqa:.1f}% vs {TT_SOLO['medmcqa']}% (delta {routed_medmcqa - TT_SOLO['medmcqa']:+.1f}pp)")
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
