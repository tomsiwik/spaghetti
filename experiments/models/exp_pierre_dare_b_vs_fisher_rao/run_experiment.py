"""exp_pierre_dare_b_vs_fisher_rao — measure B-space DARE vs Fisher-Rao Karcher
mean in Pierre's shared-A B-only architecture, with full-delta DARE as the
upper bound for architectural transfer.

Usage:
    experiment run exp_pierre_dare_b_vs_fisher_rao

Outputs:
    results.json — measured accuracies + KC verdicts per pre-registered MATH.md
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tooling.scripts.polar_train import (  # type: ignore
    PoLARLinear, RANK, SCALE, _get_layers, inject_polar_adapters,
    eval_gsm8k, eval_humaneval, eval_medqa, cleanup,
)

from compose_methods import (  # local
    compose_fisher_rao, compose_dare_b, compose_full_delta_dare,
)

# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────

MODEL_NAME = "mlx-community/gemma-4-e4b-it-4bit"
ADAPTERS_ROOT = REPO_ROOT / "data" / "adapters"

# Match the 7-adapter set used in exp_pierre_dare_vs_m2p_gated_postfix and
# exp_pierre_composition_method_ablation so K4's 73.3% reproducibility
# reference is apples-to-apples. (logical name → directory)
ADAPTERS = [
    ("strategy_full",      "strategy_full_polar"),
    ("strategy_prepare",   "strategy_prepare_polar"),
    ("strategy_act",       "strategy_act_polar"),
    ("strategy_integrate", "strategy_integrate_polar"),
    ("domain_math",        "math_polar"),
    ("domain_code",        "code_polar"),
    ("domain_medical",     "medical_polar"),
]
ADAPTER_NAMES = [a[0] for a in ADAPTERS]
ADAPTER_DIRS = dict(ADAPTERS)

N_EVAL = 50          # per benchmark
SEED = 42
DROP_RATE = 0.9
DARE_RESEARCH_REFERENCE_AVG = 73.3  # for K4 reproducibility check

# Pierre's per-domain "best single" mapping per benchmark
SINGLE_BEST_FOR_BENCH = {
    "gsm8k":     "domain_math",
    "humaneval": "domain_code",
    "medqa":     "domain_medical",
}

OUTPUT_PATH = Path(__file__).parent / "results.json"

# ─────────────────────────────────────────────────────────────────────────
# Fused-delta linear (Finding #831 canonical pattern)
# ─────────────────────────────────────────────────────────────────────────

class _FusedDeltaLinear(nn.Module):
    """base(x) + x @ fused_delta — for the full-delta DARE path."""
    def __init__(self, base_layer, fused_delta):
        super().__init__()
        self.base = base_layer
        self._fused = fused_delta

    def __call__(self, x):
        return self.base(x) + (x @ self._fused.astype(x.dtype))


# ─────────────────────────────────────────────────────────────────────────
# Adapter loading
# ─────────────────────────────────────────────────────────────────────────

def load_adapter_state(name: str) -> dict[str, dict[str, mx.array]]:
    """Load one adapter file → {layer_key: {"a", "b", "scale"}}.

    `name` is a logical name; directory comes from ADAPTER_DIRS.
    """
    dir_name = ADAPTER_DIRS[name]
    path = ADAPTERS_ROOT / dir_name / "polar.safetensors"
    if not path.exists():
        candidates = list((ADAPTERS_ROOT / dir_name).glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No safetensors in {ADAPTERS_ROOT / dir_name}")
        path = candidates[0]
    raw = mx.load(str(path))
    state: dict[str, dict[str, mx.array]] = {}
    for k, v in raw.items():
        # keys look like "layer_{L}.lora_a" or "layer_{L}.lora_b"
        layer_key, attr = k.rsplit(".", 1)
        state.setdefault(layer_key, {"scale": SCALE})
        if attr == "lora_a":
            state[layer_key]["a"] = v.astype(mx.float32)
        elif attr == "lora_b":
            state[layer_key]["b"] = v.astype(mx.float32)
    return state


def stack_B_dicts(states: list[dict]) -> list[dict[str, mx.array]]:
    """Project list of full adapter states down to B-only dicts (Pierre arch)."""
    out = []
    for st in states:
        out.append({k: v["b"] for k, v in st.items() if "b" in v})
    return out


# ─────────────────────────────────────────────────────────────────────────
# Model state setup per method
# ─────────────────────────────────────────────────────────────────────────

def reset_to_polar_path(model, modules, base_q_projs):
    """Ensure model is in PoLARLinear path (q_proj is PoLARLinear).

    `base_q_projs` is the snapshot of the original (unwrapped) base q_proj
    instances captured at the very start, BEFORE any wrapping.
    """
    layers = _get_layers(model)
    for li, layer in enumerate(layers):
        cur = layer.self_attn.q_proj
        if not isinstance(cur, PoLARLinear):
            # Currently a fused wrapper (or something else) — restore base then re-wrap
            layer.self_attn.q_proj = base_q_projs[li]
            new_polar = PoLARLinear(base_q_projs[li], rank=RANK, scale=SCALE, seed=SEED)
            layer.self_attn.q_proj = new_polar
            modules[li] = new_polar


def install_polar_state(modules, A_dict, B_dict):
    """Set lora_a and lora_b on all PoLARLinear modules from per-layer dicts."""
    for li, m in enumerate(modules):
        key = f"layer_{li}"
        if key in A_dict:
            m.lora_a = A_dict[key].astype(mx.float32)
        if key in B_dict:
            m.lora_b = B_dict[key].astype(mx.float32)
    # Realize before generation
    mx.eval(*[m.lora_a for m in modules], *[m.lora_b for m in modules])


def install_fused_delta(model, base_q_projs, fused: dict[str, mx.array]):
    """Replace q_proj with _FusedDeltaLinear at every layer."""
    layers = _get_layers(model)
    for li, layer in enumerate(layers):
        key = f"layer_{li}"
        if key not in fused:
            continue
        wrapper = _FusedDeltaLinear(base_q_projs[li], fused[key])
        layer.self_attn.q_proj = wrapper
    # Realize the fused weights to avoid lazy graph blowup at first generate
    arrs = [layer.self_attn.q_proj._fused
            for layer in layers
            if isinstance(layer.self_attn.q_proj, _FusedDeltaLinear)]
    if arrs:
        mx.eval(*arrs)


# ─────────────────────────────────────────────────────────────────────────
# Per-method evaluation
# ─────────────────────────────────────────────────────────────────────────

def run_all_evals(model, tokenizer, label: str) -> dict:
    """Run gsm8k + humaneval + medqa, return dict of scores."""
    print(f"\n=== Evaluating: {label} ===", flush=True)
    t0 = time.time()
    gsm8k = eval_gsm8k(model, tokenizer, n_eval=N_EVAL, seed=SEED)
    print(f"  gsm8k: {gsm8k:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    he = eval_humaneval(model, tokenizer, n_eval=N_EVAL)
    print(f"  humaneval: {he:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    md = eval_medqa(model, tokenizer, n_eval=N_EVAL, seed=SEED)
    print(f"  medqa: {md:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    avg = (gsm8k + he + md) / 3.0
    print(f"  avg: {avg:.1f}%", flush=True)
    return {"gsm8k": gsm8k, "humaneval": he, "medqa": md, "avg": avg}


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    print(f"=== exp_pierre_dare_b_vs_fisher_rao (N={N_EVAL}/bench) ===")

    # Load model + tokenizer
    from mlx_lm import load
    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = load(MODEL_NAME)

    # Snapshot original q_proj per layer (BEFORE injecting adapters)
    layers = _get_layers(model)
    base_q_projs = [layer.self_attn.q_proj for layer in layers]
    print(f"  {len(base_q_projs)} transformer layers")

    # Load all 7 adapter states
    print("Loading adapters...")
    adapter_states: dict[str, dict[str, dict[str, mx.array]]] = {}
    for name in ADAPTER_NAMES:
        adapter_states[name] = load_adapter_state(name)
        n_layers = len(adapter_states[name])
        print(f"  {name}: {n_layers} layers loaded")

    # Pierre's "shared A": adopt adapter[0]'s A on every layer.
    # This approximates Pierre's frozen-shared-A design using the existing
    # research adapters (which were trained with the same seed → same initial A).
    shared_A_donor = ADAPTER_NAMES[0]
    shared_A_dict = {k: v["a"] for k, v in adapter_states[shared_A_donor].items() if "a" in v}
    print(f"  shared-A donor: {shared_A_donor} ({len(shared_A_dict)} layers)")

    # Inject PoLARLinear once; we'll swap parameters per method
    print("Injecting PoLARLinear into q_proj at every layer...")
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    print(f"  {len(modules)} PoLAR modules attached")

    results: dict = {
        "config": {
            "model": MODEL_NAME,
            "n_eval_per_bench": N_EVAL,
            "seed": SEED,
            "drop_rate": DROP_RATE,
            "rank": RANK,
            "scale": SCALE,
            "shared_A_donor": shared_A_donor,
            "n_adapters": len(ADAPTER_NAMES),
            "adapter_names": ADAPTER_NAMES,
            "dare_research_reference_avg": DARE_RESEARCH_REFERENCE_AVG,
            "single_best_for_bench": SINGLE_BEST_FOR_BENCH,
        },
        "methods": {},
    }

    # ─────────────────────────────────────────────────────────────────────
    # M0: single_best per benchmark — each benchmark uses its native adapter
    # ─────────────────────────────────────────────────────────────────────
    single_best_per_bench: dict[str, float] = {}
    for bench, adapter_name in SINGLE_BEST_FOR_BENCH.items():
        reset_to_polar_path(model, modules, base_q_projs)
        st = adapter_states[adapter_name]
        A_dict = {k: v["a"] for k, v in st.items() if "a" in v}
        B_dict = {k: v["b"] for k, v in st.items() if "b" in v}
        install_polar_state(modules, A_dict, B_dict)
        print(f"\n--- single_best={adapter_name} on {bench} ---", flush=True)
        t0 = time.time()
        if bench == "gsm8k":
            score = eval_gsm8k(model, tokenizer, n_eval=N_EVAL, seed=SEED)
        elif bench == "humaneval":
            score = eval_humaneval(model, tokenizer, n_eval=N_EVAL)
        else:
            score = eval_medqa(model, tokenizer, n_eval=N_EVAL, seed=SEED)
        elapsed = time.time() - t0
        single_best_per_bench[bench] = score
        print(f"  {bench}={score:.1f}% ({elapsed:.0f}s)", flush=True)
        # Persist incremental results so partial runs are useful
        results["methods"]["single_best"] = {
            "per_bench": single_best_per_bench,
            "avg": (sum(single_best_per_bench.values()) / len(single_best_per_bench))
                   if len(single_best_per_bench) == len(SINGLE_BEST_FOR_BENCH)
                   else None,
        }
        OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    single_best_avg = sum(single_best_per_bench.values()) / 3.0
    results["methods"]["single_best"]["avg"] = single_best_avg
    print(f"\nsingle_best avg = {single_best_avg:.1f}%")

    # ─────────────────────────────────────────────────────────────────────
    # M1: fisher_rao (Pierre current default)
    # ─────────────────────────────────────────────────────────────────────
    reset_to_polar_path(model, modules, base_q_projs)
    B_lists = stack_B_dicts([adapter_states[n] for n in ADAPTER_NAMES])
    fisher_B = compose_fisher_rao(B_lists)
    install_polar_state(modules, shared_A_dict, fisher_B)
    fisher_results = run_all_evals(model, tokenizer, "fisher_rao (Pierre default)")
    results["methods"]["fisher_rao"] = fisher_results
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    # ─────────────────────────────────────────────────────────────────────
    # M2: dare_b (new candidate)
    # ─────────────────────────────────────────────────────────────────────
    reset_to_polar_path(model, modules, base_q_projs)
    dare_B = compose_dare_b(B_lists, drop_rate=DROP_RATE, seed=SEED)
    install_polar_state(modules, shared_A_dict, dare_B)
    dare_b_results = run_all_evals(model, tokenizer, "dare_b (B-space DARE)")
    results["methods"]["dare_b"] = dare_b_results
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    # ─────────────────────────────────────────────────────────────────────
    # M3: dare_full_delta (research upper bound)
    # ─────────────────────────────────────────────────────────────────────
    fused = compose_full_delta_dare(
        [adapter_states[n] for n in ADAPTER_NAMES],
        drop_rate=DROP_RATE, seed=SEED,
    )
    install_fused_delta(model, base_q_projs, fused)
    full_delta_results = run_all_evals(model, tokenizer, "dare_full_delta (research arch)")
    results["methods"]["dare_full_delta"] = full_delta_results
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    # ─────────────────────────────────────────────────────────────────────
    # KC evaluation
    # ─────────────────────────────────────────────────────────────────────
    fr_avg = results["methods"]["fisher_rao"]["avg"]
    db_avg = results["methods"]["dare_b"]["avg"]
    fd_avg = results["methods"]["dare_full_delta"]["avg"]

    k1 = db_avg >= fr_avg
    k2 = db_avg >= single_best_avg + 2.0
    k3 = abs(db_avg - fd_avg) <= 5.0
    k4 = abs(fr_avg - DARE_RESEARCH_REFERENCE_AVG) <= 3.0

    results["kill_criteria"] = {
        "K1_decision_dare_b_ge_fisher_rao": {
            "pass": bool(k1),
            "dare_b_avg": db_avg,
            "fisher_rao_avg": fr_avg,
            "delta_pp": db_avg - fr_avg,
        },
        "K2_composition_value_2pp": {
            "pass": bool(k2),
            "dare_b_avg": db_avg,
            "single_best_avg": single_best_avg,
            "delta_pp": db_avg - single_best_avg,
        },
        "K3_arch_transfer_within_5pp": {
            "pass": bool(k3),
            "dare_b_avg": db_avg,
            "dare_full_delta_avg": fd_avg,
            "abs_delta_pp": abs(db_avg - fd_avg),
        },
        "K4_reproducibility_fisher_rao_within_3pp": {
            "pass": bool(k4),
            "fisher_rao_avg": fr_avg,
            "reference_avg": DARE_RESEARCH_REFERENCE_AVG,
            "abs_delta_pp": abs(fr_avg - DARE_RESEARCH_REFERENCE_AVG),
        },
    }

    if not k4:
        verdict = "INCONCLUSIVE"  # eval pipeline drift
        decision = "Re-run with diagnostics; eval pipeline differs from research"
    elif k1:
        verdict = "SUPPORTED"
        decision = "Swap Pierre default to dare_b"
    else:
        verdict = "KILLED"
        decision = "Keep Fisher-Rao; B-space DARE does not transfer"
    results["verdict"] = verdict
    results["decision"] = decision

    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n=== {verdict}: {decision} ===")
    print(f"  K1 (decision)        : {'PASS' if k1 else 'FAIL'}  Δ={db_avg-fr_avg:+.1f}pp")
    print(f"  K2 (compose value)   : {'PASS' if k2 else 'FAIL'}  Δ={db_avg-single_best_avg:+.1f}pp vs single_best")
    print(f"  K3 (arch transfer)   : {'PASS' if k3 else 'FAIL'}  |Δ|={abs(db_avg-fd_avg):.1f}pp vs full-delta")
    print(f"  K4 (reproducibility) : {'PASS' if k4 else 'FAIL'}  |Δ|={abs(fr_avg-DARE_RESEARCH_REFERENCE_AVG):.1f}pp vs 73.3 ref")
    print(f"\nResults: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
