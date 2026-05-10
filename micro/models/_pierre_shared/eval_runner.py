"""Shared evaluation runner for Pierre composition experiments.

Each experiment under `micro/models/exp_pierre_*/` provides a single new
composition method to evaluate. This module handles all boilerplate:

  - load Gemma 4 base + tokenizer
  - load 7 PoLAR adapter states (per-adapter A and B)
  - inject PoLARLinear modules
  - capture base q_proj snapshots for fused-delta path swaps
  - run the standard 4-method matrix:
        M0 single_best per benchmark
        M1 fisher_rao  (Pierre current default — verbatim from product code)
        M2 <new method under test> (provided by experiment)
        M3 dare_full_delta (research upper bound — per-adapter A, fused delta)
  - eval each method × {gsm8k, humaneval, medqa} at N samples
  - compute KCs and write incremental results.json

Usage in an experiment's `run_experiment.py`:

    from _pierre_shared.eval_runner import (
        run_pierre_compose_experiment, MethodSpec,
    )

    def my_compose(B_lists, A_dict, **kwargs):
        ...
        return composed_B_dict   # same shape as Pierre's compose_adapters output

    run_pierre_compose_experiment(
        method=MethodSpec(name="ace_b", kind="b_only", fn=my_compose),
        kc_thresholds={...},
        out_path=Path(__file__).parent / "results.json",
    )
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import mlx.core as mx
import mlx.nn as nn

# Allow importing from repo root (scripts/polar_train.py)
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.polar_train import (  # type: ignore  # noqa: E402
    PoLARLinear, RANK, SCALE, _get_layers, inject_polar_adapters,
    eval_gsm8k, eval_humaneval, eval_medqa,
)

# ─────────────────────────────────────────────────────────────────────────
# Config (shared across all Pierre composition experiments)
# ─────────────────────────────────────────────────────────────────────────

MODEL_NAME = "mlx-community/gemma-4-e4b-it-4bit"
ADAPTERS_ROOT = REPO_ROOT / "adapters"
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
N_EVAL = 50
SEED = 42
DROP_RATE = 0.9
DARE_RESEARCH_REFERENCE_AVG = 71.3  # measured in exp_pierre_dare_b_vs_fisher_rao (full-delta DARE)
FISHER_RAO_REFERENCE_AVG = 64.7      # measured in exp_pierre_dare_b_vs_fisher_rao
SINGLE_BEST_FOR_BENCH = {
    "gsm8k":     "domain_math",
    "humaneval": "domain_code",
    "medqa":     "domain_medical",
}

# ─────────────────────────────────────────────────────────────────────────
# Fused-delta wrapper (Finding #831 canonical pattern)
# ─────────────────────────────────────────────────────────────────────────

class _FusedDeltaLinear(nn.Module):
    """base(x) + x @ fused_delta — the canonical pattern from Finding #831."""
    def __init__(self, base_layer, fused_delta):
        super().__init__()
        self.base = base_layer
        self._fused = fused_delta

    def __call__(self, x):
        return self.base(x) + (x @ self._fused.astype(x.dtype))


# ─────────────────────────────────────────────────────────────────────────
# Verbatim Fisher-Rao composition (Pierre product code copy)
# ─────────────────────────────────────────────────────────────────────────

def compose_fisher_rao(adapter_Bs, A_dict=None, weights=None):
    """Verbatim copy of Pierre's `compose_adapters` from compose.py."""
    if len(adapter_Bs) == 1:
        return adapter_Bs[0]
    if weights is None:
        weights = [1.0 / len(adapter_Bs)] * len(adapter_Bs)
    all_keys = set()
    for ab in adapter_Bs:
        all_keys.update(ab.keys())
    composed = {}
    for key in all_keys:
        tensors = [ab[key] for ab in adapter_Bs if key in ab]
        w = weights[: len(tensors)]
        composed[key] = _norm_rescaled_average(tensors, w)
    return composed


def _norm_rescaled_average(tensors, weights):
    if len(tensors) == 1:
        return tensors[0]
    w_sum = sum(weights)
    mean = sum(t.astype(mx.float32) * (w / w_sum) for t, w in zip(tensors, weights))
    norms = mx.stack([mx.linalg.norm(t.reshape(-1).astype(mx.float32)) for t in tensors])
    source_norm = mx.mean(norms)
    mean_norm = mx.linalg.norm(mean.reshape(-1))
    mx.eval(source_norm, mean_norm)
    if mean_norm.item() > 1e-8:
        return (mean * (source_norm / mean_norm)).astype(mx.bfloat16)
    return mean.astype(mx.bfloat16)


# ─────────────────────────────────────────────────────────────────────────
# Full-delta DARE (research upper bound)
# ─────────────────────────────────────────────────────────────────────────

def compose_full_delta_dare(adapter_states, weights=None, drop_rate=0.9, seed=42):
    import numpy as np
    K = len(adapter_states)
    if K == 0:
        return {}
    if weights is None:
        weights = [1.0 / K] * K
    all_keys = set()
    for st in adapter_states:
        all_keys.update(st.keys())
    rng = np.random.default_rng(seed)
    keep = 1.0 - drop_rate
    w_sum = sum(weights)
    fused = {}
    for key in sorted(all_keys):
        per_adapter = [st[key] for st in adapter_states if key in st]
        contributions = []
        for entry, w in zip(per_adapter, weights[: len(per_adapter)]):
            A = entry["a"].astype(mx.float32)
            B = entry["b"].astype(mx.float32)
            scale = float(entry["scale"])
            delta = scale * (A @ B)
            mx.eval(delta)
            mask_np = (rng.random(delta.shape) < keep).astype(np.float32)
            mask = mx.array(mask_np)
            delta_dare = (delta * mask) / keep
            contributions.append(delta_dare * (w / w_sum))
        fused_delta = contributions[0]
        for c in contributions[1:]:
            fused_delta = fused_delta + c
        mx.eval(fused_delta)
        fused[key] = fused_delta.astype(mx.bfloat16)
    return fused


# ─────────────────────────────────────────────────────────────────────────
# Adapter loading
# ─────────────────────────────────────────────────────────────────────────

def load_adapter_state(name: str) -> dict:
    dir_name = ADAPTER_DIRS[name]
    path = ADAPTERS_ROOT / dir_name / "polar.safetensors"
    if not path.exists():
        cands = list((ADAPTERS_ROOT / dir_name).glob("*.safetensors"))
        if not cands:
            raise FileNotFoundError(f"No safetensors in {ADAPTERS_ROOT / dir_name}")
        path = cands[0]
    raw = mx.load(str(path))
    state: dict = {}
    for k, v in raw.items():
        layer_key, attr = k.rsplit(".", 1)
        state.setdefault(layer_key, {"scale": SCALE})
        if attr == "lora_a":
            state[layer_key]["a"] = v.astype(mx.float32)
        elif attr == "lora_b":
            state[layer_key]["b"] = v.astype(mx.float32)
    return state


def stack_B_dicts(states: list[dict]) -> list[dict]:
    return [{k: v["b"] for k, v in st.items() if "b" in v} for st in states]


# ─────────────────────────────────────────────────────────────────────────
# Method spec
# ─────────────────────────────────────────────────────────────────────────

MethodKind = Literal["b_only", "fused_delta"]


@dataclass
class MethodSpec:
    """Description of a composition method to evaluate.

    For kind="b_only": fn(B_lists, A_dict, **kwargs) → composed_B_dict
    For kind="fused_delta": fn(adapter_states, A_dict, scale, **kwargs) → fused_delta_dict
    """
    name: str
    kind: MethodKind
    fn: Callable
    fn_kwargs: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────
# Model state setup
# ─────────────────────────────────────────────────────────────────────────

def reset_to_polar_path(model, modules, base_q_projs):
    layers = _get_layers(model)
    for li, layer in enumerate(layers):
        cur = layer.self_attn.q_proj
        if not isinstance(cur, PoLARLinear):
            layer.self_attn.q_proj = base_q_projs[li]
            new_polar = PoLARLinear(base_q_projs[li], rank=RANK, scale=SCALE, seed=SEED)
            layer.self_attn.q_proj = new_polar
            modules[li] = new_polar


def install_polar_state(modules, A_dict, B_dict):
    for li, m in enumerate(modules):
        key = f"layer_{li}"
        if key in A_dict:
            m.lora_a = A_dict[key].astype(mx.float32)
        if key in B_dict:
            m.lora_b = B_dict[key].astype(mx.float32)
    mx.eval(*[m.lora_a for m in modules], *[m.lora_b for m in modules])


def install_fused_delta(model, base_q_projs, fused: dict):
    layers = _get_layers(model)
    for li, layer in enumerate(layers):
        key = f"layer_{li}"
        if key not in fused:
            continue
        wrapper = _FusedDeltaLinear(base_q_projs[li], fused[key])
        layer.self_attn.q_proj = wrapper
    arrs = [layer.self_attn.q_proj._fused
            for layer in layers
            if isinstance(layer.self_attn.q_proj, _FusedDeltaLinear)]
    if arrs:
        mx.eval(*arrs)


# ─────────────────────────────────────────────────────────────────────────
# Eval loop
# ─────────────────────────────────────────────────────────────────────────

def run_all_evals(model, tokenizer, label: str) -> dict:
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
# Main entry point
# ─────────────────────────────────────────────────────────────────────────

def run_pierre_compose_experiment(
    method: MethodSpec,
    kc_thresholds: dict,
    out_path: Path,
    extra_config: Optional[dict] = None,
    adapter_names_override: Optional[list[str]] = None,
):
    """Run the standard 4-method matrix with `method` as M2.

    kc_thresholds keys (used by compute_kcs):
        - 'k1_min_delta_over_fisher_rao'  (e.g. 3.0 — must beat Fisher-Rao by ≥ this many pp)
        - 'k2_max_delta_under_full_delta_dare' (e.g. 4.0 — within this many pp of upper bound)
        - 'k3_max_preprocess_seconds' (e.g. 5.0 — preprocessing must complete within budget)
        - 'k4_label': str — name of the sanity check
        - 'k4_value': measured value of the sanity check (filled in by experiment)
        - 'k4_threshold': float — threshold for k4
    """
    adapters_active = adapter_names_override if adapter_names_override is not None else ADAPTER_NAMES
    print(f"=== {method.name} (N={N_EVAL}/bench, K={len(adapters_active)}) ===")
    if adapter_names_override is not None:
        print(f"  Adapter override: {adapters_active}")

    from mlx_lm import load
    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = load(MODEL_NAME)

    layers = _get_layers(model)
    base_q_projs = [layer.self_attn.q_proj for layer in layers]
    print(f"  {len(base_q_projs)} transformer layers")

    print("Loading adapters...")
    adapter_states = {}
    for name in adapters_active:
        adapter_states[name] = load_adapter_state(name)
        print(f"  {name}: {len(adapter_states[name])} layers")

    shared_A_donor = adapters_active[0]
    shared_A_dict = {k: v["a"] for k, v in adapter_states[shared_A_donor].items() if "a" in v}
    print(f"  shared-A donor: {shared_A_donor}")

    print("Injecting PoLARLinear...")
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)

    results = {
        "config": {
            "model": MODEL_NAME,
            "n_eval_per_bench": N_EVAL,
            "seed": SEED,
            "rank": RANK,
            "scale": SCALE,
            "shared_A_donor": shared_A_donor,
            "n_adapters": len(ADAPTER_NAMES),
            "adapter_names": ADAPTER_NAMES,
            "method_under_test": method.name,
            "method_kind": method.kind,
            "method_kwargs": {k: (str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v) for k, v in method.fn_kwargs.items()},
            "fisher_rao_reference_avg": FISHER_RAO_REFERENCE_AVG,
            "dare_research_reference_avg": DARE_RESEARCH_REFERENCE_AVG,
            **(extra_config or {}),
        },
        "methods": {},
        "kc_thresholds": kc_thresholds,
    }

    def save():
        out_path.write_text(json.dumps(results, indent=2, default=str))

    # M0 single_best per benchmark
    sb_per_bench = {}
    for bench, adapter_name in SINGLE_BEST_FOR_BENCH.items():
        if adapter_name not in adapter_states:
            print(f"\n--- single_best={adapter_name} on {bench}: SKIPPED (not loaded) ---", flush=True)
            continue
        reset_to_polar_path(model, modules, base_q_projs)
        st = adapter_states[adapter_name]
        A_d = {k: v["a"] for k, v in st.items() if "a" in v}
        B_d = {k: v["b"] for k, v in st.items() if "b" in v}
        install_polar_state(modules, A_d, B_d)
        print(f"\n--- single_best={adapter_name} on {bench} ---", flush=True)
        t0 = time.time()
        score = (eval_gsm8k(model, tokenizer, n_eval=N_EVAL, seed=SEED) if bench == "gsm8k"
                 else eval_humaneval(model, tokenizer, n_eval=N_EVAL) if bench == "humaneval"
                 else eval_medqa(model, tokenizer, n_eval=N_EVAL, seed=SEED))
        sb_per_bench[bench] = score
        print(f"  {bench}={score:.1f}% ({time.time()-t0:.0f}s)", flush=True)
        results["methods"]["single_best"] = {"per_bench": dict(sb_per_bench), "avg": None}
        save()
    if sb_per_bench:
        sb_avg = sum(sb_per_bench.values()) / len(sb_per_bench)
        results["methods"]["single_best"]["avg"] = sb_avg
    else:
        results["methods"]["single_best"] = {"per_bench": {}, "avg": None, "note": "skipped — single_best adapters not in override set"}
    save()

    # M1 Fisher-Rao
    reset_to_polar_path(model, modules, base_q_projs)
    B_lists = stack_B_dicts([adapter_states[n] for n in adapters_active])
    fr_B = compose_fisher_rao(B_lists)
    install_polar_state(modules, shared_A_dict, fr_B)
    fr_results = run_all_evals(model, tokenizer, "fisher_rao (Pierre default)")
    results["methods"]["fisher_rao"] = fr_results
    save()

    # M2 the new method
    print(f"\n--- preprocessing for {method.name} ---", flush=True)
    t_prep = time.time()
    if method.kind == "b_only":
        new_B = method.fn(B_lists, shared_A_dict, **method.fn_kwargs)
        prep_seconds = time.time() - t_prep
        reset_to_polar_path(model, modules, base_q_projs)
        install_polar_state(modules, shared_A_dict, new_B)
    else:  # fused_delta
        states_list = [adapter_states[n] for n in adapters_active]
        new_fused = method.fn(states_list, shared_A_dict, SCALE, **method.fn_kwargs)
        prep_seconds = time.time() - t_prep
        install_fused_delta(model, base_q_projs, new_fused)
    print(f"  preprocessing: {prep_seconds:.2f}s", flush=True)
    new_results = run_all_evals(model, tokenizer, f"{method.name} (under test)")
    new_results["preprocess_seconds"] = prep_seconds
    results["methods"][method.name] = new_results
    save()

    # M3 full-delta DARE upper bound
    fused = compose_full_delta_dare([adapter_states[n] for n in adapters_active], drop_rate=DROP_RATE, seed=SEED)
    install_fused_delta(model, base_q_projs, fused)
    fd_results = run_all_evals(model, tokenizer, "dare_full_delta (research upper bound)")
    results["methods"]["dare_full_delta"] = fd_results
    save()

    # KC evaluation
    fr_avg = fr_results["avg"]
    new_avg = new_results["avg"]
    fd_avg = fd_results["avg"]

    k1 = (new_avg - fr_avg) >= kc_thresholds.get("k1_min_delta_over_fisher_rao", 3.0)
    k2 = (fd_avg - new_avg) <= kc_thresholds.get("k2_max_delta_under_full_delta_dare", 4.0)
    k3 = prep_seconds <= kc_thresholds.get("k3_max_preprocess_seconds", float("inf"))
    k4_value = kc_thresholds.get("k4_value")
    k4_threshold = kc_thresholds.get("k4_threshold")
    k4 = bool(k4_value is not None and k4_threshold is not None
              and abs(float(k4_value) - float(k4_threshold)) <= kc_thresholds.get("k4_tolerance", 1.0))

    results["kill_criteria"] = {
        "K1_beats_fisher_rao": {
            "pass": bool(k1), "delta_pp": new_avg - fr_avg,
            "threshold_pp": kc_thresholds.get("k1_min_delta_over_fisher_rao", 3.0),
        },
        "K2_close_to_full_delta_dare": {
            "pass": bool(k2), "abs_delta_pp": fd_avg - new_avg,
            "threshold_pp": kc_thresholds.get("k2_max_delta_under_full_delta_dare", 4.0),
        },
        "K3_preprocess_within_budget": {
            "pass": bool(k3), "elapsed_s": prep_seconds,
            "threshold_s": kc_thresholds.get("k3_max_preprocess_seconds", None),
        },
        "K4_sanity": {
            "pass": bool(k4),
            "label": kc_thresholds.get("k4_label", "(no sanity check defined)"),
            "value": k4_value, "threshold": k4_threshold,
        },
    }

    # Verdict logic — K1 is the decision; K4 sanity gates inconclusive
    if not k4 and k4_value is not None:
        verdict = "INCONCLUSIVE"
        decision = "Sanity check failed; eval pipeline or method implementation suspect"
    elif k1 and k2:
        verdict = "SUPPORTED"
        decision = f"Adopt {method.name}: beats Fisher-Rao by {new_avg-fr_avg:+.1f}pp; within budget of full-delta DARE"
    elif k1:
        verdict = "SUPPORTED"
        decision = f"Adopt {method.name}: beats Fisher-Rao by {new_avg-fr_avg:+.1f}pp; gap to full-delta DARE is {fd_avg-new_avg:.1f}pp (worse than budget)"
    else:
        verdict = "KILLED"
        decision = f"Keep Fisher-Rao; {method.name} did not beat default by required margin"

    results["verdict"] = verdict
    results["decision"] = decision
    save()

    print(f"\n=== {verdict}: {decision} ===")
    print(f"  K1 beats Fisher-Rao  : {'PASS' if k1 else 'FAIL'}  Δ={new_avg-fr_avg:+.1f}pp")
    print(f"  K2 close to upper    : {'PASS' if k2 else 'FAIL'}  gap={fd_avg-new_avg:.1f}pp")
    print(f"  K3 preprocess budget : {'PASS' if k3 else 'FAIL'}  {prep_seconds:.1f}s")
    if k4_value is not None:
        print(f"  K4 sanity {kc_thresholds.get('k4_label','')}: {'PASS' if k4 else 'FAIL'}")
    print(f"\nResults: {out_path}")
    return results
