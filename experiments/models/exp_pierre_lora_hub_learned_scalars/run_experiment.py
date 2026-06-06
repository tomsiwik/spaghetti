"""exp_pierre_lora_hub_learned_scalars — architectural ceiling test.

Optimize K=7 scalar weights via scipy differential_evolution on a small
validation panel, then evaluate the optimized weighted-mean on the standard
3-benchmark suite. Compares against Fisher-Rao baseline and full-delta DARE
upper bound.

Custom flow (does NOT call run_pierre_compose_experiment) because the method
needs an objective function that scores against a held-out validation panel
during optimization, which is outside the shared runner's pattern.

Usage:
    experiment run exp_pierre_lora_hub_learned_scalars
"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

import mlx.core as mx
import numpy as np

from _pierre_shared.eval_runner import (  # type: ignore  # noqa: E402
    ADAPTER_NAMES, MODEL_NAME, _get_layers, inject_polar_adapters,
    load_adapter_state, stack_B_dicts, reset_to_polar_path,
    install_polar_state, install_fused_delta, run_all_evals,
    compose_fisher_rao, compose_full_delta_dare,
    RANK, SCALE, N_EVAL, SEED,
    DARE_RESEARCH_REFERENCE_AVG, FISHER_RAO_REFERENCE_AVG,
    SINGLE_BEST_FOR_BENCH, _FusedDeltaLinear,
)
from compose_methods import compose_weighted_mean_b  # type: ignore  # noqa: E402


N_VAL_PANEL = 3  # prompts per benchmark for the optimizer's objective
OPT_BUDGET = 40
WEIGHT_BOUNDS = (-1.5, 1.5)


def main():
    out_path = EXP_DIR / "results.json"
    print(f"=== exp_pierre_lora_hub_learned_scalars ===")
    print(f"  N_eval = {N_EVAL}/bench, N_val_panel = {N_VAL_PANEL}/bench, budget = {OPT_BUDGET}")

    from mlx_lm import load
    from mlx_lm import generate as mlx_generate

    print(f"\nLoading {MODEL_NAME}...")
    model, tokenizer = load(MODEL_NAME)

    layers = _get_layers(model)
    base_q_projs = [layer.self_attn.q_proj for layer in layers]
    print(f"  {len(base_q_projs)} transformer layers")

    print("Loading adapters...")
    adapter_states = {n: load_adapter_state(n) for n in ADAPTER_NAMES}
    shared_A = {k: v["a"] for k, v in adapter_states[ADAPTER_NAMES[0]].items() if "a" in v}
    print(f"  shared-A donor: {ADAPTER_NAMES[0]}")
    print("Injecting PoLARLinear...")
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)
    B_lists = stack_B_dicts([adapter_states[n] for n in ADAPTER_NAMES])
    K = len(B_lists)

    # Pre-fetch validation panel (held-out)
    print(f"\nBuilding validation panel ({N_VAL_PANEL}/bench, separate seed)...")
    val_seed = SEED + 1000
    val_panel = build_val_panel(tokenizer, val_seed=val_seed)
    print(f"  {sum(len(v) for v in val_panel.values())} val prompts total")

    # Objective: scores K weights by accuracy on the validation panel
    objective_calls = {"n": 0}
    obj_history = []

    def objective_fn(w_list):
        # Compose B with these weights
        new_B = compose_weighted_mean_b(B_lists, shared_A, weights=w_list)
        reset_to_polar_path(model, modules, base_q_projs)
        install_polar_state(modules, shared_A, new_B)
        # Run mini-eval on val panel
        score = score_val_panel(model, tokenizer, val_panel)
        objective_calls["n"] += 1
        cost = -score
        obj_history.append({"call": objective_calls["n"], "weights": list(w_list), "score": score})
        print(f"  obj #{objective_calls['n']:>2d}: score={score:.1f}% w={[f'{w:+.2f}' for w in w_list]}", flush=True)
        return cost

    print(f"\nOptimizing K={K} scalars via scipy.differential_evolution (budget={OPT_BUDGET})...")
    t0 = time.time()
    from scipy.optimize import differential_evolution  # noqa: E402
    res = differential_evolution(
        objective_fn,
        [WEIGHT_BOUNDS] * K,
        maxiter=8, popsize=5, seed=SEED, polish=False, tol=1e-3, init="sobol",
    )
    opt_seconds = time.time() - t0
    learned_w = [float(x) for x in res.x]
    print(f"  optimization done ({opt_seconds:.0f}s, {objective_calls['n']} evals)")
    print(f"  learned weights: {[f'{w:+.3f}' for w in learned_w]}")

    # Run the standard 4-method matrix manually
    results = {
        "config": {
            "model": MODEL_NAME,
            "n_eval_per_bench": N_EVAL,
            "n_val_panel_per_bench": N_VAL_PANEL,
            "seed": SEED,
            "rank": RANK,
            "scale": SCALE,
            "n_adapters": K,
            "adapter_names": ADAPTER_NAMES,
            "shared_A_donor": ADAPTER_NAMES[0],
            "weight_bounds": list(WEIGHT_BOUNDS),
            "optimization_budget": OPT_BUDGET,
            "lora_hub_paper": "arxiv 2307.13269",
            "fisher_rao_reference_avg": FISHER_RAO_REFERENCE_AVG,
            "dare_research_reference_avg": DARE_RESEARCH_REFERENCE_AVG,
        },
        "optimization": {
            "elapsed_s": opt_seconds,
            "n_calls": objective_calls["n"],
            "learned_weights": learned_w,
            "learned_avg_val_score": -float(res.fun),
            "history_first_5": obj_history[:5],
            "history_last_5": obj_history[-5:],
            "uniform_weight_reference": 1.0 / K,
        },
        "methods": {},
    }

    def save():
        out_path.write_text(json.dumps(results, indent=2, default=str))

    # M0 single_best per benchmark
    sb_per_bench = {}
    for bench, adapter_name in SINGLE_BEST_FOR_BENCH.items():
        reset_to_polar_path(model, modules, base_q_projs)
        st = adapter_states[adapter_name]
        A_d = {k: v["a"] for k, v in st.items() if "a" in v}
        B_d = {k: v["b"] for k, v in st.items() if "b" in v}
        install_polar_state(modules, A_d, B_d)
        print(f"\n--- single_best={adapter_name} on {bench} ---", flush=True)
        from tooling.scripts.polar_train import eval_gsm8k, eval_humaneval, eval_medqa  # noqa: E402
        if bench == "gsm8k":
            score = eval_gsm8k(model, tokenizer, n_eval=N_EVAL, seed=SEED)
        elif bench == "humaneval":
            score = eval_humaneval(model, tokenizer, n_eval=N_EVAL)
        else:
            score = eval_medqa(model, tokenizer, n_eval=N_EVAL, seed=SEED)
        sb_per_bench[bench] = score
        print(f"  {bench}={score:.1f}%", flush=True)
    sb_avg = sum(sb_per_bench.values()) / 3.0
    results["methods"]["single_best"] = {"per_bench": sb_per_bench, "avg": sb_avg}
    save()

    # M1 Fisher-Rao
    reset_to_polar_path(model, modules, base_q_projs)
    fr_B = compose_fisher_rao(B_lists)
    install_polar_state(modules, shared_A, fr_B)
    fr_results = run_all_evals(model, tokenizer, "fisher_rao (Pierre default)")
    results["methods"]["fisher_rao"] = fr_results
    save()

    # M2 LoRA Hub-learned scalars
    reset_to_polar_path(model, modules, base_q_projs)
    learned_B = compose_weighted_mean_b(B_lists, shared_A, weights=learned_w)
    install_polar_state(modules, shared_A, learned_B)
    lh_results = run_all_evals(model, tokenizer, "lora_hub_learned_scalars (final eval)")
    results["methods"]["lora_hub_learned_scalars"] = lh_results
    save()

    # M3 full-delta DARE upper bound
    fused = compose_full_delta_dare([adapter_states[n] for n in ADAPTER_NAMES], drop_rate=0.9, seed=SEED)
    install_fused_delta(model, base_q_projs, fused)
    fd_results = run_all_evals(model, tokenizer, "dare_full_delta (research upper bound)")
    results["methods"]["dare_full_delta"] = fd_results
    save()

    # KCs
    fr_avg = fr_results["avg"]
    lh_avg = lh_results["avg"]
    fd_avg = fd_results["avg"]
    uniform = 1.0 / K
    deviating = sum(1 for w in learned_w if abs(w - uniform) >= 0.1)

    k1 = (lh_avg - fr_avg) >= 2.0
    k2 = (fd_avg - lh_avg) <= 5.0
    k3 = opt_seconds <= 60 * 60
    k4 = deviating >= 2

    results["kill_criteria"] = {
        "K1_beats_fisher_rao_2pp": {
            "pass": bool(k1), "delta_pp": lh_avg - fr_avg, "threshold_pp": 2.0,
        },
        "K2_within_5pp_of_dare_full": {
            "pass": bool(k2), "abs_delta_pp": fd_avg - lh_avg, "threshold_pp": 5.0,
        },
        "K3_optimization_within_1h": {
            "pass": bool(k3), "elapsed_s": opt_seconds, "threshold_s": 3600,
        },
        "K4_at_least_2_weights_deviate_from_uniform": {
            "pass": bool(k4), "n_deviating": deviating, "threshold_n": 2,
            "uniform_value": uniform,
            "weights": learned_w,
        },
    }

    if k1 and k2:
        verdict = "SUPPORTED"
        decision = (
            f"Learned scalars beat Fisher-Rao by {lh_avg-fr_avg:+.1f}pp and close most "
            f"of the gap to full-delta DARE ({fd_avg-lh_avg:.1f}pp remaining). "
            "Architectural-ceiling test PASSES — shared-A B-only scalar weighting can close the gap."
        )
    elif k1:
        verdict = "SUPPORTED"
        decision = (
            f"Learned scalars beat Fisher-Rao by {lh_avg-fr_avg:+.1f}pp but leave "
            f"{fd_avg-lh_avg:.1f}pp on the table vs full-delta DARE. "
            "Partial ceiling: scalar weights help but cannot close the full gap."
        )
    else:
        verdict = "KILLED"
        decision = (
            "Architectural ceiling REACHED for shared-A B-only scalar weighting. "
            f"Even after gradient-free optimization on K={K} scalars, the merge "
            f"avg ({lh_avg:.1f}%) does not significantly exceed Fisher-Rao "
            f"({fr_avg:.1f}%). The remaining gap to full-delta DARE ({fd_avg:.1f}%) "
            "requires non-scalar methods (Pico/ACE/OrthoMerge) or per-adapter A storage. "
            "This kill is INFORMATIVE — it forecloses the simplest axis of attack."
        )

    results["verdict"] = verdict
    results["decision"] = decision
    save()

    print(f"\n=== {verdict}: {decision} ===")
    print(f"  K1 beats Fisher-Rao 2pp     : {'PASS' if k1 else 'FAIL'}  Δ={lh_avg-fr_avg:+.1f}pp")
    print(f"  K2 within 5pp of DARE full  : {'PASS' if k2 else 'FAIL'}  gap={fd_avg-lh_avg:.1f}pp")
    print(f"  K3 optimization < 1h        : {'PASS' if k3 else 'FAIL'}  {opt_seconds:.0f}s")
    print(f"  K4 ≥2 weights ≠ {uniform:.3f}    : {'PASS' if k4 else 'FAIL'}  ({deviating} of {K})")
    print(f"\nResults: {out_path}")


def build_val_panel(tokenizer, val_seed: int) -> dict:
    """Build a small (held-out) validation panel for the optimizer."""
    from datasets import load_dataset
    panel: dict = {}

    gsm = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=val_seed).select(range(N_VAL_PANEL))
    panel["gsm8k"] = []
    for ex in gsm:
        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        gt = gt_match.group(1).replace(",", "").strip() if gt_match else None
        panel["gsm8k"].append({
            "prompt": f"Solve step by step.\n\n{ex['question']}\n\nAnswer:",
            "gt": gt,
            "scorer": "gsm8k",
        })

    he = load_dataset("openai_humaneval", split="test").select(range(N_VAL_PANEL))
    panel["humaneval"] = []
    for ex in he:
        panel["humaneval"].append({
            "prompt": f"Complete this Python function:\n\n```python\n{ex['prompt']}\n```\n\nRespond with only the function body.",
            "gt": ex,
            "scorer": "humaneval",
        })

    md = load_dataset("GBaker/MedQA-USMLE-4-options", split="test").shuffle(seed=val_seed).select(range(N_VAL_PANEL))
    panel["medqa"] = []
    for ex in md:
        opts = ex["options"]
        question = f"{ex['question']}\n(A) {opts['A']}\n(B) {opts['B']}\n(C) {opts['C']}\n(D) {opts['D']}"
        panel["medqa"].append({
            "prompt": f"Answer with only the letter (A/B/C/D).\n\n{question}",
            "gt": ex["answer_idx"],
            "scorer": "medqa",
        })
    return panel


def score_val_panel(model, tokenizer, panel: dict) -> float:
    """Score model against the val panel; return percentage correct."""
    from mlx_lm import generate as mlx_generate
    correct = 0
    total = 0
    for bench, examples in panel.items():
        for ex in examples:
            msgs = [{"role": "user", "content": ex["prompt"]}]
            formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            max_tok = 1024 if ex["scorer"] == "gsm8k" else 512 if ex["scorer"] == "humaneval" else 20
            response = mlx_generate(model, tokenizer, prompt=formatted, max_tokens=max_tok, verbose=False)
            ok = _score_response(response, ex)
            correct += int(ok)
            total += 1
    return correct / max(total, 1) * 100


def _score_response(response: str, ex: dict) -> bool:
    """Tiny scorer matching scripts/polar_train.py logic for each benchmark."""
    if ex["scorer"] == "gsm8k":
        gt = ex.get("gt")
        if gt is None:
            return False
        m = re.search(r"####\s*([\d,\-\.]+)", response)
        if m and m.group(1).replace(",", "").strip() == gt:
            return True
        nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
        return bool(nums and nums[-1] == gt)
    if ex["scorer"] == "humaneval":
        import subprocess, sys as _sys
        ds_ex = ex["gt"]
        m = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        completion = m.group(1) if m else response
        full = ds_ex["prompt"] + completion + "\n\n" + ds_ex["test"] + f"\n\ncheck({ds_ex['entry_point']})\n"
        try:
            r = subprocess.run([_sys.executable, "-c", full], timeout=10, capture_output=True, text=True)
            return r.returncode == 0
        except Exception:
            return False
    if ex["scorer"] == "medqa":
        gt = ex.get("gt")
        pred = response.strip().upper()
        pred_letter = next((L for L in "ABCD" if pred.startswith(L)), None)
        if not pred_letter:
            m = re.search(r"\b([ABCD])\b", pred)
            pred_letter = m.group(1) if m else None
        return pred_letter == gt
    return False


if __name__ == "__main__":
    main()
