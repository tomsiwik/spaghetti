"""Stiefel post-hoc projection on existing B-matrices.

For each of 2 variants (strict, rescaled):
  - Per-adapter single eval after projection (3 native benchmarks)
  - Composition via 3 methods (simple-mean, Fisher-Rao, TIES-B)
  - Compare to standard-adapter Fisher-Rao (64.7) and TIES-B (71.3) baselines.

No training. Pure projection + eval. Runtime ~60 min for full matrix.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

import mlx.core as mx

from _pierre_shared.eval_runner import (  # type: ignore  # noqa: E402
    ADAPTER_NAMES, MODEL_NAME, RANK, SCALE, N_EVAL, SEED,
    _get_layers, inject_polar_adapters, load_adapter_state,
    stack_B_dicts, reset_to_polar_path, install_polar_state,
    SINGLE_BEST_FOR_BENCH, compose_fisher_rao,
    FISHER_RAO_REFERENCE_AVG,
)
from compose_methods import (  # type: ignore  # noqa: E402
    stiefel_project_b_dicts, compose_simple_mean,
)


TIES_B_BASELINE_AVG = 71.3
SINGLE_BEST_AVG_STANDARD = 62.0  # math=66, code=78, medical=42


def _ties_b_merge(tensors, weights, *, keep_frac=0.3):
    """TIES B-only merge (same as exp_pierre_ties_b_only)."""
    if len(tensors) == 1:
        return tensors[0]
    rank, d_out = tensors[0].shape
    D = rank * d_out
    T_flat = mx.stack([t.astype(mx.float32).reshape(-1) for t in tensors])
    w = mx.array([float(x) for x in weights], dtype=mx.float32)

    abs_T = mx.abs(T_flat)
    k = max(1, int(D * keep_frac))
    sorted_abs = mx.sort(abs_T, axis=1)
    threshold = sorted_abs[:, -k:-k + 1] if k < D else sorted_abs[:, :1]
    keep = abs_T >= threshold

    T_trim = T_flat * keep.astype(T_flat.dtype)
    weighted_sum = mx.sum(T_trim * w[:, None], axis=0)
    gamma = mx.sign(weighted_sum)
    agree = (mx.sign(T_trim) == gamma[None, :]) & (T_trim != 0)
    agree_f = agree.astype(T_flat.dtype)
    weighted_vals = T_trim * agree_f * w[:, None]
    weight_sum_per_cell = mx.sum(agree_f * w[:, None], axis=0)
    merged_flat = mx.sum(weighted_vals, axis=0) / mx.maximum(weight_sum_per_cell, 1e-8)

    merged = merged_flat.reshape(rank, d_out)
    orig_norms = mx.stack([mx.linalg.norm(t.astype(mx.float32).reshape(-1)) for t in tensors])
    mean_source_norm = mx.mean(orig_norms)
    mean_norm = mx.linalg.norm(merged.reshape(-1))
    mx.eval(mean_source_norm, mean_norm)
    if mean_norm.item() > 1e-8:
        merged = merged * (mean_source_norm / mean_norm)
    return merged.astype(mx.bfloat16)


def compose_ties_b(B_lists, A_dict=None, weights=None, keep_frac=0.3):
    if len(B_lists) == 1:
        return B_lists[0]
    if weights is None:
        weights = [1.0 / len(B_lists)] * len(B_lists)
    all_keys: set[str] = set()
    for ab in B_lists:
        all_keys.update(ab.keys())
    composed = {}
    for key in sorted(all_keys):
        tensors = [ab[key] for ab in B_lists if key in ab]
        ws = weights[: len(tensors)]
        composed[key] = _ties_b_merge(tensors, ws, keep_frac=keep_frac)
    return composed


def verify_stiefel(B_dicts):
    """Sanity check: stack per-key, compute B B^T diagonal & off-diagonal stats."""
    samples = []
    keys = sorted(B_dicts[0].keys())[:3]  # check first 3 layers
    for key in keys:
        Bs = [ab[key].astype(mx.float32) for ab in B_dicts if key in ab]
        B_all = mx.concatenate(Bs, axis=0)
        G = B_all @ B_all.T
        n = G.shape[0]
        diag = mx.array([G[i, i].item() for i in range(min(n, 10))])
        off_diag = mx.array([G[i, j].item() for i in range(min(n, 5)) for j in range(min(n, 5)) if i != j])
        samples.append({
            "key": key,
            "diag_mean": float(mx.mean(diag).item()),
            "diag_max_dev_from_1": float(mx.max(mx.abs(diag - 1.0)).item()),
            "offdiag_mean_abs": float(mx.mean(mx.abs(off_diag)).item()),
            "offdiag_max_abs": float(mx.max(mx.abs(off_diag)).item()),
        })
    return samples


def main():
    out_path = EXP_DIR / "results.json"
    print("=== exp_pierre_stiefel_b_postproj ===")
    print(f"  Strategy: joint-Stiefel projection on existing B-matrices, eval both variants")
    print(f"  N_eval = {N_EVAL}/bench")

    from mlx_lm import load
    print(f"\nLoading {MODEL_NAME}...")
    model, tokenizer = load(MODEL_NAME)

    layers = _get_layers(model)
    base_q_projs = [layer.self_attn.q_proj for layer in layers]
    print(f"  {len(base_q_projs)} transformer layers")

    print("\nLoading 7 adapter states...")
    adapter_states = {n: load_adapter_state(n) for n in ADAPTER_NAMES}
    shared_A_dict = {k: v["a"] for k, v in adapter_states[ADAPTER_NAMES[0]].items() if "a" in v}

    print("\nInjecting PoLARLinear modules...")
    modules = inject_polar_adapters(model, rank=RANK, scale=SCALE)

    # Build B-dicts in canonical order
    B_dicts_standard = stack_B_dicts([adapter_states[n] for n in ADAPTER_NAMES])

    results = {
        "config": {
            "model": MODEL_NAME,
            "n_eval_per_bench": N_EVAL,
            "rank": RANK, "scale": SCALE, "seed": SEED,
            "n_adapters": len(ADAPTER_NAMES),
            "adapter_names": ADAPTER_NAMES,
            "method_description": "Joint-Stiefel projection on B (no training, post-hoc QR)",
            "fisher_rao_baseline": FISHER_RAO_REFERENCE_AVG,
            "ties_b_baseline": TIES_B_BASELINE_AVG,
            "single_best_avg_standard": SINGLE_BEST_AVG_STANDARD,
        },
        "verifications": {},
        "single_adapter_evals": {},
        "compositions": {},
    }

    def save():
        out_path.write_text(json.dumps(results, indent=2, default=str))

    # ─── Build the two projected variants ────────────────────────────
    print("\n--- Projecting to Stiefel (strict) ---")
    t0 = time.time()
    B_dicts_strict = stiefel_project_b_dicts(B_dicts_standard, variant="strict")
    print(f"  strict projection: {time.time()-t0:.2f}s")
    results["verifications"]["strict_orthogonality"] = verify_stiefel(B_dicts_strict)

    print("--- Projecting to Stiefel (rescaled) ---")
    t0 = time.time()
    B_dicts_rescaled = stiefel_project_b_dicts(B_dicts_standard, variant="rescaled")
    print(f"  rescaled projection: {time.time()-t0:.2f}s")
    # Rescaled variant no longer strict Stiefel — log orthogonality drift
    results["verifications"]["rescaled_orthogonality"] = verify_stiefel(B_dicts_rescaled)
    save()

    # ─── Single-adapter eval — standard vs strict vs rescaled ────────
    from tooling.scripts.polar_train import eval_gsm8k, eval_humaneval, eval_medqa  # noqa: E402

    def eval_native(B_dict_list, label_prefix):
        """For each adapter, eval on its native benchmark only."""
        per_adapter_native = {}
        for bench, adapter_name in SINGLE_BEST_FOR_BENCH.items():
            idx = ADAPTER_NAMES.index(adapter_name)
            reset_to_polar_path(model, modules, base_q_projs)
            # Use adapter-native A (not shared A) for single-adapter eval
            st = adapter_states[adapter_name]
            A_d = {k: v["a"] for k, v in st.items() if "a" in v}
            install_polar_state(modules, A_d, B_dict_list[idx])
            t0 = time.time()
            if bench == "gsm8k":
                score = eval_gsm8k(model, tokenizer, n_eval=N_EVAL, seed=SEED)
            elif bench == "humaneval":
                score = eval_humaneval(model, tokenizer, n_eval=N_EVAL)
            else:
                score = eval_medqa(model, tokenizer, n_eval=N_EVAL, seed=SEED)
            per_adapter_native[bench] = score
            print(f"  {label_prefix} {bench}={score:.1f}% ({time.time()-t0:.0f}s)")
            results["single_adapter_evals"][label_prefix] = per_adapter_native
            save()
        per_adapter_native["avg"] = sum(per_adapter_native[b] for b in ["gsm8k", "humaneval", "medqa"]) / 3.0
        return per_adapter_native

    print("\n--- Single-adapter eval: standard ---")
    standard_native = eval_native(B_dicts_standard, "standard")
    print(f"  standard avg: {standard_native['avg']:.1f}%")

    print("\n--- Single-adapter eval: strict Stiefel ---")
    strict_native = eval_native(B_dicts_strict, "strict_stiefel")
    print(f"  strict Stiefel avg: {strict_native['avg']:.1f}%")

    print("\n--- Single-adapter eval: rescaled Stiefel ---")
    rescaled_native = eval_native(B_dicts_rescaled, "rescaled_stiefel")
    print(f"  rescaled Stiefel avg: {rescaled_native['avg']:.1f}%")

    # ─── Composition eval — 3 methods × 2 variants ──────────────────
    from _pierre_shared.eval_runner import run_all_evals  # noqa: E402

    composition_configs = [
        ("strict_simple_mean",   B_dicts_strict,   compose_simple_mean),
        ("strict_fisher_rao",    B_dicts_strict,   compose_fisher_rao),
        ("strict_ties_b",        B_dicts_strict,   compose_ties_b),
        ("rescaled_simple_mean", B_dicts_rescaled, compose_simple_mean),
        ("rescaled_fisher_rao",  B_dicts_rescaled, compose_fisher_rao),
        ("rescaled_ties_b",      B_dicts_rescaled, compose_ties_b),
    ]

    for label, B_list, compose_fn in composition_configs:
        print(f"\n--- Composition: {label} ---")
        composed_B = compose_fn(B_list, shared_A_dict)
        reset_to_polar_path(model, modules, base_q_projs)
        install_polar_state(modules, shared_A_dict, composed_B)
        scores = run_all_evals(model, tokenizer, label)
        results["compositions"][label] = scores
        save()

    # ─── KCs ─────────────────────────────────────────────────────────
    k1 = (strict_native["avg"] - standard_native["avg"]) >= -5.0
    k2 = (rescaled_native["avg"] - standard_native["avg"]) >= -2.0

    strict_simple_avg = results["compositions"]["strict_simple_mean"]["avg"]
    rescaled_simple_avg = results["compositions"]["rescaled_simple_mean"]["avg"]
    best_simple = max(strict_simple_avg, rescaled_simple_avg)
    k3 = best_simple >= FISHER_RAO_REFERENCE_AVG

    all_compose_avgs = {label: results["compositions"][label]["avg"]
                        for label, _, _ in composition_configs}
    best_compose_label = max(all_compose_avgs, key=all_compose_avgs.get)
    best_compose_avg = all_compose_avgs[best_compose_label]
    k4 = best_compose_avg >= TIES_B_BASELINE_AVG - 1.0

    results["kill_criteria"] = {
        "K1_strict_expressivity_within_5pp": {
            "pass": bool(k1),
            "strict_avg": strict_native["avg"],
            "standard_avg": standard_native["avg"],
            "delta_pp": strict_native["avg"] - standard_native["avg"],
        },
        "K2_rescaled_expressivity_within_2pp": {
            "pass": bool(k2),
            "rescaled_avg": rescaled_native["avg"],
            "standard_avg": standard_native["avg"],
            "delta_pp": rescaled_native["avg"] - standard_native["avg"],
        },
        "K3_simple_mean_beats_fisher_rao_baseline": {
            "pass": bool(k3),
            "best_simple_avg": best_simple,
            "fisher_rao_baseline": FISHER_RAO_REFERENCE_AVG,
            "delta_pp": best_simple - FISHER_RAO_REFERENCE_AVG,
        },
        "K4_best_compose_reaches_ties_floor": {
            "pass": bool(k4),
            "best_compose_label": best_compose_label,
            "best_compose_avg": best_compose_avg,
            "ties_b_baseline": TIES_B_BASELINE_AVG,
            "delta_pp": best_compose_avg - TIES_B_BASELINE_AVG,
        },
    }

    # Verdict
    if k1 and k2 and k3 and k4:
        verdict = "SUPPORTED"
        decision = (
            f"Stiefel projection preserves expressivity (strict Δ={strict_native['avg']-standard_native['avg']:+.1f}pp, "
            f"rescaled Δ={rescaled_native['avg']-standard_native['avg']:+.1f}pp). "
            f"Simple-mean composition reaches Fisher-Rao floor and best composition ({best_compose_label}) "
            f"hits {best_compose_avg:.1f}% (TIES-B baseline 71.3%). "
            "Stiefel-aware training next."
        )
    elif (k2 and k3) or k4:
        verdict = "PARTIAL"
        decision = (
            f"Stiefel-rescaled preserves expressivity but strict variant loses too much. "
            f"Composition reaches {best_compose_avg:.1f}% via {best_compose_label}. "
            "Proceed with rescaled variant; spec Stiefel-aware training for full gain."
        )
    elif not k2:
        verdict = "KILLED"
        decision = (
            f"Even rescaled Stiefel loses too much expressivity (Δ={rescaled_native['avg']-standard_native['avg']:+.1f}pp < -2pp). "
            "Existing training drifts too far from Stiefel manifold for post-hoc projection. "
            "Must train from scratch on Stiefel (sibling experiment exp_pierre_stiefel_b_train_single)."
        )
    else:
        verdict = "INCONCLUSIVE"
        decision = "Mixed signals; investigate before claiming."

    results["verdict"] = verdict
    results["decision"] = decision
    save()

    print(f"\n=== {verdict} ===")
    print(f"  K1 strict expressivity      : {'PASS' if k1 else 'FAIL'}  Δ={strict_native['avg']-standard_native['avg']:+.1f}pp")
    print(f"  K2 rescaled expressivity    : {'PASS' if k2 else 'FAIL'}  Δ={rescaled_native['avg']-standard_native['avg']:+.1f}pp")
    print(f"  K3 simple mean ≥ Fisher-Rao : {'PASS' if k3 else 'FAIL'}  best={best_simple:.1f}%")
    print(f"  K4 best compose ≥ TIES-B-1  : {'PASS' if k4 else 'FAIL'}  best={best_compose_avg:.1f}% via {best_compose_label}")
    print(f"\n  Decision: {decision}")


if __name__ == "__main__":
    main()
