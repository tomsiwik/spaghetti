"""
exp_followup_spectral_gap_measurement

Measure the per-layer spectral gap of attention projection weights for
BitNet-b1.58-2B-4T (ternary) and Gemma-4-E4B-4bit. Replaces the unmeasured
sqrt(30) placeholder in pro_composition_mmlu/MATH.md.

Pure measurement: load -> dequantize -> SVD -> aggregate -> dump JSON.
No training, no eval.
"""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from statistics import median

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear

HERE = Path(__file__).parent
RESULTS_PATH = HERE / "results.json"

BITNET_ID = "microsoft/BitNet-b1.58-2B-4T"
GEMMA_ID = "mlx-community/gemma-4-e4b-it-4bit"

ADAPTER_RANK_K = 16  # Pierre Pro setting per F#320
ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
SQRT_30_PLACEHOLDER = math.sqrt(30.0)  # ~5.477


# ---------------------------------------------------------------------------
# Ternary unpacking — borrowed from bitnet_spectral_surgery/run_experiment.py
# (same structure, BitNet packs 4 ternaries per uint8 as (x & 3) - 1).
# ---------------------------------------------------------------------------
def unpack_ternary(packed: mx.array, out_features: int,
                   weight_scale: mx.array, invert_scale: bool) -> mx.array:
    w0 = (packed & 3).astype(mx.float32) - 1
    w1 = ((packed >> 2) & 3).astype(mx.float32) - 1
    w2 = ((packed >> 4) & 3).astype(mx.float32) - 1
    w3 = ((packed >> 6) & 3).astype(mx.float32) - 1
    unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
    scale = weight_scale.astype(mx.float32)
    if invert_scale:
        unpacked = unpacked / scale
    else:
        unpacked = unpacked * scale
    return unpacked


# ---------------------------------------------------------------------------
# Per-matrix metric computation
# ---------------------------------------------------------------------------
def matrix_metrics(W: mx.array, k: int = ADAPTER_RANK_K) -> dict:
    """Compute spectral-gap metrics for a single weight matrix.

    W must be a 2-D mx.array. Returns floats (eagerly evaluated).
    """
    # SVD is CPU-only in mlx 0.31.1 (confirmed). Cast to f32 for numerical
    # stability. SVD returns (U, S, Vt) where S is sorted descending.
    W32 = W.astype(mx.float32)
    _, S, _ = mx.linalg.svd(W32, stream=mx.cpu)
    mx.eval(S)

    S_np = [float(x) for x in S.tolist()]
    d = len(S_np)
    sigma_1 = S_np[0]

    # Defensive: ensure descending
    monotone = all(S_np[i] >= S_np[i + 1] - 1e-6 for i in range(d - 1))
    all_finite = all(math.isfinite(x) and x >= -1e-8 for x in S_np)

    sigma_k = S_np[k - 1] if d >= k else float("nan")
    sigma_k1 = S_np[k] if d >= k + 1 else float("nan")

    abs_gap_k = sigma_k - sigma_k1 if d >= k + 1 else float("nan")
    rel_gap_k = abs_gap_k / sigma_1 if (d >= k + 1 and sigma_1 > 0) else float("nan")
    ratio_k = sigma_k / sigma_k1 if (d >= k + 1 and sigma_k1 > 0) else float("nan")

    sumsq = sum(x * x for x in S_np)
    sumabs = sum(abs(x) for x in S_np)
    stable_rank = sumsq / (sigma_1 ** 2) if sigma_1 > 0 else float("nan")
    eff_rank = (sumabs ** 2) / sumsq if sumsq > 0 else float("nan")

    return {
        "shape": list(W.shape),
        "min_dim": d,
        "sigma_1": sigma_1,
        "sigma_k": sigma_k,
        "sigma_k1": sigma_k1,
        "abs_gap_k": abs_gap_k,
        "rel_gap_k": rel_gap_k,
        "ratio_k": ratio_k,
        "stable_rank": stable_rank,
        "eff_rank": eff_rank,
        "monotone": monotone,
        "all_finite": all_finite,
    }


# ---------------------------------------------------------------------------
# BitNet weight extraction
# ---------------------------------------------------------------------------
def measure_bitnet(model) -> list[dict]:
    rows: list[dict] = []
    layers = model.layers  # BitNet 2B root has .layers
    for i, layer in enumerate(layers):
        for proj in ATTN_PROJECTIONS:
            mod = getattr(layer.self_attn, proj)
            assert isinstance(mod, BitLinear), f"unexpected {type(mod)}"
            t0 = time.time()
            W = unpack_ternary(
                mod.weight, mod.out_features,
                mod.weight_scale, mod.invert_weight_scales,
            )
            mx.eval(W)
            try:
                m = matrix_metrics(W)
                ok = True
                err = None
            except Exception as e:  # pragma: no cover - diagnostic path
                m = {"shape": [mod.out_features, mod.in_features]}
                ok = False
                err = repr(e)
            m.update({
                "model": "bitnet-2b-4t",
                "layer": i, "proj": proj,
                "wall_s": time.time() - t0,
                "svd_ok": ok, "err": err,
            })
            rows.append(m)
            del W
            mx.clear_cache()
            gc.collect()
        print(f"  bitnet L{i:02d} done  ({len(rows)} mats)")
    return rows


# ---------------------------------------------------------------------------
# Gemma 4 E4B weight extraction
# ---------------------------------------------------------------------------
def measure_gemma(model) -> list[dict]:
    rows: list[dict] = []
    layers = model.language_model.model.layers
    for i, layer in enumerate(layers):
        for proj in ATTN_PROJECTIONS:
            mod = getattr(layer.self_attn, proj)
            # Must be QuantizedLinear; dequantize with its own group_size/bits
            assert hasattr(mod, "scales"), f"unexpected {type(mod)}"
            t0 = time.time()
            W = mx.dequantize(
                mod.weight, mod.scales, mod.biases,
                group_size=mod.group_size, bits=mod.bits,
            )
            mx.eval(W)
            try:
                m = matrix_metrics(W)
                ok = True
                err = None
            except Exception as e:  # pragma: no cover
                m = {"shape": list(W.shape)}
                ok = False
                err = repr(e)
            m.update({
                "model": "gemma-4-e4b-4bit",
                "layer": i, "proj": proj,
                "wall_s": time.time() - t0,
                "svd_ok": ok, "err": err,
            })
            rows.append(m)
            del W
            mx.clear_cache()
            gc.collect()
        print(f"  gemma L{i:02d} done  ({len(rows)} mats)")
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _finite(xs):
    return [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def aggregate(rows: list[dict]) -> dict:
    out: dict = {}
    models = sorted({r["model"] for r in rows})
    for name in models:
        model_rows = [r for r in rows if r["model"] == name and r.get("svd_ok")]
        total = sum(1 for r in rows if r["model"] == name)
        ok = len(model_rows)
        summary: dict = {
            "n_total": total,
            "n_ok": ok,
            "svd_success_frac": ok / total if total else 0.0,
            "per_projection": {},
            "overall": {},
            "monotone_all": all(r.get("monotone", False) for r in model_rows),
            "finite_all": all(r.get("all_finite", False) for r in model_rows),
        }
        for proj in ATTN_PROJECTIONS + ("all",):
            if proj == "all":
                subset = model_rows
            else:
                subset = [r for r in model_rows if r["proj"] == proj]
            rel_gaps = _finite(r["rel_gap_k"] for r in subset)
            ratios = _finite(r["ratio_k"] for r in subset)
            abs_gaps = _finite(r["abs_gap_k"] for r in subset)
            stables = _finite(r["stable_rank"] for r in subset)
            effs = _finite(r["eff_rank"] for r in subset)
            sigma1s = _finite(r["sigma_1"] for r in subset)
            entry = {
                "n": len(subset),
                "rel_gap_k": {
                    "median": median(rel_gaps) if rel_gaps else float("nan"),
                    "p25": percentile(rel_gaps, 0.25),
                    "p75": percentile(rel_gaps, 0.75),
                    "min": min(rel_gaps) if rel_gaps else float("nan"),
                    "max": max(rel_gaps) if rel_gaps else float("nan"),
                },
                "ratio_k": {
                    "median": median(ratios) if ratios else float("nan"),
                    "p25": percentile(ratios, 0.25),
                    "p75": percentile(ratios, 0.75),
                },
                "abs_gap_k_median": median(abs_gaps) if abs_gaps else float("nan"),
                "stable_rank_median": median(stables) if stables else float("nan"),
                "eff_rank_median": median(effs) if effs else float("nan"),
                "sigma_1_median": median(sigma1s) if sigma1s else float("nan"),
            }
            summary["per_projection"][proj] = entry
        summary["overall"] = summary["per_projection"]["all"]
        out[name] = summary
    return out


def compute_cross_model_ratio(summary: dict) -> dict:
    g_bit = summary.get("bitnet-2b-4t", {}).get("overall", {}).get(
        "rel_gap_k", {}).get("median", float("nan"))
    g_gem = summary.get("gemma-4-e4b-4bit", {}).get("overall", {}).get(
        "rel_gap_k", {}).get("median", float("nan"))
    if not (isinstance(g_bit, float) and isinstance(g_gem, float)
            and math.isfinite(g_bit) and math.isfinite(g_gem) and g_bit > 0):
        R = float("nan")
    else:
        R = g_gem / g_bit
    return {
        "median_rel_gap_k_bitnet": g_bit,
        "median_rel_gap_k_gemma": g_gem,
        "R": R,
        "log10_R": math.log10(R) if (isinstance(R, float) and R > 0) else float("nan"),
        "placeholder_sqrt30": SQRT_30_PLACEHOLDER,
        "ratio_to_placeholder": (R / SQRT_30_PLACEHOLDER
                                 if (isinstance(R, float) and R > 0) else float("nan")),
    }


# ---------------------------------------------------------------------------
# Kill-criteria evaluation
# ---------------------------------------------------------------------------
def evaluate_kcs(summary: dict, cross: dict) -> dict:
    # K1 structural: >=95% SVD success, finite + monotone, in BOTH models
    ok = []
    for name, s in summary.items():
        ok.append(
            s["svd_success_frac"] >= 0.95
            and s["monotone_all"]
            and s["finite_all"]
        )
    k1_pass = all(ok) and len(ok) == 2

    # K2 target: R reported finite and positive
    R = cross.get("R", float("nan"))
    k2_pass = isinstance(R, float) and math.isfinite(R) and R > 0

    verdict = "SUPPORTED" if (k1_pass and k2_pass) else "KILLED"
    return {
        "K1_structural_svd_ok": k1_pass,
        "K2_target_ratio_reported": k2_pass,
        "all_pass": k1_pass and k2_pass,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    overall_t0 = time.time()
    all_rows: list[dict] = []

    # Phase 1: BitNet
    print("=== Phase 1: loading BitNet-b1.58-2B-4T ===")
    model_bit, _ = load(BITNET_ID)
    print("=== Phase 1: measuring ternary spectral gaps ===")
    all_rows.extend(measure_bitnet(model_bit))
    del model_bit
    mx.clear_cache()
    gc.collect()

    # Phase 2: Gemma 4 E4B 4-bit
    print("=== Phase 2: loading Gemma-4-E4B-4bit ===")
    model_gem, _ = load(GEMMA_ID)
    print("=== Phase 2: measuring 4-bit spectral gaps ===")
    all_rows.extend(measure_gemma(model_gem))
    del model_gem
    mx.clear_cache()
    gc.collect()

    # Phase 3: aggregate
    print("=== Phase 3: aggregate + KC evaluation ===")
    summary = aggregate(all_rows)
    cross = compute_cross_model_ratio(summary)
    kcs = evaluate_kcs(summary, cross)

    wall = time.time() - overall_t0
    results = {
        "experiment": "exp_followup_spectral_gap_measurement",
        "models": {"bitnet": BITNET_ID, "gemma": GEMMA_ID},
        "adapter_rank_k": ADAPTER_RANK_K,
        "is_smoke": False,
        "wall_s": wall,
        "per_model_summary": summary,
        "cross_model": cross,
        "verdict": kcs["verdict"],
        "all_pass": kcs["all_pass"],
        "kill_criteria": {
            "K1": {"passed": kcs["K1_structural_svd_ok"],
                   "description": "SVD >=95% success + sigmas finite/monotone in both models"},
            "K2": {"passed": kcs["K2_target_ratio_reported"],
                   "description": "R = median(g_k Gemma4) / median(g_k BitNet) finite and positive"},
        },
        "raw_rows": all_rows,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"wrote {RESULTS_PATH} | wall={wall:.1f}s | verdict={kcs['verdict']}")
    print(f"  median g_k BitNet  = {cross['median_rel_gap_k_bitnet']:.6g}")
    print(f"  median g_k Gemma4  = {cross['median_rel_gap_k_gemma']:.6g}")
    print(f"  R (Gemma/BitNet)   = {cross['R']:.6g}")
    print(f"  placeholder sqrt30 = {cross['placeholder_sqrt30']:.6g}")
    print(f"  R / placeholder    = {cross['ratio_to_placeholder']:.6g}")


if __name__ == "__main__":
    main()
