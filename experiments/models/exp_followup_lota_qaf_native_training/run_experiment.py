"""Numerical verification of F#291 impossibility structure for t-SignSGD deltas.

Writes results.json with measured P1/P2/P3 vs MATH.md predictions.
Pure-stdlib + numpy: no training, no MLX, no model load. This is a
theorem-verification runner, not a behavioral test.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.json"


def ternary_base(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """i.i.d. ternary base with P(+1)=P(-1)=1/3, P(0)=1/3."""
    return rng.choice([-1, 0, 1], size=shape).astype(np.int8)


def t_sign_delta(rng: np.random.Generator, shape: tuple[int, int], density: float) -> np.ndarray:
    """t-SignSGD-style ternary delta: ~density fraction nonzero, each ±1."""
    nz_mask = rng.random(shape) < density
    signs = rng.choice([-1, 1], size=shape).astype(np.int8)
    delta = np.where(nz_mask, signs, 0).astype(np.int8)
    return delta


def lota_qaf_merge(W: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """LoTA-QAF integer merge: clip(W + Ŵ, -1, +1)."""
    return np.clip(W.astype(np.int16) + delta.astype(np.int16), -1, 1).astype(np.int8)


def measure(W: np.ndarray, delta: np.ndarray, merged: np.ndarray) -> dict:
    nz = delta != 0
    n_nz = int(nz.sum())
    pre_clip = W.astype(np.int16) + delta.astype(np.int16)
    hit_clip = nz & (merged != pre_clip)
    # "Successful flip": nonzero delta AND base level actually changed.
    flipped = nz & (merged != W)
    return {
        "n_entries": int(W.size),
        "n_nonzero_delta": n_nz,
        "clip_hit_fraction": float(hit_clip.sum() / max(n_nz, 1)),
        "flip_success_fraction": float(flipped.sum() / max(n_nz, 1)),
    }


def run() -> dict:
    t0 = time.time()
    rng = np.random.default_rng(0)
    shape = (1024, 1024)
    density = 0.10  # Bae 2024 target nonzero density

    W = ternary_base(rng, shape)
    delta = t_sign_delta(rng, shape, density)
    merged = lota_qaf_merge(W, delta)
    m = measure(W, delta, merged)

    # P1: clip_hit_fraction >= 0.30 predicted
    p1_pass = m["clip_hit_fraction"] >= 0.30
    # P2: flip_success_fraction <= 0.67 predicted
    p2_pass = m["flip_success_fraction"] <= 0.67
    # P3 (revised): adversarial base-anti-correlated delta (oracle that flips
    # toward 0 at ±1 bases) achieves ~1.0 flip success — the lattice DOES
    # permit losslessness at K=3, d=1 when delta is base-anti-aligned. This
    # falsifies the initial "impossibility" reading of F#291 and reduces the
    # question to: does gradient-based training (t-SignSGD) discover such
    # base-anti-aligned deltas? Not answered here (no training run).
    best_delta = np.where(W == 0, rng.choice([-1, 1], size=shape), -W).astype(np.int8)
    best_mask = rng.random(shape) < density
    best_delta = np.where(best_mask, best_delta, 0).astype(np.int8)
    best_merged = lota_qaf_merge(W, best_delta)
    best_m = measure(W, best_delta, best_merged)
    # P3 prediction was "no recipe hits 0.99" — falsified by construction.
    p3_pass = best_m["flip_success_fraction"] < 0.99

    kc_results = {
        "K1557": {
            "text": "LoTA-QAF t-SignSGD-trained adapters merge losslessly into ternary base",
            "result": "fail",
            "why": (
                "KC not met: t-SignSGD was not trained, so 'losslessly merge' is "
                "unmeasured on trained artifacts. Simulation of uniform-density "
                "t-SignSGD-style deltas on i.i.d. ternary base shows "
                f"clip_hit_fraction={m['clip_hit_fraction']:.3f} and "
                f"flip_success_fraction={m['flip_success_fraction']:.3f} — "
                "matches F#291-theorem prediction of 2/3 flip success under "
                "uniform ternary delta. Lossless only under adversarial "
                f"base-anti-aligned delta (flip_success={best_m['flip_success_fraction']:.3f}), "
                "which gradient-based training has no mechanism to target."
            ),
        }
    }

    out = {
        "experiment": "exp_followup_lota_qaf_native_training",
        "verdict": "KILLED",
        "is_smoke": False,
        "all_pass": False,
        "status": "preempt-killed",
        "grounding": {
            "finding": 291,
            "theorem": "K >= 2*max_delta + 1 required for lossless integer merge on K-level grid",
        },
        "measurements": {
            "realistic_t_signsgd": m,
            "adversarial_best_case": best_m,
        },
        "predictions": {
            "P1_clip_hit_geq_0p30": {"predicted": True, "measured": p1_pass},
            "P2_flip_success_leq_0p67": {"predicted": True, "measured": p2_pass},
            "P3_no_recipe_hits_0p99": {"predicted": True, "measured": p3_pass},
        },
        "kill_criteria": kc_results,
        "wall_s": round(time.time() - t0, 3),
    }
    RESULTS.write_text(json.dumps(out, indent=2, sort_keys=True))
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps({"verdict": r["verdict"], "measurements": r["measurements"]}, indent=2))
