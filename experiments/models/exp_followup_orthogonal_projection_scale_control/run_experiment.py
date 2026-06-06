"""
exp_followup_orthogonal_projection_scale_control

Theoretical-refutation probe — no training. Verifies parent
(orthogonal_adapter_training) closure theorems C1+C2+C3 against its own
`results.json` and `PAPER.md`, then derives KC #1573 FAIL via the
routing table pre-registered in MATH.md §E.

No MLX code is required — the refutation is theorem-level, not
data-level: the relevant empirical fact (spectral gap of the frozen
ternary base) is scale-invariant.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent  # experiments/models/<name>/ -> repo root
PARENT_DIR = REPO / "micro" / "models" / "orthogonal_adapter_training"
PARENT_RESULTS = PARENT_DIR / "results.json"
PARENT_PAPER = PARENT_DIR / "PAPER.md"


def _load_parent() -> tuple[dict, str]:
    with PARENT_RESULTS.open() as f:
        results = json.load(f)
    paper = PARENT_PAPER.read_text()
    return results, paper


def _extract_spectral_gaps(paper: str) -> list[tuple[str, float]]:
    # Extract "Layer 0 q_proj: σ_15/σ_16 = 1.005" style lines.
    # Uses both Greek and '_k/_{k+1}' forms in case paper uses variants.
    pattern = re.compile(
        r"Layer\s+\d+\s+\w+_proj:\s*[^=]*=\s*([0-9]+\.[0-9]+)"
    )
    return [("raw-match", float(m.group(1))) for m in pattern.finditer(paper)]


def _check_precondition_P1(results: dict) -> dict:
    """P1: Parent k16 summary shows K1_PASS=False and K3_PASS=False."""
    s = results.get("k16_summary", {})
    return {
        "mmlu_math": s.get("mmlu_math"),
        "mmlu_math_degradation_pp": s.get("mmlu_math_degradation_pp"),
        "gsm8k": s.get("gsm8k"),
        "gsm8k_gain_pp": s.get("gsm8k_gain_pp"),
        "indist_math_ratio": s.get("indist_math_ratio"),
        "K1_PASS": s.get("K1_PASS"),
        "K2_PASS": s.get("K2_PASS"),
        "K3_PASS": s.get("K3_PASS"),
        "precondition_holds": (
            s.get("K1_PASS") is False and s.get("K3_PASS") is False
        ),
    }


def _check_precondition_P2(results: dict) -> dict:
    """P2: rho_k effectively zero (orthogonal projection operates as specified)."""
    rho = results.get("k16_rho", {})
    orth = [v.get("mean_rho") for k, v in rho.items() if k.startswith("orth_")]
    baseline = [v.get("mean_rho") for k, v in rho.items() if k.startswith("baseline_")]
    max_orth = max(orth) if orth else None
    mean_baseline = sum(baseline) / len(baseline) if baseline else None
    return {
        "max_orth_rho": max_orth,
        "mean_baseline_rho": mean_baseline,
        "rho_reduction_ratio": (
            1.0 - (max_orth / mean_baseline) if max_orth and mean_baseline else None
        ),
        "precondition_holds": bool(max_orth is not None and max_orth < 1e-4),
    }


def _check_precondition_P3(paper: str) -> dict:
    """P3: Parent PAPER.md reports flat spectral gap (≤ 1.05)."""
    gaps = _extract_spectral_gaps(paper)
    ratios = [g for _, g in gaps]
    max_gap = max(ratios) if ratios else None
    return {
        "gaps_found": gaps,
        "max_gap": max_gap,
        "precondition_holds": bool(max_gap is not None and max_gap <= 1.05),
    }


def _derive_pareto_front(results: dict) -> list[dict]:
    """MATH.md §D Thm F1 Pareto table. Linear scaling from parent's s=20 data."""
    s = results["k16_summary"]
    k1_base_pp = s["mmlu_math_degradation_pp"]  # -20 at s=20 (sign convention: positive = pp degraded)
    k2_base_pp = s["gsm8k_gain_pp"]              # +14 at s=20
    k3_base_ratio_loss = 1.0 - s["indist_math_ratio"]  # 0.5 loss at s=20

    rows = []
    for s_new in (4, 6, 8, 10):
        factor = s_new / 20.0
        k1_est = k1_base_pp * factor        # pp degraded
        k2_est = k2_base_pp * factor        # pp gain
        # K3: capacity interference (Thm C3) dominates at 80%. Linear floor at s→0
        # is not 100% — we model the residual 20% direction-interference as linear
        # and the 80% capacity-interference as rank-level (scale-INdependent within rank=16).
        # Capacity-floor loss ≈ 0.8 * 0.5 = 0.4; direction-loss scales linearly.
        k3_dir_loss = 0.2 * k3_base_ratio_loss * factor
        k3_cap_loss = 0.8 * k3_base_ratio_loss  # scale-independent
        k3_ratio = 1.0 - (k3_dir_loss + k3_cap_loss)

        # KC thresholds from parent MATH.md §D:
        # K1: degradation ≤ 15pp  -> pass if k1_est ≤ 15
        # K2: gain ≥ +3pp          -> pass if k2_est ≥ 3
        # K3: ratio ≥ 0.90        -> pass if k3_ratio ≥ 0.90
        k1_pass = k1_est <= 15.0
        k2_pass = k2_est >= 3.0
        k3_pass = k3_ratio >= 0.90
        c_all = k1_pass and k2_pass and k3_pass

        rows.append({
            "scale": s_new,
            "factor": factor,
            "k1_est_pp": round(k1_est, 2),
            "k2_est_pp": round(k2_est, 2),
            "k3_est_ratio": round(k3_ratio, 3),
            "k1_pass": k1_pass,
            "k2_pass": k2_pass,
            "k3_pass": k3_pass,
            "c_all": c_all,
        })
    return rows


def main() -> None:
    t0 = time.time()
    results, paper = _load_parent()

    p1 = _check_precondition_P1(results)
    p2 = _check_precondition_P2(results)
    p3 = _check_precondition_P3(paper)

    pareto = _derive_pareto_front(results)
    any_scale_passes_all = any(row["c_all"] for row in pareto)

    # KC #1573 routing (pre-registered in MATH.md §E):
    # - R-struct: P1 ∧ P2 ∧ P3 → C1 applies → KC FAILs
    # - R-pareto: pareto front has no row with c_all=True → KC FAILs
    r_struct = (p1["precondition_holds"] and p2["precondition_holds"] and p3["precondition_holds"])
    r_pareto = not any_scale_passes_all
    k1573_fail = r_struct or r_pareto
    k1573_pass = not k1573_fail

    evidence = {
        "P1_parent_summary": p1,
        "P2_rho_reduction":  p2,
        "P3_spectral_gap":   p3,
        "F1_pareto_front":   pareto,
        "routing": {
            "R_struct_applies": r_struct,
            "R_pareto_applies": r_pareto,
            "any_scale_passes_all": any_scale_passes_all,
        },
    }

    verdict = "KILLED" if k1573_fail else "SUPPORTED"
    out = {
        "experiment": "exp_followup_orthogonal_projection_scale_control",
        "type": "theoretical_refutation_probe",
        "parent": "orthogonal_adapter_training",
        "no_training_required": True,
        "mlx_code_used": False,
        "elapsed_s": round(time.time() - t0, 3),
        "evidence": evidence,
        "kill_criteria": {
            "1573": {
                "text": (
                    "At scales {4,6,8,10} with autograd projection, "
                    "orth-projection claim holds independent of scale"
                ),
                "pass": k1573_pass,
                "route": "R-struct" if r_struct else ("R-pareto" if r_pareto else "none"),
            }
        },
        "verdict": verdict,
        "all_pass": k1573_pass,
        "is_smoke": False,
    }
    out_path = HERE / "results.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"verdict={verdict} route={out['kill_criteria']['1573']['route']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
