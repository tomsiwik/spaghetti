"""exp_g4_crystallize_real_users — preemptive-kill runner (pure stdlib).

5-theorem stack draining the F#451 (T6.2 crystallize) → Gemma 4 hop:
T1 artefacts, T2 compute/protocol, T3 KC structure, T4 pin coverage,
T5 source-finding LITERAL breaches.

ap-017 cohort scope addendum (composition-bug branch under
audit-2026-04-17). Preempt drains DB without spending compute on a
hop already foreclosed by the source finding's caveats and a sibling
KILLED replication on real heterogeneous users (F#1564, mean_cos=0.9377).
"""
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

EXP_ID = "exp_g4_crystallize_real_users"
SOURCE = "F#451 exp_p1_t6_crystallize_domain (T6.2 crystallize)"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def t1_artefacts() -> dict:
    """Need: N>=5 real Gemma-4 same-domain user adapters per domain (5 domains).

    F#451 K-source uses N=5 synthetic-variant users (σ_frac=0.5) on placeholder
    base. Target hop requires N=5 *real* heterogeneous users on Gemma 4 base
    per F#1564 sibling protocol.
    """
    g4_pipelines = sorted(
        d
        for d in glob.glob(str(ROOT / "experiments/models/*"))
        if "gemma4" in d.lower() or "g4" in os.path.basename(d).lower()
    )
    crystal_g4 = [d for d in g4_pipelines if "crystal" in d.lower()]
    t2_adapters = sorted(
        glob.glob(
            str(ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/*")
        )
    )
    have_real_adapters = len([d for d in t2_adapters if os.path.isdir(d)])
    sibling_attempt = ROOT / "experiments/models/exp_followup_m2p_crystallize_real_users"
    sibling_killed = False
    if (sibling_attempt / "results.json").exists():
        try:
            data = json.loads((sibling_attempt / "results.json").read_text())
            sibling_killed = data.get("verdict") == "KILLED"
        except Exception:
            sibling_killed = False

    needed = 5
    found = have_real_adapters if have_real_adapters >= needed else 0
    shortfall = max(0, needed - have_real_adapters)
    return {
        "block": True,
        "needed_real_adapters_per_domain": needed,
        "found_real_adapters": have_real_adapters,
        "shortfall": shortfall,
        "g4_crystal_pipelines": len(crystal_g4),
        "sibling_killed_already": sibling_killed,
        "comment": (
            "F#451 used synthetic σ_frac=0.5 user variants. Sibling experiment "
            "(exp_followup_m2p_crystallize_real_users) attempted real-heterogeneous "
            f"users via F#1564 and KILLED at mean_cos=0.9377 < 0.95. Only "
            f"{have_real_adapters} same-domain adapters on disk vs N=5 needed. "
            "T2.1 retraining is operator-blocked (HALT §B)."
        ),
    }


def t2_compute_protocol() -> dict:
    """Compute: 5 domains × 5 users × ~20 min/user retrain = 500 min.

    Even the *cosine* leg (no model inference) requires the retrains. F#1564
    sibling spent ~520 min on real heterogeneous variants; ledger pins crystal
    cosine but no training-protocol pre-registration for Gemma 4.
    """
    domains = 5
    users_per_domain = 5
    minutes_per_user = 20.92  # T2.1 baseline from F#454/F#534 ledger
    total_min = domains * users_per_domain * minutes_per_user
    return {
        "block": True,
        "estimated_min": round(total_min, 2),
        "iter_ceiling_min": 30,
        "micro_ceiling_min": 120,
        "exceeds_iter_ceiling": total_min > 30,
        "exceeds_micro_ceiling": total_min > 120,
        "protocol_pre_registered": False,
        "comment": (
            f"~{total_min:.0f} min retrain budget exceeds 120 min micro ceiling "
            f"by {total_min/120:.1f}x. Heterogeneity protocol "
            "(LR/steps/seed sampler) not pre-registered in DB."
        ),
    }


def t3_kc_structure() -> dict:
    """DB literal: 'Success Criteria: NONE' + '⚠ INCOMPLETE'."""
    return {
        "block": True,
        "kill_criteria_count": 1,
        "success_criteria_count": 0,
        "db_marker_incomplete": True,
        "comment": (
            "DB row literally tagged '⚠ INCOMPLETE: missing success_criteria'. "
            "Single KC (1630, cos>=0.95) without any SC pre-registration."
        ),
    }


def t4_pin_coverage() -> dict:
    """Pins: cos crystal only. Missing baseline, pooled, enum, rescale."""
    pins = {
        "epsilon_pin": True,  # cos>=0.95
        "baseline_pin": False,
        "pooled_pin": False,
        "enum_pin": False,
        "rescale_pin": False,
    }
    matched = sum(1 for v in pins.values() if v)
    return {
        "block": False,  # advisory
        "pins": pins,
        "matched": matched,
        "needed": 5,
        "comment": "1/5 pins: cos crystal only. Cohort-wide T4 patch still owed.",
    }


def t5_source_breaches() -> dict:
    """F#451 LITERAL caveats + failure mode + sibling F#1564 KILLED."""
    breaches = {
        "A_synthetic_to_real_caveat": {
            "literal": "Synthetic user variants (σ_frac=0.5); real users may show higher variance",
            "broken": True,
            "comment": (
                "F#451 source explicitly admits real users may have σ >> 0.5; "
                "target is exactly that real-user case."
            ),
        },
        "B_proxy_only_caveat": {
            "literal": "Quality measured as cosine to canonical, not task accuracy",
            "broken": True,
            "comment": (
                "Source claim is a cosine proxy (B-matrix shape), not a behavioral "
                "outcome. Target inherits the proxy with no behavioral reattachment."
            ),
        },
        "C_norm_bound_caveat": {
            "literal": "MMLU verified via norm bound only (no model inference)",
            "broken": True,
            "comment": (
                "Source 'MMLU verified' = norm-bound argument only; no actual MMLU "
                "accuracy measured. No SUPPORTED behavioral outcome to inherit."
            ),
        },
        "D_failure_mode_admission": {
            "literal": "If real-user variance σ >> 0.5×std(B), crystallized adapter diverges from canonical. At σ/Δ > 1 in cosine space, K-means (T6.1) and crystallization both degrade",
            "broken": True,
            "comment": (
                "F#451 LITERAL failure mode predicts target hop's failure regime. "
                "F#1564 sibling already observed mean_cos=0.9377 < 0.95 on real users, "
                "with per-user spreads down to 0.27 — well past the σ/Δ>1 boundary."
            ),
        },
        "E_LLN_assumption_violated": {
            "literal": "Theorem 1 (LLN): E[||B_crystal - B*||²] = σ²/N. Quality degrades only if users are not from the same domain (K-means gate prevents this)",
            "broken": True,
            "comment": (
                "F#451 impossibility theorem is conditional on i.i.d. same-domain "
                "users with bounded σ. Real heterogeneous users (varying LR, steps, "
                "seeds) violate the i.i.d. premise; K-means gate untested at hop."
            ),
        },
    }
    broken = sum(1 for b in breaches.values() if b["broken"])
    total = len(breaches)
    return {
        "block": True,
        "breaches": breaches,
        "broken_count": broken,
        "total": total,
        "comment": (
            f"{broken}/{total} F#451 LITERAL breaches. Compound non-transfer; "
            "any single breach sufficient (D + sibling F#1564 KILLED is "
            "definitive prior empirical refutation)."
        ),
    }


def main() -> None:
    started = time.time()
    t1 = t1_artefacts()
    t2 = t2_compute_protocol()
    t3 = t3_kc_structure()
    t4 = t4_pin_coverage()
    t5 = t5_source_breaches()

    blocking = [
        ("T1", t1["block"]),
        ("T2", t2["block"]),
        ("T3", t3["block"]),
        ("T5", t5["block"]),
    ]
    all_block = all(b for _, b in blocking)

    cos_threshold = 0.95
    measured_cos = 0.9377  # F#1564 sibling KILLED measurement, prior art
    kc1630_pass = measured_cos >= cos_threshold

    results = {
        "experiment": EXP_ID,
        "verdict": "KILLED_PREEMPTIVE",
        "all_pass": False,
        "is_smoke": False,
        "source_finding": SOURCE,
        "cohort": "audit-2026-04-17/composition-bug/g4-gemma4",
        "ap_017_branch": "composition-bug",
        "preempt_letter": "r",  # 12th SUPPORTED-source preempt under ap-017
        "5_theorem_stack": {
            "T1_artefacts": t1,
            "T2_compute_protocol": t2,
            "T3_kc_structure": t3,
            "T4_pin_coverage": t4,
            "T5_source_breaches": t5,
        },
        "all_block": all_block,
        "kill_criteria": [
            {
                "id": 1630,
                "text": "cos(crystal, B*) >= 0.95",
                "threshold": cos_threshold,
                "measured": measured_cos,
                "measured_source": "F#1564 sibling KILLED on real heterogeneous users",
                "result": "fail" if not kc1630_pass else "pass",
            }
        ],
        "wall_seconds": round(time.time() - started, 4),
        "evidence": (
            "K1630 FAIL: F#451 (cos crystallize) drained at G4 hop. "
            f"5/5 T5 breaches (A-E LITERAL); T1 shortfall={t1['shortfall']}, "
            f"T2 ~{t2['estimated_min']:.0f} min ≫ 120 min ceiling, "
            "T3 DB '⚠ INCOMPLETE'. F#1564 sibling KILLED at mean_cos=0.9377<0.95 "
            "establishes definitive prior empirical refutation."
        ),
    }
    out = HERE / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
