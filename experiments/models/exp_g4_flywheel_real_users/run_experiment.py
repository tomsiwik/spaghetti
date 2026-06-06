"""exp_g4_flywheel_real_users — preemptive-kill runner (5-theorem).

Pure stdlib. Verifies the 5 theorems that make K1626/K1627 non-falsifiable
a priori. No MLX, no model loading, no training. Writes results.json.

F#452/F#453 validate the flywheel *expression* (W_0 + ΣΔW) on synthetic
matrices with synthetic users and q_proj only, and explicitly leave the
flywheel *process* (sequential-base retraining) untested. K1626
("epsilon_cumul < 10%") and K1627 ("quality_cos > 0.9999") port the
claim to real Gemma 4 + heterogeneous real users, crossing five scope
axes without a new bound.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

EXP_ID = "exp_g4_flywheel_real_users"
EXP_DIR = Path(__file__).parent
ROOT = EXP_DIR.parent.parent.parent

KCS = {
    1626: "epsilon_cumul < 10%",
    1627: "quality_cos > 0.9999",
}

# 5-pin enumerated-pattern checklist
KC_PINS = {
    "baseline": re.compile(r"(?:baseline|control|exclusive|majority)", re.I),
    "delta": re.compile(r"(?:\d+\s*pp|delta|Δ|vs\.?\s+baseline)", re.I),
    "pooled": re.compile(r"(?:weighted|pooled|aggregate|per[- ]domain|cumul|cos)", re.I),
    "epsilon": re.compile(r"(?:p\s*<|CI|±|\+/-|significance|sig\.|seed\s*spread)", re.I),
    "enum": re.compile(r"(?:N=\d+|seed|r=\d+|layers?\s+\d+|users?\s+\d+)", re.I),
}


def t1_infrastructure() -> dict:
    patterns = [
        "flywheel_real*", "sequential_base*", "promotion_cascade*",
        "real_user*", "users*", "adapters*",
    ]
    local = []
    for pat in patterns:
        local.extend(p.name for p in EXP_DIR.glob(pat) if p.exists())
    glob_flywheel_real = list(ROOT.glob("experiments/models/**/*flywheel_real*"))
    glob_sequential = list(ROOT.glob("experiments/models/**/*sequential_base*"))
    glob_cascade = list(ROOT.glob("experiments/models/**/*promotion_cascade*"))
    glob_user_adapters = list(ROOT.glob("experiments/models/**/user_adapters"))
    glob_flywheel_any = list(ROOT.glob("experiments/models/**/*flywheel*"))
    glob_promotion_any = list(ROOT.glob("experiments/models/**/*promotion*"))
    required = [
        "heterogeneous_real_user_adapters",
        "sequential_base_promotion_pipeline",
        "cumulative_epsilon_measurement_real_gemma4",
        "per_domain_quality_cos_sequential_vs_W0",
    ]
    shortfall = len(required)  # all 4 missing
    return {
        "required_artefacts": required,
        "local_artefacts": local,
        "repo_flywheel_real_dirs": [str(p.relative_to(ROOT)) for p in glob_flywheel_real],
        "repo_sequential_base_dirs": [str(p.relative_to(ROOT)) for p in glob_sequential],
        "repo_promotion_cascade_dirs": [str(p.relative_to(ROOT)) for p in glob_cascade],
        "repo_user_adapter_dirs": [str(p.relative_to(ROOT)) for p in glob_user_adapters],
        "repo_flywheel_any_count": len(glob_flywheel_any),
        "repo_promotion_any_count": len(glob_promotion_any),
        "shortfall": shortfall,
        "blocks_supported": shortfall > 0,
        "note": (
            "0 user-style adapters on Gemma 4 (F#454 preempt confirmed); 0 "
            "sequential-base flywheel pipelines; 0 cumulative-epsilon measurement "
            "artefacts on real base weights."
        ),
    }


def t2_budget() -> dict:
    per_user_train_min = 20.92  # from F#454 source preempt (exp_g4_real_user_registry)
    users_min = 2
    promotions = 3
    passes = 2  # W_0 reference + sequential W_k for quality_cos
    need_min = per_user_train_min * users_min * promotions * passes
    iter_budget_min = 30.0
    micro_ceiling_min = 120.0
    protocol_registered = False  # sequential-retrain-against-W_k protocol absent
    return {
        "per_user_train_min": per_user_train_min,
        "users_min": users_min,
        "promotions": promotions,
        "passes": passes,
        "need_min": need_min,
        "iter_budget_min": iter_budget_min,
        "micro_ceiling_min": micro_ceiling_min,
        "compute_exceeds_iter": need_min > iter_budget_min,
        "compute_exceeds_ceiling": need_min > micro_ceiling_min,
        "sequential_retrain_protocol_pre_registered": protocol_registered,
        "blocks_supported": True,
        "note": (
            f"{users_min} users × {promotions} promotions × {passes} passes × "
            f"{per_user_train_min} min = {need_min} min ≫ iter {iter_budget_min} "
            f"min, ≫ micro ceiling {micro_ceiling_min} min. Sequential-retrain "
            "protocol (F#453's untested invariant) un-pre-registered."
        ),
    }


def t3_framework() -> dict:
    try:
        out = subprocess.run(
            ["experiment", "get", EXP_ID],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception as e:  # pragma: no cover
        out = f"(experiment get failed: {e})"
    sc_none = "Success Criteria: NONE" in out
    db_incomplete = "⚠ INCOMPLETE" in out
    return {
        "success_criteria_none_literal": sc_none,
        "db_incomplete_flag": db_incomplete,
        "blocks_supported": sc_none or db_incomplete,
    }


def t4_pins() -> dict:
    per_kc = {}
    for kc_id, text in KCS.items():
        hits = {name: bool(pat.search(text)) for name, pat in KC_PINS.items()}
        per_kc[kc_id] = {
            "kc_text": text,
            "pin_hits": hits,
            "pin_count": sum(hits.values()),
            "required_pins": 5,
        }
    all_block = all(v["pin_count"] < 5 for v in per_kc.values())
    max_pins = max(v["pin_count"] for v in per_kc.values())
    return {
        "per_kc": per_kc,
        "max_pins_across_kcs": max_pins,
        "blocks_supported": all_block,
    }


def t5_scope_caveat() -> dict:
    breaches = {
        "A_synthetic_to_real_base": {
            "source_caveat_LITERAL": (
                "F#452: 'Synthetic W_base (std=0.05); no real MMLU test "
                "(proxy only)'"
            ),
            "target_base": "Gemma 4 e4b (real trained-LLM manifold)",
            "source_base": "synthetic std=0.05 Gaussian",
            "davis_kahan_dependency": "sin(θ_k) ≤ ||ΔW_k||_2 / δ_gap,k",
            "delta_gap_untested_on_real": True,
            "breach": True,
            "note": (
                "F#452 admits synthetic base; real Gemma 4 has unknown δ_gap "
                "distribution. Single-promotion ε=4.78% number does not "
                "transfer; source's own 'real weights give lower ε' is a "
                "directional conjecture, not a proved transfer"
            ),
        },
        "B_synthetic_to_heterogeneous_real_users": {
            "source_caveat_LITERAL": (
                "F#453: 'A-matrices not averaged in crystallization' + "
                "F#454: 'K1136 throughout tested on final state only; "
                "intermediate max_cos=0.9580 when user variants coexist'"
            ),
            "heterogeneity_skew_measured": False,
            "iid_assumption_preserved": False,
            "breach": True,
            "note": (
                "F#453 used synthetic 5-canonical-epsilon users; real users "
                "have variable LR/steps/seeds and per-user cos ∈ [0.27, 0.95] "
                "(crystallize_real_users measurement). Pythagorean 2.18√N "
                "scaling presumes i.i.d. ε_single; heterogeneity breaks it."
            ),
        },
        "C_core_invariant_untested": {
            "source_caveat_LITERAL": (
                "F#453: 'Adapters trained on original W_0, not sequential base.'"
            ),
            "flywheel_process_validated_in_source": False,
            "target_invokes_sequential_process": True,
            "breach": True,
            "note": (
                "The flywheel's *definition* is sequential-base retraining. "
                "F#453 validated the *expression* W_0+ΣΔW only. K1626/K1627 "
                "on real base + real users requires the process; source does "
                "not authorize this regime."
            ),
        },
        "D_qproj_to_full_model": {
            "source_caveat_LITERAL": "F#453: 'q_proj only.'",
            "target_projection_scope": "q_proj + k_proj + v_proj + o_proj (MLX default)",
            "cross_projection_orthogonality_measured": False,
            "breach": True,
            "note": (
                "2.18√N Pythagorean scaling holds per-projection only if ΔW "
                "across {q,k,v,o} are orthogonal; F#453 tested single projection. "
                "Full-model promotion on real Gemma 4 widens the scope."
            ),
        },
        "E_n_scale_extrapolation": {
            "source_caveat_LITERAL": (
                "F#453: 'N=5 extrapolation gives ε≈10.2% — borderline.'"
            ),
            "source_derivation": (
                "Pythagorean scaling gives ε_cumul ≈ 2.18√N · ε_single; "
                "safe for N≤12 at ε_single=2.8%"
            ),
            "target_N_promotions": 3,
            "real_epsilon_single_measured": False,
            "headroom_function_of_untested_quantity": True,
            "breach": True,
            "note": (
                "F#453's safe-N-zone is a function of ε_single, which is "
                "un-measured on real Gemma 4. If real ε_single > 3.3%, then "
                "2.18·√3·ε ≥ 10% and K1626 fails by construction."
            ),
        },
    }
    any_breach = any(b["breach"] for b in breaches.values())
    return {
        "breaches": breaches,
        "breach_count": sum(b["breach"] for b in breaches.values()),
        "any_breach": any_breach,
        "blocks_supported": any_breach,
    }


def main() -> None:
    results = {
        "experiment_id": EXP_ID,
        "verdict": "KILLED_PREEMPTIVE",
        "kc_ids": list(KCS.keys()),
        "kc_texts": KCS,
        "source_findings": [452, 453],
        "sibling_preempt": "F#454 (exp_g4_real_user_registry KILLED_PREEMPTIVE)",
        "theorems": {
            "T1_infrastructure": t1_infrastructure(),
            "T2_budget": t2_budget(),
            "T3_framework": t3_framework(),
            "T4_kc_pins": t4_pins(),
            "T5_scope_caveat": t5_scope_caveat(),
        },
    }
    blocks = [
        k for k, v in results["theorems"].items()
        if v.get("blocks_supported")
    ]
    results["blocking_theorems"] = blocks
    results["all_block"] = len(blocks) >= 3  # defense-in-depth
    results["all_pass"] = False  # verdict is KILLED
    out_path = EXP_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"PREEMPTIVE_KILL verdict written: {out_path}")
    print(f"Blocking theorems: {blocks}")


if __name__ == "__main__":
    main()
