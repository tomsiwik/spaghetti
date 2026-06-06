"""
exp_g4_adapter_class_composition_full — PROVISIONAL design-only scaffold.

Per reviewer.md §5 "PROVISIONAL (novel-mechanism design-only sub-case)" and
mem-antipattern-novel-mechanism-single-iteration-scope option (i): this scaffold
never runs the full training pipeline in-iteration. It always writes a valid
results.json with verdict=PROVISIONAL and all KCs as "untested", enumerates
structural blockers, and returns 0. The `_impl` follow-up at P3 is the ticket
for actual execution.

Blockers (design-complete, execution-deferred):
  B1. MoLoRA module: no turn-key `mlx_lm.lora --fine-tune-type molora`. Needs
      custom module `micro/utils/molora.py` with N experts + softmax gate.
  B2. 15 adapter trainings (3 classes x 5 domains) at Gemma 4 E4B 4-bit
      ~30-60 min each = 8-15h wall-clock. Exceeds 90-min single-iteration budget.
  B3. 5 domain corpora: code/math/medical/legal/law. Must be curated and
      validated for non-overlap (legal vs law is a real risk).
  B4. Composition-eval harness for N=5: must load all 5 adapters per class,
      apply class-specific composition formula, run MMLU-Pro n=1000.
  B5. Paired bootstrap statistics: 10000 resamples, 3 pairwise comparisons.
"""

import json
import os
import sys
from pathlib import Path


# Kill criteria from MATH.md §3, pre-registered and target-gated per F#666.
KILL_CRITERIA = [
    {
        "id": "K1",
        "text": (
            "Structural + adapter-class health: >=13/15 class-domain trainings "
            "converge (final_loss < 1.1 * min_train_loss AND < 0.7 * initial_loss)"
        ),
        "result": "untested",
        "note": "blocker: B1 (MoLoRA module) + B2 (15 trainings at macro scale)",
    },
    {
        "id": "K2",
        "text": (
            "Target behavioral: acc_A - max(acc_{B.1}, acc_{B.2}) >= 0.03 "
            "with 95% CI lower bound > 0 on MMLU-Pro n=1000 at N=5 composition"
        ),
        "result": "untested",
        "note": "blocker: B2 + B4 (composition-eval harness) + B5 (bootstrap)",
    },
    {
        "id": "K3",
        "text": (
            "Proxy confirmation: median(dev_D on trained DoRA at composition "
            "time) > 1e-3 (closes parent F#679 init-assumption caveat)"
        ),
        "result": "untested",
        "note": "blocker: requires trained DoRA artifacts from B2",
    },
    {
        "id": "K4",
        "text": (
            "Rank-ablation: sign of K2 is stable across r=6 and r=8 (rules "
            "out rank-specific artifacts)"
        ),
        "result": "untested",
        "note": "blocker: doubles training cost from B2",
    },
]


def phase_a_curate_domain_corpora() -> None:
    """B3: Curate 5 non-overlapping domain corpora."""
    raise NotImplementedError(
        "Phase A: 5-domain corpus curation (code/math/medical/legal/law) "
        "with non-overlap verification. Deferred to _impl."
    )


def phase_b_train_adapters() -> None:
    """B1 + B2: 15 trainings (3 classes x 5 domains)."""
    raise NotImplementedError(
        "Phase B: 15 adapter trainings. LoRA via `mlx_lm.lora --fine-tune-type "
        "lora`, DoRA via `mlx_lm.lora --fine-tune-type dora` (verify mlx-lm>=0.22), "
        "MoLoRA via custom module micro/utils/molora.py (does not yet exist). "
        "All: v_proj + o_proj, r=6, scale=6.0, 1000 steps, batch=4, max_len=2048. "
        "mx.clear_cache() between domain trainings (F#673). Deferred to _impl."
    )


def phase_c_compose_and_eval() -> None:
    """B4: Compose all 5 domain adapters per class, eval MMLU-Pro n=1000."""
    raise NotImplementedError(
        "Phase C: N=5 composition + MMLU-Pro eval. LoRA sum, DoRA elementwise-mean "
        "magnitude + renormalize, MoLoRA softmax-gated mixture. Deferred to _impl."
    )


def phase_d_ablate_rank() -> None:
    """K4: re-run at r=8."""
    raise NotImplementedError(
        "Phase D: K4 rank-ablation at r=8. Re-run Phase B + Phase C with r=8. "
        "Deferred to _impl."
    )


def phase_e_stats() -> None:
    """B5: paired bootstrap, 95% CI."""
    raise NotImplementedError(
        "Phase E: paired bootstrap (10000 resamples) for each of 3 pairwise "
        "class comparisons. Deferred to _impl."
    )


def main() -> int:
    """Graceful-failure: always write a valid provisional results.json."""
    out_dir = Path(__file__).resolve().parent
    results_path = out_dir / "results.json"

    blockers = {
        "B1_molora_module": {
            "description": "No turn-key `mlx_lm.lora --fine-tune-type molora`.",
            "fix": "Write micro/utils/molora.py with N experts + softmax gate, "
                   "unit test forward pass against manual Σ g_i B_i A_i x.",
        },
        "B2_training_wall_clock": {
            "description": "15 adapter trainings ~8-15h; exceeds 90-min cap.",
            "fix": "`_impl` iteration at P3 with dedicated compute budget.",
        },
        "B3_domain_corpora": {
            "description": "5 non-overlapping domain corpora not yet curated.",
            "fix": "HumanEval/GSM8K/PubMedQA/CaseHOLD/LegalBench "
                   "with non-overlap audit (legal vs law risk).",
        },
        "B4_composition_eval": {
            "description": "N=5 composition-eval harness does not exist for "
                           "DoRA / MoLoRA classes.",
            "fix": "Harness in `_impl`: load adapters, apply class formula, "
                   "run MMLU-Pro n=1000 per class.",
        },
        "B5_bootstrap_statistics": {
            "description": "Paired bootstrap 95% CI not yet implemented.",
            "fix": "scipy.stats.bootstrap or numpy-native; 10000 resamples.",
        },
    }

    results = {
        "experiment_id": "exp_g4_adapter_class_composition_full",
        "verdict": "PROVISIONAL",
        "is_smoke": False,
        "is_design_only": True,
        "all_pass": None,
        "kill_criteria": KILL_CRITERIA,
        "blockers": blockers,
        "iteration_scope": {
            "researcher_iteration_budget_min": 30,
            "estimated_full_pipeline_h": "8-15",
            "verdict_reason": (
                "Realistic wall-clock (~8-15h) exceeds single-iteration budget "
                "(30 min / 40 tool calls per guardrail 1009). Per "
                "mem-antipattern-novel-mechanism-single-iteration-scope option "
                "(i) and reviewer.md §5, filing PROVISIONAL-as-design with "
                "_impl follow-up at P3."
            ),
        },
        "reference_parent": {
            "experiment_id": "exp_g4_adapter_class_composition",
            "finding_id": 679,
            "status": "provisional",
            "note": (
                "Parent measured composition-geometry proxy only (dev_LoRA=0, "
                "dev_DoRA=0.089, dev_MoLoRA=0.667). Behavioral MMLU-Pro target "
                "was explicitly deferred in parent PAPER.md. This child closes "
                "that gap via K2 target KC — but execution deferred to _impl."
            ),
        },
        "antipattern_compliance": {
            "mem_novel_mechanism_single_iteration_scope": "applied (option i)",
            "mem_proxy_kc_mislabeled_target": (
                "parent K2 was proxy-labeled-as-target; this child's K2 is "
                "behavioral (MMLU-Pro accuracy) as required by F#666"
            ),
            "mem_preempt_child_parent_target_unverified": (
                "does not apply: parent proxy PASSED, child is the designated "
                "target-measurement follow-up, not an independent claim "
                "requiring parent SUPPORTED"
            ),
            "mem_schema_incomplete": (
                "all KCs reference trained artifacts; K1/K2/K3 each measure "
                "a trained-object property"
            ),
            "scope_preservation_F1_F5": "all five forbid-list items (MATH.md §0) registered",
        },
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[PROVISIONAL] Wrote {results_path}")
    print(f"[PROVISIONAL] 5 structural blockers; see results.json.blockers")
    print(f"[PROVISIONAL] 4 KCs untested; see results.json.kill_criteria")
    print(f"[PROVISIONAL] Follow-up: exp_g4_adapter_class_composition_full_impl at P3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
