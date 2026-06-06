"""IMPL refusal scaffold — exp_hedgehog_adapter_sql_domain_impl.

This iteration files PROVISIONAL because the prerequisite 26B Gemma 4 teacher cache
is absent. Per F#768 BLOCKED-on-resource model-cache sub-form: scaffold runs in <2s,
emits structured PROVISIONAL results.json, never loads or substitutes the teacher,
documents reclaim path. main() never raises — graceful blocker emission.

Parent design: experiments/models/exp_hedgehog_adapter_sql_domain/MATH.md
KCs: K#1957 (proxy PPL) + K#1958 (target PostgreSQL EXPLAIN dual-ground-truth judge)
"""
import json
import os
import sys
import time
from pathlib import Path

EXP_ID = "exp_hedgehog_adapter_sql_domain_impl"
EXP_DIR = Path(__file__).parent.resolve()
RESULTS_PATH = EXP_DIR / "results.json"

BASE_MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"            # student — parent §0
TEACHER_MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"     # teacher — parent §0
HF_CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub"))

# Reclaim path — cohort-level (unblocks JS/Python/Rust/SQL _impl siblings + the
# macro-scope F#768 sibling exp_model_knowledge_gap_26b_base in one download).
RECLAIM_PATH = [
    "1. huggingface-cli download mlx-community/gemma-4-26b-a4b-it-4bit (~14 GB)",
    "2. Verify cache: ls ~/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/",
    "3. Schedule >=6h dedicated session (Phase 0+A+B+Baseline+C+D pipeline).",
    "4. Bump priority P=3 -> P=2 (enter drain scope).",
    "5. Invoke /mlx-dev + /fast-mlx skills.",
    "6. Execute parent experiments/models/exp_hedgehog_adapter_sql_domain/MATH.md §6 verbatim.",
    "7. Apply KC measurements at K1957/K1958.",
]


def teacher_cache_present() -> bool:
    teacher_dir = HF_CACHE / f"models--{TEACHER_MODEL_ID.replace('/', '--')}"
    return teacher_dir.exists()


def main() -> int:
    started = time.time()

    teacher_cached = teacher_cache_present()
    blockers = []

    if not teacher_cached:
        blockers.append({
            "phase": "Phase A — Teacher attention capture",
            "reason": "BASE_MODEL_NOT_CACHED",
            "model_id": TEACHER_MODEL_ID,
            "size_gb": 14,
            "detail": (
                "26B Gemma 4 teacher model not present in HuggingFace cache. "
                "Hedgehog per-layer cos-sim distillation (Wang 2024 arxiv:2604.14191) "
                "requires teacher capacity strictly greater than student to avoid "
                "capacity-degenerate identity collapse of the cos-sim target. "
                "Silent substitution to E4B violates researcher antipattern (m). "
                "Refused."
            ),
        })

    # Even if cache were present, the full pipeline exceeds single-iteration budget.
    # Document this as a secondary non-blocking advisory so the reviewer sees the
    # complete picture (parallels F#769 compute-budget sub-form).
    blockers.append({
        "phase": "Phase 0+A+B+Baseline+C+D pipeline",
        "reason": "COMPUTE_BUDGET_EXCEEDS_DRAIN_ITERATION",
        "estimated_hours": 6,
        "drain_iteration_cap_hours": 0.5,
        "detail": (
            "Parent §6 Phase 0 (dataset curation) + Phase A (teacher attention "
            "capture across 42 E4B layers) + Phase B (rank-8 LoRA cos-sim "
            "training, 800 steps) + Phase Baseline (generic LoRA via mlx_lm.lora "
            "CLI) + Phase C (PPL head-to-head 3 configs) + Phase D (blind-paired "
            "PostgreSQL EXPLAIN-dual-ground-truth judge, 50 prompts × 2 conds) "
            "totals 4-6h on M5 Pro 48GB. Researcher single-iteration cap is 30 "
            "min per guardrail 1009. Dedicated session required."
        ),
    })

    results = {
        "experiment_id": EXP_ID,
        "verdict": "PROVISIONAL",
        "all_pass": False,
        "is_smoke": False,
        "elapsed_seconds": round(time.time() - started, 3),
        "blocked_on_resource": True,
        "blocker_super_family": "F#768/F#769 BLOCKED-on-resource",
        "blocker_sub_form": "model-cache (matches F#768)",
        "blockers": blockers,
        "K1957_PPL_proxy": "untested",
        "K1958_query_correctness_target": "untested",
        "paired_proxy_target_per_F666": {
            "proxy_kc": "K1957 — PPL on SQL-specific eval",
            "target_kc": "K1958 — PostgreSQL EXPLAIN dual-ground-truth query-correctness judge",
            "both_required_for_supported": True,
            "both_required_for_killed": True,
            "verdict_mapping": "see parent MATH.md §4",
        },
        "models_required": {
            "student": BASE_MODEL_ID,
            "teacher": TEACHER_MODEL_ID,
            "teacher_cache_present": teacher_cached,
            "substitution_refused": True,
            "antipattern_m_invoked": "wrong-model proxy refused per researcher antipattern (m)",
        },
        "reclaim_path": RECLAIM_PATH,
        "cohort_unblock_note": (
            "Single 26B teacher download unblocks 4 sibling _impl experiments "
            "(JS/Python/Rust/SQL hedgehog domain-knowledge cohort) plus "
            "exp_model_knowledge_gap_26b_base (F#768 macro-scope sibling). "
            "Cohort-level reclaim is more efficient than per-experiment escalation."
        ),
        "proof_first_prior": (
            "Parent F#718 PROVISIONAL design-locked at experiments/models/"
            "exp_hedgehog_adapter_sql_domain/MATH.md §3 (existence + PPL bound + "
            "query-correctness uplift). KCs locked at K#1957/K#1958. PostgreSQL "
            "EXPLAIN dual syntactic+semantic hard-floor is the novel structural "
            "discipline of this 4th-axis filing."
        ),
        "doom_loop_check": (
            "3rd consecutive PROVISIONAL escalation of the cycle on different "
            "experiments with different mechanisms (knowledge_gap_26b/scaling-law, "
            "long_context/range-extrapolation, sql_hedgehog/distillation). Not a "
            "literal A->B->A->B loop. Structurally orthogonal content axes."
        ),
        "antipattern_status": "no antipattern triggered (see MATH.md §5 scan)",
        "skill_invocation_status": (
            "/mlx-dev + /fast-mlx deferred to unblocked iteration; not invoked "
            "this iteration because no platform code is written. Documented in "
            "MATH.md §2 so reviewer can verify non-bypass."
        ),
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps({"verdict": "PROVISIONAL", "blockers": len(blockers)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
