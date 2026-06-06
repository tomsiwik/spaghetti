#!/usr/bin/env python3
"""
exp_hedgehog_behavior_adapter_formality — 2nd behavior-axis Hedgehog adapter
distilling formal-register attention routing from a 26B Gemma 4 teacher (with
π_Formal in context) into a rank-8 LoRA on (v_proj, o_proj) of Gemma 4 E4B.
Cousin of F#683 politeness adapter.

Pre-registered KCs (canonical DB text — do not edit):
  K#1879 target: Formality adapter formality auto-judge < +10pp vs base
                 (kill if TRUE — no behavioral acquisition)
  K#1880 target: Formality adapter changes factual accuracy > 2pp
                 (kill if TRUE — style leaks into substance / orthogonality fail)

DUAL-TARGET / ZERO-PROXY KC design — FIRST in the Hedgehog-framework super-
family. K1880 is a NEW KC class (style/substance orthogonality safety target).

Skills invoked: /mlx-dev (documented in MATH.md §0). /fast-mlx pending for the
_impl follow-up where the training loop lands.

Runtime structure:
  Phase 0 — Neutral-prompt curation (NEW set, NOT F#683 reuse)
  Phase A — 26B teacher attention capture under π_Formal
  Phase B — Hedgehog cos-sim student training (rank-8 LoRA on v_proj+o_proj)
  Phase C — K1879 formality auto-judge on 50 held-out neutral prompts
  Phase D — K1880 factual accuracy on 100-question MMLU subset (seed=42)
  Phase E — F#666 verdict matrix resolution

Full budget: ~8–10 h on M5 Pro 48 GB (curation + 26B teacher residency + one
Hedgehog training job at 800 steps + paired-judge eval + MMLU eval). Exceeds
single-iteration researcher cap (30 min / 40 tool calls per guardrail 1009).
This scaffold supports SMOKE_TEST=1 and full runs; Phase 0/A/B/C/D raise
NotImplementedError pending dedicated `_impl` run.

F#702 hygiene-patch PROVISIONAL is APPLICABLE (DB row shipped with hygiene
defects). mem-impossibility-f666pure-saturation-implies-f702-unavailable does
NOT fire — both K1879 and K1880 are target KCs (not F#666-pure).
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import mlx.core as mx

# M5 Pro 48 GB: leave 8 GB headroom; cap wired cache at 2 GB per mlx-dev guidance.
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

STUDENT_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
TEACHER_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 40 if IS_SMOKE else 200
N_HELDOUT = 10 if IS_SMOKE else 50
N_MMLU = 20 if IS_SMOKE else 100
N_STEPS = 50 if IS_SMOKE else 800
SEED = 42

ADAPTER_RANK = 8
ADAPTER_TARGETS = ("v_proj", "o_proj")  # F#627
LORA_SCALE = 6.0                        # <= 8 per F#328/F#330
BATCH_SIZE = 1
SEQLEN = 2048
ENABLE_THINKING = True                  # F#614/F#536

# Axis = formality (NEW behavior-axis cousin of F#683 politeness)
AXIS_NAME = "formality"
AXIS_SUB_CLUSTER = "behavior"
AXIS_COUSIN_FINDING = "F#683"

PI_FORMAL = (
    "You are a formal-register academic assistant. Reply in formal English with "
    "academic tone, no contractions, full sentences with subordinate clauses "
    "where appropriate."
)

# KCs — both targets (DUAL-TARGET / ZERO-PROXY design)
K1879_FORMALITY_DELTA_PP = 10.0          # Δ < +10pp ⇒ KILL (no acquisition)
K1880_FACTUAL_DRIFT_PP = 2.0             # |Δ| > 2pp ⇒ KILL (substance drift; two-sided)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ─────────────────────────────────────────────
# Phase 0: Neutral-prompt curation (NEW set, distinct from F#683)
# ─────────────────────────────────────────────

def curate_neutral_prompts(data_dir: Path, n_train: int, n_heldout: int) -> dict:
    """Generate 250 neutral-register prompts (200 train + 50 held-out) covering
    knowledge questions, descriptive tasks, problem-solving. Manually verify
    register neutrality — no implicit formal/informal cue. Pipeline pattern
    reuses F#683 Phase 0 curation, but the prompt content is NEW.

    NOT YET IMPLEMENTED. Requires:
      - Prompt-generation pipeline with a register-classifier filter.
      - Manual review pass for register neutrality.
      - Train/held-out split.
    """
    raise NotImplementedError(
        "Phase 0 (neutral-prompt curation) requires a curated prompt set with "
        "register-neutrality verification. Deferred to _impl follow-up."
    )


# ─────────────────────────────────────────────
# Phase A: Teacher attention capture under π_Formal
# ─────────────────────────────────────────────

def capture_teacher_attention(data_dir: Path, capture_dir: Path) -> dict:
    """26B Gemma 4 + π_Formal + Q in context. Capture {layer_idx: attn_output}
    for all 42 layers per (Q, A) pair on the train split. Sequential-phase
    eviction per F#673; pre-compute offline.

    NOT YET IMPLEMENTED. Requires 26B teacher residency on 48 GB M5 Pro.
    """
    raise NotImplementedError(
        "Teacher attention capture requires 26B model residency. See F#673."
    )


# ─────────────────────────────────────────────
# Phase B: Student training (Hedgehog cos-sim)
# ─────────────────────────────────────────────

def train_student_hedgehog(data_dir: Path, capture_dir: Path,
                           adapter_path: Path) -> dict:
    """Student rank-8 LoRA on (v_proj, o_proj) with per-layer cos-sim loss:
    L = mean_l (1 − cos(A_t_l, A_s_l)). 800 steps on N_TRAIN = 200 (Q, A)
    pairs (4 epochs). AdamW, mx.eval + mx.clear_cache between batches.
    nn.value_and_grad(student, loss_fn) functional gradients.

    NOT YET IMPLEMENTED. Requires custom MLX training loop with per-layer hooks.
    """
    raise NotImplementedError(
        "Hedgehog student training loop requires custom MLX code. "
        "Deferred to _impl follow-up."
    )


# ─────────────────────────────────────────────
# Phase C: K1879 formality auto-judge
# ─────────────────────────────────────────────

def measure_k1879_formality_delta(adapter_path: Path, held_out: list) -> float:
    """K1879: blind-paired auto-judge formality on 50 held-out neutral prompts.
    Both adapter and base generate (max_tokens=128, temperature=0); judge
    scores each pair on 0–100 rubric (lexical register, syntactic complexity,
    hedging/qualification, contraction-rate). Position swap 50/50.

    Returns Δ = mean(score_adapter) − mean(score_base) on 0–100 scale.
    Kill condition: Δ < +10 pp (no behavioral acquisition).

    NOT YET IMPLEMENTED. Requires adapter on disk + judge API + held-out set.
    """
    raise NotImplementedError(
        "K1879 formality judge eval depends on adapter existing and held-out "
        "neutral prompt set being curated."
    )


# ─────────────────────────────────────────────
# Phase D: K1880 factual accuracy on MMLU subset
# ─────────────────────────────────────────────

def measure_k1880_factual_accuracy(adapter_path: Path,
                                   mmlu_subset: list) -> dict:
    """K1880: 100 random MMLU questions (seed=42), categories balanced.
    Greedy decoding, max_tokens=64. Score against canonical answer.

    Returns dict: {acc_adapter, acc_base, abs_delta_pp}.
    Kill condition: |Δ| > 2 pp (style leaks into substance — two-sided).

    NOT YET IMPLEMENTED. Requires adapter on disk + MMLU subset.
    """
    raise NotImplementedError(
        "K1880 MMLU eval depends on adapter existing and MMLU subset being prepared."
    )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    t_start = time.time()
    log_memory("start")

    try:
        import mlx_lm
        mlx_lm_version = getattr(mlx_lm, "__version__", "unknown")
    except Exception:
        mlx_lm_version = "import_failed"

    print("Hedgehog formality behavior adapter (2nd behavior-axis cousin of F#683)",
          flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_HELDOUT={N_HELDOUT}, "
          f"N_MMLU={N_MMLU}, N_STEPS={N_STEPS}", flush=True)
    print(f"mlx-lm version: {mlx_lm_version}", flush=True)
    print(f"Axis: {AXIS_NAME} (sub-cluster {AXIS_SUB_CLUSTER}, cousin "
          f"{AXIS_COUSIN_FINDING})", flush=True)
    print(f"KC design: DUAL-TARGET / ZERO-PROXY (1st in super-family)", flush=True)

    data_dir = EXPERIMENT_DIR / "data" / "neutral_formality_prompts"
    capture_dir = EXPERIMENT_DIR / "teacher_capture"
    adapters_dir = EXPERIMENT_DIR / "adapters"
    formality_adapter = adapters_dir / "hedgehog_cos_formality_r8"

    results = {
        "is_smoke": IS_SMOKE,
        "n_train": N_TRAIN,
        "n_heldout": N_HELDOUT,
        "n_mmlu": N_MMLU,
        "n_steps": N_STEPS,
        "student_model": STUDENT_MODEL,
        "teacher_model": TEACHER_MODEL,
        "mlx_lm_version": mlx_lm_version,
        "adapter_rank": ADAPTER_RANK,
        "adapter_targets": list(ADAPTER_TARGETS),
        "lora_scale": LORA_SCALE,
        "seqlen": SEQLEN,
        "enable_thinking": ENABLE_THINKING,
        "axis": AXIS_NAME,
        "axis_sub_cluster": AXIS_SUB_CLUSTER,
        "axis_cousin_finding": AXIS_COUSIN_FINDING,
        "pi_formal": PI_FORMAL,
        "k1879_formality_delta_pp_threshold": K1879_FORMALITY_DELTA_PP,
        "k1880_factual_drift_pp_threshold": K1880_FACTUAL_DRIFT_PP,
        "phase_0_curation": None,
        "phase_a_teacher_capture": None,
        "phase_b_train": None,
        "phase_c_k1879": {},
        "phase_d_k1880": {},
        "kc": {
            "K1879_formality_delta_lt_10pp": "untested",
            "K1880_factual_drift_gt_2pp_two_sided": "untested",
        },
        "verdict": "PROVISIONAL",
        "all_pass": False,
        "kc_design": "dual-target-zero-proxy",
        "kc_design_first_in_super_family": True,
        "sub_type": "behavior-axis-extension",
        "super_family": "hedgehog-ablation",
        "super_family_sub_types": [
            "axis-extension (domain — 5 instances, closed)",
            "axis-extension (behavior — 1 instance F#683 → 2 with this)",
            "loss-variant-ablation",
            "layer-selection-ablation",
            "hyperparameter-ablation",
            "data-augmentation-ablation",
        ],
        "sub_cluster_instance_count": 2,
        "super_family_total_instances": 13,
        "provisional_number_in_pile": 9,
        "blockers": [],
    }

    # ── Phase 0: neutral-prompt curation ────────────────────
    print("\n=== Phase 0: Neutral-prompt curation (NEW set) ===", flush=True)
    try:
        results["phase_0_curation"] = curate_neutral_prompts(
            data_dir, N_TRAIN, N_HELDOUT)
    except NotImplementedError as exc:
        results["phase_0_curation"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase 0 not implemented: {exc}")

    # ── Phase A: teacher attention capture ──────────────────
    print("\n=== Phase A: 26B teacher attention capture under π_Formal ===",
          flush=True)
    try:
        results["phase_a_teacher_capture"] = capture_teacher_attention(
            data_dir, capture_dir)
    except NotImplementedError as exc:
        results["phase_a_teacher_capture"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase A not implemented: {exc}")

    # ── Phase B: student Hedgehog training ──────────────────
    print("\n=== Phase B: Student Hedgehog cos-sim training ===", flush=True)
    try:
        results["phase_b_train"] = train_student_hedgehog(
            data_dir, capture_dir, formality_adapter)
    except NotImplementedError as exc:
        results["phase_b_train"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase B not implemented: {exc}")

    # ── Phase C: K1879 formality auto-judge ─────────────────
    print("\n=== Phase C: K1879 formality auto-judge ===", flush=True)
    if formality_adapter.exists():
        try:
            k1879 = measure_k1879_formality_delta(
                formality_adapter, held_out=[])
            results["phase_c_k1879"]["K1879_formality_delta_pp"] = round(
                float(k1879), 2)
        except NotImplementedError as exc:
            results["blockers"].append(f"K1879 not implemented: {exc}")
    else:
        results["blockers"].append(
            "Phase C skipped: formality adapter not produced (Phase B blocker)")

    # ── Phase D: K1880 factual accuracy on MMLU subset ──────
    print("\n=== Phase D: K1880 factual accuracy (MMLU subset) ===", flush=True)
    if formality_adapter.exists():
        try:
            k1880 = measure_k1880_factual_accuracy(formality_adapter,
                                                   mmlu_subset=[])
            results["phase_d_k1880"] = k1880
        except NotImplementedError as exc:
            results["blockers"].append(f"K1880 not implemented: {exc}")
    else:
        results["blockers"].append(
            "Phase D skipped: formality adapter not produced (Phase B blocker)")

    # ── KC resolution ──────────────────────────────────────
    # With Phase 0/A/B not implemented, both KCs remain "untested". When the
    # _impl follow-up lands, populate kc dict from phase_c_k1879 + phase_d_k1880
    # via the F#666 verdict matrix (MATH.md §3.4).

    results["total_time_s"] = round(time.time() - t_start, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Verdict: {results['verdict']}", flush=True)
    print(f"KCs: {json.dumps(results['kc'])}", flush=True)
    print(f"KC design: {results['kc_design']} "
          f"(1st-in-super-family: {results['kc_design_first_in_super_family']})",
          flush=True)
    print(f"Sub-type: {results['sub_type']} "
          f"(behavior sub-cluster instance #{results['sub_cluster_instance_count']}; "
          f"super-family total #{results['super_family_total_instances']})",
          flush=True)
    print(f"Blockers ({len(results['blockers'])}):", flush=True)
    for b in results["blockers"]:
        print(f"  - {b}", flush=True)
    print(f"Total time: {results['total_time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
