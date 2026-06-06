#!/usr/bin/env python3
"""
exp_hedgehog_behavior_adapter_conciseness — 3rd behavior-axis Hedgehog adapter
distilling concise-output attention routing from a 26B Gemma 4 teacher (with
π_Concise in context) into a rank-8 LoRA on (v_proj, o_proj) of Gemma 4 E4B.
Cousin of F#683 politeness and F#724 formality.

Pre-registered KCs (canonical DB text — do not edit):
  K#1881 target: Conciseness adapter output length reduction < 20% vs base
                 (kill if TRUE — no behavioral acquisition)
  K#1882 target: Conciseness adapter drops task accuracy > 3pp
                 (kill if TRUE — substance degradation; ONE-SIDED — degradation only)

DUAL-TARGET / ZERO-PROXY KC design — 2nd instance in the Hedgehog-framework
super-family (1st was F#724). 1st one-sided-safety sub-variant (F#724 was
two-sided orthogonality). K1882 is an asymmetric safety-target class.

This filing triggers the behavior-axis sub-cluster standalone-memory
promotion — 3rd behavior-axis instance (F#683 politeness → F#724 formality →
THIS conciseness).

Skills invoked: /mlx-dev (documented in MATH.md §0). /fast-mlx pending for the
_impl follow-up where the training loop lands.

Runtime structure:
  Phase 0 — Neutral-prompt curation (NEW set, length-neutral + register-neutral)
  Phase A — 26B teacher attention capture under π_Concise
  Phase B — Hedgehog cos-sim student training (rank-8 LoRA on v_proj+o_proj)
  Phase C — K1881 length reduction on 50 held-out neutral prompts
  Phase D — K1882 task accuracy on 100-question MMLU subset (seed=42)
  Phase E — F#666 verdict matrix resolution

Full budget: ~8–10 h on M5 Pro 48 GB (curation + 26B teacher residency + one
Hedgehog training job at 800 steps + length eval + MMLU eval). Exceeds
single-iteration researcher cap (30 min / 40 tool calls per guardrail 1009).
This scaffold supports SMOKE_TEST=1 and full runs; Phase 0/A/B/C/D raise
NotImplementedError pending dedicated `_impl` run.

F#702 hygiene-patch PROVISIONAL is APPLICABLE (DB row shipped with hygiene
defects). mem-impossibility-f666pure-saturation-implies-f702-unavailable does
NOT fire — both K1881 and K1882 are target KCs (not F#666-pure).
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

# Length-eval and accuracy-eval token budgets (locked in MATH.md §7)
K1881_MAX_TOKENS = 256
K1882_MAX_TOKENS = 64

# Axis = conciseness (NEW behavior-axis cousin of F#683 politeness + F#724 formality)
AXIS_NAME = "conciseness"
AXIS_SUB_CLUSTER = "behavior"
AXIS_COUSIN_FINDINGS = ("F#683", "F#724")

PI_CONCISE = (
    "You are a concise assistant. Reply in the fewest words necessary. "
    "Never elaborate beyond what the question requires. No preamble, no "
    "disclaimers."
)

# KCs — both targets (DUAL-TARGET / ZERO-PROXY design)
K1881_LENGTH_REDUCTION_FRACTION = 0.20   # ρ < 0.20 ⇒ KILL (no acquisition)
K1882_ACCURACY_DROP_PP = 3.0             # drop > 3pp ⇒ KILL (substance degradation; ONE-SIDED)


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
# Phase 0: Neutral-prompt curation (length-neutral + register-neutral)
# ─────────────────────────────────────────────

def curate_neutral_prompts(data_dir: Path, n_train: int, n_heldout: int) -> dict:
    """Generate 250 neutral-register + length-neutral prompts (200 train + 50
    held-out) covering knowledge questions, descriptive tasks, problem-
    solving. Manually verify length-neutrality — no implicit "in one sentence"
    or "in detail" cue. Pipeline pattern reuses F#683/F#724 Phase 0 curation,
    but the prompt content is NEW and length-neutrality audit is specific to
    this axis.

    NOT YET IMPLEMENTED. Requires:
      - Prompt-generation pipeline with length-neutrality classifier filter.
      - Manual review pass for length-neutrality.
      - Train/held-out split.
    """
    raise NotImplementedError(
        "Phase 0 (neutral-prompt curation) requires a curated prompt set with "
        "length-neutrality verification. Deferred to _impl follow-up."
    )


# ─────────────────────────────────────────────
# Phase A: Teacher attention capture under π_Concise
# ─────────────────────────────────────────────

def capture_teacher_attention(data_dir: Path, capture_dir: Path) -> dict:
    """26B Gemma 4 + π_Concise + Q in context. Capture {layer_idx: attn_output}
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
# Phase C: K1881 length reduction
# ─────────────────────────────────────────────

def measure_k1881_length_reduction(adapter_path: Path, held_out: list) -> float:
    """K1881: length reduction on 50 held-out neutral prompts.
    Both adapter and base generate (max_tokens=256, temperature=0); report
    ρ = 1 − mean_tokens(adapter) / mean_tokens(base). Token count is post-
    tokenizer count (not character count). Mean over 50 prompts.

    Kill condition: ρ < 0.20 (no behavioral acquisition).

    NOT YET IMPLEMENTED. Requires adapter on disk + held-out set.
    """
    raise NotImplementedError(
        "K1881 length eval depends on adapter existing and held-out "
        "neutral prompt set being curated."
    )


# ─────────────────────────────────────────────
# Phase D: K1882 task accuracy on MMLU subset (one-sided degradation)
# ─────────────────────────────────────────────

def measure_k1882_task_accuracy(adapter_path: Path,
                                 mmlu_subset: list) -> dict:
    """K1882: 100 random MMLU questions (seed=42), categories balanced.
    Greedy decoding, max_tokens=64. Score against canonical answer.

    Returns dict: {acc_adapter, acc_base, drop_pp}.
    Kill condition: drop_pp > 3 pp (one-sided — degradation only; improvements
    are benign).

    NOT YET IMPLEMENTED. Requires adapter on disk + MMLU subset.
    """
    raise NotImplementedError(
        "K1882 MMLU eval depends on adapter existing and MMLU subset being prepared."
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

    print("Hedgehog conciseness behavior adapter (3rd behavior-axis; triggers "
          "sub-cluster promotion)", flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_HELDOUT={N_HELDOUT}, "
          f"N_MMLU={N_MMLU}, N_STEPS={N_STEPS}", flush=True)
    print(f"mlx-lm version: {mlx_lm_version}", flush=True)
    print(f"Axis: {AXIS_NAME} (sub-cluster {AXIS_SUB_CLUSTER}, cousins "
          f"{AXIS_COUSIN_FINDINGS})", flush=True)
    print(f"KC design: DUAL-TARGET / ZERO-PROXY (2nd in super-family; 1st "
          f"one-sided-safety sub-variant)", flush=True)

    data_dir = EXPERIMENT_DIR / "data" / "neutral_conciseness_prompts"
    capture_dir = EXPERIMENT_DIR / "teacher_capture"
    adapters_dir = EXPERIMENT_DIR / "adapters"
    conciseness_adapter = adapters_dir / "hedgehog_cos_conciseness_r8"

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
        "axis_cousin_findings": list(AXIS_COUSIN_FINDINGS),
        "pi_concise": PI_CONCISE,
        "k1881_max_tokens": K1881_MAX_TOKENS,
        "k1882_max_tokens": K1882_MAX_TOKENS,
        "k1881_length_reduction_threshold": K1881_LENGTH_REDUCTION_FRACTION,
        "k1882_accuracy_drop_pp_threshold": K1882_ACCURACY_DROP_PP,
        "phase_0_curation": None,
        "phase_a_teacher_capture": None,
        "phase_b_train": None,
        "phase_c_k1881": {},
        "phase_d_k1882": {},
        "kc": {
            "K1881_length_reduction_lt_20pct": "untested",
            "K1882_accuracy_drop_gt_3pp_one_sided": "untested",
        },
        "verdict": "PROVISIONAL",
        "all_pass": False,
        "kc_design": "dual-target-zero-proxy",
        "kc_design_sub_variant": "one-sided-safety",
        "kc_design_super_family_instance_number": 2,
        "kc_design_sub_variant_first_in_super_family": True,
        "sub_type": "behavior-axis-extension",
        "super_family": "hedgehog-ablation",
        "super_family_sub_types": [
            "axis-extension (domain — 5 instances, closed)",
            "axis-extension (behavior — 2 instances F#683/F#724 → 3 with this) [SUB-CLUSTER PROMOTION TRIGGERED]",
            "loss-variant-ablation",
            "layer-selection-ablation",
            "hyperparameter-ablation",
            "data-augmentation-ablation",
        ],
        "sub_cluster_instance_count": 3,
        "sub_cluster_promotion_triggered": True,
        "super_family_total_instances": 14,
        "provisional_number_in_pile": 10,
        "blockers": [],
    }

    # ── Phase 0: neutral-prompt curation ────────────────────
    print("\n=== Phase 0: Neutral-prompt curation (length-neutral) ===", flush=True)
    try:
        results["phase_0_curation"] = curate_neutral_prompts(
            data_dir, N_TRAIN, N_HELDOUT)
    except NotImplementedError as exc:
        results["phase_0_curation"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase 0 not implemented: {exc}")

    # ── Phase A: teacher attention capture ──────────────────
    print("\n=== Phase A: 26B teacher attention capture under π_Concise ===",
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
            data_dir, capture_dir, conciseness_adapter)
    except NotImplementedError as exc:
        results["phase_b_train"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase B not implemented: {exc}")

    # ── Phase C: K1881 length reduction ─────────────────────
    print("\n=== Phase C: K1881 length reduction ===", flush=True)
    if conciseness_adapter.exists():
        try:
            k1881 = measure_k1881_length_reduction(
                conciseness_adapter, held_out=[])
            results["phase_c_k1881"]["K1881_length_reduction_fraction"] = round(
                float(k1881), 4)
        except NotImplementedError as exc:
            results["blockers"].append(f"K1881 not implemented: {exc}")
    else:
        results["blockers"].append(
            "Phase C skipped: conciseness adapter not produced (Phase B blocker)")

    # ── Phase D: K1882 task accuracy on MMLU subset ─────────
    print("\n=== Phase D: K1882 task accuracy (MMLU subset) ===", flush=True)
    if conciseness_adapter.exists():
        try:
            k1882 = measure_k1882_task_accuracy(conciseness_adapter,
                                                 mmlu_subset=[])
            results["phase_d_k1882"] = k1882
        except NotImplementedError as exc:
            results["blockers"].append(f"K1882 not implemented: {exc}")
    else:
        results["blockers"].append(
            "Phase D skipped: conciseness adapter not produced (Phase B blocker)")

    # ── KC resolution ──────────────────────────────────────
    # With Phase 0/A/B not implemented, both KCs remain "untested". When the
    # _impl follow-up lands, populate kc dict from phase_c_k1881 + phase_d_k1882
    # via the F#666 verdict matrix (MATH.md §3.4).

    results["total_time_s"] = round(time.time() - t_start, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Verdict: {results['verdict']}", flush=True)
    print(f"KCs: {json.dumps(results['kc'])}", flush=True)
    print(f"KC design: {results['kc_design']} "
          f"({results['kc_design_sub_variant']}; super-family instance "
          f"#{results['kc_design_super_family_instance_number']})",
          flush=True)
    print(f"Sub-type: {results['sub_type']} "
          f"(behavior sub-cluster instance #{results['sub_cluster_instance_count']} "
          f"[promotion triggered: {results['sub_cluster_promotion_triggered']}]; "
          f"super-family total #{results['super_family_total_instances']})",
          flush=True)
    print(f"Blockers ({len(results['blockers'])}):", flush=True)
    for b in results["blockers"]:
        print(f"  - {b}", flush=True)
    print(f"Total time: {results['total_time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
