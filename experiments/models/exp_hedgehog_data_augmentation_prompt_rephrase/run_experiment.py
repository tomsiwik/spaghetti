#!/usr/bin/env python3
"""
exp_hedgehog_data_augmentation_prompt_rephrase — Data-augmentation ablation comparing
Hedgehog cos-sim distillation trained on 1× original prompts vs 5× rephrased prompts
(temperature 1.2, 26B Gemma 4 rephrasing model) on the politeness axis (F#683 reuse).

Pre-registered KCs (canonical DB text — do not edit):
  K#1877 target: Augmented training produces adapter with > 3pp better behavioral
                 quality than non-augmented  (kill if TRUE — augmentation helps on axis)
  K#1878 proxy:  Augmented training cos-sim variance > 0.10 (unstable)
                 (paired diagnostic per F#666)  (kill if TRUE — training instability)

Skills invoked: /mlx-dev (documented in MATH.md §0). /fast-mlx pending for the _impl
follow-up (where the training loops land).

Runtime structure (two parallel arms, identical protocol except training-data
presentation):
  Phase 0     — Reuse F#683 politeness axis corpus (1× original arm)
  Phase 0.5   — Generate 5× rephrases at temperature 1.2 + semantic-equivalence QA
  Phase A     — 26B teacher attention capture on both corpora (attn_output per layer)
  Phase B_base— cos-loss student training on 1× original (non-augmented arm)
  Phase B_aug — cos-loss student training on 5× rephrased (augmented arm)
  Phase C     — K1878 per-layer cos-sim variance over last 200 steps of augmented arm
  Phase D     — K1877 blind-paired behavioral-quality judge (both arms, F#683 rubric)

Full budget: ~12–15 h on M5 Pro 48 GB (rephrase generation + QA gate + 26B teacher
residency + two Hedgehog training jobs at 800 steps each + paired-judge on 50 held-
out pairs + eval). Exceeds single-iteration researcher cap (30 min / 40 tool calls
per guardrail 1009). This scaffold supports SMOKE_TEST=1 and full runs; Phase
0/0.5/A/B_base/B_aug/C/D raise NotImplementedError pending a dedicated `_impl` run.

F#702 hygiene-patch PROVISIONAL is applicable (DB row shipped with 3 hygiene defects).
mem-impossibility-f666pure-saturation-implies-f702-unavailable does NOT fire because
K1877 is a target KC (not F#666-pure).
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
REPHRASE_MODEL = TEACHER_MODEL  # A2: rephraser = teacher, no π_Polite context

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 40 if IS_SMOKE else 200
N_HELDOUT = 10 if IS_SMOKE else 50
N_STEPS = 50 if IS_SMOKE else 800
SEED = 42

ADAPTER_RANK = 8
ADAPTER_TARGETS = ("v_proj", "o_proj")  # F#627
LORA_SCALE = 6.0                        # <= 8 per F#328/F#330
BATCH_SIZE = 1
SEQLEN = 2048
ENABLE_THINKING = True                  # F#614/F#536

# Data-augmentation locked hyperparameters (A3/A8)
REPHRASE_DEPTH = 5                      # A3: Wei 2024 optimal median
REPHRASE_TEMPERATURE = 1.2              # Locked per experiment claim
REPHRASE_SEMANTIC_FLOOR = 0.7           # A8: drop-below threshold
REPHRASE_DRIFT_ABORT_RATE = 0.20        # A8: fail-abort if > 20% of rephrases drift

# Axis locked at politeness (F#683 sibling; mature teacher-capture; variance reduction)
AXIS_NAME = "politeness"
AXIS_SIBLING_FINDING = "F#683"

# KCs
K1877_TARGET_DELTA_PP = 3.0             # Δ > +3pp (augmented − non-augmented)
K1878_PROXY_VARIANCE_THRESHOLD = 0.10   # variance > 0.10 over last 200 steps


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
# Phase 0: Dataset reuse (F#683 politeness axis)
# ─────────────────────────────────────────────

def load_politeness_axis_corpus(data_dir: Path, n_train: int, n_heldout: int) -> dict:
    """Reuse F#683 politeness axis corpus + held-out eval slice. Non-augmented arm
    trains on N_TRAIN = 200 base pairs; augmented arm trains on N_TRAIN × 5 = 1000
    (Q_rephrased, A) pairs (after Phase 0.5 QA gate).

    NOT YET IMPLEMENTED. Requires F#683 _impl to have landed on disk.
    """
    raise NotImplementedError(
        "Phase 0 depends on F#683 politeness axis corpus existing on disk. "
        "F#683 is still PROVISIONAL (no _impl landed yet). Deferred to _impl follow-up."
    )


# ─────────────────────────────────────────────
# Phase 0.5: Prompt rephrasing + semantic-equivalence QA gate
# ─────────────────────────────────────────────

def generate_rephrases(corpus: dict, rephrase_model: str, depth: int,
                       temperature: float, rephrase_dir: Path) -> dict:
    """Generate `depth` rephrases per Q at `temperature` using 26B Gemma 4 (no
    π_Polite context — axis-neutral rephrase). Run Phase 0.5 semantic-equivalence
    auto-judge; discard rephrases below REPHRASE_SEMANTIC_FLOOR; fail-abort if
    drift-rate > REPHRASE_DRIFT_ABORT_RATE.

    NOT YET IMPLEMENTED. Requires:
      - 26B teacher residency (cache miss → ~14 GB download).
      - generate() loop with temperature=1.2 per original Q.
      - Semantic-similarity judge (sentence-transformer or model-based).
      - Drift-rate accounting; fail-abort path on drift > 20%.
    """
    raise NotImplementedError(
        "Rephrase generation requires 26B teacher residency and a generate() loop; "
        "deferred to _impl follow-up."
    )


# ─────────────────────────────────────────────
# Phase A: Teacher attention capture (both corpora)
# ─────────────────────────────────────────────

def capture_teacher_attention(data_dir: Path, capture_dir: Path,
                              corpus_name: str) -> dict:
    """Capture teacher {layer_idx: attn_output} for all 42 layers on every (Q, A)
    pair with π_Polite in context. Called TWICE: once on original corpus (for
    B_base), once on rephrased corpus (for B_aug). Same π_Polite context both
    times — only the Q varies.

    NOT YET IMPLEMENTED. Requires 26B teacher residency + sequential-phase eviction.
    """
    raise NotImplementedError(
        f"Teacher attention capture for corpus={corpus_name} requires 26B model "
        "residency on 48 GB M5 Pro. See F#673."
    )


# ─────────────────────────────────────────────
# Phase B_base: non-augmented student (baseline arm)
# ─────────────────────────────────────────────

def train_student_non_augmented(data_dir: Path, capture_dir: Path,
                                adapter_path: Path) -> dict:
    """Student rank-8 LoRA on (v_proj, o_proj) with per-layer cos-sim loss on
    attention outputs: L = mean_l (1 − cos(A_t_l, A_s_l)). Trained on N_TRAIN = 200
    original (Q, A) pairs for 800 steps (4 epochs).

    NOT YET IMPLEMENTED. Requires custom MLX training loop with per-layer hooks.
    """
    raise NotImplementedError(
        "Non-augmented student training loop requires custom MLX code. "
        "Deferred to _impl follow-up."
    )


# ─────────────────────────────────────────────
# Phase B_aug: augmented student (ablation arm)
# ─────────────────────────────────────────────

def train_student_augmented(data_dir: Path, capture_dir: Path,
                            adapter_path: Path) -> dict:
    """Student rank-8 LoRA on (v_proj, o_proj) with per-layer cos-sim loss on
    attention outputs — IDENTICAL protocol to train_student_non_augmented EXCEPT
    trained on N_TRAIN × 5 = 1000 (Q_rephrased, A) pairs for 800 steps (0.8 epochs).
    Same seed, same optimizer state-init, same rank/scale/targets. ONLY training-
    data presentation differs.

    During training, log per-layer cos-sim every step; persist to disk for K1878
    variance computation in Phase C.

    NOT YET IMPLEMENTED. Requires custom MLX training loop.
    """
    raise NotImplementedError(
        "Augmented student training loop requires custom MLX code with per-step "
        "cos-sim logging. Deferred to _impl follow-up."
    )


# ─────────────────────────────────────────────
# Phase C: K1878 cos-sim variance (augmented arm)
# ─────────────────────────────────────────────

def measure_k1878_cos_sim_variance(train_log: Path) -> float:
    """K1878: variance of per-layer cos-sim across last 200 training batches of
    the augmented arm. Returns scalar variance over last-25%-of-training cos-sim
    time series (batch-wise aggregated per layer, then layer-mean).

    Kill condition: variance > 0.10 strictly (augmentation adds measurable
    training instability).

    NOT YET IMPLEMENTED. Requires Phase B_aug cos-sim per-step log.
    """
    raise NotImplementedError(
        "K1878 depends on Phase B_aug cos-sim per-step log existing."
    )


# ─────────────────────────────────────────────
# Phase D: K1877 behavioral-quality delta (both arms)
# ─────────────────────────────────────────────

def measure_k1877_behavioral_quality_delta(base_adapter: Path, aug_adapter: Path,
                                           held_out: list) -> float:
    """K1877: blind-paired politeness-axis auto-judge on 50 held-out pairs.
    Returns Δ = mean(score_augmented − score_non_augmented) on the 0–100
    rescaled rubric (F#683).

    Kill condition: Δ > +3 pp strictly (augmented meaningfully beats non-augmented).

    NOT YET IMPLEMENTED. Requires both adapters existing and F#683 rubric + held-
    out set on disk.
    """
    raise NotImplementedError(
        "K1877 judge eval depends on both adapters existing and F#683 rubric + "
        "held-out set being prepared."
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

    print("Hedgehog data-augmentation ablation: 1× vs 5× rephrased (politeness axis)",
          flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_HELDOUT={N_HELDOUT}, "
          f"N_STEPS={N_STEPS}", flush=True)
    print(f"mlx-lm version: {mlx_lm_version}", flush=True)
    print(f"Rephrase depth: {REPHRASE_DEPTH}, temperature: {REPHRASE_TEMPERATURE}",
          flush=True)
    print(f"Semantic floor: {REPHRASE_SEMANTIC_FLOOR}, drift abort: "
          f"{REPHRASE_DRIFT_ABORT_RATE}", flush=True)
    print(f"Axis: {AXIS_NAME} (sibling {AXIS_SIBLING_FINDING})", flush=True)

    data_dir = EXPERIMENT_DIR / "data" / "politeness_qa_pairs"
    rephrase_dir = EXPERIMENT_DIR / "data" / "politeness_rephrased_5x"
    capture_dir_base = EXPERIMENT_DIR / "teacher_capture_base"
    capture_dir_aug = EXPERIMENT_DIR / "teacher_capture_aug"
    adapters_dir = EXPERIMENT_DIR / "adapters"
    base_adapter = adapters_dir / "hedgehog_cos_polite_r8_base"
    aug_adapter = adapters_dir / "hedgehog_cos_polite_r8_aug5x"
    train_log_aug = adapters_dir / "hedgehog_cos_polite_r8_aug5x_cossim_log.jsonl"

    results = {
        "is_smoke": IS_SMOKE,
        "n_train_base": N_TRAIN,
        "n_train_aug": N_TRAIN * REPHRASE_DEPTH,
        "n_heldout": N_HELDOUT,
        "n_steps": N_STEPS,
        "student_model": STUDENT_MODEL,
        "teacher_model": TEACHER_MODEL,
        "rephrase_model": REPHRASE_MODEL,
        "mlx_lm_version": mlx_lm_version,
        "adapter_rank": ADAPTER_RANK,
        "adapter_targets": list(ADAPTER_TARGETS),
        "lora_scale": LORA_SCALE,
        "seqlen": SEQLEN,
        "enable_thinking": ENABLE_THINKING,
        "axis": AXIS_NAME,
        "axis_sibling_finding": AXIS_SIBLING_FINDING,
        "rephrase_depth": REPHRASE_DEPTH,
        "rephrase_temperature": REPHRASE_TEMPERATURE,
        "rephrase_semantic_floor": REPHRASE_SEMANTIC_FLOOR,
        "rephrase_drift_abort_rate": REPHRASE_DRIFT_ABORT_RATE,
        "k1877_target_delta_pp": K1877_TARGET_DELTA_PP,
        "k1878_proxy_variance_threshold": K1878_PROXY_VARIANCE_THRESHOLD,
        "phase_0_corpus_reuse": None,
        "phase_0_5_rephrase_generation": None,
        "phase_a_teacher_capture_base": None,
        "phase_a_teacher_capture_aug": None,
        "phase_b_base_train": None,
        "phase_b_aug_train": None,
        "phase_c_k1878": {},
        "phase_d_k1877": {},
        "kc": {
            "K1877_augmented_behavioral_quality_gt_non_augmented_by_3pp": "untested",
            "K1878_augmented_cos_sim_variance_gt_0_10": "untested",
        },
        "verdict": "PROVISIONAL",
        "all_pass": False,
        "sub_type": "data-augmentation-ablation",
        "super_family": "hedgehog-ablation",
        "super_family_sub_types": [
            "axis-extension",
            "loss-variant-ablation",
            "layer-selection-ablation",
            "hyperparameter-ablation",
            "data-augmentation-ablation",
        ],
        "sub_type_instance": 1,
        "super_family_total_instances": 12,
        "provisional_number_in_pile": 8,
        "blockers": [],
    }

    # ── Phase 0: corpus reuse ───────────────────────────────
    print("\n=== Phase 0: Politeness axis corpus reuse (F#683 precedent) ===",
          flush=True)
    try:
        results["phase_0_corpus_reuse"] = load_politeness_axis_corpus(
            data_dir, N_TRAIN, N_HELDOUT)
    except NotImplementedError as exc:
        results["phase_0_corpus_reuse"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase 0 not implemented: {exc}")

    # ── Phase 0.5: rephrase generation + QA gate ────────────
    try:
        results["phase_0_5_rephrase_generation"] = generate_rephrases(
            corpus={}, rephrase_model=REPHRASE_MODEL, depth=REPHRASE_DEPTH,
            temperature=REPHRASE_TEMPERATURE, rephrase_dir=rephrase_dir)
    except NotImplementedError as exc:
        results["phase_0_5_rephrase_generation"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase 0.5 not implemented: {exc}")

    # ── Phase A: teacher attention capture (both corpora) ───
    try:
        results["phase_a_teacher_capture_base"] = capture_teacher_attention(
            data_dir, capture_dir_base, corpus_name="base")
    except NotImplementedError as exc:
        results["phase_a_teacher_capture_base"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase A base not implemented: {exc}")

    try:
        results["phase_a_teacher_capture_aug"] = capture_teacher_attention(
            rephrase_dir, capture_dir_aug, corpus_name="aug")
    except NotImplementedError as exc:
        results["phase_a_teacher_capture_aug"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase A aug not implemented: {exc}")

    # ── Phase B_base: non-augmented student ─────────────────
    try:
        results["phase_b_base_train"] = train_student_non_augmented(
            data_dir, capture_dir_base, base_adapter)
    except NotImplementedError as exc:
        results["phase_b_base_train"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase B_base not implemented: {exc}")

    # ── Phase B_aug: augmented student ──────────────────────
    try:
        results["phase_b_aug_train"] = train_student_augmented(
            rephrase_dir, capture_dir_aug, aug_adapter)
    except NotImplementedError as exc:
        results["phase_b_aug_train"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase B_aug not implemented: {exc}")

    # ── Phase C: K1878 cos-sim variance ─────────────────────
    if train_log_aug.exists():
        try:
            k1878 = measure_k1878_cos_sim_variance(train_log_aug)
            results["phase_c_k1878"]["K1878_cos_sim_variance"] = round(float(k1878), 4)
        except NotImplementedError as exc:
            results["blockers"].append(f"K1878 not implemented: {exc}")
    else:
        results["blockers"].append(
            "Phase C skipped: augmented-arm cos-sim log not produced (Phase B_aug "
            "blocker)"
        )

    # ── Phase D: K1877 behavioral quality delta ─────────────
    if base_adapter.exists() and aug_adapter.exists():
        try:
            k1877 = measure_k1877_behavioral_quality_delta(
                base_adapter, aug_adapter, held_out=[])
            results["phase_d_k1877"]["K1877_behavioral_delta_pp"] = round(
                float(k1877), 2)
        except NotImplementedError as exc:
            results["blockers"].append(f"K1877 not implemented: {exc}")
    else:
        results["blockers"].append(
            "Phase D skipped: one or both adapters not produced (Phase B blockers)"
        )

    # ── KC resolution ──────────────────────────────────────
    # With Phase 0/0.5/A/B_base/B_aug not implemented, both KCs remain "untested".
    # When the _impl follow-up lands, populate kc dict from phase_c_k1878 + phase_d_k1877
    # using the F#666 verdict matrix (MATH.md §4).

    results["total_time_s"] = round(time.time() - t_start, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Verdict: {results['verdict']}", flush=True)
    print(f"KCs: {json.dumps(results['kc'])}", flush=True)
    print(f"Sub-type: {results['sub_type']} (#{results['sub_type_instance']} in "
          f"sub-type; #{results['super_family_total_instances']} in super-family)",
          flush=True)
    print(f"Blockers ({len(results['blockers'])}):", flush=True)
    for b in results["blockers"]:
        print(f"  - {b}", flush=True)
    print(f"Total time: {results['total_time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
