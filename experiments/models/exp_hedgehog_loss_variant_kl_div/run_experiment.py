#!/usr/bin/env python3
"""
exp_hedgehog_loss_variant_kl_div — Loss-variant ablation comparing Hedgehog cos-sim
distillation against KL-divergence-on-attention-distributions on the politeness axis.

Pre-registered KCs (canonical DB text — do not edit):
  K#1870 proxy:  KL-div loss produces adapter with cos-sim < 0.70 vs cos-loss adapter
                 cos-sim > 0.80  (kill if TRUE — loss-choice matters on cos-sim)
  K#1871 target: KL-div adapter behavioral quality worse than cos-sim adapter by > 3pp
                 (pair K1870 per F#666)  (kill if TRUE — loss-choice matters behaviorally)

Skills invoked: /mlx-dev (documented in MATH.md §0). /fast-mlx pending for the _impl
follow-up (where the training loops land).

Runtime structure (two parallel arms, identical protocol except loss function):
  Phase 0     — Reuse politeness axis corpus (F#683 precedent; no new curation)
  Phase A     — 26B teacher attention capture (attn_output + attn_weights per layer)
  Phase B_cos — cos-loss student training (baseline arm)
  Phase B_kl  — KL-div student training (ablation arm; forward-KL locked)
  Phase C     — K1870 student-to-teacher per-layer cos-sim (both arms)
  Phase D     — K1871 blind-paired behavioral-quality judge (both arms)

Full budget: ~10 h on M5 Pro 48 GB (two Hedgehog training jobs at 800 steps each +
26B teacher residency + paired-judge on 50 held-out pairs + eval). Exceeds single-
iteration researcher cap (30 min / 40 tool calls per guardrail 1009). This scaffold
supports SMOKE_TEST=1 and full runs; Phase 0/A/B_cos/B_kl/C/D raise NotImplementedError
pending a dedicated `_impl` iteration.

F#702 hygiene-patch PROVISIONAL is applicable (DB row shipped with 3 hygiene defects).
mem-impossibility-f666pure-saturation-implies-f702-unavailable does NOT fire because
K1871 is a target KC (not F#666-pure).
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
N_STEPS = 50 if IS_SMOKE else 800
SEED = 42

ADAPTER_RANK = 8
ADAPTER_TARGETS = ("v_proj", "o_proj")  # F#627
LORA_SCALE = 6.0                        # <= 8 per F#328/F#330
BATCH_SIZE = 1
SEQLEN = 2048
ENABLE_THINKING = True                  # F#614/F#536

# KL-div numerical floor (prevents NaN on padded rows; see MATH.md §1 tertiary-failure)
KL_EPSILON = 1e-6
# KL direction locked at forward-KL (teacher || student); see MATH.md §1 / A7
KL_DIRECTION = "forward"

# Axis locked at politeness (F#683 sibling; mature teacher-capture; variance reduction)
AXIS_NAME = "politeness"
AXIS_SIBLING_FINDING = "F#683"


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
    """Reuse F#683 politeness axis corpus + held-out eval slice. No new curation
    needed — axis-locking is the variance-reduction choice for the ablation.

    NOT YET IMPLEMENTED IN THIS SCAFFOLD. Requires:
      - F#683 teacher-with-context (π_Polite) training corpus (training portion).
      - F#683 held-out axis eval slice (n=50 blind-paired held-out).
      - K1870 also needs a held-out cos-sim eval slice (could reuse behavioral
        held-out set; cos-sim is a per-layer aggregate over the eval forward pass).
      - Both variants train on IDENTICAL (Q, A) pairs — random splits, no per-arm
        resampling.

    The _impl follow-up (exp_hedgehog_loss_variant_kl_div_impl) should implement
    corpus loading by reusing F#683's curated data directory (once that experiment's
    _impl has landed).
    """
    raise NotImplementedError(
        "Phase 0 depends on F#683 politeness axis corpus existing on disk. "
        "F#683 is still PROVISIONAL (no _impl landed yet). Deferred to _impl follow-up."
    )


# ─────────────────────────────────────────────
# Phase A: Teacher attention capture
# ─────────────────────────────────────────────

def capture_teacher_attention(data_dir: Path, capture_dir: Path) -> dict:
    """Capture teacher {layer_idx: (attn_output, attn_weights)} for all 42 layers on
    every training (Q, A) pair with π_Polite in context. Both tensors needed:
    cos-sim variant uses attn_output (post-projection); KL-div variant uses
    attn_weights (post-softmax distributions).

    NOT YET IMPLEMENTED. Requires:
      - Load 26B teacher with π_Polite context.
      - Forward pass per training pair; hook attention modules.
      - Store tensors to capture_dir for offline streaming during student training.
      - Sequential-phase eviction on 48 GB per F#673: teacher forward → save → evict
        → student load.
      - Peak memory load-bearing: attn_weights tensor (B, H, S, S) is larger than
        attn_output (B, H, S, D). Use mixed-precision storage or chunked save.
    """
    raise NotImplementedError(
        "Teacher attention capture requires 26B model residency on 48 GB M5 Pro "
        "with sequential-phase eviction and offline precompute. See F#673."
    )


# ─────────────────────────────────────────────
# Phase B_cos: cos-loss student (baseline arm)
# ─────────────────────────────────────────────

def train_student_cos_loss(data_dir: Path, capture_dir: Path,
                           adapter_path: Path) -> dict:
    """Student rank-8 LoRA on (v_proj, o_proj) with per-layer cos-sim loss on
    attention outputs: L = mean_l (1 − cos(A_t_l, A_s_l)).

    IDENTICAL hyperparameters to train_student_kl_loss except for the loss function.

    NOT YET IMPLEMENTED. Requires:
      - Custom MLX training loop with per-layer attention-output hooks.
      - nn.value_and_grad(student, loss_fn) functional gradients (NOT .backward()).
      - AdamW, 800 steps, mx.eval + mx.clear_cache between batches (F#673).
      - Loads teacher attention-output traces from capture_dir per step.
    """
    raise NotImplementedError(
        "cos-loss student training loop requires custom MLX code with per-layer "
        "attention-output hooks. Deferred to _impl follow-up."
    )


# ─────────────────────────────────────────────
# Phase B_kl: KL-div student (ablation arm)
# ─────────────────────────────────────────────

def train_student_kl_loss(data_dir: Path, capture_dir: Path,
                          adapter_path: Path) -> dict:
    """Student rank-8 LoRA on (v_proj, o_proj) with per-layer forward-KL on
    post-softmax attention distributions:
      L = mean_l KL(softmax(Q_t K_t^T / √d + ε) || softmax(Q_s K_s^T / √d + ε)).

    KL direction locked at forward-KL (teacher || student); mode-seeking, analogous
    to teacher-forcing in SFT. ε = KL_EPSILON = 1e-6 prevents NaN on padded rows.

    IDENTICAL hyperparameters to train_student_cos_loss except for the loss function.

    NOT YET IMPLEMENTED. Requires:
      - Custom MLX training loop with per-layer attention-weights hooks.
      - Numerical stability: row-normalize softmax outputs, add ε, compute KL in
        log-space to avoid overflow.
      - nn.value_and_grad + AdamW, 800 steps, mx.eval + mx.clear_cache per F#673.
      - Loads teacher attention-weights traces from capture_dir per step.
      - Same random seed, same optimizer state-init, same (Q, A) pairs as cos-loss arm.
    """
    raise NotImplementedError(
        "KL-div student training loop requires custom MLX code with per-layer "
        "attention-weights hooks and forward-KL numerical stability. Deferred to "
        "_impl follow-up."
    )


# ─────────────────────────────────────────────
# Phase C: K1870 student-to-teacher cos-sim (both arms)
# ─────────────────────────────────────────────

def measure_k1870_cos_sim_both_arms(cos_adapter: Path, kl_adapter: Path,
                                    capture_dir: Path, eval_slice: list) -> dict:
    """K1870: per-layer student-to-teacher cos-sim for both arms on held-out eval
    slice. Returns {cos_arm: cos_sim_mean, kl_arm: cos_sim_mean}.

    Kill condition: cos-sim(KL) < 0.70 AND cos-sim(cos-loss) > 0.80 — BOTH must hold
    for K1870 KILL (cos-loss tautologically wins on cos-sim by construction; K1870
    tests whether KL also clears 0.70 as a "did training converge" signal).

    NOT YET IMPLEMENTED. Requires:
      - mlx_lm.load(STUDENT_MODEL, adapter_path=cos_adapter); forward pass on eval
        slice; capture student attention outputs per layer; compute cos-sim vs
        teacher traces from capture_dir.
      - Repeat for kl_adapter.
      - Mean over layers and eval pairs.
    """
    raise NotImplementedError(
        "K1870 depends on both cos-loss and KL-div adapters existing."
    )


# ─────────────────────────────────────────────
# Phase D: K1871 behavioral-quality judge (both arms)
# ─────────────────────────────────────────────

def measure_k1871_behavioral_quality_delta(cos_adapter: Path, kl_adapter: Path,
                                           held_out: list) -> float:
    """K1871: blind-paired politeness-axis auto-judge on 50 held-out pairs.
    Returns Δ = mean(score_cos_adapter − score_kl_adapter) on the 0-100 rescaled
    rubric.

    Kill condition: Δ > +3 pp strictly (cos-loss meaningfully beats KL behaviorally).

    NOT YET IMPLEMENTED. Requires:
      - generate(base + cos_adapter, format_prompt(Q)) per held-out pair.
      - generate(base + kl_adapter, format_prompt(Q)) per held-out pair.
      - F#683 politeness-axis rubric (register/indirection/tonal fidelity).
      - Blind-paired presentation to judge; swap order 50/50 to control for
        position bias.
      - Pinned judge (Gemma 4 E4B-as-judge or API judge).
    """
    raise NotImplementedError(
        "K1871 judge eval depends on both adapters existing and F#683 rubric + "
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

    print("Hedgehog loss-variant ablation: cos-sim vs KL-div (politeness axis)", flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_HELDOUT={N_HELDOUT}, N_STEPS={N_STEPS}",
          flush=True)
    print(f"mlx-lm version: {mlx_lm_version}", flush=True)
    print(f"KL direction: {KL_DIRECTION}, KL ε: {KL_EPSILON}", flush=True)
    print(f"Axis: {AXIS_NAME} (sibling {AXIS_SIBLING_FINDING})", flush=True)

    data_dir = EXPERIMENT_DIR / "data" / "politeness_qa_pairs"
    capture_dir = EXPERIMENT_DIR / "teacher_capture"
    adapters_dir = EXPERIMENT_DIR / "adapters"
    cos_adapter = adapters_dir / "hedgehog_cos_polite_r8"
    kl_adapter = adapters_dir / "hedgehog_kl_polite_r8"

    results = {
        "is_smoke": IS_SMOKE,
        "n_train": N_TRAIN,
        "n_heldout": N_HELDOUT,
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
        "axis_sibling_finding": AXIS_SIBLING_FINDING,
        "kl_direction": KL_DIRECTION,
        "kl_epsilon": KL_EPSILON,
        "phase_0_corpus_reuse": None,
        "phase_a_teacher_capture": None,
        "phase_b_cos_train": None,
        "phase_b_kl_train": None,
        "phase_c_k1870": {},
        "phase_d_k1871": {},
        "kc": {
            "K1870_kl_cos_sim_lt_070_and_cos_loss_gt_080": "untested",
            "K1871_cos_loss_behavioral_quality_gt_kl_by_3pp": "untested",
        },
        "verdict": "PROVISIONAL",
        "all_pass": False,
        "blockers": [],
    }

    # ── Phase 0: corpus reuse ───────────────────────────────
    print("\n=== Phase 0: Politeness axis corpus reuse (F#683 precedent) ===", flush=True)
    try:
        results["phase_0_corpus_reuse"] = load_politeness_axis_corpus(
            data_dir, N_TRAIN, N_HELDOUT)
    except NotImplementedError as exc:
        results["phase_0_corpus_reuse"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase 0 not implemented: {exc}")
    except Exception as exc:
        results["phase_0_corpus_reuse"] = {"error": str(exc)}
        results["blockers"].append(f"Phase 0 failed: {exc}")

    # ── Phase A: teacher attention capture ──────────────────
    try:
        results["phase_a_teacher_capture"] = capture_teacher_attention(
            data_dir, capture_dir)
    except NotImplementedError as exc:
        results["phase_a_teacher_capture"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase A not implemented: {exc}")
    except Exception as exc:
        results["phase_a_teacher_capture"] = {"error": str(exc)}
        results["blockers"].append(f"Phase A failed: {exc}")

    # ── Phase B_cos: cos-loss student ───────────────────────
    try:
        results["phase_b_cos_train"] = train_student_cos_loss(
            data_dir, capture_dir, cos_adapter)
    except NotImplementedError as exc:
        results["phase_b_cos_train"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase B_cos not implemented: {exc}")
    except Exception as exc:
        results["phase_b_cos_train"] = {"error": str(exc)}
        results["blockers"].append(f"Phase B_cos failed: {exc}")

    # ── Phase B_kl: KL-div student ──────────────────────────
    try:
        results["phase_b_kl_train"] = train_student_kl_loss(
            data_dir, capture_dir, kl_adapter)
    except NotImplementedError as exc:
        results["phase_b_kl_train"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase B_kl not implemented: {exc}")
    except Exception as exc:
        results["phase_b_kl_train"] = {"error": str(exc)}
        results["blockers"].append(f"Phase B_kl failed: {exc}")

    # ── Phase C: K1870 cos-sim head-to-head ─────────────────
    if cos_adapter.exists() and kl_adapter.exists():
        try:
            k1870 = measure_k1870_cos_sim_both_arms(
                cos_adapter, kl_adapter, capture_dir, eval_slice=[])
            results["phase_c_k1870"] = k1870
        except NotImplementedError as exc:
            results["blockers"].append(f"K1870 not implemented: {exc}")
        except Exception as exc:
            results["blockers"].append(f"K1870 failed: {exc}")
    else:
        results["blockers"].append(
            "Phase C skipped: one or both adapters not produced (Phase B blockers)"
        )

    # ── Phase D: K1871 behavioral quality ───────────────────
    if cos_adapter.exists() and kl_adapter.exists():
        try:
            k1871 = measure_k1871_behavioral_quality_delta(
                cos_adapter, kl_adapter, held_out=[])
            results["phase_d_k1871"]["K1871_behavioral_delta_pp"] = round(float(k1871), 2)
        except NotImplementedError as exc:
            results["blockers"].append(f"K1871 not implemented: {exc}")
        except Exception as exc:
            results["blockers"].append(f"K1871 failed: {exc}")
    else:
        results["blockers"].append(
            "Phase D skipped: one or both adapters not produced (Phase B blockers)"
        )

    # ── KC resolution ──────────────────────────────────────
    # With Phase 0/A/B_cos/B_kl not implemented, both KCs remain "untested".
    # When the _impl follow-up lands, populate kc dict from phase_c_k1870 + phase_d_k1871
    # using the F#666 verdict matrix (see MATH.md §4).

    results["total_time_s"] = round(time.time() - t_start, 1)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Verdict: {results['verdict']}", flush=True)
    print(f"KCs: {json.dumps(results['kc'])}", flush=True)
    print(f"Blockers ({len(results['blockers'])}):", flush=True)
    for b in results["blockers"]:
        print(f"  - {b}", flush=True)
    print(f"Total time: {results['total_time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
