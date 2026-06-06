#!/usr/bin/env python3
"""
exp_hedgehog_behavior_adapter_politeness — Hedgehog per-layer cos-sim distillation.

Pre-registered KCs (canonical DB text — do not edit):
  K#1782 structural: mean per-layer cos(teacher_attn_out, student_attn_out) > 0.85
                     on held-out neutral prompts at end of training (behavior captured)
  K#1783 target:     auto-judge politeness score on neutral prompts increases by >= 20%
                     (absolute) vs base Gemma 4 E4B, n>=100 prompts  (pair K1 per F#666)
  K#1784 target:     factual QA (MMLU subset) and code accuracy (HumanEval) each drop
                     < 3pp vs base — adapter is behavior, not replacement
  K#1785 ablation:   removing the polite system prompt from teacher forward pass during
                     training causes K2 target to regress by >= 10pp

Skills invoked: /mlx-dev + /fast-mlx (documented in MATH.md §0).

Runtime structure:
  Phase 0 — neutral prompt dataset prep (UltraChat-style instructions, not emotionally loaded)
  Phase A — teacher attention-output capture under POLITE_SYSTEM_PROMPT (frozen base)
  Phase B — student training: rank-8 LoRA on v_proj/o_proj, per-layer cos-sim loss (custom MLX loop)
  Phase C — K1 structural cos + K2 politeness-judge eval (API-dependent)
  Phase D — K3 non-interference: MMLU subset + HumanEval
  Phase E — K4 ablation: retrain with NEUTRAL_SYSTEM_PROMPT teacher, re-measure K2

Full budget: ~3-5h on M5 Pro 48GB (2 training jobs at 500 steps + MMLU/HumanEval evals + judge API).
Exceeds single-iteration researcher cap (30 min / 40 tool calls per guardrail 1009).
This scaffold supports SMOKE_TEST=1 and full runs; Phase B/E are NotImplementedError pending
dedicated implementation iteration (antipattern-'novel-mechanism single-iteration scope').
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 50 if IS_SMOKE else 1000
N_HELDOUT = 10 if IS_SMOKE else 100
N_JUDGE = 10 if IS_SMOKE else 100
N_STEPS = 50 if IS_SMOKE else 500
SEED = 42

ADAPTER_RANK = 8
ADAPTER_TARGETS = ("v_proj", "o_proj")  # F#627
LORA_SCALE = 6.0                        # <= 8 per F#328/F#330
BATCH_SIZE = 2
SEQLEN = 512

POLITE_SYSTEM_PROMPT = (
    "You are a deeply respectful, warm, and patient assistant. "
    "Phrase every response with courtesy, acknowledge the user's effort, "
    "and avoid any language that could feel dismissive or abrupt."
)
NEUTRAL_SYSTEM_PROMPT = ""  # K4 ablation replaces the teacher prompt with this


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
# Phase 0: Dataset
# ─────────────────────────────────────────────

def prepare_neutral_prompts(data_dir: Path, n_train: int, n_heldout: int) -> dict:
    """Curate a neutral-instruction prompt set (UltraChat filtered to non-emotionally-loaded).

    NOT YET IMPLEMENTED. Requires curation pipeline: load UltraChat, filter out prompts
    that already contain tone/etiquette cues (regex on politeness-markers), split train/held-out.
    Emotional filtering is load-bearing for K2 — judge must see prompts where politeness can
    differentiate, without the base already being polite by prompt osmosis.
    """
    raise NotImplementedError(
        "Phase 0 dataset curation not yet implemented. "
        "Requires UltraChat-filtered neutral prompts (drop politeness-marker regex matches). "
        "See PAPER.md §Measurement blockers."
    )


# ─────────────────────────────────────────────
# Phase A + B: Hedgehog custom training loop
# ─────────────────────────────────────────────
#
# NOT-YET-IMPLEMENTED. Requires:
# 1. Loading MODEL_ID via mlx_lm.load (frozen teacher).
# 2. Attaching rank-8 LoRA on v_proj + o_proj (mlx_lm.tuner.lora.LoRALinear.from_linear).
# 3. Hook on model.layers[i].self_attn output for all i in 0..L-1 to capture A_l(x).
#    Gemma 4 E4B has 42 transformer blocks; self_attn attribute path must be verified against
#    the installed mlx-lm version (attribute monkeypatching on __call__ is fragile — prefer
#    `nn.quantized.QuantizedLinear`-aware hook registered through a Module subclass).
# 4. Two forward passes per batch:
#       - teacher pass: frozen base, input = POLITE_SYSTEM_PROMPT ⊕ x
#       - student pass: base + LoRA, input = NEUTRAL_SYSTEM_PROMPT ⊕ x
#    Both on the SAME x; capture {layer_idx: attn_out_tensor} for each.
# 5. Loss = mean_l (1 − cos(teacher_attn[l], student_attn[l])).
#    Cos computed per-token then mean-pooled over tokens per layer, then mean over layers.
#    Detach teacher side (no gradient through teacher pass).
# 6. Gradient step: nn.value_and_grad(student_model, loss_fn); mlx.optimizers.AdamW.
#    mx.eval(student_model.parameters(), loss) at step boundary; mx.clear_cache() between batches.
# 7. Save LoRA weights compatible with mlx_lm adapter loading.
#
# Scope-preservation: do NOT silently fall back to cross-entropy SFT if cos-sim training fails.
# The KCs measure behavior under cos-sim distillation; substituting CE invalidates K1–K4.

def train_hedgehog_student(data_dir: Path, adapter_path: Path,
                            teacher_system_prompt: str) -> dict:
    """Phase B: train rank-8 LoRA student with per-layer cos-sim loss.

    NOT YET IMPLEMENTED. Returns NotImplementedError with structured marker so the pipeline
    can record a PROVISIONAL verdict without silent degradation.

    When landed: Phase E (K4 ablation) calls this with teacher_system_prompt=NEUTRAL_SYSTEM_PROMPT.
    """
    raise NotImplementedError(
        "Hedgehog cos-sim distillation training loop not yet implemented. "
        "Requires custom MLX: per-layer attention-output hooks (42 layers Gemma 4 E4B) on "
        "teacher + student forward passes, per-layer cos-sim loss, nn.value_and_grad + AdamW. "
        "See PAPER.md §Measurement blockers."
    )


# ─────────────────────────────────────────────
# Phase C: K1 structural cos + K2 politeness judge
# ─────────────────────────────────────────────

def measure_k1_structural_cos(adapter_path: Path, held_out_prompts: list) -> float:
    """K1: mean per-layer cos(A_l_teacher, A_l_student) on held-out neutral prompts.

    NOT YET IMPLEMENTED. Requires the same teacher/student forward-pass instrumentation as
    Phase B, applied in eval mode on held_out_prompts.
    """
    raise NotImplementedError("K1 measurement depends on Phase A/B completing.")


def measure_k2_politeness_judge(adapter_path: Path, judge_prompts: list) -> float:
    """K2: auto-judge politeness Δ = score(student) − score(base) on neutral prompts.

    NOT YET IMPLEMENTED. Requires:
      - generate(base, prompt) and generate(base+adapter, prompt) for each prompt
      - paired-compare via Claude 3.7 or GPT-4 rubric (0-100 politeness)
      - report mean(score_student − score_base) or % preference
    """
    raise NotImplementedError("K2 judge eval depends on adapter existing.")


# ─────────────────────────────────────────────
# Phase D: K3 non-interference
# ─────────────────────────────────────────────

def measure_k3_non_interference(adapter_path: Path) -> dict:
    """K3a: MMLU subset drop < 3pp. K3b: HumanEval pass@1 drop < 3pp.

    NOT YET IMPLEMENTED. Hooks into existing MMLU + HumanEval evaluators
    (see experiments/models/exp_bench_mmlu_pro* and exp_bench_humaneval*).
    """
    raise NotImplementedError("K3 non-interference eval depends on adapter existing.")


# ─────────────────────────────────────────────
# Phase E: K4 ablation
# ─────────────────────────────────────────────

def measure_k4_teacher_prompt_ablation(data_dir: Path, adapters_dir: Path,
                                        judge_prompts: list) -> float:
    """K4: retrain adapter with teacher fed NEUTRAL_SYSTEM_PROMPT (no polite cue);
    re-measure K2 on same prompts. Target: ≥10pp regression.

    Calls train_hedgehog_student(..., teacher_system_prompt=NEUTRAL_SYSTEM_PROMPT)
    then measure_k2_politeness_judge on the ablation adapter.
    """
    raise NotImplementedError("K4 ablation depends on Phase B training loop.")


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

    print(f"Hedgehog behavior-adapter distillation (politeness, Gemma 4 E4B)", flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_HELDOUT={N_HELDOUT}, "
          f"N_JUDGE={N_JUDGE}, N_STEPS={N_STEPS}", flush=True)
    print(f"mlx-lm version: {mlx_lm_version}", flush=True)

    data_dir = EXPERIMENT_DIR / "data" / "neutral"
    adapters_dir = EXPERIMENT_DIR / "adapters"
    student_adapter = adapters_dir / "hedgehog_polite_r8"
    ablation_adapter = adapters_dir / "hedgehog_polite_r8_ablation"

    results = {
        "is_smoke": IS_SMOKE,
        "n_train": N_TRAIN,
        "n_heldout": N_HELDOUT,
        "n_judge": N_JUDGE,
        "n_steps": N_STEPS,
        "model_id": MODEL_ID,
        "mlx_lm_version": mlx_lm_version,
        "adapter_rank": ADAPTER_RANK,
        "adapter_targets": list(ADAPTER_TARGETS),
        "lora_scale": LORA_SCALE,
        "seqlen": SEQLEN,
        "phase_0_dataset": None,
        "phase_b_student_train": None,
        "phase_c_k1_k2": {},
        "phase_d_k3": {},
        "phase_e_k4_ablation": None,
        "kc": {
            "K1782_per_layer_cos_gt_0_85": "untested",
            "K1783_politeness_judge_delta_ge_20pp": "untested",
            "K1784a_mmlu_drop_lt_3pp": "untested",
            "K1784b_humaneval_drop_lt_3pp": "untested",
            "K1785_ablation_regression_ge_10pp": "untested",
        },
        "verdict": "PROVISIONAL",
        "all_pass": False,
        "blockers": [],
    }

    # ── Phase 0: dataset ────────────────────────────────────
    print("\n=== Phase 0: Neutral prompt dataset ===", flush=True)
    try:
        results["phase_0_dataset"] = prepare_neutral_prompts(data_dir, N_TRAIN, N_HELDOUT)
    except NotImplementedError as exc:
        results["phase_0_dataset"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase 0 not implemented: {exc}")
    except Exception as exc:
        results["phase_0_dataset"] = {"error": str(exc)}
        results["blockers"].append(f"Phase 0 failed: {exc}")

    # ── Phase B: Hedgehog training ──────────────────────────
    try:
        phase_b = train_hedgehog_student(data_dir, student_adapter, POLITE_SYSTEM_PROMPT)
        results["phase_b_student_train"] = phase_b
    except NotImplementedError as exc:
        results["phase_b_student_train"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase B not implemented: {exc}")
    except Exception as exc:
        results["phase_b_student_train"] = {"error": str(exc)}
        results["blockers"].append(f"Phase B failed: {exc}")

    # ── Phase C: K1 structural + K2 judge ──────────────────
    if student_adapter.exists():
        try:
            k1 = measure_k1_structural_cos(student_adapter, held_out_prompts=[])
            results["phase_c_k1_k2"]["K1_cos"] = round(float(k1), 4)
        except NotImplementedError as exc:
            results["blockers"].append(f"K1 not implemented: {exc}")
        except Exception as exc:
            results["blockers"].append(f"K1 failed: {exc}")

        try:
            k2 = measure_k2_politeness_judge(student_adapter, judge_prompts=[])
            results["phase_c_k1_k2"]["K2_judge_delta"] = round(float(k2), 2)
        except NotImplementedError as exc:
            results["blockers"].append(f"K2 not implemented: {exc}")
        except Exception as exc:
            results["blockers"].append(f"K2 failed: {exc}")
    else:
        results["blockers"].append("Phase C skipped: student adapter not produced (Phase B blocker)")

    # ── Phase D: K3 non-interference ───────────────────────
    if student_adapter.exists():
        try:
            k3 = measure_k3_non_interference(student_adapter)
            results["phase_d_k3"] = k3
        except NotImplementedError as exc:
            results["blockers"].append(f"K3 not implemented: {exc}")
        except Exception as exc:
            results["blockers"].append(f"K3 failed: {exc}")
    else:
        results["blockers"].append("Phase D skipped: student adapter not produced (Phase B blocker)")

    # ── Phase E: K4 teacher-prompt ablation ────────────────
    try:
        k4 = measure_k4_teacher_prompt_ablation(data_dir, adapters_dir, judge_prompts=[])
        results["phase_e_k4_ablation"] = {"regression_pp": round(float(k4), 2)}
    except NotImplementedError as exc:
        results["phase_e_k4_ablation"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"K4 ablation not implemented: {exc}")
    except Exception as exc:
        results["phase_e_k4_ablation"] = {"error": str(exc)}
        results["blockers"].append(f"K4 ablation failed: {exc}")

    # ── KC resolution ──────────────────────────────────────
    # With Phase B not implemented, all five KCs remain "untested".
    # When Phase B lands, populate kc dict from phase_c_k1_k2 + phase_d_k3 + phase_e_k4_ablation.

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
