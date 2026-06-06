#!/usr/bin/env python3
"""
exp_hedgehog_procedural_adapter_refactor — Hedgehog per-layer cos-sim distillation.

Pre-registered KCs (canonical DB text — do not edit):
  K#1786 structural: per-layer cos(teacher, student) > 0.80 on held-out refactor tasks
                     at end of training
  K#1787 target:     refactor quality (auto-judge on CodeRefactor-style benchmark) >=
                     baseline token-space LoRA trained on the same pairs at matched params
                     (pair K1 per F#666)
  K#1788 target:     HumanEval pass@1 drop < 3pp vs base Gemma 4 E4B (general code unchanged)
  K#1789 specificity: non-refactor code tasks (gen-from-spec) accuracy drop < 2pp
                     (adapter is procedural, not general-code replacement)

Skills invoked: /mlx-dev (documented in MATH.md §0). /fast-mlx pending for the _impl follow-up.

Runtime structure:
  Phase 0 — Fowler catalog + refactor pair curation (train/held-out split)
  Phase A — teacher attention-output capture: 26B Gemma variant + catalog-in-context + c_pre
  Phase B — student training: rank-8 LoRA on v_proj/o_proj, per-layer cos-sim loss
            (custom MLX training loop — NOT available via mlx_lm.lora CLI)
  Phase Baseline — same-data token-space LoRA at matched params (for K2 head-to-head)
  Phase C — K1 structural cos + K2 auto-judge refactor quality (judge API call per pair)
  Phase D — K3 HumanEval pass@1 + K4 non-refactor code-gen

Full budget: ~4-6h on M5 Pro 48GB (two training jobs at 800 steps + teacher at 26B residency
+ judge-API per pair + HumanEval). Exceeds single-iteration researcher cap (30 min / 40 tool
calls per guardrail 1009). This scaffold supports SMOKE_TEST=1 and full runs; Phase 0/B
/ Baseline / C / D are NotImplementedError pending dedicated implementation iteration
(antipattern `mem-antipattern-novel-mechanism-single-iteration-scope`).
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import mlx.core as mx

# M5 Pro 48GB: leave 8GB headroom; cap wired cache at 2GB per mlx-dev guidance.
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
SEQLEN = 1024
ENABLE_THINKING = True                  # F#614 — load-bearing on Gemma 4 reasoning


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

def prepare_refactor_pairs(data_dir: Path, n_train: int, n_heldout: int) -> dict:
    """Curate (refactor_name, explanation, c_pre, c_post, language) pairs.

    NOT YET IMPLEMENTED. Requires:
      - Fowler catalog entry texts (summaries of each refactor; reproducible under fair-use
        summarization) keyed by canonical refactor names (Extract Method, Rename Variable,
        Inline Variable, etc.).
      - LLM-generated c_pre / c_post pairs per entry (N per refactor * M refactors),
        validated by pytest that c_post passes the same tests as c_pre.
      - Train/held-out split stratified by refactor_name so held-out contains distinct
        instances of same refactor categories (tests generalization, not memorization).

    Load-bearing for K2: judge must see held-out c_pre and score student's output against
    teacher's c_post. Curation choices here shape what K1, K2, K3, K4 measure.
    """
    raise NotImplementedError(
        "Phase 0 Fowler refactor pair curation not yet implemented. "
        "Requires catalog-entry summaries + validated c_pre/c_post pairs with pytest "
        "equivalence checks. See PAPER.md §Measurement blockers."
    )


# ─────────────────────────────────────────────
# Phase A + B: Hedgehog custom training loop
# ─────────────────────────────────────────────
#
# NOT-YET-IMPLEMENTED. Requires:
# 1. Loading STUDENT_MODEL and TEACHER_MODEL via mlx_lm.load (both frozen except student LoRA).
#    On 48GB M5 Pro the 26B teacher + E4B student may not co-reside; sequential-phase pattern:
#      - teacher forward on batch, capture {layer_idx: attn_out}; mx.eval + mx.clear_cache
#      - evict teacher (del + gc.collect + mx.clear_cache); load student if not loaded
#      - student forward + loss + step; mx.eval(student_model.parameters(), loss)
#    Alternative: pre-compute teacher attn traces for all train prompts in an offline pass,
#    write to disk, stream during student training. Trades compute for peak memory.
# 2. Attach rank-8 LoRA on v_proj + o_proj of student via mlx_lm.tuner.lora.LoRALinear
#    .from_linear; freeze all other student params.
# 3. Hook on student.layers[i].self_attn output for all i in 0..L-1 to capture A_l.
#    Gemma 4 E4B: L = 42. Teacher attribute path must be verified against installed mlx-lm
#    version (monkeypatching __call__ is fragile — prefer a Module subclass that records
#    outputs).
# 4. Loss per batch:
#       L = mean_l (1 - cos(A_l_teacher(c_pre | catalog_entry), A_l_student(c_pre)))
#    Cos computed per-token then mean-pooled over tokens per layer, then mean over layers.
#    teacher-side: stop_gradient (no grad flow through 26B).
# 5. Gradient step: nn.value_and_grad(student_model, loss_fn); mlx.optimizers.AdamW.
#    mx.eval(student_model.parameters(), loss) at step boundary. mx.clear_cache() between
#    batches (especially after teacher forward — F#673).
# 6. Save LoRA weights compatible with mlx_lm adapter loading.
#
# Scope-preservation: do NOT silently fall back to cross-entropy SFT on c_post tokens. The
# claim is cos-sim distillation transfers *procedural* knowledge via attention routing;
# substituting CE would test a different hypothesis (surface imitation of c_post tokens).

def train_hedgehog_student(data_dir: Path, adapter_path: Path) -> dict:
    """Phase B: train rank-8 LoRA student with per-layer cos-sim loss vs 26B teacher.

    NOT YET IMPLEMENTED. Returns NotImplementedError with structured marker so the pipeline
    can record a PROVISIONAL verdict without silent degradation.
    """
    raise NotImplementedError(
        "Hedgehog cos-sim distillation training loop not yet implemented. "
        "Requires custom MLX: sequential-phase teacher/student forward passes (42 layers), "
        "per-layer attention-output hooks, per-layer cos-sim loss, nn.value_and_grad + AdamW. "
        "Teacher is 26B — peak memory is load-bearing. See PAPER.md §Measurement blockers."
    )


def train_token_space_lora_baseline(data_dir: Path, adapter_path: Path) -> dict:
    """Baseline for K2: standard mlx_lm.lora next-token CE on (c_pre -> c_post) pairs at
    matched rank/targets/scale/steps. This IS available via mlx_lm.lora CLI — but running
    it in this iteration without the Hedgehog arm would produce an unpaired K2 measurement.
    Deferred to the _impl follow-up so both arms land together.
    """
    raise NotImplementedError(
        "Token-space LoRA baseline deferred to _impl follow-up to keep K2 head-to-head "
        "measurement paired with Hedgehog arm."
    )


# ─────────────────────────────────────────────
# Phase C: K1 structural cos + K2 judge
# ─────────────────────────────────────────────

def measure_k1_structural_cos(adapter_path: Path, held_out: list) -> float:
    """K1: mean per-layer cos(A_l_teacher, A_l_student) on held-out refactor prompts.

    NOT YET IMPLEMENTED. Requires same teacher/student forward-pass instrumentation as
    Phase B, applied in eval mode.
    """
    raise NotImplementedError("K1 measurement depends on Phase A/B completing.")


def measure_k2_refactor_quality(student_adapter: Path, baseline_adapter: Path,
                                 held_out: list) -> float:
    """K2: auto-judge refactor quality Δ = score(student) − score(baseline_token_space_lora).

    NOT YET IMPLEMENTED. Requires:
      - generate(student_model, format_student_prompt(c_pre)) per held-out pair
      - generate(baseline_model, format_student_prompt(c_pre)) per held-out pair
      - judge rubric (0-10): (a) unit tests pass on student's output, (b) refactor named
        correctly matches ground-truth, (c) semantic equivalence with c_post.
      - report mean(score_student - score_baseline).
    """
    raise NotImplementedError("K2 judge eval depends on both adapters existing.")


# ─────────────────────────────────────────────
# Phase D: K3 HumanEval + K4 non-refactor gen-from-spec
# ─────────────────────────────────────────────

def measure_k3_humaneval(adapter_path: Path) -> float:
    """K3: HumanEval pass@1 drop vs base Gemma 4 E4B. Hooks into existing eval
    (see experiments/models/exp_bench_humaneval*).
    """
    raise NotImplementedError("K3 HumanEval eval depends on adapter existing.")


def measure_k4_nonrefactor_codegen(adapter_path: Path) -> float:
    """K4: quality drop on non-refactor code-gen-from-spec tasks (e.g. MBPP or a
    code-gen split where there is NO refactor opportunity). Target: drop < 2pp.
    """
    raise NotImplementedError("K4 non-refactor eval depends on adapter existing.")


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

    print("Hedgehog procedural-adapter distillation (refactor, Gemma 4 E4B ← 26B)", flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_HELDOUT={N_HELDOUT}, N_STEPS={N_STEPS}",
          flush=True)
    print(f"mlx-lm version: {mlx_lm_version}", flush=True)

    data_dir = EXPERIMENT_DIR / "data" / "refactor_pairs"
    adapters_dir = EXPERIMENT_DIR / "adapters"
    student_adapter = adapters_dir / "hedgehog_refactor_r8"
    baseline_adapter = adapters_dir / "token_lora_refactor_r8"

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
        "phase_0_dataset": None,
        "phase_b_student_train": None,
        "phase_baseline_token_lora": None,
        "phase_c_k1_k2": {},
        "phase_d_k3_k4": {},
        "kc": {
            "K1786_per_layer_cos_gt_0_80": "untested",
            "K1787_refactor_quality_ge_baseline": "untested",
            "K1788_humaneval_drop_lt_3pp": "untested",
            "K1789_nonrefactor_drop_lt_2pp": "untested",
        },
        "verdict": "PROVISIONAL",
        "all_pass": False,
        "blockers": [],
    }

    # ── Phase 0: dataset ────────────────────────────────────
    print("\n=== Phase 0: Fowler refactor pair curation ===", flush=True)
    try:
        results["phase_0_dataset"] = prepare_refactor_pairs(data_dir, N_TRAIN, N_HELDOUT)
    except NotImplementedError as exc:
        results["phase_0_dataset"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase 0 not implemented: {exc}")
    except Exception as exc:
        results["phase_0_dataset"] = {"error": str(exc)}
        results["blockers"].append(f"Phase 0 failed: {exc}")

    # ── Phase B: Hedgehog student training ──────────────────
    try:
        phase_b = train_hedgehog_student(data_dir, student_adapter)
        results["phase_b_student_train"] = phase_b
    except NotImplementedError as exc:
        results["phase_b_student_train"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Phase B not implemented: {exc}")
    except Exception as exc:
        results["phase_b_student_train"] = {"error": str(exc)}
        results["blockers"].append(f"Phase B failed: {exc}")

    # ── Phase Baseline: token-space LoRA ────────────────────
    try:
        phase_baseline = train_token_space_lora_baseline(data_dir, baseline_adapter)
        results["phase_baseline_token_lora"] = phase_baseline
    except NotImplementedError as exc:
        results["phase_baseline_token_lora"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Baseline not implemented: {exc}")
    except Exception as exc:
        results["phase_baseline_token_lora"] = {"error": str(exc)}
        results["blockers"].append(f"Baseline failed: {exc}")

    # ── Phase C: K1 structural + K2 judge ──────────────────
    if student_adapter.exists():
        try:
            k1 = measure_k1_structural_cos(student_adapter, held_out=[])
            results["phase_c_k1_k2"]["K1_cos"] = round(float(k1), 4)
        except NotImplementedError as exc:
            results["blockers"].append(f"K1 not implemented: {exc}")
        except Exception as exc:
            results["blockers"].append(f"K1 failed: {exc}")

        try:
            k2 = measure_k2_refactor_quality(student_adapter, baseline_adapter, held_out=[])
            results["phase_c_k1_k2"]["K2_judge_delta"] = round(float(k2), 2)
        except NotImplementedError as exc:
            results["blockers"].append(f"K2 not implemented: {exc}")
        except Exception as exc:
            results["blockers"].append(f"K2 failed: {exc}")
    else:
        results["blockers"].append(
            "Phase C skipped: student adapter not produced (Phase B blocker)"
        )

    # ── Phase D: K3 HumanEval + K4 non-refactor ────────────
    if student_adapter.exists():
        try:
            k3 = measure_k3_humaneval(student_adapter)
            results["phase_d_k3_k4"]["K3_humaneval_drop_pp"] = round(float(k3), 2)
        except NotImplementedError as exc:
            results["blockers"].append(f"K3 not implemented: {exc}")
        except Exception as exc:
            results["blockers"].append(f"K3 failed: {exc}")

        try:
            k4 = measure_k4_nonrefactor_codegen(student_adapter)
            results["phase_d_k3_k4"]["K4_nonrefactor_drop_pp"] = round(float(k4), 2)
        except NotImplementedError as exc:
            results["blockers"].append(f"K4 not implemented: {exc}")
        except Exception as exc:
            results["blockers"].append(f"K4 failed: {exc}")
    else:
        results["blockers"].append(
            "Phase D skipped: student adapter not produced (Phase B blocker)"
        )

    # ── KC resolution ──────────────────────────────────────
    # With Phase 0 / Phase B / Baseline not implemented, all four KCs remain "untested".
    # When the _impl follow-up lands, populate kc dict from phase_c_k1_k2 + phase_d_k3_k4.

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
