#!/usr/bin/env python3
"""
exp_hedgehog_domain_adapter_js — Hedgehog per-layer cos-sim distillation for JS domain knowledge.

Pre-registered KCs (canonical DB text — do not edit):
  K#1790 structural: per-layer cos(teacher, student) > 0.80 on held-out JS-nuance prompts
  K#1791 target:     JS-specific benchmark (HumanEval-JS or custom JS-nuance suite) accuracy
                     >= base + token-space LoRA at matched params, n>=100
                     (pair K1790 per F#666)
  K#1792 target:     Python code accuracy (HumanEval) drop < 3pp vs base
                     (adapter is JS-specific, not broad code replacement)
  K#1793 target:     natural-language tasks (MMLU subset) drop < 2pp
                     (adapter is domain-narrow)

Skills invoked: /mlx-dev (documented in MATH.md §0). /fast-mlx pending for the _impl follow-up.

Runtime structure (mirrors sibling exp_hedgehog_procedural_adapter_refactor):
  Phase 0 — MDN + Eloquent-JS corpus + Q-A pair curation (train/held-out split)
  Phase A — teacher attention-output capture: 26B Gemma variant + JS-docs-in-context + Q
  Phase B — student training: rank-8 LoRA on v_proj/o_proj, per-layer cos-sim loss
            (custom MLX training loop — NOT available via mlx_lm.lora CLI)
  Phase Baseline — same-data token-space LoRA at matched params (for K1791 head-to-head)
  Phase C — K1790 structural cos + K1791 auto-judge JS-benchmark quality
  Phase D — K1792 HumanEval (Python) pass@1 + K1793 MMLU subset

Full budget: ~4-6h on M5 Pro 48GB (two training jobs at 800 steps + 26B teacher residency
+ judge per held-out pair + HumanEval + MMLU). Exceeds single-iteration researcher cap
(30 min / 40 tool calls per guardrail 1009). This scaffold supports SMOKE_TEST=1 and full
runs; Phase 0/B/Baseline/C/D raise NotImplementedError pending a dedicated `_impl` iteration
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
SEQLEN = 2048                           # JS docs blocks are longer than refactor snippets
ENABLE_THINKING = True                  # F#614 — load-bearing on Gemma 4 reasoning

JS_FOCUS_TOPICS = (
    "hoisting_and_tdz",
    "closures_and_scope",
    "this_binding_and_arrow_functions",
    "event_loop_microtasks",
    "prototype_chain",
    "async_await_error_handling",
)


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

def prepare_js_qa_pairs(data_dir: Path, n_train: int, n_heldout: int) -> dict:
    """Curate (topic, π_JS, Q, A) pairs.

    NOT YET IMPLEMENTED. Requires:
      - MDN page fair-use summaries for each of JS_FOCUS_TOPICS (reproducible under
        fair-use summarization guidelines), keyed by canonical topic name.
      - Eloquent JavaScript (CC-BY-NC 3.0) chapter summaries for matching topics.
      - LLM-generated (Q, A) pairs per topic with source-in-context — validated
        against MDN canonical answers by spot-check (n=20) + LLM-judge.
      - Train/held-out split stratified by topic so held-out contains distinct
        instances of same concept categories (tests generalization, not memorization).

    Load-bearing for K1791: judge must see held-out Q and score student's answer
    against reference A. Curation choices here shape what K1790–K1793 measure.
    """
    raise NotImplementedError(
        "Phase 0 JS corpus + Q-A pair curation not yet implemented. "
        "Requires MDN + Eloquent-JS excerpts + validated (Q, A) pairs per "
        f"{len(JS_FOCUS_TOPICS)} focus topics. See PAPER.md §Measurement blockers."
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
# 3. Module-subclass wrapper around student.layers[i].self_attn for all i in 0..L-1 that
#    records attn_out per forward pass. Gemma 4 E4B: L = 42. Prefer subclass over
#    __call__ monkeypatch (fragile across mlx-lm versions).
# 4. Loss per batch:
#       L = mean_l (1 - cos(A_l_teacher(Q | π_JS), A_l_student(Q)))
#    Cos computed per-token then mean-pooled over tokens per layer, then mean over layers.
#    teacher-side: stop_gradient (no grad flow through 26B).
# 5. Gradient step: nn.value_and_grad(student_model, loss_fn); mlx.optimizers.AdamW.
#    mx.eval(student_model.parameters(), loss) at step boundary. mx.clear_cache() between
#    batches (especially after teacher forward — F#673).
# 6. Save LoRA weights compatible with mlx_lm adapter loading.
#
# Scope-preservation: do NOT silently fall back to cross-entropy SFT on A tokens. The claim
# is cos-sim distillation transfers *domain-knowledge* routing; substituting CE would test a
# different hypothesis (surface imitation of A tokens).

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
    """Baseline for K1791: standard mlx_lm.lora next-token CE on (Q -> A) pairs at matched
    rank/targets/scale/steps. This IS available via mlx_lm.lora CLI — but running it in
    isolation without the Hedgehog arm would produce an unpaired K1791 measurement.
    Deferred to the _impl follow-up so both arms land together.
    """
    raise NotImplementedError(
        "Token-space LoRA baseline deferred to _impl follow-up to keep K1791 head-to-head "
        "measurement paired with Hedgehog arm."
    )


# ─────────────────────────────────────────────
# Phase C: K1790 structural cos + K1791 judge
# ─────────────────────────────────────────────

def measure_k1790_structural_cos(adapter_path: Path, held_out: list) -> float:
    """K1790: mean per-layer cos(A_l_teacher, A_l_student) on held-out JS-nuance prompts.

    NOT YET IMPLEMENTED. Requires same teacher/student forward-pass instrumentation as
    Phase B, applied in eval mode.
    """
    raise NotImplementedError("K1790 measurement depends on Phase A/B completing.")


def measure_k1791_js_benchmark(student_adapter: Path, baseline_adapter: Path,
                               held_out: list) -> float:
    """K1791: auto-judge JS-benchmark quality
        Δ = score(student) − score(baseline_token_space_lora).

    NOT YET IMPLEMENTED. Requires:
      - generate(student_model, format_prompt(Q)) per held-out pair + HumanEval-JS.
      - generate(baseline_model, format_prompt(Q)) per held-out pair + HumanEval-JS.
      - judge rubric (0-10): (a) syntactic validity, (b) factual correctness vs MDN
        canonical answer, (c) idiomatic JS-specific constructs.
      - report mean(score_student - score_baseline).
    """
    raise NotImplementedError("K1791 judge eval depends on both adapters existing.")


# ─────────────────────────────────────────────
# Phase D: K1792 HumanEval (Python) + K1793 MMLU subset
# ─────────────────────────────────────────────

def measure_k1792_humaneval_python(adapter_path: Path) -> float:
    """K1792: HumanEval (Python) pass@1 drop vs base Gemma 4 E4B. Hooks into existing eval
    (see experiments/models/exp_bench_humaneval*). JS-specific adapter should NOT help Python —
    target is drop < 3pp (non-interference).
    """
    raise NotImplementedError("K1792 HumanEval eval depends on adapter existing.")


def measure_k1793_mmlu_subset(adapter_path: Path) -> float:
    """K1793: MMLU subset (10 categories × 20 questions) drop vs base. Target: drop < 2pp
    (narrow domain adapter should not affect general NL).
    """
    raise NotImplementedError("K1793 MMLU eval depends on adapter existing.")


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

    print("Hedgehog JS-domain adapter distillation (Gemma 4 E4B ← 26B)", flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_HELDOUT={N_HELDOUT}, N_STEPS={N_STEPS}",
          flush=True)
    print(f"mlx-lm version: {mlx_lm_version}", flush=True)

    data_dir = EXPERIMENT_DIR / "data" / "js_qa_pairs"
    adapters_dir = EXPERIMENT_DIR / "adapters"
    student_adapter = adapters_dir / "hedgehog_js_r8"
    baseline_adapter = adapters_dir / "token_lora_js_r8"

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
        "js_focus_topics": list(JS_FOCUS_TOPICS),
        "phase_0_dataset": None,
        "phase_b_student_train": None,
        "phase_baseline_token_lora": None,
        "phase_c_k1790_k1791": {},
        "phase_d_k1792_k1793": {},
        "kc": {
            "K1790_per_layer_cos_gt_0_80": "untested",
            "K1791_js_bench_ge_token_lora": "untested",
            "K1792_humaneval_drop_lt_3pp": "untested",
            "K1793_mmlu_drop_lt_2pp": "untested",
        },
        "verdict": "PROVISIONAL",
        "all_pass": False,
        "blockers": [],
    }

    # ── Phase 0: dataset ────────────────────────────────────
    print("\n=== Phase 0: MDN + Eloquent-JS corpus + Q-A curation ===", flush=True)
    try:
        results["phase_0_dataset"] = prepare_js_qa_pairs(data_dir, N_TRAIN, N_HELDOUT)
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

    # ── Phase C: K1790 structural + K1791 judge ─────────────
    if student_adapter.exists():
        try:
            k1790 = measure_k1790_structural_cos(student_adapter, held_out=[])
            results["phase_c_k1790_k1791"]["K1790_cos"] = round(float(k1790), 4)
        except NotImplementedError as exc:
            results["blockers"].append(f"K1790 not implemented: {exc}")
        except Exception as exc:
            results["blockers"].append(f"K1790 failed: {exc}")

        try:
            k1791 = measure_k1791_js_benchmark(student_adapter, baseline_adapter, held_out=[])
            results["phase_c_k1790_k1791"]["K1791_judge_delta"] = round(float(k1791), 2)
        except NotImplementedError as exc:
            results["blockers"].append(f"K1791 not implemented: {exc}")
        except Exception as exc:
            results["blockers"].append(f"K1791 failed: {exc}")
    else:
        results["blockers"].append(
            "Phase C skipped: student adapter not produced (Phase B blocker)"
        )

    # ── Phase D: K1792 HumanEval + K1793 MMLU ──────────────
    if student_adapter.exists():
        try:
            k1792 = measure_k1792_humaneval_python(student_adapter)
            results["phase_d_k1792_k1793"]["K1792_humaneval_drop_pp"] = round(float(k1792), 2)
        except NotImplementedError as exc:
            results["blockers"].append(f"K1792 not implemented: {exc}")
        except Exception as exc:
            results["blockers"].append(f"K1792 failed: {exc}")

        try:
            k1793 = measure_k1793_mmlu_subset(student_adapter)
            results["phase_d_k1792_k1793"]["K1793_mmlu_drop_pp"] = round(float(k1793), 2)
        except NotImplementedError as exc:
            results["blockers"].append(f"K1793 not implemented: {exc}")
        except Exception as exc:
            results["blockers"].append(f"K1793 failed: {exc}")
    else:
        results["blockers"].append(
            "Phase D skipped: student adapter not produced (Phase B blocker)"
        )

    # ── KC resolution ──────────────────────────────────────
    # With Phase 0 / Phase B / Baseline not implemented, all four KCs remain "untested".
    # When the _impl follow-up lands, populate kc dict from phase_c_k1790_k1791 + phase_d_k1792_k1793.

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
