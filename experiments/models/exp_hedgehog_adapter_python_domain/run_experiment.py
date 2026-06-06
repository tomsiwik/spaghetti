#!/usr/bin/env python3
"""
exp_hedgehog_adapter_python_domain — Hedgehog per-layer cos-sim distillation for Python
domain knowledge.

Pre-registered KCs (canonical DB text — do not edit):
  K#1844 proxy:  Hedgehog cos-sim distilled Python adapter PPL on Python-specific eval
                 > base + generic LoRA  (kill if TRUE — Hedgehog strictly worse)
  K#1845 target: Python idiomaticity auto-judge < +5 pp vs base
                 (pair K1844 per F#666)  (kill if TRUE — target threshold not met)

Skills invoked: /mlx-dev (documented in MATH.md §0). /fast-mlx pending for the _impl
follow-up (where the real training loop lands).

Runtime structure (mirrors sibling exp_hedgehog_domain_adapter_js):
  Phase 0 — CPython docs + PEP-8/PEP-20/PEP-257/PEP-484 corpus + (Q, A) pair curation
  Phase A — teacher attention-output capture: 26B Gemma variant + π_Py in context + Q
  Phase B — student training: rank-8 LoRA on v_proj/o_proj, per-layer cos-sim loss
            (custom MLX training loop — NOT available via mlx_lm.lora CLI)
  Phase Baseline — generic token-space LoRA at matched params (for K1844 head-to-head)
  Phase C — K1844 PPL on held-out Python eval (base vs base+gen-LoRA vs base+Hedgehog)
  Phase D — K1845 idiomaticity auto-judge on 50 blind-paired prompts

Full budget: ~4–6 h on M5 Pro 48 GB (two training jobs at 800 steps + 26B teacher
residency + judge per held-out pair + PPL eval). Exceeds single-iteration researcher cap
(30 min / 40 tool calls per guardrail 1009). This scaffold supports SMOKE_TEST=1 and
full runs; Phase 0/B/Baseline/C/D raise NotImplementedError pending a dedicated `_impl`
iteration (antipattern `mem-antipattern-novel-mechanism-single-iteration-scope`).
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
SEQLEN = 2048                           # CPython doc blocks run longer than refactor snippets
ENABLE_THINKING = True                  # F#614 — load-bearing on Gemma 4 reasoning

PY_FOCUS_TOPICS = (
    "duck_typing_and_protocols",
    "context_managers_with_statement",
    "generators_iterators_yield",
    "decorators_and_functools",
    "gil_threading_vs_asyncio",
    "comprehensions_and_expressions",
    "descriptors_and_properties",
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

def prepare_py_qa_pairs(data_dir: Path, n_train: int, n_heldout: int) -> dict:
    """Curate (topic, π_Py, Q, A) pairs for the 7 focus topics.

    NOT YET IMPLEMENTED. Requires:
      - CPython doc page fair-use summaries for each of PY_FOCUS_TOPICS.
      - PEP canonical text (PEP-8/20/257/484) — public domain, safe to include.
      - LLM-generated (Q, A) pairs per topic with source-in-context — validated
        against canonical doc answers by spot-check (n=20) + LLM-judge.
      - Train/held-out split stratified by topic so held-out contains distinct
        instances of same concept categories (tests generalization, not memorization).
      - External Python eval corpus for K1844 PPL: top-N PyPI open-source samples,
        filtered for code-only, disjoint from CPython docs to avoid train contamination.

    Load-bearing for K1844 (PPL eval text selection) and K1845 (judge reference answers).
    Curation choices here shape what K1844 and K1845 measure.
    """
    raise NotImplementedError(
        "Phase 0 Python corpus + Q-A pair curation not yet implemented. "
        "Requires CPython docs + PEP text + validated (Q, A) pairs per "
        f"{len(PY_FOCUS_TOPICS)} focus topics + external PyPI PPL eval corpus. "
        "See PAPER.md §Measurement blockers."
    )


# ─────────────────────────────────────────────
# Phase A + B: Hedgehog custom training loop
# ─────────────────────────────────────────────
#
# NOT-YET-IMPLEMENTED. Requires:
# 1. Loading STUDENT_MODEL and TEACHER_MODEL via mlx_lm.load (both frozen except student
#    LoRA). On 48 GB M5 Pro the 26B teacher + E4B student may not co-reside; sequential-
#    phase pattern:
#      - teacher forward on batch, capture {layer_idx: attn_out}; mx.eval + mx.clear_cache
#      - evict teacher (del + gc.collect + mx.clear_cache); load student if not loaded
#      - student forward + loss + step; mx.eval(student_model.parameters(), loss)
#    Alternative: pre-compute teacher attn traces for all train prompts in an offline
#    pass, write to disk, stream during student training. Trades compute for peak memory.
# 2. Attach rank-8 LoRA on v_proj + o_proj of student via mlx_lm.tuner.lora.LoRALinear
#    .from_linear; freeze all other student params.
# 3. Module-subclass wrapper around student.layers[i].self_attn for all i in 0..L-1 that
#    records attn_out per forward pass. Gemma 4 E4B: L = 42. Prefer subclass over
#    __call__ monkeypatch (fragile across mlx-lm versions).
# 4. Loss per batch:
#       L = mean_l (1 - cos(A_l_teacher(Q | π_Py), A_l_student(Q)))
#    Cos computed per-token then mean-pooled over tokens per layer, then mean over
#    layers. teacher-side: stop_gradient (no grad flow through 26B).
# 5. Gradient step: nn.value_and_grad(student_model, loss_fn); mlx.optimizers.AdamW.
#    mx.eval(student_model.parameters(), loss) at step boundary. mx.clear_cache() between
#    batches (especially after teacher forward — F#673).
# 6. Save LoRA weights compatible with mlx_lm adapter loading.
#
# Scope-preservation: do NOT silently fall back to cross-entropy SFT on A tokens. The
# claim is cos-sim distillation transfers *domain-knowledge* routing; substituting CE
# would test a different hypothesis (surface imitation of A tokens).

def train_hedgehog_student(data_dir: Path, adapter_path: Path) -> dict:
    """Phase B: train rank-8 LoRA student with per-layer cos-sim loss vs 26B teacher.

    NOT YET IMPLEMENTED. Returns NotImplementedError with structured marker so the
    pipeline can record a PROVISIONAL verdict without silent degradation.
    """
    raise NotImplementedError(
        "Hedgehog cos-sim distillation training loop not yet implemented. "
        "Requires custom MLX: sequential-phase teacher/student forward passes (42 "
        "layers), per-layer attention-output hooks, per-layer cos-sim loss, "
        "nn.value_and_grad + AdamW. Teacher is 26B — peak memory is load-bearing. "
        "See PAPER.md §Measurement blockers."
    )


def train_generic_token_lora_baseline(data_dir: Path, adapter_path: Path) -> dict:
    """Baseline for K1844: generic token-space LoRA next-token CE on (Q -> A) pairs at
    matched rank/targets/scale/steps. This IS available via mlx_lm.lora CLI — but
    running it in isolation without the Hedgehog arm would produce an unpaired K1844
    measurement (PPL comparison loses its head-to-head anchor). Deferred to the _impl
    follow-up so both arms land together.
    """
    raise NotImplementedError(
        "Generic token-space LoRA baseline deferred to _impl follow-up to keep K1844 "
        "head-to-head PPL measurement paired with Hedgehog arm."
    )


# ─────────────────────────────────────────────
# Phase C: K1844 PPL head-to-head
# ─────────────────────────────────────────────

def measure_k1844_ppl(student_adapter: Path, baseline_adapter: Path,
                      py_eval_text: list) -> dict:
    """K1844: PPL(base), PPL(base + generic token-space LoRA), PPL(base + Hedgehog).

    Kill condition: PPL(Hedgehog) > PPL(base + generic token-space LoRA) strictly.

    NOT YET IMPLEMENTED. Requires:
      - mlx_lm.load(STUDENT_MODEL, adapter_path=...) for each of the three configs.
      - For each config, compute cross-entropy on tokenized held-out py_eval_text
        (external PyPI slice, disjoint from CPython training docs).
      - PPL = exp(mean_cross_entropy).
      - Report all three PPLs, plus the pairwise deltas needed for K1844.
    """
    raise NotImplementedError(
        "K1844 PPL head-to-head depends on both Hedgehog and baseline adapters existing."
    )


# ─────────────────────────────────────────────
# Phase D: K1845 idiomaticity auto-judge
# ─────────────────────────────────────────────

def measure_k1845_idiomaticity_judge(student_adapter: Path,
                                     held_out: list) -> float:
    """K1845: blind-paired judge scores idiomaticity of base vs Hedgehog-adapted answers
    on 50 held-out Python-nuance prompts. Returns Δ = mean(score_hedgehog - score_base).

    Kill condition: Δ < +5 pp.

    NOT YET IMPLEMENTED. Requires:
      - generate(base, format_prompt(Q)) per held-out pair.
      - generate(base + Hedgehog, format_prompt(Q)) per held-out pair.
      - judge rubric (0-10): (a) correct Python feature usage, (b) idiomatic construct
        preference (comprehensions / context managers / decorators), (c) PEP-8
        conformance where load-bearing.
      - Blind-paired presentation to judge to reduce absolute-score bias.
      - Report mean(score_hedgehog - score_base) as a pp-delta on the 0–100 rescaled
        rubric.
    """
    raise NotImplementedError("K1845 judge eval depends on Hedgehog adapter existing.")


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

    print("Hedgehog Python-domain adapter distillation (Gemma 4 E4B ← 26B)", flush=True)
    print(f"SMOKE_TEST={IS_SMOKE}, N_TRAIN={N_TRAIN}, N_HELDOUT={N_HELDOUT}, N_STEPS={N_STEPS}",
          flush=True)
    print(f"mlx-lm version: {mlx_lm_version}", flush=True)

    data_dir = EXPERIMENT_DIR / "data" / "py_qa_pairs"
    adapters_dir = EXPERIMENT_DIR / "adapters"
    student_adapter = adapters_dir / "hedgehog_py_r8"
    baseline_adapter = adapters_dir / "token_lora_py_r8"

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
        "py_focus_topics": list(PY_FOCUS_TOPICS),
        "phase_0_dataset": None,
        "phase_b_student_train": None,
        "phase_baseline_token_lora": None,
        "phase_c_k1844": {},
        "phase_d_k1845": {},
        "kc": {
            "K1844_hedgehog_ppl_not_greater_than_generic_lora": "untested",
            "K1845_idiomaticity_delta_ge_5pp": "untested",
        },
        "verdict": "PROVISIONAL",
        "all_pass": False,
        "blockers": [],
    }

    # ── Phase 0: dataset ────────────────────────────────────
    print("\n=== Phase 0: CPython + PEP corpus + Q-A curation ===", flush=True)
    try:
        results["phase_0_dataset"] = prepare_py_qa_pairs(data_dir, N_TRAIN, N_HELDOUT)
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

    # ── Phase Baseline: generic token-space LoRA ────────────
    try:
        phase_baseline = train_generic_token_lora_baseline(data_dir, baseline_adapter)
        results["phase_baseline_token_lora"] = phase_baseline
    except NotImplementedError as exc:
        results["phase_baseline_token_lora"] = {"not_implemented": str(exc)}
        results["blockers"].append(f"Baseline not implemented: {exc}")
    except Exception as exc:
        results["phase_baseline_token_lora"] = {"error": str(exc)}
        results["blockers"].append(f"Baseline failed: {exc}")

    # ── Phase C: K1844 PPL head-to-head ─────────────────────
    if student_adapter.exists() and baseline_adapter.exists():
        try:
            k1844 = measure_k1844_ppl(student_adapter, baseline_adapter, py_eval_text=[])
            results["phase_c_k1844"] = k1844
        except NotImplementedError as exc:
            results["blockers"].append(f"K1844 not implemented: {exc}")
        except Exception as exc:
            results["blockers"].append(f"K1844 failed: {exc}")
    else:
        results["blockers"].append(
            "Phase C skipped: Hedgehog and/or baseline adapter not produced "
            "(Phase B + Baseline blockers)"
        )

    # ── Phase D: K1845 idiomaticity judge ───────────────────
    if student_adapter.exists():
        try:
            k1845 = measure_k1845_idiomaticity_judge(student_adapter, held_out=[])
            results["phase_d_k1845"]["K1845_idiomaticity_delta_pp"] = round(float(k1845), 2)
        except NotImplementedError as exc:
            results["blockers"].append(f"K1845 not implemented: {exc}")
        except Exception as exc:
            results["blockers"].append(f"K1845 failed: {exc}")
    else:
        results["blockers"].append(
            "Phase D skipped: Hedgehog adapter not produced (Phase B blocker)"
        )

    # ── KC resolution ──────────────────────────────────────
    # With Phase 0 / Phase B / Baseline not implemented, both KCs remain "untested".
    # When the _impl follow-up lands, populate kc dict from phase_c_k1844 + phase_d_k1845
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
