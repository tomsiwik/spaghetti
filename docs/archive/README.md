# Archive

Frozen material that is no longer load-bearing but preserved for history. Nothing here is current truth — see `STATUS.md` and `PLAN.md` at the repo root for the live state.

## `2026-06-03-superseded/` — drifted product docs

These four documents were written ~Apr 25 2026 and present the Pierre product thesis ("strategies transfer, knowledge doesn't; orthogonal composition beats frontier; >20pp over base") as proven. The team's own later, more-rigorous experiments (May 2026) refuted the load-bearing findings they cite. They are archived rather than rewritten in place so the original framing survives as a record of what was believed at the time.

| File | Headline claim | Refuted by (DB frontier) |
|---|---|---|
| `VISION.md` | Strategies are domain-agnostic and compose orthogonally; beats frontier | F#844 (strategy B-matrices are domain-entangled), F#827/837 (cross-domain interference real), F#822/823 (orthogonality is noise at scale) |
| `ARCHITECTURE.md` | "Every component references a proven finding"; M2P 99.6%; Grassmannian-critical | F#345/820 (M2P centroid trap), F#822/823 (Grassmannian benefit is a 3-layer artifact), F#510 (pre-merge gives 0%) |
| `GAMEPLAN.md` | "Phase 0 COMPLETE"; cross-domain transfer risk is Low | F#827/837/844 — that risk materialized; survivors give only +2-4pp, not the +20pp goal |
| `RESEARCH.md` | 100+ papers validate the strategy-transfer thesis | The bibliography is real and reusable; the Pierre-findings cross-reference table is stale. Counter-evidence (arXiv:2510.03262) was already buried in the doc. |
| `pierre_research_mindmap.json` | Orphaned alignment/eval dataset taxonomy | Disconnected from the current Gemma-4 composition line |

The honest, DB-aligned version of this material lives in `STATUS.md`.

## `composer/` — pre-MLX torch/CUDA/vLLM path

The original composition CLI and training/distillation code, built on `torch`/`vllm`/RunPod GPUs. Fully dead under the MLX-first strategy (zero `mlx` imports; `compose.py` uses vLLM, `evolve.py`/`distill.py`/`rank_sweep.py` use `torch.cuda`, `runpod_exec.py` is the remote-GPU runner). Removed from `pyproject.toml` (`compose` script entry point + wheel package). Preserved in case any historical experiment needs to be re-read.

> Recover anything here with `git mv` or `git show` — full history is intact; these were moved, not rewritten.
