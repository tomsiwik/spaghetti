# MATH.md — Pierre vs raw Gemma 4 (research-path SOTA baseline)

## Hypothesis

Pierre with composed PoLAR adapters should beat raw Gemma 4 E4B 4-bit on
target benchmarks. We've never measured this directly — all prior
experiments compared Pierre composition variants against each other,
not against raw base.

This experiment establishes the **research-path SOTA baseline**: Pierre's
true win over its own base model, using direct MLX evaluation (bypassing
Pierre's HTTP harness which has known bugs per `notebooklm_briefing.md`
and the product audit).

> **By how many percentage points does Pierre+Fisher-Rao composition beat
> raw Gemma 4 E4B 4-bit on the standard 3-benchmark suite?**

## Why research path, not HTTP

Pierre product audit found multiple issues in `apps/bench/pierre_bench`:
- `direct_benchmark.py:33` uses `max_tokens=50` (truncates mid-bullet)
- `runner.py:88` builds `strategies`/`domain` keys but server drops them
- AIME/GPQA/IFEval scorers all have edge-case bugs
- `comparison_results.json` (where raw Gemma scored 20% on math) is from the broken script

Bypassing all of that, we use the research path: load model + adapters
directly into MLX, evaluate via `scripts/polar_train.py::eval_*`. Same
prompts, same generation config, same scoring as our 11+ composition
experiments.

## Methods compared

- **M_raw**: Raw Gemma 4 E4B 4-bit, no adapters.
- **M_fr**: Pierre + Fisher-Rao K=7 composition (current product default).
- **M_winner_placeholder**: Best single-adapter per benchmark (oracle routing — bound on what perfect routing would give).

When the composition winner from the queued 10-experiment family is known,
this experiment can be re-run to compare raw vs winner directly. The
current run establishes the raw and Fisher-Rao numbers on a clean rig.

## Pre-registered Kill Criteria

- **K1 (PRODUCT VALUE)** Pierre+Fisher-Rao avg ≥ Raw Gemma + 3pp. PASS = adapters demonstrably help.
- **K2 (PER-BENCHMARK)** Pierre wins on each of GSM8K, HumanEval, MedQA individually (not just on average).
- **K3 (NO REGRESSION)** Pierre per-benchmark score ≥ Raw − 2pp (composition does not destructively interfere on any axis).
- **K4 (SANITY)** Raw Gemma scores match published numbers within ±5pp:
  - GSM8K ≥ 50% (Gemma 4 E4B published is ~70% but with thinking; we eval without thinking)
  - HumanEval ≥ 65%
  - MedQA ≥ 35% (4B-class typical)

## Verdict logic

| K1 | K2 | Outcome |
|----|----|---------|
| ✓ | ✓ | **SUPPORTED** — Pierre is a credible v0 product over raw base. |
| ✓ | ✗ | **SUPPORTED with caveat** — Pierre wins on average but loses ≥1 benchmark — flag for investigation. |
| ✗ | * | **INCONCLUSIVE for product** — adapters don't pay their cost on these benchmarks; revisit composition or expand benchmark set. |

K4 failure means the eval pipeline itself is suspect — investigate before
using these numbers anywhere.

## Eval protocol

- N = 50 per benchmark, fixed seed=42
- Same `eval_gsm8k`, `eval_humaneval`, `eval_medqa` from `scripts/polar_train.py`
- Base: `mlx-community/gemma-4-e4b-it-4bit`
- Adapters: 7 PoLAR (current default) for Fisher-Rao path
- For Raw: PoLARLinear injected with rank=6 but `lora_b = zeros` (no contribution); equivalent to base inference

## Honest gaps

- Doesn't include published-leaderboard external models (Llama 3.1 8B, Qwen 2.5 7B). Establishing that comparison requires running those models on the same eval rig — separate experiment if Pierre+FR shows promise.
- N=50 has known noise (HumanEval ±23pp at N=30 from prior experiments). For the eventual product number, run high-N validation.
- AIME, MATH-500, GPQA, IFEval not included — those are in Pierre's harness but not our research eval. Separate experiment using Pierre's lm_eval_adapter once it's wired correctly.

## References

- Pierre product audit findings (research agent `a076fbcf535c6e34c`)
- Composition winner experiments: 10 P1-P3 already queued
