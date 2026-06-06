# PAPER: Correct Delta-Sum LoRA Composition at N=5 (KILLED, preemptive)

**Verdict:** KILLED (preemptive, 2026-04-18, 11th P11-style kill this week)

**Driver:** antipattern-017 — 3rd confirmed instance in 2 days.

## tl;dr

K1548 requires measuring held-out PPL under explicit `Sum(B_i @ A_i)` composition
at N=5, against a solo-adapter baseline. This requires 5 trained LoRA adapter
weight sets. Filesystem audit (2026-04-18) shows that all 5 adapter directories
referenced in `adapters/registry.json` contain only `adapter_config.json` — no
`adapters.safetensors` weights. The experiment is un-runnable until the adapters
are actually trained.

## Why this is not a run

The DB claims `exp_p1_t2_multi_domain_5` is `supported` with evidence
"K1047 PASS: all 5 adapters ≥+3pp — math+82pp, code+46pp, medical+22pp,
legal+50pp, finance+56pp". But the directories referenced as the adapters'
on-disk locations contain *only* config files. Loading any one of them via
mlx-lm's `load(..., adapter_path=...)` yields either a missing-file error or
silently falls back to the base model. In the latter case, *every* K1548
measurement would be the same base-model PPL, rendering the experiment
vacuous.

## Prediction vs measurement

| Claim / KC | MATH.md prediction | Measurement | Status |
|---|---|---|---|
| K1548: PPL_composed ≤ 2 × mean(PPL_solo) at N=5 | should pass (Thm 2: Frobenius triangle inequality bounds composed norm linearly in N) | **unmeasurable** (no adapters to compose) | **FAIL** |
| Theorem 1 (N² − N cross-terms differ) | proven in MATH | not measured | N/A (math-only) |
| Theorem 2 (bounded composed norm) | proven | not measured | N/A (math-only) |
| Theorem 3 (buggy catastrophe mechanism) | proven; corroborated by Finding #23 | not re-measured | N/A (already Finding #23) |

## Dependency state

| Required artifact | Path | State |
|---|---|---|
| math adapter (r=6) | `micro/models/exp_p1_t2_single_domain_training/adapters/math/adapters.safetensors` | **MISSING** |
| code adapter (r=6) | `.../adapters/code/adapters.safetensors` | **MISSING** |
| medical adapter (r=6) | `.../adapters/medical/adapters.safetensors` | **MISSING** |
| legal adapter (r=6) | `micro/models/exp_p1_t2_multi_domain_5/adapters/legal/adapters.safetensors` | **MISSING** |
| finance adapter (r=6) | `.../finance/adapters.safetensors` | **MISSING** |
| base model | `mlx-community/gemma-4-e4b-it-4bit` | present (F#560 used it) |
| held-out text slice | wikitext / domain-held-out | would be constructed at runtime |

5-of-5 weight files missing — this is a larger stub-proportion than J0 (4/4)
or M0 (2 missing + 2 stub = 4/4 unusable out of 4 candidate paths).

## Antipattern self-check

- ✗ antipattern-017 (weight-less stub adapters): **TRIGGERED, 5-of-5.**
  Promotes antipattern-017 to **3 confirmed instances** across the audit
  (exp_p11_baseline_eval, exp_p11_adapter_composition_thinking [J0],
  exp_p11_full_pipeline_v2 [M0], and this experiment).

  Pre-flight grep that catches this class of error:
  ```bash
  find micro/models/exp_p1_t2_single_domain_training/adapters \
       micro/models/exp_p1_t2_multi_domain_5/adapters \
       -maxdepth 2 -type f
  ```
  (expected 5 `adapters.safetensors`; observed 0.)

- ⊘ antipattern-003 (LORA_SCALE=20): N/A — no inference code executed.
- ⊘ antipattern-008 (thinking-mode strip): N/A.
- ⊘ antipattern-018 (channel tokens as SFT text): N/A — no training here.
- ⊘ antipattern-020 (cascade-dependent design): distinct — here the
  dependency is on a stub artifact (antipattern-017), not on a killed
  upstream experiment (antipattern-020's structure).
- ⊘ KC-swap: MATH.md authored fresh; no KC modification.

## Unblock path (for v2)

`P11.ADAPTER-REBUILD` — retrain the 5 domain LoRA adapters as actually-fit
weight files. Registry says rank=6, target_modules=[q_proj], 1000 steps per
domain on Gemma 4 E4B 4bit; actual retraining likely needs to resolve
mlx_lm.lora subprocess issues (see F#557). Once weights exist,
`exp_followup_composition_correct_delta_v2` can be scoped as ~15 min:
load base + 5 adapters, measure 3 PPL conditions (solo avg, correct
delta-sum, buggy delta-sum as positive control), check K1548.

## Assumptions (for record)

- Composition targets `q_proj` only (per registry). If a v2 expands to
  `q_proj+v_proj`, the catastrophe baseline may shift — still bounded by
  Thm 2 but the solo PPL floor would drop.
- Held-out PPL slice would be a stratified mixture across the 5 domains
  to avoid favoring any adapter; design to be locked in MATH.md before
  measurement in v2 to preserve KC discipline.
- 2× threshold is inherited from K1548 wording; Finding #14 would pass
  at 1/N scaling (margin is large), so K1548 at unscaled Sum is a
  meaningful stress test, not a trivially-pass version of F#14.

## References

- Finding #14 [supported] — 1/N scaling resolves composition catastrophe at N=5.
- Finding #23 [killed] — Equal-weight composition is fragile (trillion-PPL).
- Finding #199 [conclusive] — A-matrix loading bug.
- Finding #544 [killed] — evidence cites `premerge_vs_dynamic.py:415-431`
  as canonical bug location.
- F#557 — mlx_lm.lora subprocess crash without checkpoint (blocks rebuild).
- antipattern-017 (weight-less stub adapters) — instance #3.

## Handoff

Reviewer hat: verify 5-of-5 stub claim via `find` command above;
confirm MATH.md proofs stand; check KC-swap via `git diff MATH.md` (empty).
Analyst hat: promote antipattern-017 from "2 confirmed instances" to
"3 confirmed instances"; emphasize the pre-flight grep as a systemic
check for *any* composition experiment.
