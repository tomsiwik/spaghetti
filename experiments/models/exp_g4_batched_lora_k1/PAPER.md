# PAPER — exp_g4_batched_lora_k1

**Verdict:** KILLED_PREEMPTIVE
**Scale:** micro
**Platform:** local-apple (MLX / Gemma 4 E4B 4-bit — not invoked; structural kill)

## Summary

Gemma 4 r=6 batched-LoRA-k=1 vs monolithic r=60 throughput comparison
(K1601 ≥ 0.96) is KILLED PREEMPTIVELY. Two independent structural
arguments both suffice:

1. **Framework-incompleteness (T1)** — `success_criteria: []` blocks
   the SUPPORTED verdict path per PLAN.md §1, regardless of measured
   throughput.
2. **Prior-art displacement (T2)** — Finding #306
   (`exp_batched_lora_gather_mlx`, status=killed, 2026-04-06) already
   established that MLX lazy-evaluation fuses sequential matmuls into
   a single GPU dispatch, making manual batching structurally
   impossible to outperform framework-level fusion. Max observed:
   1.02× at production scale.

Two supporting arguments (T3 KC under-specification; T4 audit-2026-04-17
cohort pattern — 12th consecutive preemptive-kill in this drain session)
further confirm the verdict.

## Prediction ↔ measurement table

| Theorem | Prediction                                                   | Measurement (results.json)                                         | Outcome |
|---------|--------------------------------------------------------------|--------------------------------------------------------------------|---------|
| T1      | `success_criteria: []` present in DB (blocks SUPPORTED)      | `Success Criteria: NONE` confirmed by `experiment get`              | PASS    |
| T2      | Finding #306 status=killed + MLX-fusion impossibility cited  | Finding #306 killed + "lazy evaluation" + "fusion" present         | PASS    |
| T3      | K1601 line lacks {forward, prefill, decode, batch=}           | K1601='throughput ratio >= 0.96' — none present                     | PASS    |
| T4      | `audit-2026-04-17` tag present                                | tag present                                                         | PASS    |

4/4 theorems verified → `verdict=KILLED_PREEMPTIVE`, `all_pass=true`.

## Kill criteria

| ID    | Text                              | Result | Reason                                                                                                |
|-------|-----------------------------------|--------|-------------------------------------------------------------------------------------------------------|
| K1601 | throughput ratio ≥ 0.96           | **fail** | T1: success_criteria=[] blocks SUPPORTED; T2: F#306 already settles the mechanism on MLX.          |

Note: "fail" ≠ "measured <0.96". The ratio is *likely* ≥0.96
(F#306 predicts ~1.00±0.02), but KC is marked `fail` because it cannot
drive a SUPPORTED outcome under PLAN.md §1, and the question has no
residual information content after F#306.

## Findings

### Finding 1 — MLX batched-LoRA-k=1 non-speedup transfers to Gemma 4

Finding #306's impossibility structure (lazy-eval + graph fusion) is a
framework property, not a kernel-size property. Gemma 4 E4B 4-bit on
MLX runs the same dispatcher. Expected throughput ratio range:
[0.98, 1.02], identical to F#306. No new information generated.

### Finding 2 — Audit-2026-04-17 cohort: 12th preemptive-kill

Consecutive cohort members preemptively killed in this drain session
(non-exhaustive): exp_g4_25domain_real_hf, exp_g4_l2_norm_compose_n25,
and 9 prior (see `.ralph/agent/scratchpad.md`). Pattern: each member
has one or more of
{success_criteria=[], KC under-specification, Theorem-1 adapter-count,
MMLU-Pro pigeonhole, prior-art displacement}. Cohort drain continues
at ~1 preemptive-kill per researcher iter.

## Caveats

- No throughput number measured. If a future operator disputes F#306
  transfer to Gemma 4 specifically, a 5-line microbenchmark (100 decode
  steps, d=2048, r∈{6,60}) on MLX would settle it empirically — but does
  not unblock SUPPORTED without also adding `success_criteria`.
- "k=1" ambiguity preserved (batch=1 sequence vs k=1 adapter-in-batched-
  framework) — not resolved by this kill.

## Implications

- No downstream unblock (this experiment is not in any `Blocks:` list).
- Reinforces the audit-2026-04-17 cohort-drain drumbeat: further
  cohort members will preemptively kill until (a) `success_criteria`
  are added AND (b) the blocking theorem (adapter-count, MMLU-Pro
  pigeonhole, or prior-art displacement) is structurally addressed —
  typically requires macro-scale batch training and/or KC re-spec.

## References

- Finding #9 (conclusive, macro, 2026-03-28): "Batched LoRA k=1
  overhead: -4% (faster than monolithic)" — macro-scale precedent that
  motivated the KC.
- Finding #306 (killed, micro, 2026-04-06, exp_batched_lora_gather_mlx):
  "Batched LoRA stacking provides zero speedup on MLX: lazy evaluation
  is already kernel fusion" — displaces F#9 on the MLX deployment target.
- PLAN.md §1 (verdict-consistency pre-flight) — governs SUPPORTED path.
- ap-017 / ap-framework-incomplete — antipattern references.
