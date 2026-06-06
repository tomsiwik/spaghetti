# exp_rdt_depth_domain_2d — PAPER.md

## Verdict: KILLED (preemptive, dependency-unfulfilled)

## Summary

Preemptive-kill under F#669 (preempt-child-KCs-require-parent-target-claim-unverified).
All 4 KCs (K1749–K1752) transitively require trained artifacts from two parent
experiments. Neither parent produced them. No code was executed on the target.

## Dependency state

| Parent | Status | Trained artifacts? | Blocking KCs |
|---|---|---|---|
| exp_rdt_loop_lora_gemma4 | killed (smoke-PROVISIONAL, F#668) | No — scaffolding only, target K1740/K1741/K1742 deferred | K1749, K1750, K1752 |
| exp_method_composition_k_saturation | killed (Phase-1 teacher gate abort) | No — 0/5 methods passed teacher gate | K1749, K1752 |

## Prediction-vs-measurement table

| KC | Claim | Predicted | Measured | Pass |
|----|-------|-----------|----------|------|
| K1749 | 2D comp beats domain-only by ≥+3pp | N/A | not measured (T1: requires trained 2D artifacts) | fail |
| K1750 | Loop axis does NOT saturate at T≤6 with N=5 | N/A | not measured (T2: subset of parent target K1740/K1741) | fail |
| K1751 | avg \|cos(ΔW_d_i, ΔW_l_j)\| < 0.1 | N/A | not measured (T3: ΔW=0 at init; requires trained B) | fail |
| K1752 | Room Model cos>0.999 | N/A | not measured (T4: needs trained artifacts; F#571 already supersedes for N>1) | fail |

## Theorems (see MATH.md)

- **T1**: K1749 requires trained domain × loop artifacts from both parents.
- **T2**: K1750 is a stricter variant of parent K1740/K1741.
- **T3**: K1751 requires trained ΔW; at init, LoRA B=0 ⇒ ΔW=0.
- **T4**: K1752 Room Model requires all trained artifacts (also superseded by F#571 for N>1).

## Findings reused

- **F#669** (second reuse — promotion candidate): inter-experiment dep-preempt.
- **F#668**: parent RDT scaffolding provisional.
- **F#571**: Room Model superseded for N>1.
- **F#562**: partition-QR init orthogonality (distinct from trained-ΔW orthogonality).

## Antipattern self-audit

None triggered. This is a legitimate preempt, not a proxy, tautology, or metric
checkmark.

## Unblock path

1. Run `exp_rdt_loop_lora_gemma4_full` (macro follow-up ticket in parent LEARNINGS).
   Requires real GSM8K+MATH training at T={1..6}, full eval, saturating-exp fit.
2. Resurrect `exp_method_composition_k_saturation` with a structural fix to
   the Phase-1 teacher gate (or replace gate with a more robust method-proxy).
3. Only then can 2D composition be measured.

## Assumptions

- F#669 applies identically to double-parent-unfulfilled cases (single-parent
  precedent was exp_rdt_act_halting_throughput). This is the second reuse; if
  a third occurs, F#669 should be promoted from sub-axis to standalone.
- Room Model clause K1752 is redundant with F#571 (Room Model superseded for
  N>1); even if parents were supported, K1752 would fail by prior finding.
