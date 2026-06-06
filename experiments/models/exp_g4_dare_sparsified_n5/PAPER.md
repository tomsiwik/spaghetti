# PAPER — exp_g4_dare_sparsified_n5

**Verdict: KILLED (preemptive, 5-theorem stack)**

Cohort: audit-2026-04-17 / composition-bug / g4-gemma4 — 19th consecutive
preemptive-kill this session (18th composition-bug, 1st scale-safety).

## Prediction vs measurement

| Theorem | Prediction | Runner measurement |
|---------|-----------|--------------------|
| T1 inventory shortfall | 2 adapters missing at N=5 | **fail**: available=3 {code, math, medical}, shortfall=2 |
| T2 iter-budget breach | 2 × 20.92 min = 41.84 min > 30 min | arithmetic-level (MATH.md); not runner-tested |
| T3 success_criteria=[] | "Success Criteria: NONE" in `experiment get` | **fail**: verified present |
| T4 KC under-specification | 0/5 required adjudicatable keywords in K1609/K1610 | **fail**: matches=0 |
| T5 F#269 non-transfer | Arch + scale-rescale mismatch (BitNet-2B ternary s=20 vs Gemma 4 FP16 LoRA) | **fail**: F#269 present; BitNet/ternary and DARE substrings present |

All four runner-tested theorems fail. T5 runner passes fully — F#269 summary
directly contains "BitNet" and "DARE" substrings so no false-negative.

## Defense-in-depth

T1 (inventory shortfall=2) OR T3 (success_criteria=[]) OR T4 (KC under-spec,
0/5 adjudicatable fields) OR T5 (F#269 non-transfer) each alone blocks
SUPPORTED. T2 (iter-budget) reinforces. No single patch unblocks.

F#269 impossibility structure directly contradicts K1610: "DARE cannot fix
direction interference — MMLU math degradation -8pp persists across ALL drop
rates." K1610 ("p=0.5 loses 0% in-dist") is structurally unreachable per
the very finding it cites.

## Operator unblock (cohort-wide)

Per audit-2026-04-17 cohort convention:
1. `experiment success-add exp_g4_dare_sparsified_n5 --text "<falsifiable
   statement with epsilon, baseline, OOD/in-dist partition, rescale convention
   s/(1-p) vs 1/(1-p), per-domain n>"`
2. Train 2 Gemma 4 domain adapters (2 × 20.92 min = ~42 min; needs pueue
   handoff) OR re-scope to N=3 with available {code, math, medical}.
3. Pin K1609/K1610: define OOD vs in-dist partition; resolve "beats by 3pp"
   vs "loses 0% in-dist" as pooled/max/median with ε semantics; specify
   rescale formula for Gemma 4 FP16 (F#269's s/(1-p) is ternary-specific).
4. Accept that F#269's impossibility structure forecloses K1610 a priori —
   claim must be restated to allow -<5pp MMLU in-dist degradation.

Without (1)–(4) the claim is not falsifiable; SUPPORTED is impossible.

## References

- F#269 exp_dare_sparsified_composition (DARE p=0.5 recovers code OOD for
  BitNet-2B ternary TernaryLoRA s=20, SUPPORTED 2026-03-31)
- F#306 batched-LoRA framework displacement — ap-017 preempt (a)
- F#13/F#14 1/N regularization non-transfer — ap-017 preempt (b)
- F#44 5-domain real HF non-transfer — ap-017 preempt (c)
- F#45 BitNet-2B ternary convergence non-transfer — ap-017 preempt (d)
- F#164 λ-sweep λ>0.5 untested — ap-017 preempt (e)
- F#269 DARE p=0.5 BitNet-ternary non-transfer — ap-017 preempt (f) NEW
- ap-017 partial-cascade-insufficiency — 18→19 instances
- DARE (arXiv:2311.03099), TIES-Merging (arXiv:2306.01708)

## Routing

analyst iter 19: reference ap-017 scope addendum (18→19 instances; branches
composition-bug 18 + scale-safety 1). Register F#269 DARE non-transfer
one-liner under ap-017 preempts (f). No new antipattern.

reviewer iter 18: verify T1 (find), T3 (experiment get), T4 (K1609/K1610
text), T5 (finding-get F#269 + MATH.md direct impossibility-structure
argument). No runner false-negative. Defense-in-depth T1 ∨ T3 ∨ T4 ∨ T5.

## Assumptions (per guardrail 1007)

- Canonical N for audit-2026-04-17 composition-bug cohort = 5 (from title
  `N=5` and F#269 5-domain reference set).
- T2.1 adapter mean cost reused from cohort wall-clock measurement
  (1255.17s = 20.92 min/adapter). Absent current-session reruns.
- "OOD vs in-dist" partition undefined in K1609/K1610; T4 treats the
  ambiguity itself as the disqualifier rather than picking a partition.
- No N=3 re-scope attempted because the experiment title + cohort convention
  mandate N=5; re-scoping is operator responsibility (step 2 unblock).
