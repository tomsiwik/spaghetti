# REVIEW-adversarial — exp_delta_coding_adversarial_versions

## Verdict
**KILL (ratify preemptive-kill)**

## Summary
43rd preemptive-kill in the audit-2026-04-17 cohort. Composition-bug axis,
sub-variant *parent-finding-contradicts-assumption*. Derivation chain
substitutes a physical run with (i) parent measurement, (ii) F#157,
(iii) F#37, (iv) Eckart–Young–Mirsky; arrives at K#334 drift ≥ 33% ≫ 5%
threshold. Kill is legitimate.

## Adversarial checklist

**Consistency (a–d):**
- (a) `results.json.verdict = "KILLED"` ↔ DB `killed` ✓
- (b) `all_pass = false` ↔ killed ✓
- (c) PAPER.md verdict line = "KILLED on K#334" ↔ DB ✓
- (d) `is_smoke = false`; `run_type = derivation-only` explicit — no
  false claim of full run ✓

**KC integrity (e–g):**
- (e) K#334 (drift>5%) and K#335 (storage>70%) pre-registered on
  experiment record (2026-03-07); no post-hoc modification. Git diff
  of MATH.md has no K-ID changes ✓
- (f) No tautology: drift is a real Frobenius-norm reconstruction metric
  measured against ground-truth v_j; storage ratio is a counting
  argument. Neither passes by identity ✓
- (g) K-IDs in code (334, 335) match MATH.md quantities ✓

**Code ↔ math (h–m2):**
- (h)–(l) N/A — derivation-only stub; no LoRA composition, no
  LORA_SCALE, no routing, no shutil.copy, no hardcoded pass dict in the
  reviewer-meaningful sense (hardcoded `pass=False` is the kill verdict
  being recorded, which is the legitimate use). ✓
- (m) No model loaded; MATH.md and run_experiment.py are consistent
  (both derivation-only). ✓
- (m2) No platform-specific code executed, so skill invocation is N/A.
  This is the intended preempt pattern per PLAN.md §2. ✓

**Eval integrity (n–q):** all N/A for derivation-only preempt.

**Deliverables (r–s):**
- (r) Prediction-vs-measurement table present in PAPER.md ✓
- (s) Math check:
  - Lemma 3 (EYM): correctly stated. ✓
  - Derivation (MATH.md §Proof):
      drift = ‖v_j − v̂_j‖_F / ‖v_j‖_F
            = ‖Δ_ij − SVD_{r=2}(Δ_ij)‖_F / ‖v_j‖_F
            ≥ ‖Δ_disc‖_F / ‖v_j‖_F (assuming Δ_disc lies in truncated tail)
            ≈ √0.80 · 0.37 ≈ 33%.
    The ≥ inequality relies on Δ_disc being (approximately) orthogonal
    to the top-2 singular subspace — which is F#157's mechanism
    statement. ✓
  - Sensitivity check in PAPER.md: even 5% discriminative fraction
    yields drift = √0.05 · 0.37 ≈ 8.3% > 5% — kill robust. ✓
  - K#335 storage = 2(m+n)/(mn) is content-independent; well under 70%
    for all LLM layer shapes. ✓

## What holds
- Parent's 0.37 ‖Δ‖/‖v‖ measurement is real (exp_delta_coding_expert_versions
  PROVEN, 2026-03-07).
- F#157 is supported (2026-03-28) and directly applies — the mechanism
  in question is SVD low-rank approximation of a matrix whose signal is
  cross-domain discriminative, which is exactly what F#157 killed.
- EYM is textbook.
- Sensitivity window (5% discriminative → 8.3% drift) absorbs any
  reasonable relaxation of "adversarial".

## What does not hold (non-blocking)
- The ≥ inequality in the derivation assumes Δ_disc is fully orthogonal
  to the top-2 singular subspace. In practice some of Δ_disc may be
  captured by σ_1, σ_2, reducing drift. However, that would mean
  Δ_disc is *not actually discriminative* (it shares the smooth
  subspace) — contradiction with the adversarial premise. The
  assumption is load-bearing but internally consistent.
- No physical run means there is no error bar on the 33% estimate.
  Acceptable per PLAN.md preempt pattern when the derivation is tight;
  future v2 (rank-adaptive scheduling) lives under a separate
  experiment id.

## Assumptions (reviewer)
- F#157's hierarchical-composition mechanism generalizes to delta-
  compression. Both apply low-rank SVD to signal-is-cross-domain-
  discriminative matrices. The mechanism (averaging shared subspace,
  truncating discriminative tail) is identical. Load-bearing assumption
  flagged but accepted.
- Treat this preempt as 43rd in the cohort, not a novel T5-K axis. The
  axis (composition-bug / parent-finding-contradicts-assumption /
  proxy-with-empirical-refutation) was established at iter 34 (F#451,
  F#1564); this is a re-use, not a new branch.

## Route
- DB `experiment complete --status killed` already executed by researcher iter 49.
- Emit `review.killed` → analyst iter 33 (LEARNINGS capped 50/50 per
  HALT §C; this ratification adds +1 to the analyst-owed debt from
  preempts, carrying debt to 21).
