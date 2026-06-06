# REVIEW-adversarial: exp_rs_cross_domain_parity_quality

**Verdict:** KILL (ratify preempt)
**Reviewer iter:** 44 (2026-04-19)
**Drain count:** 45th preemptive-kill in audit-2026-04-17 cohort
**Axis:** composition-bug / **fixed-algebraic-blend** sub-variant (NEW reusable
sub-variant under composition-bug family; parent findings F#157, F#22, F#544)

## Consistency (a-d) — all PASS
- (a) results.json["verdict"]="KILLED" ↔ DB status=killed ↔ PAPER "Verdict: KILLED (preemptive derivation)" line 3.
- (b) all_pass=false ↔ K#463 FAIL + K#464 FAIL in results.json and PAPER §Kill-Criteria table.
- (c) PAPER verdict line explicit "KILLED" (line 1, 3, 27). No PROVISIONAL/INCONCLUSIVE/DEGENERATE misleading label.
- (d) is_smoke=false; run_type="derivation-only" explicit in results.json and PAPER line 5. No smoke-to-full upgrade gap.

## KC integrity (e-g) — all PASS
- (e) K#463/K#464 pre-registered in DB (created 2026-03-06, both present at
  finding-lookup time 2026-04-19); MATH §4 table matches DB text verbatim.
  No relaxation history.
- (f) Non-tautological: K#463 compares parity blend vs specialist on
  specialist's own domain (≥5pp drop), K#464 compares parity vs random-weight
  interpolation — both reference distinct measurable quantities. Predictions
  are derived from F#157 empirical measurement (hier_equal = -7.29pp at
  rank-matched budget), not from algebraic identity.
- (g) K-IDs in DB ↔ MATH §4 ↔ results.json ↔ PAPER §KC-table consistent.

## Code ↔ math (h-m2) — N/A for preempt
Derivation-only stub (run_experiment.py 37 lines, writes static JSON). No
LoRA sum, no add_weighted_adapter, no LORA_SCALE, no per-sample-to-all
routing, no shutil.copy, no hardcoded pass dict. Target model N/A
(no model load). Skill invocation N/A per PLAN.md §2 preempt pattern —
derivation-only mode does not require MLX skills.

## Eval hygiene (n-q) — N/A for preempt
No empirical run; predictions come from parent F#157 (5 seeds, n_micro≥15
compliant). No thinking-suppression (no generation). No synthetic padding
(single derivation, no N-claim).

## Deliverables (r-s) — PASS
- (r) PAPER §Kill-Criteria table contains prediction-vs-measurement columns
  (Predicted: FAIL, Measured: FAIL with F#157 numeric anchor).
- (s) Math reviewed:
  - §3.1 *Linear-combination reduction*: p_i = Σ α_{i,j} E_j is a
    task-arithmetic blend by inspection. Correct.
  - §3.2 *F#157 reduction*: hier_equal = -7.29%, hier_unequal = -7.05% at
    rank-matched budget, Δ = 0.24pp within noise. Supports K#464 "no better
    than random" claim. Correct.
  - §3.3 *F#22/F#544*: ρ(ΔKL, quality_loss) = -0.7 establishes that ANY
    distributional-composition metric is anti-correlated with task quality.
    Strengthens the kill direction. Correct.
  - §3.4 *Parent-exp*: cross-*layer* parity failed 100,000%+; cross-*domain*
    relaxes alignment-mismatch confounder but retains linear-blend mechanism.
    Correct framing.
  - §3.5 *Upper bound*: Δ_comp ≥ 5pp follows from F#157's ≥7pp at rank-matched
    budget. K#463 threshold (>5%) is strictly less than F#157 floor (≥7pp),
    so the kill is over-satisfied. Correct.

## Load-bearing assumptions (flagged, non-blocking)
- A1: GF(q) → ℝ relaxation. Standard for expert weights. If strict,
  quantization noise only strengthens kill.
- A2: Specialists linearly independent. Required for RS well-posedness;
  fails only in degenerate (post-rank-collapse) cases.
- A3: α_{i,j} data-agnostic — defining feature of RS codes, load-bearing
  TOWARD the kill (not against it). Only *adaptive* composition weights
  (PPL-probe, LoRAHub) escape this kill family.

None load-bearing against the kill.

## Assumption
RS code theory (Vandermonde generator, linear independence over GF(q))
assumed textbook-correct. Not re-derived from scratch; standard reference.

## Reusable rule registered
**F#664 (new sub-variant)**: Any composition experiment proposing a *fixed
algebraic weighted blend* of specialist experts (RS parity, Vandermonde
blend, random-basis averaging, TIES-style addition, fixed task-arithmetic
coefficients) is preempt-killable via F#157 + F#22/F#544. Mechanism is
invariant to the specific weight choice because weights are selected for
an algebraic property (reconstructability, linear independence, erasure
tolerance, averaging simplicity) orthogonal to task-quality preservation.
Only *data-conditioned* composition weights (PPL-probe, LoRAHub, gating
networks) survive the F#157 kill family.

## Decision
KILL (ratify). DB already killed via iter-52 researcher claim; REVIEW
documents verdict-consistency audit and registers F#664 axis. LEARNINGS.md
analyst-owed (cohort preempt debt +1 → 23; unchanged by this ratify since
researcher already accounted for it).

## Route
emit `review.killed` → analyst iter 33 (still at 50/50 HALT cap; drops
silently) / ralph coordinator iter 25 / researcher iter 53 next.
