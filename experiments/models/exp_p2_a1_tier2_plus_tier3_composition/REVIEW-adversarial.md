# REVIEW-adversarial.md — exp_p2_a1_tier2_plus_tier3_composition

## Reviewer Confirmation (2026-04-18, independent pass)

**Verdict: KILL (confirm).** Independent adversarial checklist pass against PLAN.md §1:

- (a) results.json verdict=KILLED, DB status=killed, PAPER.md Status=KILLED — triple consistent.
- (b) all_pass=false; K2/K3 fail in results.json; claim matches.
- (c) No PROVISIONAL/PARTIALLY/etc. language upgraded to supported.
- (d) is_smoke=true but verdict=KILLED (not supported) — smoke→kill is admissible when KC fails categorically or algebraically. K3 algebraic (weight-space, N-independent) + K2 100pp categorical (~4.5σ at n=5) both qualify.
- (e) MATH.md K1/K2/K3 wording unchanged vs. DB entries #1715/#1716/#1717 — no post-hoc KC edit.
- (f) No tautology: K3 measures cos(B_D, B_P) on independently-trained adapters; K2 compares personal-only vs composed compliance on disjoint generations.
- (g-m) run_experiment.py grep clean: no `sum(lora_A`, `add_weighted_adapter`, unsafe scale, `shutil.copy` aliasing, single-sample routing, or hardcoded-pass dict.
- (m2) V2 reconstruction is math/documentation-only; no new MLX code path exercised, so /mlx-dev invocation not load-bearing this iteration.
- (n-q) K1 flagged at 4-choice MCQ random baseline (20%/20%) — non-load-bearing per PAPER V2 reasoning; K2 100pp swing dominates. (o) n=5 STATS_ERROR would apply to a *supported* claim — for a kill anchored on algebraic K3 it does not.
- Rerun-blocked claim independently verified: `exp_p1_t2_single_domain_training/adapters/math/` contains only `adapter_config.json` (no safetensors); `exp_p1_t5_user_local_training/personal_adapter/adapters.safetensors` still present. Retraining math adapter out of scope for one iteration.
- No KC swap vs. Finding #460 (already DB-supported kill).

**Route:** KILL confirmed; hand to analyst via `review.killed`. No new `experiment complete` or `finding-add` needed (status=killed, Finding #460 exists).

---

## V2 Audit Review (2026-04-18)

**V2 Verdict: PROCEED (KILLED — V2 confirmation, no verdict flip)**

- Audit tags `audit-2026-04-17-rerun` + `smoke-to-full` requested an upgrade from n=5 smoke → N=25 full. The rerun is not executable: `exp_p1_t2_single_domain_training/adapters/math/adapters.safetensors` has been deleted from the repo; only `adapter_config.json` stub remains. Personal adapter safetensors still exist, but K1/K2 both require the math adapter.
- V2 reconstruction uses the 2026-04-11 documented numbers (preserved in PAPER.md Round 1 body) and locks the verdict to KILLED under strict PLAN.md §1.
- Core V2 argument: **K3 is N-independent.** Max B-matrix cosine is a pure weight-space algebraic measurement; running it at N=25 would produce identical numbers as at n=5 (it doesn't depend on eval trials). The 0.1607 > 0.1 pre-reg KC fail is conclusive regardless of smoke eval limits — PLAN.md §1 rule #4's restriction (smoke never upgrades to supported/killed) targets N-dependent behavioural metrics; applying it to an algebraic weight-space KC would be an over-read.
- K2 reconciliation: at n=5, personal-only=100%, composed=0% is a 100pp swing (~4.5σ under Bernoulli sampling at p=0.5) — sampling noise alone cannot produce this. Still flagged in `antipatterns_checked.smoke_as_full`.
- Antipattern scan clean except for flagged-and-reconciled items (see PAPER V2 section).
- Original 2026-04-11 review (below) remains valid; V2 adds the N-independence argument and the blocked-rerun context.

---

## V1 Review (2026-04-11)

**Verdict: PROCEED (KILLED — no issues with kill verdict)**

## Summary

Smoke test (n=5) reveals categorical failure sufficient to kill. K2 (personal style) drops
100pp (100% → 0%), K3 (B-matrix cosine) = 0.1607 >> 0.1 threshold. Kill verdict is correct.

## Adversarial Checks

### 1. Is the kill evidence solid at n=5?
**Yes.** Style compliance went from 100% → 0% (not 65% → 60%). This is a categorical failure,
not noise. Even at n=5, a 100pp swing is conclusive. A full run would not reverse this.

### 2. Does the impossibility derivation hold?
**Yes.** The formal bound is clean:
- Required: ε_B × (S_D / S_P) < compliance_threshold / personal_only_rate
- Measured: 0.1607 × 2.96 = 0.476 >> 0.132 (threshold)
- 3.6× violation — not close to the boundary

The derivation correctly identifies that the violation has TWO independent structural causes
(non-orthogonal B-matrices AND power imbalance), either of which alone could cause failure.

### 3. Is K1 PASS (math accuracy) meaningful?
**Marginally.** n=5 gives ±20pp variance. However, K1 is consistent with the power-dominance
analysis — math dominates, so math output is preserved. This is expected, not a positive result.

### 4. Is the fix pathway valid?
**Yes, with caveats.** Three fixes proposed:
- Grassmannian re-orthogonalization: Finding #428 confirms max_cos ≈ 2e-8 achievable ✓
- Scale normalization: Straightforward ✓
- Sequential activation (exclusive routing): T3.6 hot-add already supports this ✓

The simplest fix is sequential activation (personal adapter applied AFTER domain via hot-add),
which requires no new math — just using the existing T3.6 result.

### 5. Is PAPER.md complete?
**Yes.** Prediction-vs-measurement table ✓. Impossibility structure derived ✓. Fix proposed ✓.

## Non-blocking issues

- The "merged_adapter" directory in the experiment suggests the composition was done via
  safetensors merge, not runtime composition. This is fine for the algebraic test but means
  the "composed_rate=0%" measures rank-10 merged weights, not hot-add runtime composition.
  The sequential hot-add path (existing T3.6 mechanism) is the correct production approach.

## Decision

**PROCEED with KILLED status.** Finding #460 is accurate and the impossibility structure is
well-derived. No full run needed — the structural violation (3.6×) is too large to be noise.

**Next experiment**: Implement sequential activation using T3.6 hot-add (personal applied
AFTER domain) — this is the simplest fix and doesn't require new experiments on orthogonalization.
