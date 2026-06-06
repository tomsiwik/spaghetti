# REVIEW — exp_pierre_ace_merging_b_only

**Verdict: KILLED — no revision needed.**

## Checklist

- [x] MATH.md: Hypothesis stated, adaptation to shared-A explained, KCs pre-registered with verdict table
- [x] run_experiment.py: Uses shared eval_runner, correct KC thresholds, clean delegation
- [x] compose_methods.py: Faithful port of ACE algorithm (arxiv 2603.02945 + released code), well-documented
- [x] results.json: All methods measured, KC pass/fail computed, verdict correct
- [x] PAPER.md: Root cause analysis of failure, implications for future method selection

## Soundness

The -38pp collapse (26.7% vs 64.7% Fisher-Rao) is catastrophic and unambiguous. No tuning of eps/tau/k_frac could recover 38pp — the failure is structural.

Root cause is correctly identified: ACE's Theorem 1 derives per-task covariance from ΔW_t^T ΔW_t. Under shared-A, all deltas factor as s·A·B_t, so covariances are dominated by A^T·A eigenstructure. The inverse (D^{-1}) amplifies this near-rank-deficiency, producing a degenerate merge. This is a mathematical incompatibility, not an implementation bug.

## Potential concerns checked

1. **Implementation fidelity**: compose_methods.py follows the released code's structure (centering, covariance, heterogeneity flag, regularized inverse, spectral refinement). No obvious deviations.
2. **Reference baselines stable**: Fisher-Rao 64.7% and DARE 71.3% match prior experiment values — the eval harness is consistent.
3. **MedQA at 48%**: Least degraded because MCQ with 4-5 choices has a ~25% floor. Not evidence of partial success.

## Implications logged in PAPER.md

The generalization — "methods inferring per-task structure from ΔW are incompatible with shared-A" — is the key takeaway. Correctly flags RegMean and delta-derived Fisher as similarly suspect.

## Decision

Kill confirmed. No revisions. Forward to Analyst for LEARNINGS.md.
