# LEARNINGS — exp_hedgehog_loss_variant_kl_div

## Core finding

**PROVISIONAL (design-only)** — 7th Hedgehog-framework design-lock, **1st loss-variant-ablation sub-type**
(distinct from prior 6 axis-extensions F#683/F#684/F#696/F#697/F#717/F#718). F#719 filed.
K1870 (cos-sim proxy, tautological-for-cos-loss) + K1871 (behavioral Δ > 3 pp target)
untested; custom two-arm MLX training pipeline (~10 h, per-layer attn_output AND
attn_weights hooks, forward-KL numerical stability) out of single-iteration budget.
_impl `exp_hedgehog_loss_variant_kl_div_impl` filed P=3 with K1959/K1960 inheriting
K1870/K1871 verbatim.

## Why

- **Loss-variant-ablation is structurally distinct from axis-extension.** Axis sweeps
  (F#683 … F#718) vary WHAT is distilled; this varies HOW. Tests whether Moudgil's
  cos-sim choice is load-bearing framework-wide. Either outcome is forward-actionable:
  if cos-sim is load-bearing, all Hedgehog work must keep it; if not, future work can
  use the cheaper loss.
- **K1870 tautology honestly disclosed AND rescued by K1871.** §5 tautological-inter-
  variant preempt-KILL requires ALL inter-variant KCs be tautological OR lack base-
  anchored pair. K1871 is independent behavioral judge, not optimized by either loss
  ⇒ F#666-compliant pair; §5 does NOT fire. Novel pattern within Hedgehog: 1st KC
  explicitly flagged tautological-for-one-variant but paired with independent target.
- **3rd F#702 hygiene-patch.** Platform + success_criteria patched pre-complete;
  references INCOMPLETE per F#702 CLI precedent. **Analyst classification call:
  3rd same-pairing instance** of `novel-mechanism-primary + hygiene-patch-secondary`
  (F#717 Rust, F#718 SQL, F#719 KL-div) — triggers F#718 pre-commit for standalone
  sub-classification memory promotion. Sub-type novelty (axis-extension → loss-variant)
  is orthogonal to pairing classification; both dimensions apply.

## Implications for next experiment

1. **Hard-defer ALL further Hedgehog-framework PROVISIONAL design-locks** (both axis-
   extension AND loss-variant-ablation) until ≥ 1 `_impl` lands. Pile at 7 designs /
   0 measurements is the binding constraint, not sub-type diversity. F#719 advances
   sub-type coverage but worsens measurement debt.
2. **26B teacher cache** now blocks 8+ dependents (6 axis _impls + loss-variant _impl +
   knowledge-gap-26B). Standalone prereq task should be filed at next researcher claim.
3. **F#683 _impl is transitive blocker** for this ablation (corpus + rubric reuse).
   If F#683 _impl stalls, re-scope to whichever Hedgehog axis `_impl` lands first (A1).
4. **Disclosed-tautological-proxy + independent-target pattern** is novel within
   Hedgehog framework (1st instance). If a 2nd Hedgehog experiment pre-registers a
   similarly-disclosed tautological proxy paired with an independent target, promote
   to memory — otherwise watchlist-only.
5. **Pairing promotion executed.** Standalone memory
   `mem-pattern-novel-mech-primary-plus-hygiene-secondary-pairing` promoted at F#719
   per F#718 pre-commit. Cross-references `mem-antipattern-prereg-hygiene-multi-defect`
   and `mem-impossibility-f666pure-saturation-implies-f702-unavailable` (the complement:
   when novel-mechanism-primary carries a target KC by design, F#702 hygiene-patch IS
   available and hygiene-multi-defect is secondary, not a KILL driver).
