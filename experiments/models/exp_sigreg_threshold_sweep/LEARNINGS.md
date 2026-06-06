# LEARNINGS.md — exp_sigreg_threshold_sweep

## Core finding

Preempt-structural KILL (triple-fire: F#666-pure + §5 intra-detector-threshold-delta + F#669). 
**1st cross-parent instance** of the structural/parent-dependent triple-fire sub-composition
(prior instances F#728/F#729 were both F#682 children; this is F#713). The composition is
robust across parents, not tied to JEPA residual-stream geometry.

## Why

- K1890 (FPR at τ=0.05) and K1891 (FNR at τ=0.20) are both classification-accuracy proxies
  (guardrail 1007 canonical). No target-gated companion KC → F#666-pure (18th drain-window
  reuse).
- Sweeping τ ∈ {0.05, 0.10, 0.15, 0.20} on a single fixed detector yields a monotone
  FPR/FNR ROC curve. Inter-τ rank order is a tautology of the detector — no external
  anchor → §5 intra-detector-threshold-delta (12th §5 reuse; 2nd intra-instantiation
  sub-variant after F#712's intra-adapter-rank-delta).
- Default τ=0.10 is supplied by parent F#713 (`exp_sigreg_composition_monitor`, PROVISIONAL
  design-lock, empirical deferred). Any comparison vs 0.10 references unverified RHS →
  F#669 8th reuse. First same-parent-F#713 F#669 invocation.

## Implications

1. **Cross-parent triple-fire is real.** Structural/parent-dependent composition
   (F#666-pure + §5 + F#669) was first observed with JEPA F#682 parent (F#728/F#729).
   This extends it to SIGReg F#713 parent. The composition is driven by
   "proxy-only KCs + intra-single-detector sweep + PROVISIONAL parent whose default
   is the anchor" — a pattern that will recur wherever a PROVISIONAL mechanism has a
   default parameter and children sweep that parameter.
2. **Strict "same-parent-F#682" promotion trigger does NOT fire.** Parent here is
   F#713, not F#682. The cross-parent generalisation may still warrant analyst
   attention — watchlist for a 2nd cross-parent instance before promoting a
   "generalised parent-target-unverified under intra-sweep" memory.
3. **Same-parent-F#713 F#669 census = 1.** exp_sigreg_hedgehog_combined (F#714)
   did not invoke F#669 (composition was different). So this is F#713's first
   F#669-child. Watchlist: 2nd F#713-child with F#669 triggers analyst attention.
4. **Re-claim conditions** (stringent): (a) pair each proxy KC with a downstream-task
   target metric, (b) wait for F#713 `_impl` to produce empirical ground-truth events,
   (c) anchor the threshold sweep via external Neyman-Pearson / ROC argument.
   Until all three are met, any re-claim re-triggers the same triple-fire.
5. **No `_impl` follow-up.** Per preempt-structural KILL precedent, no child
   experiment is filed — this is a KC-design failure, not a mechanism failure.
