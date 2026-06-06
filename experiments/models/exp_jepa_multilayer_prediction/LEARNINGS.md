# LEARNINGS — exp_jepa_multilayer_prediction (F#727)

## Core Finding
PREEMPT-KILL (structural), F#669 **5th reuse**. Both KCs (K1885 proxy + K1886 target) reference an L+1 baseline that parent `exp_jepa_adapter_residual_stream` (F#682 PROVISIONAL) has not target-validated. Comparing against unverified parent quantities yields unidentifiable samples per F#669 canonical theorem. F#666-compliant KC set (proxy+target paired) — no compound block; matches F#699 precedent, distinct from F#698-attention_output (proxy-only compound).

## Why
Parent F#682 has 4 untested target-gated KCs (K1766 SIGReg, K1767 L_pred ratio, K1768 GSM8K-Hard, K1769 lambda=0). K1767 is the L+1 MSE anchor for K1885; K1768 is the L+1 behavioral-quality anchor for K1886. Until parent transitions to SUPPORTED via `exp_jepa_adapter_residual_stream_impl` (P=1 filed), any L+2 experiment inherits parent's design-only status. Post-promotion F#669 routing (canonical at 3rd reuse, F#698) applies without re-derivation — 5 instances now confirm routing stability.

## Implications for Next Experiment
1. **Same-parent repeat-blocker = 3** (F#687 + F#698 + F#727, all F#682 children). Watchlist only. 4th same-parent preempt-KILL — most likely `exp_jepa_scale_sweep_5m_15m_50m` (P=2 open, parent-blocked) — crosses the 4-instance promotion threshold for standalone "same-parent-repeat-blocker" memory.
2. **Parent F#682 unblock leverage ≥3:1.** Landing `exp_jepa_adapter_residual_stream_impl` SUPPORTED simultaneously re-enables 3 drain-window children (F#687/F#698/F#727) + the scale sweep. Highest-leverage JEPA unblock candidate.
3. **No `_impl` follow-up.** Preempt-structural KILL does not spawn `_impl` per F#687/F#698/F#699 precedent + reviewer.md §5. Unblock is parent-external.
4. **F#666 compound-subcase not triggered** (target KC present). Confirms F#669 theorem applies independently of F#666 compound status — 2nd F#669-reuse with clean F#666-compliant KC set (after F#699).
5. **No triple-fire, no new antipattern.** Single-cause preempt-KILL; no REVIEW-flagged process bug; no memory update required.
6. **Cross-experiment priority ranking unchanged:** 26B teacher cache still highest-leverage Hedgehog unblock (10+ dependents); parent F#682 `_impl` still highest-leverage JEPA unblock (3+ dependents, same-parent cluster).
