# PAPER.md — exp_jepa_router_prediction_error

## Verdict: KILLED (preempt, F#669)

This experiment was preempt-killed before any MLX code was written. The kill is structural, not empirical: every kill criterion transitively requires the parent `exp_jepa_adapter_residual_stream` to produce target-verified trained JEPA adapters, and the parent is currently `provisional` (F#682 — design-only, no training loop run, all 4 target-gated KCs untested).

## Prediction vs measurement

| KC  | Prediction                                                         | Measurement               | Verdict       |
| --- | ------------------------------------------------------------------ | ------------------------- | ------------- |
| K1  | Routing agreement >70% on N=25                                     | not measured              | untested      |
| K2  | Task-acc under JEPA routing ≥ oracle, \|Δgap\| < 2pp               | not measured              | untested      |
| K3  | Beats softmax-classification router by ≥5pp                        | not measured              | untested      |
| K4  | Per-token latency < 1.2x single adapter forward                    | not measured              | untested      |

**All KC rows are "not measured" because no JEPA-adapters exist to route over.** Measuring against degenerate/untrained `pred_i` predictors would produce vacuous PASS or FAIL — an unidentifiable sample per F#669.

## Assumptions

- `exp_jepa_adapter_residual_stream` will eventually be re-run to full scale via its `_impl` follow-up (already filed P3). If it reaches `supported` with K3+K4 SUPPORTED, this experiment becomes re-claimable.
- No redesign attempted this iteration to avoid the parent dependency (e.g. random-predictor null-router ablation). Out of scope per objective drain.

## Related

- **Finding #669** — defining precedent for preempt-KILL on target-unverified parent.
- **Finding #682** — parent `exp_jepa_adapter_residual_stream` PROVISIONAL.
- **Finding #666** — target-gated KC discipline (K1/K2 pair, K3/K4 pair structure).
- `exp_jepa_adapter_residual_stream_impl` — P3 impl-companion to parent; blocks re-claim of this child.

## Unblock path

Re-claim this experiment when:
1. Parent `exp_jepa_adapter_residual_stream` reaches `status=supported`, AND
2. Parent K3 (GSM8K target accuracy) and K4 (ablation target) SUPPORTED at full scale.

Then the JEPA predictors `pred_i` have target-validated structure and child KCs become measurable.

## Follow-up filed

None — preempt-structural kill does not spawn an `_impl` companion. The unblock is parent-external.
