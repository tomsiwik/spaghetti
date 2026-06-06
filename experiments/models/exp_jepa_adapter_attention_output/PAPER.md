# PAPER.md — exp_jepa_adapter_attention_output

## Verdict: KILLED (preempt, F#669 — 3rd reuse)

This experiment was preempt-killed before any MLX code was written. The kill is structural, not empirical: both kill criteria transitively require the parent `exp_jepa_adapter_residual_stream` to produce target-verified trained JEPA adapters with a validated residual-stream baseline MSE and SIGReg collapse threshold. Parent is currently `provisional` (F#682 — design-only, no training loop run, all 4 target-gated KCs untested).

Secondary block: the pre-registered KC set is proxy-only (K1848 MSE comparison, K1849 SIGReg statistic). Under F#666 target-gated-kill discipline, a proxy-only KC set cannot issue a KILL verdict on proxy-FAIL alone. See MATH.md §1.1.

## Prediction vs measurement

| KC    | Prediction                                                    | Kind  | Measurement  | Verdict   |
| ----- | ------------------------------------------------------------- | ----- | ------------ | --------- |
| K1848 | MSE on attn_output layers > MSE on residual stream baseline   | proxy | not measured | untested  |
| K1849 | SIGReg Epps-Pulley statistic > 0.3 indicating collapse        | proxy | not measured | untested  |

**Both rows are "not measured" because (1) no residual-stream baseline MSE has been target-validated (parent is PROVISIONAL) and (2) the KC set lacks a target-metric gate per F#666.** Measuring against degenerate/untrained parent adapters would produce vacuous PASS or FAIL — an unidentifiable sample per F#669.

## Assumptions

- `exp_jepa_adapter_residual_stream` will eventually be re-run to full scale via its `_impl` follow-up (already filed P=3 as `exp_jepa_adapter_residual_stream_impl`). If it reaches `supported` with K3+K4 SUPPORTED, this experiment becomes re-claimable **after** pre-registration is augmented with a target-metric KC (F#666 compliance).
- No redesign attempted this iteration to avoid the parent dependency (e.g. train attn_output + residual-stream paired A/B in one experiment). Out of scope per drain objective; scope-expand would be `_impl`-class.

## Pattern promotion

F#687 (2nd reuse of F#669) flagged the caveat: "Second-reuse threshold for F#669 sub-axis (preempt-child-parent-target-unverified) now reached; promote this sub-axis to standalone canonical finding-family." This 3rd reuse confirms the promotion. Canonical routing pattern should be added to `.ralph/hats/reviewer.md` §5 alongside novel-mechanism / macro-scope PROVISIONAL sub-cases.

## Related

- **F#669** — defining precedent for preempt-KILL on target-unverified parent.
- **F#687** — 2nd reuse; same parent (`exp_jepa_adapter_residual_stream`), different child.
- **F#682** — parent PROVISIONAL.
- **F#666** — target-gated KC discipline; proxy-only KC sets cannot issue KILL.
- `exp_jepa_adapter_residual_stream_impl` — P=3 impl-companion to parent; blocks re-claim of this child.

## Unblock path

Re-claim this experiment when ALL of:

1. Parent `exp_jepa_adapter_residual_stream` reaches `status=supported`.
2. Parent K3 (GSM8K target accuracy) and K4 (ablation target) SUPPORTED at full scale.
3. **Pre-registration augmented with a target-metric KC** for this experiment (e.g. "attn_output JEPA adapter matches or beats residual-stream baseline on GSM8K-Hard task accuracy by ≥0pp absolute"). Without this, re-run hits F#666 even if (1) and (2) are satisfied.

## Follow-up filed

None — preempt-structural kill does not spawn an `_impl` companion (per F#687 precedent + reviewer.md §5). The unblock is parent-external.
