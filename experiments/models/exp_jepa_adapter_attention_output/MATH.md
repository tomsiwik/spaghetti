# MATH.md — exp_jepa_adapter_attention_output (PREEMPT-KILL)

## Verdict: PREEMPT-KILL

This experiment is preempt-killed per **Finding #669** (preempt-child-KCs-require-parent-target-claim-unverified) before any code was run. Rationale derived below; no `run_experiment.py` MLX implementation is attempted because the parent dependency is in a state that makes every KC structurally untestable.

This is the **3rd reuse** of the F#669 pattern (F#669 → F#687 → this). F#687 explicitly flagged the second-reuse promotion threshold: "Second-reuse threshold for F#669 sub-axis (preempt-child-parent-target-unverified) now reached; promote this sub-axis to standalone canonical finding-family." This 3rd reuse confirms the promotion.

## §0 Platform / skills / model pins

Included for completeness even though no platform code is executed.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). **Not invoked** — no MLX code written; honest disclosure per reviewer checklist item (m2).
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: `v_proj + o_proj` (per F#627).
- Parent dependency: `exp_jepa_adapter_residual_stream` (status `provisional`, F#682).

## §1 Preempt-KILL theorem

**Theorem (inter-experiment target unverifiability, 2-KC variant).** Let `C` denote child experiment `exp_jepa_adapter_attention_output` with kill criteria K = {K1848 (proxy: MSE on attn_output > MSE on residual-stream baseline), K1849 (proxy: SIGReg Epps-Pulley statistic > 0.3)}. Let `P` denote parent experiment `exp_jepa_adapter_residual_stream`.

Every KC in K transitively depends on P producing *a trained JEPA adapter whose residual-stream baseline MSE is target-verified as non-degenerate*.

If `P.status ∈ {provisional, open}` — i.e. no target-verified trained residual-stream adapter exists — then:

- **K1848** ("attn_output MSE > residual_stream MSE baseline"): the right-hand side of the inequality is the parent's unverified quantity. Comparing against an unverified baseline produces either (a) vacuous PASS if the attn_output adapter trains while parent is untrained (attn_output "wins" trivially against an untrained reference), or (b) vacuous FAIL if both collapse (both sides are zero, inequality undefined up to floating-point noise). No informative MSE-ordering can be extracted without the residual-stream baseline being target-validated first.

- **K1849** (SIGReg statistic > 0.3): would be *superficially* measurable on any trained JEPA attn_output adapter in isolation — but its meaning as a KC signal requires the attn_output objective to be itself non-degenerate. The parent's SIGReg-stability claim (K1767/K1769 of F#682) is *also* untested at target scale; without that precedent, a SIGReg statistic measurement on the variant has no ground truth for "collapse threshold." Statistic > 0.3 could mean collapse OR healthy isotropic-Gaussian under a different prediction-space geometry; the variant loses the parent's interpretive anchor.

∴ ∀ k ∈ K: testing `k` while `P.status ≠ supported|proven` produces an unidentifiable sample. **QED.**

### §1.1 Compounding issue: F#666 proxy-only KC set

**Both K1848 and K1849 are PROXY KCs.** The KC set does not include a target-metric pair per F#666. Even if the parent were SUPPORTED, a KILL on this KC set would require a downstream target-metric gate (e.g. GSM8K accuracy, behavioral task quality) — absent from pre-registration. Under F#666 discipline, a proxy-only KC set forbids a KILL verdict on proxy-FAIL alone.

So this experiment has **two independent reasons it cannot be run to a SUPPORTED or KILLED verdict** as currently specified:

1. Parent dependency unverified (F#669 preempt).
2. KC set proxy-only, no target gate (F#666 violation).

Either alone suffices for preempt-KILL. Together they are redundant but mutually reinforcing.

## §2 Prior art

- **F#669** (2026-04-19) established this pattern for `exp_rdt_act_halting_throughput` over `exp_rdt_loop_lora_gemma4`.
- **F#687** (2026-04-23) 2nd reuse: `exp_jepa_router_prediction_error` over parent `exp_jepa_adapter_residual_stream` (same parent as this experiment). 4 KCs preempt-blocked. F#669 caveat triggered: "promoted to standalone finding on second reuse."
- **F#682** (2026-04-23) parent PROVISIONAL: design-only, 4 target-gated KCs untested, MLX training loop not implemented.
- **F#666** — target-gated kill discipline; proxy-only KC sets cannot issue KILL on proxy-FAIL.
- **F#498** (intra-experiment tautology) distinct: this is inter-experiment self-reference, not intra.

## §3 Predictions (registered, not measured)

All KC states are **"untested (preempt-blocked)"**:

| KC    | Claim                                                        | Kind  | Measurement status                      |
| ----- | ------------------------------------------------------------ | ----- | --------------------------------------- |
| K1848 | MSE on attn_output layers > MSE on residual-stream baseline  | proxy | untested (preempt-blocked, F#669)       |
| K1849 | SIGReg Epps-Pulley statistic > 0.3 indicating collapse       | proxy | untested (preempt-blocked, F#669)       |

## §4 Unblock condition

Re-claimable when parent `exp_jepa_adapter_residual_stream` reaches `status=supported` at full scale with its **K3 (GSM8K target-accuracy)** and **K4 (ablation target)** SUPPORTED. At that point, residual-stream baseline MSE and SIGReg collapse threshold become target-validated anchors, and attn_output variant KCs become informative.

**Additional condition to avoid F#666 violation at re-claim time:** KC set must be augmented with a target-metric KC before re-running (e.g. "attn_output JEPA adapter on Gemma 4 E4B matches or beats residual-stream baseline on GSM8K-Hard by ≥0pp absolute"). Without this augmentation, a re-run would hit F#666 even if parent is SUPPORTED.

**Alternative unblock:** redesign child to not depend on parent-trained adapters (e.g. train attn_output + residual-stream baselines *in the same experiment* as a paired A/B). Out of scope for drain window; scope-expand is `_impl`-class work.

## §5 Follow-up

No `_impl` companion filed — preempt-structural kill is self-contained. The unblock condition is external (parent's existing `exp_jepa_adapter_residual_stream_impl` at P=3; if that lands SUPPORTED, both K1848 baseline and K1849 collapse-threshold become measurable).

Unlike novel-mechanism PROVISIONAL (F#682, F#683, F#684, F#696, F#697) where an `_impl` follow-up is mandatory, preempt-structural KILL does NOT spawn an `_impl`. This distinction is canonical per F#687 and reviewer.md §5.

## §6 Scope integrity

No silent objective swap (antipattern-t): this scaffold does NOT attempt a "simpler" ablation (e.g. untrained-predictor SIGReg measurement). The pre-registered KCs are preserved verbatim; they are marked `untested (preempt-blocked)`, not replaced.

No `_impl` inline-file obligation (`mem-antipattern-impl-follow-up-delegation`): that antipattern applies to novel-mechanism PROVISIONAL. Preempt-KILL is structurally distinct and explicitly **does not** spawn `_impl` per F#687 precedent.
