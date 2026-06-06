# MATH.md — exp_jepa_router_prediction_error (PREEMPT-KILL)

## Verdict: PREEMPT-KILL

This experiment is preempt-killed per **Finding #669** (preempt-child-KCs-require-parent-target-claim-unverified) before any code was run. Rationale derived below; no `run_experiment.py` implementation is attempted because the parent dependency is in a state that makes every KC structurally untestable.

## §0 Platform / skills / model pins

Included for completeness even though no platform code is executed.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). Not invoked — no MLX code written.
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627).
- Adapter targets: `v_proj + o_proj` (per F#627).
- Parent dependency: `exp_jepa_adapter_residual_stream` (status `provisional`, F#682).

## §1 Preempt-KILL theorem

**Theorem (inter-experiment target unverifiability).** Let `C` denote child experiment `exp_jepa_router_prediction_error` with kill criteria K = {K1 (proxy routing agreement), K2 (target oracle-gap), K3 (target beats-softmax-router), K4 (target serving-latency)}. Let `P` denote parent experiment `exp_jepa_adapter_residual_stream`. Every KC in K transitively depends on P producing *trained JEPA adapters whose predictors `pred_i` are measurable functions of prompt-specific residual-stream dynamics*.

If `P.status ∈ {provisional, open}` — i.e. no target-verified trained adapters exist — then:
- For K1: "routing agreement with classification baseline" is undefined when no adapters have been trained (`pred_i` undefined for all i).
- For K2: "task accuracy under JEPA routing ≥ oracle routing" is vacuous when the routing function is evaluated on untrained-or-nonexistent predictors — the argmin_i pred_i residual is dominated by initialization noise, not learned dynamics.
- For K3: "beats softmax classification router by ≥5pp" requires a routing signal distinct from classification; that distinction is load-bearing on *predictor training* producing structure, which is exactly P's unverified claim.
- For K4: "per-token routing latency <1.2x single adapter forward" is measurable on arbitrary tensors but vacuous as a KC signal when the underlying JEPA-adapter architecture has not been target-validated — a latency measurement on a degenerate predictor is uninterpretable.

∴ ∀ k ∈ K: testing `k` while `P.status ≠ supported|proven` produces either vacuous PASS (init-artifact) or vacuous FAIL (uninformative), i.e. an unidentifiable sample. **QED.**

## §2 Prior art

- **F#669** (2026-04-19) established this pattern for `exp_rdt_act_halting_throughput` over `exp_rdt_loop_lora_gemma4`. Same structural claim: child KCs require parent target-SUPPORTED to avoid vacuousness.
- **F#682** (2026-04-23) explicitly flagged `exp_jepa_adapter_residual_stream` as novel-mechanism PROVISIONAL (design-only, 4 target-gated KCs untested, no MLX training loop implemented).
- **F#498/F#666** (intra-experiment tautology/target-gated kill) distinct: this is inter-experiment self-reference, not intra.

## §3 Predictions (registered, not measured)

All KC states are **"untested (preempt-blocked)"**:

| KC  | Claim                                                | Measurement status        |
| --- | ---------------------------------------------------- | ------------------------- |
| K1  | Routing agreement >70% on N=25                       | untested (preempt-blocked) |
| K2  | JEPA routing task-acc >= oracle, |Δgap| < 2pp        | untested (preempt-blocked) |
| K3  | Beats softmax-classification router by ≥5pp          | untested (preempt-blocked) |
| K4  | Per-token latency < 1.2x single adapter forward      | untested (preempt-blocked) |

## §4 Unblock condition

Re-claimable when parent `exp_jepa_adapter_residual_stream` reaches `supported` at full scale with K3/K4 SUPPORTED (the target KCs that prove adapters encode residual-stream dynamics — not merely SIGReg-stable tensors). At that point, JEPA-adapters exist and their `pred_i` predictors have target-validated structure; child KCs become testable.

Alternative unblock: re-design child to NOT depend on trained JEPA-adapters (e.g. use SIGReg-stable random predictors as a null-router ablation). Currently out of scope.

## §5 Follow-up

`exp_jepa_adapter_residual_stream_impl` already filed at P3 (micro) as the impl-companion to the parent PROVISIONAL. When that lands target-SUPPORTED, this child can be re-claimed.

No `_impl` companion needed for this child because the kill is preempt-structural, not design-incomplete.
