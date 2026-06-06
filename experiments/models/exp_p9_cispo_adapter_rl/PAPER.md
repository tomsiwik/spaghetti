# PAPER.md — exp_p9_cispo_adapter_rl

## Verdict: KILLED (preemptive, dependency-unfulfilled + platform-mismatch)

Two independent structural impossibility arguments concur. No execution. No
training run. No smoke. Pure preemptive kill under the drain-forward rule.

## Hypothesis (as pre-registered)
Port MiniMax's CISPO (Clipped Importance Sampling Policy Optimization,
arXiv:2506.13585) to adapter training. Hypothesis: CISPO preserves rare-token
gradients that PPO clips away, yielding stronger reasoning adapters than SFT.

## Prediction → Measurement table

| KC   | Prediction (if executable)                                       | Measurement   | Verdict       |
|------|------------------------------------------------------------------|---------------|---------------|
| 1399 | CISPO outperforms SFT by ≥ 5 pp on GSM8K                         | not measured  | FAIL (preempt)|
| 1400 | CISPO grad-magnitude ratio > 2× vs PPO on bottom-10% tokens      | not measured  | FAIL (preempt)|
| 1401 | No reward hacking / collapse over training                       | not measured  | FAIL (preempt)|

All three are flagged FAIL by the preemptive kill; the underlying truth-value
of the scientific claim is undetermined.

## Theorems (see MATH.md for proofs)
- **T1 (dep-chain unfulfilled, F#669 3rd reuse):** every KC transitively
  reduces to parent exp_p9_unsloth_rl_environment K1393, which itself depends
  on exp_p9_full_stack_integration. Both parents OPEN ⇒ no trained
  Φ_CISPO exists ⇒ KCs are 0/0.
- **T2 (platform-mismatch, F#658 reuse):** Unsloth-RL is Triton/CUDA only;
  CISPO's MiniMax codebase is CUDA+Megatron+DeepSpeed. MLX has no CISPO port.
  The target platform is MLX on Apple Silicon (PLAN.md §II,
  `feedback_mlx_first.md`). Therefore 𝓤 ∩ 𝓟 = ∅.
- **T3 (overdetermined):** T1 and T2 are independent — the preempt is
  doubly-structural, not a single-axis judgment call.

## Findings referenced
- **F#658** — infrastructure-unobtainable on target (reused here for
  MLX-platform incompatibility with Unsloth/CISPO).
- **F#669** — child-KCs-require-parent-target-claim-unverified. This is the
  **3rd reuse** (iter 70 single-parent, iter 71 double-parent, iter 72
  single-parent chain with platform co-preempt). Propose **promotion** from
  sub-axis to standalone finding.
- **F#671** — F#669 double-parent variant (precedent registration).

## Assumptions logged
- Autonomy (Guardrail 1007): no user input; most-defensible decision is to
  preempt under two concurring impossibilities rather than attempt a
  non-MLX build or await parent completion outside drain budget.
- Scope: "M5-scoped smoke build" alternative (per event payload) was evaluated
  and rejected — CISPO smoke requires at minimum Unsloth + PyTorch CUDA stack,
  which doesn't run on M5.

## Unblock path (not in drain budget)
Sequential parent completion requires:
1. exp_p9_full_stack_integration — not even in backlog
2. exp_p9_unsloth_rl_environment — OPEN P3 with MLX-port engineering of
   Unsloth kernels (months of work)
3. Then a CISPO MLX implementation (currently nonexistent)

None achievable in this loop. Preempt stands.

## Related: what a valid follow-up would look like
An MLX-native RL loop using an importance-ratio clipping scheme (CISPO's core
insight — clip weights, not probabilities) could be designed directly on top
of MLX's existing gradient machinery. That would be a new experiment under a
different architecture with pre-registered KCs that do **not** reduce to
parent K1393/K1390. Out of scope for this drain iteration.
