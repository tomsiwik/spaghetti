# PAPER.md — exp_p9_self_evolution_scaffold

## Verdict: KILLED (preemptive, dependency-unfulfilled + infrastructure-unobtainable)

Two independent structural impossibility arguments concur. No execution. No
training run. No smoke. Pure preemptive kill under the drain-forward rule.

## Hypothesis (as pre-registered)
Replicate MiniMax/Alita self-evolution: let Gemma 4 + adapters optimize its
OWN adapter training/routing scaffold over 20+ rounds (analyze failures,
modify scaffold code, re-run, keep or revert). Target: ≥10% improvement on
held-out benchmark plus ≥3 validated scaffold-bug fixes, with no regression
on unrelated benchmarks.

## Prediction → Measurement table

| KC   | Prediction (if executable)                                         | Measurement   | Verdict       |
|------|--------------------------------------------------------------------|---------------|---------------|
| 1402 | ≥10% benchmark improvement after 20 self-evolution rounds          | not measured  | FAIL (preempt)|
| 1403 | Model identifies and fixes ≥3 scaffold bugs                        | not measured  | FAIL (preempt)|
| 1404 | No regression on unrelated benchmarks during evolution             | not measured  | FAIL (preempt)|

All three are flagged FAIL by the preemptive kill; the underlying truth-value
of the scientific claim is undetermined.

## Theorems (see MATH.md for proofs)
- **T1 (dep-chain unfulfilled, F#669 4th reuse):** every KC reduces to a
  well-defined baseline scaffold Σ_0. Σ_0 is defined as the trained output of
  exp_p9_full_stack_integration, whose K1387–K1389 are all untested and whose
  own deps (cmoe_grassmannian_compose, des_reward_verifier, ttlora_moe_router)
  are open. Σ_0 undefined ⇒ Σ_t undefined for all t ≥ 0 ⇒ KCs are 0/0.
- **T2 (infrastructure-unobtainable, F#658 reuse):** Alita-style self-
  evolution requires code-exec sandbox + MCP server + revert layer + 20-round
  benchmark harness. None exist in this repo; MLX has no published MCP-agent
  framework; M5 48GB cannot co-locate Σ_t and Σ_{t-1} for comparison across
  20 rounds in feasible wall time.
- **T3 (overdetermined):** T1 and T2 are independent — infrastructure alone
  does not create Σ_0, and Σ_0 alone does not create the self-evolution
  harness.

## Findings referenced
- **F#658** — infrastructure-unobtainable on target (reused for
  agent/sandbox/MCP stack absence).
- **F#669** — child-KCs-require-parent-target-claim-unverified. This is the
  **4th reuse** (iter 70 single-parent, iter 71 double-parent F#671, iter 72
  single-parent + F#658 co-preempt F#672, and this iter — same-chain
  grand-parent with multi-branch dep-closure). Promotion to standalone
  finding was proposed at iter 72; 4th reuse strengthens the case.
- **F#672** — CISPO MLX-target preempt, sibling-branch precedent on the
  same P9 full-stack parent chain.

## Assumptions logged
- Autonomy (Guardrail 1007): no user input; most-defensible decision is to
  preempt under two concurring impossibilities rather than attempt a
  20-round agent loop without infrastructure.
- Scope: "M5-scoped smoke build" alternative (e.g. 1-round self-analysis)
  was evaluated and rejected — any sub-20-round variant changes KC text
  (K1402 is explicit on "20 rounds"). Pre-registered KCs are locked per
  PLAN.md §1.
- Terminal floor status: with this preempt, P≤2 open = 0. Criterion #1 of
  the drain objective is satisfied.

## Unblock path (not in drain budget)
1. Sequential parent completion: cmoe_grassmannian_compose +
   des_reward_verifier + ttlora_moe_router → full_stack_integration → here.
2. Build MLX-native agent orchestration layer (code sandbox + MCP server +
   revert layer) — multi-month engineering.
3. Dedicate sustained compute to 20× full-pipeline re-evaluation passes.

None achievable in this drain iteration.

## Related: what a valid follow-up would look like
A micro-scoped variant `exp_followup_self_evolution_1round_mlx` could test
whether Gemma 4 + a single adapter can propose a *single* verifiable
scaffold edit in a controlled sandbox — scope reduced from 20 rounds to 1
round, from 5 domains to 1 domain, from full benchmark to held-out probe.
That would require a new KC formulation that does not reduce to K1387–K1389.
Out of scope for this drain iteration per Guardrail 1008.
