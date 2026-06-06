# MATH.md — exp_p9_self_evolution_scaffold (PREEMPTIVE KILL)

## Status: KILLED (preemptive, dependency-chain-unfulfilled + platform-mismatch)

Two independent impossibility arguments — F#669 reuse (4th occurrence, confirms
promotion candidate from iter 72) AND F#658 reuse (agent-orchestration
infrastructure unobtainable on MLX target).

## Experiment claim (per DB)
- K1402 (K1): ≥ 10% improvement on target benchmark after 20 self-evolution rounds.
- K1403 (K2): Model successfully identifies and fixes at least 3 scaffold bugs.
- K1404 (K3): No regression on unrelated benchmarks during evolution.

## Dependency graph (verified via `experiment get` 2026-04-19)
```
exp_p9_self_evolution_scaffold              (P2, open → active, MACRO)
    └── exp_p9_full_stack_integration       (P5, OPEN — no trained artifact)
            ├── exp_p9_cmoe_grassmannian_compose     (open — untested)
            ├── exp_p9_des_reward_verifier           (open — untested)
            └── exp_p9_ttlora_moe_router             (open — untested)
```

Sibling evidence: exp_p9_cispo_adapter_rl (same parent chain, different branch)
was KILLED at reviewer iter 63 under identical F#669+F#658 co-preempt, with
F#672 registered.

Every KC of the target transitively requires:
1. A **trained full-stack adapter pipeline** (parent full_stack_integration
   K1387–K1389 all untested).
2. A **self-evolution loop harness** (code sandbox + MCP-server agent loop per
   Alita arxiv reference) — none exists in this repo.

## Theorem 1 (dep-unfulfilled — F#669 reuse, 4th occurrence)

Let Σ_t denote the scaffold (code + routing + training hparams) at
self-evolution round t, let acc_B(Σ_t) be accuracy of the pipeline configured
with Σ_t on benchmark B, and let ℬ be the set of "bugs" in Σ_t that the model
proposes and validates to fix.

**Claim.** Without a trained full-stack pipeline Σ_0 at t=0, all three KCs are
structurally unmeasurable (0/0).

**Proof.**
1. K1402 measures acc_B(Σ_20) − acc_B(Σ_0). Σ_0 is defined as "the existing
   full-stack pipeline from exp_p9_full_stack_integration at time of launch."
   If exp_p9_full_stack_integration has no trained artifact (K1387–K1389
   untested), Σ_0 is undefined. Σ_t for t > 0 is inductively undefined since
   its construction requires Σ_{t−1}. ⊥
2. K1403 measures |ℬ| ≥ 3 where ℬ = bugs-identified-and-fixed-by-model. A
   "bug" presupposes a baseline behavior to deviate from — that baseline is
   Σ_0. Without Σ_0, the notion of "scaffold bug" reduces to arbitrary code
   edit and is not well-defined. ⊥
3. K1404 measures regression on "unrelated benchmarks" across rounds. Rounds
   are indexed by Σ_t, which requires Σ_0 per (1). ⊥

All three reduce to parent K1387–K1389 ("full stack outperforms base Gemma 4
by ≥15pp on GSM8K, etc."). The target claim is an **inter-experiment
tautology**: `{K1402, K1403, K1404} ⊆ dep-closure({K1387, K1388, K1389})`.
QED.

## Theorem 2 (platform-mismatch — F#658 reuse)

Let 𝓟 = {MLX, Apple Silicon, M5 Pro 48GB} be the target platform
(PLAN.md §II, `feedback_mlx_first.md`).
Let 𝓐 = Alita-style self-evolution dependencies =
{code-execution sandbox, MCP server orchestration, git-tracked revert layer,
benchmark runner harness for 20+ round re-evaluation}.

**Claim.** 𝓐 ⊄ repo capabilities and 𝓐 is not MLX-implementable within the
drain budget; therefore exp_p9_self_evolution_scaffold cannot be built on
target without building additional agent infrastructure.

**Proof.**
- Alita (arxiv reference in DB notes) relies on a Python code sandbox with
  external tool registration via MCP. The `llm` repo has no MCP server, no
  Python-exec sandbox, no benchmark runner for 20-round re-evaluation.
- Each self-evolution round re-runs the full benchmark pipeline. With
  Gemma-4 + 5-domain pipeline (per full_stack_integration K1389 "5 domains"),
  a single round is multi-hour on M5 Pro; 20 rounds ≫ drain budget.
- Without the sandbox/MCP/revert infrastructure, "model modifies its own
  scaffold code" is ill-defined (there is no scaffold to mutate beyond the
  non-existent Σ_0). Even with infrastructure, the M5 48GB memory budget
  cannot keep Σ_t and Σ_{t−1} co-resident for comparison.
- MLX has no published MCP-agent framework; `feedback_mlx_first.md` forbids
  building on CUDA alternatives (LangGraph/OpenAgents/CUDA sandbox).

Therefore the experiment violates either the target-platform constraint or
the infrastructure-available-in-repo constraint. An
infrastructure-unobtainable structural floor prevents execution.
QED (F#658 pattern).

## Theorem 3 (combined — double-axis structural preempt)

T1 (dep-unfulfilled) AND T2 (infrastructure-unobtainable) are **independent**:
- Even if agent/sandbox/MCP infrastructure appeared tomorrow, Σ_0 is still
  undefined (full_stack_integration has no trained artifact).
- Even if full_stack_integration completed tomorrow, the 20-round loop still
  has no harness.

Two independent impossibility axes concur → preempt is **overdetermined**.

## Kill-criteria disposition (pre-registered, not modified)
| KC   | Predicted | Measured     | Verdict |
|------|-----------|--------------|---------|
| 1402 | fail      | not measured | FAIL (preempt) |
| 1403 | fail      | not measured | FAIL (preempt) |
| 1404 | fail      | not measured | FAIL (preempt) |

## Findings reused
- **F#658** (infrastructure-unobtainable, s1/d1 axis) — reused for
  agent-orchestration infra unavailable on MLX target.
- **F#669** (child-KCs-require-parent-target-claim-unverified, inter-
  experiment dep-chain) — **4th reuse**. Prior reuses: iter 70 (single-
  parent), iter 71 (double-parent variant, F#671), iter 72 (single-parent
  with F#658 co-preempt, F#672). Promotion to standalone finding was
  proposed at iter 72; this 4th reuse reinforces the promotion case.
- **F#672** (CISPO MLX-target preempt, sibling-chain precedent) — cited as
  structural precedent, same parent chain.

## What would unblock this experiment
1. Completion of exp_p9_full_stack_integration with all KCs PASS (requires
   cmoe_grassmannian_compose, des_reward_verifier, ttlora_moe_router all
   SUPPORTED) — currently all OPEN.
2. An MLX-native agent/sandbox framework with 20-round benchmark harness
   (multi-month engineering, off-scope).
3. Change of target platform (violates `feedback_mlx_first.md`).

None achievable in the drain-loop's iteration budget. Preempt stands.
