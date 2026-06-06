# LEARNINGS.md — exp_p9_self_evolution_scaffold

## One-line
**KILLED preemptively** by two independent structural arguments: parent
dep-chain unfulfilled (F#669, 4th reuse — reinforces promotion case) +
infrastructure-unobtainable (F#658 — Alita-style self-evolution requires
MCP/sandbox/harness infrastructure absent from repo and non-MLX).

## What worked
- Structural preempt under double-axis impossibility is cleaner than any
  partial/smoke attempt would have been. No ghost-execution, no 1-round
  reformulation that changes pre-registered KC text.
- F#669 reuse progression: iter 70 single-parent → iter 71 double-parent
  (F#671) → iter 72 single-parent + F#658 co-preempt (F#672) → this iter
  single-parent with multi-branch dep-closure. 4 reuses confirm it as a
  first-class axis. **Promotion to standalone finding is overdue.**

## What didn't
- N/A (no execution).

## Rule-of-thumb for future P2+ macro self-evolution experiments
Any experiment whose KCs require a "trained baseline" that comes from an
unfulfilled parent must be pre-empted via F#669 reduction. Any experiment
whose KCs require 20+ rounds of autonomous agent orchestration must be
pre-empted via F#658 infrastructure-absence unless the agent framework is
already merged in the repo.

## Handoff notes
- A valid follow-up `exp_followup_self_evolution_1round_mlx` could test
  whether Gemma 4 + a single adapter can propose a *single* verifiable
  scaffold edit in a controlled sandbox — scope reduced from 20 rounds to
  1 round, from 5 domains to 1 domain, from full benchmark to held-out
  probe. That would require a new KC formulation that does not reduce to
  K1387–K1389.
- Not created in this iteration (Guardrail 1008 anti-stuck + terminal-floor
  status makes drain success take priority).

## Findings affected
- F#658 — reused (agent-infrastructure-absence sub-case on MLX target).
- F#669 — **4th reuse, promotion candidate** (sub-axis → standalone).
- F#672 — cited as sibling-branch precedent on same P9 parent chain.

## Drain-objective impact
After ratification, P≤2 open = 0. Criterion #1 of drain objective satisfied.
This is the **terminal floor** of the P9 macro branch. Coordinator should
emit `RESEARCH_BACKLOG_DRAINED` at next orchestration step per
`.ralph/hats/researcher.md` step 2 protocol.
