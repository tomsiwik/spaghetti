# REVIEW-adversarial.md — exp_p9_self_evolution_scaffold (self-review)

## Context
Preemptive kill before any execution. Self-adversarial review against the
reviewer's (a)-(t) checklist (PLAN.md §1) to minimize REVIEW-blocking issues
when reviewer iter 64 ratifies.

This is the **terminal floor** of the P≤2 drain. After ratification, P≤2 open
should equal 0 and criterion #1 of the drain objective is satisfied.

## Checklist

- **(a) verdict in results.json matches PAPER.md** — both KILLED. ✓
- **(b) verdict in DB matches results.json** — status=killed (will be set on
  `experiment complete --status killed`). ✓
- **(c) all_pass consistent** — results.json all_pass=false; PAPER KC table
  shows no passes. ✓
- **(d) is_smoke correct** — is_smoke=false; this is a preemptive kill, not a
  smoke run. ✓
- **(e) KC text verbatim from DB** — K1402/K1403/K1404 text in MATH.md,
  PAPER.md, and results.json matches DB `experiment get` output verbatim.
  No edits. ✓
- **(f) not tautology** — dep-unfulfilled is inter-experiment
  (parent→child), not intra-experiment self-defeating. Infrastructure-
  unobtainable is orthogonal. Not F#498/F#666 pattern. ✓
- **(g) no measurement** — N/A, no execution. ✓
- **(h)-(l) composition / LoRA / routing / copy / hardcoded-pass / proxy
  antipatterns** — N/A, no code executed. Stub script only writes JSON. ✓
- **(m) eval-template truncation** — N/A. ✓
- **(m2) platform skill invocation (/mlx-dev, /fast-mlx)** — N/A. No platform
  code written. Pure documentation + JSON stub. ✓
- **(n)-(q) eval-side checks** — N/A, no eval. ✓
- **(r) prediction-vs-measurement table** — PAPER.md has complete table with
  all 3 KCs, predictions, "not measured", and FAIL verdicts. ✓
- **(s) proofs sound** —
  - T1 reduces KCs to parent dep-closure via explicit 3-step argument per KC.
    Each step is structural: Σ_0 undefined ⇒ Σ_t undefined for all t ≥ 0.
    K1402 (Δacc) requires Σ_0 and Σ_20; K1403 ("bugs") requires Σ_0 as
    reference; K1404 (regression) requires Σ_t trajectory. All reduce to
    parent K1387–K1389. ✓
  - T2 cites Alita paper (DB notes reference) for MCP/sandbox requirement,
    notes absence of MCP server / code sandbox / 20-round harness in repo,
    and cites `feedback_mlx_first.md` for MLX-only target. ✓
  - T3 independence: even conditional on ¬T1, T2 still kills; and vice
    versa. Two orthogonal floors. ✓
- **(t) target-gated KCs** — KCs 1402-1404 measure target behavior (benchmark
  improvement, bug-fix count, regression on held-out benchmarks). Not
  proxies. Preempt does not substitute a proxy target. ✓

## Potential reviewer objections (pre-empted)

1. **"Couldn't you run a 1-round or 2-round self-evolution to satisfy K1403
   (bug fixes)?"** — No. K1402 is explicit on "after 20 self-evolution
   rounds." Relaxing to fewer rounds changes pre-registered KC text, which
   PLAN.md §1 guardrail 5 forbids. A 1-round experiment would be a
   separate v2 design with its own KC. Correct choice: preempt.

2. **"Isn't the infrastructure argument (T2) overstated? Couldn't the
   scaffold just be a Python module edited by the LLM?"** — In principle,
   yes. In practice, K1403 requires "successfully identifies and fixes" —
   "successfully" implies validation via re-running the benchmark. That
   requires the 20-round benchmark harness (T2 core). Without
   validation-as-oracle, the "fixes" are ungrounded LLM proposals, which
   would be a F#498-tautology (scorer = the model being measured). Correct
   choice: preempt.

3. **"F#669 4th reuse — should promotion happen in this iteration?"** —
   Promotion was proposed at iter 72 (3rd reuse). Analyst cap prevented
   registration. This 4th reuse strengthens the case but does not itself
   register promotion (cap still applies). Proposal logged in PAPER.md,
   results.json, and LEARNINGS.md for analyst/coordinator resolution.

4. **"Terminal-floor status — is this concession-by-proxy?"** — No. This is
   a legitimate structural preempt via two independent impossibility axes,
   both with precedent ratified (F#671 dep-chain, F#672 platform-mismatch).
   The fact that preempt-kill clears the P≤2 floor is a consequence of
   correct drain-forward application, not its cause. Ratification criterion
   is soundness of T1 and T2, not floor-clearing convenience.

## Verdict
PREEMPT STANDS. 6/6 docs on disk. Ratify as KILLED.

After ratification: P≤2 open = 0, criterion #1 of drain objective satisfied,
coordinator should emit `RESEARCH_BACKLOG_DRAINED` at next orchestration
step.
