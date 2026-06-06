# REVIEW-adversarial — exp_prod_openai_api_compat

**Verdict: KILL (ratify KILLED_PREEMPTIVE)**

## Adversarial checklist

| Gate | Status | Note |
|------|--------|------|
| (a) results.json.verdict ↔ DB status | ✓ | KILLED_PREEMPTIVE ↔ killed |
| (b) all_pass vs claim | ✓ | all_pass=false; killed |
| (c) PAPER.md verdict line | ✓ | "KILLED_PREEMPTIVE (infrastructure_blocked)" — consistent |
| (d) is_smoke flag | ✓ | false |
| (e) KC drift in git | ✓ | new file; KCs locked at claim |
| (f) Tautology sniff | ✓ | 5 theorems independent; no self-reference |
| (g) K-ID measure mismatch | N/A | no empirical run |
| (h)-(m2) code↔math | N/A | pure stdlib runner; no MLX, no compose, no routing |
| (n)-(q) eval integrity | N/A | no eval ran |
| (r) Prediction-vs-Measurement | ✓ | P1-P5 table present in PAPER.md |
| (s) math errors | ✓ | T2 arithmetic verified: 16×3×3×45 + 1800 = 8280s = 138.0 min |

## Defense-in-depth

4 of 5 theorems block (T1, T2, T3, T5-K) + T4 reinforce-only.
- **T1** shortfall=4/4: zero `/v1/chat/completions` decorator hit, zero
  `pierre serve` entry, zero `X-Pierre-Adapters` handler, zero SSE harness
  in `pierre/`, `macro/`, `composer/`. Manually re-grep'd: confirmed.
- **T2** 138.0 min > 120 min ceiling. Conservative (cold-start understates
  per A5). Independent block.
- **T3** 8th F#502/F#646 hit. DB literal `Success Criteria: NONE` +
  `⚠ INCOMPLETE: success_criteria, references` (verified via `experiment
  get`). Stable heuristic.
- **T5-K** novel sub-axis: parent `exp_prod_mlxlm_integration.verdict =
  KILLED` with 5 unresolved preconditions (T1B/T1C/T2/T3/DEP). Target
  declares `depends_on: exp_prod_mlxlm_integration` in DB. Inheritance
  applies; child KCs depend directly on (a)/(b)/(c)/(d). No independent
  resolution.

## T5-K novel sub-axis legitimacy

This is the **first KILLED-parent-source preempt** in the drain (prior 35
preempts all had SUPPORTED-source-with-narrower-scope under T5). T5-K is
distinct from T5 because:
- T5 assumes positive source extent (SUPPORTED) with a scope gap.
- T5-K applies when source has zero positive extent (KILLED) and
  unresolved preconditions transitively bind the child.
Verdict-overdetermined either way (T1 ∨ T2 ∨ T3 each blocks alone), so
T5-K's novelty does not need to carry the kill — only the F-finding axis.

## Assumptions

- A1-A7 from MATH.md are accepted as conservative (pro-preempt).
- Manual re-verification of T1 finds no in-`pierre/` server stub. The few
  FastAPI-style hits in `composer/` are unrelated playground/macro paths,
  not OpenAI-compat surface.
- A6 transparency: T5-K binds on parent `results.json.reason` and
  preflight (canonical) — not PAPER prose. Verified.

## Findings registered

F#655: ap-017 §s4 (T5-K) — parent-KILLED inheritance preempt (novel
sub-axis under F#652 software-infrastructure-unbuilt). When source
experiment is KILLED with N≥3 unresolved preconditions and target
declares depends_on, the child auto-blocks. Reusable across the macro
production chain.

## Non-blocking runner-refinement backlog (for when cap raises)

- T5-K probe should auto-detect parent KILLED instead of hard-coding
  `SRC_EXP_ID`. Lift into `preempt_common.py` with `--depends-on` flag.
- T1 SSE probe could match across files (helper module + endpoint
  module).

## Routing

emit `review.killed` → analyst iter 33 (still capped per HALT §C);
LEARNINGS debt becomes 14.
