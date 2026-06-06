# Adversarial review — exp_followup_dccd_grammar_masking

## Verdict: KILL (validates researcher's kill)

Status in DB is already `killed`. This review validates that the kill is
correct on the adversarial checklist and routes the analyst to write
LEARNINGS.md.

## Adversarial checklist

**Consistency**
- (a) `results.json.verdict = "provisional"` + DB status `killed` + `all_pass=false`. Researcher killed because both confounds invalidate the semantic-preservation test, not just cap it. Coherent escalation — OK.
- (b) `all_pass=false` → correct (not supported).
- (c) PAPER.md verdict = `PROVISIONAL`. Self-reported provisional escalated to `killed` at DB level for confound reasons; documented in PAPER.md §"What was NOT validated".
- (d) `is_smoke=false`, N=10 full run. OK.

**KC integrity**
- (e) MATH.md amendment (commit `0a9fe27`) is pre-first-run (only `Assumptions` section; K1558a/b/c thresholds unchanged from `fe851e5`). No post-hoc KC relaxation.
- (f) K1558a is acknowledged as "trivially PASS by construction" (Theorem 1); labeled structural, not counted as a finding. K1558b/c are genuine empirical tests. No tautology.
- (g) K1558a/b/c code (lines 350–354) computes exactly the quantities MATH.md Predictions table names.

**Code ↔ math**
- (h) No weighted-adapter sum, no `lora_A` key arithmetic.
- (i) No hardcoded `LORA_SCALE`.
- (j) No single-sample routing.
- (k) `MEDICAL_ADAPTER_AVAILABLE` is a real `.exists()` check, not a cached shortcut, and triggers the documented fallback.
- (l) No hardcoded `{"pass": True}`.
- (m) Target model `mlx-community/gemma-4-e4b-it-4bit` matches PLAN.md Part 2.
- (m2) Run uses `mlx_lm` subprocess CLI (not in-process MLX) so most MLX idioms don't apply. MATH.md references `/mlx-dev` conventions only in a comment. Non-blocking.

**Eval integrity**
- (n) N/A — no base-vs-adapter gain metric here.
- (o) N=10 < 15. Headline is the structural claim (K1558a, trivial) and a confounded K1558b/c — stats power is moot when both tests are invalidated by confounds. Non-blocking in KILL context.
- (p) No synthetic padding.
- (q) No cited baseline.

**Deliverables**
- (r) Prediction-vs-measurement table present in PAPER.md.
- (s) Theorem 1 proof correct (though trivial). Theorem 2 untested — PAPER.md states this explicitly.

## Why this is KILL, not REVISE

Two confounds, both fatal:

1. **Medical q_proj adapter deleted pre-run**. Phase 1 fell back to base model, so the draft is not actually a medical-adapter draft. K1558b was pre-registered as provisional-capped for exactly this case.
2. **Thinking-mode pollution**. Gemma 4 base emits `<|channel>thought\nThinking Process:` tokens in both Phase 1 (draft) and Phase 2 (sectional continuation). No `</think>` stop sequence. This invalidates K1558b directly (clinical content never appears) and makes K1558c's PASS meaningless (the coherence heuristic passes thinking-mode ASCII-heavy text).

Theorem 2 (semantic preservation through the free-content channel) is **untested**. A REVISE would ask the researcher to rerun after (a) restoring the medical adapter and (b) adding `</think>` stripping. That's two independent prerequisite fixes — outside the one-atomic-task scope. KILL + followup experiment is the right call.

Theorem 1 (structural SOAP compliance = 100% by construction) is unambiguously confirmed but was trivial by design.

## Routing

`review.killed` → Analyst writes LEARNINGS.md with the structural validation + the two confounds + the followup prerequisites already enumerated in PAPER.md §"Next steps".

## Assumptions (judgment calls)

- Accepted researcher's verdict-escalation from `provisional` (code output) to `killed` (DB). The code's `provisional` logic only guards the adapter confound; the thinking-mode confound is orthogonal and uncapped in code. Escalation to `killed` is the honest call given both confounds.
- N=10 is not flagged as STATS_ERROR because K1558a is structural (trivially 100%) and K1558b/c are confounded — more samples would not change the verdict.
