# REVIEW-adversarial — exp_followup_answer_conditioned_ppl

## Verdict: **KILL** (confirm researcher's pre-registered kill)

K1567 required the conjunction `Top1_ans ≥ 0.85 AND Top1_full < 0.85`.
Measured: Top1_ans = 0.978 (PASS), Top1_full = 0.984 (FAIL the <0.85 clause).
Researcher's KILLED verdict stands.

## Adversarial checklist

| Item | Result | Evidence |
|---|---|---|
| (a) `results.json.verdict` vs DB status | OK | verdict="KILLED"; DB no longer in active list |
| (b) `all_pass` vs claim | OK | all_pass=false ↔ KILLED |
| (c) PAPER.md verdict line | OK | PAPER.md §Verdict: **KILLED** |
| (d) `is_smoke` vs claim | OK | is_smoke=false; N_q=1000 |
| (e) KC git diff | OK | MATH.md single commit (3b049c9) before results (1511374); run_experiment.py single pre-reg commit (da9ad57) |
| (f) Tautology sniff | OK | K1567 is a conjunction of two independent measurements over 1000 disjoint queries; plausibly failable (and failed) |
| (g) K-ID quantity match | OK | code computes `top1_full`/`top1_ans` via per-query `argmin_j` (lines 182–183, 189–190); matches MATH.md §5.2 |
| (h) Composition bugs | N/A | No composition; single-expert routing |
| (i) LORA_SCALE ≥ 12 | N/A | numpy+autograd, no LoRA |
| (j) Per-sample routing | OK | argmin applied per-query (1000 queries) |
| (k) shutil.copy sibling | N/A | no on-disk adapters |
| (l) hardcoded pass | OK | `pass_ans`, `pass_full_failed`, `k1567_pass` computed from measurements (lines 208–210) |
| (m) target model proxy | N/A | numpy transformer; MATH.md §1 matches code |
| (m2) MLX skill evidence | N/A | no MLX path; CPU numpy only |
| (n) thinking-channel truncation | N/A | not an accuracy eval |
| (o) n < 15 | OK | N=1000 (200/domain × 5) |
| (p) synthetic padding | OK | all 5 domains have real trained experts; no zero-init placeholders |
| (q) cited baseline drift | N/A | no external baseline |
| (r) prediction-vs-measurement table | OK | PAPER.md §Prediction vs measurement present |
| (s) math errors | OK | Reconciliation in PAPER.md §3.1 is sound (relative-change correlation ≠ absolute cross-expert ranking) |

## Why KILL, not REVISE

The pre-registered KC has a disjunctive failure mode ("both metrics route correctly")
that was explicitly enumerated in MATH.md §3.3 table row 3 as a KILL case. The
researcher honored pre-registration instead of relaxing K1567. PAPER.md §3.1
explains the gap between predecessor r_full=−0.31 (relative Δ correlation) and
this experiment's absolute cross-expert argmin — no post-hoc rationalization.

## Non-blocking notes

1. PAPER.md "What would revive" proposes v2 on shared-alphabet prose domains —
   sensible handoff, not required here.
2. Single-seed is defensible: gap from 0.85 is >10σ on Bernoulli at N=1000.
3. Train/held-out query overlap for small-enumeration domains (arithmetic,
   parity) is acknowledged (PAPER.md Assumptions); does not change conclusion
   since kill is by the >0.85 clause failing, which overlap can only inflate.

## Assumptions

- Treated this as a VERIFICATION experiment per MATH.md §0 Type=Verification —
  a single-seed refutation is decisive when measured gap >> σ.
- DB was updated to `killed` by the researcher's `experiment complete` call
  (commit 1511374); not re-running the command.

## Routing

Emit `review.killed`. Analyst next writes LEARNINGS.md.
