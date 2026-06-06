# REVIEW-adversarial — exp_g4_compose_bakeoff_top3

**Verdict: PROCEED (confirm KILL)**

29th cohort preemptive-kill ratified. 22nd composition-bug branch under
ap-017. 11th SUPPORTED-source preempt (candidate (q): F#173 theory-
aggregation-non-transfer — distinct axis from prior empirical-source
variants (a)-(p)).

## Adversarial checklist

| Item | Result | Evidence |
|------|--------|----------|
| (a) results.json ↔ DB | PASS | `verdict=KILLED_PREEMPTIVE` ↔ `Status: killed` |
| (b) all_pass ↔ claim | PASS | K1628=fail; `all_block=true`; no SUPPORTED claim |
| (c) PAPER verdict | PASS | "KILLED (preemptive, 5-theorem stack)" |
| (d) is_smoke | N/A | preemptive (no training run) |
| (e) KC diff | PASS | K1628 text unchanged: "one approach dominates others by >=3pp MMLU-Pro" |
| (f) tautology | PASS | T1 filesystem check, T3 DB-empty check, T5 source-scope audit — all external state |
| (g) K-ID ↔ claim | PASS | K1628 pass/fail of 5-theorem stack |
| (h) buggy composition | N/A | no training code executed |
| (i)-(m) training hazards | N/A | preemptive kill, no training |
| (m2) skills | N/A | pure stdlib runner, no MLX code |
| (n) thinking-suppression | N/A | no eval |
| (o) n<15 | N/A | no measurement |
| (p) synthetic padding | N/A | no composition |
| (q) baseline drift | N/A | no baseline |
| (r) prediction table | PASS | PAPER.md §"Prediction vs measurement" 5-row table |
| (s) math errors | PASS | 3/3 T5 sub-breaches (B, C, D) independently verified; (A) cosmetic runner false-neg openly disclosed |

## Defense-in-depth ratification

- **T1**: `ls micro/models/exp_p1_t2_single_domain_training/adapters/` → `{code, math, medical}` (3 of N=5; shortfall=2). `ls micro/models/ | grep runtime_lora` → 0 hits. BLOCKS alone.
- **T3**: `experiment get` output carries LITERAL "Success Criteria: NONE — add with: experiment success-add" + "⚠ INCOMPLETE". BLOCKS alone.
- **T5**: 3 of 4 sub-breaches (B, C, D) runner-verified; (A) cosmetic false-neg via `finding-get 164` summary drop (same pattern as exp_g4_task_arith_lambda_sweep previously ratified). BLOCKS alone.
- **T4**: 1/5 pins reinforcement.
- **T2**: arithmetic-only reinforcement (≥ 56.84 min > 30 min).

T1 ∨ T3 ∨ T5 each alone blocks; no single patch unblocks.

## KC-vs-source contradiction (most damning breach)

B-ii: K1628 runs DARE at **p=0.5**. F#173 top-3 recommendation (the
finding K1628 claims to test) specifies **p=0.9**. The p=0.5 value
originates from F#269 NOTES LITERAL: "Optimal p for ternary adapters
is 0.5 (not 0.9 as in DARE paper) because LoRA scale s=20 creates
effective perturbation s/(1-p)". F#269 scope is BitNet-ternary s=20,
not Gemma 4 FP16. The experiment:
  (i) contradicts its own cited source on parameter value;
  (ii) silently imports BitNet-ternary scope via p-substitution;
  (iii) F#269 LITERAL impossibility "MMLU math degradation persists
       across all drop rates" directly refutes ">=3pp MMLU-Pro"
       dominance claim for the DARE arm.

## Registration request

Register F#173 compound-non-transfer as reusable one-line preempt (q)
under ap-017, alongside (a)-(p). NEW axis: **theory-aggregation-non-
transfer** — source aggregates top-3 recommendations each riding on
distinct BitNet-scope parents (F#164, F#269, runtime-routing), target
invokes them as static bakeoff, compounding each parent's scope gap.
Distinct from prior (a)-(p) empirical-source variants.

## Assumptions (judgment calls, no operator input)

1. T5 sub-breach (A) runner false-neg treated as cosmetic because MATH-
   level F#164 base-scope (BitNet-2B ternary MLP r=8 PPL) is verifiable
   from source MATH.md; precedent exp_g4_task_arith_lambda_sweep
   accepted this pattern.
2. PAPER.md prediction-vs-measurement table satisfies (r) without
   enumerating T2 (arithmetic-only) — MATH.md captures T2 in full.

## Non-blocking followups (inherited cohort debt)

- T4 ε regex cohort-wide patch (methodology-ε keyword vs numeric
  threshold) — no ε-language in K1628, so non-blocking here.
- Analyst LEARNINGS debt (operator-owed until HALT §C cap raise): now
  7 entries {vproj_think, polar, null_space, tfidf_ridge_n25,
  tfidf_routing_no_alias, flywheel_real_users, compose_bakeoff_top3}.

## Route

`review.killed` → analyst (iter 25). Will drop silently under HALT §C
50/50 cap; coordinator drain-forward pattern re-emits research.start.
