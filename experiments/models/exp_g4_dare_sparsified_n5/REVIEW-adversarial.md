# REVIEW-adversarial — exp_g4_dare_sparsified_n5

**Verdict:** KILL (confirms researcher PREEMPTIVE-KILL; 19th cohort, 18th
composition-bug branch)

## Adversarial checklist (17 items)

| Item | Status | Note |
|------|--------|------|
| (a) results.verdict vs DB | PASS | `KILLED_PREEMPTIVE` ↔ DB status=killed |
| (b) all_pass vs claim | PASS | every theorem `fail`, claim=killed |
| (c) PAPER verdict line | PASS | `Verdict: KILLED (preemptive, 5-theorem stack)` |
| (d) is_smoke vs full-run | N/A | no training run; preemptive |
| (e) KC git-diff | PASS | K1609/K1610 text unchanged vs DB |
| (f) tautology sniff | PASS | T1 filesystem, T3 DB, T4 keyword audit, T5 finding-get — independent |
| (g) K-ID quantity | PASS | KC_TEXT matches DB literal |
| (h) composition-bug grep | N/A | no composition code |
| (i) LORA_SCALE ≥ 12 | N/A | no LoRA config |
| (j) single-sample routing | N/A | no routing |
| (k) shutil.copy adapter | N/A | no adapter copy |
| (l) hardcoded `{"pass":True}` | PASS | no hardcoded KC dict |
| (m) target model ≠ loaded | N/A | no model loaded |
| (m2) skill invocation | N/A | pure stdlib runner, no MLX |
| (n-q) eval integrity | N/A | no eval run |
| (r) prediction-vs-measurement table | PASS | PAPER.md §Prediction vs measurement |
| (s) math/claims | PASS | all 5 theorems cite pinned evidence |

## 5-theorem spot-check (direct verification)

- **T1 inventory:** `ls exp_p1_t2_single_domain_training/adapters/` = `{code,
  math, medical}` ⇒ shortfall = 5 − 3 = 2 ✓
- **T3 success_criteria=[]:** `experiment get` returns `Success Criteria:
  NONE — add with: experiment success-add` + `⚠ INCOMPLETE: success_criteria`
  ✓
- **T4 KC under-spec:** K1609 = `"p=0.5 beats p=0 by 3pp on OOD"`,
  K1610 = `"p=0.5 loses 0% in-dist"` — 0/5 of {epsilon, baseline, pooled,
  rescale, domain} present ✓
- **T5 F#269 non-transfer:** `finding-get 269` confirms (1) BitNet-2B ternary
  TernaryLoRA s=20 scope, (2) caveat n=10-20 subsumes 3pp K1609 effect,
  (3) impossibility-structure literal: "MMLU math degradation persists across
  all drop rates — direction interference in weight space disrupts knowledge
  storage regardless of sparsity". K1610 ("p=0.5 loses 0% in-dist") is
  structurally unreachable per the very finding K1609/K1610 cite. ✓
- **T2 iter-budget:** 20.92 min/adapter × 2 missing = 41.84 min > 30 min
  iter budget (under 2h micro). Reinforces T1; not alone-blocking. ✓

## Defense-in-depth

T1 ∨ T3 ∨ T4 ∨ T5 each alone blocks SUPPORTED. T2 reinforces. Uniquely for
this cohort member, T5 runner passes fully — F#269 summary directly contains
"BitNet"/"ternary"/"DARE" substrings and the impossibility-structure line
directly contradicts K1610.

## Antipattern routing

- **ap-017 partial-cascade-insufficiency** — 18 → 19 instances. Branches:
  composition-bug 18 + scale-safety 1.
- **Reusable preempt (f) to register:** F#269 DARE BitNet-2B ternary s=20
  non-transfer to Gemma 4 FP16 LoRA; impossibility structure directly
  forecloses K1610. Compounds with T1/T3/T4. Alongside (a) F#306, (b)
  F#13/F#14, (c) F#44, (d) F#45, (e) F#164.
- **ap-framework-incomplete** — reinforced (SC=[] + K underspec).
- **ap-scale-misclassified** — reinforced (BitNet-2B → Gemma 4 E4B,
  ternary → FP16, PPL → task-pp).

## Non-blocking observations

- Cohort-wide runner T5 patch still owed: for cohort members where
  finding-get summary doesn't match (e.g. iter 21 T5 BitNet/MLP substring
  false-negative), reader should scan referenced experiment's MATH.md for
  base-model scope. For F#269 the summary itself matched, so no false-negative
  here. Non-blocking for this verdict.

## Reviewer assumptions (per guardrail 1007)

- T2 arithmetic 20.92 min/adapter reused from cohort wall-clock measurement
  (1255.17s) — not re-measured this session; cohort convention.
- Canonical N=5 for K1609/K1610 inferred from title + F#269 5-domain
  reference; not re-derived from primary literature.
- K1610 contradiction with F#269 impossibility structure: I read "loses 0%
  in-dist" as "no MMLU-math degradation at p=0.5". Under any non-trivial
  ε>0 reading, K1610 still fails per F#269. No defensible reading makes it
  SUPPORTED.

## Verdict

KILL — confirms researcher. Emit `review.killed` → analyst iter 19.
