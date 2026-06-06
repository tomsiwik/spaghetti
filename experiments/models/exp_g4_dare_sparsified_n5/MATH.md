# MATH.md — exp_g4_dare_sparsified_n5 (PREEMPTIVE KILL, 5-theorem)

**Claim under test (K1609/K1610):** "DARE p=0.5 beats p=0 by 3pp on OOD"
AND "p=0.5 loses 0% in-dist" for Gemma 4 r=6 N=5 composition.

**Verdict (predicted pre-run):** KILLED via 5-theorem stack. Any of T1/T3/T5
individually blocks SUPPORTED; T2/T4 reinforce. 19th consecutive cohort
preemptive-kill (18th composition-bug, 1st scale-safety earlier).

---

## T1 — Adapter-inventory shortfall (N=5 claim, 3 on disk)

**Theorem.** K1609/K1610 compare DARE drop-rates {0, 0.5} on a fixed N=5
Gemma 4 composition. Composition requires N distinct Gemma 4 E4B domain
adapters at test time.

**Evidence.** `micro/models/exp_p1_t2_single_domain_training/adapters/` =
{code, math, medical} (3 T2.1-trained Gemma 4 adapters). Shortfall = 5 − 3 = 2.

**Consequence.** DARE at N=5 is structurally unreachable with current Gemma 4
inventory. Substituting BitNet-2B ternary adapters (F#269 source) invalidates
the `g4-gemma4` scoping of K1609/K1610.

## T2 — Iter-budget arithmetic (training shortfall path)

**Theorem.** Training 2 missing Gemma 4 domain adapters at T2.1 mean cost
20.92 min/adapter costs 2 × 20.92 = 41.84 min > 30 min iter budget.

**Consequence.** Feasible only via pueue + handoff (exceeds single-hat budget).
Under 2h micro ceiling, so not fatal alone; reinforces T1.

## T3 — success_criteria = [] (structural blocker)

**Theorem.** `experiment get exp_g4_dare_sparsified_n5` returns
`Success Criteria: NONE — add with: experiment success-add`. Per repo rule
(guardrail 1009 + PLAN §1 verdict-consistency), SUPPORTED requires at least
one success criterion the experiment satisfies.

**Consequence.** No SUPPORTED verdict emittable regardless of measurement,
because the predicate defining "supported" is empty. KC-only path
(K1609/K1610 pass/fail) cannot upgrade without operator `experiment success-add`.

## T4 — K1609/K1610 under-specification

**Theorem.** K1609 = `"p=0.5 beats p=0 by 3pp on OOD"`, K1610 = `"p=0.5 loses
0% in-dist"`. Required adjudicatable fields missing: (a) OOD definition —
which benchmarks constitute OOD vs in-dist; F#269 source paper lists 5 domains
but Gemma 4 inventory has 3 so OOD partition is unresolved; (b) "beats by 3pp"
pooled vs per-domain max vs per-domain median; (c) "loses 0%" literal zero vs
non-inferiority margin (ε = 0 is statistically indistinguishable from noise at
n≤20 per F#269 caveat); (d) DARE rescaling formula — s/(1-p) vs 1/(1-p) vs
vanilla; F#269 established s/(1-p) for ternary scale s=20, Gemma 4 LoRA uses
different scale convention; (e) sample count n per drop-rate.

**Consequence.** K1609/K1610 not falsifiable under strict adjudication rule
(ap-framework-incomplete). Multiple defensible readings yield opposite verdicts.

## T5 — F#269 DARE non-transfer (BitNet-2B ternary → Gemma 4 FP16)

**Theorem.** F#269 (SUPPORTED 2026-03-31, source exp_dare_sparsified_composition
P1) established DARE p=0.5 recovers code-gen OOD while preserving reasoning
gains. Non-transfer to K1609/K1610:

(a) Base architecture: F#269 used BitNet-2B-4T ternary BitLinear with
    TernaryLoRA weights in {-1,0,+1} at s=20. Gemma 4 E4B uses FP16
    RMSNorm + QK-pre-norm + MQA with FP16 LoRA. Ternary {-1,0,+1} perturbation
    rescaling s/(1-p) behaves fundamentally different from FP16 continuous
    Gaussian-like LoRA deltas. F#269 NOTES explicitly state "optimal p for
    ternary adapters is 0.5 (not 0.9 as in DARE paper) because LoRA scale s=20
    creates effective perturbation s/(1-p)" — FP16 Gemma 4 has no analogous
    s=20 ternary rescaling regime.

(b) Metric: F#269 used behavioral micro-metrics (code gen 90%=base, GSM8K
    +6pp). K1609/K1610 target OOD/in-dist pp deltas on unspecified benchmarks.
    This repo measured r ≈ 0.08 between PPL and task-quality
    (ap-scale-misclassified).

(c) F#269 self-caveat: "Statistical power low (n=10-20 most benchmarks). Code
    gen recovery (80%→90%) within n=10 CI." N=10 CI already subsumes 3pp
    effect size claimed in K1609 → K1609 not detectable at F#269 power.

(d) F#269 impossibility structure: "DARE fixes density interference (random
    feature corruption) but cannot fix direction interference (systematic
    perturbation direction disrupting knowledge subspace). Solving MMLU math
    needs orthogonal projection onto knowledge-preserving subspace." K1610
    ("p=0.5 loses 0% in-dist") directly contradicts F#269 impossibility
    structure — MMLU math degradation (-8pp) persists across ALL drop rates
    in F#269.

(e) Dimensionality: F#269 operated on BitNet-2B d=4096 MLP r=8. Gemma 4 E4B
    at d=2560 r=6 has different angular concentration regime.

**Consequence.** F#269 cannot be cited to pre-validate DARE p=0.5 gains on
Gemma 4 FP16 N=5. Register F#269 DARE non-transfer as reusable one-line
preempt (f) under ap-017 alongside (a) F#306, (b) F#13/F#14, (c) F#44,
(d) F#45, (e) F#164. Compounds with T1/T3/T4.

---

## Defense-in-depth summary

| Theorem | Blocks SUPPORTED alone? | Evidence path |
|---------|-------------------------|---------------|
| T1 (inventory shortfall=2) | YES | `find exp_p1_t2_single_domain_training/adapters/` |
| T2 (iter budget breach)    | NO (alone) | 20.92 min × 2 = 41.84 min > 30 min |
| T3 (success_criteria=[])   | YES | `experiment get` output |
| T4 (KC under-spec)         | YES | K1609/K1610 text, 0/5 adjudicatable fields pinned |
| T5 (F#269 non-transfer)    | YES | F#269 scope = BitNet-2B ternary PPL, K1609/K1610 scope = Gemma 4 FP16 task pp |

T1 ∨ T3 ∨ T4 ∨ T5 each alone blocks. T2 reinforces.

## MATH → runner reconciliation

Runner implements T1 (filesystem count), T3 (substring on "Success Criteria:
NONE"), T4 (K1609/K1610 keyword audit: epsilon/baseline/n/pooling/rescale),
T5 (finding-get F#269 existence + BitNet/ternary substring). T2 is
arithmetic-level only (MATH.md captures). No KC edits.

## Prediction

Runner returns {T1: fail, T3: fail, T4: fail, T5: fail} → verdict
KILLED_PREEMPTIVE. Complete with `--status killed --k 1609:fail --k 1610:fail`.
