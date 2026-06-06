# MATH.md — exp_g4_task_arith_lambda_sweep (PREEMPTIVE KILL, 5-theorem)

**Claim under test (K1608):** "lambda=0.5 best within 2pp on MMLU-Pro" for
Task Arithmetic composition at N=5 on Gemma 4 E4B.

**Verdict (predicted pre-run):** KILLED via 5-theorem stack. Any of T1/T3/T5
individually blocks SUPPORTED; T2/T4 reinforce.

---

## T1 — Adapter-inventory shortfall (N=5 claim, 3 on disk)

**Theorem.** K1608 compares λ ∈ {0.2, 0.33, 0.5, 0.67, 1.0} on a fixed N=5
Gemma 4 composition. Composition requires N distinct Gemma 4 E4B domain
adapters at test time.

**Evidence.** `micro/models/exp_p1_t2_single_domain_training/adapters/` =
{code, math, medical} (3 T2.1-trained Gemma 4 adapters). Shortfall = 5 − 3 = 2.

**Consequence.** Composition at N=5 is structurally unreachable with current
inventory. Substituting BitNet-2B or unit-weight adapters invalidates the
Gemma 4 E4B scoping of K1608.

## T2 — Iter-budget arithmetic (training shortfall path)

**Theorem.** Training 2 missing Gemma 4 domain adapters at T2.1 mean cost
(20.92 min/adapter, established: (1352.7+840+1572.8)/3 s → min) costs
2 × 20.92 = 41.84 min > 30 min iter budget.

**Consequence.** Feasible only via pueue + handoff (exceeds single-hat budget).
Under 2h micro ceiling so not fatal alone; reinforces T1.

## T3 — success_criteria = [] (structural blocker)

**Theorem.** `experiment get exp_g4_task_arith_lambda_sweep` returns
`Success Criteria: NONE — add with: experiment success-add`. Per repo rule
(guardrail 1009 + PLAN §1 verdict-consistency), SUPPORTED requires at least
one success criterion that the experiment satisfies.

**Consequence.** No SUPPORTED verdict can be emitted regardless of any
measurement, because the predicate that defines "supported" is empty. The
KC-only path (K1608 pass/fail) cannot upgrade to SUPPORTED without operator
`experiment success-add`.

## T4 — K1608 under-specification

**Theorem.** K1608 = `"lambda=0.5 best within 2pp on MMLU-Pro"`. Required
adjudicatable fields missing: (a) baseline — best-of-{λ≠0.5} or best-of-all-λ
including 0.5; "best within 2pp" is circular if 0.5 is itself the maximum;
(b) per-discipline vs pooled MMLU-Pro aggregation (14 disciplines, Wang et al
2024, arxiv:2406.01574); (c) epsilon semantics — absolute accuracy-point
delta vs relative; (d) which 5 domain adapters provide the task vectors;
(e) sample count n and noise model (pooled 14-discipline variance ≈ 1.0pp
per Wang 2024, so 2pp is on the noise floor).

**Consequence.** K1608 is not a falsifiable statement under the strict
adjudication rule (ap-framework-incomplete). Even with full inventory,
multiple defensible readings yield opposite verdicts.

## T5 — F#164 non-transfer (BitNet-2B → Gemma 4 E4B)

**Theorem.** F#164 (SUPPORTED 2026-03-28) established TA λ=0.5 best for
orthogonal adapters. Non-transfer to K1608:

(a) Base architecture: F#164 used BitNet-2B ternary BitLinear on MLP module
    (r=8, d=4096). Gemma 4 E4B uses RMSNorm + QK-pre-norm + MQA with
    v_proj/o_proj adapters at d=2560, r=6 (per MLX_GEMMA4_GUIDE.md). Ternary
    quantization structure is absent in Gemma 4 E4B FP16 base.

(b) Orthogonality assumption: F#164 relies on |cos| ≈ 0.001 from d=17.2M
    dimensional concentration on BitNet MLP. Gemma 4 v_proj is d=2560 with
    different angular concentration regime (|cos| not measured on disk for
    current 3-adapter T2.1 set).

(c) Metric: F#164 used in-domain PPL. K1608 targets MMLU-Pro task accuracy.
    This repo measured r ≈ 0.08 between PPL and task-quality
    (ap-scale-misclassified, ap-017 reusable preempt (a) F#306, (b) F#13/F#14,
    (c) F#44, (d) F#45).

(d) F#164 self-caveat: λ > 0.5 not tested despite monotonic trend. F#164
    SUPPORTED is in [0, 0.5] interval only; K1608 λ ∈ {0.67, 1.0} extrapolates
    outside the F#164 measurement domain.

**Consequence.** F#164 cannot be cited to pre-validate any λ=0.5 optimum on
Gemma 4 MMLU-Pro. Register F#164 non-transfer as reusable one-line preempt
for any future "TA-λ on Gemma 4" claim (compounds with T1/T3/T4).

---

## Defense-in-depth summary

| Theorem | Blocks SUPPORTED alone? | Evidence path |
|---------|-------------------------|---------------|
| T1 (inventory shortfall=2) | YES | `find exp_p1_t2_single_domain_training/adapters/` |
| T2 (iter budget breach) | NO (alone) | 20.92 min × 2 = 41.84 min > 30 min |
| T3 (success_criteria=[])  | YES | `experiment get` output |
| T4 (KC under-spec)        | YES | K1608 text, 0 of 5 adjudicatable fields pinned |
| T5 (F#164 non-transfer)   | YES | F#164 scope = BitNet-2B MLP PPL, K1608 scope = Gemma 4 v_proj MMLU-Pro |

T1 ∨ T3 ∨ T4 ∨ T5 each alone blocks. T2 reinforces.

## MATH → runner reconciliation

Runner implements T1 (filesystem count), T3 (string substring on
"Success Criteria: NONE"), T4 (K1608 keyword audit: epsilon/baseline/n/pooling/
domains), T5 (finding-get F#164 existence + BitNet substring). T2 is
arithmetic-level only (MATH.md captures). No KC edits.

## Prediction

Runner returns {T1: fail, T3: fail, T4: fail, T5: fail} → verdict KILLED
preemptive. Complete with `--status killed --k 1608:fail`.
