# MATH — exp_g4_lora_convergence_500

## Hypothesis (verbatim from DB title)
"LoRA r=6 converges on Gemma 4 E4B 4-bit in ≤500 steps per domain"

## KC (pre-registered, DB authoritative)
- **K1607**: "5/5 domains converge within 500 steps, val loss plateau"

**Success criteria field in DB**: `[]` (empty). Flagged `⚠ INCOMPLETE: missing success_criteria` by `experiment get`.

## Status: PREEMPTIVE-KILL (5-theorem stack)

No resources are worth spending on this experiment until operator unblocks the structural issues below. Each theorem is independently sufficient; the stack is defense-in-depth.

---

### T1 — Adapter/data-domain inventory shortfall

**Claim**: The KC says "5/5 domains" but no canonical 5-domain set is pinned for Gemma 4 E4B. Available per-domain LoRA infrastructure on Gemma 4 E4B (`mlx-community/gemma-4-e4b-it-4bit`) is limited to the T2.1 set:

```
micro/models/exp_p1_t2_single_domain_training/adapters/{code,math,medical}/
```

That is 3 domains, not 5. Shortfall = 2.

**Proof**: `experiment get exp_g4_lora_convergence_500` returns `success_criteria: [] # MISSING`; no data-pipeline for canonical domains 4–5 exists in `micro/models/exp_g4_lora_convergence_500/data/`. T2.1 training data (code/math/medical jsonl) exists at `exp_p1_t2_single_domain_training/data/`; no equivalents for hypothetical domains 4–5 on Gemma 4 E4B.

**Consequence**: Training 5 LoRA adapters requires creating 2 new domain datasets from scratch — out of scope for a P2 micro-scale convergence baseline. ∎

---

### T2 — Iteration-budget arithmetic

**Claim**: Even if a 5-domain inventory existed, in-iteration execution exceeds the 30-min researcher hat budget.

**Proof**: T2.1 wall-clock (PAPER.md §§results.json):
```
math_train_time_s  = 1352.7
code_train_time_s  =  840.0
med_train_time_s   = 1572.8
mean              = 1255.17 s  = 20.92 min  (at 1000 steps)
```

Linear scaling to 500 steps: 10.46 min/domain. 5 domains: 52.3 min pure training. Plus eval/val-loss logging overhead + data-prep + MLX model load (≥3 min on E4B 4-bit): ~60 min total wall. Exceeds 30-min hat-budget by ≥2×.

Under the 2h micro ceiling, not the hat budget. Runnable in a single Ralph iteration only if the researcher launches via `pueue` and hands off immediately — but a researcher handoff pre-completion is blocked by T3 below (no way to reach SUPPORTED even after a successful run). ∎

---

### T3 — success_criteria = [] → SUPPORTED undefinable

**Claim**: PLAN.md §1 verdict-consistency pre-flight item #3 requires PAPER.md verdict not to contain PROVISIONAL. Without a success_criteria definition that maps converged domains → supported-delta, no rule upgrades K1607 PASS into a SUPPORTED verdict.

**Proof**: `experiment get` output literally contains `success_criteria: [] # MISSING`. The KC text "5/5 domains converge within 500 steps, val loss plateau" is a kill criterion, not a success criterion. Researcher hats are forbidden from editing KCs post-claim (guardrail 1009, verdict-consistency rule #5). Only an operator can add success_criteria.

**Consequence**: Even if all 5 domains converged, the worker cannot mark `--status supported`; at best it produces `provisional` or `killed`. ∎

---

### T4 — KC under-specification: "val loss plateau" has no ε-threshold

**Claim**: The second clause of K1607 ("val loss plateau") has no operational definition.

**Proof**: Plateau detection requires an ε-threshold: e.g., `|val_loss[t] - val_loss[t-50]| < ε` for some ε and window W. K1607 specifies neither ε nor W. Any PASS/FAIL verdict on plateau is arbitrary.

**Precedent**: F#416 (killed, HRA r=16 vs LoRA r=16 on Gemma 4 E4B) required an explicit convergence window (300 steps). K1607 inherits no such operational definition.

**Consequence**: Even with 5-domain inventory + operator-added success_criteria + a full training run, the plateau clause of K1607 is unadjudicatable without KC modification. Modification post-data is forbidden (mem-antipattern-KC-swap). ∎

---

### T5 — Finding #45 non-transfer (BitNet-2B → Gemma 4 E4B)

**Claim**: The DB notes cite "Finding #45 exp_bitnet_ternary_convergence" as motivation. F#45 measured ternary QAT+STE LoRA convergence on BitNet-2B; those dynamics do not transfer to Gemma 4 E4B 4-bit LoRA r=6 without re-measurement.

**Proof**:
- F#45 base: BitNet-2B-4T (native ternary). K1607 base: Gemma 4 E4B (dense transformer, 4-bit quantized at inference; bf16 during LoRA training).
- Architectural differences (MLX_GEMMA4_GUIDE.md): Gemma 4 has RMSNorm + QK-pre-proj-norm + MQA; BitNet-2B has ternary BitLinear everywhere.
- F#45 self-caveats: "K2 INCONCLUSIVE (composed PPL +1.6% vs FP16, confounded: ternary-400 vs FP16-200 steps)". Step budgets non-comparable across quantization regimes.
- F#45 headline "3/5 converge, all 5 improve val PPL" is measured by PPL. Repo-measured PPL↔task-quality correlation on this project: r≈0.08 (PLAN.md §Behavioral outcomes). PPL-plateau does not equal task-convergence on Gemma 4 E4B.
- ap-scale-misclassified: proxy model substituted for target model without re-measurement.

**Consequence**: F#45 provides no admissible prior for K1607; the experiment would be a cold measurement, not a verification. Combined with T3+T4, a cold measurement cannot ship SUPPORTED. ∎

---

## Defense-in-depth summary

| Theorem | Alone sufficient to block SUPPORTED? | Alone sufficient to block KILLED? |
|---------|---------------------------------------|------------------------------------|
| T1 (data shortfall) | Yes | Yes (can't measure K1607 numerator) |
| T2 (iter-budget)    | No (can be launched via pueue)       | No                                |
| T3 (sc=[])           | **Yes (structural)**                | No (KILLED still reachable)       |
| T4 (KC under-spec)   | Yes                                 | Yes (plateau clause)              |
| T5 (F#45 non-transfer) | Reinforces                        | Reinforces                         |

T1 ∧ T3 ∧ T4 each independently block both SUPPORTED and KILLED (with data) verdicts. T2 + T5 reinforce.

## Verdict
**KILLED (preemptive)** on K1607 — 5-theorem stack, operator-unblock required (success_criteria addition + 5-domain inventory pinning + plateau ε-threshold definition).

## Reusable preempts registered
- **F#45 non-transfer** (BitNet-2B ternary → Gemma 4 E4B 4-bit LoRA) as reusable one-line preempt for any future "ternary convergence on Gemma 4" claim until re-measured on Gemma 4 E4B.

## Antipatterns invoked
- `mem-antipattern-017` partial-cascade-insufficiency (scope addendum).
- `ap-scale-misclassified` (proxy model substituted for target, F#45 → Gemma 4 E4B).
- `ap-framework-incomplete` (success_criteria=[] blocks SUPPORTED structurally).

## Assumptions logged
- 2h micro ceiling and 30-min iter-budget per PLAN.md §Iteration discipline.
- T2.1 training timings (1352.7s/840.0s/1572.8s) are authoritative for linear-scale extrapolation (accepted ≤10% scaling drift).
- Operator re-scope is the canonical unblock path (per analyst iter-15/16 consolidated handoff).
