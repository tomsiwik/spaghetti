# MATH.md — exp_g4_compose_bakeoff_top3 (PREEMPTIVE KILL, 5-theorem)

**Claim under test (K1628):** "one approach dominates others by >=3pp
MMLU-Pro" for the head-to-head bakeoff {Task Arithmetic λ=0.5, DARE p=0.5,
runtime LoRA} on Gemma 4 E4B N=5 composition.

**Verdict (predicted pre-run):** KILLED via 5-theorem stack. Any of
T1/T3/T5 individually blocks SUPPORTED; T2/T4 reinforce. 29th cohort
preemptive-kill this session (22nd composition-bug branch under ap-017).

---

## T1 — Triple-arm inventory shortfall (N=5, 3 domains, runtime-LoRA void)

**Theorem.** K1628 compares three merge methods on a fixed N=5 Gemma 4
composition. Each arm requires N distinct Gemma 4 E4B domain adapters at
test time; the runtime-LoRA arm additionally requires a Gemma-4 runtime-
LoRA serving pipeline with sub-batch adapter attach/detach.

**Evidence.** `micro/models/exp_p1_t2_single_domain_training/adapters/` =
{code, math, medical} (3 T2.1-trained Gemma 4 adapters). Shortfall per
arm = 5 − 3 = 2. No Gemma 4 runtime-LoRA serving pipeline exists in
`micro/models/` (verified via glob `*runtime_lora*`, `*runtime-lora*` → 0
hits); F#454 literal: "0 user-style adapters on Gemma 4". The bakeoff
would need 5 adapters × 3 arms + runtime-serving harness = structurally
unreachable.

**Consequence.** Bakeoff at N=5 is structurally unreachable with current
Gemma 4 inventory across all three arms. Substituting BitNet-2B ternary
adapters (the F#164/F#269 source architecture) invalidates the `g4-gemma4`
tag of K1628.

## T2 — Iter-budget arithmetic (training shortfall × three-arm multiplier)

**Theorem.** Training 2 missing Gemma 4 domain adapters at T2.1 mean cost
20.92 min/adapter → 41.84 min. Three-arm bakeoff then runs each arm on
the same N=5 composed model: 3 × MMLU-Pro harness at ~5-10 min/arm ≈
15-30 min additional. Total ≥ 56.84 min > 30 min iter budget; still
< 120 min micro ceiling so T2 alone does not fatal-block (reinforces T1).

**Consequence.** Feasible only via pueue + cross-hat handoff; combined
with T1 inventory-void, infeasible within single micro cycle.

## T3 — success_criteria = [] (structural blocker)

**Theorem.** `experiment get exp_g4_compose_bakeoff_top3` returns
`Success Criteria: NONE — add with: experiment success-add`. Per repo
guardrail 1009 + PLAN §1 verdict-consistency, SUPPORTED requires ≥1
success criterion the experiment satisfies.

**Consequence.** No SUPPORTED verdict emittable regardless of measurement,
because the predicate defining "supported" is empty. KC-only path
(K1628 pass/fail) cannot upgrade without operator `experiment success-add`.

## T4 — K1628 under-specification

**Theorem.** K1628 = `"one approach dominates others by >=3pp MMLU-Pro"`.
Required adjudicatable fields:
- (a) ε pinned: "≥3pp" present → ε-pin matches (via `>=3pp` substring).
- (b) baseline: ✗ — "dominates others" does not pin baseline (other two
  arms? shared base model? pre-registered control?).
- (c) pooled vs discipline: ✗ — "on MMLU-Pro" does not disambiguate
  pooled accuracy vs discipline-max vs discipline-median; MMLU-Pro has
  14 disciplines and a 3pp pooled margin can flip under per-discipline
  aggregation (F#474 precedent).
- (d) enumerated arms: ✗ — "one approach" without labelled arm IDs leaves
  tie-breaking (2-way tie at 3pp vs all three arms ≥3pp) unadjudicated.
- (e) rescale convention: ✗ — DARE p=0.5 rescale formula s/(1-p) vs
  1/(1-p) vs vanilla is unspecified; F#269 NOTES established s/(1-p)
  for BitNet ternary scale s=20, Gemma 4 LoRA scale is scope-different.

1/5 pins. T4 alone does not block (single pin > 0 by some reviewer
conventions), but reinforces T1/T3/T5.

## T5 — F#173 compound non-transfer (multi-source scope aggregation)

**Theorem.** K1628 derives from F#173 (notes: "Motivated by Finding #173
top-3 composition recommendations"). F#173 is a theory-aggregation
finding (experiment: exp_notebooklm_composition_theory) whose top-3
recommendations each ride on a distinct upstream BitNet-scope finding.
K1628 breaches F#173's scope on ≥ 3 independent axes, each sufficient.

**(A) λ=0.5 arm: F#164 BitNet-MLP non-transfer.** F#173 top-3 item (1)
"lambda 0.5-1.0" derives from F#164 (exp_lora_soups_cat). F#164 scope:
BitNet-2B-4T ternary, MLP r=8, PPL metric, 125 calibration samples,
d=17.2M soups. K1628 scope: Gemma 4 E4B FP16, v_proj/LoRA, MMLU-Pro
accuracy, 14 disciplines. F#164 caveat literal: "CAT… describes per-
LoRA-module but code does per-tensor composition" → implementation-
scope mismatch unresolved. Ap-017 preempt (e) already ratified.

**(B) DARE p=0.5 arm: F#269 BitNet-ternary non-transfer + KC-vs-source
contradiction.** F#173 top-3 item (2) recommended "DARE sparsification at
**p=0.9**". KC explicitly uses **p=0.5**, which comes from F#269 NOTES
literal: "Optimal p for ternary adapters is 0.5 (not 0.9 as in DARE
paper) because LoRA scale s=20 creates effective perturbation s/(1-p)".
K1628 thus:
  (i) contradicts its own cited source F#173 on the drop-rate value;
  (ii) silently substitutes F#269's BitNet-ternary p-optimum for the
       FP16 Gemma 4 regime, where s=20 does not apply;
  (iii) F#269 impossibility structure: "DARE fixes density interference
       but cannot fix direction interference. MMLU math degradation
       (-8pp) persists across ALL drop rates." → K1628's "dominates by
       ≥3pp on MMLU-Pro" collides with F#269's explicit negative
       impossibility result on MMLU-math.

Ap-017 preempt (f) already ratified (F#269 non-transfer).

**(C) Runtime-LoRA arm: F#173 scope is "dynamic routing", not static
bakeoff.** F#173 literal: "runtime LoRA confirmed as default **for
dynamic routing**". Dynamic routing means per-query or per-token adapter
attach/detach; a static bakeoff against merge methods compares the
wrong thing — runtime LoRA's design advantage is load-on-demand, not
composition quality under a fixed N=5. F#454 further shows 0 Gemma 4
runtime-LoRA pipelines exist.

**(D) F#173 self-caveat: non-validation disclosure.** F#173 caveats
literal: "**All recommendations need empirical validation.** Lambda
>1.0 may exit quadratic basin. **DARE untested on ternary adapters
specifically.** Shared-specific decomposition is speculative." Source
self-flags that its top-3 is theoretical; porting to Gemma 4 FP16
without first validating on BitNet-ternary doubles the unvalidated-
scope gap.

**Consequence.** F#173 cannot be cited to pre-validate the bakeoff
claim on Gemma 4 FP16 MMLU-Pro. Register as reusable ap-017 preempt (q)
alongside prior (a)-(p): F#306, F#13/F#14, F#44, F#45, F#164, F#269,
F#505, F#454, F#534, F#427, F#536, F#444, F#496, F#474, F#502, F#452/F#453.

---

## Defense-in-depth summary

| Theorem | Blocks SUPPORTED alone? | Evidence path |
|---------|-------------------------|---------------|
| T1 (inventory shortfall=2, runtime-LoRA void) | YES | `ls micro/models/exp_p1_t2_single_domain_training/adapters/`, glob `*runtime_lora*` |
| T2 (iter budget breach)    | NO (alone) | 20.92 × 2 + 3 × harness = ≥ 56.84 min > 30 min |
| T3 (success_criteria=[])   | YES | `experiment get` output |
| T4 (KC under-spec)         | NO (1/5 pins) | K1628 text: ε pinned, baseline/pooled/enum/rescale absent |
| T5 (F#173 compound non-transfer) | YES | F#164+F#269+F#454 scope audit + F#173 self-caveat |

T1 ∨ T3 ∨ T5 each alone blocks.

## MATH → runner reconciliation

Runner implements T1 (filesystem adapter count + runtime-LoRA glob), T3
(substring on "Success Criteria: NONE"), T4 (K1628 keyword audit), T5
(finding-get F#173 + presence of F#164/F#269/F#454 scope markers). T2 is
arithmetic-level only (MATH.md captures). No KC edits.

## Prediction

Runner returns {T1: fail, T3: fail, T4: fail(1/5), T5: fail} → verdict
KILLED_PREEMPTIVE (all_block=true, defense_in_depth=true). Complete with
`--status killed --k 1628:fail`.
