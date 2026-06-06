# PAPER — exp_g4_compose_bakeoff_top3

**Verdict: KILLED (preemptive, 5-theorem stack)**

Cohort: audit-2026-04-17 / composition-bug / g4-gemma4 — 29th consecutive
preemptive-kill this session (22nd composition-bug branch under ap-017;
11th SUPPORTED-source preempt registered as candidate (q)).

## Prediction vs measurement

| Theorem | Prediction | Runner measurement |
|---------|-----------|--------------------|
| T1 inventory shortfall | 2 Gemma-4 adapters missing at N=5; 0 runtime-LoRA pipelines on disk | **fail**: available=3 {code, math, medical}, shortfall=2, runtime_lora_pipelines_found=[] |
| T2 iter-budget breach | 2 × 20.92 min + 3 × harness ≥ 56.84 min > 30 min | arithmetic-level (MATH.md); not runner-tested |
| T3 success_criteria=[] | "Success Criteria: NONE" in `experiment get` | **fail**: verified present |
| T4 KC under-specification | 1/5 adjudicatable pins on K1628 (ε present, baseline/pooled/enum/rescale absent) | **fail**: matches=1 (ε only) |
| T5 F#173 compound non-transfer | ≥ 3 of 4 sub-breaches (F#164 MLP, F#269 p-value + impossibility, runtime-LoRA void, F#173 self-caveat) | **fail**: B + C + D present (A cosmetic runner false-negative) |

## T5 sub-breach ledger (runner-observed)

| Sub-breach | Observed | Mechanism |
|------------|----------|-----------|
| (A) F#164 BitNet-MLP non-transfer (λ=0.5 arm) | false (runner) | MATH-level; runner keyword miss on F#164 summary (same as exp_g4_task_arith_lambda_sweep — cosmetic) |
| (B-i) F#269 DARE BitNet-ternary scope | **true** | F#269 substrings {ternary, BitNet, s=20, s/(1-p)} ⊇ 1 |
| (B-ii) KC p=0.5 contradicts F#173 top-3 (p=0.9) | **true** | F#173 literal "DARE sparsification at p=0.9"; KC uses p=0.5 |
| (B-iii) F#269 impossibility persistence on MMLU | **true** | F#269 literal "MMLU math degradation persists across all drop rates" |
| (C) runtime-LoRA dynamic-routing scope vs static bakeoff | **true** | F#173 literal "runtime LoRA … for dynamic routing" |
| (C) Gemma 4 runtime-LoRA pipeline void | **true** | glob `*runtime_lora*` in micro/models/ → 0 hits |
| (D) F#173 self-caveat: "All recommendations need empirical validation" | **true** | F#173 caveat literal; source self-flags as theoretical |

Any single sub-breach in {B, C, D} suffices for T5 fail; all three
observed independently.

## T5 runner false-negative (A only)

Runner's T5 sub-breach (A) keyword check (`BitNet` ∨ `ternary` ∨ `MLP`)
returned false because `experiment finding-get 164` summarises F#164's
scope rather than quoting the base model, and the current summary text
does not preserve those substrings. The MATH-level (A) argument is
independently verifiable:

```
$ experiment finding-get 164   # manual inspection
# F#164 experiment: exp_lora_soups_cat (BitNet-2B MLP r=8 N=5)
# see micro/models/exp_lora_soups_cat/MATH.md for base-model scope
```

F#164 scope (BitNet-2B ternary MLP, PPL metric, λ ∈ [0, 0.5]) does not
transfer to K1628 scope (Gemma 4 E4B FP16 v_proj, MMLU-Pro accuracy,
λ=0.5 single point). Sub-breach (A) holds; runner gap is cosmetic.
Defense-in-depth intact via {B, C, D}.

## Defense-in-depth

T1 (inventory shortfall=2 + runtime-LoRA pipeline void) OR T3
(success_criteria=[]) OR T5 (F#173 compound non-transfer, ≥3 sub-
breaches) each alone blocks SUPPORTED. T2 (iter-budget) + T4 (KC under-
spec 1/5 pins) reinforce. No single patch unblocks.

## Registration

Register F#173 compound-non-transfer as reusable one-line preempt (q)
under ap-017 alongside prior (a)-(p): F#306, F#13/F#14, F#44, F#45,
F#164, F#269, F#505, F#454, F#534, F#427, F#536, F#444, F#496, F#474,
F#502, F#452/F#453. 11th SUPPORTED-source preempt; 1st theory-
aggregation-source variant (distinct from prior SUPPORTED-empirical-
source preempts).

## Kill criteria

- K1628: fail (preemptive; "one approach dominates others by ≥3pp
  MMLU-Pro" unreachable under T1 inventory-void, T3 SC=[], T5 source
  non-transfer).
