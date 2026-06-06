# PAPER — exp_g4_task_arith_lambda_sweep

**Verdict: KILLED (preemptive, 5-theorem stack)**

Cohort: audit-2026-04-17 / composition-bug / g4-gemma4 — 18th consecutive
preemptive-kill this session.

## Prediction vs measurement

| Theorem | Prediction | Runner measurement |
|---------|-----------|--------------------|
| T1 inventory shortfall | 2 adapters missing at N=5 | **fail**: available=3 {code, math, medical}, shortfall=2 |
| T2 iter-budget breach | 2 × 20.92 min = 41.84 min > 30 min | arithmetic-level (MATH.md); not runner-tested |
| T3 success_criteria=[] | "Success Criteria: NONE" in `experiment get` | **fail**: verified present |
| T4 KC under-specification | 0/5 required adjudicatable keywords in K1608 | **fail**: matches=0 |
| T5 F#164 non-transfer | Arch + metric scope mismatch (BitNet-2B MLP PPL vs Gemma 4 v_proj MMLU-Pro) | **pass (runner only)**: F#164 present but BitNet/ternary/MLP substrings absent from summary — see note below |

## T5 runner false-negative

Runner's T5 keyword check (`BitNet` ∨ `ternary` ∨ `MLP`) returned pass because
the finding-get output for F#164 summarises rather than quoting the base
model. The MATH-level T5 argument is independently verifiable:

```
$ experiment finding-get 164   # inspected manually
# Experiment: exp_lora_soups_cat (BitNet-2B MLP r=8 N=5; see
#   micro/models/exp_lora_soups_cat/MATH.md for base-model scope)
```

F#164 scope (BitNet-2B ternary MLP, PPL metric, λ ∈ [0, 0.5]) does not
transfer to K1608 scope (Gemma 4 E4B FP16 v_proj, MMLU-Pro task accuracy,
λ ∈ {0.2, 0.33, 0.5, 0.67, 1.0}). T5 argument holds; runner gap is cosmetic.
Defense-in-depth: T1 ∨ T3 ∨ T4 alone already block SUPPORTED.

## Defense-in-depth

T1 (inventory shortfall=2) OR T3 (success_criteria=[]) OR T4 (KC under-spec,
0/5 adjudicatable fields) each alone blocks SUPPORTED. T2 (iter-budget) +
T5 (F#164 non-transfer) reinforce. No single patch unblocks.

## Operator unblock (cohort-wide)

Per audit-2026-04-17 cohort convention:
1. `experiment success-add exp_g4_task_arith_lambda_sweep --text "<falsifiable
   statement with epsilon, baseline, 14-discipline pooling, per-λ n, adapter
   identities>"`
2. Train 2 Gemma 4 domain adapters (2 × 20.92 min = ~42 min; needs pueue
   handoff) OR re-scope K1608 to N=3 with available {code, math, medical}.
3. Pin K1608: define baseline (best-of-{λ≠0.5}), ε=2pp semantics (absolute
   MMLU-Pro accuracy delta), 14-discipline pooled vs micro-avg, sample count
   n per λ.

Without (1)–(3) the claim is not falsifiable; SUPPORTED is impossible.

## References

- F#164 exp_lora_soups_cat (TA λ=0.5 best, BitNet-2B MLP orthogonal adapters, PPL)
- F#306 batched-LoRA framework displacement (MLX lazy-eval) — ap-017 preempt (a)
- F#13/F#14 1/N regularization non-transfer — ap-017 preempt (b)
- F#44 5-domain real HF non-transfer — ap-017 preempt (c)
- F#45 BitNet-2B ternary convergence non-transfer — ap-017 preempt (d)
- ap-017 partial-cascade-insufficiency — 17→18 instances
- Wang et al 2024 (arxiv:2406.01574) — MMLU-Pro 14-discipline benchmark

## Routing

analyst iter 18: reference ap-017 scope addendum (17→18 instances; branches
composition-bug 17 + scale-safety 1). Register F#164 non-transfer one-liner
under ap-017 preempts (e). No new antipattern.

reviewer iter 18: verify T1 (find), T3 (experiment get), T4 (K1608 text),
T5 (finding-get F#164 + MATH.md manual-scope argument). Runner false-negative
on T5 cosmetic; defense-in-depth T1 ∨ T3 ∨ T4 holds.
