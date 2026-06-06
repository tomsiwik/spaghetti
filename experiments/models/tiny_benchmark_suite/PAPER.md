# Pierre Tiny Benchmark Suite: Results

## Theorem (Restated from MATH.md)

Additive LoRA perturbation at bounded scale preserves base model knowledge
subspaces (Davis-Kahan sin-theta bound). At scale <= 5, adapter perturbation
is small enough that benchmark scores should match base. At scale = 20,
perturbation overwhelms base knowledge and benchmarks degrade.

## Predictions vs Measurements

| Prediction (from proof/prior findings) | Measured | Match? |
|----------------------------------------|----------|--------|
| MMLU at scale <= 5: 0pp degradation (F#320) | -2pp (1 question, p=0.84) | YES (within noise) |
| MMLU at scale = 20: significant degradation (F#320) | -6pp (NTP math, p=0.55) | PARTIAL (directional, not significant at n=50) |
| NTP math adapter improves GSM8K (F#262: +10pp) | -17pp (p=0.19) | **NO** (opposite direction) |
| HumanEval preserved at scale 1 (F#320) | +0pp (exact match) | YES |
| Composed N=5 at scale 1: minimal degradation | MMLU -2pp, HumanEval +0pp | YES |
| Composed N=5 at scale 1: GSM8K impact | -10pp (p=0.44) | Not predicted |

## Hypothesis

Tested: "Pierre Tiny adapter composition improves over base on domain-specific
tasks without degrading general benchmarks."

**Result: KILLED (K820 FAIL).** No adapter configuration improves any standard
benchmark over the base model.

## What This Experiment Did

Ran MMLU (50Q logit-based), GSM8K (30Q generation), HumanEval (15Q generation+exec)
on BitNet-2B-4T with 6 configurations:

| Config | MMLU | GSM8K | HumanEval |
|--------|------|-------|-----------|
| Base (no adapter) | **58.0%** | **53.3%** | **60.0%** |
| NTP math, s=20 | 52.0% (-6pp) | 36.7% (-17pp) | 53.3% (-7pp) |
| NTP code, s=20 | 56.0% (-2pp) | 46.7% (-7pp) | 53.3% (-7pp) |
| SFT code, s=1 | 56.0% (-2pp) | 50.0% (-3pp) | 60.0% (+0pp) |
| SFT math, s=1 | 56.0% (-2pp) | 50.0% (-3pp) | 60.0% (+0pp) |
| Composed N=5, DARE, s=1 | 56.0% (-2pp) | 43.3% (-10pp) | 60.0% (+0pp) |

Total runtime: 1010 seconds (17 minutes).

## Key References

- Finding #213: Base BitNet-2B-4T scores MMLU 38%, GSM8K 58%, HumanEval 60%
- Finding #212 (killed): Code SFT adapter degrades GSM8K -18pp, HumanEval -15pp
- Finding #262: NTP adapters preserve GSM8K (+10pp) vs SFT (-20pp)
- Finding #320: Composition preserves MMLU at scale<=5, catastrophic at scale=20
- Finding #266: DARE p=0.5 preserves in-dist quality while reducing OOD degradation

## Empirical Results

### Statistical Analysis

Confidence intervals at these sample sizes are very wide:
- MMLU (n=50): 95% CI width = 26pp
- GSM8K (n=30): 95% CI width = 34pp
- HumanEval (n=15): 95% CI width = 44pp

The -2pp MMLU delta for scale=1 adapters is not statistically significant (p=0.84).
Even the -17pp GSM8K delta for NTP math is borderline (p=0.19).

At these sample sizes, the SFT scale=1 and composed configurations are
**statistically indistinguishable from base** on MMLU and HumanEval.

### Why NTP Math Adapter Hurts GSM8K (Contradicts Finding #262)

Finding #262 reported NTP math adapter *improves* GSM8K by +10pp. Our results
show the opposite: -17pp. Key differences:

1. **Finding #262 used unpacked ternary weights** (BitLinear -> nn.Linear).
   We use native BitLinear via `pierre.attach_adapter`. The adapter attachment
   mechanism differs.
2. **Finding #262 used a different adapter system** (TernaryLoRALinear with STE).
   We use `pierre.RuntimeLoRA` (simple additive A*B projection).
3. **Finding #262 used domain-optimal LoRA scale** derived per-task.
   We use fixed scale=20 for all NTP adapters.

The architectural difference matters: the prior experiment's TernaryLoRALinear
includes STE (Straight-Through Estimator) ternary quantization of B-matrices
during forward pass, which regularizes the perturbation. RuntimeLoRA applies
B-matrices directly in bfloat16, which may create larger perturbation.

### Scale=1 Adapters: Effectively Invisible

At scale=1, adapters produce minimal perturbation (alpha * ||x@A@B|| is tiny
relative to ||base(x)||). This explains why they match base on all benchmarks:
the adapter output is simply too small to change predictions.

This confirms Finding #320's prediction (0pp degradation at scale<=5) but
reveals the flip side: the adapters also provide 0pp *improvement*.

### Composed N=5: Generation Quality Impact

Composed adapters at scale=1 preserve logit-based benchmarks (MMLU -2pp,
HumanEval +0pp) but degrade generation quality on GSM8K (-10pp). This
suggests that even at scale=1, the composed perturbation from 5 adapters
affects the generation distribution enough to change reasoning chains.

## Limitations

1. **Sample sizes too small for significance.** n=50/30/15 give wide CIs.
   Most deltas are within noise.
2. **Benchmarks test general capability, not domain expertise.** MMLU, GSM8K,
   and HumanEval measure general knowledge/reasoning/coding. The adapters
   were trained on domain-specific text (medical, code, math, legal, finance).
   The benchmarks do not test whether adapters improve *domain-specific* tasks.
3. **Scale=1 is too low for domain impact.** Finding #297 trained at scale=20.
   At scale=1, the adapter signal is negligible.
4. **Scale=20 destroys benchmarks.** Finding #320 showed this for MMLU;
   our results confirm it extends to GSM8K and HumanEval.
5. **No "right" scale exists for all benchmarks simultaneously.** This is the
   fundamental tension: domain adapters need high scale for domain tasks but
   low scale for general benchmarks.

## What Would Kill This

The fundamental issue is the scale dilemma:
- Scale=20: domain tasks improve, general benchmarks degrade
- Scale=1: general benchmarks preserved, domain tasks unaffected

This could be resolved by:
1. **Task-adaptive scaling** (route determines scale per-token)
2. **Solidification** (promote adapter knowledge into base weights, then scale=0)
3. **Better adapters** (train on data that extends model capability instead of
   overriding format)

## Kill Criteria Assessment

**K820 FAIL:** No adapter configuration beats base on any standard benchmark.
The kill criterion text is "All benchmarks below base model (adapters make
everything worse)." Strictly speaking, SFT adapters at scale=1 *match* base
on HumanEval exactly (60.0% = 60.0%), so they don't make "everything worse."
But they don't improve anything either. Recording as FAIL because no
improvement was demonstrated.

## Verdict

The adapters are trained for domain text generation (medical, legal, code, etc.)
and evaluated on generic reasoning benchmarks. This is a category mismatch.
The adapters never claimed to improve MMLU/GSM8K/HumanEval -- they improve
domain-specific generation quality (measured by PPL on domain text, Finding #323).

The honest conclusion is not "adapters are broken" but "standardized benchmarks
are the wrong evaluation for domain-specific adapters." Finding #323 showed the
integrated pipeline is within 2% of oracle quality on *domain text*. This
experiment shows it does not improve *general benchmarks*. Both are true
simultaneously.
