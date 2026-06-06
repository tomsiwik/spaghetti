# Energy-Gated Composition: Proof Verification Report

## Verdict: KILLED

Energy-gated composition with absolute energy gap threshold fails because ALL adapters
reduce NLL on ALL domains (100% negative energy gap). The Neyman-Pearson gate never
fires: at tau=0, all 5 adapters are always included, making gated composition
identical to uniform composition. The fundamental assumption that adapters increase
NLL on out-of-domain data is FALSE for these LoRA-adapted models.

## Theorem (restated from MATH.md)

The energy gap Delta_E = NLL(adapted) - NLL(base) is the Neyman-Pearson optimal
test statistic for discriminating helpful vs harmful adapters. Include adapter i
for query x iff Delta_E_i(x) < tau.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| Gated beats base on >= 4/5 domains | 1/5 (math only) | NO |
| Gated beats uniform on 5/5 domains | 0/5 (identical) | NO |
| Code gated > base by >= 10% | -13.5% | NO |
| Math gated > base by >= 50% | +49.8% | MARGINAL |
| Prose domains: gated >= -2% vs base | -5.2% to -6.8% | NO |
| Overhead < 20% of gen time | 4.7% | YES |
| Energy gap selectively gates | All gaps negative, 100% inclusion | NO |

## Hypothesis

Energy-gated composition (compose only adapters with negative energy gap per query)
will beat base model on >= 4/5 domains, recovering prose quality while preserving
structured domain wins.

**Result: KILLED. The gate never activates because ALL adapters always reduce NLL.**

## What This Experiment Is

An attempt to use per-query energy gap (NLL ratio) as a runtime gate for adapter
composition. For each prompt, compute Delta_E per adapter; only compose adapters
with Delta_E < tau (they reduce NLL). When no adapter helps, fall back to base model.

## Key Finding: Universal NLL Reduction

The energy gap matrix shows that EVERY adapter reduces NLL on EVERY domain:

```
Energy Gap Matrix (mean per-sample Delta_E):
              medical    code    math    legal   finance
medical       -1.45    -1.34   -0.49   -1.63   -1.57
code          -0.79    -1.93   -0.49   -1.70   -1.61
math          -0.79    -1.36   -1.00   -1.51   -1.47
legal         -0.97    -1.53   -0.50   -2.18   -2.02
finance       -0.97    -1.54   -0.54   -2.14   -2.08
```

All 25 cells are negative. No adapter increases NLL on any domain. The gap magnitudes
range from -0.49 (math adapter on medical) to -2.18 (legal adapter on legal).

This means at any threshold tau <= 0, ALL adapters are included for every query,
producing gated = uniform = 1/5 weighting. The gating mechanism has zero
discriminative power.

## Why Finding #182 Doesn't Transfer

Finding #182 (AUC=0.851) was computed using the **lora_scale_ablation** adapters on
the **Falcon-E-3B-Instruct-1.58bit** model with ground truth from MMLU/GSM8K task
accuracy. It measured whether energy gap RANKS adapters by task quality.

This experiment uses the **real_data_domain_experts** adapters on
**BitNet-b1.58-2B-4T** with generation quality as ground truth. The disconnect:

1. **Different model:** Falcon-E-3B vs BitNet-2B. The ternary model has different
   NLL landscape.

2. **Different adapters:** scale_ablation adapters vary widely (scales 1-20, SFT/NTP).
   real_data_domain_experts are all trained with the same hyperparameters (scale=20,
   SFT), so they all strongly reduce NLL everywhere.

3. **NLL reduction != task quality:** An adapter can reduce NLL (better next-token
   prediction) while degrading generation quality (mode collapse, repetition).
   This is exactly the PPL-quality disconnect documented in Finding #178 and the
   generation_quality_test LEARNINGS.

4. **Absolute vs relative:** Finding #182's AUC=0.851 was relative (ranking adapters
   within a domain). This experiment needs absolute gating (include/exclude), which
   requires a meaningful threshold. When all gaps are negative, no threshold works.

## Empirical Results (mean across 3 seeds)

| Domain   | Base  | Uniform | Gated (tau=0) | Gated vs Base |
|----------|-------|---------|---------------|---------------|
| medical  | 0.478 | 0.449   | 0.449         | -6.2%         |
| code     | 0.336 | 0.291   | 0.291         | -13.5%        |
| math     | 0.125 | 0.188   | 0.188         | +49.8%        |
| legal    | 0.465 | 0.440   | 0.440         | -5.2%         |
| finance  | 0.488 | 0.455   | 0.455         | -6.8%         |

Gated = Uniform on ALL 5 domains (correlation = 1.0 because same weights).

**Threshold sweep (seed=42 only):**
- tau=-0.1: 1/5 domains beat base (math slightly drops to 4.8 active adapters)
- tau=0.0: 1/5 domains beat base (all 5 active)
- tau=0.1: 1/5 domains beat base (all 5 active)

Even at tau=-0.1, only 2% of prompt-adapter pairs are excluded (2 out of 50 for
math domain). The gaps are so deeply negative that no practical threshold gates
effectively.

## Kill Criteria Assessment

- **K572 FAIL:** Gated composition worse on 4/5 domains (only math wins).
  Same as uniform composition -- the gate adds nothing.

- **K573 FAIL:** Energy gap threshold is IDENTICAL to uniform (AUC diff = 0).
  The gating signal has zero discriminative power at the absolute level because
  all energy gaps are negative.

- **K574 PASS:** Energy gap computation takes 38.5s for 50 prompts x 5 adapters
  = 0.77s per prompt. Generation takes ~16.2s per prompt (base) or ~101s per prompt
  (multi-adapter). Overhead is 4.7% of base gen time, well under 20%.

## Timing

- Energy gap computation: 38.5s (once, amortized)
- Per-seed generation: base=123s, uniform=810s, gated=811s
- Total experiment: 6895s (115 min)

## What Went Wrong (Root Cause Analysis)

The fundamental error was **Assumption 1**: "Energy gap computed on prompt tokens
generalizes to generation quality." This is false because:

1. LoRA adapters add parameters (A and B matrices) that create additional capacity.
   This extra capacity ALWAYS reduces NLL (better fit to any data), even on
   out-of-domain text. NLL reduction is a property of added parameters, not
   domain relevance.

2. The energy gap discriminator (Finding #182) worked because it had GROUND TRUTH
   task accuracy to calibrate against. It measured whether adapters that reduce
   NLL more also perform better on tasks. But it did NOT establish that adapters
   with POSITIVE energy gap exist. All 17 adapters in that experiment also had
   negative energy gaps -- the AUC was computed on the RANKING of negative gaps,
   not on a positive/negative boundary.

3. The Neyman-Pearson framing assumed a natural boundary at Delta_E=0 (adapter
   helps vs hurts). But with LoRA, the boundary doesn't exist at 0 -- ALL
   adapters "help" in the NLL sense. The actual boundary between "helps" and
   "hurts" in the GENERATION sense cannot be determined from NLL alone.

## Limitations

- Only tested on BitNet-b1.58-2B-4T with real_data_domain_experts adapters
- 5 domains x 10 prompts x 3 seeds (small sample)
- Generation quality metrics are crude proxies (keyword density, etc.)
- Did not test relative energy gap ranking (top-k by gap magnitude)

## What Would Work Instead

1. **Relative energy gap ranking (top-k):** Instead of absolute threshold, select
   the adapter with the MOST negative gap per query. This is oracle top-1 routing,
   which the generation_quality_test showed works for code (+14.4%) and math (+142.1%).
   The energy gap can serve as the ROUTING signal, not the GATING signal.

2. **Generation-based discriminator:** Instead of NLL, evaluate adapter quality by
   generating a short sample and scoring it. But this is expensive (N forward passes
   per query).

3. **Task-specific fine-tuning of the gating threshold:** Train a small classifier
   on (energy_gap_vector, domain_quality) pairs to learn the non-trivial
   inclusion boundary.

## What Was Learned

1. **LoRA adapters universally reduce NLL.** This is a structural property of
   adding low-rank perturbations, not evidence that the adapter helps the task.
   The NLL reduction is proportional to adapter scale (scale=20 produces ~1-2 nats
   reduction on all data).

2. **NLL-based gating requires RELATIVE thresholds, not absolute.** The absolute
   energy gap cannot distinguish helpful from harmful adapters because all gaps
   are negative. A relative approach (select adapter with largest gap for this
   domain, or use ranking) may work.

3. **Finding #182's AUC is about RANKING, not GATING.** The discriminator ranks
   adapters by quality (Spearman rho=0.701 on math). But ranking is not the same
   as binary inclusion/exclusion. The finding should be used for routing (which
   adapter is BEST), not gating (which adapters to INCLUDE).

4. **The two-world pattern persists.** Even with energy gating, composition helps
   math (+49.8%) and hurts prose (-5% to -13.5%). This is consistent with
   generation_quality_test and suggests the problem is in the adapters themselves
   (PPL-trained adapters cause mode collapse on prose), not in the composition
   mechanism.
