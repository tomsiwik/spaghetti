# Adapter ELO Tournament: Research Digest

## Hypothesis

An ELO tournament comparing adapter variants via pairwise composition PPL will rank
adapters consistently with their individual (standalone) quality, enabling a
selection mechanism for the Evolve track (Kendall tau >= 0.7).

## What This Experiment Is

We trained 4 LoRA adapter variants per domain (3 domains: medical, math, code)
on BitNet-2B-4T, varying seed and learning rate. Then we ran a full round-robin
ELO tournament within each domain: for every pair of variants, we composed each
with the same set of other-domain adapters and compared PPL on held-out data.
The variant producing lower composition PPL wins the match. After 2 full rounds
(12 matches per domain), we computed ELO ratings and compared the ELO ranking
against the ground-truth ranking from standalone adapted PPL.

## Key References

- Elo rating: standard Bradley-Terry model, used in LMSYS Chatbot Arena (arxiv 2403.04132)
- Composition mechanism: proven in prior experiments (Grassmannian orthogonality,
  gamma=0.982 at N=25)

## Empirical Results

### Training Phase (12 adapters, 3 domains x 4 variants)

| Domain   | Base PPL | Best Variant    | Best PPL | Worst Variant | Worst PPL | Spread |
|----------|----------|-----------------|----------|---------------|-----------|--------|
| medical  | 15.05    | high_lr (2e-4)  | 7.97     | low_lr (5e-5) | 8.92      | 11.9%  |
| math     | 4.26     | alt_seed (s=99) | 3.07     | low_lr (5e-5) | 3.15      | 2.6%   |
| code     | 2.35     | baseline (s=42) | 2.06     | low_lr (5e-5) | 2.10      | 1.9%   |

Total training time: 210s (12 adapters, 100 iterations each, ~17.5s/adapter).

### Tournament Phase

Tournament time: 70.2 seconds (36 matches across 3 domains, ~2s/match).
This is well within the K2 budget of 30 minutes. At this rate, 10 adapters per domain
would require 10*9/2 * 2 rounds = 90 matches * 2s = ~3 min per domain.

**ELO rankings were perfectly consistent across all 3 domains:**

| Rank | ELO Winner  | ELO Rating |
|------|-------------|------------|
| 1    | baseline    | 1584       |
| 2    | high_lr     | 1530       |
| 3    | alt_seed    | 1472       |
| 4    | low_lr      | 1414       |

This is the SAME ordering in medical, math, and code -- a strong signal that
the tournament measures something real and stable.

### Correlation with Standalone Quality

| Domain   | ELO Order                              | Quality Order                         | Kendall tau |
|----------|----------------------------------------|---------------------------------------|-------------|
| medical  | baseline > high_lr > alt_seed > low_lr | high_lr > alt_seed > baseline > low_lr | 0.333       |
| math     | baseline > high_lr > alt_seed > low_lr | alt_seed > baseline > high_lr > low_lr | 0.333       |
| code     | baseline > high_lr > alt_seed > low_lr | baseline > high_lr > alt_seed > low_lr | 1.000       |

- Mean Kendall tau: 0.556
- Min Kendall tau: 0.333

### Kill Criteria Assessment

- **K1 FAIL:** min Kendall tau = 0.333 < 0.5 threshold. ELO ranking does not
  reliably correlate with individual adapter quality.
- **K2 PASS:** Tournament overhead = 70s, far below 30 min threshold.
- **S1 FAIL:** Mean Kendall tau = 0.556 < 0.7 threshold.

## Key Finding: ELO Measures Composition Compatibility, Not Individual Quality

The tournament produced PERFECTLY CONSISTENT rankings across all 3 domains --
the same adapter (baseline, seed=42, lr=1e-4) won every single match in every
domain. This is too consistent to be noise. The tournament is measuring a real
property, but it is **composition compatibility**, not standalone quality.

Why does "baseline" always win composition matches despite not being the best
standalone adapter? The context adapters used in all compositions are from the
"baseline" variant of each domain (same seed=42, lr=1e-4). The baseline variants
were trained with the same hyperparameters and similar random initialization.
This means the baseline variant's B-matrices have the most compatible interference
pattern with the context adapters, giving it a systematic advantage in composition.

This is the non-transitivity / non-monotonicity scenario from MATH.md section 3:
composition quality depends on pair-specific interactions, not just individual
quality. The Grassmannian A-matrices prevent catastrophic interference but do not
prevent beneficial interference patterns from emerging between similarly-initialized
adapters.

## What Was Learned

1. **ELO tournament is computationally trivial** (70s for 36 matches). K2 is not a concern.
   Scales to 10 adapters * 3 domains in ~9 min.

2. **Composition PPL != standalone PPL.** An adapter that is individually best
   (medical/high_lr, PPL=7.97) may not compose best with other adapters in the system.
   This is a fundamentally important insight for the Evolve track.

3. **The tournament measures the RIGHT thing for deployment.** If we care about
   composed system quality (which we do -- that's the whole point), then ELO
   composition ranking is MORE relevant than standalone quality. The kill criterion
   asked for correlation with individual quality, but the operationally correct
   metric IS composition quality.

4. **Context-dependent ranking.** The baseline always wins because context adapters
   are also baselines. If context adapters change, rankings would change. This means
   the tournament should be re-run when the adapter ensemble changes.

## Reframing: Is This Actually a Kill?

Formally K1 fails (tau=0.333 < 0.5). But the finding is more nuanced:

- The tournament WORKS as a ranking mechanism (consistent, reproducible, fast)
- It just measures COMPOSITION quality, not INDIVIDUAL quality
- For the Evolve track, composition quality is arguably the correct selection criterion
- The fix for correlating with individual quality: use DIVERSE context adapters
  (not all from the same variant) or evaluate each variant both standalone AND composed

**Verdict:** KILLED on the stated criterion, but the mechanism is sound for
a composition-aware selection process. Future work should either:
(a) Redefine the selection criterion to be composition quality (which this measures)
(b) Use a hybrid score: alpha * standalone_quality + (1-alpha) * composition_elo

## Limitations

- Only 4 variants per domain (minimum for meaningful tau)
- Variants differ only by seed/lr, not by data subset or architecture
- Context adapters always from "baseline" variant (biases toward baseline)
- 100 training iterations (less converged than production 200-500 iterations)
- 3 domains only (toy scale)

## What Would Kill This at Larger Scale

- If composition PPL becomes noisy (high variance across eval batches), matches
  become unreliable. Would need more eval data or multiple match rounds.
- If adapter quality spread narrows further (math/code: 2-3% spread), all rankings
  become noise-dominated.
