# Clone-Compete Evolution: Research Digest

## Hypothesis

Cloning the worst-performing BitNet-2B adapter (legal), fine-tuning the clone on
fresh domain data, and running a PPL-based tournament can improve adapter quality
monotonically without regressing other domains in the composition.

## What This Experiment Does

This is the first test of the **Evolve phase** of the Living Composable Model.
The three-phase architecture is Distill (create experts) -> Compose (merge them) ->
Evolve (improve them). Prior experiments validated Distill and Compose. This
experiment validates Evolve.

**Protocol:**
1. Start with 5 existing FP16 LoRA adapters (python, math, medical, legal, creative)
   trained on BitNet-2B-4T from `bitnet_2b_real_composition`
2. Identify worst adapter by relative PPL improvement (legal: +23.9% vs base,
   compared to math: +34.8%)
3. Clone the legal adapter (copy parameters)
4. Fine-tune clone on 500 fresh legal texts from law-stack-exchange (disjoint from
   original training data)
5. Run per-sample PPL tournament on 38 held-out samples
6. Check regression: replace original legal with clone in 5-adapter 1/N composition,
   measure PPL on all domains
7. Repeat for a second evolution round (iterability test)

**Connection to prior art:** This implements a simplified version of Sakana AI's
Evolutionary Optimization of Model Merging Recipes (2403.13187), but at the
adapter level rather than full model level. Instead of genetic crossover in
weight space, we use continued training (mutation) and PPL-based tournament
selection (fitness evaluation). The multi-armed bandit framing from the literature
suggests UCB or Thompson Sampling for adapter selection; here we use the simpler
binomial test for a two-armed comparison.

## Key References

- Sakana AI, "Evolutionary Optimization of Model Merging Recipes" (2403.13187):
  evolutionary search in model weight space and data flow space
- TIES-Merging (2306.01708): resolving sign conflicts in delta merging
- DARE (2311.03099): random drop + rescale for parameter-efficient merging
- LoRA-Flow (2402.11455): dynamic per-token fusion gates for LoRA composition
- Population-Based Training (Jaderberg et al., 2017): clone models, perturb
  hyperparameters/weights, evaluate, select winners. The clone-compete protocol
  is analogous to PBT applied to LoRA adapters with continued training as the
  perturbation operator

## Empirical Results

### Kill Criteria Assessment

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| K1: clone win rate | >60% | 62.1% (18/29 decisive), p=0.265 | INCONCLUSIVE |
| K2: tournament queries | <10K | 38 | PASS |
| K3: regression on others | <2% | max 0.06% (math) | PASS |

### Quality Trajectory (Legal Domain PPL)

| Version | PPL | vs Original | vs Base |
|---------|-----|-------------|---------|
| Base (no adapter) | 20.78 | +31.4% | -- |
| Original adapter | 15.82 | -- | -23.9% |
| Clone v1 (200 steps) | 14.50 | -8.3% | -30.2% |
| Clone v2 (200 more steps) | 13.04 | -17.6% | -37.2% |

**Monotonic improvement:** Yes. Each evolution round strictly improves PPL.

### Tournament Details

**Round 1 (original vs clone v1):**
- 38 samples evaluated, 29 decisive (9 ties)
- Clone wins: 18 (62.1%), Original wins: 11 (37.9%)
- p-value: 0.265 (not significant at alpha=0.05 -- insufficient samples)
- Aggregate PPL: clone 14.50 vs original 15.82 (-8.3%)

**Round 2 (clone v1 vs clone v2):**
- 38 samples evaluated, 28 decisive (10 ties)
- Clone v2 wins: 14 (50.0%), Clone v1 wins: 14 (50.0%)
- p-value: 1.0 (no per-sample winner)
- Aggregate PPL: v2 13.04 vs v1 14.50 (-10.1%)

The divergence between per-sample win rate and aggregate PPL in round 2 is
informative: v2 wins equally often but wins bigger on the samples it does win
(hard legal texts where continued training helps most).

### Regression Check (1/N Composition)

Replacing original legal with clone v1 in the 5-adapter composition:

| Domain | Composed (original) | Composed (clone) | Delta |
|--------|-------------------|-----------------|-------|
| python | 2.74 | 2.74 | -0.14% |
| math | 4.42 | 4.42 | +0.06% |
| medical | 6.27 | 6.26 | -0.16% |
| legal | 19.28 | 19.19 | -0.44% |
| creative | 5.87 | 5.85 | -0.24% |

No domain regresses. Most domains slightly improve (noise floor, but consistent
direction suggests the clone is marginally better for composition too).

### Timing

- Clone training: 86s (200 steps)
- Tournament: ~30s (38 samples)
- Regression check: ~60s
- Total per evolution round: ~3 minutes on Apple Silicon

## Limitations

1. **Tournament underpowered (N=38).** The per-sample binomial test requires
   ~200 decisive samples for statistical significance at delta=0.10. The
   law-stack-exchange dataset had only 38 usable samples beyond the training
   split. K1 passes the point estimate (62.1% > 60%) but the p-value (0.265)
   means we cannot reject H0 at alpha=0.05. A larger legal dataset would
   resolve this.

2. **PPL-only evaluation.** Task eval was killed at 2B scale
   (exp_bitnet_task_eval). PPL improvement does not guarantee task quality
   improvement. This is a known limitation of the entire micro-scale program.

3. **Single seed.** All results from seed 42. Prior experiments show CV=0.5%
   across seeds for this architecture (bitnet_multiseed_validation).

4. **"Worst" adapter selection is metric-dependent.** Python had the lowest
   relative improvement (+18.2%) but legal had the highest absolute PPL (15.82).
   The experiment targeted legal per the hypothesis design, which happened to
   be the domain with highest absolute PPL. The "worst adapter" auto-detection
   flagged python instead (lowest relative improvement). Both are valid targets.

5. **Clone data overlap.** The clone training data came from the same dataset
   (law-stack-exchange) as the original, just different samples. A truly
   independent data source (e.g., legalbench) would be a stronger test, but
   legalbench uses loading scripts that are no longer supported.

6. **Tournament v2 inconclusive.** Clone v2 achieves lower aggregate PPL than
   v1 (-10.1%) but wins only 50% of per-sample comparisons. This means the
   improvement concentrates on hard samples. At macro scale, this pattern
   should be monitored -- an adapter that wins 50% per-sample might not be
   universally better.

7. **No cold-start control.** The experiment does not include a fresh-from-scratch
   baseline (training a new adapter on the same data without inheriting the
   original's weights). Without this control, the improvement cannot be attributed
   specifically to the warm-start mechanism (clone inheriting original weights)
   versus simply training on more data. The evolutionary framing rests on the
   inheritance advantage; a cold-start control would disambiguate warm-start
   benefit from additional-data benefit. This is deferred to the powered
   replication (exp_bitnet_clone_compete_powered).

8. **Tournament data reuse across rounds.** Both tournament rounds evaluate on
   the same 38 held-out samples, introducing potential selection bias. If round 1
   selected a clone that happened to perform well on these specific samples,
   round 2's evaluation on the same samples is not independent. The powered
   replication should use independent held-out sets per round.

9. **Scale limitation.** 2B ternary base, d=2560, 5 adapters. The regression
   bound O(epsilon/N) predicts even lower regression at N=50 or N=500, but
   this is untested.

## What Would Kill This

**At micro scale:**
- K1 failure with adequate sample size (200+ samples, clone wins <60%)
- Quality oscillation across rounds (v3 worse than v1)
- Clone training divergence (loss increases during continued training)

**At macro scale (exp_clone_compete_evolution):**
- Clone does not win >70% of macro tournament (stronger threshold)
- Tournament requires >50K queries at 7B scale
- Task accuracy (HumanEval, MATH-500) regresses even if PPL improves
- Evolution causes measurable MMLU regression (>2pp on any subject)
- Clone-compete overhead makes it slower than simply retraining from scratch
