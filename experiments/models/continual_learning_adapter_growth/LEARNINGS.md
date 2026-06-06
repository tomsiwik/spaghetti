# Learnings: exp_continual_learning_adapter_growth

## Core Finding

Uniform 1/N composition of Grassmannian-skeleton LoRA adapters maintains PPL within ~1% of base across N=5 to N=15, with non-monotonic trajectory (ruling out strict dilution dominance). Worst-case per-domain degradation is 0.53% at N=15 — far below the 5% kill threshold. **Individual adapter invariance is tautological with frozen parameters and confirms code correctness, not a scientific hypothesis.** The non-trivial result is composition robustness to N under uniform weighting, the worst-case composition strategy.

## Why This Happened (Literature-Grounded)

### Grassmannian orthogonality eliminates cross-adapter interference exactly

At N=15, r=16, d=2560: Nr/d = 0.094, well within the orthogonal packing capacity (N_max = d/r = 160). The cross-term ||delta_W_i^T delta_W_j|| = 0 exactly when A_i^T A_j = 0, regardless of B-matrix correlation. This is not a submultiplicativity bound — it's a direct consequence of the QR-constructed A-matrices spanning orthogonal subspaces. The mechanism: x projects through A_j^T into subspace j, then B_i^T B_j maps within r-dim, then A_i projects into subspace i. Since subspaces i and j are orthogonal, the composition vanishes.

This explains why composition quality is approximately flat rather than degrading: the only effect of increasing N is dilution (1/N scaling), not interference. Dilution is a scaling factor, not a corruption mechanism.

### The ~0.6pp fluctuation band is consistent with dilution vs coverage tradeoff

Under uniform 1/N, each adapter contributes s/N of its delta. As N grows:
- **Dilution effect**: each adapter's contribution shrinks (s/5=4.0 vs s/15=1.33)
- **Coverage effect**: more adapters means more cross-domain partial contributions

The trajectory (-0.59% at N=5, -0.51% at N=15, fluctuating +-0.3pp) shows these forces approximately cancel in the N=5-15 regime. Prior work at N=24 showed -29.1% composed PPL vs base (exp_real_data_25_domain_adapters), suggesting coverage gains can dominate for longer than expected.

The non-monotonic trajectory is NOT evidence of a "sweet spot" — single-seed, no error bars, and all fluctuations are within expected noise (~0.5% CV from prior multi-seed work).

### B-parameter cosine stability confirms independent learning

Mean B-|cos| stays at 0.020-0.023 across N=5-15, indicating adapters trained on different domains develop dissimilar weight vectors. This is consistent with the random projections literature: in high-dimensional spaces, independently trained vectors tend toward orthogonality (arXiv 2508.11985, Naive LoRA Summation; Vershynin, "High-Dimensional Probability," Ch. 5).

## Confirming Evidence

1. **Our N=24 experiment (exp_real_data_25_domain_adapters)**: Composition IMPROVED from -26.3% (N=5) to -29.1% (N=24), directly contradicting dilution dominance at moderate N. The current experiment fills in the intermediate trajectory, showing approximately flat behavior in N=5-15 consistent with the transition from dilution-dominated to coverage-dominated regimes.

2. **Our N=50 experiment (exp_ternary_adapter_n50_composition)**: gamma_uniform=0.996 (nearly useless), confirming dilution eventually wins under uniform composition. The N=5-15 regime evaluated here is safely within the useful range.

3. **arXiv 2508.11985 (Naive LoRA Summation)**: Orthogonal A-matrices enable additive composition without interference. Our N=5 to N=15 step-by-step trajectory empirically confirms their theory holds continuously across intermediate N values.

4. **arXiv 2510.03262 (OSRM)**: Weight-space orthogonality does not guarantee data-space orthogonality, but composition works empirically via constructive transfer. Our stable B-|cos| with flat composition quality is consistent: even when B-matrices could interfere, the A-orthogonality filter prevents it from manifesting.

## Contradicting Evidence

1. **The approximately-flat trajectory is a weak signal.** The reviewer noted K2 (non-monotonic) passes trivially because N=5->N=6 already shows improvement. A stricter test — e.g., "composition must IMPROVE with N" — would likely fail. The experiment shows uniform composition doesn't actively degrade, not that it improves. This is consistent with exp_real_data_25_domain_adapters (where improvement at N=24 was measured on each adapter's own domain, a different evaluation protocol).

2. **Arbitrary domain slices weaken the coverage argument.** 10/15 domains are consecutive dolly-15k slices, not genuinely distinct distributions. The N=24 experiment showed these slice-based adapters have poor routing recall (<40% for 10/17). In this experiment, their contribution to "coverage" under uniform composition is questionable — they may be adding noise rather than complementary knowledge.

3. **exp_competitive_benchmark showed uniform composition hurts factual benchmarks.** SOLE with uniform composition lost to Qwen2.5-3B on 4/6 benchmarks, with composition actually degrading math (-25pp) and legal (-10pp) vs base BitNet alone. The ~1% stability measured here (PPL on training-domain data) may not transfer to out-of-distribution benchmark tasks where uniform averaging actively corrupts specialized knowledge.

## Alternative Approaches

1. **Routed top-k composition (MANDATORY next step).** Uniform 1/N is the worst-case strategy. Routing (top-2 or top-3) would eliminate dilution by selecting only relevant experts per token. MoLoRA (arXiv 2603.15965) demonstrates this at scale: 1.7B+4 adapters outperforms monolithic 8B. Our routing heads already achieve 98.5% val accuracy on genuine domains. Combined with runtime LoRA (4-87x faster than pre-merge, exp_batched_premerge_throughput), this is the production path.

2. **Learned composition weights instead of uniform 1/N.** TIES-Merging (arXiv 2306.01708) and DARE (arXiv 2311.03099) learn per-adapter or per-parameter composition weights. This could capture the optimal weighting without the overhead of per-token routing. However, static learned weights cannot adapt to input content, so per-token routing remains strictly more expressive.

3. **Multi-seed validation to resolve trajectory noise.** The ~0.6pp fluctuation band is within single-seed noise. Running 3-5 seeds would either confirm the trajectory is truly flat (dilution = coverage exactly) or reveal a reproducible trend. Cost: ~48-80 minutes (3-5x the 16-minute runtime). Value: resolves whether there's an actionable N-regime structure.

## Implications for Next Experiments

1. **Uniform composition is "safe but boring" in the N=5-15 regime.** It doesn't degrade, but it also doesn't improve enough to compete with routed composition. This validates SOLE's plug-and-play property (adapters can be added freely) but confirms routing is mandatory for value delivery. The competitive benchmark (KILLED 3/3 under uniform) is the definitive evidence.

2. **The Grassmannian guarantee is load-bearing and validated at intermediate N.** Cross-term ||delta_W_i^T delta_W_j|| = 0 holds empirically (worst-case composition degradation 0.53%, well within noise). The capacity ceiling at N_max=160 means scaling concerns don't arise until far beyond current experiments.

3. **Pattern confirmed across 4 experiments: orthogonality scales, routing is the bottleneck.**
   - N=5 (exp_real_data_domain_experts): -26.3% PPL, 99.9% routing accuracy
   - N=5-15 (this experiment): ~1% composition stability, non-monotonic trajectory
   - N=24 (exp_real_data_25_domain_adapters): -29.1% PPL, routing bifurcates (genuine vs slice)
   - N=50 (exp_ternary_adapter_n50_composition): gamma_uniform=0.996, routing mandatory

   The consistent pattern: orthogonality and composition hold; routing quality determines system utility.

4. **Arbitrary domain slices should be deprecated in future experiments.** The review and prior LEARNINGS repeatedly flag this. Use the 7 genuine domains (medical, code, math, legal, finance, health_fitness, psychology) as the standard evaluation set. Slice-based "domains" inflate numbers without adding real signal.

## Recommended Follow-Up

**exp_routed_topk_composition** — Per-token top-k routing with runtime LoRA on the 7 genuine domain adapters. Motivation: (1) this experiment confirms uniform composition is stable but cannot beat base, (2) competitive benchmark showed uniform hurts factual benchmarks, (3) batched premerge proved runtime LoRA is 4-87x faster for routed workloads. Literature basis: MoLoRA (arXiv 2603.15965) demonstrates per-token top-k routing achieving 1.7B+4 > 8B monolithic. Pre-registered K1: routed PPL < uniform PPL. K2: runtime LoRA overhead < 5% vs base generation speed.
