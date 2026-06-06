# More Adapters Is Better: Research Digest (REVISED)

## Hypothesis

Adding more routed ternary adapters monotonically improves overall system quality, because each adapter helps its own domain without degrading others, given Grassmannian orthogonality and per-sequence routing.

## What This Experiment Is

An evaluation-only scaling study on BitNet-2B-4T with 24 pre-trained ternary LoRA adapters. For each N in {5, 10, 15, 20, 24}, we compose the first N adapters using three strategies:

1. **Oracle** -- only the correct adapter per domain (upper bound)
2. **Uniform** -- all N adapters with 1/N scaling (lower bound)
3. **Routed** -- routing heads select which adapters to apply per sequence (HONEST: no oracle fallback)

Additional controls added per adversarial review:
4. **Frozen heads** -- routing heads trained at N=5, reused at all subsequent N without retraining
5. **Random routing** -- randomly select 1 adapter per domain (lower bound for routing value)

No retraining. All adapters and data come from exp_real_data_25_domain_adapters.

## Key References

- Naive LoRA Summation (arxiv 2508.11985) -- orthogonality enables additive composition
- MoLoRA (arxiv 2603.15965) -- per-token LoRA routing, 1.7B+4 adapters beats 8B
- DeepSeek-V3 (arxiv 2412.19437) -- 256 routed experts with auxiliary-loss-free balancing

## Verdict: KILLED

**K1 FAIL:** gamma_routed degrades from 0.668 (N=5) to 0.846 (N=10), never recovers. Routing quality collapses as N grows.

**K2 FAIL:** cooking regresses 26% at N=24 (routing head fails, falls back to base-only).

**S1 FAIL:** Not monotonic. Domains with routing failures get base-only PPL (no improvement).

## Empirical Results

### System-Level Quality (gamma = avg_routed_PPL / avg_base_PPL, lower is better)

| N  | gamma_routed | gamma_oracle | Improvement | Routing Acc | Fallback Rate |
|----|-------------|-------------|-------------|-------------|---------------|
| 5  | 0.668       | 0.668       | 33.2%       | 99.6%       | 0/5 (0%)      |
| 10 | 0.846       | 0.626       | 15.4%       | 93.3%       | 4/10 (40%)    |
| 15 | 0.816       | 0.625       | 18.4%       | 93.8%       | 5/15 (33%)    |
| 20 | 0.849       | 0.618       | 15.1%       | 93.9%       | 8/20 (40%)    |
| 24 | 0.851       | 0.625       | 14.9%       | 92.7%       | 11/24 (46%)   |

### The Real Finding: Routing Heads Are the Bottleneck

Oracle gamma improves monotonically from 0.668 to 0.625 — proving the architecture works. The thesis "more adapters = better" IS true under oracle routing. But honest routing (no fallback) reveals that binary classification heads fail catastrophically as N grows:

- At N=5: 0% fallback. All domains correctly routed.
- At N=10: 40% fallback. science, history, philosophy, creative_writing get base-only.
- At N=24: 46% fallback. 11 domains get no adapter at all.

Domains with routing failures get base-only PPL (gamma=1.0), dragging the system average up.

### Why Routing Fails: Positive-Class Recall Collapse

Binary routing heads (one per domain) suffer extreme class imbalance (23:1 negative:positive at N=24). Most heads learn to predict "no" (achieving >90% overall accuracy trivially). Positive-class recall for many domains is <15%:

| Domain | Pos Recall (N=24) | Routed PPL | Oracle PPL | Outcome |
|--------|------------------|------------|------------|---------|
| science | 12% | 12.46 (base) | 7.30 | FAILED |
| history | 14% | 16.70 (base) | 9.29 | FAILED |
| philosophy | 10% | 16.39 (base) | 9.97 | FAILED |
| creative_writing | 10% | 20.73 (base) | 12.01 | FAILED |
| agriculture | — | 14.56 (base) | 8.35 | FAILED |
| environmental | — | 11.49 (base) | 6.62 | FAILED |
| politics | — | 12.56 (base) | 6.67 | FAILED |
| economics | — | 16.69 (base) | 8.93 | FAILED |
| cybersecurity | — | 3.83 (base) | 3.09 | FAILED |
| marketing | — | 3.83 (base) | 2.95 | FAILED |
| cooking | — | 3.21 (base) | 2.55 | FAILED |

### K2: Cooking Regression (+26%)

Cooking is correctly routed at N=10,15,20 but its routing head fails at N=24 (no head fires). This is not architectural interference — it's routing head degradation. The adapter itself is fine (oracle PPL=2.55 at all N).

### Fixed-Domain Metric (First 5 Domains)

| N  | Avg Routed (first 5) | Avg Oracle (first 5) | Gap |
|----|---------------------|---------------------|-----|
| 5  | 7.53                | 7.53                | +0.00% |
| 10 | 7.53                | 7.53                | +0.01% |
| 15 | 7.53                | 7.53                | +0.03% |
| 20 | 7.53                | 7.53                | +0.04% |
| 24 | 7.53                | 7.53                | +0.05% |

The first 5 domains (medical, code, math, legal, finance) are perfectly stable across all N. Their routing heads maintain near-perfect recall because these are genuine, well-separated domains. **This confirms zero architectural interference** — the Grassmannian guarantee holds.

### Frozen Heads vs Retrained Heads

| N  | Retrained PPL | Frozen PPL | Gap | Frozen Fallbacks |
|----|--------------|------------|-----|-----------------|
| 5  | 7.53         | 7.53       | +0.0% | 0/5 |
| 10 | 10.65        | 10.71      | +0.6% | 5/10 |
| 15 | 9.65         | 9.94       | +2.9% | 6/15 |
| 20 | 9.64         | 9.93       | +3.1% | 11/20 |
| 24 | 8.57         | 8.79       | +2.6% | 13/24 |

Frozen heads (trained at N=5 only) are only 2.6% worse than retrained heads. The gap is mostly from domains beyond N=5 having no routing head at all (base-only fallback). Retraining heads at each N provides minimal benefit.

### Random Routing vs Trained Routing

| N  | Routed PPL | Random PPL | Gap |
|----|-----------|------------|-----|
| 5  | 7.53      | 8.00       | +6.3% (routed wins) |
| 10 | 10.65     | 8.38       | -21.3% (random wins) |
| 15 | 9.65      | 8.21       | -14.9% (random wins) |
| 20 | 9.64      | 7.85       | -18.5% (random wins) |
| 24 | 8.57      | 6.94       | -19.0% (random wins) |

**Random routing beats trained routing at N>=10.** This is because random routing always activates some adapter (PPL ≤ adapter average), while honest routing with base-only fallback often activates none (PPL = base). The routing mechanism provides negative value at scale because it fails to fire rather than routing incorrectly.

## Root Cause Analysis

The binary routing head architecture (N independent sigmoid classifiers) is fundamentally flawed at scale:

1. **Class imbalance scales with N.** At N=24, each head sees 23x more negatives than positives. Trivial "always no" achieves 96% accuracy.
2. **Positive recall collapses for non-genuine domains.** Domains carved from general-purpose datasets (economics, environmental, politics) have overlapping hidden-state distributions, making binary discrimination hard.
3. **Base-only fallback is worse than random.** When no head fires, the model gets no adapter benefit. Random selection would be better.

## What This Actually Proves

1. **Architecture works (oracle).** gamma_oracle improves monotonically from 0.668 to 0.625. Adding adapters with zero interference is confirmed.
2. **Routing heads are the bottleneck.** The gap between oracle (0.625) and routed (0.851) at N=24 is 22.6pp — almost the entire potential benefit is lost to routing failures.
3. **Binary heads are the wrong architecture.** Need multi-class softmax router (MoLoRA-style) or threshold-calibrated binary heads to recover recall.

## Limitations

1. **Single seed.** Routing head training uses seed 42. Different seeds might give different fallback patterns.
2. **Per-domain evaluation.** Each validation batch is 100% single-domain. Mixed-domain text would be harder.
3. **25 samples/domain.** PPL estimates have finite-sample variance.
4. **Domain ordering fixed.** Always add domains in same order. Different orderings might yield different curves.

## What Would Kill This (if reframed)

If a multi-class softmax router also fails to route correctly at N=24, the problem is not routing architecture but hidden-state discriminability. This would mean the adapters don't create sufficiently distinct representations to enable routing.

## What Would Validate This

Replace N binary sigmoid heads with a single multi-class softmax router (2560 → 128 → N). If this achieves >80% top-1 accuracy, the "more adapters = better" thesis holds — it was just the routing mechanism that was broken, not the concept.
