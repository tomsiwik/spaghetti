# Learnings: exp_adapter_pruning_lifecycle

## Core Finding

Adapter pools have natural redundancy: removing the 5 worst-oracle-PPL adapters
causes only +0.43% mean quality impact on remaining 19 domains (max +1.36% on any
single domain), validated with actual model composition under uniform 1/N scaling.
K1 PASS (trivially — threshold unfalsifiable at N=24). K2 PASS (Jaccard ≤ 0.25).

**Critical caveat from adversarial review:** The LOO approximation (average of
remaining oracle PPLs) is algebraically equivalent to ranking by oracle PPL.
It does NOT measure cross-adapter interference effects. True LOO would require
N-1 composition evaluation per adapter. The validation step (actual composition)
confirms the result holds, but the LOO framing oversells what the metric measures.

## Critical Discoveries

### 1. Adapters have a 6.2x quality spread

Oracle PPLs range from 2.38 (math) to 14.66 (legal). The bottom 5 (legal, finance,
psychology, creative_writing, philosophy) have oracle PPLs 2.5-6x worse than the
median. These domains are intrinsically harder for the BitNet-2B-4T base model,
not badly trained — their base PPLs are also the highest.

**Implication:** Quality variation is dominated by base model capability per domain,
not adapter training quality. Improving weak domains requires better training data
or larger base, not better adapter training.

### 2. Pruning strategies capture genuinely different signals

| Metric | What it measures | Removes |
|--------|-----------------|---------|
| LOO delta | End-to-end quality impact | Worst-performing adapters (high PPL) |
| Routing frequency | Query demand | Least-demanded (small/niche domains) |
| Delta magnitude | Perturbation size | Smallest-perturbation (well-regularized) |
| Cross-similarity | Redundancy | Most-similar B-matrix pairs |

LOO and delta magnitude have Jaccard = 0.00 — completely disjoint removal sets.
This is because small-perturbation adapters (low ||B||_F) can be high-quality
(linguistics: ||B||_F=26.2, oracle PPL=3.67), while large-perturbation adapters
(health_fitness: ||B||_F=34.0, oracle PPL=7.83) can be mediocre.

**Key insight:** ||B||_F is not a quality signal. It measures how much the adapter
changes the base model, not how much it improves it.

### 3. Grassmannian decorrelation holds for B-matrices

Mean |cos(B_i, B_j)| = 0.0238 across all 276 pairs. Max = 0.0893 (creative_writing
<-> sports). This confirms that the Grassmannian skeleton not only decorrelates
A-matrices but effectively decorrelates the full effective deltas. B-matrices trained
independently on different domains in orthogonal A-subspaces remain near-orthogonal.

**Consequence for cross-similarity pruning:** This metric adds minimal signal when
the Grassmannian guarantee holds — all pairs are already decorrelated. It becomes
useful only if decorrelation breaks down (e.g., poorly converged AP).

### 4. Same-domain comparison is the correct S1 metric

The naive metric (average oracle PPL of remaining domains vs full pool) is wrong
because it compares different domain sets. Removing 5 high-PPL domains trivially
"improves" the average. The correct metric: for the 19 remaining domains, compare
their composed PPL in the pruned pool vs the full pool.

Result: full pool avg (19 domains) = 5.052, pruned pool avg (same 19) = 5.076.
Delta: +0.43%. The remaining domains are essentially unaffected by removing 5 adapters.

### 5. Composed PPL is ~10% worse than oracle

Full pool avg oracle PPL: 6.29. Full pool avg composed PPL: 6.90.
Delta: +9.7%. This is the composition overhead — uniform 1/N scaling dilutes
each adapter's contribution. With routed (not uniform) composition, this gap
approaches 0% per exp_softmax_router_scaling.

## Design Implications

### For Evolve Track

1. **Oracle PPL ranking is the simplest effective pruning metric** — identifies
   safely removable adapters by sorting worst-to-best. Cost: 24 model loads for
   oracle PPLs (~6 min), then trivial sort. True LOO (N-1 composition per adapter)
   would measure interference effects but is 24x more expensive.

2. **Routing frequency is the cheap proxy** — single forward pass through router.
   Jaccard = 0.25 overlap with LOO suggests it's a reasonable first filter,
   but misses quality-weak adapters that are still frequently routed.

3. **Recommended lifecycle protocol:**
   - Train adapter → Evaluate oracle PPL → If worse than base, reject
   - If oracle PPL > 2x median → flag for retraining or removal
   - Periodically run LOO on full pool → prune bottom-K if delta < 2%
   - Monitor routing frequency → if never selected, candidate for removal

### For Production

Pruning 5 adapters saves ~226 MB (5 * 45.2 MB). Reduces routing classes from 24
to 19 (21% fewer softmax outputs). Eliminates coverage for 5 domains but those
domains had 2.5-6x worse PPL anyway — they were getting poor service even with
dedicated adapters.

### Uniform vs Routed Generalizability

This experiment validates pruning under uniform 1/N composition only. Under routed
composition (deployment scenario), removing an adapter means its domain's tokens go
to the nearest neighbor. The quality impact depends on inter-domain transfer quality,
which is NOT measured here. exp_softmax_router_scaling showed within-cluster misrouting
is quality-benign, which suggests routed pruning should also be safe, but this is
hypothesized, not proven.

### Per-Domain Degradation Bounds

Same-domain deltas (pruned vs full, 19 remaining domains):
- Max degradation: +1.36% (history)
- Max improvement: -1.49% (engineering)
- All 19 domains within ±1.5%
- No domain exceeds 2% individually

## Contradicting Evidence

- **exp_more_adapters_is_better:** Oracle gamma improves monotonically with N
  (0.668 → 0.625 at N=24). This suggests more adapters = better quality, which
  appears to contradict pruning. Resolution: oracle gamma measures the BEST
  possible routing (adapter matched to its own domain). Pruning removes adapters
  whose domains are too hard for the base model — oracle quality on those domains
  was poor regardless. The remaining 19 domains maintain their quality.

- **Unchosen Experts (arXiv 2402.05858):** Warns that rarely-chosen MoE experts
  still matter for tail queries. In our case, the routing frequency strategy
  (which removes rarely-chosen adapters) overlaps only 25% with LOO. Some
  rarely-routed adapters (science: freq=0.6%) have decent oracle PPL (7.24) —
  routing frequency alone misses this distinction.
