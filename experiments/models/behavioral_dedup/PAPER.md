# Activation-Based Behavioral Deduplication: Research Digest

## Hypothesis

Co-activation profiling finds functionally redundant capsules that
weight-cosine similarity misses, because ReLU creates many-to-one
mappings where different weight vectors produce identical activation
patterns on the actual data distribution.

**Falsifiable**: If behavioral analysis finds less than 5% additional
functional redundancy (measured as unique capsules in behaviorally-
redundant pairs not found by weight-cosine) above the weight-cosine
baseline, this approach adds no value.

**Result: PASS at permissive thresholds, threshold-sensitive.**
Behavioral analysis finds 19.3% of capsules in behaviorally-redundant
pairs at tau_rho=0.3, dropping to 10.8% at tau_rho=0.5 and 1.4% at
tau_rho=0.7. The finding is concentrated in Layer 0, consistent with
shared low-level feature detectors across domains. Merging produces
negligible quality change.

---

## What This Experiment Tests

Whether activation-based behavioral similarity metrics discover
functional redundancy invisible to weight-space cosine similarity.

Two genuinely behavioral metrics are computed:
1. **Co-activation Jaccard**: J(i,j) = |fire_i AND fire_j| / |fire_i OR fire_j|
2. **Output correlation**: Pearson correlation of capsule output contributions

A third metric, **conditioned output cosine**, is computed but is
actually a weight-space metric (b-vector cosine similarity) that does
not use activation data. It is included for completeness but is not
a novel behavioral metric.

Protocol:
1. Pretrain base model on ALL data (shared attention + embeddings)
2. Fine-tune only MLP weights per domain (attention frozen)
3. Compose by concatenating A and B weight matrices from both domains
4. Profile behavioral similarity on evaluation data (20 batches x 32)
5. Compare redundancy found by behavioral vs weight-cosine methods
6. Test merging quality for behaviorally-identified redundant pairs

---

## Lineage in the Arena

```
gpt -> moe -> capsule_moe -> relu_router -> capsule_dedup -> behavioral_dedup
                               (composition    (weight-cosine    (activation-based
                                by concat)      dedup, KILLED)    dedup)
```

---

## Key References

**Capsule Deduplication (Exp 8, this project)**: Found only 1.9%
redundancy at cos>0.95 with different seeds. KILLED because shared
knowledge is distributed, not concentrated in similar-weight capsules.
Note: Exp 8 used a different seed range, so its 1.9% figure is not
directly comparable to results here.

**BuddyMoE (2024)**: Co-activation profiling for expert merging.
Uses behavioral similarity rather than weight similarity. Directly
inspired the Jaccard-based approach.

**Sub-MoE (2024)**: Adaptive expert clustering with SVD subspace
merging. Demonstrates activation-based clustering at scale.

**ReDo (Klein et al. 2024)**: Activation-based dead neuron profiling
using hooks. The profiling methodology adapted here.

---

## Empirical Results

### 3-Seed Aggregate Quality (seeds 42, 123, 7)

| Method | Avg Val Loss | Std | vs Joint | vs Concat |
|--------|-------------|-----|----------|-----------|
| joint (baseline) | 0.5263 | 0.0052 | -- | -4.7% |
| concat_zero_shot | 0.5524 | 0.0023 | +4.9% | -- |
| weight_avg | 0.5317 | 0.0017 | +1.0% | -3.7% |
| weight_cos_dedup (tau=0.95) | 0.5524 | 0.0023 | +4.9% | +0.0% |
| **behavioral_jt0.5** | **0.5541** | **0.0071** | **+5.3%** | **+0.3%** |
| **behavioral_jt0.7** | **0.5543** | **0.0069** | **+5.3%** | **+0.3%** |
| **behavioral_jt0.9** | **0.5524** | **0.0008** | **+5.0%** | **+0.0%** |

### Behavioral vs Weight-Cosine Redundancy (3-seed mean, tau_rho=0.3)

| Metric | J>0.5 | J>0.7 | J>0.9 | Weight cos>0.95 |
|--------|-------|-------|-------|-----------------|
| Redundant pairs found | 2036 | 1623 | 668 | 0.3 |
| Behavioral-only pairs | 2036 | 1623 | 667 | -- |
| Unique capsules involved (%) | 22.3% | 19.3% | 13.3% | 0.0% |

**Important context on the weight-cosine comparison.** Weight-cosine
found near-zero pairs (0.3 mean, i.e. 0 in 2 seeds, 1 in 1 seed)
across these 3 seeds. This means the comparison "behavioral 19.3% vs
weight-cosine 0%" is trivially won -- the experiment demonstrates that
behavioral analysis finds *something* rather than demonstrating it finds
*more* than a functioning weight-cosine baseline. Exp 8 found 1.9%
weight-cosine redundancy at cos>0.95 with a different seed range; that
figure is cross-seed and not directly comparable to results here.

### Output Correlation Threshold Sweep (J>0.7, 3-seed)

The default tau_rho=0.3 is permissive (correlation of 0.3 on a [-1,1]
scale). This sweep tests whether the finding survives stricter thresholds.

| tau_rho | Behavioral Pairs | Behavioral-Only Capsule % | Per-Seed Values | Result |
|---------|-----------------|---------------------------|-----------------|--------|
| 0.3 | 1623 | 19.3% +/- 3.1% | 17.9%, 17.2%, 22.9% | **PASS** |
| 0.5 | 236 | 10.8% +/- 3.8% | 8.7%, 8.5%, 15.2% | **PASS** |
| 0.7 | 8 | 1.4% +/- 0.5% | 1.2%, 1.0%, 2.0% | **KILL** |

At tau_rho=0.5 (moderate correlation), 10.8% of capsules remain in
behavioral-only redundant pairs -- the finding survives, though at
roughly half the magnitude. At tau_rho=0.7 (strong correlation), only
1.4% remain, below the 5% kill threshold. The result is therefore
threshold-sensitive: substantial co-firing overlap exists, but truly
correlated output contributions are rarer.

### Per-Seed Kill Criterion Values (J>0.7, tau_rho=0.3)

| Seed | Behavioral-Only Capsule % | Total Alive | Weight-Cos Pairs |
|------|---------------------------|-------------|------------------|
| 42 | 17.9% | 398 | 0 |
| 123 | 17.2% | 394 | 0 |
| 7 | 22.9% | 463 | 1 |
| **Mean** | **19.3% +/- 3.1%** | **418** | **0.3** |

### Merge Statistics (3-seed mean)

| Method | Capsules Merged | % Merged |
|--------|----------------|----------|
| Behavioral J>0.5 | 99 | 9.6% |
| Behavioral J>0.7 | 86 | 8.4% |
| Behavioral J>0.9 | 56 | 5.5% |
| Weight-cos>0.95 | 35 | 3.4% |

### Cross-Pool Jaccard Distribution by Layer (3-seed mean)

| Layer | Mean J | P50 | P90 | P95 | Max | Pairs J>0.7 | Alive Pairs |
|-------|--------|-----|-----|-----|-----|-------------|-------------|
| 0 | 0.527 | 0.573 | 0.908 | 0.940 | 0.998 | 6,481 | 16,341 |
| 1 | 0.052 | 0.001 | 0.142 | 0.329 | 0.930 | 13 | 682 |
| 2 | 0.025 | 0.000 | 0.064 | 0.137 | 0.716 | 2 | 641 |
| 3 | 0.031 | 0.001 | 0.081 | 0.153 | 0.998 | 7 | 1,025 |

---

## Kill Threshold Analysis

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Behavioral-only capsule % (J>0.7, tau_rho=0.3) | 19.3% +/- 3.1% | >5% | **PASS** |
| Behavioral-only capsule % (J>0.7, tau_rho=0.5) | 10.8% +/- 3.8% | >5% | **PASS** |
| Behavioral-only capsule % (J>0.7, tau_rho=0.7) | 1.4% +/- 0.5% | >5% | **KILL** |
| Behavioral merging quality vs concat | +0.3% | <5% degradation | **PASS** |
| Quality vs weight averaging | +5.3% worse | Must beat | **FAIL** |

**The kill criterion (>5% behavioral-only redundancy) is met at
tau_rho <= 0.5 but fails at tau_rho=0.7.** This means behavioral
analysis finds substantial co-activation overlap (many capsule pairs
fire on the same inputs), and at moderate output correlation thresholds
this constitutes meaningful functional redundancy. However, at strict
output correlation thresholds, most of this co-activation does not
translate to truly redundant outputs.

This is a nuanced PASS:

1. **Nearly all redundancy is in Layer 0, consistent with shared
   low-level feature detectors.** Layers 1-3 contribute only ~22
   behavioral pairs combined (vs ~1601 in Layer 0). The finding is
   about one specific phenomenon -- Layer 0 capsules from both domains
   learn generic character detectors that co-fire because both domains
   share the same 26-letter alphabet. This is consistent with the
   well-known feature hierarchy principle (Yosinski et al. 2014):
   early layers learn generic features, deeper layers specialize.

2. **Quality improvement from merging is negligible.** Behavioral
   dedup achieves +0.3% vs unmerged concatenation. Weight averaging
   achieves -3.7%. Behavioral dedup is not a viable compression
   strategy -- it finds redundancy but merging barely helps.

3. **The 0 weight-cosine pairs makes this a weak comparison.**
   Weight-cosine found near-zero pairs across these seeds, so the
   experiment tests whether behavioral analysis finds *any* redundancy,
   not whether it finds *more than* weight-cosine. The key contribution
   is demonstrating that behavioral analysis detects a real structural
   phenomenon (Layer 0 co-activation) rather than outperforming a
   functioning baseline.

---

## The Real Finding: Layer 0 Co-Activation from Shared Input Statistics

Behavioral analysis finds 19.3% of capsules in redundant pairs at
tau_rho=0.3, concentrated in Layer 0. This is consistent with shared
low-level feature detectors across domains that co-fire because both
domains process the same character alphabet.

### Layer-Dependent Redundancy Structure

```
Layer 0:  J_mean = 0.527  (massive co-activation, shared character detectors)
Layer 1:  J_mean = 0.052  (10x drop from Layer 0)
Layer 2:  J_mean = 0.025
Layer 3:  J_mean = 0.031  (17x drop from Layer 0)
```

This reveals that:

1. **Composition redundancy is front-loaded.** The first layer of
   the MLP stack has enormous functional overlap between domains.
   Deeper layers are already well-specialized.

2. **Layer 0 compression is the opportunity.** If 50%+ of Layer 0
   capsule pairs are functionally identical across domains, a shared
   Layer 0 pool could reduce parameters with minimal quality impact.
   This follows directly from the feature hierarchy principle and
   does not require behavioral profiling to motivate -- but the
   experiment quantifies the effect.

3. **Behavioral analysis reveals layer structure invisible to
   weight-cosine.** Weight-cosine found near-zero redundancy
   uniformly across layers. Behavioral analysis reveals the
   concentrated Layer 0 pattern.

### Why Behavioral Diverges from Weight-Cosine

Weight-cosine measures angular similarity in R^d (64 dimensions).
Behavioral Jaccard measures functional overlap on the actual data
manifold. Two capsule detectors a_i and a_j may have:
- Low weight cosine: cos(a_i, a_j) = 0.3 (different directions in R^64)
- High Jaccard: J(i,j) = 0.95 (fire on same data points)

This happens when the directions that distinguish a_i from a_j are
orthogonal to the data manifold. Layer 0 is the extreme case: both
domains share the same character embeddings, so the data manifold
at Layer 0 is nearly identical across pools.

---

## Micro-Scale Limitations

1. **Similar domains**: a-m vs n-z names share character distributions
   heavily. With truly different domains (Python vs JavaScript), Layer 0
   redundancy might be lower (different token distributions) or higher
   (if base embeddings dominate).

2. **Character-level tokenization**: At character level, both domains
   use the same 26-character alphabet. With subword tokenization,
   domain-specific tokens would create less Layer 0 overlap.

3. **Small model**: With d=64 and P=128, the embedding space is
   relatively low-dimensional. At d=4096, behavioral divergence from
   weight-cosine may be more or less pronounced.

4. **Quality impact is negligible**: Behavioral merging produces +0.3%
   quality change. This may mean the redundancy is "harmless" (the
   model already handles it through its composition mechanism) or that
   the merging rule does not capture the right reduction.

5. **Dead capsule interaction**: ~60% of capsules are dead. Behavioral
   analysis operates only on the ~40% alive ones. The 19.3% behavioral
   redundancy is measured against total capsules, not alive capsules.
   Against alive capsules only, the percentage would be higher.

6. **Profiling on validation data**: Behavioral profiling and quality
   evaluation use the same validation data. A held-out profiling set
   would be stronger, though the two measurements capture different
   things (co-activation patterns vs loss).

7. **Threshold sensitivity**: The kill criterion result depends on
   tau_rho. At 0.3 (permissive) it passes clearly; at 0.5 (moderate)
   it still passes; at 0.7 (strict) it fails. The "right" threshold
   depends on what downstream use case the redundancy metric serves.

---

## What Would Kill This

### At Micro Scale (tested)

- **Insufficient behavioral-only redundancy at tau_rho=0.5**: NOT
  triggered. Found 10.8% vs 5% threshold at moderate correlation.
  However, at strict correlation (tau_rho=0.7), this IS triggered
  (1.4% < 5%).

- **All redundancy is dead capsules**: NOT triggered. Layer 0 has
  0-1 dead capsules (out of 256), and this is where all the
  behavioral redundancy lives.

- **Quality degradation from merging**: NOT triggered. Merging
  produces +0.3% quality change (negligible).

### At Macro Scale (untested)

- **Layer 0 redundancy is tokenizer-dependent**: With subword
  tokenization and truly different domains, Layer 0 might not show
  the same massive co-activation overlap.

- **Deeper layers may show more redundancy at scale**: With larger
  P (thousands of capsules per domain), deeper layers may develop
  more overlapping patterns.

- **Merging might matter more at scale**: At micro scale, merging
  86 capsules from 1024 is 8.4%. At macro scale with thousands of
  experts, behavioral merging could be more impactful.

---

## Implications for the Project

1. **Behavioral analysis finds co-activation redundancy concentrated
   in Layer 0, consistent with shared low-level feature detectors
   across domains.** At tau_rho=0.3, 19.3% of capsules are involved;
   at tau_rho=0.5, 10.8% remain. The mechanism works: activation-based
   similarity captures data-dependent functional overlap.

2. **The redundancy is structurally predictable.** It is concentrated
   in Layer 0 where domains share input statistics. This suggests a
   simpler intervention: share Layer 0 capsule pools across domains
   instead of duplicating and deduplicating.

3. **Merging is not the right intervention.** Quality barely changes
   with merging (+0.3%). The redundant capsules are already "free"
   in that the model handles them through its composition mechanism.
   Pruning dead capsules (~60% at zero quality loss) remains the
   dominant compression strategy.

4. **For the contribution protocol**: When composing domains, Layer 0
   capsule pools can likely be shared (not concatenated) since they
   learn identical functions. Deeper layers should be concatenated
   as before, since they show near-zero behavioral redundancy.

5. **Behavioral profiling is a useful diagnostic** even when merging
   is not the goal. It reveals the layer-dependent redundancy structure
   that guides architectural decisions.
