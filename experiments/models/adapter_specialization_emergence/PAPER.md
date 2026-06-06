# Adapter Specialization Emergence: Research Digest

## Hypothesis

When training N adapters on identical mixed-domain data with different Grassmannian
A-matrices, the orthogonality constraint forces each adapter to capture different
directions in weight space, leading to emergent domain specialization.

**Status: KILLED (K1 + K2)**

## What This Experiment Tested

We trained 10 adapters on the same mixed-domain training data (50 samples from each
of 10 domains = 500 total), using different orthogonal Grassmannian A matrices
(frozen, AP-packed) for each adapter. The only difference between adapters was
their frozen A matrix. We then evaluated each adapter on each domain's validation
data to build a 10x10 PPL matrix and measured whether adapters developed different
domain preferences.

## Key References

- FlyLoRA (arxiv 2510.08396): Frozen random A as implicit feature selector via
  JL lemma. Motivated the hypothesis that different A projections would capture
  different aspects of the data.
- MoE Self-Specialization (Shazeer et al., 2017): Experts self-specialize in
  mixed training. However, MoE uses gating gradients to reinforce specialization;
  our setup has no gating.

## Empirical Results

### K1: Specialization (silhouette >= 0.2) -- FAIL

| Metric | Value | Threshold |
|--------|-------|-----------|
| Silhouette score | 0.0000 | >= 0.2 |
| Unique best domains (of 10) | 1 | (hoped for >= 5) |
| Domain preference entropy | 0.000 | (max: 3.322) |
| Per-domain PPL CV across adapters | 0.1-0.4% | (near zero) |

**All 10 adapters have identical domain preferences.** Every adapter's best
domain is "music" (PPL ~2.46). The PPL profiles across domains are nearly
identical, with coefficient of variation 0.1-0.4% between adapters on each domain.

### K2: Quality comparison with domain-trained -- FAIL

| Domain | Base PPL | Domain-trained PPL | Best Mixed PPL | Gap |
|--------|----------|-------------------|----------------|-----|
| code | 4.98 | 3.14 | 3.32 | +5.5% |
| math | 3.84 | 2.38 | 2.81 | +18.0% |
| medical | 6.50 | 3.46 | 4.13 | +19.5% |
| legal | 21.63 | 14.66 | 16.03 | +9.4% |
| cooking | 3.21 | 2.55 | 2.66 | +4.5% |
| psychology | 17.45 | 12.20 | 13.87 | +13.7% |
| cybersecurity | 3.83 | 3.09 | 3.25 | +5.1% |
| philosophy | 16.39 | 9.97 | 10.38 | +4.0% |
| economics | 16.69 | 8.93 | 9.40 | +5.3% |
| music | 3.57 | 2.34 | 2.46 | +5.1% |

Mixed-trained adapters are worse than domain-trained on **all 10 domains**.
Mean gap: +9.0%. The gap is largest on domains with strongest domain signals
(math +18%, medical +19.5%) and smallest on easier domains (cooking +4.5%,
philosophy +4.0%).

### Positive finding: Mixed training still helps

All mixed adapters improve over the base model on every domain (mean improvement
~30%). The mixed training does teach general language modeling improvement, but
this improvement is domain-agnostic and identical across all adapters.

### Training convergence

All 10 adapters converge to essentially identical loss trajectories:
- First 50 steps avg: 1.91-1.92 (range: 0.01)
- Last 50 steps avg: 1.65-1.66 (range: 0.01)

The A-matrix projection does NOT create meaningfully different optimization
landscapes for B at this scale.

## Root Cause Analysis

### Why specialization did NOT emerge

1. **No gating gradient.** In MoE, experts specialize because the gating function
   amplifies initial specialization via gradient routing. Here, every adapter receives
   identical gradient signal from every sample. The A projection only changes which
   r-dimensional subspace the B gradient operates in -- it does not preferentially
   route certain samples to certain adapters.

2. **Identical data ordering.** All adapters see the same 500 samples in the same
   order (shuffled identically). With the same loss landscape and same data sequence,
   the only difference is a rotation of the B gradient by A^T. Since B is initialized
   at zero and the loss landscape is smooth, all adapters learn equivalent functions
   (rotated versions of the same B*A^T product).

3. **Orthogonality prevents specialization, not enables it.** The Grassmannian
   guarantee that A_i^T A_j = 0 means the adapters operate in perfectly independent
   subspaces. This is exactly what makes composition safe (no interference), but it
   also means no adapter "sees" what others are doing -- there is no competition
   pressure to differentiate.

4. **Short training + low capacity.** At 200 steps on 500 mixed samples, each adapter
   barely converges. There is insufficient training signal to develop fine-grained
   domain preference. The B matrices learn a generic "instruction following" direction
   that works across all domains equally.

## Implications for SOLE Architecture

This is an important negative result:

1. **Explicit domain training is necessary.** Orthogonal A matrices alone do NOT
   induce specialization. The Grassmannian skeleton serves as an interference
   prevention mechanism, not a specialization mechanism. This confirms the existing
   SOLE architecture: train each adapter on domain-specific data, use A for
   composition safety.

2. **Routing remains necessary.** Since mixed-trained adapters are interchangeable,
   a router cannot distinguish between them. Domain-specific training produces the
   domain signal that routing exploits.

3. **"Train once, specialize automatically" is dead.** The dream of training N
   generic adapters and having them self-specialize via projection geometry alone
   does not work at this scale.

## Limitations

1. **Short training (200 steps).** Longer training might allow divergence, though
   the identical loss trajectories suggest this is unlikely.
2. **Equal data ordering.** Different random shuffles per adapter could create weak
   specialization, but this would be a data-ordering effect, not a projection-geometry
   effect.
3. **Rank 16 on d=2560.** Higher rank or lower d might change the dynamics.
4. **10 domains only.** Testing with more domains would not change the conclusion
   since all 10 adapters are already identical.

## What Would Kill This (Retrospective)

The hypothesis was killed by K1: silhouette = 0.0, threshold was 0.2. The result
is maximally negative -- not borderline but zero specialization. The mechanism
(FlyLoRA-style implicit routing) requires either gating gradients or fundamentally
different data presentations per adapter to work.
