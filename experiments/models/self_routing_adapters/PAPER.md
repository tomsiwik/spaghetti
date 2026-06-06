# Self-Routing Adapters: Research Digest

## Hypothesis

Adapter B-matrices contain implicit routing signal: for each token, computing
similarity between hidden state and B-matrix column space should select the
correct domain expert without a learned router.

**Verdict: ORIGINAL HYPOTHESIS KILLED. B-matrix routing fails (best 23.5% top-2).
Hidden-state centroid routing is a strong closed-form baseline (87.14% top-2)
but is surpassed by a properly trained Gumbel-sigmoid router (90.41% top-2 at 6000 steps).**

## What This Experiment Is

A systematic comparison of 9 routing methods on 49-domain LoRA adapter selection,
using cached hidden states from a BitNet-2B-4T model with Grassmannian skeleton
(rank-16 adapters). Tests whether adapter weight matrices (specifically B-matrices)
can serve as routing keys, eliminating the need for a separate learned router.

## Key Finding 1: B-Matrix Routing Fails (Concentration of Measure)

All adapter-weight-based routing methods perform at or near random chance:

| Method | Top-1 | Top-2 | Top-5 | Params |
|--------|-------|-------|-------|--------|
| B-only quadratic form | 2.0% | 4.7% | 15.5% | 0 |
| SVD projection | 0.8% | 2.7% | 6.5% | 0 |
| Adapter centroid cosine | 3.3% | 5.5% | 11.4% | 0 |
| Random baseline | 2.0% | 4.1% | 10.2% | 0 |

**Why it fails (empirical observation, supported by theory):** With d=2560 and r=16,
each adapter's B-matrix spans a 16-dimensional subspace. The concentration of measure
theorem for random subspaces predicts that E[||P_S u||^2 / ||u||^2] = r/d = 0.625%
with variance O(r/d^2). While this theorem strictly requires random subspaces and
random vectors, the empirical results confirm that trained B-matrices behave
similarly: despite being domain-specific (inter-adapter cosine ~0.03), their
rank-16 subspaces are too small relative to d=2560 to develop domain-discriminative
structure. The theorem provides the correct intuition, but the empirical failure is
the actual evidence -- it is possible in principle for learned subspaces to escape
concentration, but these did not.

The per-domain analysis of the activation-norm method (Phase 3) illustrates the
failure mode: 40 out of 49 domains have 0% top-1 accuracy, with most samples
routing to the same adapter (chemistry) due to slightly larger B-matrix norms.
**Note: this per-domain analysis is for the activation-norm method specifically,
not the centroid method.**

## Key Finding 2: Hidden-State Centroids as Closed-Form Baseline

The pretrained base model's hidden states already cluster by domain. A
nearest-centroid classifier (NCC) using domain-averaged hidden states achieves
strong routing accuracy:

| Method | Top-1 | Top-2 | Top-5 | Storage |
|--------|-------|-------|-------|---------|
| **Centroid cosine (closed-form)** | **79.6%** | **87.1%** | **90.6%** | **500KB** |
| Gumbel-sigmoid (3000 steps) | -- | 86.3% | -- | 659K params |
| **Gumbel-sigmoid (6000 steps)** | **83.7%** | **90.4%** | -- | **659K params** |
| All-layer activation norm | 16.5% | 22.2% | 46.1% | 0 |

### Comparison with trained Gumbel-sigmoid router

The centroid method (87.14% top-2) approximately matches the undertrained 3000-step
Gumbel-sigmoid baseline (86.33% top-2), but this comparison is misleading.
The Gumbel-sigmoid ablation study (micro/models/gumbel_sigmoid_ablation/) shows that
simply training for 6000 steps raises the learned router to **90.41% top-2** and
**83.67% top-1**, surpassing centroid routing on both metrics.

The 0.81pp gap between centroid (87.14%) and the 3000-step baseline (86.33%) is
**not statistically significant**: with 490 validation samples, the standard error
is SE = sqrt(0.87 * 0.13 / 490) = 1.52pp, so the gap falls well within 1 SE.

**Honest framing:** Centroid routing replaces 659K *learned* parameters with 125K
*computed* parameters (49 x 2560 floats = 500KB), trading optimization for a
closed-form solution. It is a strong non-learned baseline and cheap fallback,
not a replacement for properly trained routing. Its value is:
1. **Zero training cost** -- centroids computed from 20 examples per domain
2. **Instant deployment** -- no router training step needed
3. **Initialization** -- centroid routing can serve as warm-start for learned routers
4. **Fallback** -- useful when router training budget is limited

### Statistical context

| Comparison | Gap | SE (n=490) | Significant? |
|-----------|-----|------------|-------------|
| Centroid vs 3000-step Gumbel | +0.81pp | 1.52pp | No (< 1 SE) |
| 6000-step Gumbel vs centroid | +3.27pp | 1.52pp | Marginal (~2 SE) |

## Why Hidden-State Centroids Work

The pretrained base model (BitNet-2B-4T) has already learned domain-discriminative
representations. Different text domains occupy distinct regions of the d=2560
hidden state space. NCC is a well-established technique in this setting
(cf. Prototypical Networks, Snell et al., NeurIPS 2017). The finding that
pretrained LLM hidden states cluster by domain is consistent with standard
zero-shot topic classification results.

NCC requires:
- 20 labeled examples per domain (to compute the centroid)
- One cosine similarity computation per adapter per token
- Storage: N * d floats = 49 * 2560 * 4B = 500KB

## Kill Criteria Assessment

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| K1 (id=249): Implicit routing accuracy >= 50% | top-2 >= 50% | 87.14% (centroid) | **PASS** |

K1 PASSES on the centroid method, but the winning method is NOT adapter-weight-based.
The original hypothesis (B-matrix self-routing) is KILLED:
- B-matrix methods: best 23.5% top-2 (FAIL)
- Centroid method: 87.14% top-2 (PASS, but different mechanism)

## Timing

| Method | Latency/token | Notes |
|--------|--------------|-------|
| Centroid cosine | ~0.01ms | Single matmul of h against N centroids |
| Gumbel-sigmoid | ~0.05ms | Trained router forward pass |
| Activation norm (5 layers) | 4.26ms | 49 adapters x 5 layers |

## Key References

- **Prototypical Networks (Snell et al., NeurIPS 2017)** -- nearest-centroid
  classification in pretrained feature space. Our centroid routing is NCC
  applied to adapter selection.
- **Autonomy-of-Experts (AoE)** -- activation-norm-based expert self-evaluation.
  Our activation norm results (13-16% top-1) are consistent with AoE's finding
  that this signal is weak at small scale.
- **Arrow (Ostapenko et al.)** -- SVD-based spectral routing. Our SVD projection
  result (0.8% top-1, essentially random) confirms known underperformance.
- **FlyLoRA** -- frozen random A as implicit router. Grassmannian A matrices
  are frozen by design.
- **Gumbel-sigmoid ablation (exp_gumbel_sigmoid_ablation)** -- 6000-step
  training reaches 90.41% top-2, showing learned routers outperform NCC
  when properly trained.

## Experiment DB

- Experiment ID: `exp_self_routing_adapters`
- Kill criterion ID: 249
- Status: SUPPORTED (with caveats per this revision)
- Depends on: Gumbel-sigmoid ablation for correct baseline comparison

## Limitations

1. **Centroid routing loses to properly trained Gumbel-sigmoid.** The 6000-step
   router (90.41% top-2) surpasses centroid routing (87.14%) by 3.27pp.
2. **Hidden states are from a single position (last hidden layer).** Different
   layers may provide different routing signals.
3. **Only 20 train + 10 val samples per domain.** More data may improve centroid
   quality.
4. **Clean domain labels.** Real-world text is often multi-domain, requiring
   soft/threshold routing.
5. **Centroid routing requires knowing domains a priori.** New domains need
   new centroid computation (but this is trivially cheap).
6. **Per-token routing untested.** Centroid routing on per-token hidden states
   (rather than mean-pooled sequences) may not cluster as cleanly.
7. **Statistical power is limited.** With 490 samples, only gaps >3pp are
   reliably detectable.

## What Would Kill This

- **Properly trained routers** already beat centroid routing (6000-step Gumbel
  at 90.41%). The question is cost-benefit, not superiority.
- **Multi-domain text** where mean-pooled hidden states don't cluster cleanly.
- **Scale to N=1000+** where centroid space becomes crowded.
- **Per-token routing** where individual tokens don't cluster as well as
  sequence-level means.

## Conclusions

1. **B-matrix self-routing is killed.** Adapter weight matrices do not contain
   useful routing signal at r/d = 16/2560. This is empirically confirmed and
   consistent with concentration of measure theory for low-rank subspaces.

2. **Centroid routing is a strong closed-form baseline.** At 87.14% top-2, it
   matches undertrained routers and provides a zero-training-cost fallback.
   However, it is surpassed by a properly trained 6000-step Gumbel-sigmoid
   router (90.41% top-2).

3. **The routing bottleneck is training budget, not architecture.** The
   Gumbel-sigmoid ablation showed that simply training longer (+3000 steps)
   yields +5.3pp improvement. The real question for production is whether
   the marginal gains of learned routing justify the training cost over the
   closed-form centroid baseline.
