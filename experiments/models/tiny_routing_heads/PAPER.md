# Tiny Routing Heads: Research Digest

## Hypothesis

Per-adapter binary routing heads (~82K params each) can identify their own domain
with >75% accuracy and enable top-2 composition that beats uniform 1/N averaging,
eliminating the need for a centralized router.

## What This Model Is

Each of N domain adapters carries its own tiny routing head -- a 2-layer MLP
(d -> 32 -> 1 -> sigmoid) that answers one question: "Am I useful for this input?"
The head is trained as a binary classifier on the adapter's own domain data (positive)
vs random data from other domains (negative). At inference, all heads score the input
in parallel, the top-2 highest-scoring adapters are selected, and their weights are
pre-merged with score-proportional weighting before a single forward pass.

This is a decentralized routing architecture: each adapter is self-aware of when
to activate. No centralized router is needed. Adding a new adapter requires only
training its head (0.2 seconds) -- no retraining of existing heads.

## Key References

- MoLoRA (2603.15965): per-token routing of LoRA adapters
- X-LoRA (2402.07148): dense/sparse gating over LoRA experts
- LD-MoLE (2509.25684): learnable dynamic routing for MoE LoRA
- L2R: Gumbel-sigmoid non-competing multi-adapter routing

## Empirical Results

**Platform:** Apple M5 Pro 48GB, MLX 0.31.1, BitNet-2B-4T (2.4B params, d=2560)
**Runtime:** 263 seconds ($0)
**Adapters:** 5 domains (python, math, medical, legal, creative), LoRA rank-16

### Head Classification Accuracy (K1)

| Domain | Own-Domain Acc | Negative Acc | Overall |
|--------|---------------|-------------|---------|
| python | 100% | 100% | 100% |
| math | 100% | 100% | 100% |
| medical | 100% | 100% | 100% |
| legal | 100% | 100% | 100% |
| creative | 100% | 100% | 100% |

**K1 PASS:** All heads achieve 100% accuracy (threshold: 75%). The domains are
highly separable in the base model's hidden state space.

### Head Inference Overhead (K2)

| Metric | Value |
|--------|-------|
| Base forward pass | 37.01 ms |
| All 5 heads | 0.86 ms |
| Overhead | 2.32% |

**K2 PASS:** 2.32% overhead (threshold: 5%). The heads are genuinely tiny.

### Composition Quality (K3 + S1)

| Domain | Base PPL | Individual | Uniform 1/N | Routed Top-2 |
|--------|----------|-----------|-------------|-------------|
| python | 2.74 | 2.22 | 2.52 | 2.22 |
| math | 5.54 | 3.60 | 4.94 | 3.61 |
| medical | 6.96 | 4.75 | 6.20 | 4.75 |
| legal | 21.87 | 16.56 | 20.44 | 16.58 |
| creative | 6.35 | 4.92 | 5.96 | 4.94 |
| **Average** | **8.69** | **6.41** | **8.01** | **6.42** |

**K3 PASS:** Routed PPL 6.42 vs uniform 8.01 (+19.9% improvement)
**S1 PASS:** 19.9% improvement exceeds 5% threshold

The routed composition achieves near-oracle performance: avg PPL 6.42 vs individual
oracle 6.41 (only 0.15% gap). The heads perfectly route each domain to its own
adapter, and the top-2 selection with score weighting recovers essentially all of
the individual adapter quality.

### Parameter Efficiency (S2)

| Metric | Value |
|--------|-------|
| Total head params | 409,925 (410K) |
| Total adapter params | 108,134,400 (108M) |
| Head/adapter ratio | 0.38% |

**S2 PASS:** 0.38% < 1% threshold. Heads are truly tiny relative to adapters.

### Independence (S3)

**S3 PASS:** By construction. Each head trains independently in 0.2s using only
the base model's hidden states and its own domain data. Adding adapter N+1 requires
only training head N+1.

## Comparison to Centralized Router

The previous experiment (exp_bitnet_per_token_routing) used a centralized 2-layer
MLP router (659K params) over 15 domains, achieving +13.9% improvement over uniform.

| Approach | Router Params | Improvement | Independence |
|----------|-------------|-------------|-------------|
| Centralized (N=15) | 659K (single) | +13.9% | No (retrain for new adapter) |
| Per-adapter heads (N=5) | 410K (5 x 82K) | +19.9% | Yes (train only new head) |

The per-adapter approach achieves HIGHER improvement (19.9% vs 13.9%) while being
fully decentralized. Caveat: different N (5 vs 15) -- uniform dilution is worse at
higher N, so the centralized router faces a harder problem. The key structural
advantage is independence: adding adapter N+1 does not require retraining anything.

## Limitations

1. **Only N=5 tested.** The 100% classification accuracy may degrade with more
   domains, especially similar ones (e.g., python vs javascript). The centralized
   router was tested at N=15. Scaling to N=15+ is the natural next step.

2. **Domains are highly distinct.** Python code vs medical text vs creative writing
   are easy to discriminate. The real test is closely related domains (e.g.,
   cardiology vs oncology, or Python vs JavaScript).

3. **Single seed.** No variance estimates. However, the margins are enormous
   (100% accuracy, 19.9% improvement) making seed sensitivity unlikely to change
   the verdict.

4. **Sequence-level routing only.** The heads route at sequence granularity (mean
   pooling over all tokens). Per-token routing within a sequence could further
   improve mixed-domain inputs.

5. **Same training data for adapters and heads.** The hidden states used for head
   training come from the same distribution as adapter training data. Real deployment
   would see distribution shift.

6. **Head training loss ~0.5 (not ~0).** The binary cross-entropy loss plateaus
   around 0.5 despite 100% accuracy, suggesting the logits are small but correct.
   The heads could be trained longer or with different hyperparameters for sharper
   discrimination.

## What Would Kill This

**At micro scale:**
- Accuracy drops below 75% when N > 10 with similar domains
- Routing fails on mixed-domain inputs (e.g., medical Python code)
- Head scores become uniform (all ~0.5) on out-of-distribution inputs

**At macro scale (future):**
- Per-adapter heads fail on real model (Qwen-7B) with continuous adapters
- Score calibration between heads from different training runs is inconsistent
- Overhead grows super-linearly with N (unlikely given O(N) architecture)

## Key Insight

The base model's hidden states already contain strong domain signals. A 2-layer MLP
with 82K parameters is sufficient to decode domain identity with 100% accuracy.
This means routing is not a hard problem -- it is a readout problem. The information
for routing is already in the representation; we just need to read it out.

This supports the hypothesis that any model is a sparse composition of experts:
the base model knows which knowledge is relevant, and trivially cheap probes can
extract that routing signal.
