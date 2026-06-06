# Capability Expert Taxonomy: Mathematical Foundations

## Variables and Notation

| Symbol | Definition | Dimension |
|--------|-----------|-----------|
| d | Model hidden dimension | scalar (2560 for BitNet-2B-4T) |
| r | LoRA rank | scalar (16) |
| L | Number of transformer layers | scalar (30) |
| M | Number of projection matrices per layer | scalar (7: q,k,v,o,gate,up,down) |
| N_c | Number of capability adapters | scalar (4 in this experiment) |
| N_d | Number of domain adapters | scalar (5) |
| N = N_c + N_d | Total adapters | scalar (9) |
| A_i^{l,m} | LoRA-A matrix for adapter i, layer l, projection m | (d_in, r) |
| B_i^{l,m} | LoRA-B matrix for adapter i, layer l, projection m | (r, d_out) |
| Delta_i | Flattened adapter parameter vector | (P,) where P = L * M * r * (d_in + d_out) |

## Orthogonality in High-Dimensional LoRA Space

Each adapter i is represented as a flattened vector Delta_i in R^P, where:

  P = sum over all (l,m) of r * (d_in^{l,m} + d_out^{l,m})

For BitNet-2B-4T with r=16, L=30, M=7:
  P = 420 tensors * avg_dim ~= 21.6M parameters per adapter

The cosine similarity between adapters i, j is:

  cos(Delta_i, Delta_j) = <Delta_i, Delta_j> / (||Delta_i|| * ||Delta_j||)

## Random Baseline

For random vectors in R^P, the expected |cos| follows:

  E[|cos|] ~ sqrt(2 / (pi * P))

For P = 21.6M:
  E[|cos|] ~ sqrt(2 / (pi * 21.6e6)) ~ 1.72e-4

This is the "near-random" baseline. Observed values significantly above this
indicate shared structure; values near this indicate orthogonal subspaces.

## Capability vs Domain Orthogonality Hypothesis

We hypothesize that capability adapters (which modify behavioral patterns)
occupy different subspace directions than domain adapters (which modify
knowledge content). Formally:

  H0: E[|cos(cap_i, cap_j)|] = E[|cos(dom_i, dom_j)|] = E[|cos(cap_i, dom_j)|]
  H1: Capabilities and domains occupy orthogonal subspace families

Under H1, we expect:
  - cap-cap cosines near random baseline (different behavioral modes)
  - cap-domain cosines near random baseline (orthogonal subspace families)
  - domain-domain cosines near random baseline (established from prior work)

## Composition Under 1/N Scaling

For N total adapters with 1/N scaling, the composed model is:

  W_composed = W_base + (1/N) * sum_i (B_i @ A_i)

The expected PPL improvement for each adapter's domain/capability:

  E[PPL_composed(d_i)] ~ PPL_base(d_i) - (1/N) * (PPL_base(d_i) - PPL_individual(d_i))

Under perfect orthogonality (|cos| ~ 0), cross-adapter interference is
negligible and improvements are additive at 1/N scale.

## Kill Criteria Thresholds

K1 threshold (mean |cos| < 0.01): This is ~58x the random baseline (1.72e-4).
It allows substantial room above random while still requiring near-orthogonality.
Domain adapters on this same model achieved mean |cos| = 0.001, so the
threshold is achievable.

K2 threshold (max |cos| < 0.01): Stricter than K1 because even one
high-interference pair would indicate capability entanglement.

## Worked Example

Given 4 capability adapters with observed mean |cos|:
  reasoning:    0.000831
  instruction:  0.000525
  conciseness:  0.000318
  safety:       0.000447

All are within [1.8x, 4.8x] of the random baseline (1.72e-4), meaning
capability adapters are nearly as orthogonal as random vectors in R^{21.6M}.
The max pairwise |cos| = 0.001035 (reasoning-safety) is 10x below the
kill threshold.

Cross-type ratio: cap-cap mean (0.000530) / domain-domain mean (0.000983) = 0.54x.
Capability adapters are actually MORE orthogonal than domain adapters, suggesting
behavioral modes are even more separable than knowledge domains.

## Assumptions

1. **Flattened cosine is a valid proxy for functional interference.** This is
   supported by prior SOLE experiments showing cos < 0.01 correlates with
   successful composition, but the proxy could miss layer-specific interactions.

2. **200 training steps is sufficient to learn capability-specific patterns.**
   3/4 adapters converged (loss reduction > 5%). The instruction adapter
   plateaued without converging, but still achieved 62.7% PPL improvement.

3. **BitNet-2B-4T ternary weights do not bias the orthogonality measurement.**
   The ternary constraint bounds adapter magnitudes, which may systematically
   improve orthogonality compared to FP16 bases. This is consistent with
   prior findings (bitnet_ternary_adapter_composition: -19.3% cos reduction).
