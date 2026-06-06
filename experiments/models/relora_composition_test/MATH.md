# ReLoRA Composition Test: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| d | Model embedding dimension | 64 (micro) |
| d_ff | FFN intermediate dimension | 4d = 256 |
| r | LoRA rank | 8 |
| L | Number of transformer layers | 4 |
| K | Number of ReLoRA merge cycles | 5 |
| N | Number of domain experts | 4 |
| alpha | LoRA scaling factor | 1.0 |

## 2. ReLoRA Weight Accumulation

### 2.1 Single LoRA Update

A LoRA adapter modifies weight matrix W in R^{d_out x d_in}:

    W' = W + (alpha/r) * B @ A

where A in R^{d_in x r}, B in R^{r x d_out}.

The update dW = (alpha/r) * B @ A has rank at most r.

### 2.2 Iterative Merge (ReLoRA)

**Clarification (rev2, fix 4):** ReLoRA pretraining trains ALL model
parameters (base weights W, embeddings, layer norms, AND LoRA A/B
matrices). This is NOT pure iterative-LoRA; it is standard training
where LoRA adapters are periodically merged into the base and reset.
The full-parameter training means the base weights evolve through both
(a) direct gradient updates on W and (b) periodic rank-r merges from
LoRA. At our micro scale, the LoRA contribution is one component of
the total weight evolution, not the only mechanism.

After K merge cycles, the accumulated weight is:

    W_K = W_0 + gradient_updates(steps 1..T) + sum_{k=1}^{K} dW_k

where each dW_k = (alpha/r) * B_k @ A_k is the LoRA delta at merge k,
and gradient_updates encompasses all direct updates to W during training.

**Rank bound on LoRA contributions only:**

    rank(sum_{k=1}^{K} dW_k) <= min(K * r, min(d_in, d_out))

With K=5 merges and r=8: rank <= min(40, 64) = 40.

However, because all parameters are trained, the LoRA merge contribution
is additive on top of the full-rank gradient evolution of W. The final
weight matrix is effectively full-rank.

### 2.3 ReLoRA vs Conventional Effective Rank

For a weight matrix W in R^{d_out x d_in}, the effective rank
(Roy & Vetterli, 2007) is:

    r_eff(W) = exp(H(p))

where p_i = sigma_i / sum_j(sigma_j) are the normalized singular values
and H(p) = -sum_i p_i log(p_i) is Shannon entropy.

**Experimental result (rev2, 3 seeds):**
- ReLoRA: mean r_eff = 53.36 +/- 0.60
- Conventional: mean r_eff = 53.29 +/- 0.34
- Ratio: 1.001 (essentially identical)

This confirms ReLoRA achieves comparable weight spectrum to conventional
training at micro scale.

## 3. LoRA Expert Orthogonality

### 3.1 Expert Delta Vectors

For expert i, the LoRA delta across all L layers and 2 MLP sublayers
is a vector:

    v_i = flatten(dW_i^{(1,fc1)}, dW_i^{(1,fc2)}, ..., dW_i^{(L,fc2)})

    dim(v_i) = L * (d * d_ff + d_ff * d) = L * 2 * d * d_ff
             = 4 * 2 * 64 * 256 = 131,072

### 3.2 Pairwise Cosine Similarity

For experts i, j:

    cos(v_i, v_j) = <v_i, v_j> / (||v_i|| * ||v_j||)

Under the null hypothesis (random independent training), for vectors
in R^D with D = 131,072:

    E[|cos|] ~ sqrt(2 / (pi * D)) ~ 0.0039

### 3.3 Experimental Results (rev2: corrected data splits, 3 seeds, 4 experts each)

| Metric | ReLoRA Base | Conventional Base | Ratio |
|--------|------------|-------------------|-------|
| mean|cos| | 0.0456 +/- 0.021 | 0.0269 +/- 0.004 | 1.77x |
| max|cos| | 0.149 +/- 0.001 | 0.042 +/- 0.008 | 3.55x |

**95% Bootstrap CI on cos_ratio: [0.77, 2.64]**

The wide CI reflects high variance across seeds (per-seed ratios: 0.77x,
1.90x, 2.64x). This includes a seed (42) where ReLoRA actually shows
LOWER cosine than conventional, and a seed (123) where it is 2.6x higher.

**Permutation test p-value: 0.056** (marginally non-significant at alpha=0.05)

Both mean values are well above the random expectation (0.0039), indicating
domain-specific structure in the deltas. The key observation: the difference
between ReLoRA and conventional cosine is NOT statistically significant at
the 0.05 level under the corrected experimental protocol.

### 3.4 Interference Bound Under Composition

When composing N experts additively:

    W_composed = W_base + sum_{i=1}^{N} dW_i

The interference between any pair (i,j) is bounded by:

    |<dW_i, dW_j>| / (||dW_i|| * ||dW_j||) <= cos_max

For N=4 experts, the total interference is:

    ||W_composed - W_ideal|| <= C(N,2) * cos_max * max(||dW_i||)
                               = 6 * 0.149 * ||dW|| (ReLoRA worst case)
                               = 6 * 0.042 * ||dW|| (conventional worst case)

Both are small relative to ||W_base||. At macro scale (d=896, r=16),
the gap is expected to be much smaller due to dimensionality.

## 4. Quality Analysis

### 4.1 Expert Loss on Domain Data

| Metric | ReLoRA Base | Conventional Base | Loss Ratio |
|--------|------------|-------------------|------------|
| Mean expert val loss | 0.4705 +/- 0.006 | 0.4473 +/- 0.006 | 1.052 |

**Interpretation (rev2, advisory 6):** ReLoRA expert loss is 5.2% higher
than conventional expert loss. This is reported as a direct loss ratio
(relora/conv), not as an inverted "quality" percentage.

The 5.2% gap decomposes as:
1. ReLoRA base itself has ~4.6% higher val loss (less optimal pretraining)
2. The remaining ~0.6% is the composition penalty

95% Bootstrap CI on loss_ratio: [1.041, 1.074]

### 4.2 Base Model Quality

| Base | Val Loss | Relative |
|------|----------|----------|
| ReLoRA (5 merges) | 0.539 +/- 0.005 | 1.046x |
| Conventional | 0.516 +/- 0.003 | 1.000x |

ReLoRA base is ~4.6% worse than conventional at micro scale. This is
consistent with the original ReLoRA paper which shows a small gap at
small model sizes that closes with scale.

## 5. Assumptions and Limitations

1. **Micro scale**: d=64, r=8, L=4. At macro scale (d=3584, r=16),
   cos values are expected to be ~100x smaller due to high-dimensional
   geometry.

2. **Same total steps**: Both bases see identical training data for
   the same number of steps. ReLoRA is ~30% slower (optimizer resets).

3. **All-parameter training (fix 4)**: ReLoRA pretraining trains ALL
   parameters, not just LoRA A/B. The periodic merge-and-restart of
   LoRA is layered on top of full-parameter training. This means the
   "base is just another adapter" thesis is specifically about whether
   the weight evolution PATH matters for composition, not about whether
   the base was built from pure LoRA updates.

4. **FFN-only LoRA**: Experts use LoRA on MLP layers only (fc1, fc2),
   matching our architecture choice from the FFN-only orthogonality
   experiment.

5. **Domain overlap**: Character-level name generation domains (by
   first letter) have significant overlap, inflating cos values
   compared to truly disjoint domains.

6. **Deterministic domain splits (fix 1)**: Domain data is split 80/20
   with per-domain seeded shuffle. Both conditions (ReLoRA and conventional)
   train on the same 80% and evaluate on the same 20% per domain.

## 6. Worked Example (d=64, r=8, K=5)

Setup:
- Weight matrix W_fc1 in R^{256 x 64}
- Each LoRA: A in R^{64 x 8}, B in R^{8 x 256}
- Delta: dW = (1/8) * B @ A, rank <= 8

After K=5 merges (with full-parameter training):
- W_5 = W_0 + gradient_updates + sum_{k=1}^{5} (1/8) * B_k @ A_k
- Theoretical max rank of LoRA-only contributions: min(40, 64) = 40
- Observed effective rank: ~53.4 (full-rank due to all-parameter training)

Expert training on W_5:
- Fresh LoRA: A_new in R^{64 x 8}, B_new = 0
- After 300 steps: ||dW_expert|| ~ 0.1 * ||W_5||
- cos(expert_1, expert_2) ~ 0.046 (ReLoRA) vs ~0.027 (conventional)
- Difference is marginally non-significant (p = 0.056, permutation test)
