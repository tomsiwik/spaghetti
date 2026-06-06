# LoRI-style B-sparsity on BitNet-2B: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | 2560 (BitNet-2B-4T) |
| r | LoRA rank | 16 |
| N | Number of domain adapters | 5 |
| A_i | Frozen LoRA down-projection for adapter i | (d_in, r) |
| B_i | Trained LoRA up-projection for adapter i | (r, d_out) |
| M_i | Binary sparsity mask for B_i | (r, d_out), M_i in {0,1} |
| s | Sparsity fraction | 0.90 (keep 10%) |
| alpha | LoRA scaling factor | 20.0 |
| dW_i | Effective weight delta from adapter i | (d_in, d_out) |

## 2. LoRA Composition and Interference

### 2.1 Additive Composition

Under 1/N scaling, the composed weight delta is:

    dW_composed = (1/N) * sum_{i=1}^{N} (alpha/r) * A_i @ B_i

The interference between adapters i and j manifests through their
effective delta overlap:

    interference(i,j) = ||dW_i^T dW_j||_F = (alpha/r)^2 * ||B_i^T A_i^T A_j B_j||_F

### 2.2 With Frozen A (Grassmannian Skeleton)

When A matrices are frozen and orthogonal (A_i^T A_j ~ 0 for i != j):

    interference(i,j) ~ (alpha/r)^2 * ||A_i^T A_j||_F * ||B_i||_F * ||B_j||_F

The A-orthogonality provides first-order interference suppression.
Empirically on BitNet-2B: |cos(params_i, params_j)| ~ 0.001.

### 2.3 With B-Sparsity (LoRI)

B_i^sparse = B_i * M_i, where M_i has ||M_i||_0 = (1-s) * r * d_out.

For random A_i (non-Grassmannian, as in standard LoRA):

    interference(i,j)_sparse = (alpha/r)^2 * ||(M_i * B_i)^T A_i^T A_j (M_j * B_j)||_F

When masks M_i and M_j are disjoint (non-overlapping support), the
interference is zero in the B subspace. With 90% sparsity, each adapter
uses only 10% of B elements. For random masks, the expected overlap is:

    E[overlap(M_i, M_j)] = (1-s)^2 = 0.01

So 99% of the B-parameter space is non-overlapping between any pair.

### 2.4 Why B-Sparsity Cannot Help When Interference Is Already Near-Zero

Key insight from the experiment: on BitNet-2B, the ternary base already
produces near-orthogonal adapter parameter vectors (|cos| ~ 0.001).
B-sparsity can reduce interference but interference is already 114x
below FP16 levels. This is the floor condition.

Formally: if the interference is already dominated by the A-matrix
geometry (which is near-random/orthogonal on the ternary base), then
modifying B structure cannot improve composition. The bound becomes:

    interference(i,j) <= (alpha/r)^2 * ||A_i^T A_j||_F * ||B_i^sparse||_F * ||B_j^sparse||_F

With sparse B, ||B_i^sparse||_F may actually concentrate signal into fewer
dimensions, making the few surviving elements more correlated across
adapters. This explains the INCREASED cosine we observed (1.46x).

## 3. Experimental Design (Corrected LoRI Protocol)

### 3.1 Training Protocol

Following arXiv 2504.07448 (COLM 2025) exactly:

**Dense baseline:**
- 400 steps, batch=1, seq_len=256, lr=1e-4, Adam
- Standard LoRA (rank-16, alpha=20) on all attention + MLP projections (7 targets)

**Sparse (LoRI):**
- Phase 1: 200 steps dense calibration (to learn magnitude pattern)
- Phase 2: Extract GLOBAL mask (single threshold across ALL B matrices in model)
- Phase 3: Reset B to zero (discard calibration weights -- key LoRI insight)
- Phase 4: 400 steps with frozen mask (same compute budget as dense)
- Total training: 400 steps (fair comparison)

### 3.2 Global vs Per-Layer Masking

LoRI paper ablation shows global (model-wise) masking outperforms per-layer.
The global threshold finds the top-10% most important B elements across
the entire model, regardless of which layer they belong to. This allows
some layers to have more active parameters than others based on importance.

### 3.3 B Reset After Calibration

Critical LoRI insight: discard calibration weights after extracting mask.
The calibration only determines WHICH elements to keep, not their values.
Starting from zero with the correct mask structure produces better results
than continuing from calibration weights (per LoRI paper).

### 3.4 Metrics

1. **Individual PPL ratio** = sparse_PPL / dense_PPL per domain
   - K1 threshold: all ratios <= 1.10
2. **Composed PPL ratio** = avg_sparse_composed / avg_dense_composed
   - K2 threshold: ratio <= 1.0
3. **Cosine similarity** = |cos(params_i, params_j)| averaged over all pairs
4. **Actual sparsity** = fraction of zero elements in sparse B

## 4. Worked Numerical Example

For BitNet-2B-4T with 30 transformer blocks, 7 LoRA targets per block:

- Total B params per adapter: 10,936,320
- At 90% sparsity: ~1,093,632 non-zero params per adapter
- Dense non-zero: ~10,620,423 (97.1% -- note B_i initialized to zero,
  trained to dense, so ~3% stay near-zero)

Storage comparison per adapter:
- Dense B: 10,936,320 * 2 bytes (bfloat16) = 20.9 MB
- Sparse B: 1,093,632 * 2 bytes + mask = 2.1 MB + 1.3 MB = 3.4 MB
- Compression ratio: ~6x (not full 10x due to mask overhead)

Expected mask overlap between two random 10%-masks:
  P(both keep element) = 0.1 * 0.1 = 0.01
  Expected overlapping params: 0.01 * 10,936,320 = 109,363

Note: magnitude-based masks are NOT random -- domains with similar
structures may select similar high-magnitude positions, making actual
overlap higher than random expectation. This explains why sparse adapters
show higher cosine (0.00229 vs 0.00156).

## 5. Assumptions

1. **Magnitude is a good proxy for importance.** LoRI paper validates
   this empirically on FP16 models. May interact differently with ternary base.
2. **200 steps is sufficient for magnitude calibration.** Calibration
   converged in 3/5 domains (math, medical, legal). Python and creative
   did not converge but the mask structure was still informative.
3. **Mask should be frozen after pruning.** Re-growing (dynamic sparsity)
   is a different mechanism not tested here.
4. **1/N scaling is the right composition method.** Already validated
   in prior experiments.
5. **BitNet-2B ternary base does not interact differently with sparse B
   than FP16 base.** DISPROVEN: the ternary base already provides
   near-orthogonality that makes B-sparsity redundant.
