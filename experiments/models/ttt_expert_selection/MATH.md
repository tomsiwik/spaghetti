# TTT Expert Selection: Mathematical Foundations

## Problem Statement

Given a base language model M with parameters theta, a set of N ternary LoRA adapters
{Delta_i = B_i A_i^T, i = 1..N}, and a new input sequence x = (x_1, ..., x_T),
select a subset S of k adapters that minimizes perplexity on x without a pre-trained
router.

**Notation:**
- d = 2560: model hidden dimension (BitNet-2B-4T)
- r = 16: LoRA rank
- N = 49: number of adapters (domains with data)
- k = 2: number of selected adapters
- T: sequence length
- h(x) in R^{T x d}: hidden states from base model on input x
- h_bar(x) = (1/T) sum_t h_t(x) in R^d: mean-pooled hidden state
- L(x; theta): base model loss (cross-entropy) on sequence x
- L(x; theta + sum_{i in S} alpha_i Delta_i): loss with selected adapters composed

## Strategy 1: Exhaustive Loss-Probe

**Procedure:**
1. Compute base loss L_0 = L(x; theta) via one forward pass
2. For each adapter i in {1..N}: compute L_i = L(x; theta + Delta_i)
3. Score adapter i: s_i = L_0 - L_i (loss reduction)
4. Select S = top-k({s_i})
5. Compose selected adapters and generate: theta' = theta + (1/k) sum_{i in S} Delta_i

**Cost:** N + 1 forward passes. For N=49, this is 50 forward passes -> violates K3 (>10).

**Quality:** This is the oracle upper bound for loss-based selection. Every adapter is
tested on the actual input, so the selection signal is optimal (modulo composition
interactions, which are second-order due to near-orthogonality).

## Strategy 2: Arrow-Style Projection Scoring

**Key insight (from MBC "Towards Modular LLMs"):** The relevance of adapter i to input x
can be estimated without a forward pass, by measuring how much the input's hidden states
align with the adapter's projection subspace.

**Procedure:**
1. Compute base hidden states h(x) via one forward pass -> h_bar in R^d
2. For each adapter i, compute relevance score:
   r_i = ||A_i^T h_bar||_2^2 / ||h_bar||_2^2
   This measures what fraction of h_bar's energy lies in adapter i's input subspace.
3. Select S = top-k({r_i})

**Cost:** 1 forward pass + N matrix-vector products (A_i^T h_bar, each R^{d} -> R^r).
The matrix-vector products are O(N * d * r) FLOPs = 49 * 2560 * 16 = 2.0M FLOPs,
negligible compared to one forward pass (~4B FLOPs for 2B model at T=128).

**This satisfies K3** (1 forward pass + negligible compute).

**Potential weakness:** Projection relevance != loss reduction. If adapter A-matrices
are random (not Grassmannian-optimized), the projection scores may not correlate
with actual benefit. But our A-matrices are random-uniform init, so their subspaces
DO vary across adapters, providing some discriminative signal.

## Strategy 3: Hierarchical Loss-Probe

**Procedure:**
1. Offline: cluster N adapters into C clusters by adapter parameter similarity
   (e.g., K-means on flattened vec(B_i A_i^T) or on the A-matrix column space)
2. At runtime:
   a. Compute base loss L_0 (1 pass)
   b. For each cluster c, probe the centroid adapter (C passes)
   c. Select top-2 clusters
   d. Within each cluster, probe individual adapters (~N/C passes per cluster)
3. Total: 1 + C + 2*(N/C) forward passes

**Optimal C:** Minimize 1 + C + 2N/C -> C* = sqrt(2N) ~= 10 for N=49.
Total passes: 1 + 10 + 2*5 = 21 -> still violates K3.

**With k=2 cluster selection and average cluster size N/C=5:**
1 + C + 2*(N/C) = 1 + 10 + 10 = 21 passes. Too many.

**Refinement:** Use only top-1 cluster, probe within:
1 + C + N/C = 1 + 7 + 7 = 15 passes. Still >10.

**Conclusion:** Hierarchical probe reduces N but cannot meet K3 for N=49.

## Strategy 4: Hybrid (Arrow + Selective Probe)

**Procedure:**
1. Arrow-style scoring to get top-m candidates (m << N), 1 forward pass
2. Loss-probe only the top-m candidates (m forward passes)
3. Select top-k by loss reduction

**Cost:** 1 + m forward passes. For m=3: 4 total passes -> satisfies K3.

**This is the promising approach.** Arrow-style filtering reduces the candidate set
cheaply, then loss-probing on a small set provides high-quality selection.

## Overhead Analysis (K1)

For a 512-token generation sequence, the base generation requires T forward passes
(autoregressive). The TTT selection overhead is incurred ONCE at the start:

**Strategy 2 (Arrow):** 1 forward pass on prefix (e.g., first 64 tokens).
Overhead ratio: 1 prefix pass / 512 generation passes = 0.2%.
-> K1 PASS trivially.

**Strategy 4 (Hybrid, m=3):** 4 forward passes on prefix.
Overhead ratio: 4 * (64/512) / 512 = 0.1% (prefix is short).
Actually: 4 full prefill passes vs 512 generation passes.
Each prefill pass processes 64 tokens, generation processes 1 token per step.
Cost ratio: 4 * 64 / 512 * (1/1) = 0.5 (if comparing FLOPs).
More precisely: prefill at seq_len=64 costs ~64x per-token generation cost.
4 * 64 = 256 token-equivalents. 512 generation tokens = 512 token-equivalents.
Overhead: 256/512 = 50%. Exactly at K1 threshold.

**Optimization:** Use prefix_len=32 instead of 64: 4 * 32 / 512 = 25%. K1 PASS.

## Composition After Selection

Once top-k adapters are selected, they are pre-merged into the base model:
theta' = theta + (1/k) * sum_{i in S} Delta_i

Pre-merge is a one-time weight addition (0.8% overhead per VISION.md).
Subsequent generation runs at full base-model speed.

## Worked Example (d=2560, r=16, N=49, k=2)

**Arrow scoring:**
- h_bar in R^2560 (mean-pooled hidden state from 64-token prefix)
- For adapter i: A_i in R^{2560 x 16} (random-uniform init)
- Score: r_i = ||A_i^T h_bar||^2 / ||h_bar||^2
  = (sum_j (a_j^T h_bar)^2) / ||h_bar||^2 where a_j are columns of A_i
- Each A_i^T h_bar: 2560 * 16 = 40,960 multiply-adds
- All 49 adapters: 49 * 40,960 = 2.0M multiply-adds
- For comparison: one base model forward at T=64: ~4B multiply-adds
- Arrow scoring cost: 0.05% of one forward pass

**Loss-probe (3 candidates):**
- 3 forward passes at T=64: 3 * 4B = 12B multiply-adds
- Prefix-only: 3 * 64/128 * 4B = 6B (half if model processes shorter sequences faster)
- Overhead vs 512-token generation: 6B / (512 * ~30M per token) = 6B/15B = 40%

## Assumptions

1. Mean-pooled hidden states capture domain signal (validated: router trains on them)
2. Adapter A-matrices span distinct subspaces (validated: |cos| = 0.0019 avg)
3. Prefix tokens (first 32-64) are sufficient domain signal
4. Loss reduction on prefix predicts loss reduction on full sequence
5. Adapter composition interactions are second-order (validated: |cos| < 0.01)
6. Pre-merge is equivalent to runtime LoRA for generation quality
