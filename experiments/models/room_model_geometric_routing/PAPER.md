# Room Model Piece B: Geometric Routing via Adapter Output Norms

## Theorem

**Theorem 2 (Geometric Routing Signal, from MATH.md):** For hidden state h and
adapter i with DeltaW_i = A_i @ B_i, the geometric routing score s_i(h) = ||h @ DeltaW_i||
is domain-discriminative if (a) domain centroids are separated, (b) JL-lemma
preserves separation through random A_i, and (c) trained B_i amplifies domain-i
projections. The geometric router selects i* = argmax_i s_i(h).

## Predictions

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| A-only routing ~14% (Finding #302 baseline) | 16.0% token, 12.0% seq | YES (near random) |
| DeltaW routing > 60% (K804, from JL + B amplification) | 31.2% best (layer 29) | NO (KILLED) |
| Agreement with ridge > 50% (K805) | 17.9% | NO (KILLED) |
| Improvement over A-only > 3x | 2.0x | NO |
| Mean weight on correct domain > 0.25 (1/N = 0.20 for random) | 0.201 | NO (indistinguishable from random) |
| Margin ratio (correct/max_other) > 1.2 | 0.848 | NO (correct is WEAKER than max competitor) |
| B-matrix norms roughly equal across domains | math 1.83 >> code 1.59 | PARTIALLY (15% spread, math dominates) |

## Hypothesis

The trained B-matrix amplifies domain-specific components of the random A-projection,
making adapter output norms ||h @ A_i @ B_i|| domain-discriminative.

**FALSIFIED.** The B-matrix does NOT provide sufficient domain-specific amplification.
Geometric routing scores are near-uniform across domains (mean weight on correct
domain 0.201 vs 0.200 for random). The math adapter's larger B-norm biases geometric
routing toward math regardless of input domain (math is "top confuser" for 4/5 domains).

## What This Model Is

Geometric routing uses the frozen adapter geometry (A-matrices and B-matrices) as a
zero-training, zero-parameter routing signal. For each hidden state h, it computes
s_i(h) = ||h @ A_i @ B_i|| for each adapter i and selects argmax. This is independent
of the killed pre-summing (Piece A, Findings #302/#303/#315) -- it uses the same adapter
components but for routing, not pre-merging.

## Key References

- FlyLoRA (arXiv 2510.08396): Frozen random A as implicit router via JL-lemma
- Finding #302: Room model A-subspace routing gave 14% (near random)
- Finding #310: Ridge regression router gave 98.3% token accuracy
- Finding #312: MLP per-token routing works, 6x signal over attention
- Johnson-Lindenstrauss lemma (1984): Random projections preserve distances

## Empirical Results

### Token-Level Routing Accuracy (N=5514 test tokens, 5 domains)

| Method | Accuracy | vs Random (20%) |
|--------|----------|-----------------|
| A-only (||h @ A_i||, layer 14) | 16.0% | -4.0pp |
| DeltaW single module (best: k_proj) | 19.5% | -0.5pp |
| DeltaW per-layer (best: layer 29) | **31.2%** | +11.2pp |
| DeltaW multi-layer (all 30 layers) | 18.0% | -2.0pp |
| Ridge router (Finding #310) | **98.3%** | +78.3pp |

### Sequence-Level Routing Accuracy (N=50 test sequences)

| Method | Accuracy |
|--------|----------|
| A-only | 12.0% |
| DeltaW multi-layer | 22.0% |
| Ridge router | 100.0% |

### Score Distribution

| Metric | Value |
|--------|-------|
| Mean weight on correct domain | 0.201 |
| Mean weight on other domains | 0.200 |
| Mean margin ratio (correct/max_other) | 0.848 |
| Pct tokens where correct > max_other | 18.0% |

The routing signal is **indistinguishable from random.** The mean weight on the
correct domain (0.201) is only 0.1pp above the other domains (0.200). The margin
ratio is BELOW 1.0, meaning the wrong domain typically has a higher score than
the correct one.

### B-Matrix Norm Bias

| Domain | Mean ||B|| | Consequence |
|--------|------------|-------------|
| math | 1.828 | Geometric routing defaults to math |
| medical | 1.683 | |
| legal | 1.669 | |
| finance | 1.624 | |
| code | 1.588 | Lowest norm, worst routing accuracy |

The math adapter has the largest B-norms (1.83) -- 15% above code (1.59). This
creates a systematic bias: geometric routing picks math as the winner for inputs
from ALL domains (math is the "top confuser" for medical, code, legal, and finance).

### Agreement with Ridge Router

| Metric | Value |
|--------|-------|
| Overall agreement | 17.9% |
| When they disagree (82.1% of tokens): geo right | 0.3% |
| When they disagree: ridge right | 98.1% |
| When they disagree: both wrong | 1.6% |

The geometric router adds virtually zero information. When it disagrees with ridge,
it is wrong 98.1% of the time.

### Per-Layer Analysis

| Layer | Accuracy | Interpretation |
|-------|----------|----------------|
| 0 | 24.4% | Early layer -- some signal from embedding geometry |
| 7 | 17.1% | Near random |
| 14 | 15.7% | Near random |
| 21 | 16.3% | Near random |
| 29 | **31.2%** | Last layer -- most domain-discriminative hidden states |

Layer 29 is the only layer with meaningful signal (31.2%), but still far below
the 60% kill threshold. This makes sense: later layers have the most domain-
specific representations, and the JL projection preserves more of this structure.
However, the B-matrix amplification is not domain-specific enough to push
accuracy above random + small margin.

### Latency

| Method | Single token | Batch 128 per token |
|--------|-------------|-------------------|
| Geometric (1 layer, 6 modules) | 4.32ms | 0.070ms |
| Ridge router (Finding #310) | 0.17ms | 0.001ms |

Geometric routing is 25x slower single-token and 50x slower batched than the
ridge router, while being 53x less accurate. The ridge router requires one d x N
matmul (2560 x 5); geometric routing requires N modules x (d x r + r x d_out)
matmuls.

## Kill Criteria Assessment

| Criterion | Threshold | Measured | Result |
|-----------|-----------|----------|--------|
| K804: Geometric routing accuracy | >= 60% | 31.2% (best single-layer) | **FAIL** |
| K805: Agreement with ridge router | >= 50% | 17.9% | **FAIL** |

**Verdict: KILLED**

Both kill criteria fail decisively. Best geometric routing (31.2%) is half the
60% threshold. Agreement with ridge (17.9%) is below the 50% threshold.

## Limitations

1. **Hidden states are from the FINAL layer only.** The geometric routing
   computes h @ A_i @ B_i using final-layer hidden states, but A_i and B_i
   are trained on PER-LAYER hidden states. The mismatch between final-layer h
   and the module's expected input h_l introduces noise.

2. **Norm aggregation.** Taking ||c_i|| discards directional information that
   might be domain-discriminative. An inner product c_i . v against a reference
   vector v could be more informative.

3. **5 domains only, with synthetic boundaries.** The test uses 5 cleanly
   separated domains from prior experiments.

## What Would Kill This (and Did)

The mathematical Assumption 3 (B_i training amplifies domain-i signal in the
A_i projection) was the key assumption, and it FAILED. The evidence:

1. **Near-uniform scores.** Mean weight on correct domain (0.201) is
   indistinguishable from random (0.200). The B-matrix does not amplify
   domain-specific projections.

2. **B-norm bias dominates.** The math adapter's larger B-norms (1.83 vs
   1.59-1.68 for others) create a systematic bias toward math regardless
   of input domain. The global B-norm scale overpowers any local domain
   signal in the projection geometry.

3. **JL epsilon too large at r=16.** The JL distortion epsilon ~ 0.32 at
   r=16 for N=5 means distances are preserved within 32%. Combined with the
   lack of B-matrix amplification, this 32% noise floor drowns the signal.

## Why This Fails: Structural Explanation

The fundamental reason geometric routing fails is a **mismatch between what
A-matrices are designed for and what routing needs.**

Grassmannian A-matrices are optimized for **interference prevention**:
A_i^T A_j ~ 0 ensures adapter outputs don't contaminate each other. This
means A_i projects h into a RANDOM 16-dimensional subspace that has no
special relationship with domain-i semantics.

For geometric routing to work, A_i would need to project h into a subspace
that is ALIGNED with domain-i structure. But that would contradict the
Grassmannian orthogonality constraint -- you cannot simultaneously be
orthogonal to all other adapters AND aligned with a specific domain.

The ridge router succeeds because it learns an UNCONSTRAINED linear map
from the FULL 2560-dimensional hidden state. It can exploit ALL d=2560
dimensions for routing. The geometric router is limited to r=16 random
dimensions per adapter, a 160x information bottleneck.

**Conclusion:** Adapter geometry is designed for COMPOSITION (interference
prevention), not ROUTING (domain discrimination). These are fundamentally
different objectives. The proven ridge router (98.3%, 0.17ms, 13K params)
remains the correct routing mechanism.
