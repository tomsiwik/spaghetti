# Cross-Adapter Knowledge Transfer: Research Digest

## Hypothesis

Adding a foreign domain adapter (e.g., math) to a native domain adapter (e.g., code) at sub-unit weight improves native domain PPL by >2%, and the resulting NxN transfer matrix has non-random structure reflecting domain relatedness.

**Verdict: KILLED.** Zero pairs exceed the 2% threshold. Maximum pairwise transfer is 0.24%.

## What This Experiment Tests

For each ordered pair (foreign adapter A, native domain B) among 5 domain-specialized LoRA adapters on BitNet-2B-4T:
1. Compose adapter A + adapter B with weight alpha for A, (1-alpha) for B
2. Search alpha in {0.1, 0.2, 0.3, 0.5}
3. Measure PPL on domain B's validation set
4. Compute transfer coefficient: (PPL_B_alone - PPL_AB) / PPL_B_alone

This produces a 5x5 transfer matrix mapping how each adapter affects every other domain.

## Key References

- Grassmannian orthogonality framework (this project, proven)
- OSRM diagnostic (exp_bitnet_semantic_compositionality)
- Per-token routing (exp_bitnet_per_token_routing)
- Prior composition results (exp_bitnet_2b_real_composition)

## Empirical Results

### Individual Adapter Performance (all 5 beat base)

| Domain | Base PPL | Individual PPL | Improvement |
|--------|----------|---------------|-------------|
| python | 2.74 | 2.22 | +19.1% |
| math | 5.54 | 3.59 | +35.1% |
| medical | 6.96 | 4.76 | +31.6% |
| legal | 21.87 | 16.55 | +24.3% |
| creative | 6.35 | 4.94 | +22.3% |

### Transfer Matrix (rows=foreign adapter, cols=native domain, values=transfer %)

|          | python | math | medical | legal | creative |
|----------|--------|------|---------|-------|----------|
| python   | ---    | 0.00 | 0.00    | 0.00  | 0.00     |
| math     | +0.09  | ---  | +0.03   | 0.00  | 0.00     |
| medical  | +0.24  | 0.00 | ---     | 0.00  | 0.00     |
| legal    | +0.13  | 0.00 | 0.00    | ---   | 0.00     |
| creative | +0.10  | 0.00 | 0.00    | 0.00  | ---      |

### Summary Statistics

| Metric | Value |
|--------|-------|
| Pairs tested | 20 (5x4 ordered pairs) |
| Pairs with >2% transfer | **0** |
| Max transfer | +0.24% (medical -> python) |
| Mean transfer | +0.03% |
| Matrix variance | 0.004 |
| Symmetry correlation | -0.24 |
| Structure gap (expected-high vs expected-low) | 0.02% |

### Kill Criteria

| Criterion | Threshold | Actual | Result |
|-----------|-----------|--------|--------|
| K1: Zero pairs >2% improvement | At least 1 pair | 0 pairs | **KILL** |
| K2: Matrix is random | Variance > 0.5 or structure gap > 1% | Variance 0.004, gap 0.02% | **KILL** |

## Key Insights

### 1. Grassmannian Orthogonality Works Too Well for Pairwise Transfer

The very property that makes composition safe (near-zero cosine similarity |cos| ~ 0.001) also prevents pairwise knowledge transfer. Adapters operate on disjoint subspaces of the weight space. Adding adapter A at alpha=0.1 to adapter B is almost invisible because A's contribution falls in a subspace that B's domain data doesn't activate.

### 2. The Python Column Exception

Python is the only domain that receives any positive (albeit tiny) transfer from ALL foreign adapters. This makes sense: python code contains elements of all other domains (math in algorithms, medical/legal vocabulary in docstrings, creative in variable naming). But the effect is negligible (max 0.24%).

### 3. Reconciling with Prior Composition Results

Prior experiments showed composed 5-adapter PPL (with 1/N scaling) beats naive prediction. But this experiment shows pairwise transfer is negligible. The reconciliation:

- **1/N scaling is the mechanism, not transfer.** Composing 5 adapters at 1/5 weight each means each adapter's signal is diluted to 20%. This acts as regularization against individual adapter overfitting.
- **The "constructive transfer" in prior results is actually regularization.** The composed model performs between base and individual because each adapter contributes a small domain-specific signal without interference.
- **N-way composition is NOT the sum of pairwise transfers.** The nonlinear interaction of 5 adapters through the network may produce emergent effects that pairwise analysis cannot capture.

### 4. Implications for the Architecture

This result strengthens rather than weakens the Composable Ternary Experts vision:
- Adapters are truly independent (no pairwise coupling)
- Composition works via additive domain-specific signals, not cross-domain knowledge sharing
- The router's job is to SELECT the right adapter, not to blend complementary ones
- This validates the "pointer change" model: add/remove expert = no side effects

## Limitations

1. **Only 5 domains tested.** With 15 or more domains (some highly related, like python/javascript), pairwise transfer might emerge.
2. **200 training iterations.** Undertrained adapters may not have developed transferable features.
3. **Fixed alpha grid.** Very small alpha values (0.01, 0.05) or negative alpha (subtractive composition) were not tested.
4. **PPL as sole metric.** Transfer might exist in downstream task performance (accuracy, F1) that PPL doesn't capture.

## What Would Have Saved This

- Evidence of >2% transfer on at least 1 pair
- Non-random matrix structure (variance > 0.5, structure gap > 1%)
- Symmetry correlation > 0.3 (if A helps B, B should help A)

## What This Teaches

**Orthogonal adapters are independent modules, not collaborating agents.** The composition benefit comes from having the RIGHT adapter for each token (routing), not from mixing adapters within a token. This validates per-token routing as the correct composition mechanism and kills the "knowledge graph between adapters" framing.
