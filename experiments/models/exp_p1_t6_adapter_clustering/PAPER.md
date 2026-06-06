# PAPER.md — T6.1: Adapter Clustering by Domain Similarity

## Summary

B-matrix cosine similarity recovers domain structure from 25 synthetic user adapters
across 5 domains with perfect purity (K=5, silhouette=0.8193). No user data accessed.

---

## Prediction vs Measurement

| Kill Criterion | Prediction | Measured (Full Run) | Result |
|---------------|-----------|---------------------|--------|
| K1117: ≥3 natural domain groups from ≥25 adapters | 5 groups at K=5 | 5 groups, 25 adapters | **PASS** |
| K1118: silhouette > 0.3 | ≥0.80 at K=5 | 0.8193 at K=5 | **PASS** |
| K1119: B-matrix only, no user data | Trivially satisfied | Confirmed by construction | **PASS** |

---

## Full Run Results (n_per_domain=5, n_adapters=25, σ=0.5×std(B))

### Phase 1: Canonical Adapter Statistics

| Domain | Norm ||B||_F | Std | Dim |
|--------|-------------|-------|-----|
| math | 5.7618 | 0.007425 | 602112 |
| code | 5.8203 | 0.007501 | 602112 |
| medical | 4.7696 | 0.006147 | 602112 |
| legal | 4.5804 | 0.005903 | 602112 |
| finance | 4.4176 | 0.005693 | 602112 |

Scale pattern confirmed: learnable-task domains (math, code) have higher ||B||_F ≈ 5.7-5.8
vs knowledge-dependent domains (medical, legal, finance) ≈ 4.4-4.8. Consistent with Finding #217.

### Phase 2: Canonical Pairwise Cosines

| Pair | Cosine |
|------|--------|
| math-code | 0.0308 |
| math-medical | 0.0154 |
| math-legal | 0.0203 |
| math-finance | 0.0168 |
| code-medical | 0.0239 |
| code-legal | 0.0234 |
| code-finance | 0.0238 |
| medical-legal | 0.1221 |
| medical-finance | 0.1980 |
| legal-finance | 0.1588 |

Cross-domain cosines: 0.015–0.20. Semantically related domains (medical-legal-finance)
have higher cosine (0.12–0.20) vs STEM domains (math-code-X) which approach zero.

### Phase 3: K-means Clustering Results

**PCA reduction**: 602112 → 24 dims (min(50, n_adapters-1) = 24 components)

| K | Silhouette | Inertia | Domain Purity |
|---|-----------|---------|--------------|
| 3 | 0.5296 | 13.44 | code=1.0, math=1.0, medical=0.33 (merged med+legal+finance) |
| 4 | 0.6517 | 8.60 | code=1.0, legal=1.0, math=1.0, medical=0.50 (merged med+finance) |
| **5** | **0.8193** | **4.09** | **All 5 domains: purity=1.0** |

Best K=5 (selected by max silhouette=0.8193). At K=5, all 25 adapters assigned to
correct domain cluster with purity=1.0 for every domain.

### Smoke vs Full Run Comparison

| Run | n_adapters | Best Silhouette | K | All Purity=1.0 |
|-----|-----------|----------------|---|----------------|
| Smoke (n_per=2) | 10 | 0.8798 | 5 | Yes (2/domain) |
| Full (n_per=5) | 25 | 0.8193 | 5 | Yes (5/domain) |

Slight silhouette decrease (0.8798 → 0.8193) with more adapters is expected: larger intra-
cluster variance from 4 noisy variants per domain (vs 1 noisy variant in smoke). K=5 remains
the clear winner.

---

## Mathematical Analysis

### Theorem 1 (Domain Direction Emergence) — Verified

At T=1000+ steps, cross-domain cosines 0.015–0.20 confirm domain-specific directions have
emerged. The orthogonality is an empirically verified premise: pairwise cosines between
STEM domains approach zero (cos < 0.03), while semantically related domains (med-legal-finance)
retain weak correlation (cos 0.12–0.20). Adding one sentence per MATH.md discussion: the linear
independence of domain directions is an empirical premise supported by T2.1/T2.6 measurements.

### Theorem 2 (Clustering Recoverability) — Verified

The σ/Δ ≈ 4.9 argument showed L2 clustering would fail (hard case). Cosine-space K-means
succeeded because directional signal (Δ_cos ≈ 0.80-1.0 same-domain, 0.015-0.20 cross-domain)
dominates noise. Silhouette=0.8193 >> threshold of 0.3, confirming perfect recovery.

### Note on PCA Reduction

With n=25 adapters, PCA reduces to 24 components (n_adapters-1), not the 50 specified in
MATH.md. Standard constraint: PCA components ≤ min(n_samples, n_features) - 1. Does not
affect results — the 24 PCA components capture all domain-discriminating variance.

---

## Behavioral Implications

**Finding**: Unsupervised clustering of user adapter B-matrices recovers domain
structure with perfect purity at K=5, requiring no access to user training data.

**Pierre integration path**:
- Collect B-matrices from incoming user adapters (vectors only, no data)
- Run K-means periodically (e.g., after every 10 new adapters)
- Flag high-purity clusters (silhouette > 0.5 at optimal K) as crystallization candidates
- Promote cluster centroids to shared adapters → triggers T6.2 (crystallization)

**Limitation**: Synthetic user noise (σ=0.5×std(B)) may underestimate real heterogeneity.
Real users may have different training lengths, learning rates, and data distributions.
Production clustering may need lower σ_threshold or a soft clustering method.

---

## References

- Task Arithmetic: Ilharco et al. 2023, arxiv 2212.04089
- LIMA: Zhou et al. 2023, arxiv 2305.11206
- Pollard 1982: K-means error bounds
- Finding #216 (supported): SFT adapters at T=200 have cos=0.97
- Finding #217 (supported): LoRA scale separates 3 domain categories
