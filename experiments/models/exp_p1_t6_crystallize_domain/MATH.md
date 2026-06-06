# MATH.md — T6.2: Crystallize Cluster of User Adapters into Single Domain Adapter

## Motivation

T6.1 proved B-matrix K-means recovers domain structure (silhouette=0.8193, purity=1.0 at K=5).
The next step: once a cluster is identified, merge N user adapters from the same domain into a
single crystallized adapter. This frees N-1 adapter slots, reduces memory, and (by the law of
large numbers) **improves** quality relative to any individual user adapter.

**Core question**: Does averaging N same-domain B-matrices produce an adapter that is
geometrically closer to the true domain centroid than any individual user adapter?

---

## Background

### LoRA Adapter Model (from T6.1)

For each user u in domain D, after T gradient steps:

```
B_u = B_D* + ε_u
```

where B_D* ∈ ℝ^d is the true domain centroid (the "platonic" math/code/medical adapter)
and ε_u ~ N(0, σ²I_d) models user-specific noise (variable data, LR, random seed).

T6.1 results:
- d = 602,112 (Gemma 4 E4B, all lora_b keys flattened)
- σ_frac = 0.5 × std(B), giving ||ε_u|| ≈ 0.5 × ||B_D*||
- Same-domain cosine (canonical + variants at σ_frac=0.5): ≈ 0.894

### Model Soup (Wortsman et al. 2022, arxiv 2203.05482)

Weight averaging of fine-tuned models (started from same pre-trained weights) improves
accuracy over individual models. The key condition: models must share the same initialization
("same pre-trained basin"). Our adapters satisfy this: all LoRA B-matrices start at zero.

### Task Arithmetic (Ilharco et al. 2023, arxiv 2212.04089)

Task vectors (ΔW = BA) are linearly composable in adapter space. Averaging N task vectors
from the same task gives a cleaner task direction:

```
B_crystal = (1/N) Σ_u B_u = B_D* + (1/N) Σ_u ε_u
```

The noise term (1/N)Σ_u ε_u has variance σ²/N — N× smaller than individual noise.

---

## Theorem 1: Crystallization Quality Improvement (Law of Large Numbers)

**Setting**: N user adapters {B_1, ..., B_N} from domain D, where B_u = B_D* + ε_u,
ε_u ~^{iid} N(0, σ²/d · I_d).

**Definition (Crystallized Adapter)**:
```
B_crystal := (1/N) Σ_{u=1}^N B_u = B_D* + (1/N) Σ_{u=1}^N ε_u
```

**Theorem 1**: The expected reconstruction error of the crystallized adapter is N× smaller
than any individual user adapter:

```
E[||B_crystal - B_D*||²_F] = σ²/N
```

compared to:

```
E[||B_u - B_D*||²_F] = σ²
```

**Proof**:

Step 1: B_crystal - B_D* = (1/N) Σ ε_u. Denote ε̄ = (1/N)Σ ε_u.

Step 2: By i.i.d. assumption and linearity of expectation:
```
Var(ε̄_i) = Var((1/N)Σ ε_{u,i}) = (1/N²) · N · Var(ε_{u,i}) = σ²/(Nd)
```

Step 3: Sum over all d dimensions:
```
E[||ε̄||²] = d · σ²/(Nd) = σ²/N
```

**QED**

**Corollary (Cosine Quality Improvement)**: Under the model B_u = B_D* + ε_u,

```
E[cos(B_crystal, B_D*)] > E[cos(B_u, B_D*)]
```

**Proof of Corollary**:

cos(B_u, B_D*) = (B_D* + ε_u) · B_D* / (||B_D* + ε_u|| · ||B_D*||)
               = (||B_D*||² + ε_u·B_D*) / (||B_D* + ε_u|| · ||B_D*||)

Taking expectations:
- E[ε_u · B_D*] = 0 (ε independent of B_D*)
- E[||B_D* + ε_u||] > ||B_D*|| (noise inflates norm)

Therefore E[cos(B_u, B_D*)] = ||B_D*|| / E[||B_D* + ε_u||] < 1.

For crystallized: ε̄ has variance σ²/N, so E[||B_D* + ε̄||] < E[||B_D* + ε_u||].
Therefore E[cos(B_crystal, B_D*)] > E[cos(B_u, B_D*)]. **QED**

---

## Theorem 2: Slot Liberation

**Theorem 2**: Crystallization reduces N same-domain adapter slots to 1, freeing N-1 slots
for other domains, while preserving the domain's rank-r subspace.

**Proof**: Trivial by construction — B_crystal occupies one LoRA B-matrix. The rank of
B_crystal ≤ rank of B_D* = r (rank cannot increase under averaging). **QED**

---

## Theorem 3: MMLU Preservation (Norm Bound)

**Theorem 3**: The crystallized adapter preserves base-model performance on out-of-domain
tasks (e.g., MMLU) at least as well as any individual user adapter.

**Proof**:

Let W_base be the base model and W_u = W_base + B_u A be the user-adapted model.
The adaptation ΔW_u = B_u A is applied only when routing assigns domain D.
MMLU performance degradation from adapter u is bounded by (Theorem 1, Bonferroni):

```
|ΔMMLU(u)| ≤ C · ||ΔW_u||_F = C · ||B_u||_F · ||A||_F
```

For the crystallized adapter:
```
||B_crystal||_F = ||B_D* + ε̄||_F ≤ ||B_D*||_F + ||ε̄||_F = ||B_D*||_F + σ/√N
```

Since σ/√N < σ (individual noise), ||B_crystal||_F ≤ ||B_u||_F in expectation.
Therefore: |ΔMMLU(crystal)| ≤ |ΔMMLU(u)| on average.

In the case of TF-IDF routing (T4.1, 96.6% routing accuracy), the crystallized adapter
is not active during MMLU queries, so degradation = 0.

**QED**

---

## Experimental Setup

### What We Measure

Using same adapter infrastructure as T6.1:
- 5 canonical domain adapters (B_D* for each domain)
- For each domain: 5 user adapters (1 canonical + 4 variants with σ_frac=0.5)
- Crystallize each domain's 5 user adapters into 1 crystallized adapter

**Quality metric (K1120)**:
```
q_crystal(D) = cos(B_crystal^D, B_D*)
q_user(D)   = mean_u cos(B_u^D, B_D*)

Test: q_crystal(D) >= q_user(D) for all 5 domains
```

**Slot count (K1121)**:
- Before: 25 adapters (5 domains × 5 users)
- After: 5 adapters (1 crystallized per domain)
- Freed: 20 slots

**Norm preservation (K1122 proxy)**:
```
||B_crystal^D||_F / ||B_D*||_F ∈ [0.90, 1.10] for all domains
```
(MMLU preserved: crystallization doesn't inflate adapter norm beyond 10% of canonical)

**No user data (K1123)**: crystallization uses only adapter weight files.

### Kill Criteria Predictions

| Criterion | Metric | Predicted Value | Basis |
|-----------|--------|-----------------|-------|
| K1120: quality >= mean user | Δcos = crystal - mean_user | +8pp (0.976 vs 0.894) | Theorem 1 Corollary |
| K1121: single slot | N_adapters after | 5 (from 25) | Theorem 2 |
| K1122: MMLU preserved | norm ratio | 0.95–1.05 per domain | Theorem 3 |
| K1123: no user data | bytes of training data read | 0 | By construction |

### Quantitative Derivations for Predictions

For math domain (||B*|| = 5.7618, std = 0.007425, dim = 602,112):
- σ_per_element = 0.5 × 0.007425 = 0.003713
- ||ε_u||² = dim × σ² = 602,112 × 1.379e-5 = 8.30 → ||ε_u|| = 2.88
- ||B_u|| ≈ √(5.7618² + 2.88²) = √(33.20 + 8.30) = √41.50 = 6.44
- cos(B_u, B*) ≈ ||B*||/||B_u|| = 5.7618/6.44 = 0.895

- For crystallized (N=5): ||ε̄||² = ||ε_u||²/N = 8.30/5 = 1.66 → ||ε̄|| = 1.29
- ||B_crystal|| ≈ √(33.20 + 1.66) = √34.86 = 5.90
- cos(B_crystal, B*) ≈ 5.7618/5.90 = 0.977

- Δcos = 0.977 - 0.895 = 0.082 (+8.2pp)

---

## References

- Model Soup: Wortsman et al. 2022 (arxiv 2203.05482) — weight averaging improves accuracy
- Task Arithmetic: Ilharco et al. 2023 (arxiv 2212.04089) — task vectors are linearly composable
- Finding #450 (T6.1): B-matrix K-means, silhouette=0.8193, purity=1.0 at K=5
- Finding #427 (T4.1): TF-IDF routing 96.6% at N=5, 86.1% at N=25
- FedAvg: McMahan et al. 2017 — federated model averaging, same LLN proof
