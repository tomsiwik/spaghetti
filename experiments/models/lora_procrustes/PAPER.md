# LoRA Procrustes Linear Decomposition: Research Digest

## Hypothesis

LoRA deltas (pure linear: dW = A @ B) can be decomposed into shared (always-on) +
unique (routed) components without the nonlinearity penalty that killed the
original Procrustes decomposition (exp3) on capsule groups.

**Falsifiable**: If decomposed LoRA composition degrades >3% vs concatenated LoRA,
or shared component is <10% of delta norm, the approach fails.

---

## What This Model Is

This experiment resurrects the killed Procrustes decomposition idea
(exp3_procrustes_decomp) by applying it to LoRA adapters instead of capsule
groups. The original experiment failed because ReLU breaks weight-space
decomposition: ReLU(shared(x)) + ReLU(unique(x)) != ReLU((shared+unique)(x)).

LoRA deltas are pure linear corrections to frozen base weights: dW = (alpha/r) * A @ B.
No activation function intervenes in the delta path. Therefore:
  (W_base + dW_shared + dW_unique_k) @ x = (W_base + dW_k) @ x

The decomposition is exact in both weight space AND function space.

**Protocol:**
1. Pretrain base GPT on joint data (300 steps)
2. Fine-tune LoRA adapters (rank 8) per domain, freezing base (300 steps)
3. Extract full deltas: dW_k = (alpha/r) * A_k @ B_k
4. Decompose: shared = mean(dW_k), unique_k = dW_k - shared
5. Compose: shared baked into base (always-on) + route unique deltas per token

---

## Lineage in the Arena

```
gpt
 `-- lora_gpt (LoRA adapters on MLP)
      `-- lora_procrustes (shared/unique decomposition of LoRA deltas)
```

Parent killed experiment:
```
capsule_moe
 `-- procrustes_decomp (KILLED: +5.7% vs joint, ReLU breaks decomposition)
```

---

## Key References

- **Original Procrustes decomposition** (exp3): killed because ReLU breaks
  weight-space decomposition. This experiment is the linear resurrection.
- **TIES-Merging** (Yadav et al., NeurIPS 2023): trims small deltas, resolves
  sign conflicts. Related approach to handling delta interference.
- **DARE** (Yu et al., 2023): drops and rescales delta parameters before merging.
  Complementary sparsification technique.
- **InfLoRA** (Liang & Li, 2024): orthogonal subspace constraints for LoRA
  continual learning. Enforces what we observe naturally (cos ~ 0.014).
- **LoRA** (Hu et al., ICLR 2022): the foundational low-rank adaptation method.

---

## Empirical Results

### Decomposition Analysis (3-seed aggregate)

| Metric | Mean | Range |
|--------|------|-------|
| Shared fraction of delta norm | 50.3% | 50.2% - 50.4% |
| Inter-domain cosine similarity | 0.014 | 0.008 - 0.018 |
| Max reconstruction error | 2.98e-08 | (numerically exact) |
| Linearity verification (max output diff) | <1e-05 | (confirmed exact) |

**The shared fraction is remarkably stable at ~50%.** This means exactly half of the
LoRA delta norm is shared between domains, and half is unique. The inter-domain
cosine similarity near zero confirms that LoRA deltas are naturally orthogonal
(consistent with prior findings in this project).

### Composition Quality (3-seed aggregate)

| Method | Avg Val Loss | vs Joint | vs Concat+Cal |
|--------|-------------|----------|---------------|
| Joint training | 0.5180 | baseline | -0.8% |
| Task arithmetic | 0.5249 | +1.3% | +0.5% |
| Shared only | 0.5249 | +1.3% | +0.5% |
| **Concat + calibrated** | **0.5224** | **+0.8%** | **baseline** |
| Concat + uniform | 0.5262 | +1.6% | +0.7% |
| **Decomp + calibrated** | **0.5225** | **+0.9%** | **+0.0%** |
| Decomp + uniform | 0.5262 | +1.6% | +0.7% |

### Kill Threshold Checks

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Decomp+cal vs concat+cal | +0.0% | <3% | **PASS** |
| Shared fraction | 50.3% | >10% | **PASS** |

---

## Analysis

### 1. Linearity Verification Confirmed

The linearity property is verified empirically: reconstructing domain-A weights as
base + shared + unique_A produces outputs within 1e-05 of base + delta_A. This is
floating-point noise. The decomposition is mathematically exact.

This is the fundamental difference from the killed exp3: capsule groups have ReLU
inside, breaking decomposition. LoRA deltas are pure linear, preserving it.

### 2. Decomposed == Concatenated at N=2

At N=2 domains, the decomposed model is functionally equivalent to the concatenated
model. This is an algebraic identity: for routing weights (w_A, w_B) summing to 1:

  shared + w_A * unique_A + w_B * unique_B = w_A * delta_A + w_B * delta_B

The decomposition provides no advantage at N=2 because the unique components are
simply +/- half the difference between the two deltas. Any linear combination of
shared + unique is also a linear combination of the original deltas.

### 3. Task Arithmetic == Shared Only (at N=2)

With two domains, the shared delta (mean of deltas) is identical to task arithmetic
with lambda=0.5. This is why both methods produce identical results (0.5249).

### 4. LoRA Deltas are Naturally Orthogonal

The cosine similarity between domain deltas is 0.014 (essentially zero). This
confirms the finding from earlier capsule experiments: independently trained
domain-specific parameters occupy near-orthogonal subspaces. The low cosine
means sign conflicts (which TIES-Merging addresses) are rare at this scale.

### 5. Routing Helps Minimally Over Uniform

Calibrated routing (0.5224) barely improves over uniform routing (0.5262) --
only 0.7%. This is consistent with prior findings: at micro scale with similar
domains (a-m vs n-z names), the routing signal is weak.

---

## What We Learned

### The Linearity Hypothesis is Validated

The core hypothesis -- that LoRA deltas can be decomposed exactly because they are
pure linear -- is confirmed. This was the specific failure mode of exp3, and it is
resolved here. The decomposition is exact in both weight space and function space.

### N=2 is Trivial for Decomposition

The N=2 case is algebraically degenerate: decomposition reduces to a change of
basis in a 2D space. The real test requires N >= 3 domains, where:
- The shared component captures genuine commonality across all domains
- Unique components are not simply negatives of each other
- The routing space (N unique directions) may differ from the original space (N full deltas)
- Shared knowledge is always active, reducing routing error impact

### The 50% Shared Fraction is Expected at N=2

With near-orthogonal deltas (cos ~ 0.014), the mean has approximately equal norm
to the difference. So shared/(shared+unique) approaches 50%. At N>2 with diverse
domains, the shared fraction would likely decrease (less commonality), making the
decomposition more informative.

---

## Micro-Scale Limitations

1. **Only N=2 domains tested.** At N=2, decomposition is algebraically trivial.
   The meaningful test is N >= 3, where shared structure is non-degenerate.

2. **Similar domains.** a-m and n-z names share character distributions. With
   truly distinct domains (Python vs JavaScript, or code vs prose), the shared
   fraction and orthogonality structure could differ substantially.

3. **Small LoRA rank (r=8).** Higher ranks might show different decomposition
   properties. The rank constrains the subspace of possible deltas.

4. **Short fine-tuning (300 steps).** Longer training could produce larger deltas
   with different shared/unique ratios.

5. **MLP-only LoRA.** Attention layers are frozen entirely. Adding LoRA to
   attention might change the decomposition structure.

---

## What Would Kill This

### At Micro Scale
- **N >= 3 domains with decomposed >3% worse than concatenated.** If the
  decomposition breaks down with more domains, the approach is not useful.
- **Shared fraction drops below 10% with diverse domains.** Would mean there
  is no meaningful shared structure to extract.

### At Macro Scale
- **Decomposed routing quality degrades with real domains (code vs prose).**
  If the unique components are too large relative to shared, the always-on
  shared component could dominate routing incorrectly.
- **Routing overhead exceeds savings from shared component.** If routing N
  unique deltas is not cheaper than routing N full deltas, there is no practical
  benefit.
- **Scale breaks near-orthogonality.** If LoRA deltas at d=4096 with BPE tokens
  have high cosine similarity, sign conflicts emerge and simple mean
  decomposition is insufficient (would need TIES-like conflict resolution).

---

## Next Steps

1. **exp_lora_rank_capacity**: Test how LoRA rank affects decomposition quality.
   Low rank may constrain shared subspace extraction.

2. **N >= 3 decomposition**: The critical test. Use quintary domain split
   (5 domains) to verify decomposition is non-trivial and useful.

3. **SVD-based decomposition**: Instead of simple mean, use SVD to find the
   principal shared subspace across all deltas. This is the "proper" Procrustes
   approach via Grassmann manifold alignment.

---

## Artifacts

- `micro/models/lora_procrustes/` -- model, tests, MATH.md, PAPER.md
- Parent model: `gpt` (dense GPT baseline)
- LoRA params: 20,480 per domain (rank 8, 4 layers, fc1+fc2)
- Total experiment time: ~8 minutes (3 seeds)
