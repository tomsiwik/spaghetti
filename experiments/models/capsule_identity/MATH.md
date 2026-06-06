# Capsule Identity Tracking Across Composition: Mathematical Foundations

## 1. Problem Statement

Experiment 10 (pruning_controls) established that 87% of capsule death in
composed models is training-induced, not composition-specific. But that was
an **aggregate** measurement: it compared death *rates* (percentages), not
death *identities* (which specific capsules).

The critical question:

**Q: When you profile which capsules are dead in single-domain models vs
composed models, what is the Jaccard overlap of those dead sets? Are the
SAME capsules dead, or does composition create novel death patterns?**

Three possible outcomes:
- **High overlap (J > 0.85)**: The same capsules die in both settings.
  Composition is transparent to death identity. Pre-composition profiling
  is sufficient.
- **Moderate overlap (0.50 < J < 0.85)**: Most dead capsules are shared,
  but composition creates some novel deaths. Post-composition profiling
  adds marginal value.
- **Low overlap (J < 0.50)**: Composition fundamentally reshuffles which
  capsules die. Pre-composition profiling is unreliable.

---

## 2. Notation

All notation follows pruning_controls/MATH.md and capsule_revival/MATH.md.

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
L         -- number of transformer layers (4 at micro scale)
P_total   -- total capsules per single-domain model = P * L = 512

a_i in R^d   -- detector vector for capsule i (row i of matrix A)
b_i in R^d   -- output vector for capsule i (column i of matrix B)

f_i^{single}  -- activation frequency of capsule i in single-domain model
f_i^{comp}    -- activation frequency of capsule i in composed model

D^{single}_k = {(l, c) : f_{l,c}^{single_k} = 0}
    -- dead set of domain k's model on own-domain data

D^{comp} = {(l, c) : f_{l,c}^{comp} = 0}
    -- dead set of composed model on joint data
```

In the composed model, domain A's capsules occupy indices [0, P-1] and
domain B's capsules occupy indices [P, 2P-1] in each layer. We define:

```
D^{comp}_A = {(l, c) : (l, c) in D^{comp}, c < P}    -- A-half of composed dead set
D^{comp}_B = {(l, c-P) : (l, c) in D^{comp}, c >= P}  -- B-half (re-indexed)
```

---

## 3. Set Similarity Metrics

### 3.1 Jaccard Similarity

```
J(A, B) = |A & B| / |A | B|
```

Properties:
- J = 1.0: identical sets
- J = 0.0: completely disjoint sets
- Symmetric: J(A, B) = J(B, A)
- Sensitive to size differences: if |A| != |B|, J is penalized

### 3.2 Overlap Coefficient

```
OC(A, B) = |A & B| / min(|A|, |B|)
```

Properties:
- OC = 1.0: the smaller set is a subset of the larger
- Less sensitive to size differences than Jaccard
- If single-domain dead set is a subset of composed dead set, OC = 1.0
  even if Jaccard < 1.0

This distinction matters because composition is expected to INCREASE death
(adding capsules from another domain that don't fire for the first domain's
data). So |D^{comp}_A| >= |D^{single}_A| is expected. Overlap coefficient
captures whether the single-domain dead set is *contained in* the composed
dead set, even if the composed set is larger.

### 3.3 Dice Coefficient

```
DC(A, B) = 2|A & B| / (|A| + |B|)
```

Harmonic mean of the two inclusion ratios. Less commonly used but provided
for completeness.

---

## 4. Death Decomposition

### 4.1 Four Categories

For domain A's capsules, each capsule falls into exactly one category:

```
Category                        Count    Meaning
------                          -----    -------
Dead in BOTH                    n_BB     Training-induced death (preserved)
Dead ONLY in single             n_SO     Revived by composition
Dead ONLY in composed           n_CO     Killed by composition
Alive in BOTH                   n_AA     Alive (preserved)

n_BB + n_SO + n_CO + n_AA = P * L
```

### 4.2 Expected Values Under Exp 10's Aggregate Finding

Exp 10 found that 87% of composed death is training-induced. If this holds
at the per-capsule level:

```
E[n_BB] ~ 0.87 * |D^{comp}_A|     (most composed-dead were already dead)
E[n_CO] ~ 0.13 * |D^{comp}_A|     (composition kills ~13% of composed-dead)
E[n_SO] ~ small                     (few capsules revive under composition)
```

At our micro scale (P*L = 512, ~55% single-domain death, ~63% composed):

```
E[|D^{single}_A|] ~ 0.55 * 512 = 282
E[|D^{comp}_A|]   ~ 0.63 * 512 = 323
E[n_BB]            ~ 0.87 * 323 = 281
E[n_SO]            = 282 - 281  = 1
E[n_CO]            = 323 - 281  = 42
```

Expected Jaccard:
```
J = n_BB / (n_BB + n_SO + n_CO) = 281 / (281 + 1 + 42) = 281 / 324 = 0.867
```

Expected Overlap Coefficient:
```
OC = n_BB / min(282, 323) = 281 / 282 = 0.996
```

### 4.3 Null Model (Independent Death)

If death at each capsule were independent between settings with probabilities
p_single and p_composed:

```
E[n_BB] = P_total * p_single * p_composed
E[n_SO] = P_total * p_single * (1 - p_composed)
E[n_CO] = P_total * (1 - p_single) * p_composed
E[n_AA] = P_total * (1 - p_single) * (1 - p_composed)
```

At p_single = 0.55, p_composed = 0.63, P_total = 512:
```
E[n_BB] = 512 * 0.55 * 0.63 = 177
E[J_null] = (0.55 * 0.63) / (0.55 + 0.63 - 0.55 * 0.63) = 0.347 / 0.834 = 0.416
```

A measured Jaccard significantly above 0.416 confirms that death identity
is NOT independent between settings -- the same capsules are deterministically
targeted by the death mechanism.

---

## 5. Why Composition Mostly Preserves Death Identity

### 5.1 The Shared Input Distribution Argument

In the single-domain model, capsule i fires when:

```
a_i^T x > 0    (ReLU condition)
```

where x is the hidden representation after attention. In the composed model,
the hidden representation x' differs from x only in the residual MLP
contribution from the *other* domain's capsules:

```
x'_l = x_l + sum_{c in B_alive} b_c * relu(a_c^T * norm(x_l))
```

where B_alive is the set of alive capsules from the other domain. But:

1. Attention is shared (frozen during fine-tuning), so the attention output
   is identical.
2. The MLP residual contribution is additive and bounded by the norm of
   the other domain's alive outputs.
3. Dead capsules (the majority) contribute exactly zero.

The net effect: x' ~ x + small_perturbation. If capsule i was dead because
a_i^T x < 0 by a large margin, a small perturbation to x will not revive it.

### 5.2 Why Some Capsules Are Killed by Composition

The ~29 capsules killed by composition per domain half (6% of the pool)
are those near the death boundary: a_i^T x ~ 0. The perturbation from the
other domain's capsules pushes them past zero.

### 5.3 Why Very Few Capsules Are Revived by Composition

Only ~4 capsules per domain are revived by composition. This asymmetry
(29 killed vs 4 revived) reflects that composition adds signal from a
different domain, which on average INCREASES the diversity of inputs to
each capsule -- making it harder (not easier) for dead capsules to activate,
because the perturbation is uncorrelated with the dead capsule's detector
direction.

---

## 6. Experimental Design

### 6.1 Protocol

1. Pretrain base model on ALL data (300 steps, shared attention + MLP)
2. Fine-tune only MLP weights per domain (attention frozen, 200 steps)
3. Profile each single-domain model on own-domain validation data (20 batches x 32)
4. Also profile each single-domain model on cross-domain data (control)
5. Compose by concatenating A and B weight matrices from both domains
6. Profile composed model on joint validation data
7. Also profile composed model on each domain separately
8. Compute Jaccard, overlap coefficient, and decomposition metrics
9. Repeat for 3 seeds (42, 123, 7)

### 6.2 Key Controls

**Cross-domain profiling**: Profiling the same single-domain model on
data from the other domain. This measures how much the dead set changes
due to input distribution alone (without composition).

**Composed per-domain profiling**: Profiling the composed model on data
from only one domain. This isolates whether the composed model's A-half
matches the single-domain model when seeing the same data distribution.

### 6.3 Metrics

Primary:
- Combined Jaccard: J(D^{single}_A | D^{single}_B, D^{comp})
- Per-domain Jaccard: J(D^{single}_A, D^{comp}_A), J(D^{single}_B, D^{comp}_B)

Secondary:
- Overlap coefficient (robust to size differences)
- Per-layer Jaccard (to detect layer-specific effects)
- Decomposition counts (n_BB, n_SO, n_CO)

---

## 7. Kill Criterion

```
Per-capsule death identity overlap < 50% (Jaccard < 0.50)
between single-domain and composed models.
```

If killed: composition creates fundamentally different death patterns.
Pre-composition profiling is unreliable. Post-composition profiling is
mandatory.

If passed: the same capsules die in both settings. Pre-composition
profiling is sufficient. This enables a more efficient pruning protocol:
profile -> prune -> compose (instead of compose -> profile -> prune).

---

## 8. Worked Numerical Example

At d=4, P=4, L=1 (4 capsules total, single layer):

### Single-domain model (domain A)
```
Capsule 0: dead   (a_0^T x < 0 by large margin)
Capsule 1: dead   (a_1^T x < 0 by large margin)
Capsule 2: dead   (a_2^T x ~ -0.01, barely dead)
Capsule 3: alive  (fires on 60% of inputs)
D^{single}_A = {0, 1, 2}
```

### Composed model (A-half, 4 capsules from A + 4 from B)
After composition, the hidden state x' = x + perturbation from B's capsules.

```
Capsule 0: dead   (a_0^T x' < 0, large margin survives perturbation)
Capsule 1: dead   (a_1^T x' < 0, large margin survives perturbation)
Capsule 2: alive  (a_2^T x' > 0, perturbation pushed past boundary = REVIVED)
Capsule 3: alive  (still fires)
D^{comp}_A = {0, 1}
```

### Metrics
```
J(D^{single}_A, D^{comp}_A) = |{0,1}| / |{0,1,2}| = 2/3 = 0.667
OC(D^{single}_A, D^{comp}_A) = |{0,1}| / min(3,2) = 2/2 = 1.000
n_BB = 2, n_SO = 1, n_CO = 0
```

The overlap coefficient is 1.0 because D^{comp}_A is a subset of D^{single}_A
(composition only revived capsule 2, didn't kill any new ones). Jaccard is
lower because the sets differ in size.

---

## 9. Assumptions

1. **Activation frequency is deterministic.** For a fixed model and fixed
   profiling dataset, the dead/alive classification is deterministic. Exp 12
   confirmed profiling noise is only 2.6-3.8% (well under the margins here).

2. **Capsule identity is preserved across composition.** In the composed
   model, capsules [0..P-1] in each layer correspond to domain A's capsules
   in the same order as in the single-domain model. The compose_relu_models()
   function concatenates A-matrices vertically and B-matrices horizontally,
   preserving index correspondence.

3. **Shared attention is identical.** Both single-domain and composed models
   share the same pretrained attention weights (frozen during fine-tuning).
   The attention output is identical for the same input, so the MLP input
   differs only by the other domain's residual contribution.

4. **Binary dead/alive threshold (f=0).** Same as Exp 9/10/17/18. "Nearly
   dead" capsules (0 < f < 0.01) are classified as alive. This is conservative
   and may undercount the overlap (a capsule at f=0.001 in single and f=0 in
   composed would be classified as alive/dead, reducing apparent overlap).

5. **Two-domain composition.** The experiment uses N=2 domains. At N=5+,
   the perturbation from other domains' capsules is larger, which could
   decrease overlap. The N=2 result is a best-case for overlap preservation.
