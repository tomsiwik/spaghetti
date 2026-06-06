# Split Domain Specialization: Mathematical Foundations

## 1. Setting

We build on the split_leaf_actual MATH.md. All notation carries forward.

Additional notation:
```
D_A, D_B       = domain-specific training sets (binary split: a-m, n-z)
f_c0, f_c1     = children outputs after domain-specific fine-tuning
A_A(c0)        = active capsule set of child 0 on domain A data
A_B(c0)        = active capsule set of child 0 on domain B data
J(S1, S2)      = Jaccard similarity: |S1 & S2| / |S1 | S2|
S_conv          = convergence step (first step reaching 99% of final quality)
```

---

## 2. Domain Specialization Hypothesis

### 2.1 Claim

Split children inherit feature detectors from the parent that are already
partially aligned with the data distribution. When fine-tuned on different
domains, these inherited detectors should:

1. **Converge faster** than randomly-initialized detectors, because they
   start closer to the domain-specific optimum.
2. **Specialize differently**, with each child's capsules activating on
   domain-specific patterns (low cross-domain Jaccard).

### 2.2 Convergence Speed Formalization

Let L_split(s) and L_indep(s) be the validation loss at step s for split
and independent conditions respectively.

Define convergence step:
```
S_conv(L) = min{s : L(s) <= 1.01 * L(S_final)}
```

Kill criterion KC1: split must converge >10% faster:
```
PASS iff (S_conv(L_indep) - S_conv(L_split)) / S_conv(L_indep) > 0.10
```

### 2.3 Domain Separation Formalization

For a child c trained on alternating domain data, define:
```
A_d(c, l) = {j in [0, n_c/2) : exists (x,t) in D_d such that ReLU(a_{c,j}^T x_{l,t}) > 0}
```

The set of capsule indices in child c, at layer l, that activate for at
least one token from domain d.

Cross-domain Jaccard for child c at layer l:
```
J_l(c) = J(A_{D_A}(c, l), A_{D_B}(c, l)) = |A_A & A_B| / |A_A | A_B|
```

Mean across layers and children:
```
J_combined = (1/2L) * sum_{c in {0,1}} sum_{l=0}^{L-1} J_l(c)
```

Kill criterion KC2: J_combined >= 0.95 means children use nearly identical
capsules for both domains (no specialization).

---

## 3. Why Domain Separation May Fail at Micro Scale

### 3.1 Capacity Saturation Argument

With n_c/2 = 16 capsules per child and d = 64, the ReLU activation pattern
depends on sign(a_j^T x). Each capsule partitions R^64 into two half-spaces.

For domain specialization to emerge in the Jaccard metric, some capsules
must fire exclusively on one domain. This requires:

```
exists j : a_j^T x > 0 for all x in D_A, a_j^T x <= 0 for all x in D_B
```

At d=64, the domains (names starting a-m vs n-z) produce token embeddings
in the same 64-dimensional space with high overlap. The character-level
features (bigrams, trigrams) are largely shared. With only 16 capsules,
ALL capsules fire on both domains because there aren't enough to afford
domain-exclusive detectors.

### 3.2 Information-Theoretic Bound

The domains in the names dataset differ primarily in the first character
(which determines domain membership) but share most subsequent characters.
The mutual information between domain identity and character bigrams is
low for positions beyond the first.

```
I(domain; bigram) ~ H(first_char) / L_avg ~ log2(26) / 6 ~ 0.78 bits/position
```

This low signal means that capsule detectors optimized for next-token
prediction will largely activate identically on both domains, producing
J -> 1.0.

### 3.3 Prediction

At micro scale (d=64, n_c/2=16, character-level names):
- J_combined >> 0.95 for BOTH split and independent conditions
- Domain separation is not achievable because the domains share too many features
- Split provides no specialization advantage because there are no
  domain-exclusive features to inherit

At macro scale (d=4096, n_c/2=128, subword tokenization on distinct domains
like code vs medical):
- Domains would have genuinely distinct vocabulary and patterns
- More capsules would allow domain-exclusive detectors
- Split's inherited features from the generalist parent would provide
  a stronger initialization advantage

---

## 4. Convergence Analysis

### 4.1 Split vs Random Init at Domain Fine-tuning

The split condition starts with weights derived from a trained parent:
```
theta_split = theta_parent|_{half} + epsilon,  epsilon ~ N(0, sigma^2 I)
```

The independent condition starts from random initialization:
```
theta_indep ~ N(0, sigma_init^2 I)
```

The loss landscape difference:
```
L(theta_split) = L(theta_parent|_{half}) + O(sigma)  # near parent minimum
L(theta_indep) = L(random)                            # far from any minimum
```

However, the parent was trained on ALL data. When fine-tuning on domain A
only, the parent's features for domain B are irrelevant overhead. The split
child inherits a mixture of A-relevant and B-relevant features (random
partition), giving only ~50% useful initialization.

### 4.2 Expected Speedup

With 50% relevant features inherited:
```
S_conv(split) ~ S_conv(indep) * (1 - alpha * 0.5)
```

where alpha is the fraction of convergence attributable to initialization
(vs gradient optimization). At micro scale with short training (400 steps),
alpha is small because convergence is fast for both conditions.

The >10% speedup threshold requires alpha > 0.20, which is not expected
at micro scale where both conditions converge within 100-200 steps.

---

## 5. Worked Example (d=8, n_c=4, split to n_c/2=2)

Parent leaf with 4 capsules, split into two children with 2 each:
```
Child 0: detectors a_0, a_1 (from parent capsules 0,1)
Child 1: detectors a_2, a_3 (from parent capsules 2,3)
```

Domain A input x_A: a_0^T x_A = 1.2, a_1^T x_A = -0.3 -> active set = {0}
Domain B input x_B: a_0^T x_B = 0.8, a_1^T x_B = 0.5  -> active set = {0, 1}

J(child 0) = |{0}| / |{0, 1}| = 0.5 -> good separation

But this requires the domain signals to align with individual capsule
directions. With character-level names, x_A and x_B differ only in the
first character's embedding component, which is small relative to the
shared bigram/trigram structure. Both domains produce similar activation
patterns, driving J -> 1.0.

---

## 6. Assumptions

1. **Binary Jaccard captures specialization**: We measure which capsules
   fire (binary), not how much they fire. A capsule that fires weakly on
   domain B but strongly on domain A still counts as active for both.
   Frequency-weighted metrics might show more differentiation.

2. **Alternating training produces balanced exposure**: Each domain sees
   50% of training steps. The gate learns to route without explicit
   domain labels.

3. **16 capsules per child is sufficient for specialization**: This
   assumption is tested and potentially violated. With 16 ReLU capsules,
   the space of possible activation patterns is 2^16 = 65536, but the
   actual pattern space used by the data may be much smaller.

4. **Names dataset domains are distinct enough**: Binary split (a-m vs n-z)
   produces domains that differ mainly in the first character. More
   semantically distinct domains (e.g., languages, styles) might show
   stronger separation.
