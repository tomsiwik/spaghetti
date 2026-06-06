# Cross-Domain PPL Matrix: Adapter Specificity Under Grassmannian Initialization

## A. Failure Mode Identification

**The degenerate behavior:** All 24 adapters produce identical PPL improvement regardless of which domain they are applied to. If this holds, then:
1. Routing is unnecessary (any adapter works equally well on any domain)
2. The Grassmannian initialization provides no domain-specific structure
3. Adapter training learned only a generic "quality improvement" independent of training data

**Why this is a real risk:** Finding #200 showed routed PPL (6.32) equals oracle PPL (6.32) despite only 41% routing accuracy. This is consistent with adapter interchangeability: if all adapters give the same PPL on a given domain, then routing mistakes have zero cost.

## B. The Right Question

**Wrong question:** "How do we improve routing accuracy?"
**Right question:** "Do the trained LoRA B-matrices encode domain-specific information, or only a generic quality improvement?"

This is an empirical question answerable by a cross-domain evaluation matrix, as used in LoRAuter (arXiv:2601.21795) for task affinity analysis.

## C. Prior Mathematical Foundations

### C1. LoRA Perturbation Structure

A LoRA adapter modifies a linear layer W as:
```
W' = W + s * A * B
```
where A is (d_in, r), B is (r, d_out), s is a scalar scale.

In our Grassmannian setup, A is shared structure (frozen, initialized to maximize
pairwise distance on the Grassmannian manifold), and B is trained per domain.

### C2. Cross-Domain Performance Matrix (LoRAuter, arXiv:2601.21795)

The cross-domain performance matrix M has entries:
```
M[i,j] = PPL(base + adapter_j, validation_data_i)
```

**Diagonal dominance ratio** for domain i:
```
DDR_i = mean_{j != i}(M[i,j]) / M[i,i]
```
If DDR_i > 1, the correct adapter outperforms the average of wrong adapters.

**Global diagonal dominance:**
```
DDR = (1/N) * sum_i DDR_i
```

### C3. Domain Specificity via Information Theory

If adapter j encodes domain-specific information, then applying it to domain i != j
should increase cross-entropy (PPL) compared to the matched adapter.

Let L(i,j) = cross-entropy loss of adapter j on domain i's validation set.

**Theorem (Adapter Specificity).** If the training procedure minimizes
L(i,i) for each domain i independently, and the domains have non-trivial
distributional distance (D_KL(P_i || P_j) > 0 for i != j), then
under sufficient model capacity and training convergence:

```
E[L(i,j)] > E[L(i,i)]  for j != i
```

*Proof sketch.* Each adapter B_i is trained to minimize L(i,i) = E_{x~P_i}[-log p(x | W + s*A_i*B_i)].
At convergence, B_i is a local minimum of L(i,i). For j != i, B_j was optimized for P_j, not P_i.
By the assumption that P_i and P_j are distinguishable (positive KL divergence), the optimal
B for P_i differs from the optimal B for P_j. Therefore L(i,j) > L(i,i) on average.

The magnitude of the gap depends on:
1. How different P_i and P_j are (KL divergence)
2. How much capacity the rank-r adapter has to specialize (higher r = more specialization)
3. Whether A_i and A_j are different enough to support distinct specializations

**QED (sketch)**

This is a Type 2 guided exploration: the theorem guarantees specialization EXISTS
under distributional difference, but the DEGREE of specialization is unknown.
The experiment discovers the empirical degree.

## D. Predictions

### D1. Behavioral Predictions
1. **Domain-specific adapters should outperform mismatched adapters** on average
   (follows from the theorem above, given domains are semantically distinct)
2. **Some domain pairs will show near-interchangeability** (domains with small
   KL divergence, e.g., politics/economics, sociology/psychology)
3. **The degree of specificity should correlate with domain distinctness**
   (medical vs cooking should show large gap; politics vs economics small gap)

### D2. Quantitative Predictions (derived from prior result)
From Finding #200, avg oracle PPL = 6.32, avg base PPL = 10.07 (34.8% improvement).

- **Prediction 1 (DDR):** If adapters are specialized, DDR > 1.05.
  If adapters are interchangeable, DDR ~ 1.0.
  Prior: DES-MoE (arXiv:2509.16882) shows 43-76% task drops from wrong routing,
  suggesting DDR could be 1.4-1.8.

- **Prediction 2 (Diagonal wins):** At least 18/24 domains should show diagonal
  dominance (own adapter beats mean of others), since all 24 domains showed
  > 17% improvement with own adapter.

- **Prediction 3 (Base vs wrong adapter):** Even wrong adapters should improve
  over base PPL (10.07) because the adapter adds learned structure that partially
  generalizes. Expected: wrong-adapter PPL in range [6.5, 9.0].

## E. Assumptions and Breaking Conditions

1. **Assumption: Correct A-matrix loading.** If the Grassmannian A matrices are
   not loaded correctly, all results are meaningless. (This was the bug in Finding #201.)
   **Verification:** Reuse exact `set_lora_a()` logic from fix_grassmannian_loading_retest.

2. **Assumption: Sufficient validation data.** With 20 batches of 256 tokens per
   domain, we have ~5K tokens per cell. PPL estimates may have variance ~5%.
   **Breaking:** If variance > DDR gap, we cannot distinguish diagonal dominance.

3. **Assumption: Independent training.** Each adapter was trained independently on
   its domain. If there was cross-domain leakage during training, adapters may
   not be specialized.

## F. Worked Example (2 domains, d=16, r=4)

Consider 2 domains: "medical" (domain 0) and "code" (domain 1).

Base PPL on medical validation = 6.75, on code validation = 5.73.

With adapter_0 (trained on medical):
- PPL(medical, adapter_0) = 3.54 (diagonal)
- PPL(code, adapter_0) = ? (off-diagonal, to be measured)

With adapter_1 (trained on code):
- PPL(medical, adapter_1) = ? (off-diagonal, to be measured)
- PPL(code, adapter_1) = 3.55 (diagonal)

If specialized: PPL(code, adapter_0) > 3.55 and PPL(medical, adapter_1) > 3.54
If interchangeable: PPL(code, adapter_0) ~ 3.55 and PPL(medical, adapter_1) ~ 3.54

The matrix reveals whether each adapter "knows" only its own domain.

## G. Complexity and Architecture Connection

**Measurement complexity:** O(N^2 * V * T) where N=24 domains, V=20 validation
batches, T=256 tokens per batch. Total: 24 * 24 * 20 = 11,520 forward passes.

**Memory efficiency:** Load model once, swap adapters by changing A (skeleton lookup)
and B (adapter.npz load). Model weight: ~5GB in bfloat16. Adapter overhead: negligible
(B is rank-16, ~50KB per layer per domain).

**Runtime estimate:** Each forward pass ~20ms. Total: 11,520 * 20ms ~ 4 minutes.
With adapter swapping overhead: ~10-15 minutes total.

---

## Self-Test

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   If domains are distributionally distinct and adapters are trained to convergence,
   domain-specific adapters MUST outperform mismatched adapters (by the optimality
   of each B_i for its own domain's loss).

2. **Which existing theorem(s) does the proof build on?**
   Optimality of gradient-descent solutions for convex surrogate losses (standard
   optimization theory). Cross-task evaluation methodology from LoRAuter (2601.21795).

3. **What specific numbers does the proof predict?**
   DDR > 1.05, diagonal wins >= 18/24, wrong-adapter PPL in [6.5, 9.0].

4. **What would FALSIFY the proof?**
   If DDR < 1.05 and fewer than 12/24 domains show diagonal dominance, then either:
   (a) rank-16 adapters lack capacity for domain specialization, or
   (b) the Grassmannian A matrices are not sufficiently domain-specific, or
   (c) training did not converge to domain-specific optima.

5. **How many hyperparameters does this approach add?**
   0. This is a measurement experiment, not a new mechanism.

6. **Hack check:** No fixes being added. Pure measurement.
