# MATH.md: Integrated Serving Pipeline Correctness

## Type: Verification (Type 1)

All individual components have been proven separately. This document proves that their
composition preserves the guarantees of each component. The experiment verifies these
composition predictions.

---

## A. Failure Mode Identification

**Failure mode:** When combining block-diagonal masking, per-token MLP routing, DARE
sparsification, and ridge routing into a single pipeline, uncontrolled interactions
between components could degrade quality below what each component achieves individually.

Specifically:
1. **Routing errors propagate through DARE:** Misrouted tokens get wrong adapter + random
   sparsification, compounding degradation
2. **Block-diagonal mask + MLP routing interaction:** The mask eliminates cross-segment
   attention, but per-token MLP routing still applies different LoRA weights to tokens
   in the same attention window (within a segment). Could this create within-segment
   interference?
3. **Accumulated approximation:** Each component introduces a small gap. Do these gaps
   compound (multiplicative) or stay bounded (additive)?

---

## B. Prior Mathematical Foundations

**Theorem (RoPE relative position invariance — Su et al. 2104.09864, verified Finding #322).**
For RoPE with rotation matrices R(p), the attention score between query at position a
and key at position b satisfies: q_a^T k_b = (R(a) q)^T (R(b) k) = q^T R(a)^T R(b) k
= q^T R(b-a) k. Thus attention depends only on relative position b-a, not absolute
positions. Under block-diagonal masking, each segment's within-segment attention is
identical to computing that segment in isolation.

**Verified prediction:** bd fair gap = 0.244% (Finding #322), consistent with bf16
numerical noise. RoPE reset is a no-op (0.012 mean diff).

**Theorem (MLP token-independence — Finding #313).** For an MLP layer f(x_t) that operates
pointwise on each token position independently, the output at position t depends only on
x_t and the MLP parameters, not on other tokens. Therefore, applying adapter A to tokens
in segment A and adapter B to tokens in segment B via per-token parameter selection in the
MLP produces output identical to applying each adapter in a separate forward pass, for
same-segment tokens.

**Verified prediction:** Same-segment tokens match exactly (max diff 0.000000 in Finding
#313). Cross-segment divergence bounded at 0.61% PPL (Finding #313).

**Theorem (DARE unbiased estimator — Yu et al. 2311.03099).** For adapter parameters B,
DARE with drop probability p produces B_DARE = M * B / (1-p) where M ~ Bernoulli(1-p)
elementwise. E[B_DARE] = B, so the expected perturbation is preserved. Variance:
Var[B_DARE_ij] = p/(1-p) * B_ij^2.

**Verified prediction:** DARE at p=0.5 preserves in-distribution quality while reducing
OOD degradation from 2/5 to 1/5 domains (Finding #266).

**Theorem (Ridge regression optimality).** For X in R^{n x d} and Y in R^{n x K}, the
ridge regression solution W* = (X^T X + lambda I)^{-1} X^T Y minimizes ||XW - Y||^2 +
lambda ||W||^2 and has a unique global optimum for any lambda > 0 (since X^T X + lambda I
is positive definite). This is a closed-form solution requiring zero iterative training.

**Verified prediction:** 96-98.3% routing accuracy on 5 domains (Findings #276, #310).

---

## C. Composition Conjecture

**Conjecture 1 (Integrated pipeline correctness).** Let:
- epsilon_mask be the gap from block-diagonal masking vs segment-isolated (< 0.5%, Finding #322)
- epsilon_mlp be the gap from single-pass MLP routing vs multi-pass oracle (< 0.7%, Finding #313)
- epsilon_dare be the gap from DARE sparsification (< 5% in-distribution, Finding #266)
- epsilon_route be the routing error rate (< 4%, Finding #276)

Then the integrated pipeline PPL satisfies:

  PPL_integrated <= PPL_oracle * (1 + epsilon_mask) * (1 + epsilon_mlp) * (1 + epsilon_dare) * (1 + epsilon_route * delta_misroute)

where delta_misroute is the PPL penalty of applying the wrong domain adapter (bounded
by max domain PPL difference).

**Argument (not a proof — independence is assumed, not derived).** Each component's
effect on the log-probability is modeled as additive in log-space. Block-diagonal
masking adds epsilon_mask to each token's NLL. MLP routing adds epsilon_mlp. DARE
adds epsilon_dare. Routing errors affect epsilon_route fraction of tokens with penalty
delta_misroute. We ASSUME these are independent perturbations to the log-probability.

**Critical assumption:** Independence requires that cross-derivatives
d(epsilon_mask)/d(epsilon_dare), etc., are negligible. This is plausible because each
component operates on different aspects of the forward pass (attention mask vs MLP
weights vs weight sparsification vs token routing), but these perturbations propagate
through the same nonlinear transformer layers and are therefore coupled. Independence
is NOT proven — it is supported by prior component-level measurements showing each
component's gap is consistent regardless of other components' presence.

Under the independence assumption:

  log PPL_integrated = log PPL_oracle + log(1+e_mask) + log(1+e_mlp) + log(1+e_dare) + epsilon_route * log(delta_misroute)

For small epsilons:

  PPL_integrated / PPL_oracle ~ 1 + e_mask + e_mlp + e_dare + e_route * delta_misroute

With measured values: 0.005 + 0.007 + 0.05 + 0.04 * 0.2 = 0.070 = 7%

**Predicted bound:** The integrated pipeline should be within ~7% of segment-isolated
oracle in the worst case. The measurement (-2.8%, i.e., BETTER than oracle) contradicts
the additive degradation prediction — see PAPER.md for discussion of this sign flip.

---

## D. Predictions

### Behavioral predictions:
1. The integrated pipeline produces meaningful domain-specific responses (behavioral score > 0.2)
2. Router correctly identifies domain for >90% of queries
3. Each domain's PPL is within 10% of per-sequence baseline (single adapter, no block-diag)
4. Speed exceeds 60 tok/s (native ternary inference + addmm LoRA)

### Quantitative predictions (from Conjecture 1):
| Prediction | Source | Expected |
|-----------|--------|----------|
| BD fair gap vs isolated | Finding #322 | < 0.5% |
| MLP routing gap | Finding #313 | < 1% |
| DARE degradation (in-dist) | Finding #266 | < 5% |
| Router accuracy | Finding #276 | > 90% |
| Overall pipeline vs per-seq | Conjecture 1 | < 10% |
| Speed with adapter | VISION.md | > 60 tok/s (97 tok/s proven) |

### Kill criteria derived from proof:
- **K818:** Integrated pipeline worse than per-sequence baseline. Since per-sequence
  baseline already includes routing overhead and single-adapter quality, the integrated
  pipeline (which uses the SAME adapters + block-diagonal for isolation) should be
  within 10% by Conjecture 1. PASS threshold: integrated PPL within 10% of per-sequence.
- **K819:** Speed < 60 tok/s. The pipeline adds block-diagonal mask creation and router
  inference, both sub-millisecond operations. The dominant cost is still model forward
  pass. PASS threshold: >= 60 tok/s.

---

## E. Assumptions and Breaking Conditions

1. **Adapters are SFT-trained with Grassmannian skeleton.** If adapters are trained
   differently, orthogonality guarantees may not hold. Breaking: adapter cosine > 0.05.
2. **Domains are separable in hidden-state space.** If domains overlap significantly,
   router accuracy drops. Breaking: accuracy < 70%.
3. **Block-diagonal mask preserves within-segment attention.** Verified in Finding #322
   for K=2 segments. Assumption: holds for K>2 (the mask construction is identical).
4. **DARE seed independence.** The specific random mask should not matter for expected
   quality. Breaking: variance across seeds > 10%.

---

## F. Worked Example (K=2 segments, d=2560)

Consider a mixed input of 256 tokens: tokens 0-127 are medical, tokens 128-255 are code.

1. **Router:** Forward pass on calibration data, fit W = (X^TX + I)^{-1} X^TY. For a new
   query, h = mean_pool(hidden_states), domain = argmax(h @ W). With 96% accuracy, the
   router correctly identifies "medical" and "code".

2. **Block-diagonal mask:** Create 256x256 additive mask where cross-segment (i < 128,
   j >= 128 or vice versa) entries are -inf. Within-segment entries are standard causal mask.

3. **Per-token MLP routing:** For each transformer layer, compute MLP output with adapter A
   (medical) for tokens 0-127 and adapter B (code) for tokens 128-255 using mx.where masking.

4. **DARE:** Before routing, sparsify each adapter B with Bernoulli(0.5) mask, rescale by 2.
   Expected perturbation unchanged. Variance doubles but in-distribution quality preserved.

5. **PPL:** Compute per-token NLL. Exclude boundary token (token 128, which predicts the
   first code token from medical context). The "fair" PPL should be within 2% of computing
   each segment independently.

---

## G. Complexity and Architecture Connection

| Component | FLOPs | Memory | Scaling |
|-----------|-------|--------|---------|
| Router calibration | O(N * T * d^2) one-time | O(d^2) for W | Per-domain: O(T * d^2) |
| Router inference | O(T * d + d * K) | O(d * K) | Negligible (0.46% of inference) |
| Block-diag mask | O(T^2) construction | O(T^2) | Additive to attention cost |
| Per-token MLP routing | O(T * d * r) per layer | O(K * L * r * d) for K adapters | 2x LoRA compute (both adapters) |
| DARE sparsification | O(params) one-time | O(params) for mask | Per-adapter: O(rank * d) |
| Total per-token overhead | +2x LoRA MLP (vs single adapter) | +1 adapter in memory | Dominated by base model |

Connection to production: This is the "pre-merge for always-on + runtime LoRA for routed"
architecture from VISION.md, but with all components integrated.

---

## Self-Test

1. What is the ONE mathematical property that makes the failure mode impossible?
   **Conjectured additive independence of perturbation sources:** Each component (mask,
   MLP routing, DARE, router) is CONJECTURED to contribute an independent bounded
   perturbation to log-probability. This is supported by prior component-level measurements
   but NOT formally proven. The measurement (-2.8% improvement over oracle, 18/18 samples)
   contradicts the additive degradation framework, suggesting the independence model is
   incomplete.

2. Which existing theorem(s) does the conjecture build on?
   - Su et al. (2104.09864) Theorem 1: RoPE relative position invariance
   - Finding #322: block-diagonal gap < 0.5%
   - Finding #313: MLP token-independence, single-pass gap < 0.7%
   - Yu et al. (2311.03099): DARE unbiased estimator
   - Ridge regression: closed-form global optimum

3. What specific numbers does the conjecture predict?
   - Integrated pipeline within 10% of per-sequence baseline (worst case)
   - Within ~7% of segment-isolated oracle (worst case)
   - Speed > 60 tok/s (routing overhead < 1%)
   - Router accuracy > 90%
   **NOTE:** Measurement was -2.8% (BETTER than oracle), contradicting the predicted
   direction. The bound is satisfied but vacuously so.

4. What would FALSIFY the conjecture?
   - If integrated PPL > 1.1x per-sequence for >50% of domains
   - If components interact non-additively (e.g., DARE + block-diag creates
     super-linear degradation)
   - If the -2.8% improvement is shown to be a measurement artifact (different
     code paths between isolated and integrated evaluation)

5. How many hyperparameters does this approach add?
   **0 new.** All components use previously validated settings: DARE p=0.5, ridge
   lambda=1.0, LORA_SCALE=20.0. These were set in prior experiments.

6. Hack check: Am I adding fix #N to an existing stack?
   **No.** This is integration of 5 independently proven components. Each solves a
   different problem (masking: isolation, routing: selection, DARE: OOD robustness,
   adapter: specialization, router: detection). No component is a fix for another's
   failure.
