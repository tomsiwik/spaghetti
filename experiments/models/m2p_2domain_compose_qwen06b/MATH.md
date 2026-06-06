# MATH.md: 2-Domain M2P Composition on Qwen3-0.6B — Math + Code with TF-IDF Routing

## TYPE: verification (Type 1)
## PROVEN FRAMEWORK:
  - Theorem 5 (v3/v4 MATH.md) — functional LoRA forward gives non-zero gradients
  - Finding #381 (supported) — Grassmannian A-slot composition, grad_norm > 0, TF-IDF 100%
  - Finding #389 (supported) — TF-IDF routing 100% on math/code/text real NLP domains
  - Finding #378 (supported) — M2P v4 math quality_ratio=1.433 (M2P > base, comparable SFT)

---

## A. Failure Mode Identification

### What degenerate behavior could occur?

**Failure 1 (Cross-domain interference):** B_math and B_code might share weight-space
directions, so applying B_math on a code prompt corrupts code generation (and vice versa).

This is prevented by Grassmannian A-matrix orthogonality (Theorem 1): A_math^T A_code = 0
exactly, so ΔW_math and ΔW_code have zero Frobenius inner product — they cannot interfere.

**Failure 2 (Router distribution shift):** A router trained on base-model hidden states
fails under composed adapters due to covariate shift.

Impossible by construction (Theorem 2): TF-IDF operates on raw input text, before any
model forward. The routing decision is invariant to the composed adapter state.

**Failure 3 (Code adapter insufficient capacity):** M2P rank-4 B-matrices might lack the
expressiveness to steer Qwen3-0.6B toward code output format.

Bounded by Theorem 3 (rank-r expressiveness via JL-lemma). The model already knows Python
from pretraining — M2P only needs to activate the code output mode, not learn Python from scratch.

**Root-cause framing (SIGReg):** The disease is weight-space interference. The structural
fix is not "reduce interference" but "make interference geometrically impossible via
orthogonal A-slot assignment." Three symptoms (cross-task activation, routing mismatch,
capacity) all follow from a single structural property.

---

## B. Prior Mathematical Foundations

### B1. Grassmannian A-Slot Orthogonality (cited: Finding #381, Finding #50)

**Procedure:** For N=2 domains, rank r=4, dimension d=1024 (q_proj input):

```
X ∈ R^{d × 2r}  via X_ij ~ N(0,1)
Q, _ = QR(X)
A_math = Q[:, 0:r]   (d, r)
A_code = Q[:, r:2r]  (d, r)
```

**Orthogonality identity:** A_math^T A_code = Q^T Q [0:r, r:2r] = I_{2r}[0:r, r:2r] = 0.

**Frobenius inner product:** Let ΔW_i = A_i @ B_i ∈ R^{d_out × d_in}.

⟨ΔW_math, ΔW_code⟩_F = Tr(B_math^T A_math^T A_code B_code) = Tr(B_math^T · 0 · B_code) = 0.

This holds for ANY B_math, B_code — the orthogonality is unconditional on the learned weights.

**Reference:** Finding #50 verified max|cos|=1e-08 for N=5 at d=256, r=4. Finding #381
confirmed Grassmannian orthogonality holds on Qwen3-0.6B d=1024, r=4.

### B2. TF-IDF Routing Invariance (cited: Finding #354, Finding #389)

**Theorem (LoraRetriever, arXiv:2402.09997):** For any routing function f: X → {0,1}^N
computed on input text X directly (before model forward), the router distribution is
invariant to the adapter applied at inference time.

**Why math vs. code is separable:** From Finding #389 (exp_tfidf_routing_real_text):
  cos(math, code) = 0.190 — nearly orthogonal TF-IDF centroids.
  Perfect routing (100%) was achieved with nearest-centroid TF-IDF classifier.

**Prediction for this experiment:** TF-IDF routing accuracy ≥ 97% on math vs. code
(K955 requires ≥ 85% — predicted with large margin).

**Discriminating features:**
  - Math: "how many", "each", "total", "sell", "per", number tokens
  - Code: "write", "python", "function", "called", "takes", "returns"

### B3. Functional LoRA Forward (cited: Theorem 5, v3/v4 MATH.md, Finding #376)

**Theorem 5 (v3):** If B is passed as a tensor argument (not a module attribute),
∂L/∂θ_M2P ≠ 0 for any non-degenerate input. This is the root fix for the v2 kill.

**Consequence:** Under composed adapter B = B_math (selected by TF-IDF), the gradient
flows through M2P_math's parameters. Similarly for M2P_code. No interference in the
autodiff graph.

### B4. Rank-r Expressiveness for Code Steering (cited: arXiv:2106.09685 LoRA)

**JL-Lemma application (from Finding #387):** The intrinsic dimensionality of a code
domain is bounded. For Python function generation (a structured task), the adaptation
signal is low-rank by construction — the model already has Python representations.

**Theorem (LoRA, Hu et al.):** For pre-trained weights W ∈ R^{d_out × d_in}, the
rank of the task-specific adaptation ΔW* satisfies rank(ΔW*) << min(d_out, d_in) for
most NLP tasks. r=4 captures the dominant adaptation direction.

**Prediction:** Code M2P with r=4 achieves code_single_acc significantly above base_acc
(base generates valid Python inconsistently; code M2P steers reliably toward def/return format).

---

## C. Theorems and Predictions

### Theorem 1 (Domain Isolation)

**Statement:** Under Grassmannian A-slot assignment, the composed adapter
B_composed = B_math (for math inputs) or B_code (for code inputs) via TF-IDF routing
satisfies ⟨ΔW_math, ΔW_code⟩_F = 0 for any trained B_math, B_code.

**Proof:** By the QR construction (B1 above): A_math^T A_code = 0 exactly in float32.
For any B_math ∈ R^{r × q_proj_out}, B_code ∈ R^{r × q_proj_out}:
  ⟨ΔW_math, ΔW_code⟩_F = Tr(B_math^T (A_math^T A_code) B_code) = Tr(B_math^T · 0 · B_code) = 0. □

**Prediction:** Routed composition does NOT degrade below 80% of best-single on either domain
(quality_ratio ≥ 0.80 for both math and code, K954).

### Theorem 2 (Router Invariance)

**Statement:** TF-IDF routing accuracy on math vs. code inputs is invariant to whether
the base, math-only, code-only, or composed adapter is applied.

**Proof:** TF-IDF computes f(X) = argmax_i cos(tfidf(X), centroid_i), where X is the raw
input text and centroids are computed from training prompts. f(X) depends only on X, not
on model forward(X). Therefore adapter choice cannot affect routing. □

**Prediction:** TF-IDF routing accuracy ≥ 85% (K955). Actual prediction: ≥ 97% from
cos(math, code) = 0.190 (Finding #389 measurement with identical domain pair).

### Theorem 3 (Composition Quality Lower Bound)

**Statement:** Under routed selection (TF-IDF selects one adapter at alpha=1.0),
the composed quality_ratio ≥ min(quality_ratio_math, quality_ratio_code) × routing_accuracy.

**Proof:** The composed system applies B_math or B_code selected by TF-IDF. If routing
accuracy = p, then:
  - With prob p: correct adapter applied → quality matches single-adapter
  - With prob (1-p): wrong adapter applied → quality ≈ base_acc

quality_ratio_composed = p × quality_ratio_single + (1-p) × 0
                       ≥ p × quality_ratio_single

For p ≥ 0.97 and quality_ratio_single ≈ 1.0: quality_ratio_composed ≥ 0.97 × 1.0 > 0.80. □

**Prediction:** If K955 (routing ≥ 85%) passes, K954 (quality_ratio ≥ 0.80) follows from
Theorem 3 with p=0.85 → quality_ratio ≥ 0.85 × 1.0 = 0.85 > 0.80. The theorems are coupled.

---

## D. Kill Criteria (from proofs)

| ID  | Criterion | Source | Prediction |
|-----|-----------|--------|-----------|
| K954 | Routed composition ≥ 80% of best-single on EACH domain | Theorem 3 | PASS (97% × single ≈ 0.97) |
| K955 | TF-IDF routing ≥ 85% on math vs code | Theorem 2 + Finding #389 | PASS (pred: ≥ 97%) |

**KILL condition:** K955 FAIL → composition quality is uncertain (can't isolate adapter quality from routing error). K954 FAIL with K955 PASS → fundamental composition failure.

---

## E. Prediction-vs-Measurement Table (to be filled by PAPER.md)

| Prediction | Source | Measured | Pass? |
|-----------|--------|---------|-------|
| TF-IDF routing ≥ 97% | Theorem 2 + Finding #389 | TBD | TBD |
| quality_ratio_math ≥ 0.80 | Theorem 3 | TBD | TBD |
| quality_ratio_code ≥ 0.80 | Theorem 3 | TBD | TBD |
| grad_norm > 0 at step 0 (math) | Theorem 5 | TBD | TBD |
| grad_norm > 0 at step 0 (code) | Theorem 5 | TBD | TBD |
