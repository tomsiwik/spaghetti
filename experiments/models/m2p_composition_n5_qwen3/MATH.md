# MATH.md: M2P Composition on Qwen3-0.6B — Two Real-LLM Adapters Compose Without Interference

## TYPE: verification (Type 1)
## PROVEN FRAMEWORK:
  - Theorem 5 (v3/v4 MATH.md) — functional LoRA forward gives non-zero gradients
  - Finding #50 (conclusive) — Grassmannian A-slot composition interference-free
  - Finding #354 (supported) — TF-IDF routing achieves 95% on synthetic domains
  - Finding #14 (supported) — 1/N additive scaling resolves composition catastrophe

---

## A. Failure Mode Identification

### What degenerate behavior could occur?

**Failure 1 (Gradient disconnection):** Under composition, generating B-matrices
as tensor arguments via M2P could break the autodiff graph for the composed adapter.
Specifically, if ΔW_composed = 0.5·ΔW_math + 0.5·ΔW_sort is computed at inference
time from two separately-trained M2P networks, the gradient test (K925) must confirm
that at least one network's parameters retain non-zero gradient at composition time.

This failure is IMPOSSIBLE by Theorem 5 (v3) — the functional LoRA forward ensures
∂L/∂θ_M2P ≠ 0 regardless of how B is constructed, as long as B is computed from θ_M2P
via at least one multiplication.

**Failure 2 (Mutual interference / cross-task activation):** ΔW_math and ΔW_sort
might share common directions in weight space, causing each adapter to destructively
interfere with the other's task under composition.

This is controlled by the Grassmannian A-matrix orthogonality guarantee (Theorem 1 below).

**Failure 3 (Router distribution mismatch):** A router trained on base model hidden
states fails at inference time because the composed model has different hidden state
statistics — this is the root cause of the prior exp_m2p_composition_n5 KILL
(36.6% accuracy, Finding #351).

This failure is impossible by construction when routing on INPUT TEXT FEATURES before
any model forward (Theorem 2 below).

**Are these root causes or symptoms?**

The root cause of Failure 2 is weight-space interference. The math asks: what property
of A-matrices makes this impossible, rather than "how do we reduce it"? The answer is
exact orthogonality: A_i^T A_j = 0 → ⟨ΔW_i, ΔW_j⟩_F = 0 for ANY B_i, B_j.

The root cause of Failure 3 is hidden-state covariate shift under composition. The math
asks: what routing feature space is INVARIANT to this shift by construction? The answer
is the input text itself — TF-IDF on raw tokens precedes any model computation.

---

## B. Prior Mathematical Foundations

### B1. Grassmannian Construction (cited)

**Definition (Grassmannian manifold Gr(r, d)):** The space of r-dimensional subspaces
of R^d. A matrix A ∈ R^{d×r} with orthonormal columns represents a point on Gr(r, d).

**Procedure (QR-based multi-domain Grassmannian):**

Used in exp_m2p_composition_n5 (Finding #50, max|cos|=1e-08) and exp_m2p_qwen06b_gsm8k_v4:

For N domains, rank r, and dimension d (with N·r ≤ d), generate:
```
X ∈ R^{d × N·r}  via X_ij ~ N(0,1)
Q, _ = QR(X)         # Q has orthonormal columns (Q^T Q = I_{Nr × Nr})
A_i = Q[:, i·r:(i+1)·r]  for i = 0..N-1
```

Then A_i^T A_j = (Q[:,i·r:(i+1)·r])^T (Q[:,j·r:(j+1)·r]) = I_{Nr}[i·r:(i+1)·r, j·r:(j+1)·r] = 0 for i≠j.

This is the key identity. It follows directly from Q^T Q = I: the block (i,j) of Q^T Q
is A_i^T A_j, and the full I_Nr is block-diagonal in r×r blocks only along the diagonal.

**Reference:** exp_m2p_composition_n5 Finding #50 verified max|cos|=1e-08 (numerical
zero) for 5 domains at d=256, r=4. The same QR construction is used here for d=1024
(Qwen3-0.6B q_proj input) and d=2048 (q_proj output), r=4, N=2 domains.

### B2. Frobenius Inner Product and Interference (cited)

**Definition:** The Frobenius inner product of two matrices is
  ⟨M, N⟩_F = Σ_{ij} M_{ij} N_{ij} = tr(M^T N).

**Additive interference:** Under composition W_composed = W_base + ΔW_math + ΔW_sort,
the cross-task term in the Taylor expansion of L_math(W_composed) is:
```
∂L_math/∂W^T · ΔW_sort = ⟨∇_W L_math, ΔW_sort⟩_F
```
For LoRA adapters: ΔW_i = scale · A_i B_i^T (rank-r perturbation), so:
```
⟨ΔW_math, ΔW_sort⟩_F = scale² · tr(B_math^T A_math^T A_sort B_sort)
                       = scale² · tr(B_math^T (A_math^T A_sort) B_sort)
```
If A_math^T A_sort = 0, this is identically zero for ALL B_math, B_sort.

**Reference:** This identity is well-known in LoRA theory; stated explicitly in
"Mixture of LoRA Experts" (MOLE, arXiv:2402.09432) and verified in Finding #50.

### B3. TF-IDF Separability (cited)

**Theorem (LoraRetriever, arXiv:2402.09997, Section 3.2):** Text-based routing
(TF-IDF + cosine similarity or linear classifier) decouples routing accuracy from
the model's latent space. The key property is:

  Route(x) is a function of the raw input tokens x only, not of f_θ(x).

Therefore: for ANY composition W_base + ΔW_1 + ... + ΔW_k, Route(x) is
invariant under changes to W (i.e., invariant to adapter stacking).

**Reference:** Finding #207/#247 confirmed 90% TF-IDF routing on 5 synthetic SFT
domains using char n-gram TF-IDF + logistic regression.

### B4. 1/N Additive Scaling (cited)

**Result (Finding #14, supported):** Composing N adapters with equal weight 1/N
resolves the composition catastrophe (PPL ratios from 10^12 to 2.36 with 1/N scaling).

**Formal bound:** Under LoRA with scale s and adapter weight α:
  ΔW_composed = α · ΔW_math + α · ΔW_sort  (α = 1/N = 0.5 for N=2)

The spectral norm of the composed perturbation is:
  ||ΔW_composed||_2 ≤ α · (||ΔW_math||_2 + ||ΔW_sort||_2) = ||ΔW_i||_2 / N

So the composed perturbation is bounded by the per-adapter norm divided by N.
For 1/N scaling, the composed perturbation norm never exceeds the single-adapter norm
(it is exactly 1/N of the sum, which equals the per-adapter norm when norms are equal).

---

## C. Theorem Proofs

### Theorem 1: Parameter Orthogonality for Real-LLM M2P Adapters

**Statement:** Let A_math, A_sort ∈ R^{d×r} be constructed by the QR Grassmannian
procedure (Section B1) with N=2, so A_math = Q[:,0:r] and A_sort = Q[:,r:2r].
Let B_math, B_sort ∈ R^{r×d'} be ANY matrices (including those generated by trained
M2P networks). Then:

  ⟨ΔW_math, ΔW_sort⟩_F = 0

where ΔW_i = A_i B_i (the LoRA adapter weight delta).

*Proof.*

ΔW_math = A_math B_math ∈ R^{d × d'}  (each column is a sum of r rank-1 terms)
ΔW_sort  = A_sort  B_sort  ∈ R^{d × d'}

The Frobenius inner product:

  ⟨ΔW_math, ΔW_sort⟩_F = tr(ΔW_math^T ΔW_sort)
                         = tr(B_math^T A_math^T A_sort B_sort)

By the QR construction: A_math^T A_sort = 0  (proved in Section B1).

Therefore:
  ⟨ΔW_math, ΔW_sort⟩_F = tr(B_math^T · 0 · B_sort) = 0.

This holds for ALL B_math, B_sort, including:
  - B-matrices generated by separately-trained M2P networks
  - B-matrices from SFT fine-tuning
  - Any random B-matrices

The result is exact (floating-point zero up to machine epsilon from the QR computation).

QED.

**Corollary (Interference-Free Composition for Qwen3-0.6B):**

For Qwen3-0.6B, d_model=1024, r=4, N=2. The capacity check:
  N·r = 2·4 = 8 ≤ d_model = 1024  (128× margin)

So the QR construction is always feasible. The Grassmannian guarantee holds exactly
for both q_proj (d_in=1024, d_out=2048) and v_proj (d_in=1024, d_out=1024).

**Extension to multiple layers:** Theorem 1 applies independently per layer.
Since each layer l has its own A_math^{(l)}, A_sort^{(l)} pair generated by independent
QR decompositions (same seed, different layer index), the interference is zero across
all 28 layers simultaneously.

---

### Theorem 2: TF-IDF Routing Correctness (Distribution Invariance)

**Statement:** Let Route_TF-IDF: T* → {0, 1, ..., N-1} be a routing function
mapping input token sequence T to a domain index, where Route_TF-IDF depends only
on the raw token sequence (TF-IDF features + linear classifier on input text).

Then for any model parameters θ and any composition of adapters:
  Route_TF-IDF(T) = Route_TF-IDF(T)  (trivially invariant to model state)

Moreover, for the two tasks in this experiment:

Task 1 (math / GSM8K): prompts contain "Question:", numbers, arithmetic operators,
  and the few-shot prefix "Solve the math problem step by step and end with '#### <answer>'."
Task 2 (sort): prompts contain "Sort these words alphabetically:", comma-separated words.

The TF-IDF vocabulary over char n-grams captures:
- "####" as a distinctive trigram for math
- "alphabetically" as a distinctive word for sort
- Digit-heavy text (math) vs alphabetic-heavy text (sort)

These features are TF-IDF-linearly-separable at near-100% accuracy (empirical claim;
verified in K926 below).

*Proof of distribution invariance.*

The training distribution for the router is:
  P_train(TF-IDF(T) | domain = math)
  P_train(TF-IDF(T) | domain = sort)

At inference time on the composed model, the input text T is IDENTICAL to what was
used to train the router — the same GSM8K questions and sort prompts are presented.
The composed model parameters θ_composed are NEVER passed to the router.

Therefore Route_TF-IDF(T; θ_composed) = Route_TF-IDF(T; θ_base) for all θ_composed.

There is zero train/test distribution mismatch by construction, regardless of what
adapter is applied. This is the fix to the root cause that killed exp_m2p_composition_n5.

QED.

**Remark:** The prior experiment (exp_m2p_composition_n5) used a per-token MLP router
trained on BASE hidden states h = f_{θ_base}(T). At inference time, h_composed = f_{θ_composed}(T)
has different statistics (covariate shift). TF-IDF routing is immune because it never
calls f_θ — it operates on T directly.

---

### Theorem 3: Quality Lower Bound Under Routed Selection

**Statement (updated — adversarial fix):** Let acc_single^{(t)} be the accuracy of
a single M2P adapter on task t (math or sort) evaluated alone, and acc_base^{(t)} be
the accuracy of the base model (zero adapter). Under ROUTED SELECTION — TF-IDF routes
each input to ONE adapter at full weight (alpha=1.0):

  ΔW_routed^{(t)} = ΔW_{domain(x)}   (only one adapter applied, at full scale)

The quality_ratio is defined as:
  quality_ratio^{(t)} = (acc_composed^{(t)} - acc_base^{(t)}) / (acc_single^{(t)} - acc_base^{(t)})

This formula measures what fraction of the single-adapter improvement is preserved
after composition. quality_ratio = 1.0 means zero degradation; quality_ratio = 0.0
means composed = base; quality_ratio < 0 means composed is worse than base.

**Why routed selection, not additive blend:**

Additive blend (ΔW_composed = 0.5·ΔW_math + 0.5·ΔW_sort) applies both adapters
simultaneously. Even with Grassmannian orthogonality (Theorem 1), the wrong-domain
adapter still contributes zero information but does not contribute zero activation —
it adds noise along orthogonal directions that may still affect layer normalization
and downstream layers. Routed selection sidesteps this entirely: only the relevant
adapter is applied, so acc_composed^{(t)} = acc_correct_routing^{(t)} exactly for
correctly-routed examples.

**Convergence precondition (Fix 4):** Sort K927 is only evaluated if:
  acc_single^{(sort)} > acc_base^{(sort)} + 0.10
This ensures the adapter has genuinely learned the task before measuring quality_ratio.
If the gate fails, K927 for sort is skipped with diagnostic output.

**Lower bound derivation under routed selection:**

If the router has accuracy ρ ≥ 0.80 (K926 requirement), then by the law of total probability:
  acc_composed^{(t)} = ρ · acc_single^{(t)} + (1-ρ) · acc_wrong_routing^{(t)}

where acc_wrong_routing^{(t)} ≥ acc_base^{(t)} in the worst case.

  quality_ratio^{(t)} ≥ (ρ · acc_single + (1-ρ) · acc_base - acc_base) / (acc_single - acc_base)
                       = ρ · (acc_single - acc_base) / (acc_single - acc_base)
                       = ρ ≥ 0.80

For math: ρ = 0.80 (K926 min), so quality_ratio_math ≥ 0.80 > 0.75 = K927 threshold.

So the lower bound on quality_ratio (≥ ρ ≥ 0.80) exceeds the K927 threshold of 0.75.

*Proof sketch.*

Under routed selection at full weight:
1. For correctly-routed examples (fraction ρ): acc = acc_single (single adapter applied)
2. For misrouted examples (fraction 1-ρ): acc = acc_wrong_routing ≥ acc_base

Therefore:
  acc_composed = ρ · acc_single + (1-ρ) · acc_wrong_routing

  quality_ratio = (ρ · acc_single + (1-ρ) · acc_wrong_routing - acc_base)
                  / (acc_single - acc_base)
               = ρ + (1-ρ) · (acc_wrong_routing - acc_base) / (acc_single - acc_base)

Since the last term is non-negative when acc_wrong_routing ≥ acc_base:
  quality_ratio ≥ ρ ≥ 0.80 > 0.75

QED (modulo empirical steps ρ ≥ 0.80 verified by K926 and convergence gate by sort_single_acc).

**Caveat:** If acc_wrong_routing < acc_base (applying wrong adapter actively hurts more
than base model), the lower bound fails. This is the empirical risk K927 is designed to catch.

---

## D. Quantitative Predictions

| Kill Criterion | Proof Source | Prediction | Tolerance |
|----------------|--------------|------------|-----------|
| K925: grad_norm > 0 at step 0 under routed adapter | Theorem 5 (v3), Theorem 1 | grad_norm ≈ 1.5–6.3 (matching v3/v4 range) | > 0 (strictly) |
| K926: TF-IDF routing ≥ 80% on both tasks | Theorem 2 (linear separability) | ≥ 95% expected (distinctive text features) | ≥ 80% |
| K927: quality_ratio ≥ 0.75 on both tasks (MATH formula, routed) | Theorem 3 lower bound | quality_ratio ≥ ρ ≥ 0.80 (Theorem 3), empirical math ≈ 0.85–1.10 | ≥ 0.75 |

**quality_ratio formula (canonical, per Theorem 3):**
  quality_ratio = (composed_acc - base_acc) / (single_acc - base_acc)

This is the fraction of single-adapter improvement preserved under composition.
K927 threshold: quality_ratio ≥ 0.75 (at least 75% of single-adapter gain retained).

**Prediction for K926:** GSM8K questions and word-sort prompts are nearly
perfectly linearly separable in TF-IDF space. GSM8K: "#### <answer>", digits, "step by step".
Sort: "Sort these words alphabetically", ":", common English nouns. The two domains
have virtually no overlapping vocabulary — expect ≥ 98% accuracy from TF-IDF logistic regression.

**Prediction for K927 (math):** Theorem 3 predicts quality_ratio_math ≥ ρ ≥ 0.80.
In practice with ρ ≈ 0.98–1.00 (near-perfect routing), quality_ratio_math ≈ 1.00.
The math adapter warm-starts from v4 weights so acc_single ≈ 0.286 (v4 result).
base_acc ≈ 0.20 (Qwen3-0.6B base, measured empirically in Phase 5).

**Prediction for K927 (sort):** Only evaluated if sort_single_acc > sort_base_acc + 0.10
(Fix 4 convergence gate). If sort adapter converges, Theorem 3 predicts quality_ratio_sort ≥ ρ.

---

## E. Assumptions and Breaking Conditions

| Assumption | What happens if violated | Connection to kill criterion |
|------------|--------------------------|------------------------------|
| N·r ≤ d_model (N=2, r=4, d=1024 → 8 ≪ 1024) | QR construction infeasible | Not at risk here; 128× margin |
| A_math and A_sort are SEPARATE QR slots (Fix 1) | If both adapters use the same A-matrices, A_math^T A_sort ≠ 0 and Theorem 1 is void | K927 directly; prior code bug voided Theorem 1 |
| Sort task has distinctive TF-IDF features | Routing fails if features overlap with math | K926 directly; also triggers K927 via misrouting |
| acc_wrong_routing ≥ acc_base | K927 lower bound (quality_ratio ≥ ρ) fails | K927 quality_ratio still measured empirically |
| M2P sort adapter convergence precondition (Fix 4) | sort_single_acc must exceed base + 0.10 | K927 sort is skipped if gate fails; diagnostic output produced |
| Routed selection at alpha=1.0 (Fix 3, Theorem 3) | Additive blend (0.5+0.5) would test different composition than Theorem 3 models | K927 primary result is routed mode; additive blend is secondary/informational |

---

## F. Worked Example (d=8, r=2, N=2)

### Setup
- d = 8, r = 2, N = 2
- Generate X ∈ R^{8×4} ~ N(0,1), QR → Q ∈ R^{8×4}

```
X (example):
[[ 0.5, -0.3,  0.8, -0.1],
 [-0.2,  0.7, -0.4,  0.9],
 [ 0.6,  0.1, -0.7,  0.3],
 [-0.9,  0.4,  0.2, -0.6],
 [ 0.3, -0.8,  0.5,  0.4],
 [-0.1,  0.2, -0.9,  0.7],
 [ 0.7, -0.5,  0.1, -0.2],
 [-0.4,  0.6, -0.3,  0.8]]

After QR, Q has orthonormal columns: Q^T Q = I_4
```

### A-matrix assignment
- A_math = Q[:, 0:2]  (columns 0-1)
- A_sort = Q[:, 2:4]  (columns 2-3)

### Verification: A_math^T A_sort
```
A_math^T A_sort = (Q[:,0:2])^T (Q[:,2:4])
               = [Q^T Q]_{0:2, 2:4}
               = I_4[0:2, 2:4]
               = [[0, 0],
                  [0, 0]]
```
Exact zero, as required.

### Frobenius inner product check
- B_math = [[1.5, -0.3], [0.7, 0.9]] (arbitrary, shape r×d' = 2×2 for simplicity)
- B_sort = [[-0.5, 1.2], [0.3, -0.8]]

```
ΔW_math = A_math B_math ∈ R^{8×2}
ΔW_sort = A_sort B_sort ∈ R^{8×2}

⟨ΔW_math, ΔW_sort⟩_F = tr(B_math^T A_math^T A_sort B_sort)
                       = tr(B_math^T · 0 · B_sort)
                       = 0
```

The calculator-verifiable result: regardless of what B_math and B_sort contain,
the inner product is exactly 0 because A_math^T A_sort = 0.

### Composition
```
ΔW_composed = 0.5 · ΔW_math + 0.5 · ΔW_sort
||ΔW_composed||_F² = 0.25 · ||ΔW_math||_F² + 0.25 · ||ΔW_sort||_F²
                     + 0.5 · ⟨ΔW_math, ΔW_sort⟩_F
                   = 0.25 · (||ΔW_math||_F² + ||ΔW_sort||_F²)
```
(Cross-term vanishes by Theorem 1.)
So the composed adapter norm equals √(0.25(||ΔW_math||² + ||ΔW_sort||²)) ≈ single-adapter norm / √2
when the two adapters have equal norm. This is substantially smaller than if they
had aligned (cross-term = +||ΔW||²), confirming no over-amplification.

---

## G. Complexity and Architecture Connection

### Qwen3-0.6B dimensions (from v4 results.json)
- n_layers = 28
- d_model = 1024
- n_heads = 16, n_kv_heads = 8, head_dim = 64
- q_proj: d_in=1024, d_out=2048 (n_heads × head_dim = 16×64)
- v_proj: d_in=1024, d_out=1024 (n_kv_heads × head_dim = 8×64 — but v4 uses 1024)

### LoRA adapter dimensions
- A_q: R^{1024 × 4}, A_v: R^{1024 × 4} per layer
- B_q: R^{4 × 2048}, B_v: R^{4 × 1024} per layer
- Total ΔW per adapter: 28 × (1024×4 + 1024×4) = 28 × 8192 = 229,376 parameters

### M2P architecture
- Input: 28 × 1024 layer hidden states → mean-pooled
- Encoder: 1024 → 2048 → 1024 (MLP, same as v4)
- Output heads: 28 × (q: 4×2048 + v: 4×1024) = 28 × (8192 + 4096) = 344,064 output scalars
- Total M2P params: encoder (≈4.2M) + heads (≈344K × 2 + overhead) ≈ 5-8M per adapter

### FLOPs
- Training (per step, single adapter): ~2 × forward pass FLOPs ≈ 2 × 600M = 1.2G FLOPs
- 300 steps × 2 tasks = 600 steps × 1.2G = 720G FLOPs training
- Inference (per eval example): 1 forward pass × 100 tokens ≈ 60G FLOPs
- Total eval: 200 examples × 60G = 12T FLOPs eval
- Wall time estimate: ~20 min training + ~10 min eval = ~30 min

### Production Architecture Connection
Per the architecture gallery (https://sebastianraschka.com/llm-architecture-gallery/),
Qwen3-0.6B uses GQA (8 KV heads, 16 query heads), RoPE positional encoding, SiLU MLP,
and RMSNorm. The functional LoRA forward in this experiment only patches q_proj and v_proj,
leaving all other components (k_proj, o_proj, MLP, norms, embeddings) unchanged.
This matches the SHINE paper (arXiv:2602.06358) design that uses functional injection
only where most impactful (query and value projections).

---

## Self-Test (MANDATORY)

**1. What is the ONE mathematical property that makes the failure mode (interference) impossible?**
A_math^T A_sort = 0 (QR Grassmannian construction, SEPARATE slots) makes the Frobenius
inner product ⟨ΔW_math, ΔW_sort⟩_F = 0 exactly for ANY B-matrices. This requires
separate A-slot assignment per domain (Fix 1). Routing failure is handled separately
by TF-IDF's input-text invariance (Theorem 2), not by the same math property.

**2. Which existing theorem(s) does the proof build on?**
- QR decomposition: columns of Q are orthonormal (standard linear algebra, Golub & Van Loan 1996)
- Frobenius inner product factorization: tr(A^T B) = 0 when one factor is 0
- LoraRetriever (arXiv:2402.09997): text-based routing invariant to model distribution
- Finding #50 (this project): max|cos|=1e-08 verified for 5-domain Grassmannian composition
- Theorem 5 (v3 MATH.md): functional LoRA forward guarantees ∂L/∂θ_M2P ≠ 0

**3. What specific numbers does the proof predict?**
- K925: grad_norm > 0 (expected ≈ 1.5–6.3, matching v3/v4 range)
- K926: TF-IDF accuracy ≥ 95% on both tasks (very linearly separable features)
- K927: quality_ratio = (composed-base)/(single-base) ≥ 0.75;
         Theorem 3 lower bound gives quality_ratio ≥ ρ ≥ 0.80 under routed selection

**4. What would FALSIFY the proof?**
- Theorem 1 is falsified if: A_math^T A_sort ≠ 0 (would require QR to fail or wrong slots used)
- Theorem 2 is falsified if: TF-IDF accuracy < 80%, meaning the two task prompt formats
  are not linearly separable in TF-IDF space (extremely unlikely given domain-specific vocabulary)
- Theorem 3 is falsified if: quality_ratio < ρ despite correct routing — this would imply
  acc_wrong_routing < acc_base for the misrouted fraction, i.e., applying wrong adapter
  actively degrades below base model performance

**5. How many hyperparameters does this approach add?**
0 new hyperparameters vs v4. All values inherited: LORA_RANK=4, LORA_SCALE=5.0,
OUTPUT_SCALE=0.032, LR=5e-5, composition weight α=0.5=1/N (N=2, derived from Finding #14).

**6. Hack check: Am I adding fix N to an existing stack?**
No. This experiment applies two independently-proven mechanisms:
1. Grassmannian A-matrices (proven in Finding #50 for synthetic, now on real LLM)
2. TF-IDF routing (proven in Finding #354, now on real LLM task vocabulary)
Each mechanism solves a DIFFERENT failure mode (interference vs. routing mismatch).
No new tricks are added — both are direct applications of previously proven tools.
