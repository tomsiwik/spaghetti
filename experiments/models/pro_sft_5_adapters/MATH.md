# SFT 5 Domain Adapters on Qwen3-4B: Mathematical Foundation

## Experiment Type: Guided Exploration (Type 2)

**Proven framework:** Grassmannian orthogonality on Qwen3-4B-4bit (Finding #318: exact
cos=0.000 at N=5), SFT convergence at N=5 on BitNet-2B (Finding #206), recipe transfer
across all 24 domains on BitNet (sft_24_domain_adapters experiment).

**Unknown:** Whether the SFT recipe transfers to a non-ternary quantized base (Qwen3-4B-4bit)
where the base model is significantly stronger (92% MMLU vs BitNet's ~55%). The proven
framework guarantees gradient flow and zero composition interference, but does NOT predict
SFT convergence or behavioral quality as a function of data quality vs base capability.

**Hypothesis:** The same SFT recipe (rank-16, scale 20, 300 steps, lr 1e-4, frozen
Grassmannian A, SFT loss) transfers from BitNet-2B to Qwen3-4B-4bit. The stronger base
model should produce higher behavioral quality.

---

## Step A: Failure Mode Identification

**Potential failure 1: QLoRA gradient blockage.** On BitNet-2B, we unpack ternary weights
to nn.Linear for differentiable training. On Qwen3-4B-4bit, the base layer is
QuantizedLinear. If gradients do not flow through the quantized matmul, LoRA B-matrices
receive zero gradient and training fails silently.

**Analysis:** MLX's QuantizedLinear.\_\_call\_\_ performs dequantize-then-matmul internally.
The dequantization step (mx.dequantize) produces a float tensor, so the forward pass is
differentiable with respect to the input x. The LoRA path is y = linear(x) + scale * (x @ A) @ B,
where the gradient dL/dB = scale * (x @ A)^T @ (dL/dy) flows entirely through float
operations. The quantized base weights are frozen and need no gradient. This is standard
QLoRA (Dettmers et al., 2305.14314).

**Potential failure 2: Scale mismatch.** BitNet-2B outputs have different magnitude than
Qwen3-4B-4bit outputs. The LoRA scale=20.0 was tuned for BitNet. If Qwen3-4B activations
are larger, scale=20.0 may cause instability; if smaller, the adapter contribution may be
negligible.

**Analysis:** The scale parameter controls the ratio of adapter contribution to base
output: y_total = y_base + scale * delta. For SFT, the adapter needs to shift the output
distribution toward domain-specific responses. The optimal scale depends on the ratio
||y_base|| / ||delta||. Since both models have d=2560, the activation norms should be
comparable (both models use RMSNorm, which normalizes to sqrt(d)). We use scale=20.0
as the starting point. If training diverges (K812), scale is the first parameter to
investigate.

**Potential failure 3: Format dominance.** The SFT loss masks instruction tokens. For
short training (300 steps), the adapter may learn only the shared instruction format
rather than domain-specific knowledge. This was observed on BitNet at scale=2.0/200 steps
(Finding #216) but resolved at scale=20.0/300 steps (Finding #206).

---

## Step B: The Right Question

**Wrong:** "Can we train LoRA on Qwen3-4B?"
(Trivially yes -- QLoRA is well-established.)

**Right:** "Does our specific recipe (Grassmannian frozen A, SFT loss, rank 16,
scale 20, 300 steps) produce domain-specialized adapters on a non-ternary quantized
base that exhibit behavioral domain competence?"

The distinction matters because:
1. Recipe transfer is not guaranteed across architectures (GQA vs MHA, different normalization)
2. Behavioral quality (not just loss convergence) is the target metric
3. Grassmannian A-matrices must work on QuantizedLinear (not just unpacked BitLinear)

---

## Step C: Prior Mathematical Foundations

**Theorem (QLoRA gradient flow, Dettmers et al., 2305.14314):** For a quantized linear
layer Q(x) = dequantize(W_q) @ x, the LoRA extension y = Q(x) + scale * (x @ A) @ B
has gradient:

  dL/dB = scale * (x @ A)^T @ (dL/dy)

which is independent of the quantization of W_q. The gradient flows entirely through
the float-valued LoRA path. This holds for any quantization scheme (4-bit, 2-bit,
ternary) provided the base forward pass produces float outputs.

**Grassmannian orthogonality (Finding #318):** QR-initialized A-matrices on Qwen3-4B
achieve |cos(A_i, A_j)| = 0.000 (machine precision) at N=5 across all 7 target modules
and 36 layers. This is exact, not approximate, because N*r = 5*16 = 80 << d_min = 2560.

**Composition interference bound (Finding #54):**

  ||DW_i^T DW_j||_F <= ||B_i||_F * ||A_i^T A_j||_F * ||B_j||_F / r^2

With A_i^T A_j = 0 (exact at N=5), the interference is exactly zero regardless of
B-matrix correlation. The adapters are composable by construction.

**SFT convergence (Finding #206 + sft_24_domain_adapters):** On BitNet-2B with the
identical recipe (rank=16, scale=20, 300 steps, lr=1e-4), all 24/24 domains converged
with mean val loss improvement of 16.7%. The weakest domain (finance) still improved
5.87%. The recipe is robust across diverse domains.

---

## Step D: Hypotheses (empirical, not formal proofs)

**Hypothesis 1 (QLoRA SFT convergence with frozen Grassmannian A).**
Under the following conditions:
- Base model Qwen3-4B-4bit (QuantizedLinear, d=2560)
- Standard LoRA with rank r=16, scale alpha=20, frozen Grassmannian A
- Adam optimizer with lr=1e-4
- SFT loss with response-only masking
- At least 100 training samples with mean response length >= 20 tokens

Then after 300 training steps, the validation loss L_final satisfies L_final < L_base.

*Argument.* The gradient dL/dB = alpha * (x @ A)^T @ (dL/dy) is nonzero whenever:
1. x @ A != 0 (guaranteed: A is orthonormal, so x @ A projects onto a nonzero subspace
   for any nonzero x from the residual stream)
2. dL/dy != 0 (guaranteed: the model does not predict the training data perfectly at
   initialization, so the cross-entropy loss gradient is nonzero)

Given a nonzero gradient, Adam with lr=1e-4 takes steps that decrease the local loss.
Over 300 steps with 400 training samples (cycling through data ~0.75x), the optimizer
has sufficient iterations to reduce validation loss.

This argument is empirical, not a formal convergence theorem. Its strength comes from
the N=24 replication on BitNet-2B where all 24 domains converged with the identical
recipe (independent of domain content).

**Hypothesis 2 (Behavioral quality from strong base).**
Qwen3-4B base scores 92% MMLU, indicating strong factual knowledge. SFT adapters
shift the output distribution toward domain-specific response patterns without
destroying base knowledge (the adapter is a rank-16 perturbation on a 2560-dimensional
space, affecting < 1% of the representational capacity).

The behavioral quality of the adapted model is bounded below by the base model's
domain knowledge (92% MMLU indicates strong coverage) and above by the data quality
of the training set. We predict behavioral quality > 0.5 (BitNet achieved 0.41 with
a weaker base, 55% MMLU).

---

## Step D (continued): Quantitative Predictions

| Prediction | Source | Value |
|------------|--------|-------|
| P1: Domains converging (L_final < L_base) | BitNet: 24/24 converged | 5/5 |
| P2: Mean val loss reduction | BitNet mean: 16.7% | 10-25% |
| P3: No divergence | BitNet: 0/24 diverged | 0/5 |
| P4: Per-domain training time | BitNet: ~110s per domain | 60-150s |
| P5: Mean behavioral score | BitNet: 0.41, stronger base | > 0.5 |
| P6: Peak memory during training | Qwen3-4B: 2.26 GB base + LoRA overhead | < 10 GB |

**Kill criteria (derived from predictions):**
- K812: P1 predicts 5/5 converge. FAIL if fewer than 4/5 converge.
- K813: P5 predicts behavioral > 0.5. FAIL if mean behavioral < 0.3 (generous threshold
  to account for eval methodology differences).

---

## Step E: Assumptions & Breaking Conditions

1. **QLoRA gradient flow:** MLX QuantizedLinear produces differentiable outputs.
   If violated: zero gradient, training loss does not decrease. Detectable immediately.

2. **Scale appropriateness:** scale=20.0 is appropriate for Qwen3-4B activation magnitudes.
   If violated: training diverges (scale too high) or adapter has no effect (scale too low).
   Detectable via initial training loss trajectory.

3. **Data format compatibility:** Qwen3-4B tokenizer can process the instruction/response
   format. Different tokenizer from BitNet may produce different sequence lengths and
   response-token ratios.
   If violated: response masks are empty (all instruction, no response tokens), causing
   zero SFT loss and no gradient. Detectable via initial loss values.

4. **Behavioral eval methodology:** Text generation quality depends on the generation
   procedure (temperature, top-k, max tokens) and evaluation criteria. Different from
   BitNet's eval.
   If violated: behavioral scores are not comparable. K813 threshold (0.3) is generous
   enough to absorb methodology differences.

---

## Step F: Worked Example (d=16, r=4, N=2)

Consider a toy model with hidden_dim=16, rank=4, 2 domains.

**Grassmannian A-matrices:**
Generate random 16x8 matrix, QR decompose:
- A_1 = Q[:, 0:4] (shape 16x4, orthonormal columns)
- A_2 = Q[:, 4:8] (shape 16x4, orthonormal columns)
- A_1^T @ A_2 = 0 (by QR orthogonality)

**Forward pass (domain 1 adapter on QuantizedLinear):**
- Input x: shape (1, seq_len, 16)
- Base: y_base = dequantize(W_q) @ x (computed by QuantizedLinear)
- LoRA: y_lora = scale * (x @ A_1) @ B_1
  - x @ A_1: shape (1, seq_len, 4) -- projection onto A_1's subspace
  - (x @ A_1) @ B_1: shape (1, seq_len, out_dim) -- low-rank update
- Total: y = y_base + y_lora

**Gradient (domain 1 adapter):**
- dL/dB_1 = scale * (x @ A_1)^T @ (dL/dy)
- shape: (4, out_dim) -- matches B_1's shape
- A_1 is frozen: no gradient for A_1

**Interference (composition of domains 1 and 2):**
- DW_1 = scale * A_1 @ B_1^T (shape 16 x out_dim)
- DW_2 = scale * A_2 @ B_2^T
- DW_1^T @ DW_2 = scale^2 * B_1 @ A_1^T @ A_2 @ B_2^T = 0
  (because A_1^T @ A_2 = 0)

---

## Step G: Complexity & Architecture Connection

**Per-domain training on Qwen3-4B-4bit:**
- Model load: ~1s (quantized, 2.26 GB)
- LoRA setup: 36 layers * 7 modules = 252 LoRA modules
  - Trainable params per module: r * out_dim (B-matrix only)
  - q_proj: 16 * 4096 = 65,536
  - k_proj: 16 * 1024 = 16,384
  - v_proj: 16 * 1024 = 16,384
  - o_proj: 16 * 2560 = 40,960
  - gate_proj: 16 * 9728 = 155,648
  - up_proj: 16 * 9728 = 155,648
  - down_proj: 16 * 2560 = 40,960
  - Per layer total: 491,520 params
  - Total (36 layers): 17,694,720 trainable params (~17.7M)

**Memory:**
- Base model (4-bit): 2.26 GB
- LoRA B-matrices (bfloat16): 17.7M * 2 bytes = ~35 MB
- LoRA A-matrices (frozen, bfloat16): ~same, ~35 MB
- Adam optimizer state (m, v): ~70 MB * 2 = ~140 MB
- Activation cache (seq_len=256, batch=1): ~200 MB
- **Total estimate: ~3.0 GB** (well within 48 GB)

**Total training time (5 domains):** 5 * ~120s = ~10 min

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   **QR orthogonality of A-matrices makes composition interference exactly zero at N=5.
   This is a geometric guarantee: A_i^T A_j = 0 for i != j when N*r <= d.
   NOTE: This guarantees composition, NOT SFT convergence or behavioral quality.**

2. Which existing theorem(s) does the proven framework build on?
   **QLoRA gradient flow (Dettmers et al., 2305.14314), QR decomposition orthogonality
   (Householder 1958), Grassmannian interference bound (Finding #54).**

3. What specific numbers do the hypotheses predict?
   **P1: 5/5 converge (FALSIFIED: 3/5). P2: 10-25% mean loss reduction. P5: behavioral > 0.5 (FALSIFIED: 0.391).**

4. What would FALSIFY the hypotheses?
   **If QuantizedLinear does not propagate gradients to the LoRA path (training loss
   constant). If scale=20.0 causes divergence on Qwen3-4B (training loss explodes).
   If fewer than 4/5 domains converge (recipe does not transfer). ACTUAL: 3/5
   converge due to data quality < base capability — a failure mode the hypotheses
   did not predict.**

5. How many hyperparameters does this approach add?
   **0 new. All inherited from proven BitNet recipe (rank=16, scale=20, lr=1e-4,
   steps=300).**

6. Hack check: Am I adding fix #N?
   **No. Direct recipe transfer to a new base model. Single mechanism (QLoRA with
   frozen Grassmannian A).**
