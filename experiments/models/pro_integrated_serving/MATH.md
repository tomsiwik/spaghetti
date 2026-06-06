# MATH.md: Integrated Serving Pipeline on Qwen3-4B-4bit

## Type: Verification (Type 1)

This experiment replicates the integrated pipeline (proven on BitNet-2B-4T in
exp_tiny_integrated_serving, Finding #323) on the larger Qwen3-4B-4bit model.
All individual components have been proven separately. The mathematical framework
is identical to the tiny experiment; the verification question is whether the same
composition guarantees transfer to a different architecture at larger scale.

---

## A. Failure Mode Identification

**Failure mode:** The integrated pipeline (block-diagonal masking + per-token MLP
routing + DARE + ridge router) that works on BitNet-2B-4T might fail on Qwen3-4B
due to:

1. **Scale sensitivity:** Adapters trained at scale=20 but composed at scale=5
   (per Finding #330) might have insufficient domain signal at the reduced scale,
   producing outputs indistinguishable from the base model.

2. **Architecture mismatch:** Qwen3 uses SiLU activation (not squared ReLU like
   BitNet), QK-norm (RMSNorm on query/key heads), and GQA (32/8 head ratio).
   The per-token MLP routing mechanism was proven on BitNet's architecture;
   pointwise MLP independence still holds for SiLU but the activation landscape
   differs.

3. **Quantization interaction:** Qwen3-4B-4bit uses 4-bit quantized weights.
   LoRA adapters add fp16/bf16 perturbations to 4-bit base weights. The effective
   perturbation magnitude relative to quantization noise is different from
   BitNet's native ternary weights.

---

## B. Prior Mathematical Foundations

**Theorem (RoPE relative position invariance -- Su et al. 2104.09864, verified
Finding #322).** Attention scores under RoPE depend only on relative position
b-a, not absolute positions. Block-diagonal masking zeroes cross-segment
attention without altering within-segment computation. Verified: bd fair gap =
0.244% on BitNet (Finding #322).

**Theorem (MLP token-independence -- Finding #313).** For pointwise MLP layers
f(x_t), the output at position t depends only on x_t and MLP parameters. This
holds for any elementwise activation (ReLU, squared ReLU, SiLU, GELU). Per-token
adapter routing produces output identical to separate forward passes for
same-segment tokens.

**Theorem (DARE unbiased estimator -- Yu et al. 2311.03099).** DARE with drop
probability p produces B_DARE = M * B / (1-p) where M ~ Bernoulli(1-p).
E[B_DARE] = B. Preserves expected perturbation while enabling sparse composition.

**Theorem (Ridge regression optimality).** W* = (X^TX + lambda*I)^{-1} X^TY is
the unique global minimizer for lambda > 0 (positive definite Hessian). Closed-form,
zero iterative training. Verified: 100% routing accuracy on 5 domains (Finding #323).

**Finding #320/#330 (Davis-Kahan bound on scale).** At scale<=5, MMLU degradation
is 0pp (within 7.5pp CI). At scale=20, degradation is -42pp to -60pp. The
perturbation bound ||delta||/spectral_gap becomes vacuous above scale~13.

---

## C. Composition Conjecture

**Conjecture 1 (Additive independence, from tiny_integrated_serving MATH.md).**
Let epsilon_i be the PPL gap introduced by component i. Then:

  PPL_integrated <= PPL_oracle * prod_i (1 + epsilon_i)

where the epsilon_i are:
- epsilon_mask < 0.5% (block-diagonal masking, Finding #322)
- epsilon_mlp < 1% (per-token MLP routing, Finding #313)
- epsilon_dare < 5% (DARE sparsification, Finding #266)
- epsilon_route * delta_misroute (routing error impact)

The tiny experiment measured -2.8% (BETTER than oracle), contradicting the
additive degradation prediction. The sign flip suggests the independence model
is conservative -- the integrated pipeline may benefit from block-diagonal
isolation preventing cross-segment interference that the oracle (single adapter
on single segment) cannot avoid.

**Scaling prediction for Pro:** At scale=5 (vs scale=20 used in tiny), the
adapter perturbation is 4x smaller. This reduces all epsilon_i proportionally
(smaller perturbation = smaller gap). However, domain behavioral quality may
also be reduced since the adapter signal is weaker.

**Prediction:** With scale=5 on Qwen3-4B, the integrated pipeline will:
1. Preserve base model quality (behavioral >= 0.3 across domains)
2. Show integrated vs isolated gap < 10% (same bound as tiny)
3. Router accuracy >= 90% (Qwen3's higher hidden dimension d=2560 vs BitNet's
   d=2048 provides better domain separation)

---

## D. Predictions

### Behavioral predictions:
1. Per-domain behavioral scores >= 0.2 for majority of domains at scale=5
2. Overall behavioral score >= 0.3 (K821 threshold)
3. Router accuracy >= 90% on 5 domains
4. Integrated pipeline quality within 10% of per-sequence baseline

### Quantitative predictions:
| Prediction | Source | Expected (Qwen3-4B) |
|-----------|--------|---------------------|
| Router accuracy | Finding #276/#310/#323 | >= 90% |
| Per-domain behavioral (avg) | Finding #323 analogy | >= 0.3 |
| Integrated vs per-seq gap | Conjecture 1 | < 10% |
| Speed (single adapter) | Architecture dependent | > 30 tok/s (4-bit 4B model) |

### Kill criteria:
- **K821:** Behavioral score < 0.3 on majority of benchmarks. Derived from the
  minimum useful domain quality: if the pipeline cannot produce domain-relevant
  responses, composition is pointless regardless of PPL metrics.

---

## E. Assumptions and Breaking Conditions

1. **Scale=5 preserves domain signal.** If adapters trained at scale=20 produce
   negligible outputs at scale=5, behavioral quality will match base model
   (defeating the purpose). Breaking: all domain behavioral scores equal to base
   model behavioral score (adapters have no effect).

2. **SiLU MLP is pointwise.** SiLU(x) = x * sigmoid(x) is elementwise, so
   MLP token-independence holds. No breaking condition (structural guarantee).

3. **Qwen3 QK-norm does not break block-diagonal isolation.** QK-norm applies
   RMSNorm to query and key heads BEFORE RoPE. Since RMSNorm is a per-token
   operation (normalizes each token's head independently), it does not introduce
   cross-token dependencies. Block-diagonal masking still provides exact segment
   isolation. No breaking condition (structural guarantee).

4. **4-bit quantization does not invalidate LoRA composition.** The 4-bit base
   weights are dequantized to bf16 during forward pass. LoRA perturbation is
   added in bf16. The composition is: y = dequant(W_4bit) @ x + alpha * (x @ A) @ B.
   The quantization noise in dequant() is independent of the LoRA perturbation.
   Breaking: if dequantization is non-deterministic or state-dependent (it is not).

5. **Attention LoRA asymmetry.** The integrated pipeline applies LoRA ONLY to MLP
   layers (gate_proj, up_proj, down_proj), using base weights for attention
   (q/k/v/o_proj). Both baselines (per-sequence and isolated oracle) apply
   RuntimeLoRA to ALL 7 target projections including attention. This creates a
   confound: the integrated pipeline's improvement over baselines could be caused
   by dropping attention LoRA (which may be harmful at scale=5, since adapters were
   trained at scale=20) rather than by pipeline composition quality. The comparison
   is:
   - **Integrated:** base attention + per-token MLP LoRA
   - **Per-sequence:** adapted attention + uniform MLP LoRA
   - **Isolated oracle:** adapted attention + correct MLP LoRA
   Breaking: if an MLP-only isolated control (no attention LoRA) matches the
   integrated pipeline's PPL, the improvement is from dropping attention LoRA,
   not from composition.

---

## F. Worked Example (2 segments, d=2560, Qwen3)

Input: 128 medical tokens + 128 code tokens = 256 total.

1. **Router:** Compute h = mean_pool(norm(hidden_states(text))). Project h @ W
   (W is 2560x5). argmax gives domain index. With d=2560 and 50 calibration
   texts per domain, ridge regression has ample dimensionality for 5-class separation.

2. **Block-diagonal mask:** 256x256 additive mask. Positions (i,j) where
   segment(i) != segment(j) get -inf. Within-segment: standard causal mask.
   Qwen3's QK-norm (applied before attention scores) does not break this because
   QK-norm operates per-token, not cross-token.

3. **Per-token MLP routing:** For each of 36 transformer layers, compute:
   - Attention with base weights + block-diagonal mask (no LoRA on attention)
   - MLP with per-token LoRA: tokens 0-127 use medical adapter, 128-255 use code
   - gate_out = base_gate(h) + alpha * (h @ A_medical_gate) @ B_medical_gate  [for t < 128]
   - gate_out = base_gate(h) + alpha * (h @ A_code_gate) @ B_code_gate  [for t >= 128]
   - Same for up_proj, down_proj
   - alpha = 5.0 (scale=5)

4. **DARE:** Before routing, sparsify each adapter B with Bernoulli(0.5) mask,
   rescale by 2. Expected perturbation unchanged.

5. **PPL:** Compute per-token NLL. Exclude boundary token. Fair PPL should be
   within 10% of per-sequence baseline.

---

## G. Complexity and Architecture Connection

| Component | FLOPs | Memory |
|-----------|-------|--------|
| Base model (4-bit) | O(T * d^2 * L) | ~2.3 GB |
| 5 adapters (bf16) | 5 * 252 params * 16*2560 = ~86 MB | Per-adapter: ~17 MB |
| Router W | O(T*d + d*K) | O(d*K) = 12.8 KB |
| Block-diag mask | O(T^2) | O(T^2) = 256 KB at T=256 |
| Integrated overhead | 2x MLP LoRA compute (both adapters per pair) | +1 adapter in memory |

Total memory: ~2.3 GB (model) + 0.17 GB (2 adapters) + negligible = ~2.5 GB.
Well within 48 GB M5 Pro budget.

---

## Self-Test

1. What is the ONE mathematical property that makes the failure mode impossible?
   **Conjectured additive independence of perturbation sources in log-probability
   space.** Each component (mask, MLP routing, DARE, router) contributes a bounded
   independent perturbation. This was supported (not proven) in the tiny experiment
   with 18/18 samples showing the integrated pipeline within (or better than) the
   per-sequence baseline.

2. Which existing theorem(s) does the proof build on?
   - Su et al. (2104.09864): RoPE relative position invariance
   - Yu et al. (2311.03099): DARE unbiased estimator
   - Ridge regression: closed-form global optimum (positive definite Hessian)
   - Davis-Kahan sin-theta theorem: perturbation bound for eigenspace rotation
   - Finding #322: block-diagonal masking gap < 0.5%
   - Finding #313: MLP token-independence, per-token routing gap < 0.7%
   - Finding #320/#330: scale<=5 preserves MMLU, scale=20 catastrophic

3. What specific numbers does the proof predict?
   - Router accuracy >= 90%
   - Behavioral score >= 0.3 overall
   - Integrated vs per-sequence gap < 10%
   - Speed > 30 tok/s (4-bit 4B model on M5 Pro)

4. What would FALSIFY the proof (not just the experiment)?
   - If MLP token-independence fails for SiLU (impossible: SiLU is elementwise)
   - If block-diagonal masking fails with QK-norm (impossible: QK-norm is per-token)
   - If additive independence assumption is wrong and components amplify each other's
     errors (would manifest as integrated gap >> 10%)
   - If the -3.4% vs isolated improvement is entirely from the attention LoRA
     asymmetry confound (would be confirmed if MLP-only isolated control matches
     integrated PPL). This would not invalidate the pipeline but would reattribute
     the sign flip from "composition benefit" to "attention LoRA is harmful at scale=5"

5. How many hyperparameters does this approach add?
   **0 new.** All from prior experiments: DARE p=0.5, ridge lambda=1.0,
   LORA_SCALE=5.0 (derived from Finding #330, not arbitrary).

6. Hack check: Am I adding fix #N to an existing stack?
   **No.** This is verification that 5 independently proven components compose
   correctly on a new architecture. No new mechanisms added.
