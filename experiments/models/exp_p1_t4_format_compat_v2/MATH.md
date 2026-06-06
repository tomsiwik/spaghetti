# MATH.md — T4.5v2: Adapter Format Compat with Actual PEFT/vLLM Loading (Loophole Fix)

## Background

`exp_p1_t4_adapter_format_compat` (T4.5) claimed format compatibility via:
1. `LoraConfig` instantiation (not full model loading)
2. Silent bypass when `peft` not installed
3. Metadata tag `"orthonormal_rows"` written regardless of actual deviation (measured: 0.579)

This experiment fixes all three loopholes with a falsifiable experiment:
K1: actual `PeftModel.from_pretrained()` on CPU  
K2: honest Grassmannian property reporting (no claim when deviation > 1e-4)  
K3: exact round-trip MLX → PEFT → MLX (max diff < 1e-6)

---

## Theorem 1 — MLX ↔ PEFT Bijection

**Claim:** The weight key mapping T: MLX → PEFT is an exact involutive bijection.

**MLX convention:** lora_a stored as [d_in, r] (rows = input space)  
**PEFT convention:** lora_A.weight stored as [r, d_in] (PEFT applies X @ A^T)

**Key mapping:** `path.lora_a` → `base_model.model.path.lora_A.weight`  
**Weight mapping:** W_PEFT = W_MLX^T (single transpose per tensor)

**Proof:**  
Let f_A: R^{d_in × r} → R^{r × d_in} be the transpose map.  
Round-trip: f_A^{-1}(f_A(A)) = (A^T)^T = A  
Since transposition is an involution: f_A ∘ f_A = Id  
∴ The bijection is exact, no numerical error beyond floating-point precision. □

**Prediction:** Round-trip max diff = 0.0 (float32 identity, no lossy operations).  
**Kill criterion (K3):** max diff < 1e-6 (allows FP32 epsilon, prediction is 0.0)

---

## Theorem 2 — Grassmannian Integrity Reporting

**Claim:** The metadata tag `"orthonormal_rows"` must be written if and only if  
    ||A^T A - I_r||_∞ < ε, where ε = 1e-4.

**Definition:** A Grassmannian adapter has lora_a A ∈ R^{d_in × r} with  
orthonormal columns, i.e., A^T A = I_r (Stiefel manifold Vr(R^{d_in})).

**QR initialization (T3.x baseline):**  
A = Q[:, :r] from thin QR of random R^{d_in × r}.  
By orthogonality of QR: Q^T Q = I_r exactly.  
∴ deviation_QR = ||A^T A - I_r||_∞ < 1e-12 (numerical noise only).

**Post-training drift (T4.5 measured):**  
SGD on A destroys the Stiefel constraint: deviation_trained = 0.579 >> 1e-4.  
The tag `"orthonormal_rows"` MUST NOT be written for trained adapters.

**Prediction (K2):**  
- Fresh QR adapter: deviation < 1e-12 → tag IS written
- Random (untrained) adapter: deviation ~ 0.3-0.6 → tag IS NOT written  
- The two cases are distinguishable by > 6 orders of magnitude.

---

## Theorem 3 — CPU Loadability via PeftModel

**Claim:** A PEFT adapter is loadable on CPU by `PeftModel.from_pretrained()` iff:  
(a) adapter_config.json has valid LoraConfig schema  
(b) adapter_model.safetensors keys match `base_model.model.<path>.lora_{A,B}.weight`  
(c) Weight shapes match base model architecture: lora_A.weight ∈ R^{r × d_in}, lora_B.weight ∈ R^{d_out × r}

**Proof sketch:**  
PeftModel.from_pretrained() loads base model, reads config, identifies target_modules,  
creates LoraLinear wrappers for each target layer, loads weight tensors by key matching.  
Condition (b) ensures no KeyError during state_dict loading.  
Condition (c) ensures no shape mismatch during parameter assignment.  
Condition (a) is necessary for LoraConfig construction. □

**Test design:**  
- Base model: facebook/opt-125m (CPU, 125M params, loads in ~2s)  
- Adapter: fresh QR-initialized LoRA targeting q_proj layers  
- Target layer: self_attn.q_proj, shape 768×768, rank r=4  
- PEFT path prefix: `base_model.model.model.decoder.layers.{N}.self_attn.q_proj`

**Prediction (K1):** PeftModel.from_pretrained() completes without RuntimeError, ValueError,  
or shape mismatch. Forward pass produces output shape [batch, seq, 768].

---

## Kill Criteria Summary

| ID | Criterion | Prediction | Threshold |
|----|-----------|------------|-----------|
| K1 | PeftModel.from_pretrained() CPU — no error | PASS | No exception |
| K2 | Grassmannian honesty: QR reports tag, random does not | PASS | Deviation < 1e-4 → tag; else no tag |
| K3 | Round-trip max diff < 1e-6 | PASS (exact: 0.0) | 1e-6 |

---

## References

1. Hu et al. (2021) arxiv 2106.09685 — LoRA: weight convention A,B  
2. Geva et al. (2012.14913) — attention value layers  
3. PEFT library (HuggingFace) — PeftModel.from_pretrained() CPU loading  
4. `exp_p1_t4_adapter_format_compat` LOOPHOLE_FINDING.md — original loopholes  
5. Finding #440 — Grassmannian interference bounds (T3.4, max_cos=2.25e-8)
