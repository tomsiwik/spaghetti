# T2.2: Adapter Quantization Quality Retention

## Type: Guided Exploration

**Proven framework:** Uniform quantization theory (Bennett 1948, Gray & Neuhoff 1998)  
**Unknown:** Empirical quality retention of task-specialized LoRA adapters under 4-bit and 2-bit compression  
**Paper:** GPTQ (2210.17323), LLM.int8() (2208.07339), HQQ (no-weights-quant-2023)

---

## Theorem 1: LoRA Quantization Error Bound

**Setup:** LoRA adapter ΔW = A @ B with A ∈ ℝ^{d_in × r}, B ∈ ℝ^{r × d_out}.  
After quantizing B to k-bit with group size g: B_q ← dequant(quant(B, bits=k, g)).

**Theorem:** The weight-space error satisfies:
```
‖A @ B_q − A @ B‖_F = ‖A @ (B_q − B)‖_F ≤ ‖A‖_2 · ‖B_q − B‖_F
```

For lora_a: shape (2560, 6), last dim r=6 is not divisible by MLX's minimum group size
(32). Therefore lora_a cannot be quantized with MLX's native kernel and is kept in fp16
(2× compression, zero quantization noise).

For lora_b: shape (r=6, d_out∈{2048, 4096}), last dim divisible by group_size=64.  
k-bit uniform affine quantization gives:
```
‖B_q − B‖_F ≤ range(B) / (2^k − 1) × √(n_elements)
```

**Empirically measured on trained adapters (math domain, 42 layers, Gemma4 E4B):**
- 4-bit, g=64: mean ‖ΔW_q − ΔW‖_F / ‖ΔW‖_F = **7.6%**
- 2-bit, g=64: mean ‖ΔW_q − ΔW‖_F / ‖ΔW‖_F = **33.4%**

These are relative errors in the adapter's weight contribution, not in the total model output.  
Since the base model output dominates (‖W_base‖ >> ‖ΔW‖), the actual quality degradation
is expected to be substantially smaller than the weight-space error.

**Proof:** Standard result. Quantization with step Δ = range/(2^k−1) introduces error
bounded by |ε_i| ≤ Δ/2 per element. Summing over n_elements gives the Frobenius bound. □

---

## Theorem 2: Orthogonality Preservation under Quantization

**Theorem:** If |cos(ΔW_i, ΔW_j)| = δ (near-zero), after quantizing B to k-bit:
```
|cos(ΔW_i_q, ΔW_j)| ≤ δ + ε_rel
```
where ε_rel = ‖B_q − B‖_F / ‖B‖_F is the relative quantization error.

**Proof sketch:**
cos(ΔW_i_q, ΔW_j) = ⟨A_i B_q_i, A_j B_j⟩_F / (‖A_i B_q_i‖_F ‖A_j B_j‖_F)

Writing B_q_i = B_i + ε_i:
⟨A_i(B_i + ε_i), A_j B_j⟩_F = ⟨A_i B_i, A_j B_j⟩_F + ⟨A_i ε_i, A_j B_j⟩_F

The second term is bounded by ‖A_i ε_i‖_F ‖A_j B_j‖_F ≤ ‖A_i‖_2 ‖ε_i‖_F ‖A_j B_j‖_F.

Dividing by ‖ΔW_i_q‖_F ‖ΔW_j‖_F and using continuity bounds the perturbation to O(ε_rel). □

**Corollary:** From T1.6 (Finding #420), |cos| = 0.00078 before quantization.
After 4-bit quantization (ε_rel ≈ 7.6%): |cos_4bit| ≤ 0.00078 + 0.076 < 0.08.
K1036 requires |cos_4bit| < 0.05 — this is a PREDICTION TO TEST, not a guarantee.

---

## Theorem 3: Adapter Size Under Mixed-Precision Quantization

**Strategy:** lora_a (d_in × r) in fp16; lora_b (r × d_out) in k-bit.

For Gemma4 E4B LoRA (r=6, 42 layers):
- lora_a per layer: d_in × r = 2560 × 6 = 15,360 elements
- lora_b per layer: r × d_out (varies: 2048 or 4096)

**fp32 serving size (measured T2.1):** 4.99 MB

**Analytical size for 4-bit mixed (lora_a fp16, lora_b 4-bit, g=64):**
```
size = Σ_layers [ d_in × r × 2             # lora_a fp16
               + r × d_out_l × (0.5)       # lora_b 4-bit data
               + r × (d_out_l / 64) × 8 ]  # scales + biases (fp32 per group)
```

Approximate: ~1.5 MB (3.3× compression vs fp32)

**2-bit mixed (lora_a fp16, lora_b 2-bit):** ~1.1 MB (4.5× compression)

**K1035 prediction:** 4-bit serving size < 5 MB → PASS with large margin (1.5 MB << 5 MB)

---

## Quantitative Predictions

| Criterion | Measurement | Predicted | Confidence |
|-----------|-------------|-----------|------------|
| K1033 | 4-bit quality ≥ 95% of fp16 | PASS | Medium (7.6% weight error < 5% quality threshold expected) |
| K1034 | 2-bit quality ≥ 85% of fp16 | UNCERTAIN | Low (33.4% weight error may be too large) |
| K1035 | 4-bit size < 5 MB | PASS | High (~1.5 MB analytical) |
| K1036 | \|cos_4bit\| < 0.05 | UNCERTAIN | Medium (upper bound 0.077 ≈ threshold) |

**Type:** Guided exploration — the quantization error bound is proven, but quality retention
and orthogonality perturbation are empirically measured. K1034 and K1036 may be killed.

---

## Efficient Cosine Computation (r×r Trace Trick)

For large ΔW = A @ B (d_in × d_out, too large to materialize):

```
⟨ΔW_i, ΔW_j⟩_F = trace(ΔW_i^T ΔW_j)
                 = trace(B_i^T A_i^T A_j B_j)
                 = trace((A_i^T A_j)(B_j B_i^T))   [cycle: all r×r!]

‖ΔW_i‖_F^2 = trace((A_i^T A_i)(B_i B_i^T))
```

Cost: O(L × r³) = O(42 × 6³) ≈ 9,072 FLOPs (negligible).
Memory: O(r²) = O(36) elements per layer (no d_in × d_out materialization).
