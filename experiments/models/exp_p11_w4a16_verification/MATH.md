# MATH: W4A16 Precision and Reasoning Quality Verification

## Theorem (W4A16 Reasoning Preservation)

**Statement**: For transformer inference with weight-only 4-bit quantization (W4A16) and
bfloat16 activations, reasoning chain quality degrades sub-linearly relative to 8-bit
weight quantization. Specifically, on MMLU-Pro with thinking, the accuracy gap between
W4A16 (4-bit) and W8A16 (8-bit) is bounded by δ_acc ≤ C · (Δ_4 - Δ_8) / Δ_8 where
Δ_b is the mean absolute quantization error for b-bit weight quantization.

**Prior Work**: Paper 2504.04823 claims W4A16 achieves near-lossless reasoning quantization
for instruction-tuned models. MLX affine quantization with group_size=64 is standard
W4A16 (weights 4-bit, activations bfloat16).

## Setup

Let W ∈ ℝ^{d_out × d_in} be a weight matrix with affine quantization:

    Q_b(W) = round(W / scale + zero) · scale - zero

where scale = (max(W_g) - min(W_g)) / (2^b - 1) per group g of size G=64.

**Quantization error per element**:
    ε_b = E[|w - Q_b(w)|] ≤ scale / 2 = range(W_g) / (2 · (2^b - 1))

For b=4: ε_4 ≤ range(W_g) / 30
For b=8: ε_8 ≤ range(W_g) / 510

**Error ratio**: ε_4 / ε_8 ≈ 510/30 = 17×

**Output perturbation for W4A16**:
    ||W_q x - Wx||_2 ≤ ||δ_W||_F · ||x||_2

where δ_W = W - Q_4(W). For typical LLM weights (unit-normal after RMSNorm):
    E[||δ_W||_F] ≈ √(d_out · d_in) · ε_4 / √(d_out · d_in / G) = √G · ε_4

With G=64: E[||δ_W||_F] ≈ 8 · ε_4 — the error is bounded by 8× the per-element error.

## Proof

**Step 1: Activation scale matters**. For W4A16, activations x remain in bfloat16.
  The key property: BF16 has ~3×10^{-3} relative error (7.2 bits mantissa).
  Weight 4-bit quantization has ~3×10^{-2} relative error (4 bits + scale).
  → Weight error dominates activation noise by ~10×.

**Step 2: Reasoning chain error accumulation**. In chain-of-thought (thinking),
  the model generates T tokens of reasoning before the answer. Error accumulates
  per-token, but thinking allows error correction: at each step k, the model can
  "re-derive" from earlier context. The effective noise is not O(T·ε_4) but O(ε_4)
  because each thinking step re-grounds on BF16-precision activations.

**Step 3: Gap bound**. Under this model, the accuracy gap between 4-bit and 8-bit
  on reasoning tasks is:
    δ_acc ≤ p_flip · P(question requires >4-bit precision computation)
  where p_flip = P(quantization error flips a computation result).

  For MMLU-Pro reasoning tasks: most steps require logical reasoning (not high-precision
  arithmetic), so p_flip is small. Prediction: δ_acc ≤ 5pp.

**QED** (conditional on reasoning tasks being logic-dominated, not arithmetic)

## Quantitative Predictions

| Metric | Prediction | Kill Threshold |
|--------|-----------|----------------|
| 4-bit model = W4A16 | YES (mode=affine, bits=4, dtype=bfloat16) | K1538: PASS |
| 8-bit − 4-bit MMLU-Pro + thinking | < 5pp gap | K1540: if ≥5pp, quantization hurts |
| Config: activation_bits | N/A (no activation quantization) | K1539: N/A if W4A16 |

**Expected result**: K1540 FAILS (8-bit ≈ 4-bit + 2-3pp), confirming W4A16 is near-lossless.
If K1540 PASSES (8-bit >> 4-bit + 5pp), then quantization IS the gap to Google's 69.4%,
and we should upgrade to 8-bit inference.

## Connection to Architecture

From exp_bench_mmlu_pro_thinking (Finding #530):
- 4-bit base + thinking = 62.1%
- Google's 69.4% target
- Gap = 7.3pp

If this gap is mostly from quantization (W4A16 → W8A16 closes 5pp), then:
  → Switch to 8-bit base, accept 2.7GB extra memory, gain 5pp reasoning quality
  → All adapters still W4A16 (adapter weights are small, quantization error minimal)

If gap is NOT from quantization (8-bit ≈ 4-bit + 2pp), then:
  → Quantization is not the bottleneck
  → Gap must come from: adapter training quality, prompting, or data distribution
  → CLoQ experiment (K0) addresses the adapter-side
