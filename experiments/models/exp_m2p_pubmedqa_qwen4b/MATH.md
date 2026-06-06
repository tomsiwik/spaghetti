# MATH.md: SFT-Residual M2P on PubMedQA — Generalization Verification at 4B

## TYPE: guided-exploration
## PROVEN FRAMEWORK: SFT-residual M2P (Finding #403, Theorem 1)
## UNKNOWN: Does SFT-residual M2P generalize to medical QA domain at 4B?

---

## Why This Experiment

Finding #403 proved SFT-residual M2P works for math (GSM8K) at 4B.
Finding #408 proved Grassmannian A-matrices CONFLICT with base capability when base is STRONG.

**Right question (SIGREG):** Under what condition do Grassmannian A-matrices safely improve a domain?

**Answer from existing math:** When the base model is WEAK on domain D, any A-matrix subspace
can improve quality because the base has no established circuits to protect. The conflict in
Finding #408 arose because base Qwen3-4B was already strong at code (37-42% on toy functions)
and the Grassmannian A-matrices disrupted that existing capability.

**Hypothesis:** PubMedQA is a domain where base Qwen3-4B is WEAK (expected 50-65%). In this
regime, Grassmannian A-matrices (new seed, orthogonal to math) avoid subspace conflict, and
SFT-residual M2P generalizes identically to the math case.

---

## Theorem 1: SFT-Residual Quality Floor (inherited from Finding #403)

**Statement:** For any domain D with SFT-trained B_sft[li] and M2P head initialized to zero:

    B_applied[li] = B_sft[li] + output_scale * head(z[li])

At initialization, head(z) = 0 (zero-init), so:

    B_applied[li] = B_sft[li] + 0 = B_sft[li]

The model at initialization IS the SFT model for domain D.

**Corollary:** init_quality_ratio = (SFT_acc - base_acc) / (SFT_acc - base_acc) = 1.0 exactly.

**Proof:** head output = W_head @ z = 0 @ z = 0 for zero-initialized W_head. QED.

**Prediction:** K1 (init_quality_ratio >= 0.80) passes with measured value ≈ 1.00.
This prediction is deterministic — not a confidence interval.

---

## Theorem 2: Subspace Conflict is a Function of Base Strength

**Statement:** Let φ_D(model) = accuracy(model, D) be the base model's capability on domain D.
Let A_G be Grassmannian-random A-matrices (orthogonal to other domains' A-matrices).

Case 1: φ_D(base) is HIGH (domain D is a "strong base" domain):
- The base model has established circuits {W_attn layers} that process domain D efficiently.
- Any LoRA ΔW = A_G * B disrupts these circuits in a subspace that doesn't align with
  the base's processing. Result: quality degrades (Finding #408 observed).

Case 2: φ_D(base) is LOW (domain D is a "weak base" domain):
- The base model has NO established circuits for domain D (it guesses or halluccinates).
- Any LoRA ΔW that improves domain D does not disrupt existing circuits (there are none).
- The gradient points in the improvement direction regardless of A-matrix subspace.
- Result: M2P can improve from the SFT quality floor (Theorem 1 applies).

**Mathematical statement:** Let Φ_D = {W | W improves acc(model+ΔW, D) > acc(model, D)}.
When φ_D(base) is low, |Φ_D| / |R^{d×r}| → 1 (almost all weight updates help).
When φ_D(base) is high, |Φ_D| is small — only updates in the "right" subspace help.

**Prediction for PubMedQA:**
- Expected base_acc: 50-65% (PubMedQA requires specialized biomedical knowledge)
- Random baseline: 33.3% (3-class yes/no/maybe)
- SFT should improve to 65-75% (domain-specific training)
- K3 (base < SFT) should pass with clear margin (≥ 5pp)

---

## Theorem 3: Grassmannian Isolation of Medical vs Math A-Matrices

**Statement (from Finding #404):** Random Grassmannian A-matrices drawn from different seeds
satisfy |A_i^T A_j|_F ≈ bf16_quantization_floor ≈ 1.38e-05 (measured empirically).

For n_layers=36, rank=4, d_model=2560:
- Math A-matrices: seed=0 (from m2p_qwen4b_gsm8k)
- Medical A-matrices: seed=1 (new, generated here)
- Expected: max|A_math^T A_med|_F < 1e-4 across all 36 layer pairs

This ensures that when math M2P and medical M2P are composed in the room model,
their contributions are isolated:
    ΔW_composed = A_math * B_math^T + A_med * B_med^T
The cross-terms A_math^T * A_med ≈ 0, so the domains do not interfere.

**Prediction:** All 36 layer pairs show |A_math^T A_med|_F < 1e-4.

---

## Kill Criteria

| K# | Criterion | Predicted | Theorem |
|----|-----------|-----------|---------|
| K1137 | init_quality_ratio >= 0.80 | 1.00 (exact) | Theorem 1 |
| K1138 | quality_ratio >= 0.60 after 1000 M2P steps | 0.70-1.20 | Theorem 1+2 |
| K1139 | base_accuracy < SFT_accuracy (domain weakness) | 5-15pp gap | Theorem 2 |

## Predictions Table

| Metric | Predicted Value | Reasoning |
|--------|----------------|-----------|
| base_accuracy (PubMedQA) | 50-65% | Specialized biomedical QA |
| SFT_accuracy (300 steps) | 65-75% | Domain fine-tuning helps |
| init_quality_ratio | 1.00 (exact) | Theorem 1: zero-init heads |
| quality_ratio at 1000 steps | 0.70-1.20 | Residual refinement from SFT floor |
| max Grassmannian isolation | < 1e-4 | Theorem 3 from Finding #404 |
| Runtime | ~75-90 min | SFT 300 + M2P 1000 + evals on M5 Pro 48GB |
| Peak memory | ~18-20 GB | Similar to Finding #403 (17.91 GB) |

---

## Self-Test

1. **What makes failure impossible?** Theorem 1: zero-init heads → B_applied = B_sft at step 0.
   The medical adapter IS the SFT adapter at initialization. K1 is guaranteed by construction.

2. **Cited theorems:** Finding #403 (SFT-residual math M2P). Finding #408 (A-matrix conflict
   analysis). Finding #404 (Grassmannian isolation 1.38e-05). He et al. 2016 (residual networks).

3. **What would falsify this?** If base_acc >= SFT_acc on PubMedQA (K3 FAIL), the domain
   is not a "weak base" domain and our prediction is wrong. Alternative: use a different
   domain where base is demonstrably weak.

4. **Failure mode to watch:** If PubMedQA answers are very long (beyond MAX_GEN_TOKENS),
   truncation could mask quality. We set MAX_GEN_TOKENS=16 since yes/no/maybe is short.

---

## Connection to Vision

This experiment advances from "math M2P at 4B" (Finding #403) to "multiple-domain M2P at 4B".
The room model requires: W_combined = W_base + Σ A_i * B_i^T where B_i = f_M2P_i(query).
For N=25 domains, we need 25 independently verified M2P networks.
This experiment provides domain #2.

## References

- Finding #403: SFT-Residual M2P on GSM8K, quality_ratio=1.175 at 4B
- Finding #408: A-matrix subspace conflict destroys code capability
- Finding #404: Grassmannian isolation |A^T A|=1.38e-05 at 4B
- He et al. (2016): Deep Residual Learning (arxiv:1512.03385)
- Aghajanyan et al. (2020, arxiv:2012.13255): Intrinsic Dimensionality
- PubMedQA: Jin et al. (2019), arxiv:1909.06146, 3-class biomedical QA
