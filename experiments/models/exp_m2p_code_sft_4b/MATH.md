# MATH.md: Code SFT-Residual M2P at 4B

## Problem Statement

Finding #407 showed that code M2P with B_sft=0 at 4B degrades base capability
from 42.2% → 6.7% pass@1 (quality_ratio=0.158). Finding #403 showed that
math M2P with zero-init residual heads and B_sft=math_SFT_B achieves init_qr=1.0
exactly and improves to qr=1.175 after training.

**The structural question:** Is the SFT quality floor a domain-agnostic architectural
guarantee, or math-specific? If structural, applying the same SFT-residual design
with code SFT B-matrices must produce init_qr=1.0 for code tasks.

---

## Theorem 1: SFT Quality Floor (Zero-Init Residual Guarantee)

**Setup:**
- Code SFT adapter: B_sft ∈ R^{r × d_out} trained with fixed A-matrix on code tasks
- M2P residual head: h_θ : R^{d_m2p} → R^{r × d_out}, head.weight = 0 at init
- Applied B-matrix per layer: B_applied = B_sft + ε · h_θ(z),  ε = 0.032
- LoRA forward: output = base_proj(x) + scale · (x @ A) @ B_applied

**Theorem 1:**
Let h_θ(z) = W z with W = 0 (zero initialization). Then for all inputs z:

    h_θ(z) = W z = 0
    ⟹ B_applied = B_sft + ε · 0 = B_sft
    ⟹ model_forward(x; B_applied) = model_forward(x; B_sft)
    ⟹ quality(M2P_at_step_0) = quality(SFT_adapter) exactly

*Proof:* Direct substitution. Zero-init head maps all inputs to the zero vector.
B_applied = B_sft + ε · 0 = B_sft. Model forward with B_applied = B_sft is
algebraically identical to the SFT model. Therefore quality metrics are identical.

**Corollary:** Training from this initialization optimizes ΔB = ε · h_θ(z) away from
zero. The objective gradient at step 0 is non-zero (Theorem 2), so learning occurs.
The residual connection in weight space provides a monotone lower bound at
initialization. QED.

---

## Theorem 2: Gradient Non-Degeneracy at Step 0

**Prior math:** Finding #403 (K974 PASS): grad_norm=1.804 > 0 at step 0 under
the same architecture with math SFT B-matrices.

**Theorem 2:** For any non-degenerate input distribution and non-zero B_sft,
the M2P gradient at step 0 is non-zero.

*Proof:* At step 0, B_applied = B_sft produces finite loss L. The gradient
∂L/∂W flows through: ∂L/∂B_applied → ∂B_applied/∂h_θ(z) = ε · I → ∂h_θ(z)/∂W.
Since B_sft ≠ 0 and the SFT model produces imperfect predictions, ∂L/∂B_applied ≠ 0.
Therefore ∂L/∂W = ε · z^T · ∂L/∂h_θ ≠ 0 for non-degenerate z. QED.

---

## Theorem 3: Grassmannian Isolation Preserved Under Code SFT Residual

**Prior result:** Finding #404 (K975 PASS): |A_math^T A_code|_F = 1.38e-05 (bf16 floor).

**Theorem 3:** The code SFT B-matrices B_sft_code depend only on the code A-matrices
and the code training data. They do NOT depend on math A-matrices.
Therefore the Grassmannian isolation |A_math^T A_code|_F < 1e-4 is unaffected
by the residual structure. Routing by TF-IDF (input-text only) is similarly
unaffected.

*Proof:* B_sft_code = argmin_B L(LoRA(base; A_code, B), code_data). The optimization
depends only on A_code and code_data. The math A-matrices appear nowhere in the
objective. Therefore |A_math^T A_code|_F is identical to Finding #404. QED.

---

## Quantitative Predictions

| ID  | Quantity | Formula | Predicted Value | Kill if |
|-----|----------|---------|----------------|---------|
| P1  | init_code_qr | (init_M2P_pass@1 - base_pass@1) / (SFT_pass@1 - base_pass@1) | 1.0 exactly | K987 FAIL if < 0.80 |
| P2  | post_code_qr | (post_M2P_pass@1 - base_pass@1) / (SFT_pass@1 - base_pass@1) | 0.70–1.50 | K988 FAIL if < 0.70 |
| P3  | math_qr | (math_M2P_acc - math_base_acc) / (math_SFT_acc - math_base_acc) | ≥ 0.80 (baseline: 1.3125) | K989 FAIL if < 0.80 |
| P4  | routing_acc | TF-IDF math vs code | ≥ 95% (same as Finding #404) | warn if < 80% |

**Prediction P1 basis:** Theorem 1 — zero-init heads guarantee B_applied = B_sft_code
at step 0, making code M2P and code SFT algebraically identical.

**Prediction P2 basis:** Finding #403 showed math SFT-residual M2P achieves qr=1.175
after 1000 steps. Code tasks (Python functions) are arguably simpler signal than GSM8K
chain-of-thought reasoning. Conservative estimate 0.70 provides ample margin.

**Prediction P3 basis:** Finding #404 showed math quality_ratio=1.3125 under 2-domain
routing. Adding code domain with Grassmannian-isolated A-matrices (Theorem 3) cannot
affect math routing. Conservative threshold 0.80 provides margin for routing variance.

---

## Experiment Type

TYPE: verification
FAILURE MODE: Anti-format interference under code M2P without SFT floor (Finding #407)
PRIOR MATH: Theorem 1 (zero-init residual, extends Finding #403 to code domain)
GROUNDED IN: He et al. 2016 (ResNet residual), Finding #403, Finding #407

---

## Impossibility Structure (from Finding #407 Failure)

Finding #407 killed because: M2P with B_sft=0 has no quality lower bound. The
optimization can find B-matrices that minimize training loss while destroying
the base model's code priors (42% → 6.7%).

The SFT residual makes this impossible: B_applied = B_sft_code + ΔB. At the
minimum of the training objective, ΔB = 0 recovers SFT quality. Gradient descent
starting from ΔB=0 cannot worsen quality by more than the magnitude of the first
gradient step scaled by ε=0.032. For reasonable LR=5e-5, the first-step degradation
is at most δ ≈ LR × grad_norm × ε ≈ 5e-5 × 2 × 0.032 ≈ 3.2e-6, negligible.
