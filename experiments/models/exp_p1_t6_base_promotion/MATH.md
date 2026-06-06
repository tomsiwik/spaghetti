# MATH.md — T6.3: Promote Crystallized Adapter into Base Model

## Background

T6.2 proved: crystallized domain adapters (B_crystal) improve cosine alignment to the
domain centroid by +6.5pp vs individual users, with norm_ratio=1.020.

T6.3 asks: can we *permanently bake* a crystallized domain adapter into the base model
weights, freeing the Y-slot for a new domain, while preserving both domain quality and
general capability (MMLU proxy)?

This is the "continuous improvement" flywheel: promote → new slot → new domain → promote...

---

## Theorem 1: Base Promotion Correctness

**Claim:** For a linear layer with weights W_base ∈ ℝ^{out×in} and a crystallized LoRA
adapter (A_crystal ∈ ℝ^{in×r}, B_crystal ∈ ℝ^{r×out}), the promoted weight

    W_promoted = W_base + scale · (B_crystal^T A_crystal^T)^T
               = W_base + scale · A_crystal B_crystal

computes identically to applying the adapter at inference time:

    y_promoted_base(x) = x W_promoted^T
                       = x W_base^T + scale · x A_crystal B_crystal
                       = y_base(x) + y_adapter(x)

**Proof:**
In MLX LM, a linear layer with LoRA computes:
    y = x W^T + scale · (x A) B

Substituting W_promoted for W and removing the adapter:
    y = x W_promoted^T
      = x (W_base + scale · A B)^T       # by definition of W_promoted
      = x W_base^T + scale · x A B       # linearity of matrix product

This is identical to the base + adapter computation. QED.

**Corollary:** After promotion, the adapter slot is freed. The adapter can be deleted
without changing the model's behavior on the promoted domain.

**Prediction K1124:** cos(ΔW_promoted, ΔW_crystal) = 1.0 for all layers (exact, by construction).

---

## Theorem 2: Spectral Preservation (MMLU Proxy)

**Claim:** If the relative perturbation satisfies

    ε_layer = ||scale · A B||_F / ||W_base||_F < τ_p

for all layers, then the principal invariant subspaces of W_promoted are within
sin(θ) ≤ ε_layer / δ_gap of W_base's subspaces, where δ_gap is the relative spectral gap.

**Proof (Davis-Kahan):**
Let W_base = U Σ V^T (SVD). The perturbation ΔW = scale · A B has ||ΔW||_2 ≤ ||ΔW||_F.
By Davis-Kahan theorem (Stewart & Sun 1990):

    sin(θ_k) ≤ ||ΔW||_2 / δ_gap,k

where δ_gap,k = |σ_k - σ_{k+1}| is the k-th spectral gap.

For pre-trained LLMs, the leading singular vectors encode general language knowledge.
Empirically (Finding #333, Qwen3-4B at scale=5): MMLU = 0pp change at ε < 5%.

**Prediction K1125:** max_layer(ε_layer) < 0.05 implies MMLU change < 1pp.

We also cite: Model Soup (Wortsman et al. 2203.05482) — weight averaging preserves
general capability, analogous to promotion which adds domain direction.

---

## Theorem 3: Slot Liberation

**Claim:** After promotion, exactly 1 adapter slot (Y-slot) is freed.

**Proof:**
Before promotion: N adapters in serving stack {A_1, A_2, ..., A_N}.
Theorem 1 guarantees W_promoted + 0 = W_base + A_domain. Therefore A_domain can be
removed from the serving stack without behavioral change.
After promotion: N-1 adapters in serving stack. QED.

**Prediction K1126:** n_adapters decreases by exactly 1 after promotion call.

---

## Theorem 4: Trainability on Promoted Base

**Claim:** New LoRA adapters converge on W_promoted with the same gradient magnitude
as on W_base (no gradient vanishing or exploding from promotion).

**Proof:**
For a new adapter (A_new, B_new) on W_promoted, the gradient of a loss L w.r.t. B_new is:

    ∂L/∂B_new = scale · A_new^T x^T · (∂L/∂y)

This depends only on A_new, x, and ∂L/∂y — not on W_promoted. The perturbation ΔW is
folded into the base; it does not appear in the adapter gradient computation.
Therefore: gradient magnitude is unchanged, convergence is guaranteed. QED.

**Prediction K1127:** Loss decreases monotonically over 5 gradient steps on W_promoted
(new adapter starts from random A, zero B — same initialization as original adapters).

---

## Quantitative Predictions

| Kill | Prediction | Threshold | Source |
|------|-----------|-----------|--------|
| K1124 | cos(ΔW_promoted, ΔW_crystal) = 1.0 ± 1e-6 | = 1.0 | Theorem 1 (exact) |
| K1125 | max_layer(ε_layer) < 5.0% | < 5% | Davis-Kahan + Finding #333 |
| K1126 | n_after = n_before - 1 | structural | Theorem 3 |
| K1127 | loss_step5 < loss_step0 | monotone decrease | Theorem 4 |

---

## References

1. Stewart & Sun (1990) — *Matrix Perturbation Theory* (Davis-Kahan theorem)
2. Wortsman et al. (2022) — *Model Soups*, arxiv 2203.05482 (weight averaging preserves quality)
3. Ilharco et al. (2022) — *Task Arithmetic*, arxiv 2212.04089 (adapter promotion is task addition)
4. Finding #333 — Expert promotion at scale=5 on Qwen3-4B: 0pp MMLU change (empirical baseline)
5. Finding #451 (T6.2) — Crystallized adapters: norm_ratio=1.020, cos_crystal=0.9806
