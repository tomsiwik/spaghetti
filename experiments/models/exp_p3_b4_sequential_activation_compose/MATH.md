# MATH.md — P3.B4: Pure Additive Composition as Interference-Free Baseline

## Background

**P3.B1 (Finding #462)**: Gram-Schmidt B-matrix orthogonalization gives style=60% (Δ=16pp loss).
B-GS projection MODIFIES B_P to make it ⊥ B_D. This modification may itself destroy some style.

**P3.B3 (Finding #464)**: Full ΔW null-space projection (α=1.0) → style=0%.
All style directions of ΔW_P lie in col(ΔW_D). Removing them removes style entirely.

**Gap**: No experiment has ever tested pure additive composition (rank-10 concatenation, no
projection, no modification to either adapter). P3.B1 always used B-GS-modified adapters.
P2.A1 tested simultaneous composition but was killed at smoke N=5 (too noisy to conclude).

**Sequential cross-term hypothesis (eliminated)**: The cross-term ΔW_D @ ΔW_P requires
d_out(ΔW_D) = d_in(ΔW_P) for valid matrix multiplication. For q_proj:
  - d_in = 2560 (hidden_size)
  - d_out = 2048 or 4096 (query dimension, varies by layer)
Since d_out ≠ d_in, the cross-term is NOT computable. Sequential composition via weight-space
correction is architecturally impossible for q_proj in Gemma 4.

---

## Theorem 1: Pure Additive Preserves Both Signals Unmodified

**Statement**: For pure additive composition of LoRA adapters ΔW_D = s_D A_D B_D and
ΔW_P = s_P A_P B_P:

    h_add = x @ W + x @ ΔW_D + x @ ΔW_P

Both ΔW_D and ΔW_P are applied UNMODIFIED to the same input x. Neither adapter's signal
directions (col(ΔW_D), col(ΔW_P)) are removed or projected.

**Prediction**: Since col(ΔW_P) ⊆ col(ΔW_D) (Finding #464), the style signal in col(ΔW_P)
is fully present in the additive output. The domain and style contributions coexist without
any projection-induced destruction.

**Why B-GS hurt**: B-GS projects B_P' = B_P - B_P @ Q_D @ Q_D^T, removing the components
of B_P that lie in row(B_D). Since style is encoded in B_P rows that align with B_D, B-GS
REMOVES some style — contradicting its purpose.

**QED** (trivial: pure additive modifies nothing)

---

## Theorem 2: Style Degradation in Pure Additive is Not from Projection

**Statement**: Any style degradation in pure additive h_add is NOT from projection (since no
projection occurs). It must come from one of:
(a) Power imbalance: ΔW_D is stronger → dilutes ΔW_P's style signal in shared directions
(b) Non-linear context shift: domain adapter changes attention patterns in early layers (0-25)
    before personal adapter activates (layers 26-41)
(c) A-matrix cross-term: A_P^T @ A_D ≠ 0 creates a second-order interaction in the forward pass

**Proof sketch**: For the pure additive forward pass:
    output = x @ W + s_D × x @ A_D @ B_D + s_P × x @ A_P @ B_P

The style contribution s_P × x @ A_P @ B_P is intact. However, the model's final output
depends on ALL attention layers. The domain adapter's effect in layers 0-25 (where personal
adapter doesn't fire) biases the hidden state toward domain-specific patterns before the
personal adapter engages in layers 26-41.

This is the "context shift" hypothesis: NOT weight-space interference, but distribution shift
in the hidden states that the personal adapter receives.

**QED (sketch)**

---

## Predictions (Kill Criteria)

| Kill Criterion | Predicted Value | Rationale |
|---|---|---|
| K1189: max Frobenius B-matrix cosine per layer (diagnostic) | ~ 0.16 (same as P3.B1) | Unchanged adapters, same geometry |
| K1191: style compliance, pure additive | ≥ 66% (target) | B-GS was harming style; pure additive should recover |
| K1192: math MCQ accuracy | ≥ 5% | Domain adapter unchanged |

**Key comparison**: P3.B1 (B-GS) = 60% style. Pure additive prediction = ≥ 66%.

If K1191 PASS (≥ 66%): B-GS modification was destroying style, pure additive is the right
composition method. The 16pp loss in P3.B1 included ~10pp from B-GS itself.

If K1191 FAIL (< 66%): Pure additive is no better than B-GS. The style loss is from context
shift (domain adapter biases early layers) rather than weight-space interference.
Next step: P3.B5 — retrain personal adapter ON TOP of domain-adapted model.

---

## Adapter Architecture (Pure Additive)

**Input adapters** (unchanged, from prior experiments):
- Math adapter: rank=6, scale=6.0, all 42 q_proj layers, dir: exp_p1_t2_single_domain_training/adapters/math
- Personal adapter: rank=4, scale=4.0, layers 26-41 q_proj, dir: exp_p1_t5_user_local_training/personal_adapter

**Pure additive merged adapter** (rank-10, scale=1.0, baked):
For each overlap layer l ∈ {26,...,41}:
    A_merged = [A_D_l | A_P_l]: [d_in, r_D + r_P] = [2560, 10]
    B_merged = [s_D × B_D_l; s_P × B_P_l]: [r_D + r_P, d_out] = [10, 2048]

For math-only layers l ∈ {0,...,25}:
    A_merged = [A_D_l | 0_pad]: [2560, 10]
    B_merged = [s_D × B_D_l; 0_pad]: [10, 2048]

No modification to either adapter. Both scales baked into B.

---

## Connection to Prior Literature

- **LoRA (Hu et al., 2021, arxiv:2110.04367)**: Standard additive composition ΔW = AB.
  Our pure additive is the most faithful implementation: ΔW_eff = ΔW_D + ΔW_P.

- **Finding #462** (P3.B1): B-GS projection achieves 60% style (Δ=16pp). This experiment
  establishes whether the 16pp was due to B-GS modification or inherent interference.

- **Finding #464** (P3.B3): Full projection → 0% style. Confirms style information is
  in col(ΔW_D). Additive composition keeps this information; projection removes it.
