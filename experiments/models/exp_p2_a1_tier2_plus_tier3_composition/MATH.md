# MATH.md — Tier 2 + Tier 3 Simultaneous Activation

## Problem Statement

Does activating a domain adapter (Tier 2) and a personal style adapter (Tier 3)
simultaneously preserve both behaviors? The vision 4-tier architecture requires
this to work: when a user asks a math question, they get domain-accurate answers
AND their personal style (e.g., "Hope that helps, friend!").

Prior work (Finding #425) killed all-5-adapter simultaneous activation due to
quality collapse. But that experiment activated ALL adapters with NO routing, not
2 matched adapters. This experiment tests the correct use case: exactly 2 adapters
both relevant to the current query.

## Theorem 1 (Additive Composition Behavioral Orthogonality)

**Setting:** Let ΔW_D = scale_D · lora_b_D · lora_a_D^T and
ΔW_P = scale_P · lora_b_P · lora_a_P^T be the effective weight deltas for a
domain (Tier 2) and personal (Tier 3) adapter respectively. Both target the same
projection (q_proj) at overlapping layers L_overlap.

**Definition:** The behavioral cross-talk at layer l is:
  ε_l = max_{i,j} |cos(b_D^i, b_P^j)|
where b_D^i (resp. b_P^j) is the i-th row of scale_D · lora_b_D (resp. scale_P · lora_b_P).
This measures how much each "output direction" of one adapter points along an
output direction of the other.

**Theorem:** If ε_l < ε_max for all l ∈ L_overlap, then for any input x:

  ||Δh_D + Δh_P||_cos ≤ ε_max

where Δh_D = (x @ lora_a_D) @ lora_b_D · scale_D is the domain adapter's
activation contribution and similarly for Δh_P. The cosine between the two
additive contributions is bounded by ε_max.

**Proof sketch:**
Δh_D = x @ lora_a_D @ lora_b_D · scale_D. The "effective direction" of Δh_D
in hidden space lies in span(columns of (scale_D · lora_b_D)^T). Similarly for Δh_P.
The angle between these subspaces is lower-bounded by the minimum singular value of
their cross-Gram matrix B_D^T B_P, where B_D = scale_D · lora_b_D and B_P = scale_P · lora_b_P.
The max off-diagonal entry of this cross-Gram (normalized) = max_{i,j} |cos(b_D^i, b_P^j)| = ε_l.
If ε_l < ε_max, the two contributions are nearly orthogonal in output space. □

**Corollary (Behavioral Addivity):** When ε_l < ε_max for all overlap layers:
- Domain output: ||output_composed - output_domain_only|| / ||output_domain_only|| ≤ ε_max * |L_overlap| / |L_total|
- Style output: ||output_composed - output_personal_only|| / ||output_personal_only|| ≤ ε_max * |L_overlap| / |L_total|

Both behaviors degrade by at most ε_max × (coverage fraction).

## From Prior Findings

**Finding #427** (exp_p1_t3_activation_space_bounds):
Power law: max_cos = c·N^α = 0.061·N^0.145, R²=0.94 (synthetic adapters).
At N=2: max_cos ≤ 0.061 · 2^0.145 = 0.071 (well below 0.1 threshold).
Real adapters on mismatched inputs: 0.596 (7.6× higher due to correlated lora_a init).
Key caveat: high cosines only for MISMATCHED routing. This experiment uses matched routing.

**Finding #428** (exp_p1_t3_n25_composition):
Grassmannian A-matrices: max|cos|=2.165e-8. But these are synthetically orthogonalized.
Real adapters from T2.1 + T5.1 were trained independently, NOT Grassmannian-orthogonalized.

**Finding #436** (exp_p1_t5_user_local_training):
Personal adapter: rank=4, scale=4.0, layers 26-41 (q_proj only).
Base compliance=0%, adapter compliance=76% (76pp gain).

## Quantitative Predictions

For the math adapter (T2.1: rank=6, scale=6.0, all 42 layers) and
personal adapter (T5.1: rank=4, scale=4.0, layers 26-41):

Overlap fraction: |L_overlap|/|L_total| = 16/42 = 0.381

**K1 prediction (math MCQ):**
- If ε_max < 0.1 → degradation < 0.1 × 0.381 = 3.8% of math-only accuracy
- Math-only accuracy ≈ 82% (from T2.1 PAPER.md)
- Composed accuracy prediction: 82% × (1 - 0.038) ≈ 78.9% → within 5pp of 82%
- K1 threshold: composed within 5pp of math-only (PASS predicted)

**K2 prediction (style compliance):**
- If ε_max < 0.1 → degradation < 3.8% of personal-only compliance
- Personal-only compliance = 76% (Finding #436)
- Composed compliance prediction: 76% × (1 - 0.038) ≈ 73.1% → within 10pp of 76%
- K2 threshold: composed within 10pp of personal-only (PASS predicted)

**K3 prediction (B-matrix cosine):**
- Both adapters trained independently from random initialization on different tasks
- Expected: cos(B_math, B_personal) < 0.2 per pair (uncorrelated tasks → uncorrelated directions)
- Strong prediction: max_cos < 0.1 (approximately orthogonal output subspaces)
- K3 threshold: max_cos < 0.1 (UNCERTAIN — real adapters may exceed this)

## Kill Criteria (from claim)

- K1: Math MCQ accuracy with composed adapter within 5pp of math-only adapter
- K2: Style compliance with composed adapter within 10pp of personal-only adapter
- K3: Max B-matrix cosine between math and personal adapters < 0.1

## Failure Mode

If K3 FAILS (max_cos ≥ 0.1): the two adapters' output subspaces are not sufficiently
orthogonal. The personal adapter adds domain-correlated noise to math outputs (domain
accuracy degrades) AND the math adapter distorts style outputs. This requires
Grassmannian re-orthogonalization of both adapters (as in Finding #428) before composition.

If K1 or K2 FAIL with K3 PASS: behavioral degradation despite low B-matrix cosine suggests
the activation-space interference is larger than predicted (perhaps because lora_a alignment
matters, not just lora_b). Fix: apply Grassmannian A-matrix constraint.

## Composition Implementation

For layers l ∈ {0..25} (math-only):
  lora_a_merged = [lora_a_math | 0_{d_in×4}]  ∈ ℝ^{d_in × 10}
  lora_b_merged = [6.0 × lora_b_math; 0_{4×d_out}] ∈ ℝ^{10 × d_out}
  scale_merged = 1.0

For layers l ∈ {26..41} (overlap):
  lora_a_merged = [lora_a_math | lora_a_personal]  ∈ ℝ^{d_in × 10}
  lora_b_merged = [6.0 × lora_b_math; 4.0 × lora_b_personal] ∈ ℝ^{10 × d_out}
  scale_merged = 1.0

Forward: output += 1.0 × (x @ lora_a_merged) @ lora_b_merged
       = 6.0 × (x @ lora_a_math) @ lora_b_math + 4.0 × (x @ lora_a_personal) @ lora_b_personal ✓

## References

- Finding #427: Activation power law (c=0.061, α=0.145), R²=0.94
- Finding #428: Grassmannian composition N=25, max|cos|=2.165e-8
- Finding #436: Personal adapter 76pp compliance gain
- exp_p1_t2_single_domain_training: Math adapter (rank=6, 1000 steps, all layers)
- exp_p1_t5_user_local_training: Personal adapter (rank=4, 300 steps, layers 26-41)
