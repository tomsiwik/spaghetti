# Attention-Layer Orthogonality at Macro Scale

## Setup

Given N LoRA adapters trained on distinct domains, each adapter modifies both
attention (q, k, v, o projections) and MLP (gate, up, down projections) layers.

For adapter pair (i, j), define the flattened attention delta vector:

    δ_attn_i = flatten([B_q A_q, B_k A_k, B_v A_v, B_o A_o]_i across all layers)

Similarly for MLP:

    δ_mlp_i = flatten([B_gate A_gate, B_up A_up, B_down A_down]_i across all layers)

## Metric

Pairwise absolute cosine similarity:

    cos(i, j) = |⟨δ_i, δ_j⟩| / (‖δ_i‖ · ‖δ_j‖)

## Theoretical Bound

From concentration of measure on Grassmannian Gr(r, d):

    E[cos] ≤ sqrt(r/d)

For Qwen2.5-7B with d=3584, r=16:

    sqrt(16/3584) = sqrt(1/224) ≈ 0.0668

Note: this bound applies to the per-module subspace overlap. The flattened
delta vector has dimension D = sum over all layers of (d_in × r + r × d_out)
for each attention projection. D >> d, so concentration is even stronger.

## Hypothesis

For dissimilar domains (different clusters: STEM, programming, writing,
reasoning, professional), attention-layer cosines should be:
1. Below sqrt(r/d) = 0.0668 for >80% of pairs
2. Below 0.1 for all pairs

## Kill Criteria

K1: Attention cos exceeds sqrt(r/d) for >20% of dissimilar domain pairs
K2: Attention cos for dissimilar domains exceeds 0.1 at d≥896

## Prior Evidence

- Structural orthogonality proof (micro): MLP cos = 0.002-0.021 at d=64-1024,
  all far below sqrt(r/d). Tested MLP-only.
- FFN-only vs all-modules (macro): attention cos=0.85 for math-medical (RELATED
  domains within STEM cluster). This is a within-cluster measurement and does not
  test the dissimilar-domain hypothesis.
- Pilot 50 composition: cos=0.0002 at d=896 (all-modules, dissimilar pairs) —
  but this averaged over all modules, not attention-specific.

## Design

- 50 pilot adapters across 5 domain clusters
- All-modules LoRA rank-16 on Qwen2.5-7B (d=3584)
- Compute attention-only and MLP-only cosines separately
- Classify pairs as similar (same cluster) or dissimilar (different cluster)
- Per-transformer-layer breakdown to identify which layers have highest overlap
- CPU-only computation (no GPU needed, ~5 min estimated)
