# T1.6: Algorithm Bake-off — LoRA vs HRA at Equal Params

TYPE: verification
PRIOR MATH:
- HRA paper (arxiv 2405.17484), Table 2: HRA matches/beats LoRA at equal params on LLaMA-2
- T1.1 (Finding #415): sr(H_r...H_1) ≈ r EXACT, HRA 2× fewer params than LoRA at same rank
- T1.2 (Finding #416): KILLED — equal-rank ≠ equal-params; HRA r=16 (40k) vs LoRA r=16 (106k)
  Impossibility: (1) capacity disadvantage, (2) multiplicative vs additive update
- T1.5 (Finding #419): KILLED — V-collapse from rank-1 gradient; PoLAR excluded from bake-off

## Background

T1.2 killed because comparison was at equal RANK, not equal PARAMS. HRA r=16 has 40k params
vs LoRA r=16 has 106k params. The HRA paper explicitly compares at equal params.

T1.5 killed: PoLAR's V-collapse requires joint U×V product-manifold retraction. Excluded.
Cayley (T1.4): GPU linalg.inv not available in MLX 0.29.x → CPU-only, too slow for training.
Givens (T1.3): 16 params at r=16 → insufficient capacity for SFT tasks. Excluded.

This experiment corrects T1.2 by comparing at equal parameter budgets.

## Theorem 1: Equal-Params HRA Capacity

**Setup:** Qwen3-4B q_proj with d_in = 2560, d_out = 4096, 36 layers.

**Param counts:**
- LoRA r=6:  A(2560,6) + B(4096,6) = 39,936 params/layer  (1.44M total)
- HRA r=16:  V(16,2560)             = 40,960 params/layer  (1.47M total)
- LoRA r=16: A(2560,16) + B(4096,16) = 106,496 params/layer (3.83M total)
- HRA r=42:  V(42,2560)             = 107,520 params/layer  (3.87M total)

Low budget: LoRA r=6 ≈ HRA r=16 ≈ 40k params/layer
High budget: LoRA r=16 ≈ HRA r=42 ≈ 107k params/layer

**Theorem 1 (HRA Effective Rank):**
HRA r=16 has the same stable rank as LoRA r=16 at 38% of the parameter cost.
From T1.1 (Finding #415): sr(H_1...H_r - I) = r EXACTLY (algebraic result).
At equal params (HRA r=42 ≈ LoRA r=16): HRA has sr=42 vs LoRA's sr≈16.
This gives HRA a HIGHER effective rank at the same parameter budget.

**Proof sketch:**
1. HRA ΔW_eff = W_base(H_r...H_1 - I). The factor (H_r...H_1 - I) has sr = r from T1.1.
2. LoRA ΔW = B@A. sr(B@A) ≤ min(r, sr(A), sr(B)) = r (random init → sr≈r from T1.1 correction).
3. At equal params r_HRA × d_in = r_LoRA × (d_in + d_out):
   r_HRA = r_LoRA × (1 + d_out/d_in) = 16 × (1 + 4096/2560) = 16 × 2.6 = 41.6 → r_HRA = 42
4. sr_HRA = 42 > sr_LoRA = 16 at same param budget. Higher sr = broader adaptation. □

## Theorem 2: Adapter Orthogonality in High-Dimensional Param Space

For two adapters A₁, A₂ ∈ ℝᴺ with N >> 1, initialized independently:

**Theorem 2 (JL Orthogonality, from T0.1 Finding #417):**
cos(A₁, A₂) ~ N(0, 1/N). For N = 1,474,560 (HRA r=16 total):
E[|cos|] = sqrt(2/π) / sqrt(N) ≈ 5.2 × 10⁻⁴

Predicted: |cos(math adapter, code adapter)| ≈ 10⁻³ << 0.01 threshold.

After training on different domains: gradient vectors g_math, g_code are not perfectly
orthogonal (both domains involve logical reasoning), but parameter updates accumulate
with random noise → final cos stays << 0.01 for large N.

## Theorem 3: Stable Rank After SFT

For winner config with r_eff reflections/rank:
sr(V) where V ∈ ℝ^{r×d_in} after training:
- Random Marchenko-Pastur predicts sr_init ≈ r (all singular values equal, spectral flatness)
- SFT on rank-1 GSM8K gradient: one dominant direction, sr decreases
- Lower bound: sr ≥ sqrt(r) by sub-Gaussian matrix concentration
- For r=42: sr ≥ sqrt(42) ≈ 6.5 > 3 (K1026 threshold) — predicted PASS

## Quantitative Predictions

| Prediction | Value | Kill Criterion |
|-----------|-------|---------------|
| P1: HRA r=42 GSM8K ≥ LoRA r=16 GSM8K | measured | K1024 winner |
| P2: composite(HRA r=42) ≥ composite(LoRA r=16) | HRA r=42 wins due to sr_eff > sr_LoRA | K1024 |
| P3: \|cos\| < 0.01 (math vs code adapters) | ~10⁻³ from Theorem 2 | K1025 |
| P4: sr(winner) ≥ 3 | sr ≥ 6.5 from Theorem 3 | K1026 |
| P5: all configs train ≤ 1 hour | ~5 min per config × 4 = 20 min | K1027 |

## Kill Criteria

| ID | Criterion | PASS Condition |
|----|----------|----------------|
| K1024 | Winner identified (quality × 1/params × 1/time) | max composite score config identified |
| K1025 | Winner adapters orthogonal for 2 domains | \|cos\| < 0.01 |
| K1026 | Winner stable rank ≥ 3 at nominal rank | sr ≥ 3 |
| K1027 | All configs train ≤ 1 hour | max training time ≤ 3600s |

## Impossibility Structure (from killed experiments)

**Cayley excluded:** MLX 0.29.x linalg.inv/solve is CPU-only (T1.4). Training 36 layers with
CPU linear solves at each step → prohibitively slow. Fix: wait for MLX GPU linalg support.

**PoLAR excluded:** V-collapse from rank-1 GSM8K gradient (T1.5). Fix: joint U×V product-manifold
retraction. This retraction is nontrivial in MLX without Riemannian Adam. Excluded pending Fix.

**Givens excluded:** At r=16 (16 angle params), insufficient capacity for SFT (T1.3 verified isometry
only). To match LoRA r=6 params would require r≈20,000 rotation pairs (d_in/2=1280 × 16 params each
if using the full rotation budget) — not the sparse Givens regime.
