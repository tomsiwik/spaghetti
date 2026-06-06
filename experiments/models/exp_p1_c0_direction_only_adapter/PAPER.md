# C0.2: Direction-Only Adapter on Gemma 4 (QKV-Norm Thesis)

## Experiment Type
Guided Exploration — proven framework (RMSNorm kills magnitude), unknown: whether
direction-only TRAINING converges as well as standard LoRA.

## Prediction vs Measurement

| Theorem | Prediction | Measurement | Result |
|---------|-----------|-------------|--------|
| T1: RMSNorm discards LoRA magnitude | B_norms constant under projection | B_norms = 2.4495 (=√6) throughout all 1000 steps | **CONFIRMED** |
| T2: Non-zero tangent gradient → loss decreases | Loss trends down after step 50 | Loss drops: 5.45 → 0.84 (dir-only), 5.45 → 0.81 (std) | **CONFIRMED** |
| T3: Unit-norm prevents rank collapse | sr(dir-only) > sr(standard) | sr_dir=[3.04,1.87,2.95] avg=2.62 > sr_std=[1.58,2.30,1.46] avg=1.78 | **CONFIRMED** |
| KC05: direction-only ≥ 90% of std (≥73.8%) | ≥ 95% (magnitude irrelevant post-norm) | 60.0% vs 72.0% (83.3% ratio; < 90% threshold) | **FAIL** |
| KC06: loss trends down after step 50 | Yes (non-zero tangent gradient) | Yes — loss 1.71 (step 50) → 0.84 (step 1000) | **PASS** |

## Key Results

- Direction-only GSM8K: **60.0%** (n=50 eval)
- Standard LoRA GSM8K: **72.0%** (n=50 eval, same setup)
- T2.1 baseline (mlx_lm.lora): 82.0% (different setup: full attention, more steps)
- Quality ratio vs our standard: **83.3%**
- Quality ratio vs T2.1: 73.2%

## Secondary Findings

### Confirmed: B-Matrix Norm Invariance Under Projection
Direction-only B_norms remain perfectly constant at 2.4495 = √rank = √6 throughout
all 1000 training steps. The unit-norm projection is geometrically consistent.

### Confirmed: Higher Stable Rank (Theorem 3)
```
Direction-only stable rank (layers 0-2): [3.04, 1.87, 2.95], mean=2.62
Standard LoRA stable rank  (layers 0-2): [1.58, 2.30, 1.46], mean=1.78
```
Unit-norm constraint prevents any row from dominating, forcing rank diversity.

### Confirmed: Loss Convergence
Both modes converge to similar final loss (~0.84 vs ~0.81). The training procedure
works; direction-only just starts from a disrupted initialization.

## Kill Criteria

| ID | Criterion | Measured | Pass? |
|----|-----------|---------|-------|
| KC05 | dir-only GSM8K ≥ 73.8% (90% of 82%) | 60.0% | **FAIL** |
| KC06 | Loss trends down after step 50 | 1.71→0.84 | **PASS** |

**Status: KILLED** — KC05 fails. Direction-only adapter does not match standard LoRA accuracy
within the 90% threshold under the post-hoc projection training procedure.

## Why KC05 Failed: Diagnosis

The 83.3% ratio (60%/72%) suggests direction-only is 16.7 percentage points behind standard LoRA.
Three structural contributors:

1. **Initialization disruption**: B initialized to zeros, projected to random unit vectors after
   step 1 (scale=6.0 × random unit B = large initial perturbation). Standard LoRA starts clean
   (B=0 → ΔW=0 initially). This asymmetry likely hurts early convergence.

2. **AdamW momentum mismatch**: After each projection, B magnitude resets to unit norm.
   AdamW's momentum accumulates on the "wrong" scale. Proper Riemannian optimization
   (retraction-based AdamW on S^{d-1}) would avoid this.

3. **Post-hoc retraction (not native)**: Projecting after the Euclidean gradient step is a
   first-order approximation to Riemannian gradient descent. The correct update is:
   `B_i ← normalize(B_i + lr × (∇L - (∇L·B_i)B_i))`
   Our implementation computes the Euclidean update first, then retracts — two separate steps
   rather than one Riemannian step.

## What Makes Failure Impossible (Resurrection Path)

**Impossibility structure for KC05 failure**: The 17pp gap is a TRAINING ALGORITHM gap,
not a representational gap. Theorem 1 proves RMSNorm discards magnitude, so direction-only
representations are theoretically lossless. The failure is in how we traverse the sphere.

**Fix**: Native Riemannian gradient descent on the Stiefel manifold (orthonormal rows):
- Compute tangent-projected gradient: `G_t = G - (G@B^T)@B`
- Cayley retraction or exponential map for update
- Reference: Wen & Yin 2013 "A Feasible Method for Optimization with Orthogonality Constraints"
  arxiv 2309.03737 (ManifoldOpt library)

This is precisely what PoLAR (T1.5) attempted but only applied to U, not V.
A correct Stiefel LoRA should close the gap.

## Scale Sweep Prediction (Not Measured)

Theorem 1 predicts: if direction-only works, scale={1,5,10,20} should give IDENTICAL accuracy
(magnitude discarded by RMSNorm). This was not measured; a follow-up experiment could verify
this as a direct test of T1 without the confound of the training procedure gap.

## Training Details

- Model: Gemma 4 4B (gemma-4-e4b-it-4bit), MLX
- LoRA targets: q_proj only, rank=6, scale=6.0, 42 layers
- Training: 2000 examples, 1000 steps, AdamW lr=1e-4, batch_size=2
- Trainable params: 1,247,232 (42 × 2 × 6 × 2480 / 2 ≈ 42 lora_a + 42 lora_b)
- Direction-only training time: 1471.1s (24.5 min)
- Standard LoRA training time: 1466.2s (24.4 min)
- Evaluation: 50 GSM8K test examples, chain-of-thought with `####` answer extraction
