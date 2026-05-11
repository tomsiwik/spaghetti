# MATH.md — Joint-Stiefel multi-adapter training (K=7)

## The structurally-honest version

Single-adapter Stiefel (`exp_pierre_stiefel_b_train_single`) gives
*per-adapter* row-orthonormality (B_k B_k^T = I_r). That doesn't guarantee
cross-adapter orthogonality — two different adapters' rows can still
align.

The mathematical version that delivers Pierre's promise: **joint Stiefel
across K adapters** per layer.

Stack:
```
B_all = [B_1; B_2; ...; B_K] ∈ ℝ^(K·r × d_out)
```
Constrain:
```
B_all B_all^T = I_{K·r}     (all K·r rows mutually orthonormal)
```

This forces both intra-adapter orthonormality (B_k B_k^T = I_r) AND
inter-adapter orthogonality (⟨row_i(B_k), row_j(B_l)⟩ = 0 for k ≠ l).

The composition consequence:
```
B_merged = (1/K) Σ_k B_k
‖B_merged row i‖² = ‖(1/K) Σ_k row_i(B_k)‖²
                  = (1/K²) Σ_k Σ_l ⟨row_i(B_k), row_i(B_l)⟩
                  = (1/K²) Σ_k ‖row_i(B_k)‖²       (cross terms = 0)
                  = (1/K²) · K · 1 = 1/K
```
So merged rows have *exactly* known magnitude 1/K — predictable, not
needing Fisher-Rao's empirical correction. And the merge is interference-
free by Pythagoras.

## Training procedure

Multi-task data loader (math + code + medical, balanced batches).
Each batch contains examples from all K adapters' native domains.
Loss accumulates per-adapter contributions; gradient is computed on
joint B_all but each B_k gets gradient from its own task examples only.

Per training step:
1. Sample batch from each adapter's data.
2. Forward through each adapter independently, compute per-task loss.
3. Backward → per-adapter B gradients.
4. Stack gradients: `g_all = [g_1; ...; g_K]`.
5. Stack B's: `B_all = [B_1; ...; B_K]`.
6. Riemannian tangent project on joint Stiefel:
       `T = g_all - B_all sym(B_all^T g_all)`
7. Update + retract: `B_all_new = QR(B_all - lr * T)`.
8. Slice back: `B_k = B_all_new[k·r:(k+1)·r, :]`.

This is **per-step joint optimization**. Per-adapter losses still drive
per-adapter gradients, but the manifold constraint mixes them at
retraction time.

## Pre-registered Kill Criteria

- **K1 (CONVERGENCE)** Each adapter k reaches its native benchmark
  within 5pp of independent single-task training.
  PASS → joint constraint doesn't kill any adapter's learning.

- **K2 (JOINT ORTHOGONALITY)** `||B_all B_all^T - I_{Kr}||_F` after
  training ≤ 1e-3 averaged over layers.

- **K3 (CROSS-CONTRIBUTION ZERO)** Adding K-1 other adapters' contributions
  to adapter k's forward perturbs k's output by ≤ 1% on its native
  task. This is the mathematical promise being measured.

- **K4 (COMPOSED ACCURACY)** K=7 composition via simple 1/K mean reaches
  TIES-B baseline (71.3%) on the 3-bench average.
  PASS → mathematical guarantee delivers TIES-B-quality composition via
  trivial averaging. Composition arc closed *with theorem*.

## Verdict matrix

| K1 | K2 | K3 | K4 | Outcome |
|----|----|----|----|---------|
| ✓ | ✓ | ✓ | ✓ | **SUPPORTED** — joint Stiefel works. Mathematical composition guarantee delivered. Migrate Pierre. |
| ✓ | ✓ | ✓ | ✗ | **PARTIAL** — orthogonality achieved but composition operator wrong. Try weighted variants (Karcher mean on Stiefel). |
| ✓ | ✓ | ✗ | * | **CONTRADICTION** — joint orthogonality but high cross-contribution. Implementation bug. |
| ✓ | ✗ | * | * | **RETRACTION BUG** — orthogonality lost during training. QR sign issues or retraction frequency. |
| ✗ | * | * | * | **KILLED** — joint constraint too restrictive at K=7, r=6, d_out=2048. Training can't converge under the manifold. Either reduce K or relax to per-adapter Stiefel. |

## Implementation status

**SPEC + ALGORITHM — implementation is the heaviest of the arc.**

Required:
1. Multi-task data pipeline: round-robin or balanced sampling across
   math/code/medical training sets (~50 LoC, datasets already exist).
2. Joint forward+backward: per-adapter forward, accumulate per-task loss
   (~100 LoC, modifies polar_train.py training loop).
3. Joint Stiefel retraction: stack-QR-slice (~50 LoC).
4. Validation: every N steps, check `||B_all B_all^T - I||_F` per layer.
5. ~3.5h training for K=7 adapters at convergence.

Estimated total: **6-8h MLX impl + 3.5h training**.

## Why this is the strongest version of the thesis

Every other composition method in our queue is a *correction* applied
either at merge time (Fisher-Rao norm-rescale, TIES trim+elect, ACE
covariance-weight, Pico calibration) or post-hoc projection of trained
weights. Joint-Stiefel training is the only path where:

- Cross-adapter orthogonality is enforced *during* training
- Composition becomes simple weighted addition without correction
- The mathematical guarantee (zero cross-contribution) holds by theorem
- The merge is reversible: given the composed B and one adapter's B,
  you can recover the contribution of any other adapter exactly

This is Pierre's design goal — composition without interference because
the math forbids it.

## References

- Sibling: `exp_pierre_stiefel_b_postproj` (P1) — must complete first
- Sibling: `exp_pierre_stiefel_b_train_single` (P2) — single-adapter
  feasibility
- Sibling: `exp_pierre_stiefel_b_composition` (P2) — composition eval
  using these trained weights
- Pierre PoLAR theorem: this is the analogous constraint on B that A has
  via Grassmannian.
- Riemannian Optimization for LoRA on Stiefel (arxiv 2508.17901)
- StelLA (arxiv 2510.01938)
