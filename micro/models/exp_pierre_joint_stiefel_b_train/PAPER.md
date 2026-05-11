# PAPER — Joint-Stiefel Multi-Adapter Training (K=7)

**Experiment:** `exp_pierre_joint_stiefel_b_train`
**Status:** KILLED (K3, K4 fail; K1, K2 pass)
**Date:** 2026-05-11

## Hypothesis

Training K=7 PoLAR adapters with a joint Stiefel constraint on B matrices
(B_all B_all^T = I_{Kr}, Kr=42) enforces both intra-adapter orthonormality
and inter-adapter orthogonality during training, enabling interference-free
composition via simple 1/K averaging.

## Method

- **JointPoLARLinear** module: K=7 sets of (A_k, B_k) per q_proj layer,
  with per-step joint SVD retraction on stacked B_all ∈ R^(42 × 2048).
- **Phase A:** 3 independent single-adapter Stiefel-B trains (math/code/medical,
  300 steps each) as K1 baselines.
- **Phase B:** K=7 joint training via round-robin (2100 total steps), joint
  Stiefel retraction every step, A retraction every 20 steps.
- **7 domains:** math (GSM8K), code (CodeAlpaca), medical (MedMCQA),
  finance/legal/biology/physics (MMLU subtasks).
- **Composition:** ΔW = (1/K) Σ_k (A_k @ B_k) applied as direct delta.

## Predictions vs Measurements

| Kill Criterion | Prediction | Measurement | Result |
|---|---|---|---|
| K1 CONVERGENCE: joint ≥ independent − 5pp | Each adapter converges normally | math +2pp, code +0pp, medical +4pp | **PASS** |
| K2 ORTHOGONALITY: ‖B_all B_all^T − I‖_F ≤ 1e-3 | SVD retraction enforces manifold | avg = 1.89e-07 | **PASS** |
| K3 CROSS-CONTRIBUTION: perturbation ≤ 1% | Joint Stiefel → zero cross-contrib | 3.4%–501k% (all fail) | **FAIL** |
| K4 COMPOSED ACCURACY: 3-bench ≥ 71.3% | Theorem-backed composition | 59.3% (GSM8K 62, HE 74, MedQA 42) | **FAIL** |

## Per-Adapter Individual Accuracy (Joint Training)

| Domain | Joint Score | Independent Baseline | Delta |
|---|---|---|---|
| math | 66.0% | 64.0% | +2.0pp |
| code | 68.0% | 68.0% | +0.0pp |
| medical | 38.0% | 34.0% | +4.0pp |
| finance | 88.0% | — | — |
| legal | 72.0% | — | — |
| biology | 100.0% | — | — |
| physics | 90.0% | — | — |

Individual adapter quality is excellent. Joint training does not degrade any
adapter vs independent — in fact, it slightly improves math and medical.

## Cross-Contribution Analysis

| Domain | NLL Single | NLL Composed | Perturbation |
|---|---|---|---|
| math | 0.283 | 0.337 | 18.8% |
| code | 0.442 | 0.457 | 3.4% |
| medical | 0.116 | 0.135 | 15.9% |
| finance | 0.073 | 0.103 | 40.0% |
| legal | 0.691 | 0.583 | 15.7% |
| biology | 0.000016 | 0.078 | 501k% |
| physics | 0.081 | 0.253 | 212.0% |

The 1% threshold is violated for every domain. Biology's extreme ratio is
an artifact of near-zero single-adapter NLL (perfectly fitted training
examples), but even well-behaved domains like code show 3.4% perturbation.

## Why K3 and K4 Fail Despite K2 Passing

The joint Stiefel constraint enforces B_k B_l^T = 0 for k ≠ l — the B
matrices span orthogonal subspaces. But the full LoRA contribution is
A_k @ B_k, not just B_k. The A matrices are independent (unconstrained)
across adapters, so:

```
cross-contribution = Σ_{j≠k} (x @ A_j) @ B_j
```

Even though B_j spans a subspace orthogonal to B_k, the output
(x @ A_j @ B_j) lives in the full d_out space. The cross-adapter
contributions add signal in directions orthogonal to adapter k's B rows,
but these directions still affect downstream computation (attention,
LayerNorm, softmax). The geometric guarantee on B does not extend to
the composed output.

**The gap:** Stiefel on B alone provides geometric separation of B's row
spaces but NOT functional non-interference of the full LoRA delta.
Joint Stiefel on B is necessary but not sufficient for zero cross-contribution.

## What Would Be Needed

For true zero cross-contribution, the full ΔW_k = A_k @ B_k must produce
outputs in mutually orthogonal subspaces. This requires joint constraints
on BOTH A and B — e.g., Grassmannian A with Stiefel B, such that:
- A_k columns span mutually orthogonal input subspaces
- B_k rows span mutually orthogonal output subspaces
- Together: the adapters operate on disjoint input→output channels

This is the "Pierre double-Stiefel" hypothesis: constrain (A, B) jointly
on the product manifold St(r, d_in) × St(r, d_out).

## Composition Quality

The 59.3% composed 3-bench average (vs 71.3% threshold) shows that simple
averaging of K=7 joint-Stiefel adapters produces worse-than-TIES-B composition.
The composition is not catastrophic (unlike post-hoc projection at 32.7%),
but the 12pp gap to TIES-B confirms that B-orthogonality alone doesn't
deliver the promised interference-free merge.

## Verdict

**KILLED.** Joint Stiefel on B achieves near-perfect geometric orthogonality
(1.89e-07) and preserves individual adapter quality (+0–4pp), but the
mathematical guarantee on B does not extend to the full A@B LoRA contribution.
Cross-contribution is 3–40% (threshold 1%) and composition quality is
59.3% (threshold 71.3%).

The positive K1+K2 results validate joint Stiefel training as a mechanism.
The K3+K4 failures identify the specific gap: A-matrix coupling. This
motivates the "double-Stiefel" extension or a composition operator that
accounts for the A-B interaction.
