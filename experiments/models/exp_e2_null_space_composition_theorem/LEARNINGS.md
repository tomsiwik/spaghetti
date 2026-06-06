# E2: Null-Space Composition Theorem — Learnings

## Core Finding
Grassmannian adapters do NOT occupy the base model null space. The measured null-space fraction matches (d − r_eff)/d exactly — the random-chance rate from rank deficiency. Inter-adapter orthogonality (A_i ⊥ A_j) is independent of adapter-base orthogonality (A_i ⊥ W).

## Why
The Grassmannian partition-QR construction operates on a random matrix independent of W_base. By construction it guarantees mutual orthogonality between adapters, but makes no claim about alignment with W's row/null space. The exact match to the rank-deficiency formula (3 decimal places, all 42 layers) proves this is geometry, not a statistical fluke.

## Implications for Next Experiments

1. **E14 (Grassmannian ⟹ Activation Orthogonality):** Must focus on output-space orthogonality, not input-space null-space claims. The Grassmannian guarantee is A_i ⊥ A_j, period.

2. **E15 (Composition Residual Decomposition):** F#752's tau ≈ 0.48 is confirmed as genuine nonlinear coupling (LayerNorm, softmax, SiLU across layers), not null-space leakage. Any composition fix must address cross-layer nonlinearity, not per-layer linear algebra.

3. **E19 (Privacy via null-space):** Requires explicit null-space reparameterization (F#494). Grassmannian alone does not provide it.

4. **General principle:** Per-layer linear composition is exact (tau ≈ 0.067 is numerical noise). The composition problem lives in the nonlinear inter-layer coupling. Solutions must operate at the network level, not the layer level.

## Architecture Discovery (reusable)
Gemma 4 E4B alternates two attention configs every 6 layers: standard (v_proj 512×2560) and wide (v_proj 1024×2560). Adapter strategies treating all layers uniformly miss this structure.
