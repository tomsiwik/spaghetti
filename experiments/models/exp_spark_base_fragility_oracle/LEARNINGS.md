# LEARNINGS — exp_spark_base_fragility_oracle

**Core finding.** An isotropic random-noise probe of the frozen base alone (KL divergence under
fixed-norm random ε, K=8 draws, zero adapter information) does not predict per-token composition
damage: Spearman rho = 0.0085 (floor 0.30) and top-decile AUC = 0.574 (floor 0.62) — both
kill-2305 clauses fired, verdict KILLED.

**Why.** Rho is a genuine null (both arrays have nonzero variance; the ~0 correlation is real
independence, not an artifact). Composition damage is driven by the structural alignment between
the adapter delta direction and the base geometry — an anisotropic, directional property —
not by the base's direction-averaged isotropic steepness. A zero-adapter-information predictor
cannot recover this signal because it knows nothing about the structured Σ(BᵢAᵢ)h displacement
direction.

**Implication for the next experiment.** This closes the complementary escape from the
F#864/867/868 dead class: neither adapter-delta magnitude alone (prior kills) nor base isotropic
curvature alone (this kill) predicts interference. The open frontier is the directional alignment
angle between the adapter delta and the base geometry — cos θ between Σ(BᵢAᵢ)hₜ and the
dominant eigenvectors of the base's local Hessian (or Jacobian) at hₜ. That anisotropic,
adapter-aware signal has not been tested.
