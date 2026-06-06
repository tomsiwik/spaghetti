# LEARNINGS — exp_pierre_ace_merging_b_only

**Status:** KILLED | **Finding:** Shared-A breaks covariance-based merging

## Core Finding

ACE-Merging (arxiv 2603.02945) collapses to 26.7% avg vs 64.7% Fisher-Rao baseline (-38pp).
The failure is structural, not tunable.

## Why

ACE infers per-task importance from ΔW_t via covariance Σ̂_t = ΔW_t^T ΔW_t.
Under shared-A, every delta factors as s_t · A · B_t, so all covariances are dominated
by A^T·A. The regularized inverse D^{-1} amplifies this near-rank-deficiency,
producing degenerate merge weights that destroy task signal.

## Generalization (Finding #ACE-1)

**Any merge method that infers per-task structure from ΔW is incompatible with shared-A.**
The rank-1 factorization means ΔW carries shared structure (A), not task identity.
Suspect methods: RegMean, delta-derived Fisher, any covariance/gradient weighting on full deltas.

## Implication

For Pierre's shared-A architecture, only methods operating directly on B_t
(the task-specific component) or using external signals (held-out loss, Fisher on activations)
can produce meaningful per-task weighting. Simple methods (Fisher-Rao, DARE) that
don't attempt per-task importance inference remain the correct baselines.
