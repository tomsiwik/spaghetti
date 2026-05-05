# Learnings — mHC Normalized Composition

## Key Finding
Sinkhorn-Knopp normalization cannot be naively applied to LoRA deltas via exp/log round-trip. The spectral norm bound applies to the doubly-stochastic matrix M, not to log(M). This destroyed the model (0% accuracy, spectral norm 26,934×).

## Reusable Insights

1. **DARE's spectral norm is already small**: Before SK, max spectral norm was only 2.98 across 42 layers. This means DARE doesn't actually have a "spectral norm explosion" problem — the original hypothesis was false.

2. **The -6.7pp GSM8K regression doesn't exist**: DARE baseline here got 63.3% GSM8K = same as single adapter. Confirms finding from exp_pierre_per_task_routing_math: there's no systematic DARE regression to fix.

3. **Post-hoc normalization of deltas is dangerous**: Unlike training-time constraints (which the optimizer can adapt to), post-hoc projection destroys the learned structure. The delta's meaning comes from its relationship to the base weights — arbitrary norm constraints break this.

4. **DSv4 mHC is training-time, not inference-time**: The technique works because the model learns to operate within the doubly-stochastic constraint. Applying it after training is a different (invalid) operation.

## Implications for Pierre
- DARE composition is already well-behaved (spectral norm ~3, accuracy preserved)
- No further "regularization" of composed deltas is needed
- If future work needs spectral control, use training-time constraints (e.g., spectral normalization during LoRA training), not post-hoc projection
