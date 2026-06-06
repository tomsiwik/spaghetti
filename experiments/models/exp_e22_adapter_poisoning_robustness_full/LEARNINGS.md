# E22-full Learnings: Adapter Poisoning Robustness at Scale

## Core Finding
Grassmannian poisoning protection (F#821: 55pp margin) was a 3-layer artifact. At 35 layers, clean composition alone destroys model coherence (84%→2%), making all differential measurements noise (-3 to +2pp). F#823 falsifies F#821.

## Why
Synthetic adapters (B ∝ W @ A^T) accumulate 175 rank-6 ΔW perturbations at 35 layers. Total perturbation overwhelms base model regardless of A-matrix orthogonality. Input-space feature isolation is real per-layer but irrelevant when the aggregate destroys coherence.

## Implications

1. **Synthetic adapter construction is scale-limited.** Any experiment using B ∝ W@A^T at >10 layers produces floor-level accuracy. Future composition experiments must use trained adapters or restrict layer count.

2. **Grassmannian has no demonstrated benefit at scale.** E14-full killed activation decorrelation (0.0018 benefit = noise). E22-full kills poisoning protection (margin = noise). Both 3-layer smoke results failed to replicate.

3. **3-layer → full-scale is the dominant failure mode.** E14-full and E22-full both showed effects at 3 layers that vanished at 35. This is a systematic bias in the synthetic adapter methodology, not experiment-specific.

4. **Research backlog complete.** 21 experiments (E1-E16, E14-full, E19-E22, E22-full) all KILLED. Key surviving insights: (a) NRE ∝ N^1.3 sub-quadratic scaling, (b) composition residual is nonlinear (cross-layer GELU/softmax), (c) B-matrix coupling is structural via shared W, (d) proxy-behavioral divergence confirmed repeatedly (F#666).
