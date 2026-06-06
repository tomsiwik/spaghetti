# LEARNINGS — exp_prod_red_team_adapter_injection

## Core Finding

N=2 additive null-space LoRA composition **leaks** membership-inference (+15pp
probe advantage, K1667) and subspace structure (~58% mean overlap of composed
ΔW's rank-6 SVD with each user's B basis, K1668), while **blocking** verbatim
canary extraction via greedy decode on 15-token prefix (0/20, K1669). Verdict:
KILLED — two of three attacks succeed; null-space isolation alone is not
operational privacy.

## Why

The leak is mechanistic, not statistical. Dependency's K1644 measured
cross-user B cosine = 0.39 (> 0.30 threshold). Under additive composition
`ΔW = s·Q·(B_A A_A + B_B A_B)`, the rank-2r output-side span is
span(B_A) ∪ span(B_B); because the two bases share ~39% cosine, a rank-r SVD
of the observable composed delta projects onto each individual user's B at
~50–60%. No training-time orthogonality constraint ⇒ no composition-time
separability.

K1669 PASS is the surprising behavioral decouple: at training loss ~0.10 each
adapter could memorize, but loading both adapters simultaneously perturbs the
shared null-space input channel (effective scale 2s through overlapping
direction), shifting the greedy-decode trajectory enough to destroy verbatim
recall. Membership-inference ≠ verbatim recall under additive composition.

## Implications for Next Experiment

1. **Critical path**: retrain with Gram-Schmidt on B_B against B_A during
   each step (project out B_A's column span from B_B's gradient update).
   Target: cross-user B cosine < 0.10 ⇒ predict K1667 probe advantage → ≤ 2pp
   and K1668 subspace overlap → < 5%. Tools already proven at N=1
   (`exp_prod_privacy_redesign` K1643 PASS).
2. **K1669 is a weak test** — keep as monitoring KC but do not rely on it
   for a SUPPORTED verdict in v2. Use stronger threat model: prefix_len=30,
   n_gen=50, temperature-0.7 multi-seed, or random-token canaries during
   training (per MATH.md §K1669 original spec, weakened in run_experiment.py).
3. **Composition-as-defense is real but secondary**. Worth a side experiment
   only after Gram-Schmidt path closes K1667/K1668 — test whether the
   verbatim-blocking persists post-orthogonalization or was collateral from
   B-overlap-induced perturbation.
4. **Success criterion for next iteration**: privacy experiment SUPPORTED
   only when K1667 probe < 1pp AND K1668 overlap < 1% AND K1669 canary ≤ 1/20
   — all three must hold simultaneously, no decouple gaming.
