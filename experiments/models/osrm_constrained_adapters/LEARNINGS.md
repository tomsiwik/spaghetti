# LEARNINGS: OSRM-Constrained Adapter Init

## Core Finding

**The "orthogonality hypothesis" for adapter composition is now CLOSED.**

Three independent experiments have tested whether orthogonal separation between
adapters is the mechanism behind successful multi-adapter composition:

1. **Finding #68** (exp_bitnet_semantic_compositionality): Weight-space orthogonality
   does not imply data-space orthogonality. 100% of adapter pairs fail OSRM
   data-orthogonality. Yet composition works (4/5 pairs, PPL 8.35 vs 8.58).

2. **This experiment** (exp_osrm_constrained_adapters): Data-aware A-matrix init
   (OSRM covariance-constrained) reduces cross-domain activation by 15% but does
   not improve composition quality. Random, Grassmannian, and OSRM produce
   identical results within 1%.

3. **Finding #164** (exp_lora_soups_cat): Simple scaling (Task Arithmetic lambda=0.5)
   beats all learned composition methods. The composition problem is about **scaling**
   (how much of each adapter to apply), not **orthogonality** (how separated adapters are).

**Verdict: Orthogonality — whether in weight space, data space, or achieved via
constrained initialization — is not the mechanism. Stop investing in orthogonality.**

## Why Orthogonality Doesn't Matter at d=2560

1. **Dimensional concentration provides it for free.** At d=2560 with rank-16
   subspaces, random A matrices have pairwise |cos| ~ 1/sqrt(d) ≈ 0.02. This
   is already near-zero. OSRM pushes it to ~0.015 — a 15% improvement on a
   negligible baseline. There's no meaningful interference to prevent.

2. **B compensates.** Even with frozen A, the B matrix is unconstrained and
   learns to map rank-16 projections into whatever output subspace minimizes
   loss. 200 gradient steps erase any starting advantage of A's orientation.
   The optimization landscape has more than enough capacity.

3. **1/N scaling is the dominant regularizer.** With N=5 adapters and 1/5
   scaling, each adapter contributes 20% of its perturbation. This aggressive
   dilution prevents any single adapter from dominating, regardless of
   subspace overlap.

## What IS the Mechanism?

Based on findings #68, #164, #169:

- **Constructive cross-domain transfer:** Adapters learn perturbations that
  improve the base model for their domain. When composed, these perturbations
  add constructively because the base model provides the shared structure.
  Cross-domain activation is not noise — it's a feature.

- **Scaling strategy:** The composition quality is determined by how much of
  each adapter signal to include. Lambda=0.5 (Task Arithmetic) > lambda=0.2
  (uniform). Higher lambda = more adapter signal = better, up to some
  interference ceiling we haven't found yet.

- **Base model quality:** The base model provides the function; adapters only
  encode direction of change. The binding constraint is base model quality
  and adapter training quality, not subspace geometry.

## Confirming Evidence

- Finding #68: weight orth != data orth, composition works regardless
- Finding #164: scaling > weighting for orthogonal adapters
- OSRM paper (arXiv:2505.22934): reported +12.78% but on highly disjoint tasks
  where cross-interference IS the bottleneck (different from our domain setup)
- FlyLoRA (arXiv:2510.08396): frozen random A works fine, JL-lemma orthogonality

## Contradicting Evidence

- OSRM paper: +6-9pp improvement at their scale/task setup. May work when
  domains are truly disjoint AND d is smaller AND cross-interference is
  the actual bottleneck.
- LoRI (arXiv:2504.07448): frozen A + sparse B, 17.3% better merge. But LoRI's
  improvement may come from B sparsity (regularization), not A orthogonality.

## Practical Recommendations

1. **Use random QR A init.** No benefit from Grassmannian or OSRM at d=2560.
2. **Focus on routing and scaling** — top-k selection, lambda tuning, entropy gating.
3. **Focus on adapter training quality** — better data, longer training, STE ternary.
4. **OSRM might matter at d < 256** where random subspace overlap is significant.
   Don't test this — it's not our production scale.

## Follow-Up

No dedicated follow-up recommended. The orthogonality research direction is
settled. Priority should shift to deployment track: generation quality,
task accuracy, and end-to-end demo.
