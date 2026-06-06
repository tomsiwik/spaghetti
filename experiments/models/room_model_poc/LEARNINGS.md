# LEARNINGS: exp_room_model_poc (KILLED)

## Core Finding

**Per-module adapter pre-summing is algebraically exact (MSE 5.6e-7), but full-model
pre-summing is structurally impossible in any transformer with nonlinear layers.
LayerNorm, SiLU, and attention softmax create multiplicative cross-terms between
adapter deltas that compound through 30 layers (logit MSE: 53.9). A-subspace
projection routing is dead — Grassmannian A-matrices are geometrically orthogonal
but semantically arbitrary (14% accuracy vs 20% random baseline).**

## Why This Happened

### 1. Nonlinear Compounding Kills Full-Model Equivalence

The distributive property x @ (A+B) = x@A + x@B is an axiom for single linear modules.
But transformers interleave linear modules with LayerNorm, SiLU, and softmax. Each
nonlinear layer creates cross-terms: LayerNorm(h + delta) != LayerNorm(h) + LayerNorm(delta).
These cross-terms compound multiplicatively through L=30 layers.

This is a known theoretical result. **Zhong et al. (2504.10957, ICLR 2025 oral)**
provide the first formal generalization analysis of task vectors on nonlinear
transformers, proving that task arithmetic (weight-space addition of fine-tuned deltas)
is effective ONLY when tasks are "irrelevant or aligned" — i.e., when the delta
subspaces do not interact through the nonlinear layers. Our 5 domain adapters
(medical/code/math/legal/finance) are definitionally NOT irrelevant — they modify
overlapping hidden state dimensions — so pre-summing falls outside the provably
effective regime.

The original Task Arithmetic paper (Ilharco et al., 2212.04089) demonstrated that
weight-space addition works empirically for CLIP models, but noted performance
degradation at higher scaling coefficients — precisely the nonlinear interaction
effect we measured. Our experiment with N=5 simultaneous deltas at full alpha
amplifies this effect 5x.

### 2. Orthogonality Serves Composition, Not Routing

This is the third confirmation of a recurring project finding:
- **Finding #115:** Content-aware routing from hidden states: 26.5% accuracy (killed)
- **Finding #192:** Softmax multi-class routing at N=24: 39.4% (killed)
- **Finding #246:** Weight-space orthogonality (99.6%) produces only 0.3-5.7% PPL
  differentiation — decorrelation ≠ specialization
- **This experiment:** A-subspace projection: 14% accuracy (worse than random)

Grassmannian A-matrices are optimized for geometric packing (minimizing mutual
coherence), not for semantic alignment with domain-specific hidden states. The
A-matrices span random r-dimensional subspaces of R^d that are maximally spread
on the Grassmannian manifold — this is a packing property, not a semantic property.

**OSRM (Yadav et al., 2505.22934)** confirms this from the merging side: orthogonal
subspaces help prevent interference during merging, but the orthogonality must be
CONSTRAINED DURING FINE-TUNING to ensure task-relevance, not just imposed
geometrically. Our Grassmannian initialization achieves orthogonality without
task-awareness, which is why it serves composition (non-interference) but not routing.

### 3. Bandwidth Confirms Prior Findings

Room model at 39.2 tok/s confirms the bandwidth model from Findings #300/#301:
4.17 GB W_combined at 273 GB/s = 15.3 ms minimum. Even if equivalence were solved,
the bandwidth cost makes full pre-summing unviable on M5 Pro.

## Confirming Evidence

- **2504.10957** (Zhong et al., ICLR 2025): Task vector arithmetic provably effective
  only for "irrelevant or aligned" tasks on nonlinear transformers. First theoretical
  characterization of when weight-space addition fails.
- **2212.04089** (Ilharco et al., 2022): Original task arithmetic paper. Empirically
  shows degradation at high scaling coefficients — the nonlinear interaction effect.
- **2505.22934** (OSRM): Orthogonal subspaces for robust merging. Confirms orthogonality
  must be task-aware during training, not just geometric initialization.
- **2510.03262** (Rethinking Inter-LoRA Orthogonality): Shows inter-LoRA orthogonality
  does NOT guarantee semantic disentanglement — orthogonal ≠ specialized.
- **2505.15875** (Decouple and Orthogonalize): Data-free LoRA merging that decouples
  before merging — acknowledges that naive summation creates interference.
- **Finding #300, #301:** Bandwidth model confirmed across 4 configurations.

## Contradicting Evidence

- **LoRA Soups (2410.13025):** Shows that simple averaging of LoRA adapters CAN work
  for skill composition. Key difference: they average at 1/N scale (norm-controlled),
  not full-alpha sum. Our v3 NRE composition is actually closer to this approach.
- **2601.18350** (Adapter Merging Reactivates Reasoning): Shows that merging can
  actually activate latent traces. But this is for single-merged models at inference,
  not N simultaneous deltas at full strength.

The contradiction resolves on scale: merging with coefficient 1/N (averaging) is
within the "small perturbation" regime where nonlinear effects are manageable.
Pre-summing at full alpha (N deltas, each at alpha=1) is 5x outside that regime.

## Alternative Approaches

1. **Per-token adapter routing — MoLoRA (2603.15965):** Routes individual tokens to
   specific adapters, avoiding simultaneous multi-adapter activation entirely. Achieves
   composable specialization: Qwen3-1.7B exceeds Qwen3-8B on 4 reasoning benchmarks.
   Uses learned gating, not geometric projection. Directly addresses our failure mode:
   instead of applying all deltas simultaneously, apply ONE per token.

2. **Fused factored serving — Punica BGMV (2310.18547):** Eliminates per-adapter dispatch
   overhead via fused kernels while keeping adapters in factored form. S-LoRA (2311.03285)
   extends this to thousands of adapters. Addresses the speed limitation without
   requiring pre-summing.

3. **Task-level routing — LoRAuter (2601.21795):** Routes at the request level using
   validation-set similarity, bypassing token-level routing entirely. Achieves 101.2%
   of oracle performance at 1500+ adapters. Already identified as follow-up from
   Finding #298's LEARNINGS.

4. **Constrained orthogonal fine-tuning — OSRM (2505.22934):** If multi-adapter
   composition is needed, constrain A-subspace during training to align with
   task-relevant directions. This would make orthogonality semantic, not just geometric.

## Implications for Next Experiments

1. **The "all adapters active" paradigm is dead.** Any approach that applies N adapter
   deltas simultaneously at full strength will hit the nonlinear compounding wall.
   The provably effective regime (2504.10957) requires task irrelevance or alignment —
   which means at most 1-2 related adapters, not 5+ diverse domains.

2. **v3 factored RuntimeLoRA (73 tok/s) is the correct architecture.** It applies
   ONE adapter per forward pass (via NRE composition or single selection), staying
   within the single-perturbation regime where nonlinear effects are small.

3. **Speed gains must come from the serving layer, not the composition layer.**
   Punica-style BGMV fusion (eliminate dispatch overhead) or MoLoRA-style per-token
   routing (eliminate simultaneous activation) are the proven paths.

4. **Orthogonality is a composition guarantee, not a routing signal.** This is now
   confirmed across 4 experiments (#115, #192, #246, #303). Future work should treat
   Grassmannian orthogonality as interference prevention only and use separate
   mechanisms for routing (task-level similarity, learned gating, or ridge regression).

## Recommended Follow-Up

**Per-token adapter routing (MoLoRA-style)** — Motivation: MoLoRA (2603.15965) proves
per-token routing is both optimal (work N vs K*N) and effective (1.7B matches 8B).
Our existing v3 RuntimeLoRA already selects one adapter; extending to per-token
selection within a sequence would handle mixed-domain inputs (e.g., "explain this
medical code") without simultaneous multi-adapter activation. This directly addresses
the structural impossibility found here: instead of summing N deltas, route each
token to its best single adapter.
