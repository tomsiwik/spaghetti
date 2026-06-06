# LEARNINGS: exp_topology_stress_n_sweep

## Core Finding

Weight-space topology is radically robust to adapter composition at any practical N (5-50), under both averaging and additive schemes. Zero high-persistence features are lost across 60 persistent homology computations. Under 1/N averaging, topology actually IMPROVES with N (Spearman rho=-1.0), making the composition scheme self-stabilizing in weight space. This definitively closes the 4-experiment pathway-preservation research track: topological feature loss is a non-problem.

## Why This Happened

The Algebraic Stability Theorem (Cohen-Steiner et al. 2007, Theorem 5.2) gives d_B <= max||delta_i||, but this bound is derived from worst-case pointwise rearrangements. Low-rank adapter perturbations (Delta = A @ B) confine all row movements to a rank-r subspace, creating highly structured perturbations that use only a fraction of the bound's worst-case budget. The measured ratio d_B/max||delta|| ranges from 0.06-0.23, confirming the bound is 4-17x loose.

Under 1/N averaging, the incoherent adapter case gives ||Delta_row(N)|| ~ scale/sqrt(N), which DECREASES with N. The 1/N factor dominates the sqrt(N) growth from vector addition, creating a self-stabilizing dynamic in weight space. Empirically confirmed: perturbation norms decay from 0.88 (N=5) to 0.42 (N=50).

Under additive composition (stress test), norms grow superlinearly (4.4 to 21.1 for N=5 to 50), indicating partial adapter coherence. Even at N=50 additive where the vulnerability window (103x) exceeds median persistence (38.7) by 2.7x, zero features are destroyed -- the bound remains vacuously loose.

## Confirming Evidence

- **Cohen-Steiner, Edelsbrunner, Harer (2007):** The stability theorem itself is well-established. Our application to row-wise point clouds is a direct and valid use.
- **arXiv:2410.11042 (Persistent Topological Features in LLMs):** Demonstrates that persistent topological features exist in LLM weight spaces and are stable under normal perturbations. Our finding extends this: they are stable even under 50-adapter additive composition.
- **Finding #225:** 0/17,223 H0 features lost at N=5 (parent experiment).
- **Finding #230:** 0/11,962 H1 features lost at N=5 with 19/35 modules in vulnerability window.
- **Finding #228:** Bridge correction counterproductive -- SVD components carry useful learning signal (Task Arithmetic, Delta-CoMe, PiSSA arXiv:2404.02948 all confirm adapter perturbations are task-specific vectors, not damage).

## Contradicting Evidence

- **arXiv:2510.03262 (Rethinking Inter-LoRA Orthogonality):** Weight-space geometric orthogonality does NOT predict semantic disentanglement. Adapters can be topologically isolated yet still interfere through activations, residual streams, and LayerNorm. This is the fundamental limitation of our entire topology track: we measured the wrong thing. Topological preservation in weight space says nothing about functional interference.
- **Our own composition experiments (SOLE adversarial review):** Equal-weight 1/N averaging caused PPL explosion to trillions due to logit-scale mismatch from a single dominant adapter. Topology was PRESERVED while behavior was CATASTROPHIC. This is the strongest evidence that weight-space topology is disconnected from composition quality.
- **LoRA-LEGO (Rank-Wise Clustering):** Standard merging destroys modular adapter nature despite weight-space metrics looking acceptable. Preserving "Minimal Semantic Units" requires adapter-level structural awareness, not weight-space topology.
- **MoTE (Mixture of Ternary Experts):** At larger structural scales (replacing entire FFN layers), composition causes significant degradation. Our rank-16 perturbations are too small to trigger this regime, but it exists.

## Alternative Approaches

For predicting safe scaling limits of adapter composition (the practical question behind the topology track):

1. **Sub-additive safety bounds** -- Transformer composition errors are sub-additive across layers with ~75% dampening. Residual streams + LayerNorm make composition 11.5x safer than raw FFN. Our own experiments predicted 0.106% degradation at N=50, measured 0.098%. No topology needed.
2. **Domain similarity collision modeling** -- Within-cluster cosine similarity is 7.84x higher than cross-cluster. Collision rate follows power-law decay (beta=-0.575), reaching 1.23% at N=20. Predicts practical limits from adapter statistics, not weight-space topology.
3. **Canary queries** -- Behavioral probes with 2.0% FNR vs 33.8% FNR for geometric metrics (validated in our architecture). The only reliable runtime monitor for composition quality.
4. **PPL-probe weighting** -- K+1 forward passes for runtime behavioral probing, r=0.990 correlation with oracle. Catches the exact failure mode (logit-scale mismatch) that topology misses.
5. **W2T: LoRA Weights Already Know What They Can Do (arXiv:2603.15990)** -- Infers adapter capabilities from weight statistics. More practical than topology for adapter selection/routing.

## Implications for Next Experiments

1. **Pathway-preservation track is definitively closed.** Four experiments form a complete arc:
   - exp_pathway_graph_bitnet2b: KILLED (sparsification artifact, not real topology)
   - exp_persistence_diagram_diff: SUPPORTED but vacuous (stability bound 10-100x loose)
   - exp_persistence_bridge_extraction: KILLED (bridge counterproductive, SVD is the learning signal)
   - exp_topology_stress_n_sweep: SUPPORTED (0 features lost even at N=50 additive, averaging self-stabilizes)

   Weight-space persistent homology provides no actionable signal for adapter composition at any practical scale.

2. **The critical insight is the DISCONNECT between topology and behavior.** Our own experiments proved this most powerfully: topology was preserved while PPL exploded to trillions (the logit-scale mismatch). arXiv:2510.03262 provides the theoretical backing: weight-space geometry does not predict functional-space behavior because activation-level interactions (residual streams, LayerNorm) create non-linear interference invisible to weight-space analysis.

3. **Behavioral monitoring is the only reliable approach.** Canary queries (2.0% FNR), PPL-probe weighting (r=0.990), and leave-one-out screening catch real composition failures. Geometric/topological metrics catch none of them (33.8% FNR).

4. **For scaling predictions, use algebraic bounds not topology.** Sub-additive error bounds (0.106% predicted vs 0.098% measured at N=50) and domain similarity collision modeling (power-law decay) give accurate, actionable scaling limits without persistent homology computation.

## Recommended Follow-Up

No follow-up from the topology track. The research line is closed with high confidence across 4 experiments.

Priority should shift entirely to the P0 deployment track:
- **exp_generation_quality_test** -- Does routed composition produce better text? THE existential test. Motivated by: all topology/metric work is meaningless without behavioral validation (the core lesson of this entire track).
- **exp_task_accuracy_real_benchmarks** -- MMLU/GSM8K/HumanEval with composition. Motivated by: behavioral outcomes over metrics (Finding #228, arXiv:2510.03262).
