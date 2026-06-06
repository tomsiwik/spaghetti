# Learnings: exp_bitnet_meta_scaffold

## Core Finding
MAML-style bilevel optimization for scaffold composition is fundamentally flawed at three levels: the FOMAML implementation computed gradients at the wrong point (restored scaffold, not inner-loop endpoint), the composition penalty was gradient-disconnected, and unconstrained outer-loop updates destroyed scaffold quality (12x PPL degradation after ternary quantization). The one positive finding — adapter resilience (3% degradation on 12x worse scaffold) — needs an adapter/base ratio caveat.

## Why This Happened (Literature-Grounded)

### 1. FOMAML's irreducible bias with long inner loops
Nichol et al. (2018, arXiv:1803.02999) proved that FOMAML introduces **irreducible bias** to the true meta-gradient. This bias grows with the number of inner-loop steps K. At K=50, FOMAML drops the entire second-order term d(theta^K)/dW that captures how scaffold changes affect the inner-loop trajectory — the critical signal for scaffold optimization. The experiment's implementation was even worse: gradients were computed at the restored scaffold + mean-adapter, not at the inner-loop endpoint, making it "direct scaffold optimization" rather than FOMAML at all.

### 2. Unconstrained meta-updates destroy pretrained distributions
The outer loop had no scaffold preservation term (KL divergence, PPL constraint, or weight magnitude penalty). Without such constraints, Adam updates push weights into arbitrary regions that catastrophically interact with ternary quantization. This is analogous to the well-known "catastrophic forgetting" in continual learning — unconstrained gradient updates on a new objective destroy previously learned representations.

### 3. Ternary quantization amplifies meta-learning damage
FOMAML's continuous Adam updates produce weight distributions that straddle ternary quantization thresholds. The 12x PPL degradation after quantization (vs pre-quantization 6.8x) confirms that meta-updated weights land in quantization-hostile regions. STE-aware outer loops (as used in QAT) would be necessary to keep weights quantization-friendly.

### 4. Composition penalty provided zero gradient signal
The pairwise adapter cosine was computed as a detached Python float and never entered the differentiable computation graph. Even conceptually, at micro scale (d=256) with random LoRA init, adapter cosines are already ~0.002 — providing negligible gradient signal regardless.

## Confirming Evidence

1. **Nichol & Schulman (2018, arXiv:1803.02999)**: FOMAML has irreducible bias that grows with inner-loop steps. Performance degrades noticeably after K>4 when using cycling minibatches. At K=50, the approximation is severely degraded.

2. **Rajeswaran & Finn et al. (2019, arXiv:1909.04630)**: Implicit MAML (iMAML) was specifically developed because FOMAML and full MAML both fail at long inner loops. iMAML's memory is independent of K, solving the scalability problem via implicit differentiation + conjugate gradient.

3. **Recover-LoRA (arXiv:2510.08600)**: Confirms that model degradation (from quantization, pruning, etc.) can be partially recovered via LoRA adapters — consistent with our adapter resilience finding. However, Recover-LoRA uses synthetic data distillation, not meta-learning.

4. **Our own GaLore scaffold experiment**: GaLore's gradient low-rank projection naturally produces composition-friendly weight distributions (comp ratio 1.045x) without any explicit composition objective. This suggests that the composition problem is better solved by training dynamics than by explicit meta-optimization.

## Contradicting Evidence

1. **Meta-LoRA (arXiv:2510.11598)**: Successfully uses two-stage MAML for LoRA task adaptation. Key difference: Meta-LoRA optimizes the **adapter initialization** (not the scaffold), uses shorter inner loops, and optimizes for single-task performance (not multi-adapter composition). The success of Meta-LoRA does not transfer to scaffold optimization because the bilevel structure is fundamentally different.

2. **"Meta-Learning the Difference" (TACL, doi:10.1162/tacl_a_00517)**: Uses bilevel optimization to update base weights with low-rank task reparameterization. This work succeeds because it (a) uses full second-order gradients, (b) includes regularization on the base weights, and (c) operates in FP16 (no quantization amplification). All three conditions were violated in our experiment.

3. **MSLoRA (ScienceDirect, 2025)**: Meta-learns layer-wise scaling factors for LoRA, not base weights. Succeeds because the optimization target (scalar scaling factors) is low-dimensional and well-conditioned, unlike full scaffold weight optimization.

## Alternative Approaches (What We Could Try Instead)

### 1. Reptile (OpenAI, 2018) — Recommended
Reptile performs SGD on each task without unrolling computation graphs or computing second derivatives. It approximates the meta-gradient via the direction from initial weights to task-adapted weights. **Advantages for scaffold optimization**: no computation graph, memory-efficient, naturally regularizes toward the initial point (implicit scaffold preservation). **Risk**: may be too conservative to produce meaningful scaffold changes.

### 2. Implicit MAML (iMAML, arXiv:1909.04630)
Eliminates backpropagation through the inner loop entirely by using implicit differentiation + Hessian-vector products. Memory is O(1) in inner-loop steps K. **Advantages**: correct meta-gradient regardless of K, principled. **Risk**: Hessian-vector products may be expensive on Apple Silicon, and still needs STE integration for ternary weights.

### 3. Scaffold-free composition (our GaLore direction)
The GaLore scaffold already achieves comp ratio 1.045x without any meta-optimization. Rather than meta-learning a better scaffold, invest in better adapter training (orthogonality constraints, Grassmannian init) on an existing scaffold. This sidesteps the entire bilevel optimization problem.

### 4. Distillation-based scaffold optimization
Instead of bilevel optimization, use knowledge distillation: train a student scaffold to match the output distributions of a teacher (pretrained model) while simultaneously minimizing adapter composition interference. This avoids the meta-learning machinery entirely.

### 5. Evolutionary scaffold search
Replace gradient-based meta-optimization with evolutionary strategies (ES). ES is gradient-free, naturally handles non-differentiable ternary quantization, and can optimize any black-box objective including composition metrics. Downside: sample efficiency.

## Implications for Next Experiments

1. **MAML for scaffold optimization is deprioritized.** The combination of FOMAML's irreducible bias, ternary quantization amplification, and the GaLore baseline being hard to beat makes this direction unpromising without iMAML + STE + scaffold preservation — which is complex and speculative.

2. **Adapter resilience finding needs validation at realistic ratios.** The 3% degradation finding used adapter/base ratio ~5.4% (r=16, d=256). At production scale (r=16, d=2560), the ratio drops to ~0.5%. The resilience may not hold. This should be tested in exp_bitnet_scaffold_fresh_adapters.

3. **GaLore scaffold is the strongest base-free candidate.** It produces good composition without explicit optimization. Future work should improve adapter quality ON GaLore scaffolds rather than trying to build better scaffolds.

4. **If scaffold meta-optimization is revisited**, use Reptile or iMAML, not FOMAML. Include STE in the outer loop and a scaffold preservation term (KL or PPL constraint).

5. **consecutive_kills=2** (this + basefree_exploration). The base-free track needs a win. exp_bitnet_scaffold_fresh_adapters (training ON random scaffold, not testing against it) is the next best candidate.

## New References to Add

1. **iMAML**: Rajeswaran et al., "Meta-Learning with Implicit Gradients", NeurIPS 2019, arXiv:1909.04630
   - Relevance: Correct meta-gradient computation independent of inner-loop length
   - Nodes: exp_bitnet_meta_scaffold (killed), future scaffold meta-optimization

2. **Reptile**: Nichol et al., "On First-Order Meta-Learning Algorithms", 2018, arXiv:1803.02999
   - Relevance: FOMAML bias analysis + Reptile as simpler alternative
   - Nodes: exp_bitnet_meta_scaffold (killed), future scaffold meta-optimization

3. **Recover-LoRA**: "Recover-LoRA: Data-Free Accuracy Recovery of Degraded Language Models via Low-Rank Adaptation", arXiv:2510.08600
   - Relevance: LoRA-based recovery from model degradation (confirms adapter resilience)
   - Nodes: exp_bitnet_meta_scaffold, exp_bitnet_scaffold_fresh_adapters
