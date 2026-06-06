# Learnings: exp_bitnet_scaffold_fresh_adapters

## Core Finding

Fresh LoRA adapters trained directly on a random ternary scaffold converge (43-58% loss
reduction) but achieve 36-642x worse PPL than identical adapters on pretrained BitNet-2B-4T.
This kills the base-free scaffold path at two levels (pretrained adapters: 319M PPL,
fresh adapters: 186-2887 PPL) while confirming FreezeNet's gradient flow principle
extends to 2B-scale ternary models.

## Why This Happened (Literature-Grounded)

Three converging mechanisms explain the 36-642x gap:

**1. Information-Theoretic Capacity Bottleneck**

Rank-16 LoRA applied to all 7 projections across 30 layers yields ~23.5M trainable
parameters (0.98% of the 2.4B base). With ternary quantization (1.58 bits/param), this
is ~3.4MB of information capacity. A language model at 2B scale encodes gigabytes of
knowledge in its pretrained weights — the adapter cannot reconstruct this from scratch.
This is consistent with Hu et al.'s (arXiv:2106.09685) observation that LoRA exploits
the "low intrinsic dimensionality" of the *adaptation* task, not the *pretraining* task.

Aghajanyan et al. (arXiv:2012.13255) showed pretrained models have low intrinsic
dimension for fine-tuning: RoBERTa achieves 90% of full performance with only 200
trainable parameters projected back into full space. The critical insight is that
pretraining *creates* this low-dimensional structure — without it, the intrinsic
dimension of language modeling is enormous, far exceeding rank-16 capacity.

**2. Random Features Hit Known Capacity Ceiling**

Our experiment is structurally equivalent to Rahimi & Recht's (NIPS 2007) random
features framework: a fixed random projection (the scaffold) plus a thin learned layer
(the LoRA adapters). Random feature methods provably approximate kernel methods but
with capacity bounded by the number of random features, not the dimensionality of the
feature space. At rank-16, the adapter can only express a rank-16 perturbation of a
random function — fundamentally insufficient for language modeling.

The "LoRA Without Regret" analysis confirms that lower-rank LoRAs fall off the
minimum-loss curve when the adapter exhausts its capacity, and this threshold
correlates directly with rank. On a pretrained base, rank-16 suffices because the
adaptation task is low-dimensional. On a random base, the "adaptation" task IS
pretraining — which requires full-rank updates.

**3. FreezeNet Validates But At Wrong Scale**

FreezeNet (arXiv:2011.14087) showed random frozen weights support gradient flow for
classification on MNIST/CIFAR (achieving 99.2% of baseline on MNIST). Our experiment
confirms this extends to 2B-scale ternary language models: all 4 domains converge with
43-58% loss reduction. But FreezeNet's success was on classification tasks where the
output space is small (10-1000 classes). Language modeling has a 32K-token output space
at every position — the capacity demands are qualitatively different.

## Confirming Evidence

| Paper | Key Finding | Relation |
|-------|-------------|----------|
| Aghajanyan et al. (arXiv:2012.13255) | Pretrained models have low intrinsic dimension; pretraining creates this structure | **CONFIRMS** rank-16 sufficient for adaptation, insufficient for pretraining from scratch |
| Hu et al. LoRA (arXiv:2106.09685) | Weight updates during adaptation have low "intrinsic rank" | **CONFIRMS** LoRA exploits pretrained structure, not general compression |
| Rahimi & Recht (NIPS 2007, 2008) | Random features approximate kernels with capacity bounded by feature count | **CONFIRMS** thin trainable layer on random base hits capacity ceiling |
| FreezeNet (arXiv:2011.14087) | Random frozen weights support gradient flow; 99.2% baseline on MNIST | **CONFIRMS** convergence on scaffold; we extend to 2B ternary scale |
| TernaryLM/Continual QAT (arXiv:2602.07374, arXiv:2502.11895) | 16→1.58-bit training strategy beats full 1.58-bit from scratch | **CONFIRMS** pretrained knowledge bootstraps ternary learning |
| ReLoRA (arXiv:2307.05695) | Low-rank updates need "warm start" full-rank phase for training from scratch | **CONFIRMS** pure low-rank on random init is insufficient; iterative merge + warm start needed |

## Contradicting Evidence

| Paper | Key Finding | Discrepancy |
|-------|-------------|-------------|
| FreezeNet (arXiv:2011.14087) | 99.2% baseline with few trainable params on random base (MNIST) | Apparent contradiction resolved by task scale: MNIST classification (10 classes) vs language modeling (32K vocabulary). FreezeNet's trainable parameter fraction was also higher (~10-30% vs our 0.98%). |
| Lottery Ticket Hypothesis (arXiv:1803.03635) | Random networks contain winning subnetworks | Not directly contradicting — lottery tickets require the FULL network to be trainable, then pruned. Our experiment freezes the base and trains only 0.98% via LoRA. The winning tickets exist but cannot be found with a rank-16 probe. |

No paper was found demonstrating successful language model training with rank-16 LoRA
on a fully random base. The closest positive result is ReLoRA, which achieves comparable
performance to full-rank training but requires (a) warm-start full-rank phase and
(b) iterative merge-and-reinitialize cycles that accumulate effective rank over time.

## Alternative Approaches (What We Could Try Instead)

**1. GaLore Scaffold (exp_bitnet_galore_scaffold — SUPPORTED, next in pipeline)**

GaLore (arXiv:2403.03507) projects gradients into a low-rank subspace for memory
efficiency while maintaining full-parameter learning capability. Unlike LoRA-on-scaffold,
GaLore updates ALL base parameters through low-rank gradient projections. This sidesteps
the capacity bottleneck: the scaffold progressively accumulates full-rank knowledge
through iterative subspace rotations. GaLore has demonstrated LLaMA-7B pretraining
on consumer GPUs (24GB), reducing optimizer memory by 65-82%. GaLore 2 (arXiv:2504.20437)
addresses scalability with FSDP integration. This is the most promising base-free path.

**2. ReLoRA Iterative Training (untested)**

ReLoRA (arXiv:2307.05695) trains low-rank LoRA modules, periodically merges them into
the base weights, and reinitializes fresh LoRA modules. This progressively builds
full-rank representations from an initial low-rank budget. Key requirement: a short
full-rank "warm start" phase before switching to low-rank updates. Could be tested
on random ternary scaffold: warm start with small full-rank phase, then ReLoRA cycles.
Predicted gap: much smaller than 36-642x, potentially viable if warm-start is sufficient.

**3. Knowledge Distillation into Scaffold (BitDistill approach)**

Use pretrained BitNet-2B-4T as teacher, random scaffold as student. The student learns
to mimic the teacher's output distribution, effectively transferring knowledge into the
scaffold weights. This preserves the scaffold's ternary structure while encoding pretrained
knowledge. Not tested in our pipeline but well-established technique.

**4. SLTrain: Sparse + Low-Rank (NeurIPS 2024)**

SLTrain combines sparse and low-rank updates for memory-efficient training from scratch.
Unlike pure LoRA, the sparse component provides higher effective rank. Could offer a
middle ground between full-rank warm start and pure low-rank training on scaffold.

## Implications for Next Experiments

1. **The pretrained base is load-bearing**: 36-642x gap quantifies that >99% of model
   utility comes from pretrained weights, not adapter tuning. BitNet-SOLE must use a
   pretrained base for the foreseeable architecture. Base-free is a long-term research
   goal, not a near-term option.

2. **GaLore is the right next step for base-free**: It's the only method that updates
   all parameters through low-rank projections, avoiding the capacity bottleneck.
   exp_bitnet_galore_scaffold should proceed with high priority.

3. **ReLoRA deserves a node**: If GaLore scaffold also kills, ReLoRA iterative training
   with warm start is the natural fallback. Consider adding an HYPOTHESES.yml node for
   exp_bitnet_relora_scaffold with deps on galore_scaffold.

4. **FreezeNet at 2B ternary is a citable positive result**: K2 PASS (all domains
   converge) extends FreezeNet from CIFAR/MNIST to 2B-param ternary language models.
   This is a novel data point worth highlighting in future publications.

5. **Orthogonality is geometric, not knowledge-dependent**: Mean |cos| 0.0021 on
   scaffold vs 0.0029 on pretrained confirms that adapter orthogonality is a property
   of Gr(r, d) at d=2560, independent of base quality. This strengthens the Grassmannian
   skeleton theory.

## New References to Add

| Paper | ArXiv ID | Relevance |
|-------|----------|-----------|
| Intrinsic Dimensionality Explains Fine-Tuning | arXiv:2012.13255 | Explains why LoRA works on pretrained but not random: pretraining creates low-dimensional structure |
| ReLoRA: High-Rank Training Through Low-Rank Updates | arXiv:2307.05695 | Iterative merge approach for training from scratch; requires warm start |
| GaLore 2: Large-Scale LLM Pre-Training | arXiv:2504.20437 | Scaled gradient low-rank projection; direct competitor to ReLoRA for base-free |
| SLTrain: Sparse + Low-Rank | NeurIPS 2024 | Combines sparse and low-rank for from-scratch training |
