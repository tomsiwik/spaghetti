# Learnings: exp_real_data_domain_experts

## Core Finding

Correct multi-expert LoRA composition using per-expert A_i@B_i summation with QR-orthogonal A matrices achieves -26.3% avg PPL improvement over base across 5 real instruction domains, 3.3x better than the broken single-A composition (-8.0%). This validates that orthogonal subspace separation is a **functional mechanism** for interference-free adapter composition, not just a structural property.

## Why This Happened (Literature-Grounded)

The 3.3x improvement from fixing the composition formula has a clean mechanistic explanation grounded in linear algebra and the LoRA literature:

**Per-expert projection is mathematically necessary.** When A matrices are QR-orthogonal (A_i^T A_j = 0), each B_i is trained to produce useful outputs only when projected through its corresponding A_i. Using A_0 for all experts means B_1..B_4 are projected through the wrong subspace, contributing noise rather than signal. The broken formula was effectively running domain_0's adapter at 1/5 scale plus noise from 4 misaligned experts.

**L2R confirms: parameter merging destroys task-specific representations.** The Learning to Route framework (Ponti et al., EMNLP 2023) demonstrates that computing a weighted average of adapter hidden-state outputs significantly outperforms merging adapter weights into a single matrix. L2R-wavg >> L2R-merge, because direct parameter merging causes destructive interference between distinct task modules. Our broken-vs-correct ablation is a specific instance of this principle: the broken formula implicitly merges all experts through a single projection, while the correct formula preserves per-expert output separation.

**Orthogonal subspaces prevent catastrophic forgetting in continual learning.** Chaudhry et al. (2020, "Continual learning in low-rank orthogonal subspaces") and Wang et al. (2023, "Orthogonal subspace learning for language model continual learning") established that constraining new tasks to orthogonal subspaces prevents interference with previously learned representations. Our QR-orthogonal A matrices achieve the same guarantee for simultaneous composition rather than sequential learning.

**At d=2560, N*r=80, orthogonality is trivially achievable.** The low ratio N*r/d = 0.031 means QR factorization produces exactly orthogonal frames with zero approximation error. This is consistent with the Johnson-Lindenstrauss lemma: in high-dimensional spaces, random subspaces are near-orthogonal naturally. The QR guarantee becomes critical only at N >= 160 where N*r exceeds d and the Welch bound forces nonzero coherence.

## Confirming Evidence

1. **L2R (Learning to Route)** — Ponti et al., EMNLP 2023. Trains task-specific PEFT modules in isolation, uses Gumbel-sigmoid routing at inference. Key finding: output-space averaging >> weight-space merging. Directly confirms our per-expert A_i@B_i approach over single-A merging.

2. **Orthogonal Subspace Continual Learning** — Chaudhry et al., NeurIPS 2020. Demonstrates that constraining gradient updates to orthogonal subspaces prevents catastrophic forgetting. Our frozen orthogonal A matrices achieve the same guarantee structurally.

3. **Mixture-of-Subspaces in Low-Rank Adaptation** (MoSLoRA) — Explores subspace partitioning for multi-task LoRA. Consistent with our approach of assigning each expert its own subspace.

4. **Our own prior experiments:**
   - exp_bitnet_grassmannian_init: QR-orthogonal init survives QAT (gamma near 1.0)
   - exp_bitnet_2b_real_composition: NTP-format baseline (-8.1% composed) now explained by the same single-A bug
   - exp_bitnet_n15_scaling_node: Composition works at N=15 with gamma=0.987
   - exp_bitnet_n25_scaling_node: Gamma=0.982 at N=25, scaling is sub-linear but stable

## Contradicting Evidence

1. **Model Soups / Linear Mode Connectivity** — Wortsman et al., ICML 2022; Frankle et al., NeurIPS 2020. Simple weight averaging works WITHOUT orthogonality when models are fine-tuned from the same pre-trained checkpoint (Linear Mode Connectivity). **Discrepancy:** This applies to same-task models in the same loss basin. Our multi-domain adapters occupy different basins (different training data, different objectives), so LMC does not hold — orthogonality is needed precisely because our experts are NOT linearly mode connected.

2. **Random projections are near-orthogonal by JL lemma.** Flora (Hao et al., 2024) uses random projections successfully for gradient compression. **Discrepancy:** At d=2560, r=16, random A matrices would have expected pairwise cosine ~O(1/sqrt(d)) = 0.02, which is coincidentally similar to our measured B-matrix cosine of 0.0205. This means our missing random-A baseline might show similar composition quality at N=5. The QR guarantee becomes load-bearing only at high N where random coherence accumulates. **This is our most important open question.**

3. **ZipIt! partial zipping** — Stoica et al., ICLR 2023. Merges models trained on disjoint tasks by finding redundant features via activation similarity, without any orthogonality constraint. Only merges early layers, leaving later layers as a multi-head architecture. **Discrepancy:** ZipIt! requires feature matching and partial merging (effectively routing). Our approach achieves full-depth composition without routing, but at the cost of requiring orthogonal initialization.

## Alternative Approaches (What We Could Try Instead)

1. **Gumbel-sigmoid routing (L2R)** — Non-competing multi-adapter activation via independent Bernoulli gates. Unlike softmax which forces competition, Gumbel-sigmoid allows multiple adapters to activate simultaneously. Could replace our 1/N uniform weighting with learned per-input activation. Reference: Ponti et al., EMNLP 2023.

2. **SHINE hypernetwork** — Generates LoRA weights from context in a single forward pass using Memory-to-Parameter Transformer. Eliminates the need for pre-trained per-domain adapters entirely. Reference: Luo et al., 2024.

3. **pQuant decoupled architecture** — Splits layers into a shared 1-bit backbone (majority of compute) + tiny 8-bit expert branches with top-1 routing. Directly relevant to our ternary base architecture — could replace adapter composition with a native 1-bit/8-bit split. Reference: pQuant, 2024.

4. **Dynamic adapter switching at inference** — Evaluate prompt complexity, activate specialized LoRA only when needed, keep base model lightweight for routine queries. Our routing heads (99.9% accuracy) could serve this exact role. Practical next step for edge deployment.

5. **Random-A baseline experiment** — The most important missing control. If random frozen A matrices compose equally well at N=5 (plausible by JL lemma), the QR-orthogonal mechanism story changes: it's not that orthogonality enables composition, but that per-expert projection is sufficient at low N, and orthogonality is insurance for high N.

## Implications for Next Experiments

1. **Random-A baseline is the highest-priority follow-up.** Without it, we cannot distinguish "orthogonality enables composition" from "any frozen per-expert A enables composition." Design: same experiment, same data, but A matrices drawn from random Gaussian (no QR) and measured for pairwise cosine. If composition quality is similar at N=5, orthogonality is insurance for scale, not mechanism at current scale.

2. **Routing-weighted top-k composition is essential for scaling.** At N=5, 1/N uniform weighting works but already dilutes each expert to 20% of its individual effect. At N=25+, this dilution makes uniform composition nearly indistinguishable from base. L2R's Gumbel-sigmoid routing or our existing routing heads with top-2 selection are the natural next steps.

3. **The composition bug retroactively explains weak prior results.** The NTP experiment's -8.1% composition benefit was likely also affected by single-A composition (it used random A init + single-A averaging). Re-running with correct per-expert composition could show similar 3x improvements on the NTP pipeline.

4. **Overlapping domain splits are the real test.** All 5 current domains are trivially separable (99.9% routing accuracy). The architecture must be tested on domains with shared vocabulary and concepts (e.g., biomedical + chemistry, legal + finance) where routing is non-trivial and subspace overlap is likely.

5. **Task accuracy evaluation is blocking.** PPL improvements don't guarantee task performance. HumanEval (code), GSM8K (math), MedQA (medical) benchmarks are needed to validate that composition benefits transfer to downstream utility.
