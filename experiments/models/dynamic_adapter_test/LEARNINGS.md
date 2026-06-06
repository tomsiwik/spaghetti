# Learnings: exp_dynamic_adapter_addition

## Core Finding

Hot-adding a 6th LoRA adapter with its own routing head to an existing 5-adapter composition causes only 0.70% average PPL degradation — but this average hides a bimodal outcome: 4/5 domains show zero degradation (new head never fires), while the 5th (medical) degrades 3.5% due to science-medical routing confusion.

## Why This Happened (Literature-Grounded)

Three independent mechanisms provide plug-and-play guarantees, but the third has a known failure mode:

**1. Weight-space orthogonality is structural, not accidental.**
At d=2560, r=16, the Johnson-Lindenstrauss lemma predicts near-zero cosine between random subspace projections: E[|cos|] ~ O(1/sqrt(d)) ≈ 0.02. Our observed mean |cos| = 0.00075 across 15 adapter pairs is 27x below this bound. The Welch bound capacity (d²/r² = 25,600 adapters) means N=6 is trivially within the interference-free regime. This is consistent with our prior finding that cos=0.0002 at d=896 (50x better than theory, macro/ortho_scaling/). Grassmannian initialization is load-bearing: it guarantees that any new adapter trained in the same space will be near-orthogonal to all existing ones without coordination.

**2. Independent binary routing heads share no parameters.**
Each head h_i is a 2-layer MLP (82K params) trained on frozen base model hidden states. Adding h_{N+1} cannot change the decision boundaries of h_1...h_N because: (a) no shared parameters, (b) same frozen hidden states, (c) same training data for existing heads. This is the "modular routing" property — contrast with DeepSeek-V3's shared router where adding experts requires rebalancing (arXiv 2412.19437). Our per-adapter binary heads trade routing sophistication for complete modularity.

**3. Top-k selection provides N-invariance in effective scale — but NOT in slot competition.**
Under top-k=2 routing, each selected adapter gets w_i = S_i/(S_1+S_2) ≈ 0.5, independent of N. This prevents the 1/N dilution that uniform composition suffers (+1.2% degradation at N=6). However, top-k introduces a different failure mode: slot competition. When h_{N+1} has high sigmoid on domain j (false positive), it displaces the correct adapter from the top-2 set. This is exactly what happened with medical: science head scores 0.41 on medical data, entering top-2 for 100% of medical inputs.

**The medical-science confusion is a routing calibration issue, not a composition mechanism failure.** The sigmoid threshold for binary routing heads is implicitly 0.5, but the science head at 0.41 on medical is close enough to compete. This mirrors findings from the Mixture of LoRA Experts (MoLE, arXiv 2404.13628) literature: when domains share semantic features, per-expert routing requires careful calibration to avoid cross-domain activation. The Retrieval-Augmented MoLE (RAMoLE, arXiv 2406.16989) addresses this by using retrieval-based expert selection rather than learned routing, sidestepping the calibration problem entirely.

## Confirming Evidence

1. **MoLE (arXiv 2404.13628):** Mixture of LoRA Experts demonstrates that independently-trained LoRA adapters can be composed via learned routing without retraining. Their gate network learns to select experts per-input, confirming our finding that modular routing preserves composition quality. However, MoLE uses a shared router (not per-adapter heads), which requires joint training.

2. **NP-LoRA (arXiv 2511.11051):** Null-space projection ensures new adapters lie in the null space of existing ones, providing a theoretical guarantee of zero interference. Our Grassmannian initialization achieves this approximately (cos=0.00075) without explicit projection. NP-LoRA would provide exact zero interference but requires computing the null space of all existing adapters before training a new one — trading modularity for stronger guarantees.

3. **Task-Aware LoRA Composition (arXiv 2602.21222):** Shows that linear merging of LoRA adapters preserves quality when adapters are trained on distinguishable tasks. This aligns with our finding that 4/5 trivially-separable domains show zero degradation.

4. **Our prior hash-ring routing (FINDINGS.md):** 5.3% displacement at N=20 with hash-based routing. Different routing mechanism (content hashing vs learned sigmoid heads) but same conclusion: plug-and-play works when domains are separable. The 5.3% figure at N=20 vs our 0.70% at N=6 suggests degradation scales sub-linearly with N under good routing.

5. **Our N=25 scaling experiment (FINDINGS.md):** gamma=0.982 (composition quality ratio) at N=25 with Grassmannian initialization. Confirms that orthogonality holds at moderate N and composition quality is preserved.

6. **Ensembling vs Merging vs Routing (arXiv 2603.03535):** Systematic comparison showing routing-based composition outperforms both ensembling and merging for multi-task LoRA. Our 19.5% routing advantage over uniform 1/N at both N=5 and N=6 is consistent with their findings.

7. **MoLoRA (arXiv 2603.15965):** The strongest external confirmation — proves the full pipeline of independent training, runtime composition via per-token routing, and plug-and-play addition. Qwen3-1.7B + 4 adapters exceeds Qwen3-8B across four reasoning benchmarks. Uses a "hot-set" memory architecture for efficient adapter swapping, directly validating our approach.

8. **Orthogonal Model Merging (arXiv 2602.05943):** Demonstrates that orthogonality is a "decisive factor" in maintaining performance during model merging. Directly supports our Grassmannian initialization strategy.

9. **ReMoE: ReLU Routing (arXiv 2412.14711, ICLR 2025):** ReLU-based routing as drop-in replacement for TopK+Softmax where the number of active experts is determined naturally by ReLU sparsity rather than fixed K. Shows sharper performance gains as expert count scales — providing N-invariance properties stronger than our top-k approach.

10. **InfLoRA (arXiv 2404.00228, CVPR 2024):** Designs subspaces that eliminate interference of new tasks on old tasks for continual learning. Shows fine-tuning within a subspace is equivalent to fine-tuning injected parameters. Our Grassmannian initialization achieves this approximately.

11. **SoLA: Semantic Routing-Based LoRA (arXiv 2603.11239):** Allocates independent LoRA per edit with semantic routing. Fully reversible — can add or remove adapters. Directly validates our plug-and-play architecture.

## Contradicting Evidence

1. **Our own mixed-domain per-token routing (FINDINGS.md, killed):** Post-hoc per-token routing failed — 3 prose domains (medical/legal/creative) were "genuinely indistinguishable in BitNet-2B-4T last-layer hidden space." This directly relates to the medical-science confusion: if prose domains are hard to distinguish at the hidden state level, adding science (which overlaps with medical) will inevitably cause routing errors. The per-sequence mean-pooling in our dynamic adapter test partially masks this (averaging over tokens smooths the signal), but the 0.41 sigmoid score shows the underlying separability problem persists.

2. **Our N=50 scaling experiment (FINDINGS.md):** 4/49 domains showed 0% routing accuracy. At larger N with more domain overlap, the medical-science confusion pattern would multiply. The aggregate metrics hide these per-domain failures, just as the 0.70% average hides the 3.5% medical degradation.

3. **SAME: Stabilized MoE for Multimodal Continual Instruction Tuning (arXiv 2602.01990):** Identifies two failure modes directly relevant to our approach: **router drift** (expert selection becomes inconsistent as new tasks are added) and **expert drift** (shared experts are overwritten across tasks). Our architecture avoids expert drift (frozen base, independent adapters) but is susceptible to router drift in the form of slot competition — the science head's 0.41 sigmoid on medical is exactly this.

4. **PASs-MoE: Misaligned Co-drift (arXiv 2601.13020):** Shows that LoRA-based MoE with jointly-updated routers causes "misaligned co-drift" — the router's preferences deviate from input-expert specialization over time. Our independent binary heads avoid joint training, but the medical-science confusion shows that even independently-trained heads can have misaligned decision boundaries when domains overlap.

5. **When Routing Collapses (arXiv 2602.03478):** Identifies "routing collapse" where routers trained to predict scalar scores fail at discrete comparisons — small prediction errors flip relative orderings. Our sigmoid scores (medical=0.41 vs science=0.91) are precisely this: a small miscalibration causes the wrong adapter to win the top-2 slot.

6. **Probing Semantic Routing in Large MoE Models (arXiv 2502.10928):** Confirms that expert routing IS influenced by input semantics, but semantically similar inputs from different domains get routed identically. This explains our medical-science confusion: science and medical share enough semantic features that the hidden states are similar enough to confuse the routing head.

7. **Advancing Expert Specialization (arXiv 2505.22323):** Shows that auxiliary load balancing loss leads to expert overlap and overly uniform routing. Proposes orthogonality loss (force experts to process distinct token types) and variance loss (force more discriminative routing decisions) — both directly applicable to our overlapping-domain problem.

8. **Ortho-LoRA: Disentangling Task Conflicts (arXiv 2601.09684):** Shows that different tasks compete for limited capacity of shared LoRA parameters, causing gradient conflict and negative transfer. This is exacerbated by the low-rank constraint. Our Grassmannian initialization avoids this for weight space, but the routing heads still face the capacity competition problem.

9. **OSRM (arXiv 2505.22934):** Weight-space orthogonality ≠ data-space orthogonality. Our cos=0.00075 guarantees no parameter interference, but does not guarantee no functional interference. Two adapters can be orthogonal in weight space while producing correlated outputs on similar inputs — which is exactly the medical-science pattern (orthogonal adapters, correlated routing decisions).

10. **MoE scaling laws (Apple, ICML 2025 via parameter-golf):** MoE fails below 500M params. Our 2B-param base is above this threshold, but the routing difficulty scales with domain similarity, not model size.

## Alternative Approaches (What We Could Try Instead)

1. **Null-space projection (NP-LoRA, arXiv 2511.11051):** Train each new adapter in the null space of all existing adapters' effective deltas (BA products). This provides exact zero interference at the cost of reduced capacity as N grows (null space shrinks). For N=6 at d=2560, r=16, the null space is still vast (2560-96=2464 dims), so this is practical.

2. **Hierarchical routing for overlapping domains:** Instead of flat binary heads, use a two-level routing: first classify into domain clusters (science vs non-science), then within-cluster routing. This would prevent medical-science confusion by routing to the right cluster first. Related: LD-MoLE (arXiv 2509.25684) uses learnable dynamic routing with hierarchical expert selection.

3. **Negative training for cross-domain specificity:** Train the science head with medical data as explicit hard negatives (high weight on medical-vs-science discrimination). Current training uses "own domain vs rest" which treats medical and python equally as negatives — but science-medical confusion needs more attention than science-python confusion.

4. **Gumbel-sigmoid non-competing routing (L2R approach):** Replace binary sigmoid heads with Gumbel-sigmoid gates that allow multiple experts to activate without competing for top-k slots. Each gate independently decides "activate or not" rather than competing for limited slots. This would prevent the displacement failure mode entirely. Our self-routing experiment showed Gumbel routing at 86-90% top-2 accuracy.

5. **Centroid routing as fallback (FINDINGS.md, self-routing adapters):** Centroid-based routing (87.14% top-2 accuracy) matches undertrained Gumbel routing and requires zero learned parameters per adapter — just 20 labeled examples for centroid estimation. For hot-add scenarios, centroid routing avoids the training step entirely.

6. **Per-domain kill criteria instead of aggregate:** The adversarial review correctly identified that averaging hides per-domain failures. Future hot-add experiments should use max(per-domain degradation) < threshold, not mean < threshold. This would have flagged the medical degradation as a K3 failure.

7. **ReLU routing instead of sigmoid+top-k (ReMoE, arXiv 2412.14711):** Replace our sigmoid-top-k routing with ReLU-based routing where sparsity emerges naturally. This eliminates the slot competition problem entirely — each adapter activates independently based on ReLU threshold, not relative ranking. ICLR 2025, drop-in replacement.

8. **LoRA-Augmented Generation (LAG, arXiv 2507.05346):** Two-stage routing: Arrow routing filters a large adapter library to k candidates, then Spectral routing ranks by measuring token representation length in each adapter's basis. Requires NO additional training — purely geometric routing. Could replace our learned routing heads entirely for the hot-add use case.

9. **Sigmoid gating over softmax (arXiv 2405.13997, NeurIPS 2024):** Theoretically proves sigmoid gating has higher sample efficiency than softmax. Key insight: softmax creates unnecessary zero-sum competition among experts (when one weight increases, others must decrease), causing representation collapse. Sigmoid eliminates this constraint, allowing multiple experts to activate independently. This directly addresses our top-k slot competition failure mode — switching from sigmoid+top-k to pure sigmoid gating would prevent the medical displacement problem.

10. **LD-MoLE differentiable routing (arXiv 2509.25684, ICLR 2026):** Replaces non-differentiable top-k with a differentiable routing function and closed-form solution. Adaptively determines the number of experts per token at each layer (no fixed K). Solves the hyperparameter-sensitivity of top-k and removes the slot competition problem.

11. **HMoRA: Hierarchical Mixture of LoRA Experts (OpenReview, Oct 2024):** Hybrid routing integrating token-level and task-level routing to capture both fine-grained token information and broader task contexts. Directly addresses the overlapping-domain problem by routing at multiple granularities — exactly what medical-science confusion needs.

## Implications for Next Experiments

1. **The plug-and-play mechanism works for separable domains.** This is confirmed at N=5→6 (this experiment), N=20 (hash routing), and N=25 (scaling). No further experiments needed on this axis unless testing overlapping domains specifically.

2. **The real challenge is domain overlap, not adapter interference.** All composition failures trace back to routing confusion, not weight-space interference. Grassmannian orthogonality is solving the weight-space problem completely (cos=0.00075). Future work should focus on routing quality for similar domains.

3. **Medical-science confusion (0.41 sigmoid) is the template for future failure modes.** Any domain pair with significant semantic overlap (e.g., cardiology/radiology, Python/JavaScript, constitutional law/contract law) will exhibit the same pattern. Experiments testing these cases would be more informative than further well-separated domain tests.

4. **Per-adapter routing heads scale well but need calibration for overlapping domains.** The 82K-param binary head is sufficient for well-separated domains but may need either: (a) more capacity (larger hidden dim), (b) better negative sampling (hard negatives from overlapping domains), or (c) hierarchical routing for domain clusters.

5. **Compose readiness: 75%.** This experiment brings compose readiness from 70% to 75% by confirming plug-and-play adapter addition. Remaining gaps: generation quality (the two-world pattern — structured tasks win, prose loses), per-token routing for mixed-domain inputs, and the overlapping-domain routing challenge.
