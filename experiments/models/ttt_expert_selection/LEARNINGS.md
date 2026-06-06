# Learnings: exp_ttt_expert_selection_mlx

## Core Finding
Test-time expert selection via loss-probing and cosine centroids cannot beat the learned Gumbel-sigmoid router at N=49 (K2 FAIL: 15.27 vs 15.07, 1.3% gap), but reveals that the routing problem is largely solved by simple cosine similarity (97% of router quality with zero training) and that direct loss measurement provides a quality ceiling the router has not reached.

## Why This Happened (Literature-Grounded)

**The routing problem at N=49 is a nearest-centroid classification problem.** Our 49 domain-specialized adapters produce well-separated clusters in BitNet-2B-4T's hidden-state space. Mean-pooled hidden states carry strong domain signal because the base model's representations already encode domain identity (a property of pre-trained LLMs). The Arrow/MBC paper ("Towards Modular LLMs by Building and Reusing a Library of LoRAs") confirms this: their Model-Based Clustering groups tasks by adapter parameter similarity, and Arrow zero-shot routing achieves competitive performance by exploiting this separability without retraining.

**Loss-probing beats the router because it measures the objective directly.** The learned router approximates loss reduction through a 2-layer MLP on hidden states -- an indirect proxy. Exhaustive probing evaluates the actual objective (next-token prediction loss on the input). This is expected: any approximation loses information. The 93.9% vs 86.3% accuracy gap (the cleaner comparison, per reviewer -- the 14.91 vs 15.07 PPL comparison has partial data leakage from probe-eval text overlap) quantifies how much signal the router's bottleneck discards.

**Arrow-style projection fails due to Johnson-Lindenstrauss concentration.** With d=2560 and r=16, random-uniform A-matrices project all inputs into nearly equidistant subspaces. The JL lemma predicts that random projections preserve pairwise distances up to epsilon, meaning projection energies ||A_i^T h||^2 concentrate around the same value regardless of adapter identity. The 49% accuracy (near combinatorial baseline for top-2/49) confirms this concentration effect empirically.

**The K2 fail is marginal because our setting differs from standard MoE.** In standard MoE (Switch Transformer, GShard), the router and experts co-train -- experts specialize based on router decisions, creating a symbiotic relationship that loss-probing of frozen experts cannot replicate. Our setting has pre-trained, frozen experts with a post-hoc router. This weaker coupling means the router's advantage over simple heuristics is smaller. The gumbel_sigmoid_ablation LEARNINGS confirmed this: our router training regime differs fundamentally from co-trained MoE.

## Confirming Evidence

- **Arrow/MBC (Towards Modular LLMs)**: Zero-shot Arrow routing on a LoRA library matches or outperforms joint training, confirming that adapter selection is tractable without learned routing when domains are separable. Our cosine centroid (97% quality) is a simplified version of this principle.
- **Our gumbel_sigmoid_ablation**: Found training budget is primary accuracy driver, LB loss has zero aggregate benefit. Consistent with TTT finding that simple heuristics nearly match the router -- the router's learned features capture marginal value over direct measurement.
- **Our N=50 composition experiment**: gamma_routed=0.632 (37% PPL improvement, 49/49 domains below base). Routing captures 99.6% more composition benefit than uniform averaging. This confirms routing matters enormously for composition quality, even if the specific routing mechanism matters less.
- **Standard MoE literature**: Loss-probing as oracle is a known upper bound in MoE evaluation. It's never used in production due to O(N) cost, but it serves as a diagnostic ceiling (our exact usage).

## Contradicting Evidence

- **Co-trained MoE routers outperform probing of frozen experts.** In standard Switch Transformer / GShard settings where router and experts co-train, the router develops decision boundaries that experts actively adapt to. This symbiotic specialization creates a compounding advantage that post-hoc methods (including loss-probing) cannot access. Our setting (frozen pre-trained experts, router-only training) is fundamentally different -- the gap between learned routing and heuristics should be larger in co-trained settings.
- **Cosine similarity degrades at scale.** NotebookLM notes that at higher N with overlapping domains, cosine centroid accuracy will degrade as the JL concentration effect applies to centroids too -- distances between centroids converge. Our 9 PPL=1.0 domains (trivially routed) inflate the 65% accuracy; on real-data domains only, accuracy is ~55-60%. At N=500+, this could drop below utility threshold.
- **Single-sample evaluation inflates confidence.** The reviewer correctly identified that our 1-sample-per-domain evaluation (vs router's 10-sample) produces high-variance accuracy estimates. The 93.9% for exhaustive probe could shift significantly with multi-sample evaluation.

## Alternative Approaches (What We Could Try Instead)

1. **Contrastive routing (InfoNCE/ArcFace):** Train the router with contrastive loss to push domain embeddings apart in latent space, overcoming JL concentration. This directly addresses the cosine centroid's failure mode at scale -- margin-based losses maintain distinct decision boundaries even as domains become more similar. Could rescue projection-based scoring that Arrow fails at.

2. **Attention-weighted pooling:** Replace mean pooling with a learned attention head that weights tokens by routing salience. Mean pooling washes out domain-distinctive tokens (e.g., "translate" vs filler words). The gumbel_sigmoid_ablation found dialogue unroutable via mean-pooling -- attention pooling could specifically fix this.

3. **Prototypical networks:** Maintain moving-average prototype embeddings per expert (updated during training). Route via Euclidean distance to prototypes. More robust than fixed centroids for domains with high intra-class variance. Borrows from few-shot classification literature.

4. **Hierarchical/tree routing:** Route first to coarse categories (code vs prose vs science), then to fine-grained experts. Reduces the N=49 routing problem to multiple smaller problems. Our data shows clear coarse clusters (the mixed-domain experiment found router learned 2 effective classes: code/math vs prose).

5. **Model-Based Clustering (MBC):** From the Arrow paper -- cluster adapters by parameter similarity rather than input features. Could improve composition by grouping adapters that interact constructively, rather than routing purely by input domain.

6. **Loss-probe as router validation metric:** Don't deploy loss-probing, but use it offline to measure router quality ceiling. If router PPL >> probe PPL, the router needs improvement. This is the key practical takeaway.

## Implications for Next Experiments

1. **Cosine centroid is the deployment baseline.** For rapid deployment or new domains, compute centroids from training data (no router training needed). 97% quality with zero training cost.

2. **The router's value is in hard cases.** The 14% of domains where cosine fails (chemistry, dialogue, debate, wikitext overlap) are where learned routing adds value. Future router work should target these confusion domains specifically, perhaps via contrastive training.

3. **Arrow projection needs Grassmannian A-matrices.** Random A-init killed projection scoring, but our Grassmannian-optimized A-matrices are specifically designed to be discriminative. This is an untested combination worth exploring -- it could rescue the Arrow mechanism.

4. **Longer training defaults confirmed.** The gumbel_sigmoid_ablation found training length is the primary accuracy driver. TTT's 32-token prefix is short context. Both point to the same lesson: give the routing system more signal (longer training, longer prefix).

5. **Multi-sample evaluation is mandatory.** Single-sample-per-domain evaluation inflates variance and produces unreliable accuracy estimates. All future routing experiments should use >= 5 samples per domain.

6. **Loss-probing should become a standard diagnostic.** Run exhaustive probe on any new routing method to measure its ceiling gap. If gap > 5%, the router architecture is losing substantial information.
