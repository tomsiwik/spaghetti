# LEARNINGS: DUME Ridge Regression Router

## Core Finding

Ridge regression on mean-pooled hidden states achieves 96% routing accuracy at N=5, but this matches the much simpler nearest-centroid cosine method exactly. The Woodbury incremental update (12ms vs 100ms refit) is the only differentiating feature. Given that ALL centroid/linear methods collapse from 96% to 33% at N=24 (finding #257), ridge regression will face the same phase transition — this is a feature-space limitation, not an algorithm limitation.

## Why This Happened

**Ridge = nearest centroid at N=5** because domain separation in BitNet-2B-4T's hidden space is strong enough that any reasonable linear method saturates. The Fisher discriminant ratio of 1.24 is sufficient for 5 well-separated domains (medical/code/math/legal/finance), but the legal-finance centroid cosine of 0.981 already signals the collapse mechanism.

The equivalence between ridge regression and nearest centroid for well-separated Gaussian classes is a known result in statistical learning theory. When class-conditional distributions are Gaussian with shared covariance and equal priors, the ridge solution W* converges to the LDA (Fisher Linear Discriminant) solution, which is equivalent to nearest-centroid classification in Mahalanobis space (Fisher, 1936; Hastie et al., Elements of Statistical Learning, Ch. 4.3).

DUME (arXiv 2603.29765) demonstrated this at 115M and 3B Llama scale with multiple domain experts. Our result independently confirms the mechanism but reveals that the closed-form solve provides no accuracy advantage over the trivially simple centroid approach at small N.

## Confirming Evidence

1. **DUME** (arXiv 2603.29765): Ridge regression gating for training-free MoE upcycling. Validated at 115M and 3B Llama. Directly inspires our experiment. Key claim: incremental expert addition via Woodbury identity.

2. **Self-Routing** (arXiv 2604.00421, Mohamud et al., Mila 2026): Parameter-free routing using a subspace of hidden states directly as expert logits — no learned router at all. Competitive with learned routers at GPT-2 scale with 17% higher routing entropy. Confirms hidden states contain sufficient routing signal without any classifier.

3. **Finding #254** (our exp_lorauter_task_routing): Sentence-embedding routing achieves 96% at N=5, matching our ridge result. Confirms that N=5 domain routing is trivially solved by multiple methods.

4. **Finding #276** (this experiment): Lambda insensitivity across [0.01, 10.0] confirms domains are well-separated and the regularization parameter is not load-bearing.

## Contradicting Evidence

1. **Representation Collapse in Sparse MoE** (arXiv 2204.09179, Chi et al., NeurIPS 2022): Hidden representations cluster around expert centroids as N increases, collapsing to an N-dimensional subspace. Expert vectors span at most N dimensions, so the routing subspace shrinks relative to the full d-dimensional hidden space. This predicts that ridge regression (a linear method on this same space) will degrade as N grows.

2. **Finding #257** (our exp_sentence_embedding_routing_n24): ALL centroid-based methods collapse from 96% (N=5) to 33.3% (N=24). Root cause: general-purpose feature spaces embed domain identity as a minor feature. At N=24, inter-centroid distances shrink below the margin needed for reliable routing. "No routing algorithm can fix it without domain-specific embeddings."

3. **Finding #256** (our exp_sentence_embedding_routing_n24): Fisher ratio 2.93 at N=24 looks healthy, but 91/276 centroid pairs exceed confusion threshold. The aggregate statistic masks the bimodal distribution: 7 domains >= 80% accuracy (margin > 0.10), 6 domains = 0% (margin < 0.05).

4. **Finding #28** (softmax router): Softmax router already matches oracle at N=24. Learned routing solves N=24; linear/centroid methods do not.

## Alternative Approaches

1. **Routing-Free MoE** (arXiv 2604.00801, Liu et al., 2026): Eliminates external routers entirely. Each expert determines its own activation through continuous gradient flow. Outperforms baselines with better scalability. However, requires training — not applicable to our zero-training setting.

2. **HiLoRA** (NPoMZuiHnM, OpenReview 2025): Training-free hierarchical LoRA routing using Gaussian likelihoods over rank-one components. Operates at a finer granularity than per-adapter routing. Could address the centroid collapse by routing at the rank-one level instead of the adapter level.

3. **Softmax router with Gumbel-sigmoid** (our existing approach): Already proven at N=24 with oracle-matching accuracy. Requires ~330K trained parameters but is the only method that has survived the N=24 phase transition.

4. **Domain-specific embeddings**: Finding #257's impossibility structure says general-purpose feature spaces cannot separate 24+ domains. The fix is domain-aware features — either fine-tuned embeddings or adapter-conditioned hidden states (extract hidden states WITH adapters loaded, not just base model).

## Implications for Next Experiments

1. **Ridge regression adds no accuracy value over nearest centroid** at any N we've tested. The Woodbury update is the only advantage, and it's only useful if the underlying routing is accurate. At N=24, it won't be.

2. **The N=24 routing problem is unsolved by linear methods.** Ridge regression, nearest centroid, TF-IDF, and sentence embeddings all fail. The learned softmax router succeeds. This points to a fundamental gap: learned nonlinear routing captures structure that linear methods cannot.

3. **For the deployment track (P0)**, ridge regression is a viable quick-start router for N<=5 (where everything works), but the softmax router remains necessary for N=24 scaling. Ridge could serve as initialization for the softmax router — cold-start with ridge, then fine-tune.

4. **Woodbury incremental update is genuinely useful for hot-adding adapters** if combined with a method that works at scale. A hybrid: softmax router at N=24, Woodbury for fast provisional routing of the (N+1)th adapter before full retraining.

## Recommended Follow-Up

**No new experiment recommended.** The ridge regression router is a nice-to-have convenience for N<=5 but does not advance the critical path. The deployment track needs exp_generation_quality_test (P0), not more routing methods.

If routing at scale is revisited, the promising direction is **adapter-conditioned hidden states**: extract routing features from the model WITH each adapter loaded, creating adapter-specific representations that carry domain signal. This is motivated by finding #257's impossibility structure and by the observation that Self-Routing (2604.00421) uses hidden states from within the MoE forward pass (where experts are active), not from the bare base model.
