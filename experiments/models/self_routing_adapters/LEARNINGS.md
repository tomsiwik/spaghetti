# Learnings: Self-Routing Adapters

## Core Finding

B-matrix routing is killed by concentration of measure at r/d=0.006 (rank-16 in d=2560). Hidden-state centroid routing (87.14% top-2) is a strong closed-form baseline but is surpassed by a properly trained Gumbel-sigmoid router (90.41% at 6000 steps). The routing signal lives in the base model's representations, not in adapter weight matrices.

## Why This Happened (Literature-Grounded)

**B-matrix failure:** The concentration of measure phenomenon in high-dimensional spaces causes projections onto low-rank subspaces to become indistinguishable. L2R (arxiv 2601.21349) formally identifies this as "angular concentration" — in raw high-dimensional representation spaces, similarity scores concentrate, reducing effective separability among experts. Our empirical result (2% top-1, indistinguishable from random) at r/d=0.006 is the extreme case of this well-characterized phenomenon. The Grassmannian MoE paper (arxiv 2602.17798) directly addresses this by parameterizing routing via concentration parameters of Matrix Bingham distributions on the Grassmannian manifold, acknowledging that naive subspace-based routing fails exactly as we observed.

**Centroid success:** The finding that pretrained LLM hidden states cluster by domain is consistent with decades of work on learned representations. Prototypical Networks (Snell et al., NeurIPS 2017) established that nearest-centroid classification in pretrained feature space is a strong baseline when the encoder produces separable features. Our centroid routing is NCC applied to adapter selection — a well-known technique in a new context. The Arrow paper (Ostapenko et al., 2405.11157) similarly exploits pretrained representations for zero-shot routing, using SVD of LoRA matrices to craft prototype vectors. Our SVD projection result (0.8% top-1) confirms Arrow's known limitation: SVD bases of individual adapters are too generic to discriminate inputs without the full Arrow protocol.

**Why centroids lose to learned routers:** The 3.27pp gap between centroid (87.14%) and trained Gumbel-sigmoid (90.41%) reflects what contrastive/metric learning literature predicts — static centroids from few examples cannot capture the nonlinear decision boundaries that gradient optimization discovers. The centroid method uses the base model's representation as-is, while the Gumbel-sigmoid router learns to reshape this space for routing-specific discrimination.

## Confirming Evidence

1. **L2R (arxiv 2601.21349):** Identifies "representation mismatch, angular concentration, and scale-sensitive scoring" as jointly undermining routing discriminability in raw high-dimensional spaces. Proposes low-rank latent routing space with Saturated Inner-Product Scoring (SIPS) as fix — confirms our finding that routing must happen in a transformed space, not directly through adapter matrices.

2. **Grassmannian MoE (arxiv 2602.17798):** Explicitly models routing on the Grassmannian manifold of subspaces with concentration-controlled gating. Their framework acknowledges that naive subspace routing concentrates — the entire paper is a response to the failure mode we empirically confirmed.

3. **"Towards Modular LLMs" / Arrow (arxiv 2405.11157):** Uses adapter weight similarity (MBC) only for *library building* (offline clustering), NOT for test-time routing. For inference routing, they use Arrow (SVD-based prototypes) — implicitly confirming that raw adapter weights don't route well. Arrow itself uses the top right singular vector of each LoRA as a prototype, a more principled version of our SVD projection.

4. **SpectR (arxiv 2504.03454):** Improves on Arrow by leveraging the *entire* spectrum of the LoRA covariance matrix rather than just the top singular vector. Explicitly addresses Arrow's known weakness — confirms that single-vector SVD routing (our Method C) is insufficient.

5. **Effective LoRA Adapter Routing (arxiv 2601.21795):** Uses compact "task representations" for routing, confirming that some form of learned/computed representation (not raw weights) is needed for effective adapter selection.

## Contradicting Evidence

1. **Arrow claims to work:** Arrow (Ostapenko et al.) reports effective zero-shot routing using SVD of LoRA matrices. Our SVD projection result (0.8% top-1) appears to contradict this. **Resolution:** Arrow uses a multi-step protocol — SVD extraction → prototype construction → per-layer/per-token cosine matching with softmax — that is more sophisticated than our single-layer SVD projection. Arrow's effectiveness may come from the aggregation across layers, not from any single projection's discriminative power.

2. **MBC weight clustering works for library building:** The same paper shows that adapter parameter similarity successfully clusters related tasks. **Resolution:** This is an offline meta-learning step over many adapters, not per-token routing. Weight similarity captures structural relationships between tasks (which tasks benefit from shared parameters) but not input-conditional routing signal.

3. **Higher rank could help:** At higher adapter ranks (r=64, r=128), the r/d ratio increases from 0.006 to 0.025-0.05, potentially providing enough subspace volume for discriminative projections. We only tested r=16. **Caveat:** Higher ranks defeat the efficiency goals of LoRA at our target platform (M5 Pro 48GB).

## Alternative Approaches (What We Could Try Instead)

1. **Contrastive centroid training (ArcFace/InfoNCE):** Train a small projection head that pushes domain centroids apart using margin-based losses, combining the zero-overhead of centroid routing with learned discrimination. Cost: ~5K extra params, single training pass. This bridges the centroid-to-Gumbel gap without a full router.

2. **SpectR-style full-spectrum routing (arxiv 2504.03454):** Instead of projecting onto a single SVD vector, use the full LoRA covariance matrix for routing. Avoids the information loss of our Method C while remaining training-free.

3. **Vector database retrieval (arxiv 2602.21222):** Treat adapter selection as information retrieval — embed the input, retrieve top-k adapters from a vector store. Task-Aware LoRA Adapter Composition via Similarity Retrieval proposes exactly this. Scales to N=1000+ adapters without concentration issues.

4. **Loss-probing / TTT:** For high-stakes routing, run a quick forward pass through candidate adapters and select by minimum loss. Expensive (N forward passes) but provides ground-truth routing signal. Our TTT expert selection experiment (P1 priority) explores this direction.

5. **Centroid-initialized Gumbel:** Use centroid routing as warm-start initialization for the Gumbel-sigmoid router (reducing the 6000-step training budget). The centroid already achieves 87% — a warm-started router might reach 90%+ in 1000-2000 steps instead of 6000.

6. **Hybrid routing (LoRA-Mixer, arxiv 2507.00029):** Serial attention routing that coordinates multiple LoRA experts, potentially combining centroid pre-selection with learned fine-grained mixing.

## Implications for Next Experiments

1. **Centroid routing is our production fallback.** At 87% top-2 with zero training cost, it's the default routing method for any new adapter deployment until a trained router is available. Store 500KB of centroids alongside each adapter library.

2. **B-matrix routing is permanently closed.** At r/d=0.006, no amount of cleverness in projection methods will recover discriminative signal from adapter weights. This is a geometric constraint, not an implementation failure. The only escape is higher rank (which we don't want) or contrastive training of the B-matrices themselves (which changes the optimization objective).

3. **Router training budget is the real bottleneck.** The Gumbel-sigmoid ablation showed 3000→6000 steps = +5.3pp. The centroid baseline tells us where learned routing starts: 87%. The question for production is whether the marginal 3.27pp gain justifies the training compute.

4. **Centroid-warm-started routing is the natural next step.** Initialize Gumbel-sigmoid router from centroid similarities, then fine-tune. This could dramatically reduce training steps to reach 90%+ accuracy.

5. **Per-token routing is the untested risk.** All our results are sequence-level (mean-pooled hidden states). Per-token routing — needed for MoLoRA-style composition — may not cluster as cleanly. This should be tested before committing to centroid-based production routing.

6. **L2R's insight applies to our architecture.** Routing in a low-rank *latent* space (not the raw d=2560 space, and not the adapter's r=16 subspace) may be the sweet spot. L2R's shared latent routing space is conceptually similar to our Gumbel-sigmoid router but with explicit geometric control.
