# Learnings: exp_gumbel_sigmoid_ablation

## Core Finding

Training length (3000 to 6000 steps) is the dominant driver of 49-expert routing accuracy (+5.3pp top-2), not auxiliary L1 gate regularization. L1's only measurable value is recovering the single hardest zero-accuracy domain (wikitext, cos=0.996 with history). The experiment was killed (K1 FAIL: 4.49% vs 5% threshold against canonical 86.33% baseline).

## Why This Happened (Literature-Grounded)

Three mechanisms explain the training-length dominance:

**1. Expert divergence requires sufficient gradient signal.** With 49 experts, batch_size=1, and 3000 steps, each domain sees ~61 gradient updates. At 6000 steps, ~122 updates. The router receives noisy gradient signals when experts haven't diverged sufficiently in their hidden state representations. Switch Transformers (Fedus et al., 2022, arXiv 2101.03961) notes that router training stability is coupled to expert specialization -- the router can only learn sharp boundaries after experts have differentiated.

**2. L1 regularization is not load-balancing.** The auxiliary loss (L_aux = alpha * sum_i mean_batch(g_i)) is L1 gate activation penalty, not the Switch Transformer f_i * p_i capacity fraction loss. L1 penalizes total gate mass, pushing non-target gates toward zero. This is structurally different from load-balancing: it doesn't enforce uniform utilization, it enforces sparsity. At 3000 steps, this sparsity is premature -- it suppresses exploration before the router has learned which gates should be active. At 6000 steps, the effect is marginal because the router has already learned sharp boundaries.

**3. Temperature catastrophe at tau=0.1 is a known Gumbel pathology.** The Gumbel noise term g = -log(-log(U)) has support on (-inf, +inf) with typical values reaching 5-10. At tau=0.1, effective noise magnitude is 50-100x the logit scale, completely swamping learned routing preferences. Jang et al. (2017, arXiv 1611.01144) recommend tau >= 0.5 for stable training and annealing from high to low temperature.

## Confirming Evidence

**Training budget scaling with expert count.** Our own scaling trajectory shows routers are consistently undertrained relative to expert count. At N=5, routers converge quickly (400 steps per expert, ~2000 total). At N=49, the per-expert budget of ~61 steps at 3000 total is 6.5x less per expert. The 6000-step control's +5.3pp confirms the deficit. This is consistent with MoLoRA (arXiv 2603.15965) which uses longer router training for larger expert pools.

**Over-regularization hurts at small scale.** The L1 alpha=0.5 config achieves only 52.45% accuracy (17 zero-acc domains), showing that strong regularization is catastrophic. Even alpha=0.1 at 3000 steps hurts (-1.22pp). This confirms the finding from "Towards Modular LLMs by Building and Reusing a Library of LoRAs" (MBC framework) that router regularization must be calibrated to the training budget -- premature sparsity prevents exploration.

**Temperature robustness in [0.5, 2.0].** Fixed tau=2.0 (85.71%), tau=1.0 (84.90%), tau=0.5 (84.49%) are all within 1.2pp of the annealed baseline (85.10%). This is consistent with the Gumbel-softmax literature: moderate temperatures produce similar gradient quality because the sigmoid saturates above tau~0.3, making the exact value non-critical.

## Contradicting Evidence

**Load-balancing IS essential for dead expert prevention.** Our result that "LB hurts at 3000 steps" contradicts the Switch Transformer finding that auxiliary losses are critical to prevent expert collapse. The resolution: our experts are pre-trained LoRA adapters with fixed weights -- only the router trains. There are no "dead experts" because expert quality is determined at LoRA training time, not router training time. This is fundamentally different from standard MoE where experts co-train with the router. In our setting, L1's role is narrower: suppressing off-target gate activations, not preventing collapse.

**L1 recovers wikitext (cos=0.996 with history).** While LB's aggregate effect is negligible-to-negative, it uniquely recovers wikitext from 0% to 40% top-2 accuracy. This specific value cannot be achieved by training length alone (6000 steps no-LB: still 0% wikitext). For domains with near-identical centroids (cos > 0.99), L1 suppression of the confusing neighbor's gate is the only mechanism that works short of architectural changes.

**Softmax (competing) gates show surprising viability.** Softmax anneal 2->0.5 achieves 86.12% with only 1 zero-acc domain, compared to sigmoid's 85.10% with 2. The L2R framework (Learning to Route) argues sigmoid is superior because adapter gates are independent Bernoulli variables. But our result shows the competition inherent in softmax can actually help -- it prevents the "many weakly active gates" failure mode that sigmoid allows.

## Alternative Approaches (What We Could Try Instead)

**1. Contrastive router training.** Apply InfoNCE or triplet loss to the router's latent space, forcing embeddings of different domains apart. This directly attacks the root cause of zero-accuracy domains: high cosine similarity between centroids (chemistry/science_qa: 0.992, wikitext/history: 0.996). ArcFace-style margin losses from face recognition enforce minimum angular separation between class boundaries.

**2. Attention-weighted pooling instead of mean-pooling.** Dialogue fails (4.375 variance, 13x typical) because mean-pooling averages over heterogeneous token distributions. A learnable attention head could weight salient tokens (topic words, style markers) while ignoring noise. CLS token routing is another option: prepend a learnable token whose self-attention state serves as the routing feature. Both are cheap (~1K params overhead).

**3. Hierarchical routing.** Coarse-grained first pass (e.g., code vs text vs science) followed by fine-grained expert selection within the cluster. This prevents high-variance domains from blurring decision boundaries of specialized domains. Our hidden state similarity analysis already provides natural clusters: code domains form a tight cluster, science domains overlap with each other, dialogue is an outlier.

**4. Prototypical networks for routing.** Maintain moving-average prototype embeddings per expert. Route by nearest-prototype distance (Euclidean). This method from few-shot classification is naturally robust to high intra-class variance because prototypes smooth over individual samples. Could replace the MLP router entirely.

**5. Model-Based Clustering (MBC) from "Towards Modular LLMs."** Cluster tasks by adapter parameter similarity rather than input embedding similarity. This sidesteps the centroid-confusion problem entirely: even if chemistry and science_qa have similar hidden states, their learned adapter parameters (B matrices) may be quite different.

**6. Simply train longer.** The most boring and most effective intervention. The 3000->6000 step improvement (+5.3pp) suggests the router is still undertrained at 6000. A 12000-step run (matching ~244 updates per domain) would be informative. This is cheap (~40s per 3000 steps on M5 Pro with cached hidden states).

## Implications for Next Experiments

1. **Default router training to 6000+ steps for N~50.** The 3000-step default was inherited from N=5 experiments where it was sufficient. For N=49, per-expert gradient budget matters more than total steps. Future experiments should scale router training budget proportionally to expert count.

2. **L1 regularization is a precision tool, not a default.** Use alpha=0.1 L1 only when zero-accuracy domain recovery is specifically needed. For aggregate accuracy optimization, longer training alone is superior and simpler.

3. **The dialogue routing problem is architectural, not parametric.** No configuration change fixes dialogue (0% across all 22 configs except softmax at 20%). Mean-pooled features fundamentally cannot represent this domain. Any future experiment addressing dialogue routing MUST change the feature extraction method (attention-weighted pooling, per-token routing, or CLS token).

4. **Contrastive router training is the highest-value next step.** The zero-accuracy domains are caused by high centroid similarity -- a problem that contrastive objectives directly address. This is a cleaner intervention than L1 regularization because it reshapes the representation space rather than post-hoc suppressing confusing activations.

5. **Temperature is a solved problem in this setting.** Any fixed value in [0.5, 2.0] or annealing from high-to-low works. Do not spend future experiment budget on temperature tuning. Avoid tau < 0.3 categorically.

6. **Re-examine softmax vs sigmoid.** The prior assumption (from L2R) that sigmoid is strictly superior for multi-adapter routing was not confirmed. Softmax achieved comparable accuracy with fewer zero-acc domains. A dedicated sigmoid vs softmax experiment at 6000 steps would be informative.
