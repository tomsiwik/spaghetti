# LEARNINGS: Sentence-Embedding Routing at N=24

## Core Finding

Off-the-shelf sentence embeddings (MiniLM-L6-v2) catastrophically fail for centroid routing at N=24 (33.3% accuracy vs 96% at N=5) because transformer sentence embeddings are **anisotropic** — baseline cosine similarity ~0.8 regardless of content — causing the minimum per-domain margin, not the average Fisher ratio, to govern routing accuracy. This is the seventh routing method killed at N=24, confirming the disease is **feature space non-separability**, not the routing algorithm.

## Why This Happened

### 1. Hubness and Anisotropy in Sentence Embeddings

Radovanovic et al. (2010) identified the **hubness phenomenon**: in high-dimensional spaces, some points become "universal nearest neighbors" as dimensionality grows. Mu & Viswanath (2018, arXiv:1702.01417, "All-but-the-Top") showed that transformer embeddings are dominated by a few principal components, creating an anisotropic distribution where most vector pairs have cosine similarity ~0.7-0.8. Our measured mean inter-centroid cosine of 0.798 matches this prediction exactly.

The proof assumed isotropic Gaussian concentration around centroids (standard for random vectors via JL lemma), but MiniLM-L6-v2 is a **trained** model that compresses semantic content into a narrow cone. At N=5, the 5 chosen domains happened to lie in different sub-cones. At N=24, many domains (cooking, sports, music, creative_writing) share sub-cones, making centroids indistinguishable.

### 2. Average vs Minimum Margin — The Real Predictor

Fisher discriminant ratio R=2.93 (within predicted 2.0-4.0) measures **average** separability. But nearest-centroid routing errors are governed by the **worst-case margin** per domain. The bimodal pattern is sharp:
- 7 domains with margin > 0.10: all achieve ≥80% accuracy
- 6 domains with margin < 0.05: all achieve ≤10% accuracy
- 91 confused centroid pairs (vs 3-6 predicted)

This is a well-known gap in the classification theory literature (Cover & Hart, 1967): average Fisher ratio is necessary but not sufficient; minimum margin governs error rate.

### 3. Seven Methods, One Root Cause

All N=24 routing methods fail in the same accuracy band (28-40%):
- Hidden-state mean pooling: 32.5% (Finding #192-193)
- TF-IDF centroids: 35% (Finding #247)
- Binary routing heads: 39.6% (Finding #190-191)
- Softmax multi-class: 39.4% (Finding #192-193)
- Energy-gap argmin: 8.3% (Finding #189)
- Embedding-layer: 25-28%
- **Sentence embeddings: 33.3% (Finding #257)**

The convergence around 30-40% across radically different feature extraction methods is itself diagnostic: the ceiling is not in the routing algorithm but in the **adapter specialization** at N=24. Oracle PPL (19.16) is *worse* than base PPL (18.97) on average — the adapters themselves provide no meaningful signal to route on.

## Confirming Evidence

- **Radovanovic et al. (2010)** — Hubness in high-dimensional spaces. Nearest-neighbor hubs emerge when dimensionality grows, making centroid-based methods unreliable. Directly explains the failure mechanism.
- **Mu & Viswanath (2018, arXiv:1702.01417)** — "All-but-the-Top": sentence embeddings occupy a narrow cone; removing top principal components restores isotropy. Whitening would increase inter-centroid margins.
- **Timkey & van Schijndel (2021, arXiv:2109.04404)** — "All Bark and No Bite": embedding isotropy alone doesn't predict downstream task performance. Even if whitening increases margins, routing accuracy may not follow.
- **Ethayarajh (2019, arXiv:1909.00512)** — Showed contextual word representations become MORE anisotropic in higher layers of BERT/GPT-2. Our use of last-layer hidden states for sentence embeddings maximizes the anisotropy problem.
- **Finding #192-193** (this project) — Centralized softmax routing matches binary heads at N=24 (39.4% vs 39.6%), confirming the bottleneck is feature quality, not architecture.

## Contradicting Evidence

- **LoRAuter (arXiv:2601.21795)** — Claims to route across 1500+ adapters using sentence-embedding retrieval with 101.2% of oracle quality. Key difference: they use **trained** task embeddings from adapter metadata, not off-the-shelf sentence embeddings from query text. The routing is over task *descriptions*, not query *content*.
- **Task-Aware LoRA Composition (arXiv:2602.21222)** — Routes across 22 datasets using frozen all-MiniLM embeddings. Possible explanation for their success: 22 datasets spanning 6 task *families* (classification, QA, summarization, etc.) are more separable than 24 general-knowledge *domains* (economics, history, sociology all look similar to MiniLM).
- **Finding #255** (this project) — Sentence embeddings achieved 96% at N=5. Not contradicting per se, but shows the method *does* work when domains are well-separated. The failure is scale-dependent.

## Alternative Approaches

### 1. Contrastive Fine-Tuning of Routing Embeddings
**LoraRetriever (arXiv:2402.09997):** Trains a contrastive retriever on (query, correct-adapter) pairs. Creates adapter-specific embeddings where inter-adapter distances are maximized. At N=24, this would learn to push cooking away from sports in embedding space, directly fixing the margin collapse. **Directly addresses the root cause** (anisotropy in general-purpose embeddings).

### 2. Whitening / Isotropy Enforcement
**Mu & Viswanath (2018, arXiv:1702.01417):** Remove top-k principal components from sentence embeddings before computing centroids. Computationally free (single SVD), but Timkey & van Schijndel (2021) caution that isotropy alone doesn't guarantee task performance. Worth testing as a zero-cost preprocessing step.

### 3. Hierarchical Routing
**Already tested (micro/models/hierarchical_routing_n24/):** Cluster selection works (97.3%) but within-cluster routing collapses (41.5%). Root cause is identical: adapters lack specialization, so within-cluster discrimination is impossible regardless of routing method.

### 4. Per-Token Routing (Bypass Query-Level Routing)
**MoLoRA (arXiv:2603.15965):** Routes at per-token granularity, avoiding the query-level classification bottleneck entirely. Qwen3-1.7B + 4 adapters > Qwen3-8B. This sidesteps the N=24 centroid problem by never needing to classify the query into a single domain.

### 5. Adapter Quality First
**LIMA (arXiv:2305.11206):** SFT teaches format alignment, not domain knowledge. Our N=24 adapters may not have learned meaningful specialization (oracle PPL ≈ base PPL). Before improving routing, verify adapters are worth routing to. The N=5 adapters achieved 26.5% PPL improvement; N=24 adapters achieve ~0%.

## Implications for Next Experiments

1. **The routing problem at N=24 may be unsolvable with the current adapters.** Oracle PPL being worse than base PPL means even perfect routing yields no benefit. The priority should shift from "better routing" to "better adapters" — or accept that 5-8 well-specialized domains is the practical operating point.

2. **The margin > 0.10 threshold** is an actionable design rule: for any embedding space, precompute pairwise centroid margins and only attempt routing when all margins exceed 0.10. Domains below this threshold should be merged or their adapters retrained with harder negatives.

3. **The ten-level proxy chain (Findings #236-257)** is now complete. Every proxy metric tested — PPL, MMLU, cosine, Fisher ratio, TF-IDF, hidden states, sentence embeddings — fails to predict the next level. The meta-lesson: only behavioral evaluation on actual generation tasks is reliable.

4. **Contrastive fine-tuning is the principled next step** if routing at N>8 is required — it directly addresses the root cause (anisotropy) rather than switching algorithms. But the adapter quality question (oracle ≈ base at N=24) should be resolved first.

## Recommended Follow-Up

### Option A: Contrastive Routing at N=24 (if adapter quality confirmed)
- **Motivation:** Finding #257 identifies anisotropy as root cause; LoraRetriever (arXiv:2402.09997) shows contrastive training fixes exactly this.
- **Prerequisite:** Verify N=24 adapters have meaningful specialization (oracle PPL << base PPL). If not, retrain adapters first.
- **Would fix:** The specific margin collapse observed (91 confused pairs → target <10).

### Option B: Accept N=5-8 as Operating Point, Focus on P0 Track
- **Motivation:** Seven methods killed at N=24; N=5 routing solved at 96% (Finding #255). The P0 deployment track needs generation quality proof, not more routing experiments.
- **Literature:** MoLoRA (arXiv:2603.15965) shows 4 adapters suffice for substantial gains (1.7B+4 > 8B).
- **Would advance:** The existential question — does routed composition produce useful text?

### Option C: Whitening as Zero-Cost Preprocessing
- **Motivation:** Mu & Viswanath (2018) — removing top-3 PCs restores isotropy. Zero implementation cost.
- **Caveat:** Timkey & van Schijndel (2021) — isotropy ≠ task performance. May increase margins without improving routing.

## References Added
- Mu & Viswanath (2018, arXiv:1702.01417) — "All-but-the-Top": isotropy via PCA whitening
- Ethayarajh (2019, arXiv:1909.00512) — Layer-wise anisotropy in contextual representations
- Timkey & van Schijndel (2021, arXiv:2109.04404) — Isotropy alone insufficient for downstream tasks
- LoraRetriever (arXiv:2402.09997) — Contrastive retriever training for adapter selection
