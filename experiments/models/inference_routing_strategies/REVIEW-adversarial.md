# Peer Review: Inference Routing Strategies

## NotebookLM Findings

Skipped. The experiment was self-killed by the researcher on K3 (quality capture 41.5% vs 90% threshold). The review below confirms the kill is justified and identifies additional issues.

## Mathematical Soundness

**MATH.md quality model (Section 1):** The additive quality decomposition is internally consistent. The quality matrix construction (base + home domain bonus + within-cluster bonus) correctly models a tiered specialization structure. No errors in the indicator-function formulation.

**Complexity analysis (Section 2):** All O(.) claims are correct. Hash ring is O(log(N*V)), embedding similarity is O(N*E), hierarchical is O(C*E + log(N*V/C)). The classifier scaling claim of "O(D*H + H*N)" in PAPER.md uses D for embedding dimension (inconsistent with MATH.md which uses E). This is a notation inconsistency, not a mathematical error.

**Quality capture bound (Section 3.4):** The bound quality_capture <= (s * domain_accuracy + q_base_mean) / (s + q_base_max) is correct for N/D = 1. The extension "with N/D > 1, the bound tightens to quality_capture <= 0.90 * D/N" is a rough approximation, not a derivation. It assumes that within a domain, the router randomly selects among N/D experts, which only holds for embedding-based routers that cannot distinguish same-domain experts. This is stated but not proven. **Non-blocking.**

**Hidden assumption: quality = Q[d, selected_expert].** The quality measurement assumes the router's single expert selection determines quality. For pre-merge, the paper uses Q[d, :].mean(). This is correct for uniform averaging but ignores that real pre-merge quality is determined by the merged model's actual loss, not the arithmetic mean of per-expert quality scores. The synthetic quality matrix is a proxy for per-expert contribution to merged loss, and this proxy assumes linearity of composition. The paper lists this as Assumption 1 (Section 6), which is appropriate.

**Worked example (Section 5):** The embedding similarity calculation claims "expected quality ~ (0.95 + 0.36) / 2 = 0.655" but then states "domain accuracy is ~27%". These two statements are disconnected. With 27% domain accuracy, the expected quality from embedding similarity would be closer to 0.27 * 0.655 + 0.73 * Q_random, not 68.9% quality capture. The worked example conflates the "if domain routing is perfect" scenario with the actual measured scenario. **Non-blocking but misleading.**

## Novelty Assessment

**Prior art overlap is moderate.** The experiment cites Switch Transformers, Mixtral, DeepSeek-V3, Soft MoE, and semantic-router. These are all relevant. However, the experiment does not test any of the routing mechanisms these papers actually propose (learned linear router, top-k gating, soft assignment). Instead, it tests a set of simpler strategies (hash ring, cosine similarity, 2-layer MLP, hierarchical). This is a deliberate choice given the micro constraints, and it is acceptable.

**The key insight -- that routing latency is negligible and quality differentiation is minimal at micro scale -- is consistent with the three prior killed experiments** (content_aware_routing, semantic_router, pre-merge vs dynamic). This experiment adds the Pareto analysis framing and the quality sweep, which are useful consolidation work rather than novel findings.

**Missing comparison:** The MoRAM (associative memory routing) reference in REFERENCES.yml was not considered. MoRAM proposes self-routing where experts intrinsically contain routing information, which is conceptually different from all five tested strategies. Not blocking for a killed experiment.

## Experimental Design

**The experiment tests what it claims.** Five strategies are compared on latency and quality capture against an oracle. The methodology is clean: synthetic quality matrices with tunable specialization, multiple seeds, and a comprehensive sweep over specialization strength and expert-to-domain ratio.

**Critical design issue: the K3 criterion is fundamentally unachievable.** The paper acknowledges this in Section 3.4 of MATH.md, but the kill criterion was set at >90% oracle quality capture before this limitation was understood. The bound shows 90% is only achievable with perfect domain routing at N/D=1. Since the experiment tests N/D up to 10:1, the 90% threshold is mathematically impossible for most configurations. This means the kill criterion was poorly calibrated -- it was guaranteed to kill the experiment from the start.

The researcher correctly identifies this: "the kill criterion is unfairly strict." This is a legitimate observation. A better K3 would have been: "best strategy achieves >90% of the quality capture that is theoretically achievable given the N/D ratio." This would be: quality_capture_observed / (0.90 * D/N) > 0.90, or equivalently quality_capture > 0.81 * D/N. At N=5, D=5 (1:1): threshold 0.81. At N=30, D=15 (2:1): threshold 0.405. The best result at N=5 is 0.415, which still falls below the corrected 1:1 threshold of 0.81.

**The conclusion is the same either way: routing quality is poor.** But the kill should be attributed to "embedding-based routing cannot distinguish experts within a domain" rather than "no strategy reaches 90%."

**Classifier training uses oracle labels.** The tiny classifier is trained on (embedding, best_expert) pairs where best_expert comes from the oracle. This gives the classifier an unfair advantage: it has access to oracle-quality supervision that would not exist in production. Despite this advantage, it still only achieves 41.5% quality capture. This actually strengthens the kill -- even with oracle supervision, routing fails.

**No top-k (k>1) routing tested.** All strategies select a single expert. The SOLE architecture (VISION.md) mentions top-k=2 selection. Averaging the top-2 experts by cosine similarity could improve quality capture. This is a missing configuration, but unlikely to change the verdict given the magnitude of the gap.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry for `exp_inference_routing_strategies` lists three kill criteria that match the paper's K1, K2, K3 exactly. K1 and K2 pass, K3 kills. The evidence entries are complete and accurately reflect the paper's findings. Status is correctly marked as `killed`.

The experiment depends on `exp_inference_latency_vs_N` (proven), which is appropriate. It blocks nothing, which is also correct since the SOLE architecture already committed to hash ring + pre-merge before this experiment ran.

## Macro-Scale Risks (advisory)

1. **Pre-merge dilution at large N is the real question.** This experiment's most important finding is that pre-merge quality is indistinguishable from hash ring quality (both ~23% capture). But this is in a synthetic model. At macro scale with N=500 pre-merged experts, each contributing 0.2% of the delta, the question is whether the merged model retains any expert specialization at all. If it does not, then the Distill phase is wasted for pre-merge. This is acknowledged in the paper's "What Would Kill This" section.

2. **Real expert quality profiles may be very different.** The distillation pilot shows 98% win rate with 42.2% PPL improvement. This means real experts are much more specialized than the synthetic model (s=0.8 over a 0-0.2 background). With stronger specialization, routing quality lift over pre-merge would increase. The macro experiments should re-test routing with real expert quality profiles.

3. **Sentence-level embeddings would dramatically change routing quality.** The experiment uses synthetic Gaussian embeddings at E=64. Production would use sentence-transformers (E=768 or E=1024) with domain-discriminative pretraining. The domain classification accuracy, which limits quality capture, could be much higher with real embeddings. The experiment's directional finding (routing quality is limited) may not transfer to macro.

## Verdict

**KILL (confirmed)**

The researcher's self-kill is justified. The experiment is well-executed and the findings are sound within its scope. The key contributions are:

1. Routing latency is definitively negligible (<5us at N=100) for all tested strategies.
2. No embedding-based routing strategy achieves meaningful quality lift over pre-merge.
3. The mathematical bound on quality capture explains why: embeddings cannot distinguish same-domain experts.

The kill criterion K3 was poorly calibrated (mathematically unachievable for most configurations), but even under a corrected criterion, the results would still kill. The quality lift from routing (best: 9.1% of achievable gap) is too small to justify any routing complexity over pre-merge.

No revision is warranted. The experiment has served its purpose: it confirms hash ring + pre-merge as the correct SOLE production strategy at current scale, while leaving the door open for re-evaluation at macro scale with real expert profiles and pretrained embeddings.
