# Learnings: exp_speculative_expert_selection

## Core Finding

Expert selection autocorrelation is real but domain-dependent and practically irrelevant for speedup. Overall hit rate 63.3% across 24 domains (K1 PASS), with strong bimodal distribution: 6 domains >80% (psychology 95.3%, finance 89.6%, medical 88.2%) and 14 domains 40-62%. The router overhead is only 0.46% of total inference, making maximum possible speedup 0.46% -- the "routing overhead" concern is definitively closed.

## Why This Happened

### Bimodal autocorrelation reflects semantic distinctiveness
The 6 high-autocorrelation domains (psychology, finance, medical, health_fitness, math, legal) have distinctive vocabulary and semantics that map consistently to specific experts. These correspond to the "singleton" or tight clusters in exp_softmax_router_scaling's confusion matrix. The 14 low-autocorrelation domains (philosophy, history, science, creative_writing, etc.) sit in the "confused cluster" where the router oscillates between semantically similar experts -- making expert selection token-dependent even within a single domain.

### Per-token routing oscillates within clusters
Even within single-domain text, the router uses 13-24 different experts per domain. This refines the exp_pointer_routing_no_merge finding that "per-sequence = per-token": that finding measured PPL equivalence, not routing consistency. The router DOES change experts frequently at the token level, but the PPL impact is benign because all selected experts are within the same semantic cluster.

### Router overhead is negligible by construction
The softmax router (2560->128->24) has only 330K parameters and executes in 0.166ms. Total generation is ~36ms/token, dominated by the weight-bandwidth of the 2B parameter base model. No routing optimization can provide meaningful speedup. This is a structural property of the MoE-LoRA architecture where the router is tiny relative to the model.

## Confirming Evidence

1. **exp_softmax_router_scaling:** Identified the semantic clustering that explains why some domains have high autocorrelation (distinctive clusters) and others don't (confused cluster). The 40% classification accuracy with oracle-matching PPL confirms within-cluster misrouting is benign -- and our per-token oscillation IS within-cluster oscillation.

2. **exp_molora_per_token_routing:** Measured 0.58% router overhead, consistent with our 0.46% measurement. Also found per-token routing equivalent to per-sequence on clean domains (-0.46% delta), which our autocorrelation data reinterprets: the equivalence is because within-cluster oscillation doesn't affect PPL, not because the router picks the same expert.

3. **exp_pointer_routing_no_merge:** Proved per-sequence routing is the correct granularity. Our data shows WHY: the router oscillates at token level but the oscillation is within semantically equivalent clusters.

## Contradicting Evidence

1. **Psychology 95.3% autocorrelation seems too high.** This might indicate the router has learned to "always pick expert X" for psychology regardless of token content, rather than genuine semantic routing. A per-token accuracy test against ground truth labels would clarify.

2. **The prior claim "per-sequence = per-token" was about PPL equivalence.** Our finding that the router uses 13-24 experts per domain at the token level appears to contradict this, but the contradiction dissolves when you realize PPL-equivalence and routing-consistency are different metrics.

## Implications for SOLE Architecture

1. **Routing overhead is a non-problem.** Do not invest in routing optimization (caching, prediction, speculation). The router is 0.46% of total inference -- essentially free. This permanently closes Track B "routing overhead" concerns.

2. **Per-sequence routing is validated from a different angle.** Token-level routing oscillates within clusters but produces identical PPL. Caching the router decision once per sequence is sufficient and correct.

3. **Domain-distinctive adapters are more "sticky" in routing.** Medical, finance, legal, math adapters get consistent routing; humanities/social science adapters oscillate. This suggests quality differences between these groups may be partly due to routing stability, not just adapter quality.

## Numbers to Remember

| Metric | Value |
|--------|-------|
| Overall autocorrelation (hit rate) | 63.3% |
| Best domain (psychology) | 95.3% |
| Worst domain (environmental) | 40.2% |
| High-autocorrelation domains (>80%) | 6/24 |
| Low-autocorrelation domains (<60%) | 14/24 |
| Router per-token cost | 0.166ms |
| Router fraction of total inference | 0.46% |
| Max possible speedup | 0.46% |
| Boundary detection rate | 91.3% |
| Mean self-transition probability | 0.559 |
| Mean transition entropy | 2.28 bits |
| Experiment runtime | 74.2s |
| Peak memory | 5.40 GB |
