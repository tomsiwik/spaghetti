# Peer Review: Semantic Router

## NotebookLM Findings

Skipped. The experiment self-killed on K1 with a thorough information-theoretic explanation. The analysis is transparent and the conclusions are well-supported. Deep review is not needed for a clean kill with honest diagnostics.

## Mathematical Soundness

**Section 4.1 (Stationary distribution sensitivity bound):** The bound `||pi_1 - pi_2|| <= epsilon / (1 - lambda_2)` is a standard result from Markov chain perturbation theory (e.g., Cho and Meyer 2001). The application here is correct: with noise_scale=0.15 and domain_bias scaled by 0.5, the effective perturbation in transition probabilities is O(0.075), and the stationary distribution difference is further compressed by the spectral gap. The estimate of 0.5-1.3 bits of distinguishing information vs the 2.32 bits needed is directionally sound, though not rigorously derived (it relies on intuitive entropy estimates rather than a formal mutual information calculation). This is acceptable for a micro experiment.

**Section 4.2 (Information bottleneck decomposition):** The decomposition `acc_domain ~ acc_cluster * acc_within_cluster` assumes conditional independence of cluster-level and within-cluster routing errors, which is approximately valid when the cluster routing is near-perfect (95-97%). The predicted ~24% matches the observed ~25% well. This is the strongest part of the math -- it explains why the kill criterion was never achievable at this scale with this data.

**JL lemma application (Section 2.2):** The random projection from D_raw=224 to D=64 is a ~3.5x reduction. The JL distortion bound for preserving pairwise distances within (1 +/- epsilon) requires D >= O(log(n)/epsilon^2). With n=4500 data points (15 domains * 300 train), D=64 gives epsilon ~ 0.4 -- substantial distortion. This is not flagged in the paper. However, since ALL routers (including keyword frequency which operates in the original 32-dim space) hit the same ceiling, the projection distortion is not the bottleneck. The information bottleneck is upstream in the data generation, not the embedding.

**LSH collision probability:** Correctly stated. The SimHash guarantee `Pr[match] = 1 - theta/pi` is textbook.

**Latency model (Section 5):** Correct asymptotic complexities. The production-scale extrapolations (cosine ~10us, utterance ~500us at N=500) are reasonable order-of-magnitude estimates.

**Minor issue:** The MATH.md states the projection matrix has "i.i.d. N(0, 1/D) entries (column-normalized)" but the code generates N(0, 1) entries and then column-normalizes. After normalization the initial variance is irrelevant, so this is a documentation inconsistency, not a bug.

## Novelty Assessment

**Prior art:** The semantic-router library (aurelio-labs) is correctly cited and the utterance matching strategy directly adapts its pattern. The experiment adds cosine centroid matching and LSH as additional baselines, which is a reasonable extension. No novelty is claimed beyond the controlled comparison.

**Delta over content_aware_routing:** The predecessor experiment tested 4 strategies (hash ring, keyword, cosine, MLP classifier) with unigram-only features. This experiment adds: (1) bigram+trigram features, (2) LSH routing, (3) utterance matching (1-NN and aggregated). The key new finding is that richer features (224-dim n-grams vs 32-dim unigrams) provided negligible improvement (27.3% vs 26.5%), confirming the bottleneck is in the data, not the features.

**Reinvention check:** The semantic-router reference is in `references/semantic-router/`. The researcher correctly adapted this for the utterance matching strategy rather than reinventing it.

## Experimental Design

**Does it test the stated hypothesis?** Yes. The hypothesis is that a semantic router can achieve >70% domain accuracy. Six strategies were tested across 3 seeds. The strongest achieved 27.3%, decisively failing K1.

**Could a simpler mechanism explain the results?** The information bottleneck analysis already provides this simpler explanation: within-cluster domains are too similar in their observable statistics (stationary distribution of Markov chains) to discriminate with ANY frequency-based method. This is a data limitation, not a routing limitation.

**Controls:**
- Hash ring as content-agnostic baseline: appropriate.
- Oracle as upper bound: appropriate.
- 3 seeds with std reporting: adequate for a kill decision.
- Train/test split (300/100 per domain): appropriate.

**One design concern:** The keyword frequency router operates on raw unigram features (32-dim) while cosine/LSH/utterance routers operate on projected n-gram features (64-dim). This means keyword and cosine are not a clean apples-to-apples comparison of "unigram vs n-gram" -- they differ in both feature space AND similarity metric. However, since keyword still wins on domain accuracy (27.3% vs 25.4% cosine), and the paper's conclusion is about an information ceiling rather than feature superiority, this does not undermine the kill verdict.

**Missing control:** No ablation isolating bigram/trigram contribution. The paper claims n-grams did not help (27.3% vs 26.5% prior), but this comparison is across experiments with different code, different seeds, and different router implementations. A within-experiment ablation (cosine router with unigram-only vs n-gram features) would have been cleaner. Non-blocking: the ceiling argument holds regardless.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry `exp_semantic_router` correctly records:
- Kill criteria match the paper: K1 (<70% accuracy), K2 (>5ms latency), K3 (>2% overhead)
- Status: killed
- Evidence lines accurately summarize results
- Depends_on: [] (no dependencies, appropriate for a routing-only test)
- The evidence correctly notes this confirms content_aware_routing findings

The kill is clean and well-documented.

## Integration with VISION.md

VISION.md already records "Content-aware routing killed at micro" and recommends "Hash ring + pre-merge" as the routing strategy. This experiment reinforces that decision. The hierarchical routing recommendation (cluster-level semantic + within-cluster hash ring) is noted in the PAPER.md but not yet formalized as a hypothesis in HYPOTHESES.yml. This is a minor gap -- the cluster routing result (97%) is a genuinely useful finding that could inform macro-scale architecture.

## Macro-Scale Risks (advisory)

1. **The kill may not transfer to macro.** The information bottleneck is specific to micro-scale synthetic Markov chain data with V=32 and noise_scale=0.15. Real domains (Python vs JavaScript vs medical) have dramatically richer distinguishing features. Pretrained sentence embeddings would provide orders of magnitude more discriminative power than character n-grams. The paper acknowledges this clearly in the Limitations section.

2. **Cluster routing at 97% is a real result.** At macro scale with real embeddings, hierarchical routing (semantic cluster selection + hash ring within cluster) is worth testing. The cluster routing signal appears robust across all strategies.

3. **Utterance matching scales as O(NK*D).** At N=500, K=50, D=4096, this is 100M ops per query (~500us). FAISS or other ANN methods would be needed. The paper correctly flags this.

## Verdict

**KILL** -- confirmed. Accept the experiment's own kill verdict.

The experiment cleanly fails K1 (best domain accuracy 27.3% vs 70% threshold). The information-theoretic analysis in MATH.md Section 4 provides a satisfying explanation: within-cluster domain discrimination requires 2.32 bits but only ~1 bit is available from the stationary distribution differences at noise_scale=0.15. This is a data limitation, not a mechanism failure.

**What the experiment proved despite the kill:**
1. Cluster routing is trivially solved (82-97%) across all semantic methods
2. All routing strategies are comfortably within production latency budgets (<6us)
3. Richer features (n-grams) do not overcome information-limited data
4. Hierarchical routing (semantic cluster + hash within) is the right architecture for production

**No revisions needed.** The math is sound, the experimental design is adequate, the kill is honest and well-explained, and the positive findings (cluster routing, latency) are correctly extracted. The HYPOTHESES.yml status is already set to "killed." This experiment can be archived as-is.
