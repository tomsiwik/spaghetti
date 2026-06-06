# Peer Review: Routing at Scale

## NotebookLM Findings
Skipped -- the experiment is straightforward engineering measurement (latency benchmarking), not a novel mathematical mechanism. NotebookLM deep review would add little value here.

## Mathematical Soundness

**Latency analysis: sound.** The complexity classes (O(1), O(log N), O(N*E), O(E*H+H*N)) are correctly stated for each strategy. The power-law fit T = a * N^b across 3 data points is a reasonable empirical summary, though fitting a 2-parameter model to 3 points leaves zero degrees of freedom for goodness-of-fit assessment. The measured exponents are directionally consistent with theory: hash ring 0.08 (expected ~0 from log factor), embedding sim 0.289 (expected 1.0, but overhead-dominated at small N*E), classifier 0.63 (expected ~1.0, clamped hidden dim). The sub-theoretical exponents are correctly attributed to constant overhead domination at these N values.

**Quality analysis: sound but tautological.** The quality capture derivation in MATH.md Section 3.1 is correct: E[Q_premerge] = 0.1 + (0.8 + (D_c-1)*0.24)/N converges to 0.1 as N grows, giving ~10% quality capture against oracle. The measured values (16-21%) are higher because the actual oracle mean is <1.0 (finite sampling), which the paper correctly notes. However, this entire quality analysis is measuring properties of the synthetic data generation, not of the routing algorithms. The quality matrix is designed so that pre-merge quality dilutes as 1/N -- that is a property of the experiment setup, not a finding.

**Scaling projections: fragile.** Projecting from 3 points (N=100, 500, 1000) to N=10,000 using a power law is acceptable for order-of-magnitude estimates but the paper presents specific numbers (e.g., "hash ring 2us at N=10,000") with false precision. The projection is safe directionally (all strategies remain sub-millisecond) but the specific microsecond estimates should not be relied upon.

**Domain accuracy math error (minor).** MATH.md Section 3.2 states random domain accuracy = D/N = 10% because N/D = 10 experts share each domain. This is correct for the "hits home domain" metric, not the "hits oracle expert" metric. The paper conflates these in the text but the code correctly implements the home-domain metric (line 454: `expert_home = selected % Q.shape[0]`). The `% Q.shape[0]` implementation is correct only because expert assignment is round-robin (`expert_to_domain[i] = i % n_domains`). This is fragile -- it would break with any other assignment scheme.

## Novelty Assessment

**Low novelty, but that is acceptable.** This is an engineering validation experiment, not a mechanism proposal. The experiment confirms what the complexity analysis already predicts: O(log N) routing is fast at N=1000. The value is in the empirical confirmation with specific microsecond numbers and the introduction of FAISS ANN as a comparison point.

**Prior art is well-cited.** Switch Transformers, Mixtral, DeepSeek-V3, FAISS, and consistent hashing are all referenced. The experiment does not claim novelty in routing algorithms, only in measuring their scaling properties for the SOLE use case.

**No reinvention detected.** The experiment uses faiss-cpu directly rather than reimplementing ANN search.

## Experimental Design

**Strength: good controls.** Testing 6 strategies at 3 scale points with 3 seeds and 2000+ queries per seed is solid methodology for a latency benchmark. The warmup phase (200 iterations) and high iteration count (5000) for timing are appropriate.

**Strength: honest K2 assessment.** The paper correctly identifies that K2 (routing accuracy >50%) is structurally inapplicable to SOLE's pre-merge architecture. Rather than hiding this, it explains why the criterion mismatch exists and points to the correct experiment (exp_cross_domain_dilution_vs_k, r=0.990) that validates quality selection for SOLE.

**Weakness 1: synthetic embeddings are not embeddings.** The "embeddings" are Gaussian noise around cluster centroids at E=64. Real sentence embeddings from models like all-MiniLM-L6 (E=384) or BGE (E=768) have very different distance distributions -- they tend to have high baseline cosine similarity (0.3-0.6) with domain-specific variation on top. The Gaussian synthetic embeddings have near-zero baseline similarity. This could mean that in production, embedding-based routing has either better or worse domain separation than measured here. The paper acknowledges this in Limitations but does not quantify the impact. This is acceptable at micro scale.

**Weakness 2: Apple Silicon CPU timing.** The latencies are measured on Apple Silicon, which has different cache hierarchy, memory bandwidth, and branch prediction characteristics than production servers (typically x86 with Intel/AMD). The absolute microsecond numbers are not transferable. The relative rankings and scaling exponents are likely stable across platforms. The paper should state this more explicitly.

**Weakness 3: hash ring routes by query index, not content.** The hash ring router hashes `f"query_{query_idx}"` (line 168), which means it routes based on query position in the test set, not query content. In production, the hash key would be derived from the query text or embedding. This does not affect latency measurement (MD5 hash + bisect is the same cost regardless of input), but it means the quality/domain-accuracy numbers for hash ring are effectively random assignment -- which is the correct interpretation for a content-unaware router, so this is defensible.

**Weakness 4: classifier is untrained at scale.** At N=1000, the classifier has a 128x1000 output layer (128K parameters) trained on 3000 examples for 30 epochs. With a 1000-class problem and 3 examples per class on average, this classifier cannot possibly learn meaningful routing. The domain accuracy of 1.5% (below 10% random baseline) confirms this. The classifier latency numbers are valid, but comparing its "quality" to other strategies is misleading. The paper should note that the classifier is capacity-starved at N=1000, not that classifier routing fundamentally fails.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry marks this as "proven" with status based on K1 PASS, K3 PASS, and K2 as criterion mismatch. This is reasonable. The kill criteria in HYPOTHESES.yml match the code exactly.

The K2 criterion ("routing accuracy drops below 50% at N=500") was poorly specified from the start -- it assumes selective routing, which SOLE does not use. The paper's recommendation to mark K2 as NOT APPLICABLE rather than KILL is the correct interpretation. The experiment should update its status language from "K2 KILL (criterion mismatch)" to "K2 N/A (criterion assumes selective routing; SOLE uses pre-merge)" for clarity.

## Macro-Scale Risks (advisory)

1. **E=384-4096 changes the picture.** At production embedding dimensions, brute-force cosine becomes 6-64x slower. FAISS ANN would likely show real advantage over brute-force at E>=384 and N>=1000. The experiment correctly identifies this as a future test.

2. **Batch routing is the real production scenario.** Single-query latency is measured but production inference uses batched requests. GPU-based batch routing (e.g., batched matrix multiply for embedding similarity) would have completely different scaling. This is the most important unmeasured dimension.

3. **Memory footprint at scale.** Hash ring with 150 virtual nodes at N=10,000 = 1.5M entries. FAISS IVF index at N=10,000 with E=768 = ~30MB. Neither is concerning, but should be measured at macro.

## Verdict

**PROCEED**

This is a clean engineering validation experiment that definitively answers its core question: routing latency is not a scaling concern for SOLE at any foreseeable expert count. The methodology is sound, the controls are adequate, and the paper honestly handles the K2 criterion mismatch rather than hiding it.

Minor issues that do not block proceeding:

1. The K2 status language should be clarified from "KILL (criterion mismatch)" to "N/A (criterion assumes selective routing)" in HYPOTHESES.yml to avoid confusion in future reviews.
2. The scaling projections to N=10,000 should be described as order-of-magnitude estimates, not precise predictions.
3. The classifier quality comparison at N=1000 is misleading due to capacity starvation (3 examples per class); a note acknowledging this would strengthen the paper.
4. Future macro validation should prioritize batch routing on GPU and E>=384 embedding dimensions, as these represent the real production scenario that micro cannot test.
