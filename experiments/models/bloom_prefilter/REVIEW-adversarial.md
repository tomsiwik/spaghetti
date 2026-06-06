# Peer Review: Bloom Filter Pre-Filtering

## NotebookLM Findings

Skipped. The experiment is self-killed with thorough root cause analysis. The failure mode is fundamental (exact membership vs approximate similarity), not a tuning issue. Deep review would not surface additional concerns beyond what the researcher already identified.

## Mathematical Soundness

The math is correct throughout MATH.md.

1. **Bloom filter FPR formula** (Section 1.3): Standard result, correctly stated. The approximation via e^(-kn/m) is valid for large m.

2. **Optimal k derivation** (Section 1.4): k_opt = (m/n)ln(2) is textbook. Correctly applied.

3. **Saturation calculation** (Section 3.3): At m=256, k=4, n=20000: E[bits_set] = 256 * (1 - (1-1/256)^80000). Since (1-1/256)^80000 = ((1-1/256)^256)^(80000/256) ~ e^(-312.5) ~ 0, all bits are set. The calculation is correct and explains the 0% elimination at small m.

4. **Coverage calculation** (Section 3.2): 20480 / 16.7M = 0.12% coverage. Correct. This is the core quantitative argument for why the approach fails: the profiling set cannot cover the quantization space.

5. **Key space size** (Section 3.2): 256^8 = ~1.8e19 at full resolution, 8^8 = 16.7M at 3-bit quantization. Both correct. The gap between profiled tokens (~20K) and key space (~16.7M even at coarse quantization) is five orders of magnitude.

6. **Parameter count** (Section 5): Correctly computed. The Bloom filter storage is non-learned and separate from the 204K learned parameters, which is an appropriate accounting.

One minor note: the MATH.md Section 3.4 states "76-85% of top-k experts that SHOULD fire are eliminated" while the PAPER.md Phase 2 table shows FN-in-top-k ranging from 76.4% to 99.1%. These are consistent (MATH.md gives the range for threshold=0.5 only; PAPER.md includes threshold=1.0). No error.

## Novelty Assessment

**Prior art check**: The paper correctly identifies that no published work uses Bloom filters specifically for MoE expert pre-filtering. The closest references are:

- Hash Layers (Chen et al., NeurIPS 2021): replaces routing with hashing, does not pre-filter
- PEER (He et al., DeepMind 2024): product-key retrieval for large expert pools, different mechanism

The experiment is novel in the specific combination (Bloom filter + softmax two-stage), even though the result is negative. The negative result itself is the contribution: it establishes that exact membership data structures are fundamentally unsuited for continuous-valued routing, which is a useful finding for the research program.

**REFERENCES.yml check**: No Bloom filter specific reference exists. The LSH experiment (`lsh_capsule_routing`) is referenced appropriately as the similarity-preserving alternative.

**FINDINGS.md check**: An earlier experiment used a "Bloom filter gate" (Phase 2 in an older ablation study) with zero effect on training loss. That was a different mechanism (training-time gate vs inference-time pre-filter), but the researcher does not cite it. Minor omission -- the mechanisms are different enough that this is not reinvention.

## Experimental Design

**Does it test the hypothesis?** Yes. The hypothesis states: "eliminate at least 30% of expert-token pairs with FPR under 20%." The experiment sweeps m_bits (64 to 100K), activation thresholds (0.1 to 2.0), and uses 3 seeds. Both kill criteria are tested directly.

**Controls adequate?** Yes. Baseline is the same model without Bloom filtering (bloom_mask = all-ones). The comparison is apples-to-apples since the underlying model is identical.

**Could a simpler explanation account for the results?** No alternative explanation is needed. The saturation at small m is a straightforward mathematical consequence. The high FN rate at large m follows directly from the coverage gap. The researcher correctly identifies the root cause.

**Hypothesis graph consistency**: The HYPOTHESES.yml node `exp_bloom_prefilter` has status "disproven" with kill criteria matching exactly what was tested. The evidence summary accurately reflects the experimental findings.

**One design question the researcher did not explore**: Locality-sensitive quantization (e.g., Gray codes across bin boundaries, or overlapping bins where each value maps to 2-3 adjacent bins). This would partially address the bin boundary problem. However, the researcher correctly notes in Section 4 of MATH.md that this would be reinventing LSH with extra steps, which is already validated. Not a flaw in the experiment -- just a road not taken, with good reason.

## Macro-Scale Risks (advisory)

Not applicable. The experiment is killed. The failure mode (exact vs approximate membership) is scale-independent, as the researcher correctly argues. Stronger expert specialization at macro scale would improve coverage but cannot fix the quantization boundary problem.

## Minor Issues

1. The `BloomFilter` scalar class (line 235-276) uses a different quantization scheme (8 bins, range [-3, 3]) than the `VectorizedBloomBank` (256 bins, range [-4, 4]). The scalar class is only used in unit tests, so this inconsistency does not affect results, but it means the unit tests do not exactly match the production code path.

2. The `profile_batch` method in `BloomPrefilterGPT` (line 462-476) computes `h = layer.norm2(x + layer.attn(layer.norm1(x)))` and then calls `x = layer(x)`, which recomputes the same attention and norms. This is a 2x compute overhead during profiling. Acceptable for a micro experiment but worth noting.

## Verdict

**PROCEED** (as a completed, killed experiment)

The experiment is well-designed, mathematically sound, thoroughly analyzed, and correctly killed. The researcher identified the fundamental failure mode (exact membership vs approximate similarity), quantified it precisely (0.12% coverage of key space, 76-99% routing false negatives), and correctly pointed to validated alternatives (LSH, hierarchical trees). The HYPOTHESES.yml node is marked "disproven" with appropriate evidence.

No revisions needed. The negative result is clean and informative. It advances the research program by definitively ruling out exact-membership data structures for continuous-valued routing and strengthening the case for similarity-preserving alternatives.
