# Peer Review: m2p_safe_dissolve_comparison (RE-REVIEW)

## Experiment Type
Guided exploration

## Previous Issue Verification

### Blocking Issues

**1. Add Self-Test section to MATH.md (all 6 items)**
PROPERLY APPLIED. Self-Test section present at end of MATH.md (lines 159-186) with all 6 items completed. Each answer is substantive and specific. The impossibility property correctly notes this is a guided-exploration (no impossibility theorem required) and identifies the S1 structural finding. Cited theorem (Eckart-Young-Mirsky) is real, conditions stated, limitation acknowledged. Predictions reference the table. Falsification condition targets S3 specifically. HP counts listed per strategy. Hack check correctly identifies this as a comparison study.

**2. Reframe S1 no-merge result as negative finding**
PROPERLY APPLIED. Section 5 of PAPER.md now leads with "Negative result: at this scale, naive loss-gating cannot safely merge any cross-domain adapter." The paragraph explicitly states S1 merged 0/10 adapters, that the enriched base is byte-identical to the original, and that 90.15% quality is "unmodified base quality -- not the result of any dissolve." The finding is framed as "loss-gating is structurally equivalent to 'do not promote' when cross-domain interference exceeds tau for every adapter." This is clear, honest, and correctly scoped to this scale.

**3. Re-evaluate K884 Pareto dominance -- S3 as true winner**
PROPERLY APPLIED. K884 entry in PAPER.md Section 3 now reads: "S3 is the Pareto winner among merge strategies: same median quality as naive (89.17%), 5/5 domains protected (vs 4/5 for naive), zero merge time overhead, zero inference time overhead. S1 also satisfies K884 vacuously (0 adapters merged), but S3 is the operationally meaningful Pareto winner." Section 5 reinforces S3 as "the actual winner among strategies that do something." Section 7 (Implications) recommends S3 as default, not S1. This fully addresses the original critique.

**4. State proven framework precisely -- cite theorem, not just a finding**
PARTIALLY APPLIED. MATH.md lines 7-13 now provide a more precise statement with specific numbers (parity: 0.59 -> 3.73, +532%) and scopes the vulnerability to "parity-class domains (SFT delta < 0.05 nats)." However, it labels this "Theorem (prior result, Finding #353)" which is misleading -- Finding #353 is an empirical observation from exp_m2p_cross_domain_graph, not a proven theorem. The notation "O(scale * ||DW_i|| * kappa(W))" is an unproven asymptotic claim presented in mathematical notation. For a guided-exploration, the requirement is to cite the proven framework -- calling an empirical finding a "Theorem" does not make it one.

This is NOT blocking because: (a) the experiment type is guided-exploration, not verification; (b) the specific empirical numbers (0.59->3.73, +532%) are real and well-documented; (c) the exploration successfully operates within this framework. But the label should be "Empirical result (Finding #353)" or "Prior finding," not "Theorem."

### Non-Blocking Issues

**5. Median reporting inconsistency (excl. vs incl. parity)**
PROPERLY APPLIED. PAPER.md Section 4 now says "Median (all 5 domains): 90.15%" with explicit note that "parity (-1887%) is the minimum, excluded by the median statistic." This is accurate and transparent. The code computes median over all 5 domains, and the paper correctly reports this.

**6. Pareto dominance code 0.95 relaxation alignment**
NOT APPLIED. The code at line 907 still uses `median_quality >= naive_median * 0.95` (5% relaxation). MATH.md says "same or better median" with no mention of relaxation. PAPER.md does not document the relaxation either. This does not affect the outcome (S3 has exactly the same median as naive, 0.8917 = 0.8917, so it would pass without relaxation), but the code-spec discrepancy remains. Non-blocking since the result is unaffected.

**7. S4 quality ceiling language qualification**
PROPERLY APPLIED. PAPER.md Section 5 now says "S4 (null-space) is the quality ceiling for non-parity domains, but fails catastrophically on parity (-2497%)." Section 7.3 qualifies: "quality ceiling for non-parity domains if parity-class domains are explicitly excluded before merge" and adds "must not be used without explicit parity exclusion." Clear and properly scoped.

## Hack Detector
- Fix count: 5 strategies compared head-to-head (comparison study, not additive fixes -- not flagged)
- Is MATH.md a proof or a description? Description of 5 strategies with equations describing what each computes. No Theorem/Proof/QED block. This is acceptable for guided-exploration type.
- Metric used as evidence: median quality ratio + domain protection count + cost metrics. Quality ratio is (base_loss - enriched_loss) / (base_loss - sft_loss), a reasonable proxy. Not proven to predict behavioral outcomes, but adequate for a comparison study.
- Kill criteria source: derived from problem statement (engineering targets: 90%, 5%, 2x). Not mathematically derived, but appropriate for guided-exploration.

## Self-Test Audit
1. One-sentence impossibility property: PASS. Correctly states this is guided-exploration (no impossibility theorem) and identifies the S1 structural finding as the key impossibility-like result.
2. Cited theorems: PASS. Eckart-Young-Mirsky is real. Limitation (weight-space vs hidden-state distance) explicitly stated. The "Theorem (prior result, Finding #353)" label is misleading but the content is accurate.
3. Predicted numbers: PASS. Table gives specific ranges per strategy. Post-hoc accuracy assessment (3/5 correct) is honest.
4. Falsification condition: PASS. "S3 does NOT achieve 5/5 domain protection at <=2x inference memory cost." Specific, testable, targets the mechanism.
5. Hyperparameter count: PASS. Counts per strategy: S0=1, S1=2, S2=2, S3=1, S4=3. Notes S3 has lowest HP count among protective strategies.
6. Hack check: PASS. Comparison study, not fix iteration.

## Mathematical Soundness

The five strategies are mechanically described, not proven. This is acceptable for a guided-exploration. The key mathematical claims that DO hold:

- S3 structural protection: routing parity to original base guarantees zero degradation. Verified: parity enriched_base_loss = 0.5855 = base_loss (from results.json). SOUND.
- S4 SVD projection: Eckart-Young-Mirsky guarantees optimal rank-k approximation in Frobenius norm. Applied correctly. The noted failure (hidden-state projection does not capture weight-space interference) is an honest limitation, not a bug.
- S1 loss-gating: the tau=5% threshold correctly rejects all 10 adapters. The code evaluates each adapter independently (greedy). The conclusion (all adapters individually exceed tau on at least one domain) is verified by results.json (merged=0, skipped=10).

One notation issue: "O(scale * ||DW_i|| * kappa(W))" on line 9 of MATH.md is stated as a bound but has no derivation. This is an empirical scaling observation, not a proven bound. Non-blocking for guided-exploration.

## Prediction vs Measurement

PAPER.md Tables 1-3 present predictions vs measurements for all 5 strategies. Assessment:

| Strategy | Quality Prediction | Protection Prediction | Verdict |
|----------|-------------------|----------------------|---------|
| S0 Naive | 91.5% vs 89.17% (-2.3pp) | Parity fails: confirmed | Correct direction |
| S1 Loss-gated | 85-90% vs 90.15% (above range) | 5/5: confirmed | Trivially correct (no merge) |
| S2 Headroom | 88-92% vs 88.54% (in range) | ~0% vs +854%: WRONG | Prediction failed |
| S3 Selective | 91.5% vs 89.17% (-2.3pp), 0%: confirmed | 5/5: confirmed | Correct |
| S4 Null-space | 80-88% vs 90.66% (above range) | <5% vs +760%: WRONG | Prediction failed |

3/5 predictions correct directionally. 2/5 failed on parity protection. Root cause analysis provided (Section 6): cross-domain interference penetrates both headroom scaling and null-space projection. This is honest and informative.

## NotebookLM Findings
Skipped -- manual analysis was sufficient.

## Novelty Assessment
This is a practical engineering comparison, not a novel theoretical contribution. The five strategies are standard techniques (gated merging, proportional scaling, selective routing, null-space projection). Value is in the head-to-head comparison within the M2P framework, producing an actionable recommendation (S3). No prior art concerns.

## New Issues Found in Revision

1. **Minor: S3 "zero inference time overhead" is technically inaccurate.** results.json shows S3 inference_overhead_s = 0.001 vs S0's 0. The 1ms is negligible, but the PAPER.md claims "zero inference time overhead" in multiple places (lines 117, 60). Should say "negligible (~1ms)" not "zero." NON-BLOCKING.

2. **Minor: "Theorem (prior result, Finding #353)" mislabeling.** As noted in issue #4 above, Finding #353 is an empirical result, not a theorem. The asymptotic notation O(scale * ||DW_i|| * kappa(W)) has no derivation. Should be labeled "Empirical result" or "Prior finding." NON-BLOCKING for guided-exploration.

3. **Minor: Pareto code 0.95 relaxation still undocumented** (carried over from issue #6). NON-BLOCKING.

## Macro-Scale Risks (advisory)

1. S1's "skip everything" result may not transfer -- at larger scale with better-trained adapters, some may pass loss-gating. The negative result is scoped to micro scale with 10 cross-domain adapters.
2. S3's 2x memory cost scales linearly with base model size. At Qwen3-4B (8GB), this means 16GB for two copies. Feasible on M5 Pro 48GB but consumes significant budget.
3. The hybrid S3+S4 suggestion (route parity to original, use null-space for hard domains) is untested. Worth a follow-up experiment.

## Verdict

**PROCEED**

All 4 blocking issues from the previous review have been addressed (3 properly applied, 1 partially applied but non-blocking for guided-exploration). The Self-Test section is complete and substantive. The S1 negative result is properly framed. S3 is correctly identified as the operational winner. K884 Pareto assessment is properly contextualized. The 3 remaining minor issues (inference overhead rounding, theorem mislabeling, code relaxation) are non-blocking presentation concerns that do not affect the validity of the finding.

The experiment successfully narrows the unknown (which protection strategy gives the best quality/cost tradeoff) to a clear answer: S3 (selective routing) is the Pareto winner among strategies that actually merge adapters. This is a valid guided-exploration result.
