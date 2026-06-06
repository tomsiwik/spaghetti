# Peer Review: Behavioral Deduplication (Re-Review After Revision)

## Prior Review Summary

The first review issued REVISE with 5 required fixes:

1. Fix `_count_unique_capsules_in_pairs` set-difference bug
2. Sweep tau_rho at {0.3, 0.5, 0.7}
3. Report per-seed kill criterion values in PAPER.md
4. Reframe narrative away from "finds what weight-cosine misses"
5. Acknowledge 0 weight-cosine pairs issue more prominently

All 5 fixes were applied. This re-review assesses whether the revised experiment meets the bar for PROCEED.

## Fix Verification

### Fix 1: Set-Difference Bug -- RESOLVED

The function `_count_unique_capsules_in_pairs` at line 525 of `/Users/tom/Code/tomsiwik/llm/micro/models/behavioral_dedup/behavioral_dedup.py` now correctly computes the set difference:

```python
if pair_type == "behavioral_only":
    behavioral_set = set((p["i"], p["j"]) for p in lr["behavioral_pairs"])
    weight_set = lr.get("weight_pairs", set())
    behavioral_only_set = behavioral_set - weight_set
    for i, j in behavioral_only_set:
        unique.add((lr["layer"], i))
        unique.add((lr["layer"], j))
```

The `weight_pairs` set is now stored in the per-layer results dict (line 431: `"weight_pairs": weight_pairs`), enabling correct set subtraction. The fix is structurally sound. Since weight-cosine found 0-1 pairs across seeds, the numerical results are unchanged, but the code now correctly implements the metric it claims to compute.

### Fix 2: tau_rho Sweep -- RESOLVED

PAPER.md now reports a 3-level sweep (Section "Output Correlation Threshold Sweep"):

| tau_rho | Behavioral Pairs | Behavioral-Only Capsule % | Result |
|---------|-----------------|---------------------------|--------|
| 0.3     | 1623            | 19.3% +/- 3.1%            | PASS   |
| 0.5     | 236             | 10.8% +/- 3.8%            | PASS   |
| 0.7     | 8               | 1.4% +/- 0.5%             | KILL   |

This is the most informative addition. It reveals that the finding is threshold-sensitive: substantial co-firing exists (Jaccard) but truly correlated outputs (high rho) are much rarer. The paper correctly acknowledges this.

### Fix 3: Per-Seed Kill Criterion Table -- RESOLVED

PAPER.md now includes a per-seed table at J>0.7, tau_rho=0.3:

| Seed | Behavioral-Only Capsule % | Total Alive | Weight-Cos Pairs |
|------|---------------------------|-------------|------------------|
| 42   | 17.9%                     | 398         | 0                |
| 123  | 17.2%                     | 394         | 0                |
| 7    | 22.9%                     | 463         | 1                |

All 3 seeds individually exceed the 5% kill threshold at tau_rho=0.3. At tau_rho=0.5 (per-seed: 8.7%, 8.5%, 15.2%), all 3 seeds still exceed 5%. The variance is notable (seed 7 is consistently higher, likely due to having more alive capsules and 1 weight-cosine pair), but not problematic for a 3-seed micro experiment.

### Fix 4: Narrative Reframing -- RESOLVED

PAPER.md's main finding is now titled "The Real Finding: Layer 0 Co-Activation from Shared Input Statistics" and the text explicitly states: "the experiment tests whether behavioral analysis finds *any* redundancy, not whether it finds *more than* weight-cosine." The framing is honest and accurate.

### Fix 5: 0 Weight-Cosine Pairs Acknowledgment -- RESOLVED

The paper now includes a prominent paragraph: "Important context on the weight-cosine comparison. Weight-cosine found near-zero pairs (0.3 mean, i.e. 0 in 2 seeds, 1 in 1 seed)... the comparison 'behavioral 19.3% vs weight-cosine 0%' is trivially won."

## Mathematical Soundness (Re-Verified)

No changes to the mathematical content from the first review. The derivations remain correct:

- **Jaccard computation**: Standard, correctly implemented.
- **Output correlation factorization**: Correct under rank-1 capsule structure. The `H[i,j] * B_dot[i,j]` decomposition holds.
- **Merging error bound**: Still stated but not numerically evaluated. Acceptable at micro scale since the empirical quality test (+0.3% vs concat) validates that merging is harmless.
- **Conditioned output cosine**: Still a weight-space metric, now explicitly acknowledged in PAPER.md: "is actually a weight-space metric (b-vector cosine similarity) that does not use activation data."

One concern from the first review that remains unaddressed but is not blocking: the merging bound assumes high Jaccard implies similar activation magnitudes. This is not guaranteed -- two capsules can co-fire on identical inputs with very different magnitudes. The +0.3% quality result empirically validates that this is not a problem in practice, but the mathematical argument in MATH.md Section 3 is still incomplete.

## Experimental Design Assessment

### Strengths After Revision

1. **The tau_rho sweep is the key contribution.** It reveals the layered structure of redundancy: massive co-firing (Jaccard) but progressively less output correlation. The 7x drop from tau_rho=0.3 to tau_rho=0.7 (19.3% to 1.4%) is itself an interesting finding -- it means Layer 0 capsules fire on the same inputs but produce outputs in different directions (different b vectors). This is structurally meaningful.

2. **The per-seed table confirms robustness.** All 3 seeds individually pass at tau_rho=0.3 and tau_rho=0.5. The minimum per-seed value at tau_rho=0.5 is 8.5% (seed 123), still above the 5% threshold.

3. **The narrative is now accurate.** The paper no longer overclaims. It correctly frames this as "behavioral analysis reveals a structural phenomenon (Layer 0 co-activation) rather than outperforming a functioning baseline."

### Remaining Weaknesses (Non-Blocking)

1. **Quality impact is negligible.** Behavioral dedup achieves +0.3% vs concat. Weight averaging achieves -3.7%. The paper correctly notes "behavioral dedup is not a viable compression strategy." This limits the practical value of the finding, but the paper does not overclaim utility.

2. **Profiling on validation data.** Still uses `joint_val` for both profiling and quality evaluation. As noted in the first review, this creates a subtle optimistic bias. Non-blocking for micro.

3. **The simpler baseline question remains.** Computing input representation cosine similarity per layer would likely show that Layer 0 inputs are near-identical across domains, making co-activation predictable without behavioral profiling. The paper partially addresses this by noting "This follows directly from the feature hierarchy principle and does not require behavioral profiling to motivate -- but the experiment quantifies the effect." Fair.

4. **3 seeds with n=3 statistical power.** Standard limitation for micro experiments. The per-seed values are consistent enough (no seed below 5% at tau_rho <= 0.5) that this is acceptable.

## Novelty Assessment

Unchanged from first review. The technique (co-activation Jaccard for MoE deduplication) adapts BuddyMoE/Sub-MoE ideas to the capsule composition setting. The finding (Layer 0 redundancy from shared features) confirms Yosinski et al. 2014 in this context. The novelty is in the application and quantification, not the technique or finding.

The practical insight -- "share Layer 0, concatenate deeper layers" -- is the most valuable output and is well-supported by the data.

## Hypothesis Graph Consistency

The `HYPOTHESES.yml` entry for `exp_behavioral_dedup` has status `proven` with evidence citing "18.3% of capsules in behaviorally-redundant pairs." The revised PAPER.md reports 19.3% (slightly different, likely from the set-difference fix or rounding). The hypothesis entry should be updated to match the revised numbers and to note the threshold sensitivity (PASS at tau_rho <= 0.5, KILL at tau_rho = 0.7). The current evidence text does not mention the tau_rho sweep, which is the most important finding of the revision.

Minor discrepancy: the HYPOTHESES.yml evidence says "0 pairs" for weight-cosine while the revised PAPER.md says "0.3 mean (0 in 2 seeds, 1 in 1 seed)." Should be consistent.

## Macro-Scale Risks (Advisory)

Unchanged from first review:

1. **Subword tokenization**: Layer 0 co-activation depends on shared tokens. With domain-specific subword vocabularies, the effect may weaken substantially.
2. **Quadratic cost**: (P_total)^2 Jaccard matrix at P_total=8192 requires careful batching.
3. **Threshold choice at scale**: The tau_rho sensitivity (19.3% at 0.3 vs 1.4% at 0.7) means the "right" threshold for macro experiments is underdetermined.

## Verdict

**PROCEED**

All 5 required fixes from the first review have been applied correctly. The revised paper is honest about its limitations, the tau_rho sweep reveals genuine structure in the data, and the per-seed table confirms robustness. The kill criterion is met at tau_rho <= 0.5 across all seeds.

The experiment's value is not as a compression strategy (merging barely helps) but as a diagnostic that reveals layer-dependent redundancy structure. The practical implication -- share Layer 0 capsule pools across domains -- is well-supported and actionable.

Remaining minor items (not blocking):

- Update `HYPOTHESES.yml` evidence to mention the tau_rho sweep and use the revised 19.3% figure.
- The merging error bound in MATH.md Section 3 remains mathematically incomplete (does not account for activation magnitude differences), but the empirical quality result (+0.3%) makes this a documentation issue, not a correctness issue.
