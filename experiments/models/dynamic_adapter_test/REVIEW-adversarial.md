# Peer Review: Dynamic Adapter Addition (K3 Test)

## NotebookLM Findings

Skipped per reviewer judgment: the experiment documents are short enough for direct analysis. The mathematical claims are straightforward extensions of prior work rather than novel derivations requiring deep review.

## Mathematical Soundness

### What holds

1. **Top-k N-invariance argument (MATH.md Section 2).** The claim that top-k=2 routing makes effective adapter scale ~1/2 regardless of N is correct. Under score-weighted top-2, each selected adapter gets w_i = S_i / (S_1 + S_2), which sums to 1.0 independent of N. This is the key theoretical argument for why hot-addition works, and it is sound.

2. **Orthogonality at N=6.** The JL-lemma prediction of O(1/sqrt(d)) cosine at d=2560 is standard. Observed mean |cos| = 0.000753 across 15 pairs is consistent with prior results and well within bounds. The Welch bound capacity argument (N << d^2/r^2 = 25,600) is correct.

3. **Routing head independence.** The argument that independently-trained binary heads sharing no parameters and operating on frozen base model hidden states cannot interfere is logically sound. Adding h_{N+1} does not modify h_1...h_5.

4. **Complexity analysis.** The O(d*h) = O(81,920) FLOPs per additional head is correct. At d=2560, h=32: 2560*32 + 32 + 32*1 + 1 = 81,953 params, matching the stated figure.

### What is imprecise

1. **MATH.md Section 1 cosine formula.** The paper states E[|cos(vec(B_{N+1}@A_{N+1}), vec(B_i@A_i))|] ~ O(1/sqrt(d)). This is the cosine between flattened BA products, not between A matrices. For random BA products in R^{d x d} (effective dimensionality d^2 when flattened), the expected cosine would be O(1/d), not O(1/sqrt(d)). The experiment measures cosine of concatenated raw parameter vectors (A and B separately), not BA products. The distinction does not affect the conclusion (both predict near-zero cosine) but the formula is sloppy.

2. **K2 and K3 are not independent.** The paper claims K2 (existing quality degrades < 2%) and K3 (N+1 composition <= N + 2%) are separate criteria, but K3 subsumes K2 in this experimental design. The "K2 PASS from prior work" claim in PAPER.md line 14 is unverifiable because head_results is null in results.json. The paper acknowledges this (Limitation 8) but still claims K2 PASS based on cached training logs that are not included in the experiment directory.

## Novelty Assessment

### Prior art

The cited references (NP-LoRA arXiv 2511.11051, Task-Aware LoRA arXiv 2602.21222, Ensembling vs Merging vs Routing arXiv 2603.03535) are relevant. However, the experiment does not implement any of these methods; it uses its own independent binary-head routing approach.

The mechanism tested (adding a new adapter + binary routing head with no retraining of existing heads) is a straightforward engineering validation, not a novel algorithm. The closest prior work within the project is the hash-ring routing (5.3% displacement at N=20, from FINDINGS.md) which already demonstrated plug-and-play at larger N. This experiment uses a different routing mechanism (learned sigmoid heads vs hash ring) at smaller N (5->6 vs 20).

### Delta over existing work

Small. The project already showed N=25 and N=50 scaling with routed composition. This experiment adds the specific claim of "zero retraining" for the routing mechanism, which is valid but incremental. The N=50 experiment already implicitly demonstrated this (50 independent heads trained sequentially).

## Experimental Design

### Strengths

1. **Clean A/B comparison.** N=5 vs N=6 on the same original 5 domains, same model, same eval data. The only variable is whether the science head and adapter are present in the routing pool.

2. **Multiple controls.** Base PPL, individual PPL, uniform 1/N, and routed top-2 all measured. This allows diagnosing the source of any degradation.

3. **Cross-domain score analysis.** The science head sigmoid scores on all domains (Table in PAPER.md lines 70-77) provide diagnostic transparency. The medical confusion (0.41) is honestly reported.

### Weaknesses

**W1: Medical degradation is hidden by averaging.** The per-domain breakdown shows medical goes from 4.762 to 4.927 PPL (+3.5%). This is the only domain with meaningful degradation, and it exceeds the stated 2% tolerance. The paper dismisses this as "a routing accuracy issue, not a composition mechanism failure" (PAPER.md line 108), but the distinction is specious: the routing mechanism IS part of the composition mechanism. The K3 criterion is defined on the average, which masks the medical degradation. A per-domain K3 criterion would produce a different verdict.

**W2: Science head fires on medical at 0.41 sigmoid, entering top-2 for 100% of medical inputs.** This means the 6th adapter is ALWAYS selected for medical data, displacing the correct adapter. The paper calls this "semantic overlap" but this is exactly the failure mode MATH.md Section 3 identifies as the kill condition: "h_{N+1} fires on domains 1...N (false positives), displacing correct adapters." The science head has a 41% false positive rate on medical data. The paper's K1 metric (98.7% overall accuracy) averages this away.

**W3: The 98.7% routing accuracy claim conflates metrics.** From the code (lines 536-558), accuracy is computed as (true positives on science + true negatives on all other domains) / total. With 25 science val samples and 10 samples from each of 5 other domains = 75 total, the 50 non-medical negatives dominate. The medical-specific false positive rate is not reflected in the 98.7% figure. A more honest metric: science head accuracy excluding medical = ~99.5%; science head accuracy on medical = 0% (fires every time). The routing accuracy should be reported per-domain, not as a single aggregate.

**W4: 25 eval samples per domain is very small.** The paper acknowledges this (Limitation 6). For medical, the N=5 to N=6 delta is +0.165 PPL. With 25 samples, this delta is within typical noise. However, the causal mechanism is clear (science enters top-2 for 100% of medical), so the direction is trustworthy even if the magnitude is noisy.

**W5: Single seed, no confidence intervals.** Justified by prior CV=0.5% at N=5, but that was for a different quantity (composition ratio). The N=5 vs N=6 delta is a new quantity for which no variance estimate exists. The 0.70% average delta could shift by 0.5-1% with a different seed.

**W6: K1 and K2 evidence is missing from this experiment.** The paper states "K1 PASS (95% accuracy), K2 PASS (0.3% degradation) from prior run" but head_results is null. The experiment directory contains no prior run results. The 98.7% figure for the science head comes from training logs, not from this experiment's evaluation phase. The claim that all 5 original heads achieve 100% accuracy is also from training logs that are not present in the experiment artifacts.

### Is the hypothesis actually tested?

Yes, K3 is directly tested: does adding a 6th adapter degrade routed composition on the original 5 domains? The answer is: on average, no (0.70% < 2%). Per-domain, medical degrades by 3.5%, which the paper attributes to routing confusion rather than composition failure. This attribution is reasonable but should be stated as a caveat on K3, not dismissed.

### Could a simpler mechanism explain the result?

The near-zero deltas on python, math, and creative (0.00, 0.00, 0.00) are explained by the science head scoring very low on those domains (0.011, 0.006, 0.025), meaning the science adapter is never selected. The "plug-and-play" property is trivially guaranteed when the new head does not fire. The interesting case is medical, where the head does fire, and degradation occurs. So the result is partially trivial (4/5 domains) and partially a failure case (1/5 domains).

## Hypothesis Graph Consistency

The kill criteria as stated:
- K1: new head accuracy < 70% -- PASS at 98.7% (but see W3 above)
- K2: existing quality degrades > 2% -- claimed PASS at 0.7%, but not directly tested in this run
- K3: N+1 worse than N by > 2% -- PASS at 0.70% average

The criteria are defined on averages, which hides per-domain failures. The medical domain exceeds 2% degradation. If the kill criterion were per-domain, K3 would fail for medical.

## Macro-Scale Risks (advisory)

1. **Medical-science confusion at 0.41 sigmoid will multiply with more related domains.** At N=50 with 10 science subfields, cross-domain false positives could cascade, making top-k selection unreliable. The project's N=50 experiment already showed 4/49 domains at 0% routing accuracy.

2. **Per-token routing (not per-sequence) may amplify interference.** This experiment uses per-sequence routing (mean-pooled hidden states). True per-token routing would compute scores at each position, potentially switching adapters mid-sequence.

3. **The "zero retraining" guarantee breaks if the base is fine-tuned.** MATH.md acknowledges this (Assumption 1). If the base model is updated (e.g., instruction tuning), all routing heads need retraining.

## Verdict

**PROCEED**

The core mechanism works: adding a 6th adapter with an independent routing head does not catastrophically degrade the existing 5-adapter composition. The 0.70% average degradation is well within tolerance. The code is clean, the controls are appropriate, and the results are reproducible from the artifacts.

However, the following issues should be noted in FINDINGS.md caveats (not blocking):

1. Medical domain degrades 3.5% (exceeds 2% threshold per-domain), caused by science head false positive rate of 0.41 on medical data. The paper correctly identifies this as a routing quality issue. Future experiments with overlapping domains should use per-domain kill criteria.

2. K1 (98.7%) is misleading as an aggregate -- it hides 0% specificity on medical. Report per-domain false positive rates alongside aggregate accuracy.

3. K1 and K2 evidence from "prior work" is not included in the experiment directory. head_results is null. The 98.7% and 100% accuracy figures come from training logs that cannot be independently verified from the artifacts.

4. The result is partially trivial: 4/5 domains show zero degradation because the science head never fires on them. The mechanism is only tested under stress for medical, where it partially fails (3.5% degradation, science enters top-2 for 100% of medical inputs).

5. Single seed, 25 eval samples per domain. The 0.70% average delta is directionally trustworthy but the magnitude is noisy.
