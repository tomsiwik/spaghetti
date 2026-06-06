# Peer Review: Hash Ring Expert Removal

## NotebookLM Findings

Skipped. The experiment is sufficiently straightforward (consistent hashing properties + synthetic quality model) that deep review via NotebookLM would add latency without insight. The mathematical claims reduce to well-known results from Karger et al. (1997) plus a synthetic quality simulation.

## Mathematical Soundness

### Consistent hashing properties (Sections 1.1-1.3 of MATH.md): CORRECT

The displacement bound, neighbor redistribution, and add-remove symmetry are textbook consistent hashing results. The proofs in MATH.md are correct restatements of Karger et al. (1997). There is nothing novel here mathematically -- these are known properties being verified empirically, which is appropriate for a micro-experiment validating infrastructure.

### Load balance analysis (Section 1.4): CORRECT but reveals a problem

The variance formula Var[load_i] ~ 1/(NV) is standard. The empirical observation that FNV1a produces 1.8x imbalance at N=8 is honestly reported. This is important: at N=4, expert 2 handles 38.7% of tokens (1.55x the theoretical 25%), which directly drives degradation above the 5% threshold at N=4.

### Quality degradation model (Section 2): CIRCULAR BY DESIGN

The quality model defines Q[i,i] = 1.0 (expert is perfect on its own domain) and Q[i,j] = 1 - s*d(i,j). The "domain" of a token is defined as the expert it was assigned to BEFORE removal (line 325 of run_experiment.py: `domain = assignments_before[i]`). This is circular: a token's "true domain" is defined by its hash ring assignment, not by any actual semantic property. The quality degradation is therefore an artifact of the synthetic model, not an empirical measurement of real quality loss.

This is acknowledged in the Limitations section and is acceptable for a micro-scale routing experiment. The structural guarantees (zero false moves, 100% neighbor accuracy, perfect symmetry) are the real contribution. The quality model serves only to show that degradation is bounded by (load_fraction * specialization), which is a trivially correct mathematical consequence of the model definition.

### Degradation bounds (Section 2.3): SLIGHTLY MISLEADING

MATH.md predicts expected degradation ~ 1.88% at N=8, s=0.3. The measured values are -1.43% to -1.56%, which is consistent. However, the PAPER.md reports "Mean: -2.23%, Max: -4.53%" for the N=8 kill criteria assessment -- these come from the edge_case test (removing experts 0, 4, 7) where expert 0 has pathological load due to FNV1a (22.5% vs 12.5%). The mean is inflated by expert 0's disproportionate load. This is not wrong, but the framing of "-2.23% mean" obscures that the mean for non-pathological experts is closer to -1.2%.

### N=4 degradation exceeds 5% threshold

Results.json shows N=4 degradation of -9.0% to -9.7% across seeds. PAPER.md reports this as "-9.40%" in the scaling table. The kill criteria are specified at N=8, so this does not trigger a kill. However, MATH.md's scaling table (Section 2.4) shows "Measured: ~3.07% (mean, mid)" for N=4, which contradicts the results.json showing -9.0% to -9.7%.

**This is a discrepancy.** The MATH.md table says N=4 measured degradation is ~3.07%, but results.json shows -9.03% to -9.73%. Looking more carefully: MATH.md says "mid" specialization -- perhaps this refers to a different specialization value? The experiment runs s=0.3 as default. The PAPER.md scaling table says N=4 degradation is -9.40%. The MATH.md table appears to contain incorrect values for N=4 and N=16.

**Specific discrepancy:** MATH.md Section 2.4 claims N=16 measured = ~1.51%, but the PAPER.md scaling table also says -1.51%. These match. For N=4, PAPER.md says -9.40% but MATH.md says ~3.07%. MATH.md is wrong for N=4.

### Neighbor accuracy check (lines 250-264 of run_experiment.py): WEAKER THAN CLAIMED

The code checks whether each displaced token's new assignment is in the SET of all clockwise neighbors of the removed expert's virtual nodes:

```python
neighbor_set = set(neighbors.values())
...
if new_expert in neighbor_set:
    n_to_neighbor += 1
```

This checks set membership, not per-virtual-node correctness. For example, if expert 4 has virtual nodes with neighbors {2, 5, 7}, and a token that SHOULD go to expert 2 instead goes to expert 5, this would still count as "to neighbor." However, by consistent hashing construction, this cannot happen -- each token walks clockwise to the specific next virtual node, not to an arbitrary neighbor. So the check is weaker than it could be, but the result (100%) is correct by construction. The code is correct in practice even though the verification is imprecise.

A stronger test would verify that each displaced token goes to the SPECIFIC clockwise neighbor of the virtual node it was closest to. This would catch implementation bugs that the current test cannot.

## Novelty Assessment

**Zero novelty.** This experiment verifies known properties of consistent hashing (Karger et al. 1997). The paper correctly frames it as validating the dual of the previously proven add-expert operation. The contribution is not theoretical novelty but completing the operational lifecycle (add + remove) needed for clone-and-compete.

This is appropriate. Not every experiment needs to be novel. Infrastructure validation is necessary work.

The prior `consistent_hash_routing` experiment already validated add-expert. This experiment validates remove-expert. Together they unblock `exp_clone_compete_evolution`, which is the actual novel contribution in the roadmap.

## Experimental Design

### Strengths

1. **Seven distinct tests** covering scaling, edge cases, redistribution destinations, false moves, virtual node sensitivity, specialization sensitivity, and add-remove symmetry. Thorough.
2. **Three seeds** for stochastic tests. Adequate for a micro experiment.
3. **100K tokens** provides sufficient statistical power for displacement measurements.
4. **Honest reporting** of FNV1a load imbalance rather than hiding it.

### Weaknesses

1. **The experiment is almost entirely tautological.** Consistent hashing's removal guarantee is a mathematical theorem, not an empirical hypothesis. The experiment is really testing the code implementation, not the mechanism. This is fine for infrastructure validation but should not be overclaimed.

2. **Quality degradation is synthetic.** The Q matrix is constructed, tokens' "domains" are defined by their hash assignments, and the degradation formula is an algebraic consequence of the model definition. There is no empirical content in the quality result beyond "our formula computes what our formula says it should compute." The paper acknowledges this ("Routing-only test") but the kill criteria assessment still frames it as if quality degradation was empirically measured.

3. **Token hashes are uniform random, not actual hidden state projections.** The real system hashes projected hidden states via `x @ proj -> FNV1a`. The distribution of these hashes may not be uniform if the projection or data has structure. This is acknowledged but should be tested at macro.

4. **Virtual node sensitivity results are suspicious.** At V=500, displacement drops to 2.92% (0.23x theoretical), but at V=1000 it rises to 5.37% (0.43x). This non-monotonicity is unexplained. It likely reflects FNV1a collision patterns with the specific expert ID (4) and virtual node count, but it undermines confidence in the hash function.

### Does it test the stated hypothesis?

The hypothesis is: "Removing one expert redistributes only ~1/N of tokens, all to the nearest clockwise neighbor, with <5% quality degradation and no recalibration."

- **~1/N displacement**: Partially confirmed. Displacement equals the expert's actual load fraction, not 1/N. This is correctly identified as a hash quality issue.
- **All to nearest clockwise neighbor**: Confirmed (100%), but this is guaranteed by construction.
- **<5% quality degradation**: Confirmed at N=8 with synthetic quality model. Not confirmed at N=4 (-9.4%).
- **No recalibration**: Confirmed by the structural guarantee (no other assignments change).

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry for `exp_hash_ring_remove_expert` lists kill criteria:
1. ">5% degradation when removing one expert from N=8 without recalibration"
2. "removed expert tokens not redistributed to nearest neighbor (>20% go to non-neighbor)"

The experiment tests both at N=8. K1 passes (max -4.53%). K2 passes (100% neighbor accuracy). The status should be updated from "active" to "proven" or "supported."

The experiment correctly blocks `exp_clone_compete_evolution` per the dependency graph.

## MATH.md Discrepancy

MATH.md Section 2.4 scaling table claims N=4 "Measured: ~3.07% (mean, mid)" but results.json shows -9.03% to -9.73% and PAPER.md shows -9.40%. The MATH.md value appears to be the predicted value (~3.75%) rather than the measured value. This should be corrected.

## Macro-Scale Risks (advisory)

1. **FNV1a load imbalance at large N.** The 1.8x imbalance at N=8 may worsen or improve at N=500 depending on hash collision patterns. Production should use xxHash or MurmurHash3 as noted.

2. **Real expert specialization may violate the linear quality model.** If removing a medical expert redistributes tokens to a code expert, the quality drop could be catastrophic for those specific tokens, even if the aggregate is small. The (1/N) * s bound only holds for the linear Q model.

3. **Cascading removals during evolution sweeps.** Removing K experts sequentially is bounded by the harmonic series but may create transient load spikes on specific experts if FNV1a clustering is pathological.

4. **Hash uniformity of real projections.** The uniform random assumption must be validated with actual hidden state projections at macro scale.

None of these are blocking for the micro experiment. They are natural items for macro validation.

## Verdict

**PROCEED**

The experiment correctly validates consistent hash ring removal properties at micro scale. The structural guarantees (zero false moves, 100% neighbor redistribution, perfect add-remove symmetry) hold by mathematical construction and are correctly verified by the implementation. The quality degradation model is synthetic but appropriately bounded. Both kill criteria pass at N=8.

Two non-blocking fixes recommended:

1. **Correct MATH.md Section 2.4 scaling table.** The N=4 measured value is listed as "~3.07%" but results.json shows -9.03% to -9.73%. PAPER.md correctly reports -9.40%. Update MATH.md to match.

2. **Strengthen neighbor accuracy verification.** The current check tests set membership (displaced token goes to ANY clockwise neighbor of the removed expert). A per-virtual-node check would verify that each token goes to the SPECIFIC correct neighbor. This would catch potential implementation bugs. Not blocking because the result is correct by construction, but it would make the test more precise.

One advisory note: the HYPOTHESES.yml status should be updated from "active" to "supported" with appropriate caveats about the synthetic quality model. The structural guarantees are proven; the quality bound is model-dependent and should be validated at macro with real experts.
