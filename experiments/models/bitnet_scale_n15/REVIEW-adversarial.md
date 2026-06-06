# Peer Review: bitnet_scale_n15 (RE-REVIEW)

## Previous Review Status

The first review (REVISE) requested 5 specific fixes. All 5 have been addressed:

1. **Composed/base ratio promoted as primary metric**: YES. The summary table now leads with composed/base ratio marked "(PRIMARY)" and bold formatting. The composition ratio is demoted to a secondary row. An explanatory paragraph (lines 63-74) explains why it grows mechanically with domain diversity.

2. **Medical-health overlap analyzed**: YES. A dedicated section (lines 93-119) reports the medical-health cosine (0.000268), contextualizes it relative to all 105 pairs, and explains via high-dimensional geometry why semantically similar adapters appear orthogonal. The paper honestly notes the ablation (N=14-without-health) was not run but presents geometric evidence for dilution over interference. The paper also flags that cosine may be a nearly unfalsifiable metric at this dimensionality -- a valuable self-critical observation.

3. **Full 15-domain composed/individual + composed/base table**: YES. Lines 146-165 provide a complete table for all 15 domains with five columns (Individual PPL, N=15 Composed PPL, Base PPL, Composed/Individual, Composed/Base). Physics is highlighted as worst case. Key observations are documented.

4. **HYPOTHESES.yml status consistency**: YES. Status changed from "killed" to "supported" with notes framing K3 as dilution-driven under uniform weighting only. The dependency graph (blocks exp_bitnet_scale_n25) is now consistent -- "supported" does not block downstream nodes.

5. **Composed/base ratio highlighted with explanatory note**: YES. The value 0.938 is prominently featured in the summary table, the explanatory paragraph, and the per-domain table. The note about mechanical inflation of the composition ratio metric is clear and well-placed.

## NotebookLM Findings

Skipped (authentication not configured in this session). Review proceeds from direct document analysis.

## Mathematical Soundness

### Composition formula: Correct
The 1/N uniform scaling and ternary STE quantization are standard and correctly specified. No errors in the derivation.

### Composition ratio analysis: Now properly qualified
The previous review's main mathematical concern -- that avg_composed/best_individual grows mechanically with domain diversity -- is now explicitly acknowledged in the paper (lines 68-74). The composed/base ratio (0.938) is correctly promoted as the primary cross-N comparison metric.

### Dilution prediction: Validated
MATH.md predicted medical as the only domain capable of exceeding 10% degradation under complete dilution. The experiment confirmed this (medical +15.06%, all others under 7%). The upper bounds in MATH.md (medical max 22.4%, code max 9.2%, etc.) all held. This is a strong point: the math predicted the outcome correctly.

### Cosine metric weakness: Honestly flagged
The paper's observation that cosine similarity may be "nearly unfalsifiable" at d=2560 with r=16 (lines 117-119) is mathematically sound. With 21.6M-dimensional parameter vectors, even semantically overlapping adapters (medical/health) produce cosine of 0.000268. The K2 criterion (mean |cos| < 0.01) has ~9x margin and would likely pass at N=100+. This is an honest limitation acknowledgment.

### One remaining concern: K1 threshold is weakly informative
The K1 criterion (ratio N=15 < 2x ratio N=5) passes at 1.78x. But as the paper now acknowledges, this metric is dominated by domain mix changes. The 1.78x factor tells us more about the PPL distribution of the 10 new domains than about composition quality. This is not a mathematical error -- the paper correctly identifies it -- but K1 is functioning as a sanity check rather than a discriminating test. This is acceptable for a micro experiment. The composed/base ratio (0.886 to 0.938, a +5.9% shift toward base) is the actual informative signal.

## Novelty Assessment

No novelty claim is made. This is an engineering scaling test extending the project's own N=5 results. Appropriate for its purpose.

## Experimental Design

### Strengths (unchanged from first review)
- Reuses validated adapters from proven multiseed experiment
- Consistent training pipeline across all 15 adapters
- MATH.md predictions validated by results
- 15/15 individual adapters beat base (training pipeline generalizes)
- All 15 domains have composed/base < 1.0 (no domain regresses below base)

### Remaining minor weaknesses (not blocking)
1. **Single seed**: Justified by multiseed CV=0.5% at N=5, but the scaling behavior itself (N=5 to N=15) is untested across seeds. Not blocking at micro scale.
2. **Short sequences for some domains**: openbookqa averages 54 chars. Noted in Limitations.
3. **Physics outlier**: composed/base = 0.866 means physics still benefits from composition, but the 3.14x composed/individual ratio is large. The paper correctly attributes this to dilution under 1/15 weighting, not interference.

### The K3 "FAIL" framing is now well-handled
The paper makes a clear distinction: K3 fails under the strict kill criterion (medical +15.06% > 10%), but the mechanism is dilution (predictable, bounded), not interference (which would be a real composition failure). The HYPOTHESES.yml status ("supported" with caveat) correctly reflects this nuance. The paper explicitly notes that K3 will inevitably fail for ALL domains at sufficiently large N under uniform weighting, confirming that per-input routing is mandatory -- which is the production design.

## Hypothesis Graph Consistency

- Status: "supported" -- consistent with K1 PASS, K2 PASS, K3 FAIL-with-caveat
- Blocks: exp_bitnet_scale_n25 and exp_bitnet_basefree_exploration -- both unblocked by "supported" status
- Notes: Accurately describe the K3 caveat and routing requirement
- No graph inconsistencies remain

## Macro-Scale Risks (advisory)

1. **Composed/base ratio will approach 1.0 at large N**: Under uniform 1/N weighting, adapter signals become negligible. At N=50, each adapter contributes 2% of total signal. The composed model will be indistinguishable from base. This is expected and validates the routing requirement, but macro experiments should not use uniform weighting as their primary condition.

2. **Cosine metric may not detect functional interference at macro**: The medical-health pair (cosine 0.000268) demonstrates that geometric orthogonality does not imply functional independence. At scale with hundreds of adapters, functional interference metrics (per-domain PPL under targeted composition) should replace or supplement cosine.

3. **K1 metric needs redesign for N=25+**: The composition ratio (avg_composed/best_individual) will grow unboundedly with domain count. Macro experiments should use composed/base ratio or per-domain composed/individual ratios as primary metrics.

## Verdict

**PROCEED**

All 5 requested revisions from the first review have been adequately addressed. The composed/base ratio is now the primary metric with clear explanation. The medical-health overlap is analyzed with honest acknowledgment of limitations. The full 15-domain table is present. The HYPOTHESES.yml status is consistent with the graph. No mathematical errors, no broken mechanisms, no missing controls.

The core finding is sound: ternary composition on BitNet-2B scales from N=5 to N=15 with predictable dilution-driven degradation, no packing pressure, and no domain regressing below base. The K3 failure under uniform weighting correctly identifies routing as mandatory for production, which is the SOLE architecture design.
