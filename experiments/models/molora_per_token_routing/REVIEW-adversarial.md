# Peer Review: MoLoRA Per-Token Routing (v2 -- Revision Review)

## Context

This is a revision review. The v1 review issued a REVISE verdict with 6 required fixes. The researcher applied all 6 as documentation updates (no code changes). This review verifies the fixes and checks for new issues.

## Fix Verification

### Fix 1: K1 redefined (per-token vs per-sequence on same 5 domains)

**VERIFIED.** PAPER.md lines 40-45 now define K1 as "Per-token PPL <= per-sequence PPL on same 5 domains" with result "NULL RESULT (7.63 > 7.60, -0.46%)". The invalid 13.65 baseline is explicitly flagged as confounded by at least 4 variables. The K1 note (line 45) provides a clear explanation of why the old comparison was invalid.

### Fix 2: Evaluation methodology flaw documented

**VERIFIED.** PAPER.md lines 74-83 contain a detailed "Evaluation Methodology Limitation" section. It correctly explains the per-group forward pass problem (tokens at positions 0..t-1 conditioned on wrong adapter), identifies the per-token PPL as an approximation (likely upper bound), and notes that the -0.46% gap is within the error range of this artifact. The section also prescribes correct alternatives (single-pass per-layer adapter switching, or causal-consistent evaluation).

### Fix 3: MATH.md "first-token proxy" replaces "average weight"

**VERIFIED.** MATH.md lines 76-78 now clearly state: "Where w_rep is the weight vector of the **first token** assigned to group S, used as a proxy for the group." An explicit implementation note (lines 78) references the code location (run_experiment.py ~line 600-606) and confirms the code uses first-token weights, not a true average. The bounding argument (within-group weight variance is small for confident activations) is present.

### Fix 4: All "44% improvement" claims removed

**VERIFIED.** Grep for "44%" in PAPER.md and MATH.md returns zero matches. The only references to the 13.65 baseline in PAPER.md appear in the K1 note (line 45) and the Gumbel-sigmoid discussion (line 92), both of which explicitly flag the comparison as confounded and invalid. No inflated language ("superior", "dramatically", "breakthrough") found anywhere in the paper.

### Fix 5: Revised kill criteria section in PAPER.md

**VERIFIED.** PAPER.md lines 37-45 contain a properly structured kill criteria table with columns for Criterion, Definition, Metric, Threshold, and Result. K1 is defined with a proper null hypothesis (per-token <= per-seq), K2 and K3 retain their original definitions. The table clearly shows K1 as NULL RESULT, K2 as PASS, K3 as PASS. Success criteria (lines 47-53) are separately defined and correctly show S1 as FAIL.

### Fix 6: FLOPs corrected (327,680)

**VERIFIED.** MATH.md line 98 now reads "2*d*h + 2*h*N = 327,680 + 640". Checking: 2 * 2560 * 64 = 327,680. Correct.

## Check for New Issues

### Claims-Evidence Consistency

The paper now leads with "Informative Null Result" as the primary finding. All three claims in the "What the Gumbel-Sigmoid Router Proved" section are directly supported by experimental evidence:

1. Independent binary gates allow multi-adapter blending -- supported by diversity metric (2.42 groups/seq)
2. 164K router params suffice -- supported by routing accuracy and PPL results
3. 0.58% overhead -- supported by timing measurements

The Gumbel-sigmoid vs softmax paragraph explicitly states the comparison is confounded and no causal claim can be made. This is honest.

### Math-Code Consistency

MATH.md now matches the code. The first-token proxy is documented with a code reference. The pre-merge formulation correctly uses w_rep instead of bar{w}_S. No discrepancies found.

### Framing Integrity

The paper maintains a consistent framing throughout: per-token routing is not better than per-sequence routing on cleanly separated domains, but the mechanism works (low overhead, non-degenerate routing patterns, correct MLX implementation). The limitations section is comprehensive (5 items) and does not attempt to downplay the null result.

### HYPOTHESES.yml

The experiment does not have its own dedicated HYPOTHESES.yml entry (it falls under `exp_bitnet_per_token_routing`). This is acceptable for a micro experiment that produced a null result -- the evidence could be appended to the existing node. Not blocking.

## Mathematical Soundness

Verified in v1, no changes to the math. The router architecture parameter count (164,229), Gumbel-sigmoid formulation, top-k selection, and pre-merge grouping are all correct. The FLOPs arithmetic is now correct.

## Novelty Assessment

Unchanged from v1. The contribution is modest: Gumbel-sigmoid routing on MLX/Apple Silicon with an informative null result comparing per-token vs per-sequence routing. The null result itself is useful for the project's hypothesis graph.

## Experimental Design

Unchanged from v1. The evaluation methodology flaw is now properly documented as a known limitation. The controls (uniform 1/N, per-sequence, oracle) are adequate. The per-domain diversity analysis adds diagnostic value.

## Macro-Scale Risks (advisory)

1. **Mixed-domain data is the real test.** The null result on cleanly separated domains is expected. Per-token routing's value proposition requires sequences where different tokens genuinely need different experts (e.g., code+comments, legal+mathematical formulas). This is the critical macro experiment.
2. **Evaluation architecture.** The per-group forward pass approach is not viable at scale. A single-pass architecture with per-layer adapter switching is needed for production.
3. **Token-level supervision.** Training with per-domain labels teaches per-sequence patterns. True per-token routing needs token-level training signals.

## Verdict

**PROCEED (v2)**

All 6 required fixes have been properly applied. The paper now:
- Leads with the honest null result (per-token not better than per-sequence on clean domains)
- Explicitly flags the confounded cross-experiment comparison as invalid
- Documents the evaluation methodology limitation with error bound discussion
- Maintains math-code consistency (first-token proxy)
- Uses corrected arithmetic throughout
- Makes no claims beyond what the evidence supports

The experiment is correctly positioned as an informative null result that validates the Gumbel-sigmoid routing mechanism on MLX while showing that per-token routing requires mixed-domain data to demonstrate its value. This is useful directional evidence for the project's hypothesis graph.
