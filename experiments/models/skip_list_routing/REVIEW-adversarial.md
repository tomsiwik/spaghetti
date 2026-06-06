# Peer Review: Skip-List Multi-Resolution Routing (Revision 1 Re-Review)

## NotebookLM Findings

Skipped -- the mathematical structure and revision changes are transparent enough
for direct review. The revision is primarily a documentation/framing correction
plus one new experimental control (ensemble).

## Mathematical Soundness

### What holds

1. **Stick-breaking weights sum to 1**: Correct. Standard Dirichlet process
   stick-breaking. The telescoping identity is valid. Implementation in
   `__call__` (lines 162-180 of skip_list_routing.py) correctly accumulates
   `cum_pass_through`. Unit test `test_level_usage_sums_to_one` covers this.

2. **Parameter overhead calculation**: d*(2N-1) + L*(d+1) is correct. For d=64,
   N=8: reported +1.3% (2816 extra params over 204,160) checks out.

3. **Level structure**: ceil(N/2^k) hierarchy producing [8, 4, 2, 1] for N=8
   is correctly implemented in `_level_size`.

4. **Coarse expert construction**: Recursive weight-averaging in
   `_get_coarse_expert_output` (lines 126-148) correctly averages children.
   The recursion terminates at level 0 with actual expert parameters.

5. **Training cost analysis (new)**: MATH.md now correctly states 32 leaf expert
   forward passes per token (N*(L+1) = 8*4 = 32) vs 2 for flat top-k=2, yielding
   16x higher expert evaluation cost. This is arithmetically correct and honestly
   reported.

### Residual concern (non-blocking)

6. **Hypothetical routing cost calculation**: The MAD calculation in MATH.md
   (284 vs 512 MADs, "~45% reduction") is internally consistent but applies to
   an unimplemented hard routing mode. The paper correctly flags this as
   hypothetical. No change needed, but the HYPOTHESES.yml evidence still reads
   "64.0% routing depth reduction" which could be misread as actual savings.
   This is a metadata hygiene issue, not a paper issue.

## Novelty Assessment

### Prior art

The mechanism is a multi-resolution ensemble with learned stick-breaking depth
weighting. The "skip list" analogy remains loose -- actual skip lists enable
O(log N) search by skipping elements, while this model computes all levels and
blends them. However, the analogy is reasonable for the *structure* (geometric
spacing of levels with express lanes), and the paper does not overclaim.

Relevant prior art:
- **Stick-breaking processes** (Sethuraman 1994): The confidence gates are
  exactly stick-breaking weights. Cited in PAPER.md references.
- **Mixture of Depths** (Raposo et al. 2024): Per-token adaptive computation.
  Cited and correctly distinguished (layer-level vs within-layer routing).
- **Hierarchical softmax** (Morin & Bengio 2005): Not cited. Hierarchical
  softmax organizes outputs in a tree for efficient normalization. The analogy
  is structural, not functional, so absence is acceptable.
- **Feature Pyramid Networks** (Lin et al. 2017): Multi-scale fusion in vision.
  Different domain, similar structural intuition. Not cited, acceptable.

The delta over the project's own hierarchical_tree is genuine: learned soft
level selection vs fixed full-depth traversal. This is a meaningful contribution
within the research program.

### Reinvention check

No existing code in `references/` implements this mechanism.
`references/skip-list-routing/` contains only a README. No reinvention detected.

## Experimental Design

### What the revision fixed correctly

1. **Ensemble confound (original Fix #2)**: The 4x ensemble control is well-
   designed. Result: ensemble +0.59% worse than single flat, skip adaptive
   -1.51% better than ensemble. This cleanly rules out simple output averaging
   as the explanation. The ensemble has 4x parameters (816,640) yet still loses,
   which strengthens the argument that shared expert structure (coarse experts
   reusing leaf parameters) provides genuine value.

2. **Routing stats on validation set (original Fix #4)**: Now measured over
   20,480 tokens per layer per seed (20 batches * 32 batch_size * 32 seq_len),
   aggregated across 3 seeds and 4 layers = 12 measurements. Standard deviations
   reported. This is adequate for a micro experiment.

3. **Parameter counts (original Fix #5)**: All variants now report trainable and
   total counts. The 780-param confound between adaptive and fixed-depth is
   explicitly acknowledged with appropriate hedging ("0.38% difference ... should
   be noted when interpreting the 0.60% quality gap").

4. **Routing cost claims (original Fix #1)**: MATH.md and PAPER.md now contain
   prominent caveats distinguishing level-weight concentration from actual FLOP
   savings. The "Important Caveat" section in MATH.md and the clarification boxes
   in PAPER.md are clear and honest.

5. **Recursive computation cost (original Fix #3)**: Limitation #4 in PAPER.md
   now correctly identifies 32 leaf expert forward passes per token vs 2 for
   flat top-k=2 (16x factor). This is the most important disclosure.

### Remaining minor issues (non-blocking)

1. **Double router evaluation**: In `SkipListCapsulePool.__call__`, when
   `n_at_level > top_k`, the router is called twice (line 191 for probs,
   line 199 for scores). Mathematically equivalent but wasteful. This was noted
   in the original review and not fixed. It does not affect correctness or
   results -- it is a code quality issue only.

2. **The 67.2% coarsest-level weight interpretation**: The paper frames this
   as "most tokens handled here" and "confidence gates learn token difficulty."
   An alternative interpretation remains valid: at micro scale with character-
   level names, experts are insufficiently specialized, so averaging them
   produces a decent generalist -- the model discovers that individual expert
   selection adds little value. Both interpretations are consistent with the
   data. The paper's Analysis section does acknowledge this ("consistent with
   the LSH finding that all routing strategies equivalent at small G"), which
   is fair. The Limitations section (#1) further notes that macro-scale diverse
   text may produce a broader distribution. This is adequately hedged.

3. **Fixed-depth control is not perfectly param-matched**: The 780-param
   difference (0.38%) is acknowledged but not resolved. A tighter control
   would add those 780 params elsewhere. Given the magnitude (0.38% params
   vs 0.60% quality gap), the confound is plausibly but not certainly
   negligible. The paper's hedging is appropriate.

## Hypothesis Graph Consistency

The experiment targets `exp_skip_list_multiresolution_routing` with kill criteria:
- KC1: skip-list routing >2% worse than flat softmax -- passes at -0.93%.
- KC2: adaptive depth doesn't reduce average routing cost vs fixed depth --
  passes via level-weight concentration (60.6% above Level 0).

KC2 as stated ("average routing cost") is ambiguous. The level-weight
interpretation passes. The actual FLOPs interpretation would fail (training
is 16x more expensive). The HYPOTHESES.yml evidence entry still reads
"64.0% routing depth reduction" which could mislead readers unfamiliar with
the caveat. Recommend updating to "64.0% level-weight concentration above L0
(not FLOP savings)" for clarity. This is metadata hygiene, not a blocking issue.

## Macro-Scale Risks (advisory)

1. **Coarse experts become poor with diverse experts**: Averaging a code expert
   and a poetry expert produces gibberish. The 67% coarsest-level weight would
   collapse toward zero at macro scale, potentially eliminating the routing
   benefit. Test with genuinely dissimilar expert domains.

2. **O(N * L) training cost is intractable at scale**: N=256, L=8 would require
   2048 leaf expert evaluations per token. Hard routing must be implemented and
   validated before macro experiments.

3. **GPU batch parallelism**: Different tokens stopping at different levels
   creates irregular computation. Padding to max depth negates savings. This
   needs architectural solutions (e.g., sorting tokens by predicted depth).

4. **Soft-hard gap**: Training with soft blending but deploying with hard
   early stopping introduces a distribution shift. The gap is bounded by the
   concentration of level weights (tighter concentration = smaller gap), but
   this bound is not formally derived or tested.

## Verdict

**PROCEED**

The revision adequately addresses all five issues raised in the original REVISE
verdict:

| Original Issue | Status |
|---|---|
| Routing cost claims misleading | Fixed: clear caveats in MATH.md and PAPER.md |
| Ensemble confound uncontrolled | Fixed: 4x ensemble tested, skip wins by -1.51% |
| Recursive computation cost hidden | Fixed: 16x factor disclosed in Limitations |
| Routing stats on single sequence | Fixed: 20K tokens/layer/seed on val set |
| Param counts unreported | Fixed: all variants reported with confound acknowledged |

The mechanism is mathematically sound (stick-breaking level weights, correct
hierarchy). The kill criteria are passed: -0.93% better than flat (threshold
was >2% worse), 60.6% level-weight concentration above Level 0. The ensemble
confound is experimentally refuted. The paper is honest about what was and was
not measured -- the 16x training cost, the hypothetical nature of routing cost
savings, and the parameter mismatch in the fixed-depth control.

The remaining issues (double router call, metadata hygiene in HYPOTHESES.yml,
imperfect param matching of fixed-depth control) are non-blocking code quality
and metadata items. The core contribution -- learned stick-breaking level
selection for multi-resolution expert routing that matches or beats flat routing
quality while learning meaningful level-weight distributions -- is validated at
micro scale.

The serious risks are all macro-scale (coarse expert quality with diverse
domains, O(N*L) training cost, soft-hard gap, GPU parallelism). These should
be tested in macro experiments, not at micro scale.
