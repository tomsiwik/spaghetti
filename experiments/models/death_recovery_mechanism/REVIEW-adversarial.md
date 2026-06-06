# Peer Review: Death Recovery Mechanism (Exp 20) -- Post-Revision

## NotebookLM Findings

Skipped per time constraints. This is a post-revision review based on direct
reading of MATH.md, PAPER.md, the implementation code, HYPOTHESES.yml,
ADVERSARIAL_REVIEW.md, VISION.md, and the first-round review.

---

## Review of Applied Fixes

The first-round review issued 6 required fixes. Here is the status of each:

### Fix 1: Verify and document embedding freeze status
**ADDRESSED.** MATH.md Section 3.3 now explicitly states that
`freeze_specific_mlp_layers()` calls `model.freeze()` first, which freezes
ALL parameters including wte, wpe, norm0, and lm_head. PAPER.md adds a
dedicated "Embedding freeze status" paragraph. I verified the code: line 65
calls `model.freeze()` (MLX's nn.Module.freeze, which freezes all params),
then lines 67-68 unfreeze only capsule pools, then lines 70-71 re-freeze
specified layers. This matches the `_freeze_attention` pattern from
test_composition.py (line 64). The claim that x^0 = embed(tokens) is fixed
is correct. No embedding drift confound.

### Fix 2: Explain the revival rate discrepancy
**ADDRESSED.** MATH.md Section 3.5 provides a thorough two-part explanation:
(a) offsetting new deaths (baseline L1 creates 7.0 A->D transitions vs 0.3
when frozen), and (b) different anchor dead sets (|D^1_100| = 28 in
train_only_L0 vs 94.7 in baseline). PAPER.md includes a "New Death Analysis"
table and "Net Revival" table confirming the explanation with data. The net
revival formula in MATH.md (r_net including alive->dead offsets) is correctly
specified. This is a satisfactory explanation.

### Fix 3: Report |D^l_100| dead counts per condition
**ADDRESSED.** PAPER.md includes a full "|D^l_100| Per Condition" table with
all 8 conditions and 4 layers. The code (lines 276-293) prints these counts
in the analysis output. The table reveals important context: train_only_L0
has only 28 L1 dead vs 95 in baseline, which directly enables the reader to
assess small-denominator effects on the 95.7% revival figure.

### Fix 4: Separate upstream vs downstream in self-revival code
**ADDRESSED.** Lines 378-409 now explicitly iterate over upstream (other_l <
layer) and downstream (other_l > layer) layers separately, printing distinct
"Revival in upstream layers" and "Revival in downstream layers" lines. The
PAPER.md Table 2 (single-layer training) correctly shows separate upstream
and downstream columns.

### Fix 5: Soften language from "confirmed" to "strongly supported"
**ADDRESSED.** PAPER.md uses "strongly supported" in the result summary and
notes "n=3 seeds, no significance test; directional evidence with large
effect sizes." The kill criterion check in code (line 489-493) uses the same
phrasing. No instance of "confirmed" found in claims about the mechanism.

### Fix 6: Exclude L0 from self-revival reporting
**ADDRESSED.** Code line 385 iterates `range(1, N_LAYERS)`, skipping L0.
Lines 411-419 print an explanatory note about L0 exclusion with the per-seed
dead counts. PAPER.md notes L0 exclusion with the explanation (one seed has
0/0 denominator, mean 20.3 dead which is too variable for reliable rates).

---

## Mathematical Soundness

### What holds

1. **The causal intervention logic is correct.** Freezing upstream MLP layers
   is a valid way to block the inter-layer coupling pathway. With embeddings
   and attention also frozen (verified), frozen upstream MLP layers truly
   produce fixed outputs. The only remaining input change to layer l comes
   from layer l's own alive capsules feeding back through the residual stream,
   which is the self-revival path.

2. **The dead neuron gradient argument is standard and correct.** ReLU with
   pre-activation <= 0 on all inputs produces zero gradient. The detector
   vector a_i^l cannot change through gradient descent. Revival must come
   from input distribution shift.

3. **The net revival formula (Section 3.5) is correctly specified.** It
   accounts for both D->A (revival) and A->D (new death) transitions,
   making cross-condition comparisons more meaningful.

4. **The kill criterion is well-specified, falsifiable, and correctly
   implemented.** Lines 467-494 check whether any of layers 1-3 shows
   revival reduction exceeding 5pp OR freeze revival below 50% of baseline.
   The OR logic matches MATH.md Section 5.

### Remaining issues (non-blocking)

5. **The norm2 self-revival characterization is improved but still imprecise.**
   MATH.md Section 3.3 now calls it "a first-order effect through a
   within-layer pathway" rather than a "second-order numerical artifact."
   This is more accurate. However, the 2-8% residual revival is attributed
   to this path without direct evidence -- it could also come from profiling
   noise (Exp 12 showed 2.6-3.8% measurement disagreement, which overlaps
   with the 2-8% range). The paper should note that the residual revival
   is within the noise floor established by Exp 12, so it may not represent
   genuine self-revival at all. This does not change the conclusion but
   affects interpretation of the mechanism's completeness.

6. **The "79-94% reduction" range is computed from means of 3 seeds.** The
   per-seed variance is honestly reported (e.g., L3 baseline: 37.6% +/- 23.1%),
   but the reduction percentage itself is a ratio of means, not a mean of
   ratios. With such high variance on L3 baseline (23.1% std on 37.6% mean,
   a CV of 61%), the 79% reduction figure for L3 is less stable than it
   appears. Computing reduction per-seed and reporting the range would be
   more honest. This is a presentation issue, not a validity issue -- even
   the worst-case seed likely shows substantial reduction given the L3 frozen
   revival is only 7.8% +/- 3.3%.

---

## Novelty Assessment

Unchanged from first review. The delta is modest but real: first direct
causal test of inter-layer coupling as a revival mechanism. Gurbuzbalaban
et al. (2024) observed revival but never isolated the mechanism via freezing.
ReDo (2024) focused on reinitializing dead neurons, not understanding natural
revival. No reference in REFERENCES.yml covers this specific intervention.

---

## Experimental Design

### Strengths (confirmed by revision)

1. **Complementary conditions provide converging evidence.** The
   freeze-upstream conditions show revival suppression (79-94% reduction).
   The train-only-one conditions show downstream-only revival (0% upstream).
   Both point to the same mechanism.

2. **The discrepancy explanation is now convincing.** The new death analysis
   table provides direct evidence that baseline L1 creates ~7.0 new dead
   capsules vs ~0.3 when frozen. Combined with the smaller denominator
   explanation, the apparent paradox (fewer trainable layers = more revival)
   is fully resolved.

3. **The anchor dead count table enables proper interpretation.** Readers
   can now see that train_only_L0's 95.7% revival in L1 applies to 28 dead
   capsules, not 95.

### Remaining weakness (non-blocking)

4. **No direct test of optimizer momentum as confound.** The experiment
   rules out inter-layer coupling as non-dominant by showing freezing
   suppresses revival. But it does not directly test whether the residual
   2-8% is from optimizer momentum vs norm2 self-revival vs profiling noise.
   This is acknowledged implicitly (the paper lists all three as possible
   sources) and is not critical -- the point is that the residual is small
   regardless of source.

---

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp20_death_recovery_mechanism` correctly:
- Has status: proven
- Lists the kill criterion: "freezing upstream layers does NOT reduce revival
  in downstream layers" (not triggered)
- Records both the original evidence and the review-revision evidence
- Notes all 6 fixes applied

The evidence entry accurately summarizes the revised findings including the
specific numbers and the discrepancy explanation.

---

## Macro-Scale Risks (advisory)

1. **SiLU/SwiGLU irrelevance.** This entire analysis is ReLU-specific.
   Exp 15 showed SiLU models have 0% truly dead neurons. If the macro
   architecture uses SiLU (as most modern LLMs do), inter-layer coupling
   revival is moot. The finding remains useful for understanding ReLU-based
   capsule pools but may not apply to the final macro architecture.

2. **Coupling attenuation with depth.** At 24-32 layers, does L0 still
   revive L24? The micro result (L0 revives L3 through frozen intermediaries
   at 81-98%) is suggestive of strong long-range coupling, but 4 layers is
   not 32. The residual stream norm dynamics may differ at scale.

3. **Attention unfreezing changes the picture.** All conditions froze
   attention. In full training, attention weight updates provide an
   additional inter-layer coupling pathway. The headline finding
   characterizes "MLP inter-layer coupling with attention frozen" but the
   macro protocol may unfreeze attention, making the coupling stronger (or
   weaker if attention changes dominate MLP changes).

---

## Verdict

**PROCEED**

All 6 required fixes from the first review have been adequately addressed:

- Embedding freeze is verified by code inspection and documented in both
  MATH.md and PAPER.md
- The revival rate discrepancy (train_only_L0 vs baseline) is explained by
  offsetting new deaths and smaller denominators, with supporting data tables
- Anchor dead counts are reported per condition
- Self-revival code correctly separates upstream from downstream
- Language appropriately softened to "strongly supported"
- L0 excluded from self-revival with explanation

The core finding stands: upstream MLP layer training drives downstream
revival through the residual stream (79-94% reduction when upstream frozen,
0% upstream revival in single-layer conditions). The experimental design is
sound, the causal intervention is clean, and the remaining issues (norm2
self-revival vs noise floor, per-seed reduction variance) are non-blocking
presentation concerns that do not threaten the conclusion.

The experiment achieves what it set out to do: provide directional evidence
at micro scale that inter-layer coupling is the dominant revival mechanism
for dead ReLU neurons. This is a clean mechanistic finding that informs the
composition protocol (frozen base = stable dead neuron set) and pruning
timing recommendations.
