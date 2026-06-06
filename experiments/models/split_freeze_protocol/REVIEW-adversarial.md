# Peer Review: Split-and-Freeze Protocol (Re-Review After Revision)

## NotebookLM Findings

Skipped -- manual deep review conducted. The revised documents are sufficiently
detailed to evaluate directly from MATH.md, PAPER.md, code, and the previous
REVIEW-adversarial.md.

## Revision Assessment

The previous review issued REVISE with three required fixes. Assessment of each:

### Fix 1: KC1 relabeled from "split vs from-scratch" to "warm-start vs cold-start"

**Status: Addressed thoroughly.**

MATH.md Section 2.2 now contains a prominent note: "The KC1 experiment did NOT
test this split operation." Section 2.3 is a new section titled "What KC1
Actually Tests: Warm-Start vs Cold-Start" that correctly describes the
experiment. PAPER.md lines 95-109 provide detailed clarification including the
explicit statement that `split_leaf()` was "implemented but was not invoked."
The protocol specification (PAPER.md lines 213-246) distinguishes "WARM-START
(validated) / SPLIT (untested)."

The relabeling is honest and complete. No remaining confusion between the
mathematical split operation and the warm-start experiment.

### Fix 2: subtree_grafting overlap made prominent

**Status: Addressed thoroughly.**

PAPER.md lines 186-207 now contain a dedicated section "Relationship to
subtree_grafting Findings" that:

1. Cites the specific prior result (root-only calibration +2.42%, all-gates
   recalibration sufficient)
2. Cites the minimal_graft_recal refinement (3/7 gates sufficient)
3. Articulates the novel delta: in the freeze-and-graft scenario (vs
   subtree_grafting's compose-by-grafting), the grafted subtree's **leaves**
   (not just gates) must be trainable during calibration
4. Provides the specific numbers: gates-only +2.5% (borderline KILL) vs
   right-tree +0.09% (clean PASS)

This is a genuine contribution beyond subtree_grafting: the frozen scenario
imposes a stronger requirement on calibration scope.

### Fix 3: per-seed V2 numbers documented as not preserved

**Status: Addressed honestly.**

PAPER.md lines 149-157 document the gap: per-seed values were printed at
runtime (confirmed in `run_experiment_v2.py` line 222) but not captured to a
log file. The paper explains why the means are still trustworthy: 3 seeds with
a 2% kill threshold applied per-configuration-mean. Limitation 6 reiterates
the gap and notes a rerun would recover the values.

This is an acceptable documentation-of-a-gap. The `run_experiment_v2.py` code
(lines 218-223) does print per-seed degradation, so a rerun is straightforward.
Not blocking.

## Mathematical Soundness

### KC1 math: Correct for what it claims

The split derivation (MATH.md Section 2.2) correctly shows f_child0 + f_child1
= f_parent when noise is zero. The derivation is now clearly labeled as
theoretical and not tested. The warm-start vs cold-start comparison (Section
2.3) is correctly described with no mathematical claims beyond "same
architecture, same params, same budget."

### KC2 math: Correct

The degradation analysis (Sections 3.2-3.4) correctly identifies three
mechanisms of frozen branch degradation: root gate redistribution, beam
selection competition, and normalization effects. The calibration scope
analysis in Section 3.4 matches the empirical sweep.

The informal bound (degradation = f(root gate classification error on domain
A)) is directionally correct and no tighter bound is claimed.

### Parameter counts: Verified

KC1: 33,028. Matches code (2 leaves x 2 matrices x d x n_c x 4 layers + 1
gate x (d+1) x 4 layers = 4*(8192+65) = 33,028). KC2: 66,576 for right-tree.
Matches code (4 leaves x 2 matrices x d x n_c + 4 gates x (d+1), all x 4
layers = 4*(16384+260) = 66,576).

## Novelty Assessment

### Prior art: Correctly cited

Progressive Neural Networks (freeze old, add new), PackNet (per-neuron
freezing), Fast Feedforward Networks (binary tree FFN). The delta is the tree
lifecycle protocol combining split/freeze/graft.

### Overlap with subtree_grafting: Now properly disclosed

The revised paper honestly states that KC2 "confirms and extends" the
subtree_grafting finding. The novel contribution is clearly scoped: in the
freeze scenario, leaf-level calibration is required, not just gate-level. This
is a legitimate refinement, not a reinvention.

### Warm-start result novelty: Low but acknowledged

The KC1 result (warm-start = cold-start at micro scale, -0.03%) is not novel.
This is a well-known outcome at small scale with sufficient training budget.
The paper correctly positions it as a negative result: "The inherited capsules
provide no advantage but also no disadvantage." The hypothesized macro-scale
advantage (faster convergence from inherited features) remains untested.

## Experimental Design

### KC1: Adequate for the revised claim

With the relabeling, KC1 now tests exactly what it claims: warm-start vs
cold-start on an existing leaf pair. Controls are proper (same architecture,
same trainable params, same budget, same data). 3 seeds with tight per-seed
spread (split: 0.5080-0.5200, scratch: 0.5081-0.5190). The -0.03% gap is
well within noise.

The `split_leaf()` function remains dead code in the experiment. This is
acceptable given the relabeling -- the function is documented as "implemented
but not invoked." It would be a stronger experiment to actually test it, but
the revised claim does not depend on it.

### KC2: Well-designed with dose-response evidence

The V2 diagnostic sweep is methodologically strong:

- 6 configurations testing 3 calibration scopes x 2 step budgets
- 3 seeds per configuration (18 total runs)
- Clear monotonic progression: root-only (+13.3%) > all-gates (+2.5%) >
  right-tree (+0.09%)
- Structural verification: frozen weight drift = 0
- Pre-graft baseline measured per-seed

The dose-response pattern (more calibration scope = less degradation) is
convincing evidence that the mechanism works in principle. The gates-only
result (+2.5%, on the kill boundary) vs right-tree (+0.09%, clean pass)
clearly demonstrates the leaf-trainability requirement.

### Missing control: "unfreeze everything" baseline

Neither PAPER.md nor the code includes a baseline where everything is
unfrozen and retrained. This would measure the cost of freezing vs simply
retraining. The paper acknowledges this as Limitation 4 ("calibration budget
confounds") and correctly notes that the value of freezing is in the
contribution model (structural preservation guarantee), not in compute
savings per se at micro scale.

Not blocking -- the contribution model argument is valid.

## Hypothesis Graph Consistency

### Kill criteria alignment

HYPOTHESES.yml kill criteria:
1. "split branch quality >5% worse than training a new flat expert from scratch"
2. "frozen branches degrade >2% when new branches are grafted alongside"

KC1 tests criterion 1 but via warm-start, not split. The kill criterion text
in HYPOTHESES.yml still says "split branch" despite the KC1 relabeling. The
evidence entry (HYPOTHESES.yml line 832) correctly says "Warm-start provides
no advantage but no disadvantage" but the kill_criteria text has not been
updated to match the relabeling.

**Minor inconsistency**: the kill_criteria text should say "warm-started leaf
quality >5% worse than cold-started" to match the revised experiment, or
note that split was not directly tested. This is a HYPOTHESES.yml housekeeping
issue, not a blocking problem for the experiment itself.

KC2 tests criterion 2 and the conditional pass (right-tree calibration
required) is correctly documented in the evidence. Status "proven" is
appropriate given the clear protocol specification.

## Macro-Scale Risks (advisory)

1. **Calibration cost at scale**: Right-tree calibration uses 66,576 params at
   micro. At d=4096, n_c=256, this could reach tens of millions of params. If
   calibration cost approaches retraining cost, the economic argument for
   freezing weakens. The paper acknowledges this (Limitation 4).

2. **Warm-start advantage may emerge at scale**: KC1's null result (-0.03%)
   may become a positive result at macro scale where random initialization
   takes more steps to converge. The split mechanism (untested) would also
   need validation. This is a genuine open question.

3. **Multi-domain sequential grafting**: With N>2 domains, each graft requires
   recalibrating all unfrozen parameters. The cumulative calibration cost
   could grow superlinearly. The Huffman tree shaping and minimal graft
   recalibration findings may help but are untested in the sequential
   freeze-and-graft scenario.

4. **Attention layer interaction**: The experiment freezes attention during
   grafting. At macro scale with continued training, attention representation
   shifts could indirectly degrade frozen subtree quality. Acknowledged in
   PAPER.md "What Would Kill This" section.

## Verdict

**PROCEED**

All three revision requests have been addressed:

1. KC1 is honestly relabeled as warm-start vs cold-start throughout MATH.md,
   PAPER.md, and the protocol specification. The `split_leaf()` function is
   explicitly documented as implemented-but-untested.

2. The subtree_grafting overlap is now prominent with a dedicated section
   articulating the novel delta (leaf-level calibration required in freeze
   scenario).

3. The per-seed V2 data gap is honestly documented with explanation of why
   the means are still trustworthy.

### What is solid

- KC2 provides convincing dose-response evidence that calibration scope
  determines frozen branch stability. The right-tree calibration finding
  (+0.09%, 3 seeds) is a genuine refinement over subtree_grafting.
- The protocol specification correctly includes the "grafted leaves must be
  trainable" requirement -- this is the actionable contribution.
- Weight drift verification is structural and correct.
- Limitations are honest and comprehensive.
- The experiment builds on existing infrastructure without reinvention.

### Minor housekeeping (non-blocking)

- HYPOTHESES.yml kill_criteria[0] still says "split branch" rather than
  "warm-start vs cold-start." Should be updated for consistency with the
  revised PAPER.md and MATH.md, but this is a metadata issue, not a
  scientific one.
