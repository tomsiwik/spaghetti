# Peer Review: Minimal Graft Recalibration

## NotebookLM Findings

Skipped -- the experiment is straightforward enough that deep review via NotebookLM would not surface additional issues beyond direct analysis.

## Mathematical Soundness

**Gate classification (Section 1.2): Correct.** The decomposition into root / graft-point / deep categories follows directly from the binary tree topology and the grafting protocol. No hidden assumptions.

**Parameter counting (Section 2.2): Correct.** Each gate is Linear(d, 1, bias=True) = 65 params. 4 layers times gate count gives the stated totals (260, 780, 1820). Verified by the unit tests, which assert these exact numbers.

**Cost ratios (Section 2.3): Correct.** 3/7 = 0.429 and 1/7 = 0.143 are arithmetic facts.

**Depth generalization (Section 6): Correct but incomplete.** The formula 3/(2^D - 1) assumes grafting always occurs at depth 1 (immediately below root). For deeper grafts or multi-level grafting (e.g., grafting at depth 2 in a depth-5 tree), the graft-point set changes. The paper acknowledges this in limitations ("variable-depth grafting"), so this is not blocking.

**Interface mismatch hypothesis (Section 1.3): Reasonable but informal.** The claim that "distribution mismatch concentrates at the root-to-subtree boundary" is stated as an intuition, not derived. This is acceptable for a micro experiment -- the empirical test IS the derivation. The result (+0.19% gap) supports the hypothesis directionally.

**Quantitative prediction (Section 4.1): Inconsistency with actual results.** MATH.md cites parent experiment numbers of "+2.42% root-only" and "+0.67% all-gates" (both vs weight averaging). But PAPER.md reports this experiment's numbers as "+1.27% root-only" and baseline (all-gates) vs all-gates. The metrics are measured differently (parent: vs weight averaging; this experiment: vs all-gates). This is not an error per se, but the prediction section in MATH.md conflates the two reference frames. The prediction "root+graft-point gap ~= 0.67% + epsilon" is in the parent's metric space, while the result (+0.19% vs all-gates) is in a different metric space. The prediction is not falsified but also not cleanly confirmed because the measurement changed.

## Novelty Assessment

**Prior art check: No direct prior art found in references/.** The REFERENCES.yml does not contain entries specifically about selective gate recalibration in hierarchical MoE trees. Jordan & Jacobs (1994) HME is correctly cited as the foundational topology.

**Is this novel?** Moderately. The idea that interface gates matter more than interior gates after grafting is intuitive and likely not surprising to anyone who has worked with tree-structured mixtures. However, the specific quantification (3/7 gates capture 85% of improvement, savings scaling as 3/(2^D-1)) is a useful practical result. The novelty is in the protocol, not the insight.

**Delta over parent experiment:** Incremental. The parent (subtree_grafting) already showed root-only is insufficient and all-gates works. This experiment fills in the middle point (root+graft-point). The experimental delta is one additional condition tested.

## Experimental Design

**Does it test what it claims? Yes, cleanly.** The three conditions share the same grafted model, differ only in which gates are unfrozen, use matched step budgets (100 steps), and are evaluated on the same val sets. This is a well-controlled ablation.

**Potential confound: matched steps vs matched compute.** All conditions run 100 calibration steps, but they train different numbers of parameters per step. Root-only updates 260 params per step; all-gates updates 1820. At 100 steps, root-only has seen 100 gradient updates to its 1 gate, while all-gates has seen 100 gradient updates to all 7 gates. The per-gate step budget is identical across conditions, which is the right comparison for the stated hypothesis. No confound here.

**Potential confound: optimizer state.** Each condition creates a fresh Adam optimizer. This is correct -- reusing optimizer state from a different gate set would be invalid.

**Missing control: calibration convergence.** 100 steps may be insufficient for all conditions to converge. If root-only has not converged but all-gates has, the gap may be a convergence artifact rather than a capacity limitation. A convergence check (loss plateau) would strengthen the result. However, the paper's per-seed data shows root+GP is quite consistent (spread 0.0054), suggesting reasonable convergence for that condition.

**The elephant in the room: grafting loses to weight averaging.** All three grafting conditions (+2.05% to +3.35% vs weight averaging) are worse than the simple baseline. The experiment optimizes the calibration protocol for a composition method that is already dominated. The paper acknowledges this (Finding 4) and frames grafting's value as "2x cheaper fine-tuning," which is fair. But the "99.8% of all-gates quality" headline claim obscures this -- 99.8% of a method that itself is 2% worse than the simpler alternative.

**The "99.8%" claim is misleading.** It is computed as 1 - (0.19%/100%) = 99.8%, i.e., the absolute quality ratio. A more informative metric is stated elsewhere in the paper: root+GP captures 85% of the root-to-all-gates improvement. The 99.8% number, while technically correct, inflates the perceived significance. For comparison, a random baseline achieving 98% absolute quality would also sound impressive despite being poor. The HYPOTHESES.yml evidence field uses this 99.8% framing, which should be revised.

**Kill criteria are reasonable and well-matched.** The 3% and 1.5% thresholds for root-only and root+GP respectively are pre-registered in HYPOTHESES.yml and match what is tested.

## Macro-Scale Risks (advisory)

1. **Depth-1 grafting assumption may not hold.** At macro scale, grafting might occur at arbitrary depths (not just depth 1). The cost analysis assumes the graft-point set is always {1, 2}, but multi-level grafting changes this. The savings formula 3/(2^D-1) would need revision.

2. **More diverse domains may shift the interface.** Character names a-m vs n-z are extremely similar. With truly distinct domains (code vs prose), the "deep gates need no recalibration" finding may not transfer -- the input distribution shift could propagate deeper into subtrees.

3. **Interaction with other composition methods.** If grafting is combined with Huffman shaping (variable-depth trees) or splay routing, the clean root/graft-point/deep classification may break down.

4. **Absolute numbers are not competitive.** Even the best grafting result (+2.05% vs weight averaging) is worse than weight averaging, which itself is +0.43% vs joint. At macro scale, a 2% gap may be unacceptable depending on the application.

## Verdict

**PROCEED**

The experiment is well-designed, the math is correct, the controls are adequate, and the kill criteria are properly pre-registered and survived. The result is directionally useful: it confirms that selective gate recalibration is sufficient after grafting, with clear cost savings that scale exponentially with tree depth.

Reservations that do not block PROCEED:

1. The "99.8% of all-gates quality" framing in the HYPOTHESES.yml evidence field should be revised to "85% of the root-to-all-gates improvement at 43% parameter cost" or simply "+0.19% vs all-gates (within 1.5% threshold)." The 99.8% absolute ratio is technically correct but misleadingly impressive.

2. The experiment is incremental over the parent (one additional ablation condition). This is appropriate for a micro experiment but should not be treated as a major finding.

3. The broader context matters: this optimizes a method (grafting) that loses to the simpler baseline (weight averaging). The practical value is contingent on grafting's cost advantage (2x cheaper fine-tuning) mattering more than its quality disadvantage (+2% vs weight averaging) at macro scale. This is an open question for macro validation, not a flaw in the micro experiment.
