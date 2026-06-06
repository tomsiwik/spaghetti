# Peer Review: Subtree Grafting Composition

## NotebookLM Findings

Skipped -- the experiment is straightforward enough that a manual deep review suffices. The math is simple (set partitioning + weight averaging vs. assignment), and the code is short.

## Mathematical Soundness

### What holds

1. **Partition completeness (Section 1.3)**: The decomposition of a depth-3 binary tree into root + left subtree {1,3,4} + right subtree {2,5,6} is correct. Gate indices and leaf indices partition cleanly. The QED is valid.

2. **Sigmoid non-commutativity (Section 2.1)**: The claim that sigma((w_A+w_B)/2 * x) != (sigma(w_A*x) + sigma(w_B*x))/2 is correct by Jensen's inequality (sigma is neither convex nor concave globally, but the inequality holds generically). This is the correct motivation for why weight averaging distorts routing decisions.

3. **Parameter counts (Section 3)**: Arithmetic checks out. 7*65 = 455, 8*2*64*32 = 32768, total 33223 per layer. Subtree: 3*65 = 195, 4*2*64*32 = 16384, total 16579. The 49.9% reduction claim is correct (16579/33223 = 49.9%).

4. **Leaf probability decomposition (Section 4.2)**: P(leaf=l|x) = P(subtree(l)|x) * P(leaf=l|subtree(l),x) is a correct application of the chain rule of probability, given the tree structure.

### What does not hold (or is misleading)

5. **"Function-space gap absent within subtrees" (Section 2.3 table)**: This claim is only true BEFORE calibration. The paper acknowledges in Section 4 of PAPER.md (Limitation 4) that all-gates calibration drifts internal gates from their fine-tuned values. But the MATH.md table presents "Absent within subtrees" without qualification. After all-gates calibration, the internal gates ARE modified, and the function-space gap is reintroduced within subtrees. The table should say "Absent before calibration; partially reintroduced during all-gates calibration."

6. **"Routing decisions: Exact per subtree" (Section 2.3 table)**: Same issue. Only exact before calibration. After 100 steps of all-gates calibration on mixed data, internal gates have drifted. The experiment's own finding undermines this selling point.

7. **The core theoretical argument is self-defeating**: The paper argues grafting is better because it preserves routing decisions. But the experiment shows root-only calibration (which actually preserves internal routing) performs poorly (+2.42%). All-gates calibration is needed (+0.67%), which modifies the internal routing the paper claims to preserve. The mechanism that was supposed to make grafting superior is the mechanism that needs to be undone for grafting to work. The PAPER.md acknowledges this honestly in its findings, but the MATH.md does not.

### Hidden assumptions

8. **Domain-subtree assignment is arbitrary**: The paper assigns domain A to the left subtree and domain B to the right. But after pretraining on mixed data, the left and right subtrees may already have developed specializations that do not align with the a-m / n-z split. The experiment does not test whether assignment matters (e.g., swapping left/right) or whether the pretrained base's existing subtree specializations help or hurt.

## Novelty Assessment

### Prior art

- **Hierarchical Mixtures of Experts (Jordan & Jacobs, 1994)**: The PAPER.md cites this correctly. HME already uses tree-structured gating. The grafting composition protocol (assigning subtrees to domains post-hoc) is a novel extension of HME, not a reinvention.

- **TIES Merging / DARE Merging**: Cited correctly as alternatives that address parameter blending. Grafting sidesteps blending entirely, which is a genuinely different approach. Neither TIES nor DARE operates on tree-structured models.

- **Fast Feedforward Networks (Belcak & Wattenhofer, 2024)**: Cited for tree decomposition. Not directly comparable (FFN routing vs. expert composition).

- **No direct prior art found for subtree grafting as a composition protocol for tree-structured MoE.** The idea of assigning tree branches to domains and composing by concatenation is novel within the MoE composition literature. However, the idea is simple enough that its novelty is primarily in the testing, not the concept.

### Delta over existing work

The delta is small. Weight averaging already achieves +0.27% vs joint. Grafting achieves +0.94% vs joint. Grafting is worse, not better. The practical contribution is the 2x fine-tuning cost reduction, which is legitimate but modest.

### References check

The experiment correctly builds on `hierarchical_tree` and reuses its architecture. No reinvention of existing code in `references/`. The `SubtreeGraftingGPT` class is a thin wrapper (correct approach).

## Experimental Design

### Does the experiment test the stated hypothesis?

**Yes, adequately.** The hypothesis is: "grafting will match or beat weight averaging because it preserves routing decisions." The experiment tests this with matched calibration budgets (v2), 3 seeds, and per-domain evaluation. The result is a clear negative on the "beat" part and a clear positive on the "match" part (within 3% threshold).

### Controls

1. **Joint training baseline**: Present and correctly implemented. Joint model trains for pretrain+finetune steps (500 total), matching the total training budget of the composition methods.

2. **Matched calibration budget (v2)**: Correctly addresses the v1 confound. Both methods get 100 steps of all-gates calibration. This is a well-designed fix.

3. **Per-domain evaluation**: Present. Tests kill criterion 2 (donor preservation).

### Concerns with experimental design

4. **Same random seed for both domain fine-tunes**: In `run_experiment_v2.py` line 149 and 186, both domain-specific models are initialized with `mx.random.seed(seed)` and then loaded with base_weights. This means both start from identical states. For weight averaging, both models fine-tune the FULL tree (same architecture, same init, different data). For grafting, they fine-tune different subtrees. This is correct -- the seed controls the optimizer's random sampling, not the model init (which comes from base_weights). No issue here.

5. **Weight averaging calibrates all gates; grafting also calibrates all gates (v2)**: This is the correct matched comparison. But it means the experiment is really comparing "blend all params then recalibrate gates" vs "graft subtrees then recalibrate gates." Since both recalibrate all gates, the remaining difference is only in the leaf weights (blended vs. preserved). The gate recalibration equalizer makes this primarily a test of whether blended vs. domain-specific leaf weights matter.

6. **No confidence intervals**: With 3 seeds and no standard deviations reported, we cannot assess whether the +0.67% gap is statistically significant. Looking at per-seed values: weight_avg = [0.5134, 0.5284, 0.5248], graft = [0.5205, 0.5284, 0.5282]. The gap on seed 42 is 0.0071, on seed 123 it is 0.0000, on seed 777 it is 0.0034. The variance is high relative to the mean gap. The +0.67% could easily be noise. This does not invalidate the "passes kill criterion" conclusion (the threshold is 3%, and even the worst seed is well within that), but it does mean the directional claim "weight averaging is slightly better" may not be robust.

7. **v1 result included as evidence of a "calibration artifact"**: The v1 experiment used a genuinely unfair comparison (root-only 50 steps vs all-gates 100 steps). The paper correctly identifies this and runs v2 to fix it. The diagnostic sweep (Section "Diagnostic: Calibration Budget Sweep") is well-designed and informative. Good scientific practice.

### Hypothesis graph consistency

The experiment matches its HYPOTHESES.yml node (`exp_subtree_grafting_composition`). Kill criteria are:
- "subtree grafting composition >3% worse than weight averaging composition" -- tested, passes at +0.67%
- "grafting produces >5% degradation on the donor subtree's original domain" -- tested, passes at +1.34% max

Status is "proven" which is correct for "survived kill criteria" even though the hypothesis direction (grafting beats averaging) was not confirmed.

There is a duplicate node: `exp_subtree_grafting` (line 919) with identical evidence. This should be cleaned up -- one node, not two.

## Macro-Scale Risks (advisory)

1. **N>2 domains**: The binary subtree assignment is fundamentally limited to N=2 (or N as a power of 2). The paper acknowledges this. At macro scale with 5-20 domains, this requires either (a) much deeper trees with unequal subtree sizes, or (b) a different approach entirely. The Huffman tree experiment addresses (a) but with its own limitations.

2. **All-gates calibration cost at scale**: The finding that all-gates calibration is necessary means grafting does not actually reduce calibration cost compared to weight averaging. The only savings are in fine-tuning (each domain trains half the tree). At macro scale, if calibration is the bottleneck (not fine-tuning), the practical advantage disappears.

3. **Subtree capacity ceiling**: Each domain gets L/2 leaves. At D=5 (32 leaves), each domain gets 16 -- probably adequate. At D=3 (8 leaves), 4 leaves per domain may be too few for complex domains. The capacity concern is inversely related to tree depth.

4. **The "routing preservation" argument gets weaker with more calibration**: The diagnostic shows the gap shrinks with more calibration steps (200 steps: +1.34%). With enough calibration, the gates are fully retrained and the grafting/averaging distinction is purely about leaf weights. At macro scale with longer calibration, this distinction may vanish entirely.

## Verdict

**PROCEED**

The experiment is well-designed, honestly reported, and passes its stated kill criteria. The researcher correctly identified the v1 confound, ran diagnostics, and fixed it in v2. The findings are nuanced and the limitations are acknowledged.

Specific strengths:
- Good iterative methodology (v1 failure -> diagnostic -> v2 fix)
- Matched calibration budgets in v2
- Honest reporting that the core hypothesis (routing preservation superiority) is not confirmed
- Correct identification that the practical value is in fine-tuning cost, not composition quality

Issues that do not block PROCEED but should be noted:
1. The MATH.md table in Section 2.3 should qualify "Absent within subtrees" and "Exact per subtree" with "before calibration only" to match the actual experimental findings.
2. The duplicate HYPOTHESES.yml node (`exp_subtree_grafting_composition` at line 621 and `exp_subtree_grafting` at line 919) should be consolidated to one entry.
3. No confidence intervals / standard deviations reported. With 3 seeds, the +0.67% gap may not be statistically significant. The kill criterion is comfortably passed regardless, but the directional claim should be hedged.
4. The domain-subtree assignment was not ablated (swapping left/right). This is a minor gap -- at micro scale with similar domains, it likely does not matter, but it is an untested assumption.

The experiment advances the project by establishing that subtree grafting is a viable (if not superior) composition method for tree-structured experts, with the practical benefit of 2x cheaper per-domain fine-tuning. The finding that all-gates recalibration is necessary is itself valuable -- it constrains the design space for macro-scale tree composition.
