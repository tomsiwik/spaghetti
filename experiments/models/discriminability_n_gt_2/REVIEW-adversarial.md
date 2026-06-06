# Peer Review: discriminability_n_gt_2

## NotebookLM Findings

Skipped (experiment already completed with detailed MATH.md and PAPER.md; manual deep review performed instead).

## Mathematical Soundness

### What holds

1. **N=2 gradient derivation is correct.** At N=2, k=2, the router gradient is indeed proportional to (f_1(x) - f_2(x)), which is exactly the discriminability. The MATH.md correctly identifies this as the reason r^2=0.95 at N=2.

2. **The qualitative argument for attenuation is sound.** At N=8, k=2, only 2 of 8 experts contribute to each token's forward pass. The k/N=0.25 dilution factor and the argument about selection noise averaging over selected vs non-selected tokens are both directionally correct.

3. **The Gram-Schmidt basis construction is mathematically valid.** The expert generation procedure creates vectors with controlled cosine to a reference direction by decomposing into parallel and perpendicular components, then rotating in the perpendicular subspace. The additional random orthogonal noise for angular diversity among the N-1 synthetic experts is a reasonable approach.

### What does not hold or is incomplete

4. **MATH.md equation on line 61-63 is wrong.** The gradient expression for selected expert i is written as:
```
dL/ds_i = dL/df_comp * sum_{j in T_k} (dw_i/ds_i) * f_i(x)
        + dL/df_comp * sum_{j in T_k} w_j * (dw_j/ds_i) * f_j(x)
```
This conflates two different effects. The correct expression for the composed output f_comp = sum_{j in T_k} w_j * f_j is:
```
df_comp/ds_i = sum_{j in T_k} (dw_j/ds_i) * f_j(x)
```
The written form double-counts: the first term isolates f_i but sums over j (why?), and the second term has w_j factors that should not be there (w_j is already accounted for through the chain rule on the renormalized softmax). This is not a fatal error since the qualitative conclusion (gradients are weaker at N>2) is supported empirically, but the derivation is sloppy.

5. **The "expected correlations" section (lines 117-129) is hand-waving, not math.** Predicting r^2 > 0.3 "because discriminability controls gradient magnitude for the selected pairs" is an informal argument, not a derivation. A tighter analysis would decompose the total gradient variance into within-selection-set variance and between-selection-set variance, predicting the r^2 drop as a function of k/N. This would give a falsifiable prediction rather than a post-hoc rationalization.

6. **Mean pairwise discriminability is an inadequate summary statistic, and the paper acknowledges this (Assumption 2) but does not address it.** The router gradient for expert i depends on the discriminability of the *selected pair* containing i, not all 28 pairs. The correct predictor would be E[D_ij | i,j in T_k(x)], i.e., the expected discriminability conditioned on selection. This is measurable from the experiment data but was not computed. The gap between the "right" statistic and the "used" statistic partially explains why r^2 drops from 0.95 to 0.46.

### Hidden assumptions

7. **Expert 0 (deltas_a) is always the reference expert.** All 7 synthetic experts are generated relative to expert A. This means expert 0 is qualitatively different from experts 1-7 (it is a real trained expert, while the others are geometric projections). At N=2, both experts were real. At N=8, this asymmetry could systematically bias router behavior toward or away from expert 0.

8. **The cosine control is between each synthetic expert and expert A, not between all pairs.** The MATH.md acknowledges this (std=0.42 at target cos=0.0), but the experiment treats "target cosine" as if it controls mean pairwise cosine. The actual mean pairwise cosine at target=0.0 is -0.06 with std=0.42, meaning some pairs are substantially correlated (cos > 0.3) while others are anti-correlated. This makes "target cosine" a poor independent variable for N=8 and could explain the non-monotonic gradient profile.

## Novelty Assessment

**Prior art:** The paper cites arXiv:2504.12463 ("Dense Backpropagation Improves Training for Sparse MoE") which directly addresses the gradient bottleneck in top-k routing. The "stability gap" reference is also relevant. No other published work studies discriminability as a specific predictor of router gradient magnitude, so the core question is novel.

**Delta over parent:** The parent experiment (gap_causal_mechanism) proved discriminability drives gradients at N=2. This experiment asks a natural follow-up: does it generalize? The delta is the transition from mixing-only to selection+mixing, which is a genuine qualitative change in the optimization landscape.

**No reinvention detected.** The experiment reuses infrastructure from gap_as_signal and gap_causal_mechanism appropriately via imports.

## Experimental Design

### What works

1. **Direct N=2 baseline with identical infrastructure.** The N=2 trials use the same base model, same data, same calibration procedure, same seeds. This is a proper controlled comparison.

2. **Multi-seed design (3 seeds).** Adequate for a micro experiment. The mean-curve analysis across seeds is the right choice for 6 cosine levels.

3. **Both pooled and mean-curve r^2 reported.** The paper correctly identifies mean-curve as the appropriate statistic and pooled as informationally weak.

### What does not work

4. **Only 2 training domains for 8 experts.** Calibration cycles through 2 datasets (train_a, train_b) for 8 experts. This means the router can only distinguish "a-m names" vs "n-z names" -- a binary signal being split across 8 expert slots. The router has no way to learn meaningful routing for experts 2-7 since there are no corresponding data distributions. This is a fundamental confound: the low r^2 at N=8 might reflect insufficient routing signal from 2 domains, not inherent attenuation of the discriminability mechanism. With 8 domains and 8 experts, the discriminability-gradient correlation could be much higher.

5. **Synthetic experts from 2 basis vectors.** All 8 experts live in a 2D subspace (spanned by deltas_a and the perpendicular component of deltas_b, plus small noise). Real N=8 experts trained on 8 distinct domains would span a much higher-dimensional subspace. The experiment tests discriminability in a constrained geometry that may not represent the real N=8 scenario.

6. **The non-monotonic peak at cos=0.3 is unexplained.** The paper notes it but attributes it to "stochastic selection effects." This is not an explanation. A simpler hypothesis: at cos=0.3, the pairwise cosine variance is large enough that some pairs are near-orthogonal (strong gradient) while others are moderately correlated (weak gradient), and the router preferentially selects the near-orthogonal pairs, boosting the mean gradient. This could be tested by examining which expert pairs are actually selected at each cosine level. The data is in the experiment but was not analyzed.

7. **KC2 threshold (shape r^2 >= 0.5) was set without justification.** Why 0.5? The HYPOTHESES.yml says "expert selection gradients behave qualitatively differently from mixing-ratio gradients" -- this is a qualitative criterion, not a quantitative one. A shape r^2 of 0.489 vs 0.500 is not a meaningful distinction. The borderline result is an artifact of an arbitrary threshold.

### Simpler alternative explanation

The positive result (r^2=0.46) could be explained without the discriminability mechanism: higher target cosine produces experts with more similar outputs, which trivially reduces the loss landscape's gradient (there is less to learn when experts produce the same output). This is a confound between discriminability-as-cause and discriminability-as-correlation. The parent experiment at N=2 had the same confound but the near-perfect r^2=0.95 made it less concerning. At r^2=0.46, the confound is significant.

## Hypothesis Graph Consistency

The experiment matches its HYPOTHESES.yml node. The kill criteria are:
- KC1: r^2 < 0.3 at N=8, top_k=2 -- tested correctly, passes at 0.462
- KC2: "expert selection gradients behave qualitatively differently" -- tested via shape correlation, borderline at 0.489

The "partial" status is appropriate given KC1 pass and KC2 borderline. The evidence recorded in HYPOTHESES.yml is accurate.

## Macro-Scale Risks (advisory)

1. **The 2-domain/8-expert confound vanishes at macro.** With real multi-domain LoRA experts (each trained on a distinct domain), the router has a real 8-way signal. The discriminability-gradient correlation could be significantly stronger than the micro result suggests. This makes the micro r^2=0.46 a pessimistic estimate.

2. **Gradient magnitude at large N.** The 5-7x attenuation at N=8 implies ~20-30x at N=32. If calibration steps scale linearly with attenuation, large expert pools become expensive to calibrate. Dense backpropagation (cited in the paper) is the obvious fix but has not been tested.

3. **At real scale (cos~0.0002), discriminability is always maximal.** The paper correctly notes this makes the entire experiment somewhat academic for the practical system. The contribution is mechanistic understanding, not practical optimization.

## Verdict

**PROCEED**

The experiment achieves its primary goal: demonstrating that the discriminability-gradient mechanism survives the transition from mixing-only (N=2) to selection+mixing (N=8). The core finding -- KC1 passes at r^2=0.46, mechanism generalizes but attenuates -- is sound and informative.

The weaknesses I identified are real but fall into two categories:

(a) **Limitations the paper already acknowledges:** synthetic experts, 2 domains for 8 experts, imprecise cosine control. These are inherent micro-scale constraints.

(b) **Missed analyses that would strengthen the paper but do not invalidate it:** the selected-pair discriminability statistic, the expert selection pattern analysis at cos=0.3, and the alternative confound explanation.

The partial status is honest. The practical implication (Section "Implications for the Architecture" point 2) is the most important takeaway: at real scale, all experts are maximally discriminable, so the attenuation is irrelevant. This experiment confirms that the mechanism is not broken at N>2, which is what needed to be established before building on it.

No further micro experiments needed on this branch. The discriminability question is answered: it works, it weakens with selection noise, and it does not matter in practice because real experts are always in the strong-gradient regime.
