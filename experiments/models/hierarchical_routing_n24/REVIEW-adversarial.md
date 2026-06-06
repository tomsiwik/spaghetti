# Peer Review: Hierarchical Two-Stage Routing at N=24

## Experiment Type
Frontier extension (stated correctly in MATH.md).

Proven result being extended: N=5 routing at 100% accuracy (Finding #179).
Mathematical gap: Whether confusion-graph clustering reduces N=24 to K solved
sub-problems of size <=5.

## Hack Detector
- Fix count: 1 (hierarchical decomposition replacing flat router). Clean, not a patch stack.
- Is MATH.md a proof or a description? **Description dressed in equations.** Theorem 1 is a straightforward expectation decomposition into 3 cases. It is correct but trivial -- any probability-weighted cost decomposition gives this bound. The real claim (that confusion clustering makes delta_intra small) is an ASSUMPTION (A2), not a proven guarantee. The "QED" covers the algebra, not the mechanism.
- Metric used as evidence: Top-1 routing accuracy (primary), PPL gamma (secondary). Accuracy is appropriate for the routing question. PPL is appropriate for the cost question.
- Kill criteria source: K593 (accuracy >= 60%) is derived from the proof's P2 prediction (overall >= 50%) but set higher. K594 and K595 are derived from predictions P3 and P4. Reasonable.

## Self-Test Audit

1. **One-sentence impossibility property:** "Confusion-graph clustering guarantees that misroutable domains are in the same cluster, making within-cluster misrouting PPL-benign by construction." -- This is circular. The clustering groups confused domains together (by definition of using a confusion matrix). The claim that this makes misrouting PPL-benign is the HYPOTHESIS, not a guarantee. Confusion in representation space does not mathematically imply PPL-benignness. Finding #192 is empirical evidence, not a theorem. **FLAG: The impossibility property is actually the untested assumption.**

2. **Cited theorems:** Fiedler (1973) and von Luxburg (2007) are real and correctly described. However, they guarantee optimal graph partitioning, not PPL-benign misrouting. The gap between "minimizes normalized cut on confusion graph" and "minimizes PPL cost" is not bridged by any theorem. Finding #192 is correctly cited as empirical.

3. **Predicted numbers:** P1-P5 are specific and falsifiable. Good.

4. **Falsification condition:** "If domains that confuse each other in representation space do NOT produce similar PPL." This is well-targeted at the core assumption. However, the experiment revealed something even more fundamental: there is essentially NO PPL signal at all (oracle = base). The falsification condition did not anticipate this failure mode.

5. **Hyperparameter count:** 1 (K). Honest. K=4 was data-driven via Ward clustering.

6. **Hack check:** Clean. This replaces the flat router rather than patching it.

## Mathematical Soundness

**Theorem 1 is correct but vacuous in this setting.**

The decomposition:

  E[PPL_routed / PPL_oracle] <= 1 + p_c(1-p_w)*delta_intra + (1-p_c)*delta_inter

is a valid expectation bound. The algebra is sound. But:

1. **The bound is trivially satisfied when adapters provide no benefit.** With oracle PPL = 10.05 and base PPL = 10.06 (0.04% gap), ALL routing strategies trivially satisfy any reasonable PPL bound. The bound tells you nothing about whether routing works -- it tells you adapters do not matter.

2. **Step C3 contains the critical logical error** (correctly identified post-hoc in PAPER.md): "N=5 with distinct domains is a fundamentally different problem than N=5 with similar domains." The proof assumed that reducing class count to <=5 was sufficient (citing Finding #179), but Finding #179 proved that 5 DISTINCT domains are separable, not that any 5 domains are. The N<=5 was a necessary but not sufficient condition. The proof failed to formalize what "separable" means beyond class count.

3. **Corollary 2 arithmetic is correct** but the assumed values (p_c >= 0.80, delta_intra <= 0.05) were reasonable a priori. The actual p_c = 0.973 exceeded the assumption, but the overall framework fails because the question was wrong: routing accuracy is irrelevant when adapters provide 0.04% benefit.

## Prediction vs Measurement

PAPER.md contains a proper prediction-vs-measurement table. Results:

| Prediction | Measured | Verdict |
|-----------|----------|---------|
| P1: cluster acc >= 60% | 97.3% | PASS (massively) |
| P2: overall acc >= 50% | 40.4% | **FAIL** |
| P3: routed PPL < uniform | 10.06 < 10.07 | PASS (0.1%, meaningless) |
| P4: overhead < 15% | 0.85% | PASS |
| P5: delta_intra < 0.10 | See below | PASS (vacuous) |

P3 and P5 pass only because the adapters provide essentially zero benefit (oracle-base gap = 0.04%). Any routing strategy -- including random -- would pass these criteria. These are vacuous passes.

The meaningful kill criterion (K593 / P2) correctly fails.

## NotebookLM Findings

Skipping NotebookLM step -- the mathematical and empirical analysis is straightforward enough that automated deep review would not surface additional concerns beyond what is already clear from the documents.

## Novelty Assessment

The hierarchical routing idea is standard (HMoRA at ICLR 2025 is correctly cited). The specific application to confusion-graph decomposition is a reasonable adaptation. The real novelty is the NEGATIVE finding: this is the seventh routing kill at N=24, and this experiment finally surfaces the root cause -- the adapters themselves provide near-zero benefit (oracle PPL = base PPL), making routing irrelevant regardless of mechanism.

Finding #198 (24 adapters provide near-zero PPL benefit) is genuinely important and should redirect all future routing work.

## Macro-Scale Risks (advisory)

The root cause finding (adapters provide negligible benefit) is likely specific to this setup:
- 400 training samples per domain
- Rank-16 LoRA on a 2.4B ternary model
- Character/token-level evaluation

At macro scale with better-trained adapters that genuinely specialize (oracle PPL << base PPL), hierarchical routing based on confusion clustering could work. The mechanism is sound in principle -- the preconditions (meaningful adapter specialization) just do not hold here.

## Critical Observations

1. **Seven routing kills all explained by one root cause.** Oracle PPL (10.05) is within 0.04% of base PPL (10.06). There is nothing to route. Every routing mechanism produces the same PPL because the adapters are functionally identical to no-adapter. This should have been the FIRST thing checked in any routing experiment, not discovered on the seventh attempt.

2. **The experiment is well-executed despite the flaw.** The code correctly implements Ward clustering on cosine distances, trains appropriate routing heads, and measures all relevant metrics. The failure is in the research question, not the implementation.

3. **PAPER.md correctly identifies the root cause.** The analysis in "The Deeper Lesson" section is honest and precise. This is good scientific practice -- the experiment killed the hypothesis and correctly diagnosed why.

4. **The proof's real gap:** Theorem 1 bounds routing cost assuming adapters provide meaningful specialization. It never proves (or even checks) that this precondition holds. A complete mathematical framework would have started with: "Theorem 0: Under training conditions X, adapters achieve oracle PPL at least Y% below base. Proof: ..."

## Verdict

**KILL**

The experiment is correctly killed by K593 (40.4% < 60% threshold). PAPER.md correctly diagnoses the root cause. The findings (#197, #198) are valuable negative results.

**Specific issues for the record:**

1. **Theorem 1 is correct but vacuous** in this setting because its precondition (meaningful adapter specialization) does not hold. The proof should have included a precondition check as Theorem 0.

2. **The Self-Test impossibility property is circular** -- it restates the hypothesis as a guarantee. Confusion-graph clustering does not mathematically guarantee PPL-benignness; it groups domains that confuse a classifier, which is a different property.

3. **Step C3 contains a category error** that PAPER.md correctly identifies post-hoc: N<=5 is necessary but not sufficient; the N=5 proof exploited domain distinctness, not small N.

4. **Finding #198 is the most important output** of this experiment and of the entire 7-experiment routing sequence. All future routing experiments at N=24 must first demonstrate that oracle PPL is meaningfully below base PPL before testing any routing mechanism.

**Recommendation:** No more routing experiments at N=24 until adapter training is improved to produce oracle PPL at least 5% below base PPL. The routing problem is currently undefined (there is nothing to route).
