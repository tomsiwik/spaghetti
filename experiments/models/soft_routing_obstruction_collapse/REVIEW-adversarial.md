# Peer Review: Soft-Routing Obstruction Collapse

## Experiment Type
Verification (Type 1) -- MATH.md contains Theorem 1 with Proof/QED, makes quantitative predictions, experiment measures against them.

## Hack Detector
- Fix count: 0. No new mechanism introduced. This is a measurement experiment.
- Is MATH.md a proof or a description? **Proof with QED** (Theorem 1 is a genuine proof). However, see Mathematical Soundness for caveats on the proof's scope.
- Metric used as evidence: Mean activation count K and fraction K>=3. These are direct observables predicted by the theorem -- appropriate.
- Kill criteria source: **Derived from proof** (K1 from Binomial(5,0.5) baseline, K2 from Theorem 1 corollary).

## Self-Test Audit
1. One-sentence impossibility property: "K>=3 implies every 1-cycle filled by 2-simplex, hence H^1=0." -- **PASS.** Single property, clearly stated.
2. Cited theorems: Finding #242, Jang et al. 2017 (ICLR, real paper), Borsuk 1948 nerve theorem (real). -- **PASS.** However, the application of the nerve theorem deserves scrutiny (see below).
3. Predicted numbers: E[K]>=2.5, E[K]>=3.0 learned, frac>=3 >0.50, PPL ratio <=1.05. -- **PASS.** Specific and falsifiable.
4. Falsification condition: "K>=3 but H^1 != 0." -- **PASS.** Targets the proof, not just the experiment.
5. Hyperparameter count: 1 (temperature tau). -- **PASS.** Minimal, and tau=1 is justified from prior work.
6. Hack check: "No new mechanism." -- **PASS.** This is a measurement, not an intervention.

## Mathematical Soundness

### What holds

**Theorem 1's core logic is correct.** If K>=3 adapters are all active on the same token x, then every pair among those K adapters shares x in their intersection, creating 2-simplices that fill any 1-cycle among those adapters. The topological argument is sound: for any three active adapters {i,j,k}, the triple intersection is non-empty (contains x), so the 2-simplex (i,j,k) exists in the nerve.

**The Gumbel-sigmoid activation probability formula is correct.** P(g>0.5) = exp(-exp(-l)) follows directly from the Gumbel CDF. The worked example in Section F checks out numerically.

### What does not hold (or is misleading)

**Issue 1: The prediction E[K]>=2.5 "at neutral logits" is wrong for BCE-trained routing.** The proof assumes logits could be neutral (near zero), yielding p_i ~ 0.5 per gate, giving E[K] ~ 2.5 from Binomial(5,0.5). But this assumption is not grounded in any analysis of what BCE training actually produces. MATH.md Section A even acknowledges "learned routing may push logits negative," yet the prediction table still claims E[K]>=2.5. The prediction contradicts its own setup.

The worked example (Section F) already shows E[K]=2.2 for moderately informative logits -- below the 2.5 threshold. This should have been a red flag BEFORE running the experiment. The MATH.md essentially predicts its own failure in the worked example and then proceeds to predict success in the prediction table. This is inconsistent.

**Issue 2: The prediction E[K]>=3.0 "with learned routing" is unjustified.** The only citation is "Theorem 1 + Finding #185." Finding #185 is about energy-gap routing accuracy, not about activation count. There is no derivation showing that learned routing would push logits positive for 3+ adapters. The prediction is a hope, not a theorem consequence.

**Issue 3: Theorem 1 has limited scope.** The theorem proves: IF K>=3 THEN H^1=0. It does NOT prove: learned routing will achieve K>=3. The experiment tests the antecedent (whether K>=3 occurs), not the theorem itself. The theorem is trivially unfalsifiable by this experiment -- it is a conditional statement. What was actually tested is the hypothesis that "BCE-trained Gumbel-sigmoid routing achieves K>=3," which has no theorem backing it.

**Issue 4: The nerve theorem application is subtly different from Finding #242.** Finding #242 computed the Cech nerve of a top-k specialization cover (deterministic, each token assigned to exactly k adapters). Theorem 1 applies this to a stochastic routing cover where K varies per token. The proof handles this correctly (it argues per-token), but the Corollary about "average topological obstruction" is hand-wavy -- you cannot simply average Betti numbers across tokens. Betti numbers are integers describing global topology, not per-sample quantities that can be averaged.

### Severity

Issues 1-2 are prediction methodology failures but do not invalidate the theorem. Issue 3 means the experiment was testing an untethered hypothesis. Issue 4 is a minor mathematical imprecision in the Corollary.

## Prediction vs Measurement

PAPER.md contains the required prediction-vs-measurement table. Results:

| Prediction | Measured | Match |
|------------|----------|-------|
| E[K] >= 2.5 | 0.93 | NO -- 2.7x below threshold |
| E[K] >= 3.0 | 0.93 | NO -- 3.2x below threshold |
| Frac K>=3 > 0.50 | 0.000 | NO -- maximally wrong |
| PPL(k=3)/PPL(k=2) <= 1.05 | 1.000 | YES but vacuous |
| H^1 = 0 | 0 | YES but vacuous (no overlaps at all) |

The experiment is a clean kill. Every non-vacuous prediction failed. The two "passing" criteria are vacuous because the router is so sparse that forced-k regimes add near-zero weight (sigmoid(-3.2) = 0.039), making PPL identical across all regimes.

**Credit:** PAPER.md correctly identifies the vacuousness and does not claim K2 as a real pass. The analysis of WHY K=1 (BCE optimal logits) is insightful and well-derived. The SIGReg-style "disease vs symptom" analysis in the implications section is strong.

## NotebookLM Findings

Skipping NotebookLM deep review -- the experiment is already killed with clear analysis. The mathematical issues are identifiable from direct reading.

## Novelty Assessment

The experiment is a measurement, not a new method. The novel contribution is the negative result: BCE-trained Gumbel-sigmoid routing is fundamentally sparse (K=1) because domain-classification loss optimizes for selectivity. This is a useful finding that closes off a research path.

The connection between activation count and Cech nerve cohomology (Theorem 1) is a reasonable extension of Finding #242, though the conditional nature limits its novelty -- it says "if you can get K>=3, good things happen" without providing a mechanism to get there.

## Macro-Scale Risks (advisory)

Not applicable -- experiment killed. However, the insight that domain-classification routing is inherently sparse is scale-invariant and should transfer to macro. Any future routing design must use a composition-aware loss (not BCE) if multi-adapter activation is desired.

## Verdict

**KILL** -- correctly killed by the experimenters.

The experiment is well-executed and the kill is appropriate. The PAPER.md analysis is honest, identifies the root cause correctly (BCE loss induces sparsity by construction), and provides actionable next steps (composition-aware loss, forced top-k, diversity regularizer).

However, two issues should be noted for future experiment design:

1. **The prediction E[K]>=2.5 was contradicted by MATH.md's own worked example (E[K]=2.2).** The experiment could have been killed analytically before running. The BCE optimal-logit analysis in PAPER.md (logit(0.80)=1.39 for primary, logit(0.05)=-2.94 for others) should have been in MATH.md as a counter-prediction. Future experiments should check whether their own worked examples satisfy kill criteria before running code.

2. **Theorem 1 is unfalsifiable by this experiment** because it is a conditional (IF K>=3 THEN H^1=0). The experiment tested the antecedent, not the theorem. This is fine as a measurement experiment, but the framing as "proof verification" is misleading. It would be more accurately described as "testing the feasibility of a precondition."

The negative finding (domain-classification routing is inherently sparse) should be recorded as a supported finding for the research program.
