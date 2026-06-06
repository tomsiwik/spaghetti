# Peer Review: Scale Reconciliation Behavioral

## Experiment Type
Guided exploration

MATH.md declares this as guided exploration. It cites LoRA perturbation theory
(Hu et al., 2021) and the LIMA hypothesis (Zhou et al., 2023) as the proven
framework, and identifies the unknown as "whether uniform scale=2.0 matches
per-domain optimal behavioral quality." This is a legitimate Type 2 framing.

## Hack Detector
- Fix count: 0. This is a measurement experiment, not a mechanism.
- Is MATH.md a proof or a description? **Description dressed in equations.** There is no Theorem/Proof/QED block. The perturbation ratio rho(s) = s * ||B^T A^T|| / ||W_base|| is a definition, not a theorem. The LIMA hypothesis is cited as motivation but never formalized into a provable claim. The "predictions" are reasonable engineering intuitions, not derived consequences of a theorem.
- Metric used as evidence: Behavioral scores (numerical answer match, syntax parse, factual recall). These are execution-based, which is better than PPL alone.
- Kill criteria source: K1 and K3 are derived from the LIMA hypothesis framing. K2 is derived from the perturbation ratio argument. Reasonable for guided exploration.

## Self-Test Audit

1. **One property:** "The perturbation ratio rho(s) controls whether the adapter overrides base behavior or merely adjusts format." -- This is one property. PASS.
2. **Cited theorems:** LoRA (2106.09685) is real. LIMA (2305.11206) is real. However, LIMA is an empirical observation (a hypothesis), not a theorem with formal conditions. The self-test says "existing theorems" and lists LIMA -- this is a stretch. LIMA has no formal preconditions to check. MINOR FLAG.
3. **Specific numbers:** Math at s=2.0 predicted 0.10-0.30. Legal/finance predicted >= per-domain optimal. These are specific and falsifiable. PASS.
4. **Falsification:** "The framework is wrong if s=2.0 achieves math >= 0.40." This targets the perturbation dominance model, not just the experiment. PASS.
5. **Hyperparameters:** 0 added. PASS.
6. **Hack check:** No hacks. PASS.

Self-test is complete with one minor flag (LIMA called a "theorem").

## Mathematical Soundness

**What holds:**
- The perturbation ratio definition rho(s) = s * ||B^T A^T|| / ||W_base|| is dimensionally correct and the linear scaling in s is trivially true for the LoRA composition formula y = (W_base + s * B^T A^T) x.
- The worked example (Section F) is arithmetically correct: rho(2) = 0.2, rho(20) = 2.0.
- The prediction that math degrades at low scale follows logically from the observation that math requires large perturbation (8/10 at s=20 vs 1/10 at base).

**What does not hold / is missing:**
- There is no formal proof of anything. The "perturbation ratio controls behavioral regime" is stated as fact but never proved. It is a plausible hypothesis supported by the data pattern, but it could equally be explained by other mechanisms (e.g., attention pattern activation thresholds, layer-specific effects, nonlinear gating).
- The LIMA analogy is informal. LIMA says 1000 examples suffice for format learning. The leap to "therefore low LoRA scale suffices for format" conflates data quantity with parameter perturbation magnitude. These are different axes of variation. A formal connection would need to show that format features live in a low-norm subspace of the weight perturbation, which is never attempted.
- Assumption A1 (monotonicity of behavioral quality in scale for learnable tasks) is stated but unproven and could easily be false. The paper acknowledges this but does not bound its impact.
- The rho(s) ratio is computed for ||B^T A^T|| as a whole, but LoRA adapters are applied per-layer. Different layers may have very different perturbation ratios. The single-number characterization is a significant simplification.

**For guided exploration, this level of formalism is acceptable.** The framework provides directional predictions that were tested and confirmed. The lack of a formal theorem is consistent with Type 2 (the unknown IS whether the framework's predictions hold).

## Prediction vs Measurement

PAPER.md contains the required table. Checking each prediction:

| Prediction | Predicted | Measured | Verdict |
|-----------|-----------|----------|---------|
| P1: Format preserved at s=2.0 | Coherent all domains | 10/10 format OK all 5 | CONFIRMED |
| P2: Math degrades at s=2.0 | 0.10-0.30 | 0.100 | CONFIRMED (at floor of range) |
| P3: Legal/finance preserved | >= per-domain optimal | legal 0.097>=0.096, finance 0.181>=0.155 | CONFIRMED |
| P4: s=2.0 within 20% on >=3/5 | >=3 domains | 4/5 (medical, legal, finance, math[trivially]) | See note |

**Note on P4:** The paper claims 4/5 domains within 20%, but the framing is odd. Math at s=2.0 is 0.100 vs per-domain 0.800 -- that is 87.5% degradation. The paper counts only 1/5 as "worse" because it uses a >20% threshold, but math clearly fails spectacularly. The claim "4/5 within 20%" appears to count math as NOT worse because it says "worse on 1/5 (math only, 87.5% degradation)" -- meaning 4/5 are within 20%. This arithmetic is correct but the framing buries the lede: the one failure is catastrophic.

However, K2 catches this correctly: math loses 100% of the gain, triggering FAIL and killing the experiment. The paper does not try to hide this.

**Statistical concern:** n=10 per domain. Legal scores of 0.097 vs 0.096 -- a difference of 0.001 on n=10 is noise, not signal. The claim "legal at s=2.0 >= per-domain optimal" is not statistically supportable. Same for finance (0.181 vs 0.155 on n=10). These differences are within sampling noise. P3 should be "INCONCLUSIVE" not "CONFIRMED."

## NotebookLM Findings

Skipped -- the experiment is straightforward enough that a deep review is not warranted for the mathematical content. The core question is empirical.

## Novelty Assessment

This is not novel research -- it is an internal measurement experiment resolving a discrepancy between two prior findings (#217 and #246). It correctly identifies that PPL and behavioral metrics can disagree, which is a well-known phenomenon in the alignment literature. The per-domain scale finding is useful architectural input for the routing system.

No prior art is being duplicated. This is original measurement within the project.

## Macro-Scale Risks (advisory)

1. The two-regime model (format vs capability) may not generalize beyond BitNet-2B-4T. Larger models with stronger base capabilities may have different phase transition points.
2. The per-domain scale requirement adds complexity to the routing system (must output scale, not just adapter selection). Worth confirming this is architecturally feasible.
3. n=10 is insufficient for any claims about the exact location of phase transitions. Macro needs n>=50 minimum.

## Verdict

**PROCEED**

Justification:

1. **Type 2 framework is sound.** The experiment correctly identifies a proven framework (LoRA perturbation scaling), states the unknown (behavioral quality at s=2.0), and narrows it with measurements. This is what guided exploration is supposed to do.

2. **Predictions were specific and testable.** All four predictions were confirmed or falsified cleanly. The kill criterion K2 correctly triggered, leading to a killed finding. This is the proof-first process working as intended.

3. **The finding is valuable.** The PPL-vs-behavioral discrepancy is an important result. It confirms that per-domain scale is necessary and that uniform s=2.0 cannot replace it. This directly informs the routing architecture.

4. **No mathematical errors.** The formulas are correct (if informal). The predictions follow from the framework. The measurements match.

Minor issues that do not block PROCEED:

- P3 (legal/finance preserved) should be marked INCONCLUSIVE given n=10 sample size. The differences are within noise.
- LIMA is cited as a "theorem" in the self-test but is an empirical hypothesis. This is a terminology issue, not a framework error.
- The "description dressed in equations" flag applies, but for Type 2 guided exploration, the standard is "state the proven framework and identify the unknown" -- not "prove a new theorem." The experiment meets the Type 2 standard.

The experiment was correctly killed based on K2. The finding (per-domain scale is necessary; PPL is not a reliable proxy for behavioral quality) is well-supported and useful for architectural decisions.
