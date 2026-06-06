# Peer Review: Scale Phase Transition

## Experiment Type
Guided exploration (Type 2). Proven framework: LoRA perturbation theory + two-regime model (Finding #249). Unknown: shape of the transition function f(s) for math reasoning.

## Hack Detector
- Fix count: 0 (pure measurement experiment). No mechanisms or losses introduced. CLEAN.
- Is MATH.md a proof or a description? Description of competing models with predictions. Acceptable for Type 2 -- this is a guided exploration, not a verification experiment. No proof is expected; the framework (LoRA perturbation theory) is cited correctly.
- Metric used as evidence: Math accuracy (exact numerical match). Binary and well-defined. Appropriate for the claim.
- Kill criteria source: Derived from the three competing models. K1 and K2 discriminate between models. K3 tests a foundational assumption. Well-designed.

## Self-Test Audit
1. One-sentence impossibility property: "The shape of the scale-behavior transition function f(s)." -- This is the unknown being explored, not an impossibility property. Acceptable for Type 2 but imprecise. MINOR.
2. Cited theorems: LoRA (Hu et al., 2106.09685) -- real, applied correctly. Nanda et al. (2301.05217) on grokking phase transitions -- real paper, but used as analogy only, not as a theorem with conditions. The analogy (scale ~ training time) is hand-wavy but acknowledged as motivation, not proof. ACCEPTABLE.
3. Specific numbers: Prediction table provided for all 3 models across 9 scales. Falsifiable. PASS.
4. Falsification condition: K1/K2/K3 target the model discrimination, not arbitrary thresholds. PASS.
5. Hyperparameters: 0 added. Sweeping existing parameter. PASS.
6. Hack check: No. Pure measurement. PASS.

## Mathematical Soundness

MATH.md is well-structured for a Type 2 exploration. It correctly frames three competing models with distinct quantitative predictions and designs kill criteria to discriminate between them. No formal proof is required or claimed.

**Issues found:**

1. **The sigmoid fit is degenerate (MAJOR).** The fit uses `bounds=([1.0, 0.1], [20.0, 10.0])`, meaning tau is lower-bounded at 0.1. The fitted tau=0.1 hits the lower bound exactly. This means the optimizer wants tau < 0.1 (i.e., a pure step function) but is prevented by the bound constraint. The reported tau=0.1 is an artifact of the bound, not a meaningful fit parameter. The R-squared of 0.989 is achieved because a step function at s_mid=5.7 fits the data well -- but this is just saying "the data looks like a step function," which is already obvious from the raw numbers. The sigmoid fit adds no information beyond what the raw transition table shows. PAPER.md correctly notes this ("only because it collapses to a step function") but still reports the specific numbers s_mid=5.7 and tau=0.1 as if they are meaningful findings.

2. **Binomial confidence intervals at n=10 are severe (MAJOR for claims, ACKNOWLEDGED).** The paper acknowledges this in Limitations but then makes strong claims anyway. Specific issues:
   - At s=4, observed 1/10 correct. Binomial 95% CI: [0.003, 0.445]. True accuracy could be as high as 0.44.
   - At s=6, observed 7/10 correct. Binomial 95% CI: [0.348, 0.933]. True accuracy could be as low as 0.35.
   - The claimed "jump of 0.60" has overlapping confidence intervals. A Fisher exact test for 1/10 vs 7/10 gives p=0.020 (two-sided), which is significant at alpha=0.05 but marginal. With 9 pairwise comparisons across the sweep, no multiple-testing correction is applied, so this p-value is not as strong as it appears.
   - The claim "transition width is less than 0.2 scale units" is not supportable. The data only samples s={4, 6}; the transition could span the entire [4, 6] interval (width = 2 scale units). The tau=0.1 is a bound artifact (see point 1).

3. **The single correct prompt is the same across all low scales (IMPORTANT).** At base, s=1, s=2, and s=4, the SAME prompt (James/partner teaching years, gt=70) is correct. This means the "base rate" of 0.10 is entirely driven by one easy prompt. This is not an issue per se (it is what it is at n=10), but it means the transition from 1/10 to 7/10 is really "6 additional prompts crossing threshold." The per-prompt analysis would be more informative than aggregate accuracy.

4. **The generation format changes qualitatively at s=6 (OBSERVATION).** Comparing s=4 vs s=6 for the same prompt (Mr. Grey): at s=4 the model generates a verbose prose solution that gets the wrong answer; at s=6 it generates GSM8K-style "<<3*26=78>>" format and gets the right answer. This is not just "math reasoning activating" -- the adapter at s=6 is imposing the training data's FORMAT (GSM8K step-by-step with <<>> markers), which directly enables answer extraction. The "phase transition" may be an artifact of the answer extraction regex matching the training format, not a genuine reasoning threshold. This is a confound: does s=6 activate reasoning, or does it activate the GSM8K output format that happens to produce extractable correct answers?

## Prediction vs Measurement

| Prediction | Predicted | Measured | Verdict |
|-----------|-----------|----------|---------|
| P1: Jump >= 0.4 between adjacent scales | >= 0.4 | 0.60 (s=4 to s=6) | MATCH (but CI overlaps, see above) |
| P2: f(10) ~ 0.45 (sigmoid midpoint) | 0.45 | 0.80 | REFUTED -- supports step model |
| P3: CoT without correct answers at intermediate s | Present | Absent (CoT + correct at s=6) | REFUTED -- no two-threshold |
| P4: Monotonic | Non-decreasing | 0.70 at s=16 vs 0.80 at s=12 | VIOLATED (but within noise) |

Model discrimination is clear: Model 1 (phase transition) is the best fit. Models 2 and 3 are refuted. This part of the analysis is sound.

## Novelty Assessment

The observation that LoRA scale has a threshold effect on behavioral capability is interesting and practically useful for the composable experts architecture. The two-regime model from Finding #249 is refined with a specific boundary estimate. The finding that binary scale suffices (FORMAT vs CAPABILITY) simplifies the routing architecture.

No prior art found that specifically maps LoRA scale transitions for behavioral activation. The closest is the grokking literature (Nanda et al.), which studies training dynamics rather than inference-time scaling.

## Specific Findings

### Finding 1: Phase transition claim is directionally correct but overstated (MAJOR)
The jump from 1/10 to 7/10 between s=4 and s=6 is real (p=0.020, Fisher exact). However:
- The transition width claim (tau=0.1, < 0.2 scale units) is unsupported -- it is a bound artifact
- The precise location s*=5.7 is a fit artifact from a degenerate sigmoid
- At n=10, the true accuracy at s=6 could be anywhere in [0.35, 0.93]
- The GSM8K format confound (point 4 above) needs investigation

**Required fix:** Weaken the transition width claim. Report s* in [4, 6] as the transition interval. Do not report tau=0.1 as a meaningful parameter. Acknowledge the format-vs-reasoning confound.

### Finding 2: Non-monotonicity (K3 FAIL) is noise (AGREE)
7/10 vs 8/10 at n=10 is well within binomial noise (p=1.0 by Fisher exact). The paper correctly identifies this. No action needed.

### Finding 3: Architectural implication (binary scale) is premature (MODERATE)
The claim that "s=6 suffices (no need for s=20)" for capability domains is based on:
- 7/10 at s=6 vs 8/10 at s=20 -- indistinguishable at n=10
- Math domain only -- code and medical are untested at intermediate scales
- 10 prompts -- the 2 always-wrong prompts may respond to s=20 with more prompts/harder tests

The claim is reasonable as a hypothesis but should not be stated as an "architectural implication" without testing on other domains and with larger n.

**Required fix:** Label this as a hypothesis for follow-up, not a conclusion. The current data cannot distinguish s=6 from s=20 performance.

### Finding 4: CoT rate finding is genuine and interesting (MINOR POSITIVE)
The observation that the base model already has 90% CoT rate, and that low-scale adapters (s=1, s=4) actually REDUCE CoT rate, is a genuine and useful finding. It shows adapters at low scale are disruptive rather than neutral. This deserves more attention in the paper.

### Finding 5: The sigmoid fit should be removed or explicitly marked as degenerate (MINOR)
Reporting s_mid=5.7 and tau=0.1 with R-squared=0.989 gives a false sense of precision. The R-squared is high because any step function near s=5 would fit well. The fit has 2 free parameters for a dataset with effectively 2 regimes (low and high). A simpler analysis -- "the transition occurs between s=4 and s=6" -- is more honest and equally informative.

## Macro-Scale Risks (advisory)
- The phase transition location likely depends on adapter rank, training data size, and base model size. The specific value s* in [4, 6] is for rank-16 LoRA on BitNet-2B-4T with this specific training set. Do not assume it transfers.
- The GSM8K format confound matters more at macro scale where evaluation benchmarks have diverse output formats.
- At larger n, the "phase transition" may reveal itself as a steep sigmoid (tau ~ 1) rather than a true step function. The micro data cannot distinguish these.

## Verdict

**REVISE**

The experiment is well-designed and the core finding (sharp transition between s=4 and s=6, supporting Model 1 over Models 2 and 3) is sound. However, the claims overreach the evidence in three specific ways that must be fixed before recording findings:

1. **Remove or explicitly flag the sigmoid fit as degenerate.** tau=0.1 hits the optimizer bound. Report the transition as "between s=4 and s=6" without false precision. Do not claim "transition width < 0.2 scale units."

2. **Weaken the architectural implication.** Change "s=6 suffices" to "s=6 may suffice for math; verification needed for code and medical at n>=50." The binary scale claim is a hypothesis, not a conclusion.

3. **Acknowledge the format confound.** The s=6 generations switch to GSM8K format (<<>> markers), which may drive the accuracy jump through better answer extraction rather than genuine reasoning improvement. A follow-up should test whether the model produces correct intermediate steps or just matches the training format.

4. **Add per-prompt transition analysis.** The data is already collected. Show which of the 10 prompts flip from wrong to right between s=4 and s=6, and whether the "always correct" prompt (James/70) remains correct throughout. This is more informative than aggregate accuracy at n=10.

None of these issues kill the experiment. The directional finding (sharp transition, not gradual) is credible. The required fixes are about precision of claims, not validity of the core observation.
