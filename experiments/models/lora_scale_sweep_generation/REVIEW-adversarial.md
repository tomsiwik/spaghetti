# Peer Review: LoRA Scale Sweep Generation

## Experiment Type
Guided exploration

## Hack Detector
- Fix count: 0 (parameter sweep of existing mechanism, no new mechanisms added)
- Is MATH.md a proof or a description? **Description dressed in equations.** Weyl's inequality is cited correctly but never used to derive a quantitative prediction. The "perturbation ratio rho(s)" is defined but no theorem states what happens at rho=1 in terms of generation quality. The framework says "when rho > 1, overwrite regime" but this is a heuristic label, not a proven bound.
- Metric used as evidence: Domain-specific composite scores (factual recall, syntax, answer correctness). These are reasonable execution-based metrics but their connection to the perturbation theory is never formalized.
- Kill criteria source: Reasonable for guided exploration. K620/K621/K622 are derived from the research question (does scale matter, does it break code dominance, does low scale work). Not derived from a proof, because there is no proof.

## Self-Test Audit

1. **One-sentence impossibility property:** "The perturbation ratio rho(s) controls the regime." This is a mechanism description, not an impossibility property. An impossibility property would be: "No adapter can overwrite base representations when rho(s) < epsilon, because Weyl's inequality bounds the singular value shift." The current statement is a restatement of the hypothesis, not a mathematical guarantee. **FLAG: evasion.**

2. **Cited theorems:** Weyl's inequality -- real theorem, correctly stated. LIMA (2305.11206) -- empirical finding, not a theorem. Calling it a theorem is a stretch. Weyl's conditions apply (matrix perturbation of sigma_i), but the leap from "singular values are bounded" to "generation quality follows a sweet spot curve" is never proven. **FLAG: gap between cited math and predictions.**

3. **Predicted numbers:** P1 (alpha > 0.10), P3 (format > 0.3), P4 (alpha(20, legal) < 0). These are specific and falsifiable. However, P1 and P3 thresholds are not derived from the perturbation theory -- they are plausible but arbitrary. P4 is a replication of a prior finding, not a prediction from the proof. **Acceptable for guided exploration.**

4. **Falsification condition:** "If ALL scales produce identical quality, the framework is wrong." This is reasonable but very easy to pass -- almost any noise would create differences. A stronger falsification would be: "If the relationship between scale and quality is non-monotonic in a pattern inconsistent with the augment/overwrite transition." **Acceptable but weak.**

5. **Hyperparameter count:** 0. Correct -- sweeping an existing parameter.

6. **Hack check:** No. This is a clean parameter sweep. No concern.

## Mathematical Soundness

**What holds:**
- Weyl's inequality is correctly applied: |sigma_i(W + s*B*A) - sigma_i(W)| <= s*||B*A||_2
- The perturbation ratio rho(s) = s*||B*A||_2 / ||W||_2 is well-defined
- The qualitative framework (low scale = negligible, high scale = overwrite) is reasonable

**What does not hold:**
- There is no theorem connecting rho(s) to generation quality Q(s,d). The entire framework is: Weyl says singular values shift proportionally to s, therefore quality must follow. This is a plausible hypothesis, not a derivation.
- The "augmentation" vs "overwrite" regimes are defined by rho < 1 vs rho > 1, but rho was never measured. We do not know what rho(20) actually is for any of these adapters. Without measuring ||B*A||_2 and ||W||_2, the theory is unfalsifiable at the mechanism level.
- The domain-dependent prediction (P5: "s* is higher for code/math than for prose") is a LIMA-based intuition, not a derivation from Weyl. LIMA is an empirical observation about SFT behavior, not a theorem with conditions.

**Verdict on math:** For a guided exploration, this is acceptable. The framework provides reasonable hypotheses grounded in real math. It is not a proof, and should not be called one. The experiment is essentially: "Weyl + LIMA suggest scale matters differently per domain type; let us measure whether this is true."

## Prediction vs Measurement

PAPER.md contains the required table. Assessment of each prediction:

| Prediction | Verdict | Concern |
|------------|---------|---------|
| P1: alpha > 0.10 for at least 1 domain | PASS | Math +700% is real but n=10 binary scores (8/10 vs 1/10). SE is large. |
| P2: Code NOT best on all 5 | PASS | But 2 of the "domain wins" are ties or within noise. See below. |
| P3: Format > 0.3 at s<=2 | PASS | Trivially -- 0.93 is far above 0.3. The threshold was too easy. |
| P4: alpha(20, legal) < 0 | PASS | Replication of Finding #209. Valid. |
| P5: s* varies by domain (prose < structured) | **PROBLEMATIC** | See detailed critique below. |

### P5 Detailed Critique

PAPER.md states: "prose avg s*=8.3, structured avg s*=20.0." But in the Findings section (Finding 1), the domains are categorized as:
- Learnable-task: math (s*=20)
- Structured-output: code (s*=20), medical (s*=20)
- Knowledge-dependent: legal (s*=4), finance (s*=1)

If "prose" means knowledge-dependent domains (legal, finance), their average s* is (4+1)/2 = 2.5, not 8.3. The only way to get 8.3 is to include medical: (20+4+1)/3 = 8.33. But the paper itself classifies medical as "structured-output," not prose/knowledge-dependent. **The P5 measurement is computed using a category assignment that contradicts the paper's own taxonomy.** This is post-hoc category gerrymandering to make the prediction match.

The actual data shows: all 3 domains with s*=20 (math, code, medical) vs 2 with s*<20 (legal at 4, finance at 1). The real pattern is "most domains peak at s=20, two knowledge-weak domains do not." This is a weaker claim than "domain type determines optimal scale."

### K621 Detailed Critique

The K621 comparison has methodological issues:

1. **Finance: domain=0.1766 vs code=0.1766 at s=1.** These are IDENTICAL to 4 decimal places. PAPER.md reports this as "Tie (finance)" and counts it as a domain win for the "4/5" claim. This is not a domain win -- the code adapter produces exactly the same score because at s=1 the adapter effect is negligible for both.

2. **Legal: domain=0.0995 vs code=0.0938 at s=4.** Difference is 0.006, while stderr is ~0.023. This difference is not statistically significant (t < 0.2). Calling this a "domain win" is unjustified.

3. **The fair comparison** should use the code adapter at its OWN best scale for each eval domain, not at the domain adapter's best scale. Code adapter at s=20 on medical gets 0.278, while medical adapter at s=20 gets 0.310 -- this is a real comparison. But for legal, comparing code@s=4 vs legal@s=4 is comparing code at a suboptimal scale. Code adapter's best legal score might be at a different scale.

Actually, looking at the data: code_cross_domain at s=4 for legal = 0.094, code at s=20 for legal = 0.082, code at s=2 = 0.097, code at s=1 = 0.098. So code adapter on legal peaks at s=1 (0.098) vs domain adapter at s=4 (0.100). Still within noise, but the comparison in the paper uses the wrong scale for the code adapter. **The K621 comparison should compare each adapter at its own best scale, not both at the domain adapter's best scale.**

### Statistical Concerns

1. **Math scores are binary** (correct/incorrect). At n=10, going from 1/10 to 8/10 is "+700%" but the 95% CI for the s=20 score is approximately [0.44, 0.97] (binomial). The true effect is real but the magnitude is uncertain.

2. **Finance scores are bimodal.** Looking at the raw scores: {0.44, 0.49, 0.25, 0.45, 0.02, 0.01, 0.01, 0.06, 0.01, 0.0}. Five prompts get ~0.3-0.5, five get ~0.0. This bimodality means the mean is misleading and the standard error (0.063) understates uncertainty.

3. **No multiple comparison correction.** With 5 domains x 5 scales = 25 comparisons, some "best" values are expected by chance.

4. **Code adapter scores at s=1 and s=2 for code domain are identical to base.** Scores: {0.071, 0.067, 0.729, 0.038, 0.809, 0.082, 0.022, 0.809, 0.073, 0.82} at s=1 vs base {0.071, 0.033, 0.729, 0.038, 0.809, 0.082, 0.022, 0.809, 0.777, 0.82}. Most individual scores are identical -- the adapter at s=1 is effectively doing nothing (consistent with the perturbation theory). But then the -16% "advantage" at s=1 is from 2-3 prompts where the adapter slightly changed output. This is noise, not a real effect.

## NotebookLM Findings

Skipping -- the mathematical and statistical issues are clear from direct analysis.

## Novelty Assessment

This is a parameter sweep, not a novel contribution. The finding that "different domains have different optimal hyperparameters" is well-known in the adapter literature (e.g., AdaLoRA, 2303.10512, which proposes per-layer rank adaptation). The specific finding that scale interacts with domain type is useful for this project's architecture but is not novel in the broader literature.

The operationally useful finding is: "at s=20, math/code adapters work well but legal/finance do not." This is a practical calibration result, not a theoretical advance.

## Macro-Scale Risks (advisory)

1. The legal/finance degradation at high scale may be entirely due to the 2B base model lacking domain knowledge (acknowledged in Limitations). At 7B+ this finding likely disappears.
2. The "three domain categories" taxonomy is derived from n=5 domains. With more domains, the categories may not hold.
3. The perturbation ratio rho(s) was never measured -- at macro scale, measuring ||B*A||_2 / ||W||_2 for each adapter would ground the theory properly.

## Verdict

**REVISE**

The experiment is well-executed as a parameter sweep and produces useful operational data. However, several issues need fixing before the findings are trustworthy:

1. **Fix P5 measurement.** The "prose avg s*=8.3" uses medical as a prose domain while the Findings section classifies it as structured-output. Either reclassify medical consistently, or report the honest numbers: knowledge-dependent avg s*=2.5, structured/learnable avg s*=20. The prediction "s*_prose < s*_code" holds either way -- just report it honestly.

2. **Fix K621 comparison methodology.** Compare each adapter at its OWN best scale, not both at the domain adapter's best scale. Report: "domain adapter at its best scale vs code adapter at its best scale for that eval domain." Currently the comparison is rigged (mildly) against the code adapter.

3. **Report statistical significance for K621 domain wins.** For legal (delta=0.006, SE~0.023) and finance (delta=0.000), these are not significant wins. Honest reporting: "domain adapters significantly win on 2/5 domains (medical, math), tie on 2/5 (code, finance), and insignificantly lead on 1/5 (legal)."

4. **Acknowledge the hack detector finding in MATH.md.** The framework is a well-grounded hypothesis, not a proof. Change "Framework predictions (from perturbation theory + LIMA)" to "Hypotheses (motivated by perturbation theory + LIMA)." The Self-Test "impossibility property" should be rewritten as an honest statement that this is exploration, not verification.

5. **Measure rho(s) for at least one adapter.** Computing ||B*A||_2 / ||W||_2 for the code adapter at each layer would take minutes and would ground the perturbation theory in actual data. Without it, the "augmentation vs overwrite regime" framing is just a label.

None of these require re-running the experiment. Items 1-4 are documentation fixes. Item 5 is a small addition to the analysis script.
