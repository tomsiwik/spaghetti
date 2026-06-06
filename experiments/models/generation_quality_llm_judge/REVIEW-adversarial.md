# Peer Review: generation_quality_llm_judge

## Experiment Type
Guided exploration (Type 2) -- proven framework (routed LoRA composition changes outputs), unknown being explored (whether prior keyword-density kill was an evaluation artifact).

## Hack Detector
- Fix count: 1 (replaced evaluation metric). Not a hack -- this is the right response to a suspected evaluation artifact.
- Is MATH.md a proof or a description? **Description dressed in equations.** MATH.md describes a statistical testing framework (Wilcoxon, Bonferroni, Spearman) but proves no theorem about the architecture. The statistical machinery is standard and correctly applied, but there is no novel proof here -- just a well-structured experimental design document.
- Metric used as evidence: LLM-as-judge composite score (1-5). **Not proven to predict the behavioral outcome.** The researcher correctly identified this post hoc.
- Kill criteria source: K1 derived from prior experiment's threshold (3/5 domains worse). K2 threshold (r>0.7) is reasonable but somewhat arbitrary -- why 0.7 and not 0.5 or 0.8?

## Self-Test Audit
1. **One-sentence impossibility property:** States Wilcoxon is distribution-free. This is a property of the test, not an impossibility property of the architecture. Weak but acceptable for Type 2.
2. **Existing theorems:** Wilcoxon (1945), Bonferroni, Spearman, Cohen -- all real, all correctly applied. Pass.
3. **Specific numbers:** >99.9% power for medium effects (pre-experiment), updated post-experiment with effective-n table showing the power analysis was inapplicable. Good intellectual honesty. Pass.
4. **Falsification:** "The framework assumed ordinal judge scores with sufficient variance. This assumption FAILED." Honest and correct self-assessment. Pass.
5. **Hyperparameters:** 0 for statistical framework. The judge prompt is a design choice. Fair characterization. Pass.
6. **Hack check:** Single change (new metric). Correctly notes the replacement metric also failed, in a different way. Pass.

Self-test is complete and honest. No blanks or evasions.

## Mathematical Soundness

The statistical framework is correct but ultimately moot:

**What holds:**
- Wilcoxon signed-rank is the right test for ordinal paired data. Correct.
- Bonferroni at alpha/5 = 0.01 is conservative and appropriate. Correct.
- Power analysis derivation is mathematically correct for the assumed conditions. Correct.
- Spearman correlation is appropriate for rank-order agreement. Correct.

**What doesn't hold (and the researcher knows this):**
- The power analysis assumed sufficient variance in judge scores. With 0-11 nonzero differences out of 50, the Wilcoxon test is massively underpowered. The researcher documented this clearly.
- The "two independent metrics agree on direction" argument (p = (1/2)^5 = 3.1%) is **not quite right.** The argument assumes the two metrics independently and uniformly choose a direction for each domain. But: (a) the keyword-density metric showed base winning on ALL 5 domains in this experiment, not 3/5 as in the prior experiment, suggesting systematic bias in the old metric; (b) the judge metric is near-constant, so "direction" is determined by noise in 0-11 samples. The probability calculation is technically correct under its assumptions but those assumptions (independence, equal probability of either direction) are questionable.

**Critical methodological issue: Ties counted as losses.** Line 908 defines `routed_wins = routed_judge_mean > base_judge_mean`. When scores are tied (code: 3.96 vs 3.96, finance: 4.0 vs 4.0), `routed_wins` is False, incrementing `routed_worse_count`. The PAPER.md reports "5/5 domains worse or tied" but K1 counts this as 5/5 worse. With a near-constant judge, ties are the expected null outcome, not evidence of degradation. If ties were excluded, K1 would be 2/5 (math and legal genuinely worse) or 3/5 (adding medical's -0.027 delta), which is borderline.

This is not necessarily wrong -- the K1 criterion was defined as "routed worse on >= 3/5 domains" and the researcher could argue ties indicate failure to improve. But it does mean the "5/5 worse" headline overstates the finding when the judge cannot discriminate.

## Prediction vs Measurement

PAPER.md contains a proper prediction vs measurement table. Assessment:

| Prediction | Verdict | Comment |
|-----------|---------|---------|
| P1 (code routed > base) | Failed | Tied. But judge non-discriminating. |
| P2 (math routed > base) | Failed | Judge scored WORSE despite 24x correctness gain. Reveals judge flaw, not architecture flaw. |
| P3 (medical reverses) | Failed | Near-tied (3.97 vs 4.0). Judge non-discriminating. |
| P4 (legal stays bad) | Confirmed | 3.62 vs 3.80, directionally correct. |
| P5 (finance unknown) | N/A | Tied at 4.0. |
| P6 (metrics disagree) | Confirmed | r = 0.107. |
| P7 (math correctness) | Confirmed | 48% vs 2%. Strong. |

The prediction table is present and honestly assessed. The researcher did not cherry-pick.

## Key Findings Assessment

### 1. Is the kill justified?

**Partially.** The kill is justified as a statement about evaluation methodology: "we cannot demonstrate generation quality improvement using available evaluation methods at micro scale." It is NOT justified as a statement about the architecture: "routed composition hurts generation quality."

The distinction matters because:
- The judge outputs near-constant scores (effective discrimination: 0-11 out of 50 pairs)
- The old metric (keyword density) has known format sensitivity
- Neither metric captures correctness (the math 24x finding)
- Two broken rulers agreeing does not make them right

The researcher correctly identified all of this in the analysis section but still called it a kill. This is defensible -- the experiment was designed to test whether a better metric reverses the verdict, and it did not. But the experiment also revealed the replacement metric is broken, which weakens the "double kill" narrative.

### 2. Is BitNet-2B-4T a valid judge?

**No, and the researcher knows this.** With 49/50 identical scores on medical and 50/50 on finance, the model is outputting a fixed pattern (R=4, C=3, I=5 composite=4.0) regardless of input. This is a 2B model asked to perform a capability (nuanced text quality assessment) that likely requires 7B+ parameters. The self-evaluation paradigm cited (Zheng et al. 2023) used GPT-4, not 2B models.

The researcher correctly identified this as a limitation but could have predicted it before running the experiment. A 2B ternary model's ability to discriminate text quality on a 5-point scale should have been validated with a quick pilot (10 samples, check variance) before committing 70 minutes of compute.

### 3. Is the "two independent metrics agree" statistical argument sound?

**Weak but not wrong.** The p = 0.031 from (1/2)^5 is a valid sign test under independence. However:
- Independence is questionable -- both metrics evaluate the same texts, and both are computed from surface features (keywords) or near-constant outputs (judge)
- The "direction" for tied domains (code, finance) is determined by floating-point noise
- The old metric showed 5/5 base wins in this experiment vs 3/5 in the prior, suggesting either different prompts or systematic bias affected the old metric too

The argument provides "weak but meaningful evidence" as the researcher states. I would not elevate it beyond that.

### 4. Does the math adapter 24x finding change the overall verdict?

**Yes, significantly.** This is the most important finding in the experiment and is buried in the analysis. The math adapter demonstrates that domain-specific routing works exactly as the architecture intends -- for structured tasks where correctness is measurable. The 48% vs 2% correctness gap is:
- Large effect (not noise)
- Measured by an objective metric (answer correctness, not subjective judgment)
- Directly relevant to the architectural thesis (experts improve domain performance)

This finding should be extracted as a separate supported finding, not killed alongside the prose domains. The experiment conflates "generation quality" (prose fluency) with "domain capability" (task correctness). The architecture may excel at the latter while failing at the former.

### 5. Are the task-specific metrics well-designed?

**Math correctness:** Yes. Checking whether the model produces the correct numerical answer to GSM8K-style problems is objective and meaningful.

**Code syntax validity:** Partially. Using Python's `ast.parse` is a reasonable proxy for syntax correctness but not semantic correctness. The reversal from prior experiment (60% to 48%) is concerning and suggests instability or prompt sensitivity.

### 6. What should the project do next?

The experiment has established:
1. Prose generation quality cannot be reliably measured at micro scale with available tools
2. Task correctness (math) shows clear routing benefit
3. The architecture's value may be in structured/factual tasks, not open-ended prose

Recommended next steps (in priority order):
1. **Extract the math correctness finding as a separate supported result.** Do not let it be killed by the prose domains.
2. **Test routing on objective benchmarks** (GSM8K accuracy, code-eval pass@1) where correctness is measurable. This avoids the evaluation problem entirely.
3. **Do NOT retry with a bigger judge.** The evaluation problem is not solvable at micro scale. Accept this limitation and focus on objective metrics.
4. **Investigate the code syntax reversal.** Why did it flip from +12% to -17%? Is there prompt sensitivity or is the prior 10-prompt result just noise?

## Novelty Assessment

This is a retest with improved methodology, not a novel contribution. The LLM-as-judge paradigm is well-established (Zheng et al. 2023). The experimental design (Wilcoxon + Bonferroni + Spearman correlation) is standard statistics applied correctly. The contribution is empirical: establishing that the evaluation artifact hypothesis does not hold (with caveats about judge quality).

The math 24x correctness finding, while not novel in mechanism, is a genuinely useful data point for the project's direction.

## Macro-Scale Risks (advisory)

1. **Judge quality scales with model size.** At macro scale (7B+ judge), the evaluation picture may change entirely. The current finding is specific to 2B self-evaluation.
2. **Prose quality may require generation-aware training.** Adapters trained on perplexity may inherently produce text that scores poorly on surface quality metrics. DPO/RLHF-style training could fix this but changes the training paradigm.
3. **The structured/prose split is architecturally important.** If routing only helps structured tasks, the project should reframe accordingly rather than claiming general-purpose improvement.

## Verdict

**PROCEED** (with conditions)

The experiment is well-executed, honestly reported, and the kill is methodologically defensible as stated (the replacement metric did not reverse the verdict). However, the review identifies several issues that must be addressed:

### Required actions before closing:

1. **Record the math 24x correctness finding as a separate supported finding.** It should not be buried in a killed experiment. Use `experiment finding-add` with status `supported` and title something like "Domain routing produces 24x math correctness improvement (48% vs 2%) despite aggregate generation quality kill."

2. **Acknowledge the tie-counting issue in PAPER.md.** Add a note that K1's "5/5 worse" includes 2 domains where scores were exactly tied (code, finance) and the judge was non-discriminating. The honest count is "2-3/5 worse, 2 tied, 0 better" which still triggers K1 at threshold 3 but changes the narrative.

3. **Record a finding about 2B self-evaluation limitations.** This is a useful negative result: "BitNet-2B-4T cannot discriminate text quality at 1-point granularity on a 5-point scale, outputting near-constant scores (0-11 nonzero diffs out of 50 pairs)." This prevents future experiments from repeating the same mistake.

4. **Update MATH.md post-experiment addendum** to note that a pilot study (5-10 samples, check judge variance) should be standard practice before committing to LLM-as-judge experiments at this scale.

The kill of "routed composition improves general prose generation quality" is justified. The kill of "routing is useless" is NOT justified -- the math finding directly contradicts it. The project should pivot toward objective task benchmarks where the architecture's value is measurable.
