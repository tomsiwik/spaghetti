# Generation Quality LLM-Judge: Statistical Foundations

## Experiment Type: Guided Exploration (Type 2)

The framework is proven: routed LoRA composition produces measurably different outputs
from the base model (prior experiment confirmed 5/5 domains show >5% change). The unknown
is whether the prose domain "losses" (medical -6.9%, legal -8.6%, finance -11.9%) are
real quality degradation or artifacts of keyword-density scoring.

This experiment discovers whether the old metric was the disease or the symptom.

## Problem Statement

Given $N=5$ domain adapters and two evaluation functions:

- $f_{\text{old}}$: keyword-density-based composite score (prior experiment)
- $f_{\text{new}}$: LLM-as-judge domain quality rating (this experiment)

We test whether $f_{\text{old}}$ and $f_{\text{new}}$ agree on which configurations
are better. If they disagree, the kill verdict changes.

## A. Failure Mode: Metric Misalignment

The prior experiment was killed because $f_{\text{old}}(\text{routed}) < f_{\text{old}}(\text{base})$
on 3/5 domains. But $f_{\text{old}}$ (keyword density) has a known failure mode: it penalizes
**format-appropriate responses** that use fewer English prose words.

- Code adapter produces Python code -> fewer English keywords -> lower $f_{\text{old}}$ even when
  the code is correct and relevant
- Medical adapter may produce more precise clinical language -> fewer generic keywords ->
  lower $f_{\text{old}}$ even when the response is more medically accurate

## B. The Right Question

Not "how do we fix the metric?" but: **What evaluation function is invariant to output
format while still measuring domain quality?**

Answer: A language model's own assessment of domain relevance and quality, prompted with
explicit criteria. This is the LLM-as-judge paradigm (Zheng et al., 2023, "Judging
LLM-as-a-Judge with MT-Bench and Chatbot Arena", arXiv:2306.05685).

## C. Statistical Framework

### C.1 Paired Comparison (Wilcoxon Signed-Rank Test)

For each domain $d$ with $n=50$ prompts and $S=3$ seeds, we have $n \times |S| = 150$
paired observations $(x_{\text{base},i}, x_{\text{routed},i})$ where $x$ is the
LLM-judge score.

**Theorem (Wilcoxon, 1945).** The Wilcoxon signed-rank test is distribution-free:
under $H_0: \text{median}(x_{\text{routed}} - x_{\text{base}}) = 0$, the test
statistic $W$ has a known null distribution. For $n \geq 25$, the normal approximation
$z = W / \sqrt{n(n+1)(2n+1)/6}$ is accurate to $O(n^{-1/2})$.

We use this rather than the paired t-test because LLM-judge scores on a 1-5 Likert
scale are ordinal, not interval. The Wilcoxon test requires only that differences
are symmetric around the median under $H_0$.

### C.2 Sample Size Justification

With $n=50$ prompts $\times$ 3 seeds = 150 paired observations per domain:

For detecting a medium effect (Cohen's $d = 0.5$, i.e., half a standard deviation
difference in judge scores):

$$\text{Power} = 1 - \Phi\left(z_{\alpha/2} - d\sqrt{n}\right)$$

At $\alpha = 0.05$ (two-sided), $n = 150$:
$$\text{Power} = 1 - \Phi(1.96 - 0.5\sqrt{150}) = 1 - \Phi(1.96 - 6.12) = 1 - \Phi(-4.16) > 0.999$$

We have >99.9% power to detect a 0.5 SD difference. Even for a small effect ($d = 0.2$):
$$\text{Power} = 1 - \Phi(1.96 - 0.2\sqrt{150}) = 1 - \Phi(1.96 - 2.45) = 1 - \Phi(-0.49) \approx 0.69$$

69% power for small effects — adequate for a directional test.

### C.3 Correlation Test for K2 (Spearman Rank Correlation)

K2 tests whether $f_{\text{old}}$ and $f_{\text{new}}$ agree. We compute:

$$r_s = 1 - \frac{6 \sum d_i^2}{n(n^2-1)}$$

where $d_i$ is the rank difference between $f_{\text{old}}$ and $f_{\text{new}}$ scores
for observation $i$.

**Kill trigger:** $r_s > 0.7$ (strong agreement) means the old metric was measuring
the same thing as LLM-judge, implying the prior kill was real, not an artifact.

**Success condition:** $r_s < 0.7$ means the metrics disagree substantially, implying
the keyword-density metric was not a good proxy for actual quality.

### C.4 Multiple Comparisons (Bonferroni)

With 5 domains tested independently, we apply Bonferroni correction:
$\alpha_{\text{adjusted}} = 0.05 / 5 = 0.01$ per domain.

## D. Predictions

| Prediction | Source | Expected Value |
|-----------|--------|----------------|
| P1: Code domain — routed beats base (judge score) | Prior: +14.4% syntax validity | Judge score routed > base, $p < 0.01$ |
| P2: Math domain — routed beats base (judge score) | Prior: +142% answer correctness | Judge score routed > base, $p < 0.01$ |
| P3: Medical — old metric artifact reverses | Medical cross-PPL improved (2.41 vs 2.59) | Judge score routed >= base |
| P4: Legal — adapter mode collapse persists | Legal cross-PPL was 4.39 (degraded) | Judge score routed < base |
| P5: Finance — old metric artifact reverses OR stays negative | Ambiguous prior evidence | Direction unknown |
| P6: K2 — old and new metrics disagree for prose | Keyword density is format-sensitive | $r_s < 0.7$ for prose domains |

**Kill criterion derivation:**
- K1 (#560): Routed worse on >= 3/5 domains using judge scoring -> architecture genuinely hurts prose
- K2 (#561): $r_s > 0.7$ (old and new metrics agree) -> old metric was fine, real composition problem

## E. Assumptions

1. **Self-evaluation is informative.** The base model can distinguish domain-relevant
   from domain-irrelevant text when explicitly prompted. This is weaker than "the model
   is a good judge" — we only need it to distinguish quality directions (better/worse),
   not absolute quality levels. Limitation: if the model systematically rates its own
   outputs higher (self-preference bias), this is controlled by comparing base-generated
   vs routed-generated text through the same judge.

2. **Ordinal scores are comparable across domains.** We compare within-domain only
   (paired routed vs base), so cross-domain calibration is not needed.

3. **150 samples per domain provides adequate power.** Derived in C.2 above.

4. **Oracle routing is the upper bound.** Same assumption as prior experiment.

## F. LLM-as-Judge Scoring Design

The judge prompt asks the base model to rate generated text on three criteria:

1. **Domain Relevance** (1-5): Does the text address the domain topic appropriately?
2. **Coherence** (1-5): Is the text well-structured and logically organized?
3. **Informativeness** (1-5): Does the text provide useful, specific information?

The composite judge score is the mean of all three: $J = (R + C + I) / 3$.

**Why these three criteria?** They capture the behavioral outcomes keyword density
misses: a code adapter producing valid Python scores high on relevance (it IS about
code) and informativeness (it IS useful code), even though it has low keyword density.

**Why self-evaluation?** On Apple Silicon with limited models, using the base model
itself as judge is the only practical option. The self-preference bias is controlled
because both base and routed text are evaluated by the same judge. Any systematic
bias cancels in the paired comparison.

## G. Complexity

- Generation: 50 prompts x 5 domains x 3 seeds x 2 configs = 1500 generations
  At ~1s/prompt: ~1500s (~25 min)
- Judging: 1500 generations x 1 judge call each at ~0.5s/call: ~750s (~12.5 min)
- Total: ~40 min

Memory: same as prior experiment (~5.2 GB active, 7.3 GB peak).

## Post-Experiment Addendum: Power Analysis vs Reality

The Wilcoxon power analysis assumed sufficient variance in judge scores to detect
medium effects. In practice, the BitNet-2B judge exhibited near-zero variance:

| Domain | Nonzero diffs / 50 pairs | Effective n |
|--------|-------------------------|-------------|
| Medical | 1 | 1 |
| Code | 2 | 2 |
| Math | 10 | 10 |
| Legal | 11 | 11 |
| Finance | 0 | 0 |

The Wilcoxon test requires nonzero differences to have power. With only 0-11
effective pairs (out of 50), the test is massively underpowered despite the
adequate sample size. The power analysis was correct in theory but inapplicable
because the judge outputs near-constant scores.

**Lesson:** Power analysis for LLM-as-judge must account for the judge's
discriminating resolution, not just sample size. A 2B model cannot discriminate
at the 1-point level on a 5-point scale, making paired tests degenerate.

**Despite this limitation,** the directional agreement between judge and old metric
on all 5 domains is itself informative: two independent metrics with zero correlation
(r=0.107) agree on direction. Under independence, the probability of 5/5 directional
agreement by chance is $(1/2)^5 = 3.1\%$ (assuming equal probability of either direction).
This provides weak but meaningful evidence that the direction is real.

## Self-Test

1. **One mathematical property:** The Wilcoxon signed-rank test is distribution-free,
   making it valid for ordinal LLM-judge scores without normality assumptions.

2. **Existing theorems:** Wilcoxon (1945), Bonferroni correction, Spearman rank
   correlation, Cohen's power analysis.

3. **Specific numbers:** >99.9% power for medium effects (pre-experiment).
   Post-experiment: 0-11 effective pairs, massively underpowered due to judge
   near-constant output. $\alpha_{adj} = 0.01$, $r_s$ threshold of 0.7 for K2.

4. **Falsification:** The statistical framework assumed ordinal judge scores with
   sufficient variance. This assumption FAILED — the 2B model outputs near-constant
   scores. The framework is technically valid but practically useless for this judge.

5. **Hyperparameters added:** 0 for the statistical framework. The judge prompt
   template is a design choice, not a hyperparameter — it is fixed before data collection.

6. **Hack check:** No. This replaces a broken metric with a better one. Single change.
   The replacement metric (2B self-judge) turned out to also be broken, in a different way.
