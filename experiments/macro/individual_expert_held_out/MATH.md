# Individual Expert Held-Out Evaluation: Mathematical Framework

## Problem Statement

The N=50 composed model showed average delta = -3.67pp on held-out MMLU
relative to the Qwen2.5-7B base. This regression has two candidate explanations:

1. **Distillation quality failure**: Each adapter individually harms base model
   performance on non-training-domain tasks. The regression is intrinsic to
   each adapter, and composition merely aggregates N individual harms.

2. **Composition interference**: Each adapter individually is neutral or
   beneficial, but composing 50 in weight space creates destructive
   interference. The regression emerges from the composition, not the parts.

This experiment isolates the cause by testing adapters individually.

## Notation

| Symbol | Description | Typical value |
|--------|-------------|---------------|
| $\theta_0$ | Base model parameters | Qwen2.5-7B (NF4) |
| $\Delta\theta_i$ | LoRA adapter $i$ parameters | rank-16, all modules |
| $S$ | Set of MMLU subjects used for evaluation | \|S\| = 57 (all available) |
| $D_s$ | Test set for MMLU subject $s$ | 100 -- 1534 questions |
| $N$ | Number of adapters tested | 20 (top by training PPL) |

## Individual Accuracy

For adapter $i$ on MMLU subject $s$, log-probability scoring gives:

$$\hat{y}(p; \theta) = \arg\max_{c \in \{A,B,C,D\}} \log P(c \mid p; \theta)$$

Individual adapter accuracy on subject $s$:

$$a_i^s = \frac{1}{|D_s|} \sum_{(p,y) \in D_s} \mathbb{1}\big[\hat{y}(p; \theta_0 + \Delta\theta_i) = y\big]$$

Base accuracy on subject $s$:

$$a_0^s = \frac{1}{|D_s|} \sum_{(p,y) \in D_s} \mathbb{1}\big[\hat{y}(p; \theta_0) = y\big]$$

Per-subject delta for adapter $i$:

$$\delta_i^s = a_i^s - a_0^s$$

## Aggregate Metrics

**Overall accuracy for adapter $i$** (micro-averaged across all subjects):

$$a_i = \frac{\sum_{s \in S} \text{correct}_i^s}{\sum_{s \in S} |D_s|}$$

**Overall delta for adapter $i$**:

$$\delta_i = a_i - a_0$$

**Mean individual delta across all adapters**:

$$\bar{\delta} = \frac{1}{N} \sum_{i=1}^{N} \delta_i$$

This is the primary diagnostic metric. It answers: on average, does adding a
single adapter help or hurt base model performance across diverse MMLU subjects?

## Kill Criteria (Formal)

**K1 (composition interference)**: $\bar{\delta} > -1\text{pp}$

If individual adapters are roughly neutral ($\bar{\delta} > -1\text{pp}$), but
composed model shows $-3.67\text{pp}$, then the regression source is
composition. The gap $\Delta_{\text{interference}} = -3.67 - \bar{\delta}$
quantifies the composition penalty.

**K2 (distillation memorization)**: $\bar{\delta} < -3\text{pp}$

If individual adapters already cause $> 3\text{pp}$ regression on their own,
distillation has created adapters that actively harm generalization. Composing
50 of them naturally accumulates this harm.

**Decision matrix**:

| $\bar{\delta}$ range | Diagnosis | Action |
|----------------------|-----------|--------|
| $> -1\text{pp}$ | Composition interference | Fix composition (top-k, selective routing) |
| $[-3, -1]\text{pp}$ | Inconclusive | Both distillation and composition contribute |
| $< -3\text{pp}$ | Distillation memorization | Fix training (more data, better teacher, curriculum) |

## Decomposing the Regression

The -3.67pp composed model regression can be decomposed into:

$$\delta_{\text{composed}} = \bar{\delta} + \Delta_{\text{interference}}$$

where $\Delta_{\text{interference}}$ is the composition-specific penalty
(cross-adapter weight-space interference, dilution, etc.).

If $\bar{\delta} \approx 0$ and $\delta_{\text{composed}} = -3.67\text{pp}$,
then $\Delta_{\text{interference}} \approx -3.67\text{pp}$ -- the entire
regression is from composition.

If $\bar{\delta} \approx -3.67\text{pp}$, then $\Delta_{\text{interference}}
\approx 0$ -- the composition adds no additional harm; each adapter is
independently harmful.

## Statistical Considerations

### Per-subject standard error

For a subject with $n$ questions and base accuracy $p \approx 0.70$:

$$\text{SE}(a^s) = \sqrt{\frac{p(1-p)}{n}} \approx \sqrt{\frac{0.21}{n}}$$

At $n = 50$ (with --max-per-subject 50): $\text{SE} \approx 0.065$ (6.5pp).

Individual subject-level comparisons will be noisy. The power comes from
aggregating across all subjects.

### Overall standard error

With $N = 20$ adapters, the SE of $\bar{\delta}$ is:

$$\text{SE}(\bar{\delta}) = \frac{\sigma_\delta}{\sqrt{N}}$$

where $\sigma_\delta$ is the standard deviation of individual adapter deltas.
If $\sigma_\delta \approx 3\text{pp}$ (plausible for diverse adapters):

$$\text{SE}(\bar{\delta}) \approx \frac{3}{\sqrt{20}} \approx 0.67\text{pp}$$

This gives sufficient resolution to distinguish K1 ($> -1\text{pp}$) from K2
($< -3\text{pp}$) -- the gap between thresholds is $2\text{pp} \approx 3$
standard errors.

### Bootstrap confidence interval

To report uncertainty on $\bar{\delta}$, bootstrap resample adapters (with
replacement, $B = 1000$ iterations):

$$\bar{\delta}^{(b)} = \frac{1}{N} \sum_{i \in I_b} \delta_i, \quad b = 1, \ldots, B$$

where $I_b$ is a bootstrap sample of adapter indices. The 95% CI is
$[\bar{\delta}^{(0.025)}, \bar{\delta}^{(0.975)}]$.

## Domain-Matched vs Cross-Domain Analysis

A secondary analysis decomposes each adapter's delta into:

- **In-domain delta**: accuracy on MMLU subjects that match the adapter's
  training domain (using the mapping from pilot50_held_out_eval)
- **Out-of-domain delta**: accuracy on all other MMLU subjects

If in-domain $\delta_i^{\text{in}} > 0$ but out-of-domain $\delta_i^{\text{out}} < 0$,
the adapter learned its domain but actively interferes with unrelated knowledge.
This would suggest the training creates overly strong biases.

## Computational Cost

- Model load: ~30s (NF4 quantized, 4GB VRAM)
- Per question: ~0.5s (forward pass for log-prob scoring)
- Per adapter: 20 subjects x 50 questions x 0.5s = 500s ~ 8 min
  Plus model reload (~30s per adapter for fresh base): ~9 min
- 20 adapters: ~180 min ~ 3 hours
- Base model eval: 20 subjects x 50 questions x 0.5s = 500s ~ 8 min

**Total: ~3 hours, ~$0.50 at $0.16/hr A5000**

With max_per_subject=50, total questions = 20 x 50 = 1000 per model.
For 21 models (1 base + 20 adapters) = 21,000 forward passes.

## What This Does NOT Test

1. Composition quality (tested separately by exp_selective_composition_mmlu
   and exp_small_n_held_out_eval)
2. Domain-specific improvement (pilot50_held_out_eval already tests this)
3. Training data quality directly (would need comparison with different data)
4. The effect of rank, learning rate, or training steps
5. Whether retrained adapters (new rank-16 all-modules config) differ
