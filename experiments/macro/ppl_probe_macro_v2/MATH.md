# PPL-Probe Weighted Composition at Macro Scale: Mathematical Foundations

## 1. Problem Statement

We have a frozen base model $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and $N = 5$
LoRA adapters $\Delta_i = B_i A_i$ where $B_i \in \mathbb{R}^{d_{\text{out}} \times r}$,
$A_i \in \mathbb{R}^{r \times d_{\text{in}}}$, $r = 16$, $d = 3584$ (Qwen2.5-7B hidden dim).

Equal-weight composition $W = W_0 + \sum_{i=1}^{N} \Delta_i$ produces catastrophic PPL
(trillions at N=5). We seek weights $\alpha_i \geq 0$, $\sum_i \alpha_i = 1$, such that:

$$W_\alpha = W_0 + \sum_{i=1}^{N} \alpha_i \Delta_i$$

produces quality approaching single-expert performance on relevant domains while
maintaining generality on irrelevant domains.

## 2. PPL-Probe Weighting Mechanism

### 2.1 Probe Buffer

For each MMLU subject $s$, we maintain a probe buffer $\mathcal{P}_s$ of $n = 10$ examples
drawn from the test set (answer-labeled). This is a held-out calibration set, NOT the
evaluation set.

### 2.2 Per-Adapter PPL Scoring

For adapter $i$ and subject $s$, compute answer-only PPL on the probe buffer:

$$\text{PPL}_i(s) = \exp\left(\frac{1}{|\mathcal{P}_s|} \sum_{(x,y) \in \mathcal{P}_s} -\log p_{W_0 + \Delta_i}(y | x)\right)$$

where $p_{W_0 + \Delta_i}(y | x)$ is the probability of the correct answer token $y$
given the question $x$ under model $W_0 + \Delta_i$.

**Key**: this requires $N+1$ forward passes per probe example (base + each adapter individually).
We also compute base PPL for reference:

$$\text{PPL}_{\text{base}}(s) = \exp\left(\frac{1}{|\mathcal{P}_s|} \sum_{(x,y) \in \mathcal{P}_s} -\log p_{W_0}(y | x)\right)$$

### 2.3 Softmax Weight Computation

Convert PPL scores to composition weights via temperature-scaled softmax over negative
log-PPL (lower PPL = higher weight):

$$\alpha_i(s) = \frac{\exp\left(-\log \text{PPL}_i(s) / \tau\right)}{\sum_{j=1}^{N} \exp\left(-\log \text{PPL}_j(s) / \tau\right)}$$

**Temperature selection**: Micro results (exp_ppl_probe_temperature_sensitivity) showed
$\tau = 0.5$ is optimal for synthetic losses, but the specific value is not transferable
to real PPL scales. We test $\tau \in \{0.1, 0.5, 1.0, 2.0\}$ at macro scale.

**Simplification**: $-\log \text{PPL}_i = -\frac{1}{n} \sum_j -\log p_i(y_j | x_j)
= \frac{1}{n} \sum_j \log p_i(y_j | x_j)$, i.e., the mean log-probability of the
correct answer. Higher log-prob = lower PPL = higher weight.

### 2.4 Composition

Given weights $\alpha(s)$ for subject $s$:

$$W_\alpha(s) = W_0 + \sum_{i=1}^{N} \alpha_i(s) \cdot \Delta_i$$

This is applied across ALL LoRA-adapted layers simultaneously with the SAME weight
vector $\alpha$. Per-layer weights would require $L \times N$ probing passes and are
not justified at $N=5$.

## 3. Conditions

### 3.1 Base (C0)
No adapters. $W = W_0$.

### 3.2 Equal-Weight (C1)
$\alpha_i = 1/N$ for all $i$. This is the current default and the broken baseline.
Note: at N=5, each adapter contributes with weight 0.2. At N=50 pilot this produced
-3.67pp regression; at N=5 with these specific adapters it produced trillions PPL.

**Critical note on scaling**: The catastrophic PPL came from UNSCALED addition
($\alpha_i = 1.0$ for all $i$, i.e., full sum). Equal-weight with $\alpha_i = 1/N$
may already fix part of the problem. We test BOTH:
- C1a: $\alpha_i = 1/N = 0.2$ (scaled equal-weight)
- C1b: $\alpha_i = 1.0$ (unscaled, reproducing the catastrophe)

### 3.3 PPL-Probe Weighted (C2)
$\alpha_i(s) = \text{softmax}(-\log \text{PPL}_i(s) / \tau)$ as derived above.
Tested at $\tau \in \{0.1, 0.5, 1.0, 2.0\}$.

### 3.4 Top-1 Selection (C3)
Select the single adapter with lowest PPL on the probe:

$$k^*(s) = \arg\min_i \text{PPL}_i(s)$$
$$W = W_0 + \Delta_{k^*}$$

This is the simplest routing strategy.

### 3.5 Best Single Adapter (C4) -- Oracle-ish
For each subject, evaluate ALL 5 adapters individually and pick the one with highest
MMLU accuracy. This is an oracle upper bound on single-adapter selection.

## 4. Computational Cost Analysis

### 4.1 Probe Phase (Per Subject)

For $N = 5$ adapters and $n = 10$ probe examples:
- Load base model: 1x (amortized)
- For each adapter: load adapter, run 10 forward passes, unload
- Total forward passes: $N \times n = 50$ per subject
- At ~20ms per forward pass (512 tokens, NF4): $50 \times 0.02 = 1.0$ seconds per subject

For 57 MMLU subjects: $57 \times 1.0 = 57$ seconds for the probe phase.

**Optimization**: Load each adapter once, evaluate across ALL 57 subjects' probe buffers.
Total: $5 \times 57 \times 10 = 2850$ forward passes = ~57 seconds (same, just reordered).

### 4.2 Evaluation Phase (Per Subject)

For each condition (6 total: base, 2x equal-weight, 3x PPL-probe temps, top-1):
- Compose model with computed weights (PEFT weighted_adapter)
- Evaluate on held-out examples (~40-50 per subject after removing probe)
- Total per condition: $57 \times 50 = 2850$ forward passes = ~57 seconds

Total: $6 \times 57 = 342$ seconds for evaluation.

### 4.3 Total Runtime Estimate

| Phase | Forward Passes | Time (est.) |
|-------|---------------|-------------|
| Probe profiling (5 adapters x 57 subjects x 10) | 2,850 | ~5 min |
| Base eval (57 x ~50) | 2,850 | ~5 min |
| C1a eval (scaled equal-weight) | 2,850 | ~5 min |
| C1b eval (unscaled equal-weight) | 2,850 | ~5 min |
| C2 eval (PPL-probe, 4 temps) | 11,400 | ~20 min |
| C3 eval (top-1 selection) | 2,850 | ~5 min |
| C4 eval (best single, 5 adapters x 57 subj) | 14,250 | ~25 min |
| **Total** | **~40,000** | **~70 min** |

With overhead (model loading, PEFT adapter swap, GC): ~2 hours.

**Key optimization**: For C4, we can reuse the probe PPL scores as the selection
criterion rather than separately evaluating all 5 adapters on all eval data. The
probe IS the selection mechanism. This eliminates C4 as a separate eval pass.

### 4.4 Latency Analysis (K2 Kill Criterion)

Per-query PPL-probe latency at inference:
- Run each of $N=5$ adapters on $n=10$ probe examples: $5 \times 10 = 50$ forward passes
- At ~20ms each: $50 \times 0.02 = 1.0$ seconds PER QUERY

This exceeds the 100ms K2 threshold. However, this is an amortized cost:
- The probe scores only need to be computed ONCE per subject/domain
- At serving time, we cache the weights $\alpha(s)$ for each known subject
- For new/unknown subjects: fall back to equal-weight or nearest-subject weights
- Actual per-query overhead: weight lookup + single PEFT merge = <1ms

**K2 clarification**: The 100ms threshold applies to per-query SERVING latency, not
calibration-time latency. Calibration is a one-time offline cost.

## 5. Metrics

### 5.1 Primary: MMLU Accuracy (0-shot)

For each condition $c$ and subject $s$:

$$\text{Acc}_c(s) = \frac{\text{correct}_{c,s}}{\text{total}_{c,s}}$$

Aggregate: micro-average across all subjects (weighted by subject size).

### 5.2 Delta vs Base

$$\delta_c = \text{Acc}_c - \text{Acc}_{\text{base}} \quad \text{(in percentage points)}$$

### 5.3 Kill Criteria Metrics

**K1 (routing quality)**: For each adapter $i$, compute PPL on its training-domain
probe. If $\text{PPL}_{\text{composed}}(s) > 2 \times \text{PPL}_i(s)$ for the
adapter's home domain $s$, the routing failed for that domain.

$$\text{K1} = \frac{|\{s : \text{PPL}_{\text{composed}}(s) > 2 \times \min_i \text{PPL}_i(s)\}|}{|\text{home domains}|} < 50\%$$

**K2 (latency)**: Per-query serving latency. Measured as wall-clock time for:
1. Look up cached $\alpha(s)$ (or compute if uncached)
2. PEFT weighted merge
3. Single forward pass

Target: <100ms total.

**K3 (improvement)**: $\max_\tau \text{Acc}_{\text{probe},\tau} - \text{Acc}_{\text{equal}} > 2\text{pp}$

### 5.4 Probe-Oracle Correlation

Pearson $r$ between PPL-probe weights and oracle weights (oracle = accuracy-optimal
weights found by grid search or best-single selection):

$$r = \text{corr}(\alpha_{\text{probe}}, \alpha_{\text{oracle}})$$

At micro scale this was $r = 0.990$. We expect some degradation at macro but $r > 0.5$
would still indicate useful signal.

## 6. Worked Example

**Setup**: 5 adapters (bash, math, medical, python, sql), MMLU subject "college_mathematics".

**Step 1**: Compute PPL on 10 probe examples for each adapter:

| Adapter | PPL on math probes | -log(PPL) |
|---------|-------------------|-----------|
| bash | 8.2 | -2.10 |
| math | 3.1 | -1.13 |
| medical | 7.5 | -2.01 |
| python | 6.9 | -1.93 |
| sql | 9.4 | -2.24 |

**Step 2**: Compute softmax weights at $\tau = 1.0$:

| Adapter | score = -log(PPL)/1.0 | exp(score - max) | weight |
|---------|----------------------|------------------|--------|
| bash | -2.10 | $e^{-0.97}$ = 0.379 | 0.104 |
| math | -1.13 | $e^{0}$ = 1.000 | 0.275 |
| medical | -2.01 | $e^{-0.88}$ = 0.415 | 0.114 |
| python | -1.93 | $e^{-0.80}$ = 0.449 | 0.124 |
| sql | -2.24 | $e^{-1.11}$ = 0.329 | 0.091 |
| | | **sum** = 3.636 | |

Math adapter gets weight 0.275 (highest), sql gets 0.091 (lowest).
Compare equal-weight: 0.200 each.

**Step 3**: At $\tau = 0.5$ (sharper):

| Adapter | score = -log(PPL)/0.5 | exp(score - max) | weight |
|---------|----------------------|------------------|--------|
| bash | -4.20 | $e^{-1.94}$ = 0.144 | 0.054 |
| math | -2.26 | $e^{0}$ = 1.000 | 0.377 |
| medical | -4.02 | $e^{-1.76}$ = 0.172 | 0.065 |
| python | -3.86 | $e^{-1.60}$ = 0.202 | 0.076 |
| sql | -4.48 | $e^{-2.22}$ = 0.109 | 0.041 |
| | | **sum** = 2.654 | |

Math adapter gets weight 0.377 at $\tau = 0.5$ vs 0.275 at $\tau = 1.0$.
More discriminative, as expected from micro temperature sensitivity results.

**Step 4**: Compose $W = W_0 + 0.377 \cdot \Delta_{\text{math}} + 0.076 \cdot \Delta_{\text{python}} + \ldots$

**Step 5**: Evaluate accuracy on held-out math questions.

## 7. Assumptions and Risks

### Assumptions
1. **PPL on 10 answer-only examples discriminates adapters at macro scale.** Micro
   showed $r = 0.990$ but with random weights and synthetic tasks. Real adapters have
   more subtle differences.
2. **Per-subject weights are stable across examples within a subject.** The probe
   buffer may not represent the full distribution of a subject's questions.
3. **Linear weight composition is valid.** $W_0 + \sum \alpha_i \Delta_i$ assumes
   adapter effects combine linearly. Nonlinearities in the model may invalidate
   this at large $\alpha$ but at $\alpha \leq 0.4$ (softmax with 5 adapters) this
   should be safe.
4. **The 5 adapters span useful knowledge.** bash, math, medical, python, sql cover
   a narrow slice of MMLU's 57 subjects. Most subjects have no relevant adapter.
5. **0-shot MMLU is a valid eval.** Standard MMLU uses 5-shot. Our adapters were
   trained on instruction data, so 0-shot may understate their benefit.

### Risks
1. **The catastrophic PPL may come from a single bad adapter (sql).** Dropout
   robustness showed sql removal drops PPL from 31.6T to 17,683. PPL-probe
   weighting may just be learning to zero-out sql, not genuinely routing.
   We can detect this by examining the weight distributions.
2. **MMLU subjects without a relevant adapter.** For 40+ of 57 subjects, none
   of the 5 adapters are domain-relevant. PPL-probe might still discriminate
   (some adapters may generalize), or it might produce near-uniform weights
   (which is the correct behavior -- fall back to near-base).
3. **Probe-eval contamination.** We use the FIRST 10 test examples as probes
   and the remaining as eval. This is standard calibration-eval split. The
   contamination risk is that probe performance correlates with eval performance
   for trivial reasons (e.g., both sets have the same difficulty). This is
   mitigated by the per-adapter comparison (all adapters see the same probe).
