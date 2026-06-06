# Logit-Space Ensembling vs Weight-Space Merging: Mathematical Foundations

## Problem Statement

The N=50 composed model showed -3.67pp MMLU regression relative to the
Qwen2.5-7B base. This regression has two competing explanations:

1. **Weight interference**: LoRA weight deltas destructively interfere when
   summed in weight space, corrupting representations even though individual
   experts are sound.

2. **Distillation quality**: Individual adapters already degrade the base
   model on held-out tasks. Composition merely aggregates N individual harms.

This experiment separates these hypotheses by comparing weight-space merging
(one forward pass with summed weights) against logit-space ensembling
(N independent forward passes, averaged logits). Logit ensembling is
**interference-free by construction** --- each adapter operates independently
on the base model, so any regression under ensembling must come from the
adapters themselves, not from weight-space interactions.

## Notation

| Symbol | Shape / Type | Description |
|--------|-------------|-------------|
| $\theta_0$ | model params | Base model (Qwen2.5-7B, NF4) |
| $\Delta_i = B_i A_i$ | (d, d') per layer | LoRA adapter $i$'s weight delta (rank $r=16$) |
| $N$ | scalar | Number of composed adapters |
| $S$ | set | MMLU evaluation subjects, $\|S\| = 15$ |
| $D_s$ | dataset | Test examples for subject $s$, $\|D_s\| \le 50$ |
| $p$ | string | Formatted MMLU prompt (question + 4 choices) |
| $c \in \{A,B,C,D\}$ | token | Answer choice |
| $z(\cdot; \theta)$ | $(V,)$ | Final-position logit vector under parameters $\theta$ |
| $\alpha$ | scalar | Accuracy (fraction correct) |

## Method 1: Weight-Space Merge

The merged model has effective parameters:

$$\theta_{\text{merge}}^N = \theta_0 + \frac{1}{N} \sum_{i=1}^{N} \Delta_i$$

Prediction on prompt $p$:

$$\hat{c}_{\text{merge}}(p) = \arg\max_{c \in \{A,B,C,D\}} z_c(p;\, \theta_{\text{merge}}^N)$$

This requires **one forward pass** per example. The 1/N scaling ensures the
total perturbation norm stays bounded as N grows.

### Why Merging Can Cause Interference

Even with near-orthogonal LoRA deltas (cos $\approx$ 0.0002), the merged
model's output is a **nonlinear** function of the summed weights. Consider
the output at any layer $l$:

$$h^{l+1} = \text{RMSNorm}\Big(\text{residual} + f\big(W^l h^l + \sum_i \Delta_i^l h^l\big)\Big)$$

where $f$ is SiLU (for MLP) or softmax-attention. Even if
$\langle \Delta_i, \Delta_j \rangle \approx 0$, the nonlinear activation
can create cross-terms:

$$f(W h + \Delta_1 h + \Delta_2 h) \ne f(W h + \Delta_1 h) + f(W h + \Delta_2 h) - f(W h)$$

The residual error from this nonlinear interaction accumulates across layers.
For $L$ layers, the total interference is bounded by (from micro experiments):

$$\|\text{interference}\| \le \alpha_{\text{amp}} \cdot L \cdot \sum_{i<j} |\langle \Delta_i, \Delta_j \rangle_F|$$

where $\alpha_{\text{amp}} = 0.022$ (residual + RMSNorm amplification ratio).
At production cosines this is small, but at $N=50$ with $\binom{50}{2} = 1225$
pairs, even tiny per-pair interference accumulates.

## Method 2: Logit-Space Ensemble

Each adapter runs independently on the base model. The ensemble averages
logits before softmax:

$$\bar{z}(p) = \frac{1}{N} \sum_{i=1}^{N} z(p;\, \theta_0 + \Delta_i)$$

$$\hat{c}_{\text{ens}}(p) = \arg\max_{c \in \{A,B,C,D\}} \bar{z}_c(p)$$

This requires **N forward passes** per example. Crucially, there is
**zero weight-space interaction** between adapters --- each sees the
pristine base model plus only its own delta.

### Why Ensembling Isolates Distillation Quality

If adapter $i$ degrades base model accuracy on subject $s$ (i.e.,
$\alpha(\theta_0 + \Delta_i, D_s) < \alpha(\theta_0, D_s)$), this
degradation appears in the ensemble because adapter $i$'s logits
contribute 1/N of the average.

If all N adapters individually degrade accuracy, the ensemble **must**
degrade accuracy. No amount of averaging can recover signal that no
individual adapter possesses.

Conversely, if individual adapters are neutral or mildly helpful, the
ensemble should match or exceed the base --- the averaging smooths out
noise and preserves signal (the classic ensemble benefit).

## Decomposition of Regression Sources

Define:

- $\alpha_0$ = base model accuracy on $S$
- $\alpha_{\text{merge}}^N$ = accuracy of weight-merged model with $N$ adapters
- $\alpha_{\text{ens}}^N$ = accuracy of logit-ensemble with $N$ adapters

The total regression decomposes as:

$$\underbrace{\alpha_{\text{merge}}^N - \alpha_0}_{\text{total regression}} = \underbrace{\alpha_{\text{ens}}^N - \alpha_0}_{\text{distillation effect}} + \underbrace{\alpha_{\text{merge}}^N - \alpha_{\text{ens}}^N}_{\text{interference effect}}$$

### Diagnostic Decision Matrix

| Observation | Diagnosis | Action |
|-------------|-----------|--------|
| $\alpha_{\text{ens}}^N \approx \alpha_0$ and $\alpha_{\text{merge}}^N \ll \alpha_0$ | **Weight interference** | Fix composition (selective top-k, reweight) |
| $\alpha_{\text{ens}}^N \approx \alpha_{\text{merge}}^N \ll \alpha_0$ | **Distillation quality** | Fix training (better data, longer, curriculum) |
| $\alpha_{\text{ens}}^N < \alpha_0$ but $\alpha_{\text{ens}}^N > \alpha_{\text{merge}}^N$ | **Mixed** (both contribute) | Fix both, prioritize larger factor |
| $\alpha_{\text{ens}}^N > \alpha_0$ and $\alpha_{\text{merge}}^N < \alpha_0$ | **Pure interference** | Composition method is sole culprit |

Quantitative thresholds (applied to the average ensemble-minus-merge gap
across all N values):

- **Gap > 2.0pp**: Weight interference is the dominant factor
- **Gap < 0.5pp**: Distillation quality is the dominant factor
- **0.5pp < Gap < 2.0pp**: Mixed regime

## Expected Outcomes Under Each Hypothesis

### Hypothesis A: Weight Interference Dominates

Under this hypothesis, individual adapters are roughly neutral on MMLU
(they learned their domain without harming general capability). The
regression appears only when weights are summed.

**Predicted pattern:**

| N | Merge delta | Ensemble delta | Gap |
|---|-------------|----------------|-----|
| 5 | ~-0.5pp | ~+0.5pp | ~1pp |
| 10 | ~-1.5pp | ~+0.5pp | ~2pp |
| 25 | ~-2.5pp | ~+0.3pp | ~2.8pp |
| 50 | ~-3.7pp | ~+0.2pp | ~3.9pp |

The merge delta worsens with N (interference accumulates) while ensemble
delta stays near zero or slightly positive (more voters, slight ensemble
benefit).

**Scaling:** Under orthogonal interference, the merge penalty should scale
as $O(\sqrt{N})$ (random walk in weight space, each pair contributing
independent noise). If we observe $\Delta_{\text{merge}} \propto \sqrt{N}$,
this confirms diffuse interference from many small cross-terms.

### Hypothesis B: Distillation Quality Dominates

Under this hypothesis, each adapter individually degrades MMLU performance
(format mismatch, overfitting to synthetic instruction data). The regression
is intrinsic to the adapters.

**Predicted pattern:**

| N | Merge delta | Ensemble delta | Gap |
|---|-------------|----------------|-----|
| 5 | ~-3.0pp | ~-3.2pp | ~+0.2pp |
| 10 | ~-3.5pp | ~-3.5pp | ~0pp |
| 25 | ~-3.6pp | ~-3.6pp | ~0pp |
| 50 | ~-3.7pp | ~-3.7pp | ~0pp |

Both methods regress by similar amounts because the harm is in the
individual adapters. The merge delta is nearly constant with N because
the 1/N weighting limits total perturbation magnitude, and the ensemble
captures the same per-adapter bias.

**Key prediction:** If individual adapters each cause ~-3.7pp regression
(consistent with the 3 adapters tested in pilot50_held_out_eval showing
-3.71pp average), then the ensemble of any subset also regresses by ~-3.7pp
regardless of N.

## Statistical Framework

### Per-Subject Accuracy Standard Error

For subject $s$ with $n_s$ questions and true accuracy $p_s$:

$$\text{SE}(\hat{\alpha}_s) = \sqrt{\frac{p_s(1 - p_s)}{n_s}}$$

At $n_s = 50$ and $p_s = 0.70$: SE $\approx$ 0.065 (6.5pp).

### Aggregate Accuracy (Micro-Averaged)

$$\hat{\alpha} = \frac{\sum_{s \in S} \text{correct}_s}{\sum_{s \in S} n_s}$$

With $|S| = 15$ subjects at $n_s = 50$: total $n = 750$ questions.

$$\text{SE}(\hat{\alpha}) = \sqrt{\frac{p(1-p)}{n}} \approx \sqrt{\frac{0.21}{750}} \approx 0.017 \text{ (1.7pp)}$$

### Gap (Ensemble - Merge) Standard Error

The gap $\gamma^N = \alpha_{\text{ens}}^N - \alpha_{\text{merge}}^N$ is
computed on the **same 750 questions** (paired comparison). By McNemar's
theorem, the SE of a paired accuracy difference is:

$$\text{SE}(\gamma) = \sqrt{\frac{p_{01} + p_{10}}{n}}$$

where $p_{01}$ is the fraction of questions where merge is correct but
ensemble is wrong, and $p_{10}$ the reverse. In practice, with
$p_{01} + p_{10} \approx 0.10$ (rough estimate):

$$\text{SE}(\gamma) \approx \sqrt{\frac{0.10}{750}} \approx 0.012 \text{ (1.2pp)}$$

This gives sufficient resolution to distinguish the 2.0pp threshold
(gap / SE $\approx$ 1.7) from the 0.5pp threshold (gap / SE $\approx$ 0.4).
For a definitive result, the gap must exceed ~2.3pp (2-sigma from zero).

### McNemar Test for Significance

For each N, the formal test of whether merge and ensemble have different
accuracy uses McNemar's chi-squared:

$$\chi^2 = \frac{(b - c)^2}{b + c}$$

where $b$ = number of questions correct under ensemble but wrong under merge,
$c$ = the reverse. Under H0 (no difference), $\chi^2 \sim \chi^2(1)$.

Reject H0 at $\alpha = 0.05$ if $\chi^2 > 3.84$.

**Required discordant pairs for significance at each effect size:**

| True gap | $b - c$ needed | $b + c$ needed (power 0.80) |
|----------|---------------|----------------------------|
| 1pp | ~7.5 | ~56 |
| 2pp | ~15 | ~56 |
| 3pp | ~22.5 | ~56 |

With 750 questions, even 1pp difference produces ~7.5 discordant pairs on
net, which is detectable if $b + c \ge 56$ (i.e., $\ge 7.5\%$ of questions
flip between methods --- plausible).

## N-Scaling Analysis

The experiment sweeps N = {5, 10, 25, 50}. This enables testing scaling laws:

### Under Weight Interference

If interference causes the merge penalty:

$$\alpha_{\text{merge}}^N - \alpha_0 = -\beta \cdot N^\gamma$$

where $\gamma = 0.5$ for diffuse pairwise interference (random walk),
$\gamma = 1.0$ for additive interference (each adapter adds fixed harm).
Fit $\gamma$ from the 4 data points via log-log regression.

### Under Distillation Quality

If distillation quality causes regression:

$$\alpha_{\text{ens}}^N - \alpha_0 \approx \text{const} \quad \forall N$$

The ensemble delta should be roughly constant with N, because averaging
N copies of the same bias preserves the bias. (The variance decreases as
$1/N$, but the mean bias is preserved.)

Slight deviation: if adapters have heterogeneous biases (some positive, some
negative), the ensemble approaches the mean adapter effect as N grows. We'd
expect convergence to some stable value by N=25.

## Computational Cost Analysis

### Weight Merge

For each N value:
- Load base model: ~30s (once, reused)
- Compose N adapters: ~10s (PEFT add_weighted_adapter)
- Evaluate 15 subjects x 50 questions: ~375s (0.5s/question)
- Total per N: ~385s

### Logit Ensemble

For each N value:
- Load base model: ~30s (once, reused)
- For each of N adapters:
  - Load adapter: ~3s
  - Forward pass on 750 questions: ~375s
  - Unload adapter: ~2s
- Total per N: ~30 + N x 380s

| N | Merge time | Ensemble time | Ensemble/Merge ratio |
|---|-----------|---------------|---------------------|
| 5 | ~6.4min | ~31.7min | 5.0x |
| 10 | ~6.4min | ~63.3min | 9.9x |
| 25 | ~6.4min | ~158.3min | 24.7x |
| 50 | ~6.4min | ~316.7min | 49.5x |

**K2 kill criterion**: Ensemble overhead must be < 10x weight merge. This
is satisfied at N=5 (5.0x) and N=10 (9.9x), but violated at N=25 and N=50.

**Practical mitigation**: The overhead is inherent to ensembling (N forward
passes vs 1). K2 asks whether ensembling is "practical as a diagnostic
baseline," not whether it's practical for production. At N=5, the 5x overhead
is well within K2. At N=50, the 50x overhead violates K2 but the diagnostic
value remains: if N=5 and N=10 already show the pattern, N=50 is confirmatory.

**Optimization**: The script accumulates logits across adapters, avoiding
storing N x 750 logit vectors. Memory overhead is O(vocab_size) regardless
of N. Each adapter is loaded, scored, and unloaded sequentially.

**Total estimated runtime**: ~30 + 4 x 6.4 + (5+10+25+50) x 380/60 $\approx$
56 + 570 $\approx$ 626 minutes $\approx$ 10.4 hours. This is expensive but
bounded.

**SMOKE_TEST mode**: With --max-per-subject 5 (50 questions total instead
of 750), total runtime drops to ~1/15th: ~42 minutes.

## Assumptions

1. **MMLU logit-space scoring is consistent.** Both methods use the same
   log-probability scoring: $\arg\max_{c} \log P(c | p; \theta)$. The
   ensemble averages raw logits (before softmax), which is mathematically
   equivalent to geometric mean of probabilities (log-linear model).

2. **Adapter loading/unloading is clean.** Each adapter is loaded onto a
   fresh copy of the base model (via PeftModel.from_pretrained), ensuring
   no weight contamination between ensemble members.

3. **4-bit quantization does not differentially affect methods.** Both
   weight merge and logit ensemble operate on the same NF4-quantized base.
   The merge adds float16 deltas before the quantized forward pass; the
   ensemble adds each float16 delta independently. Any quantization artifact
   affects both equally.

4. **Equal weighting (1/N) is the right comparison.** The weight merge uses
   1/N scaling per adapter, and the logit ensemble averages with equal 1/N
   weights. This ensures a fair comparison. Optimal weighting is a separate
   research question (exp_top_k_embedding_routing).

5. **15 MMLU subjects provide sufficient coverage.** Selected subjects
   span science (algebra, anatomy, astronomy, physics), humanities (ethics),
   CS (computer science, security), and medicine (clinical knowledge,
   college medicine). Not all 57 subjects are tested due to time constraints.

6. **Top-ranked adapters (by training PPL) are used.** The adapter selection
   ranks by contaminated PPL improvement. This biases toward adapters that
   showed strongest training signal, which is the relevant subset for
   diagnosing the N=50 regression.

## Worked Numerical Example

Consider a single MMLU question with 4 choices, base logits:

$$z_0 = [2.1, 0.3, -0.5, 1.8] \quad \text{(correct answer: A)}$$

**Under weight merge** (N=2 adapters, 1/2 weight each):

Adapter 1 (math): shifts toward C (wrong): $\Delta z_1 = [-0.4, 0.1, 0.8, -0.2]$
Adapter 2 (medical): shifts toward D (wrong): $\Delta z_2 = [-0.3, -0.1, 0.2, 0.6]$

Merged logits: $z_{\text{merge}} = z_0 + \frac{1}{2}(\Delta z_1 + \Delta z_2) = [1.75, 0.30, 0.0, 2.0]$

Prediction: D (wrong). The two off-domain adapter biases combined to flip
the answer from A to D.

**Under logit ensemble** (N=2, averaged):

Adapter 1 logits: $z_0 + \Delta z_1 = [1.7, 0.4, 0.3, 1.6]$ -- prediction: A (correct)
Adapter 2 logits: $z_0 + \Delta z_2 = [1.8, 0.2, -0.3, 2.4]$ -- prediction: D (wrong)

Ensemble average: $\bar{z} = \frac{1}{2}([1.7, 0.4, 0.3, 1.6] + [1.8, 0.2, -0.3, 2.4]) = [1.75, 0.30, 0.0, 2.0]$

Prediction: D (wrong).

**Key observation**: In this linear case, the ensemble and merge give
**identical** logits when adapter effects are additive! The difference
emerges only through **nonlinear** interactions in the model (layer norms,
activations, attention softmax).

This illustrates why the experiment is informative: if merge and ensemble
give the same results, the regression is in the linear (additive) adapter
effects, not in nonlinear interactions. A gap between them quantifies the
nonlinear interference.

### Correction: Single-Adapter vs Merged Multi-Adapter

The worked example above shows a subtlety. In weight merge, the effective
delta per adapter is $\Delta_i / N$ (because of 1/N weighting). In the
ensemble, each adapter contributes its full delta $\Delta_i$ to its own
forward pass. This is **not** equivalent to merge when the model is
nonlinear.

More precisely:

$$z_{\text{merge}}(p) = f\big(\theta_0 + \frac{1}{N}\sum_i \Delta_i,\, p\big)$$

$$\bar{z}_{\text{ens}}(p) = \frac{1}{N}\sum_i f(\theta_0 + \Delta_i,\, p)$$

These differ because:

1. Merge sees each adapter at 1/N strength but all simultaneously.
2. Ensemble sees each adapter at full strength but independently.

The merge uses diluted but interacting effects. The ensemble uses undiluted
but non-interacting effects. If the ensemble also regresses, it means
even a single adapter at full strength hurts MMLU --- the distillation
quality problem.

## Connection to Prior Results

### pilot50_held_out_eval (-3.71pp individual)

Three individually-tested adapters (math, medical, python) showed -3.71pp
average regression on their MMLU subsets. This suggests the distillation
quality hypothesis (H_B) may be dominant. However:

- Only 3 of 50 adapters were tested
- These were tested on their **related** MMLU subjects only
- The -3.67pp composed regression is on a **different** set of 15 subjects

This experiment provides a clean comparison on the **same** subjects.

### Structural orthogonality (cos $\approx$ 0.0002)

Near-perfect orthogonality suggests interference should be minimal. This
**favors Hypothesis B** (distillation quality). If weights are orthogonal,
merging should be nearly lossless, and the ensemble-merge gap should be
small.

### Micro safety bound (alpha = 0.022)

The amplification ratio of 0.022 means each layer's nonlinear interference
is dampened 45x by residual+RMSNorm. Even if per-layer interference exists,
it's heavily suppressed. This again **favors Hypothesis B**.

**Prior evidence leans toward distillation quality being the dominant cause.**
This experiment will confirm or refute this with direct evidence.
