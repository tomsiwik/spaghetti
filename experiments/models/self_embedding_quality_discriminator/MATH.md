# Energy-Gap Discriminator for Adapter Composition Quality

## Type: Guided Exploration

**Proven framework:** Autoregressive language models define energy functions over
sequences. Adapter perturbations shift the energy landscape. The energy gap between
base and adapted model is a theoretically motivated signal for composition quality.

**Unknown:** Whether the energy gap achieves sufficient AUC on ternary BitNet-2B-4T
to serve as a practical quality gate for the Evolve system.

## A. Failure Mode Identification

**The disease:** All current quality metrics are broken for adapter composition:
- PPL correlates with task quality at r=0.08 (Finding #178)
- LLM-as-judge outputs constants (Finding #178)
- Keyword density is unreliable
- Math adapter improves correctness 24x but scores WORSE on all proxy metrics (Finding #179)

Without a reliable quality discriminator, the Evolve system (at 20%) cannot gate which
adapter compositions to keep and which to discard. The composition landscape is smooth
and convex (Finding #41), but we have no oracle to navigate it.

**The degenerate behavior:** A quality metric that is uncorrelated with actual task
performance is worse than random — it systematically selects the wrong compositions.

## B. The Right Question (Reframe)

**Wrong question:** "How do we build a better quality metric?"
**Right question:** "What information does the model ALREADY contain about whether
an adapter helps on a given input, and how do we extract it without external judges?"

The answer: the model's own conditional probability p(x|theta) is the richest signal
available. When an adapter helps, the model becomes MORE confident on in-domain inputs
(higher probability = lower energy). When it hurts, confidence drops (lower probability
= higher energy).

## C. Mathematical Framework

### Definition 1: Energy Function

An autoregressive language model with parameters theta defines an energy function
over token sequences x = (x_1, ..., x_T):

    E(x | theta) = -sum_{t=1}^{T} log p(x_t | x_{<t}, theta)

This is the negative log-likelihood (NLL), which equals T * ln(PPL). By the
maximum likelihood principle, the trained model assigns lowest energy to sequences
from its training distribution.

**Source:** This is the standard energy-based interpretation of autoregressive models
(LeCun et al., "A Tutorial on Energy-Based Learning," 2006; Grathwohl et al.,
"Your Classifier is Secretly an Energy-Based Model," ICLR 2020).

### Definition 2: Energy Gap

For base parameters theta_0 and adapter perturbation delta_a:

    Delta_E(x, a) = E(x | theta_0 + delta_a) - E(x | theta_0)

    = sum_{t=1}^{T} [log p(x_t | x_{<t}, theta_0) - log p(x_t | x_{<t}, theta_0 + delta_a)]

This is the per-sequence log-likelihood ratio. Negative gap means the adapter
INCREASED probability (helps). Positive gap means DECREASED probability (hurts).

### Definition 3: Domain-Conditioned Energy Gap

For evaluation samples S_d from domain d, the domain-conditioned energy gap is:

    Delta_E_d(a) = (1/|S_d|) * sum_{x in S_d} Delta_E(x, a) / T(x)

where T(x) is the sequence length (normalizing for length).

### Proposition 1: Energy Gap as Quality Signal

**Claim:** If adapter a was trained on domain d data via NLL minimization, then
Delta_E_d(a) < 0 (the adapter reduces energy on in-domain data).

**Justification:** By construction, NLL training on domain d minimizes
E(x|theta_0+delta_a) for x ~ P_d. If training converges to a better local
minimum than theta_0 alone, the energy gap is negative. This is not guaranteed
(the adapter could overfit to training data while hurting held-out data), but
it is the expected behavior for well-regularized training.

**The discrimination problem:** Given the energy gap Delta_E(x, a) for adapter a
on input x, can we predict whether adapter a improves TASK ACCURACY (not just NLL)
on domain d?

This is where the exploration begins. Finding #178 showed that PPL (which IS the
energy) correlates poorly with task accuracy (r=0.08). However, there are two
key differences in our approach:

1. **Relative gap, not absolute energy.** PPL of the adapted model is a single
   number. The energy GAP between base and adapted is a relative signal that
   cancels out difficulty effects.

2. **Per-sample discrimination, not corpus average.** PPL averages over the
   entire corpus. The energy gap can be computed per-sample, enabling finer
   discrimination.

### Proposition 2: Energy Gap Distribution Separation

If adapter a helps on domain d, the distribution of per-sample energy gaps
{Delta_E(x, a) : x in S_d} should be shifted LEFT (more negative) compared
to a non-helpful adapter b:

    E_x[Delta_E(x, a)] < E_x[Delta_E(x, b)]  when a helps and b doesn't

The AUC of a classifier based on thresholding Delta_E measures how well the
energy gap separates helpful from harmful adapters. AUC = 0.5 means no signal;
AUC > 0.75 means practically useful.

**Note on Neyman-Pearson:** The energy gap IS the log-likelihood ratio between
two model configurations, which is the test statistic that the Neyman-Pearson
lemma identifies as optimal for simple distributional hypothesis tests. However,
our hypothesis is NOT distributional membership (P0 vs P1) — it is whether an
adapter improves task accuracy, a behavioral question. The NP lemma's optimality
guarantee does not apply here. We use the energy gap as an empirically motivated
discriminator inspired by the likelihood ratio structure, not as a theoretically
optimal test. Whether the energy gap correlates with task accuracy is the
empirical question this experiment investigates.

### The Self-Embedding Extension (ATLAS-inspired)

ATLAS uses the model's own hidden-state embeddings (the final-layer representations)
rather than just the scalar log-probability. The d-dimensional embedding h(x|theta)
captures richer information about the model's internal state than a single scalar.

The embedding energy is:

    E_embed(x | theta) = ||h(x | theta) - mu_d||^2

where mu_d is the centroid of domain d embeddings. Inputs that the model
"understands well" cluster tightly; inputs where the adapter fails scatter.

However, this requires computing centroids and is more complex. We test both:
1. **Scalar energy gap** (pure log-probability ratio) — simplest
2. **Embedding distance** (hidden-state clustering) — richer but more complex

## D. Predictions

### Behavioral
1. Energy gap Delta_E separates adapters that improve task accuracy from those
   that don't, achieving AUC > 0.75 on held-out domains.
2. The energy gap signal outperforms a random baseline (AUC > 0.5 + margin).
3. Ranking adapters by energy gap produces the correct task-accuracy ordering
   on at least 2/3 tested domains (with statistically significant rho).

### Quantitative (derived from framework)
- **P1:** Mean energy gap for helpful adapters < 0 (negative = helps)
- **P2:** Mean energy gap for harmful adapters > 0 or less negative
- **P3:** AUC of energy-gap classifier > 0.75 (K566)
- **P4:** Energy gap beats random baseline (K567)
- **P5:** Correct adapter ranking on >=2/3 tested domains with significant rho (K568)
  *(Originally predicted 3/5 domains, but only 3 domains were available for testing.)*

### Null predictions (what "killed" looks like)
- If energy gap AUC <= 0.5: the model's own probabilities contain NO useful
  signal about task quality, confirming Finding #178 that PPL is truly useless
- If embedding distance also fails: self-embeddings carry no composition-quality
  information at d=2560 on ternary models

## E. Assumptions & Breaking Conditions

**A1:** The adapter perturbation is small enough that the energy landscape
changes smoothly (perturbation regime). CONFIRMED: rho < 0.15 at all tested
scales (lora_scale_ablation).

**A2:** Task accuracy correlates with RELATIVE energy gap even if it doesn't
correlate with ABSOLUTE energy (PPL). This is the key hypothesis being tested.
If violated, K566 fails.

**A3:** The ternary quantization of BitNet-2B-4T does not destroy the energy
signal. Ternary weights {-1, 0, 1} discretize the output distribution more
coarsely, which could flatten energy differences. If violated, K567 fails
(random baseline ties or beats).

**A4:** Ground truth task accuracy from lora_scale_ablation (n=50 GSM8K, n=20
MMLU per domain) is sufficient to define "helps" vs "hurts." With n=20, the
95% CI is +/-22%, making marginal cases ambiguous.

## F. Worked Example (d=2560, 3 adapters)

Base model: GSM8K=0.440 (22/50)

Adapter s1.0__sft__math: GSM8K=0.560 (28/50). Delta = +0.120. HELPS on math.
Adapter s1.0__sft__medical: GSM8K=0.420 (21/50). Delta = -0.020. NEUTRAL/HURTS.
Adapter s1.0__ntp__medical: GSM8K=0.420 (21/50). Delta = -0.020. NEUTRAL/HURTS.

For a GSM8K test problem x:
- Compute E(x|base) = -sum log p(x_t | x_{<t}, theta_0) = NLL_base
- Compute E(x|base+math_adapter) = NLL_adapted
- Energy gap = NLL_adapted - NLL_base

Prediction: The math adapter energy gap is more negative (lower energy) on
math problems than the medical adapter's gap. The medical adapter's gap is
near zero or positive on math problems.

If this separation achieves AUC > 0.75 across a set of test problems and
adapter pairs, K566 passes.

## G. Complexity & Architecture Connection

**Compute cost per (adapter, sample) pair:**
- One forward pass through the model: O(T * d^2 * L) where T=seq_len, d=2560, L=layers
- Two forward passes needed (base + adapted)
- For 30 adapters x M test samples: 2 * 30 * M forward passes
- At 172 tok/s base throughput: ~30 adapters * 50 samples * 128 tokens * 2 / 172 = ~2200s
- Optimization: cache base model logits (only compute adapted once per adapter)

**Memory:** One model in memory at a time (~1.7GB). Adapter weights loaded per-adapter.

**Integration with Evolve:** The energy-gap score C(x, a) serves as the quality gate:
- When composing adapters, compute energy gap on held-out validation set
- Reject compositions where Delta_E > 0 (adapter hurts)
- Rank compositions by Delta_E magnitude for the specific domain

## Self-Test (MANDATORY)

1. **What is the ONE mathematical property?**
   The log-likelihood ratio (energy gap) between base and adapted model measures
   how much the adapter shifts the probability mass toward the evaluation domain.

2. **Which existing theorem(s)?**
   Energy-based model framework (LeCun et al., 2006). Log-likelihood ratio
   structure inspired by Neyman-Pearson (1933), though NP's optimality guarantee
   does not apply here (we test task accuracy, not distributional membership).

3. **What specific numbers does the proof predict?**
   Energy gap negative for helpful adapters, positive for harmful. AUC > 0.75
   for binary classification. Ranking correlation > 0.5 on >=2/3 tested domains.

4. **What would FALSIFY the proof?**
   If the energy gap is uncorrelated with task accuracy (AUC ~ 0.5), the
   log-probability signal is useless for composition quality, and the ternary
   model's output distribution is too coarse for energy-based discrimination.

5. **How many hyperparameters?**
   Count: 0 for the energy gap discriminator. The threshold for "helps vs hurts"
   is derived from the base model's energy (Delta_E > 0 or < 0).

6. **Hack check:** No stack of fixes. One signal: energy gap. One question:
   does it discriminate?
