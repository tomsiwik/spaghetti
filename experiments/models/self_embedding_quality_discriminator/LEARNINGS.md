# LEARNINGS: Self-Embedding Energy Discriminator

## Core Finding

The energy gap (NLL_adapted - NLL_base) discriminates helpful from harmful adapters
at AUC=0.942 on math (GSM8K), resolving the quality-gate problem that blocked Evolve.
The key insight: **relative NLL works where absolute NLL fails** because the
per-sample difficulty cancellation transforms an uninformative metric (PPL, r=0.08)
into a strong discriminator (rho=0.701).

## Why This Happened

### The PPL Paradox Resolution

Finding #178 showed PPL correlates at r=0.08 with task accuracy. This seemed to
kill NLL-based quality scoring entirely. But the failure was in *absolute* NLL,
not *relative*. The energy gap cancels sample difficulty:

    Delta_E(x, a) = NLL(x|theta+delta_a) - NLL(x|theta_0)

Hard samples have high NLL under both models; the gap isolates the adapter's
contribution. This is why embedding distance fails (AUC=0.568) — cosine distance
in hidden-state space doesn't cancel difficulty, it adds noise.

This pattern is well-documented in the compression literature. Jaiswal et al.
(arxiv 2409.11233) showed SparseGPT and Wanda maintain perplexity at 50% sparsity
while suffering significant task degradation — perplexity masks real performance
changes. Velickovic et al. (arxiv 2601.22950) proved formally that for decoder-only
Transformers, confident-correct predictions on one sequence imply existence of
wrong predictions with near-zero perplexity. Both confirm: absolute PPL is
fundamentally disconnected from task accuracy.

### Why Relative Signals Work Better

The energy gap has the structure of a log-likelihood ratio test statistic
(Neyman-Pearson, 1933). While NP's optimality guarantee doesn't apply here
(we test task accuracy, not distributional membership), the structural insight
transfers: differencing cancels shared nuisance factors (sample difficulty,
base model biases), isolating the adapter-specific signal.

This is analogous to paired t-tests outperforming unpaired ones when within-pair
correlation is high — the base model's per-sample NLL is highly correlated with
the adapted model's NLL, so the difference has much lower variance than either
absolute value.

## Confirming Evidence

- **LeCun et al. (2006)** — Energy-based learning framework establishes the
  mathematical foundation: lower energy = higher probability = better model fit.
- **Grathwohl et al. (arxiv 1912.03263, ICLR 2020)** — "Your Classifier is
  Secretly an Energy-Based Model" shows classifiers can be reinterpreted as
  energy functions, validating energy-gap reasoning for discriminative tasks.
- **ATLAS (itigges22/ATLAS)** — Self-referential energy scoring for code quality.
  Uses the model's own confidence as quality signal, same principle as our energy gap.
- **Jaiswal et al. (arxiv 2409.11233)** — Compression preserves PPL but degrades
  tasks, confirming absolute PPL is unreliable. Proposes JS-Divergence as
  alternative — a relative metric, consistent with our finding.
- **Our Finding #179** — Math adapter produces 24x more correct answers despite
  lower judge scores. The energy gap correctly identifies this adapter as helpful
  (gap=-0.274), resolving the format-accuracy tradeoff that confused other metrics.

## Contradicting Evidence

- **Velickovic et al. (arxiv 2601.22950)** — Proves perplexity fundamentally cannot
  distinguish right from wrong predictions in Transformers. While our energy *gap*
  sidesteps this by being relative, the underlying theorem suggests caution: at
  sufficient scale/sequence length, even relative NLL signals could degrade.
- **Our code domain results** — 0/17 adapters helped on code MMLU, producing a
  degenerate case where AUC=0.500 by definition. The energy gap has nothing to
  discriminate when no adapter helps. This isn't a failure of the discriminator
  but a limitation: it requires positive examples to be useful.
- **Medical fragility** — AUC=0.938 from a single positive sample is statistically
  meaningless. The discriminator may appear to work in domains where it's really
  just getting lucky.

## Alternative Approaches

### Proven Methods for Adapter Quality/Selection

1. **LoRAuter (arxiv 2601.21795)** — Routes via task representations derived from
   small validation sets. Matches oracle performance (101.2%) when task-aligned
   adapters exist. More general than our energy gap but requires task embeddings.
   Could complement energy gap: LoRAuter for routing, energy gap for gating.

2. **MoLoRA (arxiv 2603.15965)** — Per-token adapter routing with learned gating.
   Already cited in our VISION.md. Addresses a different problem (which adapter
   per token) vs our problem (is this adapter helpful at all).

3. **L-MoE (arxiv 2510.17898)** — Lightweight gating network dynamically composes
   LoRA adapters per token. End-to-end trained, so quality gating is implicit
   in the learned gates.

4. **JS-Divergence (from arxiv 2409.11233)** — Proposed as PPL replacement for
   compression evaluation. Could be adapted for adapter quality scoring. More
   expensive (requires full distribution comparison) but captures distributional
   shifts that scalar NLL misses.

5. **KR-Test (our Finding #56)** — Knowledge retention delta rank-correlates
   perfectly with task accuracy (rho=1.000, n=4). Complementary to energy gap:
   KR-Test measures what the adapter retains, energy gap measures what it adds.

## Implications for Next Experiments

### 1. The Evolve Quality Gate is Unblocked
The energy gap at AUC=0.942 (math) is the first metric exceeding AUC>0.75 in
this project. The Evolve system can now use Delta_E as a quality gate for
adapter composition selection: reject compositions with Delta_E > 0, rank by
magnitude. This is zero-parameter, requires only 2 forward passes per
(adapter, sample) pair, and works on the existing infrastructure.

### 2. Composition Energy Gaps Are the Critical Next Test
All results are for single adapters. The original motivation was composition
quality. Key question: does the energy gap decompose linearly for multi-adapter
compositions? If Delta_E(x, a+b) ≈ Delta_E(x, a) + Delta_E(x, b), single-adapter
gaps predict composition quality. If not, composed evaluation is required.

### 3. The Relative-vs-Absolute Pattern May Generalize
If relative NLL works where absolute NLL fails, other relative metrics may also
work: relative entropy, relative KL-divergence, relative embedding shift
(normalized by base). This project should default to relative/differential
metrics going forward.

### 4. Domain Coverage Remains a Weakness
Math is the only domain with strong signal. Scaling to more domains with
balanced positive/negative adapters is needed to confirm generality. The code
domain's 0/17 positive rate suggests some domains may be fundamentally
harder for LoRA adaptation at this scale.

## Recommended Follow-Up

**Composition energy gaps** — Test Delta_E on multi-adapter compositions from
lora_scale_ablation data. Motivation: Finding #182 only validates single adapters;
the Evolve use case requires composition scoring. Literature support: MoLoRA
(arxiv 2603.15965) shows per-token composition works, but quality gating of
composed outputs is untested.

**Cross-domain energy gap** — Evaluate whether a math adapter's energy gap on
code data predicts code accuracy (and vice versa). Motivation: the same-domain
confound (PAPER.md) means in-domain signal is partially tautological. The
non-obvious test is cross-domain prediction. If cross-domain gaps work, the
energy discriminator becomes a universal quality gate rather than domain-specific.
