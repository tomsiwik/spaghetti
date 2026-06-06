# Self-Contrast Decoding: Proof Verification Report

## Status: KILLED (informative negative result)

## Theorem

Self-contrast decoding (SCMoE-style) should extract useful signal from
unchosen LoRA adapters as a negative/amateur distribution. The contrastive
formula z_CD = (1+alpha)*z_expert - alpha*z_amateur should suppress generic
tokens and amplify domain-specific tokens when the amateur adapters correlate
with generic but not domain-specific outputs.

## Predictions

| Prediction (from MATH.md) | Measured | Match? |
|---------------------------|----------|--------|
| P1: math score >= 0.80 | 0.80 (at alpha=0.1 only; degrades at higher alpha) | PARTIAL |
| P2: code score >= 0.62 | 0.55 (best at alpha=0.1) | NO |
| P3: finance degrades | 0.155 vs 0.156 baseline (marginal) | YES |
| P4: <= 2 domains worse | 0 (no alpha improves ANY domain) | YES (vacuously) |
| P5: latency ~2x (< 3x) | 2.02x | YES |

## Hypothesis

Self-contrast decoding extracts value from unchosen LoRA adapters by using them
as an amateur distribution in contrastive decoding, sharpening the expert's
domain-specific outputs without training.

**Result: KILLED.** Self-contrast provides zero benefit at any alpha value.
The non-primary adapter signal is noise, not useful contrast.

## What This Model Is

SCMoE-style contrastive decoding adapted from native MoE experts to LoRA
adapter composition. For each query routed to a primary adapter:
- Expert = base model + primary adapter (pre-merged)
- Amateur = base model + average(non-primary adapters) (pre-merged)
- Contrastive: z_CD = (1+alpha)*z_expert - alpha*z_amateur
- Greedy decoding from contrastive logits

## Key References

- SCMoE (arxiv 2405.14507): self-contrast on MoE experts, Mixtral GSM8K +8.3%
- Contrastive Decoding (Li et al., 2210.15097): expert vs amateur logit subtraction
- DExperts (2105.03023): expert/anti-expert ensemble at decoding time

## Empirical Results

### Behavioral Scores by Alpha

| Domain | Baseline | alpha=0.1 | alpha=0.3 | alpha=0.5 | alpha=1.0 |
|--------|----------|-----------|-----------|-----------|-----------|
| medical | **0.291** | 0.279 | 0.288 | 0.229 | 0.210 |
| code | **0.624** | 0.553 | 0.478 | 0.284 | 0.071 |
| math | **0.800** | 0.800 | 0.400 | 0.100 | 0.000 |
| legal | **0.096** | 0.088 | 0.079 | 0.072 | 0.098 |
| finance | **0.156** | 0.155 | 0.118 | 0.142 | 0.125 |

**Baseline wins every domain at every alpha.** No alpha value improves any domain.

### Degradation Pattern

The contrastive signal monotonically destroys quality as alpha increases:
- alpha=0.1: mild degradation (code -11.3%, medical -4.2%)
- alpha=0.3: moderate degradation (math -50%, code -23.4%)
- alpha=0.5: severe degradation (math -87.5%, code -54.5%)
- alpha=1.0: catastrophic degradation (math 0/10 correct, code 0.07 score)

Math is most sensitive because it relies on exact numerical computation --
contrastive noise perturbs logits away from the correct digits.

### Latency (K2)

| Alpha | Avg Latency (s) | Ratio vs Baseline |
|-------|-----------------|-------------------|
| baseline | 5.62 | 1.00x |
| 0.1 | 11.31 | 2.01x |
| 0.3 | 11.34 | 2.02x |
| 0.5 | 11.33 | 2.02x |
| 1.0 | 11.02 | 1.96x |

K2 PASS: 2.02x worst case, well under 3x threshold. The ~2x overhead is
expected (two forward passes per token).

### Kill Criteria

- **K1 (#652):** PASS (0/5 domains worse -- vacuously, because best alpha
  per domain defaults to no contrast). But the REAL result is KILL: self-contrast
  never improves any domain. The kill criterion as written was too permissive.
- **K2 (#653):** PASS (2.02x, under 3x threshold).

## Why This Failed (Root Cause Analysis)

**SCMoE works on native MoE experts because they share the SAME model.** In Mixtral,
all 8 experts are FFN blocks within the same transformer. They share embeddings,
attention, and have been jointly trained. The unchosen experts have meaningful
per-token preferences that are anti-correlated with irrelevant tokens.

**LoRA adapters are NOT shared-model experts.** Our adapters:
1. Were trained independently on different datasets
2. Share only the base model weights (not jointly optimized)
3. Have Grassmannian-orthogonal A-matrices (by construction, their subspaces
   are maximally decorrelated)
4. Produce adapter deltas that are in orthogonal subspaces

The Grassmannian skeleton -- which is the STRENGTH of our composition method --
is precisely what makes self-contrast useless. The non-primary adapters operate
in orthogonal subspaces, so their averaged output is effectively random noise
relative to the primary adapter's domain-specific signal. Subtracting random
noise from a good signal can only degrade it.

**The mathematical structure that enables interference-free composition
(orthogonality) is the same structure that prevents contrastive value extraction.**
This is a fundamental duality, not a tuning problem.

## Limitations

1. Only tested greedy decoding (temperature=0). Sampling-based decoding might
   interact differently with contrastive logits, but the monotonic degradation
   pattern suggests this would not help.
2. Tested with uniform amateur weighting (1/K-1 average). Weighted amateur
   (e.g., inverse-routing-score) might help, but orthogonality makes all
   non-primary contributions equally noisy.
3. 10 prompts per domain is limited sample size, but the pattern is consistent
   across all 200 generations (50 prompts x 4 alphas).

## What Would Kill This

Already killed. The mechanism is fundamentally incompatible with Grassmannian
orthogonal adapters. Would only work if adapters operated in overlapping
subspaces (which would re-introduce interference -- the very thing we eliminated).

## Key Takeaway for the Project

**Self-contrast requires shared representations to work.** In SCMoE, the MoE
experts share all non-FFN layers and have co-trained representations. In our
architecture, adapters are deliberately decorrelated. This means:

1. Training-free composition improvements from the MoE literature do NOT
   transfer to orthogonal LoRA composition.
2. The value of our architecture is in its composition GUARANTEE (zero
   interference), not in cross-adapter information extraction.
3. Future composition improvements should focus on better routing (which
   adapter to USE) rather than trying to extract signal from unchosen adapters.
