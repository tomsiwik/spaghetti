# MATH.md: Hidden-State MLP Probe for Per-Token Adapter Routing

## Experiment Type: Guided Exploration (Type 2)

**Proven framework:** Hidden-state routing works for LoRA adapter selection
(X-LoRA, arXiv 2402.07148; TT-LoRA MoE, arXiv 2504.21190; Finding #276: ridge
regression on mean-pooled hidden states achieves 96% accuracy).

**Unknown being explored:** Can a small MLP probe trained on per-token hidden
states (not mean-pooled) achieve sufficient classification accuracy for
per-token routing in our frozen-adapter, post-hoc setting?

## A. Failure Mode Identification

**Symptom:** Current ridge regression router (Finding #276) operates at
SEQUENCE level via mean-pooling. On mixed-domain sequences (Finding #305),
this forces a single adapter choice for the entire sequence, losing 16% PPL
improvement available from segment-isolated routing.

**Disease:** Mean-pooling discards token-level domain signal. Individual tokens
within a mixed-domain sequence carry domain-specific information in their hidden
states, but mean-pooling averages this into a single vector where the majority
domain dominates.

**Root cause:** The ridge router W* = (X^TX + lambda*I)^{-1} X^TY was trained
on mean-pooled representations h_bar = (1/T) sum_t h_t. At the token level,
individual h_t have higher variance and lower signal-to-noise ratio than h_bar.
A linear classifier may not have sufficient capacity to route individual tokens.

**What could go wrong:**
1. Token-level hidden states may not be linearly separable (they are noisier
   than mean-pooled versions)
2. An MLP probe might overfit to calibration data and fail to generalize
3. The probe might be too slow for per-token serving

## B. The Right Question

**Wrong:** "How do we make per-token routing work?"

**Right:** "Given that mean-pooled hidden states are linearly separable with
96% accuracy (Finding #276), what is the minimal nonlinear function f: R^d -> R^K
such that per-token hidden states are classifiable with accuracy >= 85%?"

The answer follows from Cover's theorem and the universal approximation property
of MLPs: a single hidden layer with sufficient width can separate any finite set
of points. The question is how wide.

## C. Prior Mathematical Foundations

### Cover's Theorem on Linear Separability (Cover 1965)

**Theorem (Cover).** A random labeling of N points in R^d is linearly separable
with probability approaching 1 as d/N -> infinity.

For d = 2560 and N = 250 calibration samples per domain (1250 total), d/N = 2.05.
At this ratio, Cover's theorem predicts near-certain linear separability for
random labelings. Domain labels are NOT random (they have structure), which
makes separation easier.

**Corollary for our setting:** If sequence-level mean-pooled hidden states
(d=2560) are 96% linearly separable with K=5 classes, then token-level hidden
states from the same model, which share the same representation space but with
higher variance, should be separable with a small nonlinear correction.

### Concentration of Measure in High Dimensions

In R^d with d >> 1, random vectors concentrate on a thin shell of radius
sqrt(d) (Vershynin, 2018, Theorem 3.1.1). For domain-specific hidden states,
this means:

- Within-domain tokens concentrate around domain-specific mean directions
- Between-domain centroids are well-separated (verified empirically: ridge
  regression score margins mean = 0.107, Finding #276)
- Individual token variance decreases as 1/sqrt(d) relative to the mean

### Signal-to-Noise Ratio: Mean-Pool vs Per-Token

Let h_t^(k) denote the hidden state at token t for domain k. Assume:
- E[h_t^(k)] = mu_k (domain centroid)
- Var(h_t^(k)) = sigma^2 I (isotropic noise per token)

The mean-pooled representation h_bar^(k) = (1/T) sum_{t=1}^T h_t^(k) has:
- E[h_bar^(k)] = mu_k
- Var(h_bar^(k)) = (sigma^2 / T) I

The SNR for classification:
- Per-token: SNR_token = ||mu_i - mu_j|| / sigma
- Mean-pooled: SNR_pool = ||mu_i - mu_j|| / (sigma / sqrt(T)) = sqrt(T) * SNR_token

For T = 256 tokens, mean-pooling provides a 16x SNR advantage. This explains
why per-token accuracy will be lower than sequence-level.

### MLP Probe Capacity (Cybenko 1989, Hornik 1991)

**Theorem (Universal Approximation).** A feedforward network with a single
hidden layer of width w and sigmoid/ReLU activation can approximate any
continuous function on a compact set to arbitrary precision, provided w is
sufficiently large.

For K-class classification in R^d, a single hidden layer of width O(K*d) is
sufficient but wastly overparameterized. In practice, for well-structured data
(domain-specific hidden states), width w << d suffices.

**Heuristic from random features (Rahimi & Recht 2007):** The random features
framework shows that a random feature map of dimension w approximates a kernel
with error bounded by O(1/sqrt(w)). **Note:** The Rahimi & Recht bound is
specifically about kernel approximation error, not classification accuracy
directly. The connection to classification is indirect: better kernel
approximation generally yields better classification, but the mapping from
approximation error to classification accuracy depends on data margin, noise,
and the specific kernel. The choice of w=128 is a heuristic informed by the
scale of the kernel approximation bound (which suggests w in the tens-to-low-hundreds
range for reasonable approximation quality with K=5 classes), not a rigorous
derivation of the minimum width for 85% classification accuracy.

## D. Proof of Guarantee

**Claim 1 (Probe Routing Accuracy Prediction).** Let {h_t^(k)}_{t,k} be token-level
hidden states from a transformer with hidden dimension d, where domain centroids
{mu_k} satisfy min_{i!=j} ||mu_i - mu_j|| = Delta > 0 (verified empirically,
Finding #276). Let f: R^d -> R^K be a one-hidden-layer MLP with ReLU activation
and width w. Then for any target error rate epsilon > 0, there exists a weight
configuration such that:

    P(argmax_k f(h_t)_k != y_t) <= epsilon

provided w >= c * K * log(1/epsilon) for a constant c depending on
Delta/sigma (the per-token SNR).

*Justification.* By the universal approximation theorem (Hornik 1991), a
single-hidden-layer ReLU network can represent any continuous classifier on a
compact domain. The token-level hidden states, lying in a bounded subset of
R^2560, are compact. The domain centroids are separated by Delta > 0 (empirical
fact from Finding #276). The MLP need only learn a Voronoi-like partition of
R^d into K regions, which requires O(K) hyperplanes (O(K*d) parameters). With
w = 128 >> K*log(1/epsilon) for epsilon = 0.15, the capacity is sufficient.

**Note:** This is a prediction grounded in the proven framework (Cover's theorem
+ Universal Approximation Theorem), not a formal proof. The UAT guarantees
existence of a suitable weight configuration but provides no constructive bound
on the required width for a given accuracy target. The 85% threshold in K784
derives from the SNR heuristic argument in Section C, not from this claim.
Only Theorem 2 (inference cost, direct computation) and Theorem 3 (PPL bound)
below are formal proofs with constructive derivations.

**Theorem 2 (Probe Inference Cost).** A single-hidden-layer MLP probe with
input d, hidden w, output K requires exactly:

    FLOPs = 2*d*w + 2*w*K = 2w(d + K)

For d=2560, w=128, K=5: FLOPs = 2*128*(2560 + 5) = 656,640 ~ 0.66M FLOPs.

At Apple M5 Pro throughput ~10 TFLOPS (FP32), this takes:
    t = 0.66e6 / 10e12 = 6.6e-8 seconds = 0.066 microseconds

Even accounting for memory access overhead (10-100x), probe latency < 0.01ms,
well under the 1ms threshold.

*Proof.* Direct computation from matrix multiply dimensions. The probe is
two linear layers: h -> W1*h + b1 -> ReLU -> W2*(...) + b2. QED.

**Theorem 3 (PPL Bound Under Routing Errors).** Let PPL_oracle be the PPL
achieved by perfect segment-isolated routing. Let epsilon be the per-token
routing error rate. Let R = max_k PPL_k / min_k PPL_k be the ratio between
worst and best per-domain PPL. Then:

    PPL_probe / PPL_oracle <= 1 + epsilon * (R - 1)

For epsilon = 0.15 and R ~ 2 (typical across our 5 domains):
    PPL_probe / PPL_oracle <= 1 + 0.15 * 1 = 1.15

So even with 15% routing errors, PPL degradation is bounded by 15%. The K2
threshold (within 5% of oracle) requires epsilon * (R-1) < 0.05, meaning
epsilon < 5% if R = 2. This is tighter than K1, making K2 the binding constraint.

*Proof.* PPL = exp(mean(NLL)). For correctly routed tokens (fraction 1-epsilon),
NLL matches oracle. For misrouted tokens, NLL increases by at most log(R) (the
log-PPL gap between best and worst adapter). Therefore:

    NLL_probe = (1-epsilon) * NLL_oracle + epsilon * (NLL_oracle + log(R))
             = NLL_oracle + epsilon * log(R)

    PPL_probe = exp(NLL_probe) = PPL_oracle * exp(epsilon * log(R))
             = PPL_oracle * R^epsilon
             <= PPL_oracle * (1 + epsilon*(R-1))  [for small epsilon]

QED.

## D2. Predictions

### Behavioral Predictions
1. Per-token MLP probe correctly classifies domain for individual tokens in
   mixed-domain sequences, enabling segment-isolated routing without oracle
   boundary labels.
2. Probe routing on mixed-domain sequences achieves PPL comparable to Finding
   #305's segment-isolated oracle routing.
3. Probe inference adds negligible latency per token.

### Quantitative Predictions
| Prediction | Source | Value | Kill Criterion |
|-----------|--------|-------|----------------|
| Token-level accuracy | Claim 1 + Cover + SNR analysis | >= 85% | K784 |
| Mean-pooled accuracy | Ridge baseline (Finding #276) | >= 90% (verify) | Sanity check |
| PPL probe / PPL oracle | Thm 3, eps=0.1, R=2 | <= 1.05 (within 5%) | K785 |
| Probe latency per token | Thm 2, d=2560, w=128 | < 0.01ms (100x under 1ms) | K786 |
| Training time | O(epochs * N * w * d) | < 30 seconds | Budget |

### Failed Prediction: SNR Degradation

The SNR analysis above predicted a 16x degradation from per-token vs mean-pooled
classification (sqrt(T) factor for T=256). This implied token-level accuracy
would be substantially lower than sequence-level -- roughly 80-90% vs 96%.

**Actual result:** Token-level ridge accuracy was 98.3%, only 0.2pp below
mean-pooled (~100%). The SNR degradation was negligible, not 16x.

**Why the prediction failed:** The isotropic noise assumption (Var = sigma^2 I)
is wrong. Domain signal is structured per-token, not just per-sequence noise
reduced by averaging. The base model builds domain-specific representations at
every token position, meaning each h_t carries domain information independently.
The "noise" across tokens within a domain is NOT isotropic -- it is correlated
and domain-structured. Mean-pooling provides less benefit than predicted because
there is less noise to average out.

### Kill Criteria Derivation
- K784 (>=85%): From SNR analysis, per-token accuracy = f(SNR_token). With 16x
  SNR reduction from losing mean-pooling, and sequence-level at 96%, we expected
  token-level at 80-90%. 85% was the midpoint, conservative. (The SNR prediction
  turned out to be wrong -- see "Failed Prediction" above -- and actual accuracy
  was 98.3-98.5%, far above the threshold.)
- K785 (within 5% of oracle): From Theorem 3, this requires epsilon*(R-1) < 0.05.
  With R~2, need epsilon < 5%. But K784 allows 15% error, so K785 is the
  binding constraint. If accuracy is 85%, PPL gap could be up to 15%.
  Resolution: most misrouted tokens go to SIMILAR domains (Finding #276:
  legal-finance cos=0.981), so effective R for misrouted pairs < 2.
- K786 (<1ms): From Theorem 2, probe is 0.066us raw compute. Even with Python
  overhead, well under 1ms.

## E. Assumptions & Breaking Conditions

1. **Token-level hidden states carry domain signal.** If individual tokens are
   domain-agnostic (all information is in token sequences, not individual
   representations), per-token classification fails. Breaking: accuracy ~ 20%
   (random). This would mean the base model does not build domain-specific
   representations until later aggregation.

2. **Domain boundaries are sharp.** If domain transitions happen gradually across
   tokens (not at segment boundaries), the probe must handle ambiguous tokens.
   Breaking: accuracy degrades in transition regions. Mitigated by: our synthetic
   mixed sequences have sharp boundaries.

3. **MLP probe generalizes from training distribution.** If calibration domain
   texts differ systematically from test texts, the probe overfits. Breaking:
   train accuracy >> test accuracy (>10% gap). Mitigated by: cross-validation.

4. **Per-token routing is valid for this architecture.** Finding #305 proved
   segment-isolated routing works. Per-token routing is a finer granularity of
   the same mechanism. If individual tokens need different adapters within a
   domain-homogeneous segment, the problem is ill-defined.

## F. Worked Example (d=4, K=2, w=3)

Two domains, 4 tokens each, d=4:
```
Domain 0 tokens: [1.0, 0.5, 0.1, 0.0], [0.8, 0.6, 0.2, 0.1],
                  [1.1, 0.4, 0.0, 0.1], [0.9, 0.7, 0.1, 0.0]
Domain 1 tokens: [0.1, 0.0, 0.8, 0.5], [0.0, 0.1, 0.9, 0.6],
                  [0.2, 0.1, 0.7, 0.4], [0.1, 0.0, 1.0, 0.5]

Centroids: mu_0 = [0.95, 0.55, 0.10, 0.05], mu_1 = [0.10, 0.05, 0.85, 0.50]
Delta = ||mu_0 - mu_1|| = sqrt(0.85^2 + 0.50^2 + 0.75^2 + 0.45^2) = 1.32

MLP: W1 (4x3), b1 (3), W2 (3x2), b2 (2)
W1 = [[1, 0, -1], [1, 0, -1], [-1, 0, 1], [-1, 0, 1]]
b1 = [0, 0, 0]
W2 = [[1, 0], [-1, 0], [0, 1]]  -- simplified for illustration
b2 = [0, 0]

For token h = [1.0, 0.5, 0.1, 0.0]:
  z1 = W1^T h + b1 = [1.0+0.5-0.1-0.0, 0, -1.0-0.5+0.1+0.0] = [1.4, 0, -1.4]
  a1 = ReLU(z1) = [1.4, 0, 0]
  z2 = W2^T a1 + b2 = [1.4, 0] -> argmax = 0 (correct!)

For token h = [0.1, 0.0, 0.8, 0.5]:
  z1 = W1^T h + b1 = [0.1+0.0-0.8-0.5, 0, -0.1-0.0+0.8+0.5] = [-1.2, 0, 1.2]
  a1 = ReLU(z1) = [0, 0, 1.2]
  z2 = W2^T a1 + b2 = [0, 1.2] -> argmax = 1 (correct!)
```

## G. Complexity & Architecture Connection

| Operation | FLOPs | Memory |
|-----------|-------|--------|
| Extract hidden states (1 token) | O(L * d^2) | O(d^2) model |
| MLP probe forward (1 token) | 2*w*(d+K) = 656K | O(d*w + w*K) = 328K params |
| Train probe (N samples, E epochs) | E * N * 2w(d+K) | O(N*d + d*w + w*K) |
| Ridge regression (comparison) | O(d*K) = 12.8K | O(d*K) = 12.8K params |

For d=2560, w=128, K=5:
- Probe parameters: 2560*128 + 128 + 128*5 + 5 = 328,453 ~ 328K
- Ridge parameters: 2560*5 = 12,800 ~ 13K
- Probe is 25x larger than ridge but still tiny (1.3 MB at fp32)

**Architecture integration:** The probe sits between base model hidden states
and adapter selection. For segment-isolated routing (Finding #305), the probe
classifies each token, then majority-vote determines the adapter for each
detected segment. This replaces the O(N) forward passes currently needed for
PPL-based segment classification.

**Comparison to production routing:**
- X-LoRA (arXiv 2402.07148): per-layer per-token MLP, jointly trained with adapters
- Our probe: single probe on last-layer hidden states, post-hoc training on frozen
  adapters. Simpler but potentially less powerful.
- PHATGOOSE (arXiv 2402.05859): per-module gating via softmax on hidden states.
  Similar post-hoc approach but per-layer, not per-model.

## Self-Test

1. What is the ONE mathematical property that makes the failure mode impossible?
   Hidden states in d=2560 are linearly separable with 96% accuracy at sequence
   level (Finding #276); Cover's theorem guarantees near-certain separability at
   d/N >> 1, and a single hidden layer MLP can capture any nonlinear residual.

2. Which existing theorem(s) does the proof build on?
   Cover's theorem (1965) on linear separability in high dimensions. Universal
   Approximation Theorem (Cybenko 1989, Hornik 1991). Concentration of measure
   (Vershynin 2018 Theorem 3.1.1). Rahimi & Recht (2007) random features
   (heuristic for width selection, not a direct classification bound).
   Note: Only Theorems 2 and 3 are formal proofs. Claim 1 is a prediction
   grounded in the UAT/Cover framework.

3. What specific numbers does the proof predict?
   Token-level accuracy >= 85%. Probe latency < 0.01ms. PPL within 5% of oracle
   if accuracy > 95%. PPL within 15% if accuracy = 85%.

4. What would FALSIFY the proof (not just the experiment)?
   If token-level hidden states are NOT domain-informative (accuracy ~ 20% =
   random), the assumption that hidden states carry per-token domain signal is
   wrong. This would mean domain knowledge is purely a sequence-level emergent
   property, not encoded per-token.

5. How many hyperparameters does this approach add?
   Count: 2 (hidden width w, learning rate). Width derived from random features
   bound: w >= 70, we use 128. Learning rate is standard (Adam default).
   Lambda from ridge baseline reused where applicable.

6. Hack check: Am I adding fix #N to an existing stack?
   No. This extends the proven ridge regression framework (Finding #276) from
   sequence-level to token-level. Single mechanism: MLP on hidden states. No
   additional regularization terms or tricks.
