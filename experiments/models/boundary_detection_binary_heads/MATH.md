# Boundary Detection via Sliding-Window Domain Classification

## Type: Guided Exploration

**Proven framework:** Segment isolation eliminates cross-attention contamination by
construction (Finding #305, Theorem 1). Per-adapter PPL-based classification achieves
95.2% segment accuracy at N=5 (Finding #305).

**Unknown:** How accurately can we detect domain BOUNDARIES (not just classify domains)
by observing WHERE the argmax adapter changes in a sliding window? What window size is
optimal? What is the end-to-end PPL gap vs oracle boundaries?

## Step A: The Disease (Not Symptoms)

The segment isolation approach (Finding #305) achieves +16% PPL improvement over
per-sequence routing, but it requires **oracle boundaries** -- the exact token position
where one domain ends and another begins. In production, these boundaries are unknown.

The disease is not "how to classify domains" (already solved at 95.2% accuracy) but
"how to LOCATE where one domain ends and another begins" in a streaming token sequence.

**This is a change-point detection problem**, not a classification problem.

Prior killed findings (#190-192) showed that binary heads collapse at N=24 for domain
CLASSIFICATION. But boundary DETECTION is fundamentally simpler: we only need to detect
that the domain CHANGED, not identify which domain it changed to (that is handled
post-split by the segment classifier).

## Step B: The Right Question

**Wrong:** "How do we classify each token's domain in a mixed-domain sequence?"
(This is the classification problem that collapses at N=24.)

**Right:** "Given a sequence of window-level domain classifications, at what positions
does the classification change, and how close are these change-points to the true
domain boundaries?"

This reframes the problem as **change-point detection on a categorical time series**,
a well-studied problem in statistics (Page 1954, CUSUM; Basseville & Nikiforov 1993).

## Step C: Mathematical Foundation

### Change-Point Detection Theory

The classical change-point detection framework (Basseville & Nikiforov 1993) considers
a sequence of observations x_1, ..., x_T drawn from distribution P_theta, where theta
changes at unknown time tau: theta = theta_0 for t < tau, theta = theta_1 for t >= tau.

In our setting:
- Each "observation" is the argmax adapter classification of a sliding window
- theta_0 = domain A, theta_1 = domain B
- tau = the true boundary position
- We detect tau by finding where the classification sequence changes

### Window Classification Accuracy Model

Let p be the probability that a window centered at position t is classified correctly
(i.e., assigned to the true domain for the tokens in that window).

From Finding #305, we know p = 0.952 for pure-domain segments of length 128.

**Lemma 1 (Window Purity).** A sliding window of size w centered at position t in a
sequence with boundary at position tau contains tokens from a single domain if and
only if either t + w/2 <= tau or t - w/2 >= tau. The window is "mixed" (contains tokens
from both domains) when |t - tau| < w/2.

*Proof.* Direct from the definition. The window spans [t - w/2, t + w/2). It overlaps
the boundary when both t - w/2 < tau and t + w/2 > tau, i.e., |t - tau| < w/2. QED.

**Theorem 1 (Boundary Detection via Argmax Change).** Let c_t denote the argmax
domain classification for a window centered at position t. Suppose:
1. Pure windows (|t - tau| >= w/2) are classified correctly with probability p
2. Mixed windows (|t - tau| < w/2) are classified to either adjacent domain with
   some probability that depends on the fraction of each domain in the window

Then the detected boundary position tau_hat (the position where c_t changes) satisfies:

   |tau_hat - tau| <= w/2

with probability at least p^2 (the probability that both the last pure window before
the boundary and the first pure window after it are classified correctly).

*Proof.* Consider the sequence of classifications c_1, c_2, ..., c_T.

For positions well before the boundary (t + w/2 < tau), c_t = domain_A with probability p.
For positions well after the boundary (t - w/2 > tau), c_t = domain_B with probability p.

A change in c_t can occur:
(a) Within the "mixed zone" |t - tau| < w/2, where the window spans the boundary.
    This change point is within w/2 of the true boundary by construction.
(b) Outside the mixed zone, due to misclassification. This requires at least one pure
    window to be misclassified, which occurs with probability (1-p).

The detected boundary tau_hat is the first position where c_t changes from domain_A to
domain_B (or vice versa). Case (a) places |tau_hat - tau| <= w/2 by Lemma 1. Case (b)
creates a false positive with probability <= 1 - p per window.

For the true change to be detected within w/2 tokens: the last pure-A window and the
first pure-B window must both be correct. This occurs with probability at least p^2.

Therefore P(|tau_hat - tau| <= w/2) >= p^2. QED.

**Corollary 1 (F1 Prediction).** If we define a boundary detection as "correct" when
|tau_hat - tau| <= delta for some tolerance delta >= w/2, then:

- Precision >= p^2 / (p^2 + (1-p) * (T/w - 1))
  [true positives / (true positives + false positives from misclassification)]
- Recall >= p^2
  [probability the true boundary is detected within tolerance]

For p = 0.952, w = 32, T = 256, delta = w/2 = 16:
- p^2 = 0.906
- False positives: expected number = (1-p) * (T/w - 1) = 0.048 * 7 = 0.336
  (But false positives require TWO adjacent misclassifications that happen to differ,
   which is (1-p)^2 * (1 - 1/N) per window pair = 0.0023 * 0.8 = 0.0018)
- Refined: Expected false positives = 0.0018 * 7 = 0.013
- Precision >= 0.906 / (0.906 + 0.013) = 0.986
- Recall >= 0.906
- F1 >= 2 * 0.986 * 0.906 / (0.986 + 0.906) = 0.944

**Predicted F1 >= 94% for w=32, p=0.952, sharp boundaries.**

### Window Size Trade-off

**Theorem 2 (Window Size Optimality).** The boundary localization error |tau_hat - tau|
is bounded by w/2, but classification accuracy p(w) depends on w:
- Larger w: more tokens per window, higher p(w), but coarser boundary resolution
- Smaller w: finer boundary resolution, but lower p(w) due to fewer tokens for PPL estimation

The optimal window size w* minimizes the expected boundary error:

   E[|tau_hat - tau|] = w/2 * p(w)^2 + T/2 * (1 - p(w)^2)

The first term is the error when detection succeeds (within w/2); the second is the
error when detection fails entirely (random guess over T).

For this experiment, we explore w in {16, 32, 64, 128} to empirically determine p(w)
and thus w*. This is the Type 2 (guided exploration) component.

### PPL Gap from Boundary Error

**Theorem 3 (PPL Degradation from Boundary Error).** Let PPL_oracle be the PPL under
perfect segmentation, and PPL_detected be the PPL under detected boundaries with
localization error epsilon = |tau_hat - tau|.

The PPL ratio is:

   PPL_detected / PPL_oracle = exp((epsilon / T) * (NLL_wrong - NLL_right))

where NLL_wrong is the per-token NLL under the wrong adapter and NLL_right under the
correct adapter.

*Proof.* With boundary error epsilon, approximately epsilon tokens are assigned to the
wrong adapter. The total NLL changes by epsilon * (nll_wrong - nll_right).

   log(PPL_detected) = log(PPL_oracle) + (epsilon / T) * (NLL_wrong - NLL_right)
   PPL_detected / PPL_oracle = exp((epsilon / T) * (NLL_wrong - NLL_right))

QED.

From Finding #305 data: per-sequence PPL = 4.815, segment-oracle PPL = 4.042.
The per-token NLL difference between wrong and right adapters is approximately:
   NLL_wrong - NLL_right ~ ln(4.815) - ln(4.042) = 1.572 - 1.397 = 0.175 per token

For epsilon = w/2 = 16 tokens, T = 256:
   PPL_detected / PPL_oracle = exp((16/256) * 0.175) = exp(0.011) = 1.011

**Predicted: detected-boundary PPL within 1.1% of oracle.** This is well within the
5% K776 threshold.

## Step D: Quantitative Predictions

| Prediction | Source | Value |
|-----------|--------|-------|
| Boundary F1 (sharp, w=32) | Theorem 1 + Corollary 1, p=0.952 | >= 94% |
| Boundary localization error | Theorem 1, w=32 | <= 16 tokens |
| PPL gap vs oracle | Theorem 3, epsilon=16, T=256 | <= 1.1% |
| False positive rate (per sequence) | Corollary 1 | <= 0.013 |
| Window classification accuracy at w=32 | Extrapolation from p(128)=0.952 | >= 85% (conservative) |

**Kill criteria derived from predictions:**
- K775 (F1 >= 70%): Theorem 1 predicts F1 >= 94%. FAIL if p < 0.84 (i.e., window
  classification accuracy drops below 84% at all tested window sizes)
- K776 (PPL within 5%): Theorem 3 predicts 1.1%. FAIL if NLL_wrong - NLL_right is
  much larger than estimated (would need 7.8x our estimate to hit 5%)
- K777 (latency < 5ms): Computational: N windows * N adapters * forward_pass_time.
  At w=32, stride=16: 16 windows. With N=5 adapters: 80 forward passes of 32 tokens.
  Each ~0.05ms on M5 Pro. Total ~4ms. Tight but should pass.

## Step E: Assumptions and Breaking Conditions

1. **Sharp boundaries**: Sequences have a single clean domain transition. If domains
   blend gradually over many tokens, the "mixed zone" grows and Theorem 1's w/2 bound
   is optimistic. Breaking: F1 drops, but PPL impact should be small (Theorem 3).

2. **p is stable for short windows**: We assume per-window classification accuracy
   remains high even for w=32. If PPL estimates on 32 tokens are too noisy for reliable
   classification, p drops and F1 drops. Breaking: need larger w, coarser boundaries.

3. **Single boundary per sequence**: With multiple boundaries, each must be detected
   independently. False positive rate accumulates. Breaking: for K boundaries, expected
   false positives multiply by K, degrading precision.

4. **N=5 domains (well-separated)**: Prior results at N=5 showed 95.2% classification.
   At N=24, classification dropped to 40%. Boundary detection may be more robust (only
   needs to detect CHANGE, not identify domain), but this is unproven at N>5.

## Step F: Worked Example (w=32, T=256, N=5)

**Setup:** Sequence of 256 tokens. Tokens 0-127 from domain "python", tokens 128-255
from domain "legal". True boundary at position 128. Window size w=32, stride s=16.

**Window positions:** Centers at t = 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240.
(15 windows total)

**Classification per window** (given p=0.952):
- Windows 1-7 (t=16..112): fully within python. Each classified as "python" with prob 0.952.
- Window 8 (t=128): spans positions 112-143. Mixed: 16 python + 16 legal tokens.
  Classification uncertain (could go either way based on which domain dominates PPL).
- Windows 9-15 (t=144..240): fully within legal. Each classified as "legal" with prob 0.952.

**Detected boundary:** The classification changes between window 7 (python) and window 9
(legal). Window 8 could go either way. If window 8 is classified as "python", detected
boundary is between windows 8 and 9: position ~136. Error = |136 - 128| = 8 tokens.
If window 8 is classified as "legal", detected boundary is between windows 7 and 8:
position ~120. Error = |120 - 128| = 8 tokens.

**Either way: |tau_hat - tau| <= w/2 = 16 tokens.** Confirms Theorem 1.

**PPL impact:** 8 misgrouped tokens out of 256.
PPL_detected / PPL_oracle = exp((8/256) * 0.175) = exp(0.0055) = 1.0055 (0.55% gap).

## Step G: Complexity and Architecture Connection

**Boundary detection cost:**
- Method 1 (exhaustive PPL): N_windows * N_adapters * cost_per_forward
  - For w=32, stride=16, T=256: 15 windows * 5 adapters = 75 forward passes of 32 tokens
  - Much cheaper than full-sequence forward passes (75 * 32 = 2400 tokens vs 5 * 256 = 1280)
  - But each is independent (no KV cache reuse)

- Method 2 (softmax router): 1 forward pass to get hidden states + N_windows cheap
  classifications. O(T * d_hidden + N_windows * d_hidden * N_adapters). Much faster.

- Method 3 (differential PPL): Only compute PPL for ONE adapter on all windows, detect
  where it CHANGES sharply. Then classify only the boundary windows with all N adapters.
  This reduces to ~15 + 5 = 20 forward passes.

**Memory:** No additional model weights (uses existing adapter infrastructure). Only
storage for N_windows * N_adapters PPL values (~75 floats).

**Integration with Finding #305 pipeline:**
1. [NEW] Run boundary detection on incoming sequence
2. [Existing] Split at detected boundaries
3. [Existing] Classify each segment via PPL-based routing (Finding #305)
4. [Existing] Evaluate each segment independently with best adapter

## Self-Test

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   The sliding window guarantees that the detected change-point is within w/2 tokens
   of the true boundary, because pure-domain windows on either side of the boundary
   are classified correctly with probability p^2 > 0.9.

2. **Which existing theorem(s) does the proof build on?**
   Change-point detection theory (Basseville & Nikiforov 1993, "Detection of Abrupt
   Changes in Signals and Dynamical Systems"). Window-purity is a direct geometric
   argument. PPL decomposition follows from NLL additivity.

3. **What specific numbers does the proof predict?**
   F1 >= 94% (Corollary 1), localization error <= 16 tokens (Theorem 1),
   PPL gap <= 1.1% (Theorem 3), false positive rate <= 0.013 per sequence.

4. **What would FALSIFY the proof (not just the experiment)?**
   The proof is wrong if: (a) p(w=32) is much lower than p(w=128) (i.e., short-window
   PPL estimation is too noisy for domain classification), or (b) the NLL difference
   between adapters is not approximately additive across misassigned tokens.

5. **How many hyperparameters does this approach add?**
   Count: 2 (window size w and stride s). Window size is bounded by Theorem 1 (must be
   large enough for reliable classification, small enough for localization). Stride is
   constrained by s <= w (coverage) and s >= 1 (efficiency). The optimal w is the
   Type 2 unknown this experiment discovers.

6. **Hack check:** No. This is a single mechanism (sliding window change-point detection)
   applied to the proven segment isolation framework (Finding #305).

## Post-Experiment Analysis: Where the Theory Failed

### Theorem 1 (Localization): CONFIRMED
Mean localization error 19.6 tokens at w=64 (predicted <= 32). The geometric bound holds.

### Corollary 1 (F1): PARTIALLY CONFIRMED
Predicted F1 >= 94% at w=32, measured 61.2%. At w=64, measured 88.2% (closer to prediction).
The gap comes from Corollary 1 underestimating false positives. The model assumed false
positives require two ADJACENT misclassifications that differ, giving P(FP) = (1-p)^2 * (1-1/N).
In reality, per-window PPL has correlated noise: if one window is noisy, adjacent windows
(overlapping by 50%) are also noisy, creating BURSTS of argmax flickering. The independence
assumption in Corollary 1 is violated.

### Theorem 3 (PPL Gap): FALSIFIED
Predicted 1.1% gap, measured 32.9%. The theorem modeled only epsilon misassigned tokens at
the boundary. The actual failure mode is different: false positive boundaries create ENTIRE
MISROUTED SEGMENTS, not just misassigned tokens. A sequence split into 3 segments at wrong
positions can have one entire segment routed to a domain that is wrong for all its tokens --
this is O(T/3) misassignment, not O(epsilon).

**Corrected model:** Let k be the expected number of detected boundaries (true + false).
With 1 true boundary and 0.26 false positive boundaries per sequence:
- Expected segments: k+1 = 2.26
- Each false-positive segment has probability (1 - 1/N) of being misrouted = 0.8
- Misrouted segment has average length T/(k+1) ~ 113 tokens
- Expected misassigned tokens: 0.26 * 0.8 * 113 = 23.5 tokens
- PPL gap: exp((23.5/256) * 0.175) = exp(0.016) = 1.6% -- still less than measured.

The remaining discrepancy (1.6% predicted vs 32.9% measured) suggests additional effects:
short segments have less context for accurate PPL estimation, and the Phase 2 implementation
only used the FIRST detected boundary (ignoring subsequent ones), which may have been a
false positive rather than the true boundary.

### Latency Prediction: GROSSLY WRONG
Predicted ~4ms, measured 3017ms. The error was assuming 0.05ms per forward pass of 32 tokens.
Actual cost: ~40ms per forward pass (800x). This is because even a 32-token sequence requires
a full model forward pass through all 26 layers of BitNet-2B-4T, each involving large matrix
multiplications. The cost is dominated by weight loading (memory bandwidth), not by the number
of input tokens.

### Key Takeaway
The mathematical framework (change-point detection via sliding-window classification) is sound.
Localization bounds hold. But the IMPLEMENTATION via per-adapter PPL is doubly impractical:
(a) too many forward passes for latency budget, and (b) PPL noise at short window sizes
creates correlated false positives that cascade into misrouted segments. A viable boundary
detector needs a lightweight, smooth signal (hidden states, entropy) rather than expensive,
noisy PPL computation.
