# Topology Stress Test: Scaling N in Adapter Composition

## Experiment Type: Guided Exploration (Type 2)

The stability theorem framework is proven (Cohen-Steiner et al., 2007). The unknown
is empirical: at what adapter count N does the perturbation norm enter the vulnerability
window and destroy high-persistence topological features?

## Step A: Diagnose the Disease

**Problem:** At N=5 with rank-16 adapters, composition is fully topologically lossless
(Finding #225: 0/17,223 H0 features lost; Finding #230: 0/11,962 H1 features lost).
The stability bound is 10-100x loose. But this is only ONE operating point. As N grows,
the composed perturbation Delta = (scale/N) * sum_i(A_i @ B_i) changes in norm, and the
topological safety margin shrinks or grows depending on adapter coherence.

**Root cause question:** Is the topological losslessness at N=5 a robust property that
persists to N=50, or does it break at some critical N? If it never breaks at any
practical N, the entire pathway preservation research track solves a non-problem.

## Step B: The Right Question

NOT: "How do we protect topology at high N?"
RIGHT: "At what N does max_i ||Delta_i|| cross the vulnerability threshold
2 * median_persistence, making feature loss geometrically possible?"

## Step C: Prior Mathematical Foundations

### Stability Theorem (Cohen-Steiner, Edelsbrunner, Harer, 2007, Theorem 5.2)

d_B(Dgm(P), Dgm(P')) <= max_i ||delta_i||_2

where delta_i = i-th row of the composed perturbation Delta.

### Corollary: Vulnerability Threshold

Features with persistence p survive iff p > 2 * max_i ||delta_i||_2. The critical
N is the value where:

  max_i ||Delta_i(N)||_2 >= median_persistence / 2

### Perturbation Norm Scaling with N

For N adapters composed as Delta(N) = (scale/N) * sum_{i=1}^N (A_i @ B_i):

**Case 1: Incoherent adapters.** If the adapter contributions A_i @ B_i have
uncorrelated row vectors, by concentration:

  ||Delta_row(N)|| ~ (scale/N) * sqrt(N) * ||single_adapter_row|| = scale * ||single_adapter_row|| / sqrt(N)

The perturbation DECREASES with N (1/sqrt(N) law).

**Case 2: Coherent adapters.** If adapters are correlated:

  ||Delta_row(N)|| ~ (scale/N) * N * ||single_adapter_row|| = scale * ||single_adapter_row||

Independent of N (constant).

**Case 3: Additive composition (no 1/N averaging).** If Delta(N) = scale * sum_i(A_i @ B_i):

  ||Delta_row(N)|| ~ scale * sqrt(N) * ||single_adapter_row||   (incoherent)
  ||Delta_row(N)|| ~ scale * N * ||single_adapter_row||          (coherent)

This is the stress scenario where topology WILL break.

### Key Paper: Persistent Topological Features in LLMs (arXiv:2410.11042)

Demonstrates that persistent topological features exist in LLM weight spaces and
are functionally meaningful. Their persistence similarity metric shows these features
are stable under normal training but can be disrupted by large perturbations.

## Step D: Proof of Guarantee (Bounded Degradation)

**Theorem 1 (Scaling of Bottleneck Distance with N).**
Let W be a weight matrix and {B_i}_{i=1}^N be adapter B-matrices with a shared
skeleton A. Define the composed perturbation at scale N as:

  Delta(N) = (alpha/N) * sum_{i=1}^N A_i @ B_i

where alpha is the LoRA scale. Then:

  d_B(Dgm(W), Dgm(W + Delta(N))) <= max_j ||(alpha/N) * sum_{i=1}^N (A_i @ B_i)_j||_2

where (...)_j denotes the j-th row.

*Proof.* Direct application of Theorem 5.2 (Cohen-Steiner et al., 2007) to the
row-wise correspondence W_j <-> W_j + Delta(N)_j. QED.

**Theorem 2 (Critical N for Feature Loss).**
Let p_med be the median persistence of H0 features in Dgm(W). Feature loss becomes
possible (vulnerability window is non-empty at the median) when:

  max_j ||Delta(N)_j||_2 >= p_med / 2

Under the 1/N averaging scheme, if sigma_j(N) = ||(1/N) sum_i (A_i @ B_i)_j|| is
the per-row norm of the unscaled average:

  Critical condition: alpha * sigma_j(N) >= p_med / 2

*Proof.* By Corollary 1 of the stability theorem, features with persistence > 2*delta
survive. Setting delta = p_med/2 gives the threshold at which the median feature
enters the vulnerability window. QED.

### Predictions

**P1 (Monotonicity):** d_B(N) should be monotonically related to ||Delta(N)||.
Since ||Delta(N)|| scales sublinearly or linearly with N (depending on coherence),
d_B should grow monotonically with N if adapters are not adversarially anti-correlated.

**P2 (Feature Loss Threshold):** From the N=5 data, max||delta_i|| ~ 0.3-2.0 while
median persistence ~ 30-58. The vulnerability threshold is p_med/2 ~ 15-29.
Under 1/N averaging, ||Delta(N)|| ~ ||Delta(5)|| * sqrt(N/5) (incoherent case),
giving a critical N of:
  N_crit = 5 * (p_med / (2 * max||delta_5||))^2 ~ 5 * (30/4)^2 ~ 281

Under additive composition (no 1/N), ||Delta|| grows with sqrt(N):
  N_crit = (p_med / (2 * max||delta_single||))^2 ~ (30 / (2*0.4))^2 ~ 1406

Both estimates suggest topology is robust far beyond N=50.

**P3 (Stress test with additive scaling):** To actually observe feature loss within
N <= 50, we may need to use ADDITIVE composition (no 1/N averaging) or increase the
LoRA scale. This is the stress test scenario.

**P4 (Empirical check):** For N in {5, 10, 15, 24, 50}, measure:
- max_i ||Delta_i|| (perturbation norm)
- d_B for H0 and H1 (bottleneck distance)
- Number of features with persistence below vulnerability window
- Spearman correlation between N and d_B

## Step E: Assumptions & Breaking Conditions

**A1: Synthetic adapters are representative.** We only have 5 real adapters.
For N > 5, we generate synthetic adapters by sampling from the distribution of
real adapter B-matrices (matching mean, variance, and spectral profile). If synthetic
adapters differ qualitatively from real ones, N>5 results are approximate.

**A2: Row subsampling is representative.** Same as parent experiment. 500/2560 rows.

**A3: 1/N averaging is the realistic composition scheme.** If production uses
different weighting, the scaling law changes.

## Step F: Worked Example (N=5 vs N=10, single module)

From the N=5 experiment, layer_0 q_proj:
- max_delta_norm = 0.41
- median_persistence_h0 ~ 30
- vulnerability_window = 2 * 0.41 = 0.82

At N=10 (synthetic, 1/N averaging, incoherent):
- Expected max_delta ~ 0.41 * sqrt(10/5) / (10/5) ~ 0.41 * 1.41 / 2 ~ 0.29
- The norm DECREASES because 1/N averaging dominates sqrt(N) growth
- Vulnerability window ~ 0.58 (LESS than N=5)

At N=10 (additive, no 1/N):
- Expected max_delta ~ 0.41 * sqrt(10/5) ~ 0.41 * 1.41 ~ 0.58
- Vulnerability window ~ 1.16 (MORE than N=5)

## Step G: Complexity

For each N value, we compute PH on 1-2 representative layers (not all 5) with
a focused set of projection types. This keeps runtime at ~5-10 min per N point,
total ~30-40 min for the full sweep.

## Self-Test

1. **What is the ONE mathematical property?**
   The stability theorem bounds bottleneck distance by max row perturbation norm,
   so topology is safe whenever perturbation norm stays below half the median
   persistence.

2. **Which existing theorem(s)?**
   Algebraic Stability Theorem (Cohen-Steiner et al., 2007, Theorem 5.2).

3. **What specific numbers does the proof predict?**
   Under 1/N averaging, the perturbation norm DECREASES with N (incoherent case),
   so NO features should be lost at any N. Under additive composition, critical
   N ~ 1400 (far beyond our sweep). Feature loss within N=50 requires unrealistic
   adapter coherence.

4. **What would FALSIFY the proof?**
   If d_B exceeds max||delta_i|| (stability theorem violation = implementation bug).
   If features are lost when persistence > 2*max||delta_i|| (same). The proof itself
   cannot be falsified; it can only be vacuously true (bound too loose).

5. **How many hyperparameters?**
   Count: 0 new. We use existing adapter scale, rank, and subsample size.

6. **Hack check:** Single measurement sweep, no fixes stacked. Clean exploration.
