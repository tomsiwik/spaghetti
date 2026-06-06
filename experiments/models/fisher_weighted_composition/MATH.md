# Fisher-Weighted Adapter Composition: Mathematical Analysis

## Type: Guided Exploration (Type 2)

**Papers:**
- Fisher Merging (arXiv:2111.09832, Matena & Raffel 2022) -- Fisher-weighted averaging for model merging
- EWC (Kirkpatrick et al. 2017, arXiv:1612.00796) -- Fisher diagonal for parameter importance
- DeLoRA (arXiv:2503.18225) -- magnitude carries task-specific information
- FroM (arXiv:2506.02478) -- Frobenius-norm adaptive merging

**Prior findings:**
- Finding #279 -- Frobenius equalization: 50% log-compression works but factor is unprincipled
- Finding #277 -- DC-Merge: cross-domain scale imbalance (20:1) is root cause
- Finding #278 -- Spectral surgery structurally inverted for Grassmannian compositions

**Proven framework:** Grassmannian-orthogonal adapters compose near-losslessly.
Partial Frobenius equalization works but the 50% compression factor is empirical.

**Unknown:** Whether diagonal Fisher information provides per-adapter importance
weights that differ meaningfully from raw Frobenius norms. If decorrelated,
Fisher provides a principled replacement for the ad hoc compression factor.

## A. Failure Mode: Unprincipled Scale Compression

### The Disease

Finding #279 established that the 21.6:1 energy ratio across 5 domain adapters
causes spectral pathology (Gini 0.49) in the composed weight matrix. Partial
Frobenius equalization with a 50% log-space compression factor reduces this to
Gini 0.39, yields 4/5 domains within 5% PPL, and improves mixed PPL by 1.2%.

**The 50% factor is the disease.** It was chosen empirically as a midpoint between
full equalization (which kills high-scale domains) and no equalization (which
silences low-scale domains). There is no mathematical justification for 50% vs
30% or 70%. It may overfit to the N=5, r=16 setup.

The composition needs **per-adapter importance weights** that reflect how much
each adapter's parameters contribute to domain-specific capability, not just
how large they are (Frobenius norm mixes training artifact with genuine signal).

## B. The Right Question (Reframe)

**Wrong:** "What compression factor best balances equalization vs preservation?"
**Right:** "What is the per-parameter importance measure that naturally separates
capability signal from training artifact in LoRA adapter scales?"

The answer is the Fisher Information Matrix. The Fisher diagonal F_i[j] measures
how much the loss changes when parameter j of adapter i is perturbed. High-Fisher
parameters encode genuine capability; low-Fisher parameters encode training noise.

## C. Prior Mathematical Foundations

### Theorem (Fisher Merging -- Matena & Raffel 2022, Theorem 1)

For N models with parameters theta_1, ..., theta_N, each with Fisher Information
Matrix F_i = E_{x~D_i}[grad_theta log p(x|theta_i) grad_theta log p(x|theta_i)^T],
the Fisher-weighted average:

  theta* = (sum_i F_i)^{-1} (sum_i F_i theta_i)

minimizes the weighted sum of KL divergences:

  sum_i KL(p(x|theta*) || p(x|theta_i))

under a local quadratic approximation to the log-likelihood around each theta_i.

**Key insight:** Fisher merging weights each parameter by its importance to the
task, not by its magnitude. A large parameter with low Fisher importance contributes
less than a small parameter with high Fisher importance.

### Adaptation to LoRA Composition

In our setting, each adapter i has delta Delta_i = s_i * B_i^T @ A_i^T.
Rather than merging full model parameters, we weight the per-adapter *contribution*
to the composed delta:

  Delta_composed = sum_i alpha_i * Delta_i

where alpha_i is the Fisher-derived importance weight for adapter i.

### Definition: Adapter-Level Fisher Importance

For adapter i with parameters constituting Delta_i, the Fisher importance is:

  w_i = sum_j F_i[j] * (Delta_i[j])^2

where j indexes the (flattened) parameters of Delta_i, and F_i[j] is the diagonal
Fisher evaluated on domain i's data.

**Interpretation:** w_i measures the total "information content" of adapter i --
parameters that are both large AND important (high Fisher) get high weight.
Parameters that are large but unimportant (low Fisher) get discounted.

This is exactly the decomposition we need: Frobenius norm = sum_j (Delta_i[j])^2
weights all parameters equally, while Fisher-weighted importance = sum_j F_i[j] *
(Delta_i[j])^2 weights by task relevance.

### Proposition (Fisher vs Frobenius Decomposition)

**Claim.** The Fisher importance w_i can be decomposed as:

  w_i = ||Delta_i||_F^2 * E_j[F_i[j] * (Delta_i[j])^2 / ||Delta_i||_F^2]
      = ||Delta_i||_F^2 * <F_i, (Delta_i)^2>_weighted

where <F_i, (Delta_i)^2>_weighted is the Fisher-weighted average of squared
parameters, normalized by total Frobenius norm.

If F_i[j] is constant across all parameters (all parameters equally important),
then w_i = const * ||Delta_i||_F^2, and Fisher weights reduce to Frobenius norms.

If F_i[j] varies significantly across parameters, the Fisher importance captures
structure that Frobenius norms miss.

**This is the key testable prediction:** Fisher weights differ from Frobenius
weights if and only if there is significant variance in per-parameter Fisher
values within each adapter.

## D. Proof of Guarantee

### Theorem 1 (Fisher-Weighted Composition Minimizes Approximate KL)

**Theorem.** Under the local quadratic approximation to the log-likelihood,
the Fisher-weighted composition:

  Delta_composed = sum_i alpha_i * Delta_i,    alpha_i = w_i / sum_j w_j

minimizes:

  sum_i E_{x~D_i}[(grad_theta f(x))^T (Delta_composed - Delta_i)^2 (grad_theta f(x))]

among all convex combinations of {Delta_i}. Here f is the base model log-likelihood.

**Proof sketch.** (Follows from Matena & Raffel 2022, Section 3.1.)

The KL divergence between composed and individual models, under quadratic
approximation, is:

  KL(composed || model_i) approx 0.5 * (theta_composed - theta_i)^T F_i (theta_composed - theta_i)

For our LoRA setting, theta_composed - theta_i = Delta_composed - Delta_i
(restricted to the LoRA subspace).

The objective sum_i KL_i is a quadratic in alpha (the composition weights).
Setting the gradient to zero:

  d/d_alpha_i [sum_j (alpha - e_j)^T F_j (alpha - e_j)] = 0
  => sum_j F_j alpha = sum_j F_j e_j
  => alpha = (sum_j F_j)^{-1} sum_j F_j e_j

For diagonal Fisher (our approximation), this reduces to per-parameter independent
optimization. Aggregating to per-adapter level using the trace inner product gives:

  alpha_i propto Tr(F_i * Delta_i Delta_i^T) = sum_j F_i[j] * Delta_i[j]^2 = w_i

QED.

### Theorem 2 (Fisher Weights are Scale-Aware by Construction)

**Theorem.** Fisher importance w_i incorporates the adapter scale s_i through
both the gradient magnitude and the delta magnitude:

  w_i = sum_j F_i[j] * (s_i * [B_i^T A_i^T]_j)^2
      = s_i^2 * sum_j F_i[j] * [B_i^T A_i^T]_j^2

But F_i[j] itself depends on s_i: adapters with larger scale produce larger
gradients during Fisher computation, so F_i[j] ~ O(s_i^2) for the parameters
affected by the adapter.

**Consequence:** w_i ~ O(s_i^4) in the naive case (double-counting scale).
To avoid this, we compute Fisher on the BASE MODEL with adapter applied, not
on the adapter parameters themselves. This makes F_i independent of s_i
(it measures the base model's parameter sensitivity on domain data).

**Alternative (our approach):** Compute Fisher on the composed model's
parameters (base + adapter delta), measuring how much each parameter position
matters for domain i's data. Then the Fisher importance reflects the
*composed position's* sensitivity, which naturally accounts for whether
the adapter's contribution at that position is beneficial.

### Prediction: Fisher Weight Distribution

Given the known adapter structure:
- B-matrix norms are nearly identical (29.1-31.5 across domains)
- Scale factors are {medical:20, code:20, math:20, legal:4, finance:1}
- Frobenius norms are {medical:627, code:604, math:629, legal:118, finance:29}

**If Fisher weights track Frobenius norms (K3 triggers):**
Fisher weights would be approximately proportional to s_i^2 * ||B_i||_F^2,
giving the same 21.6:1 ratio. The rank correlation would be >0.9.

**If Fisher weights provide new information (K3 passes):**
Fisher weights would rerank the adapters or change the relative spacing.
For example, if finance has high per-parameter Fisher despite low scale,
its Fisher weight would be elevated relative to its Frobenius rank.

The most likely scenario: Fisher partially decorrelates from Frobenius because
the high-scale domains (medical/code/math) have many low-importance parameters
(padded by the large scale), while the low-scale domains (legal/finance) have
concentrated importance. The Fisher-to-Frobenius ratio should be *higher* for
low-scale domains.

## E. Quantitative Predictions (Testable)

### P1: Fisher computation time
**Prediction:** N_samples=20 per domain, 5 domains, forward+backward passes on
sequence length 256. Each pass ~0.15s. Total: 5 * 20 * 0.15 = 15s.
With overhead: <60s total.
**Kill criterion K707:** Fisher computation < 10 minutes.

### P2: Fisher-Frobenius rank correlation
**Prediction:** Spearman rho between Fisher weights and Frobenius norms will be
between 0.7 and 0.9 -- correlated (same-direction) but not perfectly so.
If rho > 0.9, Fisher adds no information beyond Frobenius norms (K708 FAIL).
If rho < 0.7, Fisher provides substantially different weighting.

### P3: Fisher-weighted mixed PPL
**Prediction:** If Fisher decorrelates from Frobenius (P2), Fisher-weighted
composition should improve mixed PPL over partial equalization because it
provides principled per-adapter weights rather than uniform compression.
**Kill criterion K706:** Fisher-weighted mixed PPL < partial equalization mixed PPL (6.508).

### P4: Per-domain PPL pattern
**Prediction:** Fisher weights should naturally downweight the three high-scale
domains' training artifacts while preserving their genuine capability signal.
We expect:
- High-scale domains (med/code/math): within 5% of raw sum (Fisher preserves
  the capability component of their scale)
- Low-scale domains (legal/finance): improved over raw sum (Fisher recognizes
  their per-parameter importance despite low scale)

### P5: Fisher-to-Frobenius ratio per domain
**Prediction:** w_Fisher_i / ||Delta_i||_F^2 should NOT be constant across domains.
If it were constant, Fisher = rescaled Frobenius. We expect this ratio to be
*highest* for finance (low scale but concentrated importance) and *lowest* for
medical/code/math (high scale but diffuse importance from many padding parameters).

## F. Assumptions and Breaking Conditions

1. **Diagonal Fisher approximation is sufficient.** The full Fisher is O(d^2)
   which is impractical. Diagonal Fisher drops cross-parameter correlations.
   If the LoRA parameters have strong cross-parameter structure, diagonal
   Fisher may miss important information.
   **Breaking:** If Fisher diagonal has near-zero variance within each adapter,
   it collapses to a scalar per adapter and provides no more info than Frobenius.

2. **N_samples=20 per domain is sufficient for Fisher estimation.** Fisher
   converges as O(1/sqrt(N)). At N=20, the relative error is ~22%.
   **Breaking:** If Fisher estimates are too noisy, the weights will be unreliable.
   We measure coefficient of variation of Fisher estimates across samples.

3. **Grassmannian orthogonality holds.** Same assumption as Finding #279.
   Verified at |cos|=0.026 (Finding #225).

4. **PPL is a meaningful proxy.** Same caveat as all prior experiments (r=0.08
   correlation with task quality). We add generation samples as behavioral check.

5. **Fisher computed on validation data is representative.** If the adapter's
   importance pattern differs between train and validation distributions,
   Fisher weights may be misleading. We use validation data (same distribution
   as evaluation).

## G. Worked Example (N=2, r=2, d=4)

Two adapters: s_1=20, s_2=1. B and A matrices (r=2, d=4):

B_1 = [[0.5, 0.3, 0.2, 0.1],    A_1^T = [[1, 0, 0, 0],
        [0.4, 0.2, 0.1, 0.05]]             [0, 1, 0, 0]]

B_2 = [[0.4, 0.3, 0.2, 0.15],   A_2^T = [[0, 0, 1, 0],
        [0.3, 0.25, 0.15, 0.1]]            [0, 0, 0, 1]]

Delta_1 = 20 * B_1^T A_1^T (4x4 matrix)
Delta_2 = 1 * B_2^T A_2^T (4x4 matrix)

||Delta_1||_F = 20 * ||B_1||_F = 20 * 0.742 = 14.84
||Delta_2||_F = 1 * ||B_2||_F = 1 * 0.600 = 0.600

**Frobenius weights:** alpha_1 = 14.84^2 / (14.84^2 + 0.6^2) = 0.998, alpha_2 = 0.002
(Domain 2 is essentially invisible.)

**Now suppose Fisher diagonal on domain 1 data gives:**
F_1 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01] (uniform -- low per-parameter importance, spread across many positions)

**And Fisher diagonal on domain 2 data gives:**
F_2 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] (concentrated -- each parameter highly important despite small scale)

w_1 = sum(F_1 * Delta_1^2) = 0.01 * 14.84^2 = 2.203
w_2 = sum(F_2 * Delta_2^2) = 0.5 * 0.600^2 = 0.180

**Fisher weights:** alpha_1 = 2.203 / (2.203 + 0.180) = 0.924, alpha_2 = 0.076

Fisher gave domain 2 a 38x boost over Frobenius (from 0.002 to 0.076) because
the Fisher diagonal recognized domain 2's parameters as individually important
despite their small scale.

**However:** If F_1 and F_2 are proportional to the squared parameters (i.e.,
F_i[j] propto Delta_i[j]^2), then w_i propto sum(Delta_i^4), and the ranking
may not change much from Frobenius (sum of Delta_i^2). The experiment determines
which case holds.

## H. Complexity and Architecture Connection

**Fisher computation:** O(N * N_samples * forward_pass_cost)
= O(5 * 20 * seq_len * d^2 * L) per domain.
For our setup: 5 domains * 20 samples * ~150ms = ~15s total.

**Weight computation:** O(N * n_params) = O(5 * 210 * 16 * d_out) ~ negligible.

**Memory:** Fisher diagonal has the same shape as the adapter parameters.
Per adapter: 210 keys * 16 * d_out * 4 bytes ~ 35 MB. For 5 adapters: 175 MB.
Well within budget.

**Integration:** Fisher weights replace the empirical 50% log-compression factor.
Applied once at composition time as a pre-processing step.

## Self-Test (MANDATORY)

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   Fisher information weights each parameter by its task-relevance (gradient
   sensitivity), making it impossible for training-artifact scale to dominate
   composition weights -- only capability-encoding parameters contribute.

2. **Which existing theorem(s) does the proof build on?**
   Fisher Merging Theorem (Matena & Raffel 2022, arXiv:2111.09832, Section 3.1):
   Fisher-weighted average minimizes sum of KL divergences under quadratic approx.
   EWC (Kirkpatrick et al. 2017): diagonal Fisher as per-parameter importance.

3. **What specific numbers does the proof predict?**
   P1: Fisher computation < 60s (K707 threshold: 10 min).
   P2: Spearman rho(Fisher, Frobenius) in [0.7, 0.9] (K708 threshold: 0.9).
   P3: Fisher-weighted mixed PPL < 6.508 (partial eq baseline, K706).

4. **What would FALSIFY the proof (not just the experiment)?**
   If Fisher diagonal has zero variance within each adapter (all parameters
   equally important), then Fisher = rescaled Frobenius and the decomposition
   provides no new information. This would mean the quadratic approximation
   to KL captures no adapter-internal structure.

5. **How many hyperparameters does this approach add?**
   1: N_samples for Fisher estimation (theory says more is better, practical
   constraint is compute time). We use N_samples=20 matching evaluation set.

6. **Hack check: Am I adding fix #N to an existing stack?**
   No. This replaces the unprincipled 50% log-compression with a single
   theoretically justified weighting scheme. One mechanism: Fisher-weight, then sum.
