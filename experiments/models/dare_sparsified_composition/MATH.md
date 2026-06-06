# MATH.md: DARE Sparsified Adapter Composition for Ternary LoRA

## Type: Guided Exploration (Type 2)

The proven framework is DARE (Yu et al., arXiv:2311.03099). The unknown is
the optimal drop rate p for ternary-valued adapters (weights in {-1, 0, +1}).

## A. Failure Mode Identification

**Disease:** Adapter composition at scale s degrades out-of-distribution (OOD)
benchmarks. Finding #263 shows MMLU degrades by 5-6pp regardless of training
objective (NTP or SFT). Finding #260 shows GSM8K -15pp, code -10pp for SFT.

**Root cause:** The perturbation Delta_W = sum_i(B_i^T @ A_i^T) * s introduces
a dense modification to base weights. On OOD inputs x where the adapter domains
are irrelevant, the perturbation acts as structured noise:

  y_composed = W*x + s * Delta_W * x

The interference term s * Delta_W * x has magnitude proportional to
||Delta_W||_F * ||x||, which grows with the number of non-zero entries in
Delta_W. OOD inputs receive full perturbation strength despite receiving no
benefit from domain specialization.

**Why this is the disease, not a symptom:** The interference is proportional to
the DENSITY of Delta_W. Reducing density while preserving expected magnitude
directly attacks the root cause rather than treating downstream symptoms.

## B. Reframing the Question

**Wrong question:** "How do we prevent OOD degradation during adapter composition?"

**Right question:** "What is the sparsest representation of Delta_W that preserves
the adapter's in-distribution effect while minimizing the expected interference
on OOD inputs?"

**DARE's answer (Yu et al., 2311.03099):** Apply a Bernoulli mask M ~ Bernoulli(1-p)
elementwise and rescale by 1/(1-p):

  Delta_W_DARE = (1/(1-p)) * (M odot Delta_W)

This has the property E[Delta_W_DARE] = Delta_W (unbiased estimator), but
||Delta_W_DARE||_0 = (1-p) * ||Delta_W||_0 in expectation.

## C. Prior Mathematical Foundations

### Theorem (DARE Unbiasedness, Yu et al. 2311.03099, Section 3.1)

For delta parameters delta_ij, let M_ij ~ Bernoulli(1-p) i.i.d. Then:

  delta_ij^DARE = delta_ij * M_ij / (1-p)

satisfies E[delta_ij^DARE] = delta_ij for all i,j.

**Proof.** E[delta_ij * M_ij / (1-p)] = delta_ij * E[M_ij] / (1-p) = delta_ij * (1-p)/(1-p) = delta_ij. QED.

### Variance of DARE Estimator

Var[delta_ij^DARE] = delta_ij^2 * p / (1-p)

**Proof.** Var[M_ij/(1-p)] = Var[M_ij]/(1-p)^2 = p(1-p)/(1-p)^2 = p/(1-p).
Since delta_ij^DARE = delta_ij * M_ij/(1-p), and delta_ij is deterministic,
Var[delta_ij^DARE] = delta_ij^2 * p/(1-p). QED.

### Corollary: Variance Explosion at High Drop Rate

At p=0.9: Var = 9 * delta_ij^2. At p=0.95: Var = 19 * delta_ij^2.
The noise-to-signal ratio scales as sqrt(p/(1-p)).

### DARE with Ternary Adapters: Special Structure

Standard DARE operates on FP16 delta parameters with continuous values. Our
adapters use a Grassmannian skeleton (A matrices, shared across domains) with
trained B matrices. The composed delta is:

  Delta_W = scale * B^T @ A^T

where B in R^{r x d_out}, A in R^{r x d_in}, r=16.

The actual delta is a rank-16 matrix. Applying DARE to Delta_W (the
materialized delta) rather than to B or A individually preserves the unbiasedness
property while sparsifying the full d_out x d_in matrix.

**Key insight for ternary interaction:** The base model has ternary weights
{-1, 0, +1}. The LoRA deltas (B^T @ A^T) are continuous-valued (bfloat16).
DARE sparsification applies to the continuous deltas, not the ternary base.
There is no ternary quantization incompatibility because DARE operates on the
perturbation, not the base weights.

### OOD Interference Reduction (informal bound)

For an OOD input x, the expected squared interference is:

  E[||Delta_W_DARE * x||^2] = ||Delta_W * x||^2 + sum_{ij} Var[delta_ij^DARE] * x_j^2

The second term is the variance cost. However, the key observation is that for
OOD inputs, Delta_W*x already acts as noise. The VARIANCE of the DARE estimator
adds noise-on-noise, but the SPARSITY means most entries contribute ZERO
perturbation. The net effect depends on the input distribution.

**Why DARE helps OOD despite variance:** On OOD inputs, the adapter delta is
already harmful (random direction). DARE zeroes out (1-p) fraction of entries,
creating a sparser perturbation. While surviving entries are rescaled larger,
the law of large numbers means the EXPECTED perturbation magnitude per-output-
dimension scales as:

  E[||Delta_W_DARE * x||_inf] ~ (1-p) * (1/(1-p)) * E_active[|delta_ij * x_j|]
                               = E[|delta_ij * x_j|]

So the infinity-norm is preserved, but the actual perturbation is SPARSE.
For transformer hidden states where different dimensions encode different
features, sparsity means FEWER features are corrupted, even if corrupted
features are corrupted by the same amount.

This is the key qualitative prediction: DARE should reduce the NUMBER of
corrupted features, not the amount of corruption per feature.

## D. Predictions

### Behavioral Predictions

1. **P1:** DARE at p=0.9 will reduce OOD degradation on MMLU from ~5pp to <= 3pp,
   because fewer knowledge-critical dimensions are perturbed.

2. **P2:** DARE at p=0.9 will preserve >= 80% of in-distribution behavioral gains
   (GSM8K accuracy for math adapter), because E[Delta_W_DARE] = Delta_W.

3. **P3:** Higher drop rates (p=0.95) may further reduce OOD degradation but will
   increase variance, potentially degrading in-distribution performance.

4. **P4:** DARE is compatible with ternary base weights. No degenerate output.
   The ternary base is untouched; only the continuous LoRA delta is sparsified.

### Quantitative Predictions

| Metric | No DARE (baseline) | DARE p=0.9 (predicted) | Source |
|--------|-------------------|----------------------|--------|
| MMLU degradation | -5pp to -6pp | <= -3pp | Sparsity reduces corrupted dimensions |
| GSM8K (math adapter) | +10pp vs base | >= +8pp vs base | Unbiased estimator preserves expectation |
| Code gen (code adapter) | -10pp | <= -7pp | Fewer features corrupted |
| In-dist math correctness | ~80% | >= 64% (80% of 80%) | K2 threshold: 50% |

### Kill Criteria Derivation

- **K681:** DARE composition STILL degrades OOD by >= 5pp on majority (>=3/5) domains.
  Derived from: if DARE does not reduce corrupted dimensions, sparsification is
  ineffective for this architecture. Threshold 5pp matches Finding #260/263 baseline.

- **K682:** In-distribution gains drop below 50% of non-DARE composition.
  Derived from: unbiased estimator property guarantees E[effect] = original effect.
  If actual effect drops >50%, the variance cost overwhelms the mean, indicating
  the drop rate is too aggressive for this rank.

- **K683:** DARE + ternary produces degenerate output.
  Derived from: DARE modifies the continuous delta only, not the ternary base.
  Degenerate output would indicate a bug, not a mathematical limitation.

## E. Assumptions & Breaking Conditions

1. **Independence assumption:** DARE assumes entries of Delta_W are somewhat
   independent in their contribution to task performance. If the rank-16 structure
   creates strong correlations, random dropping may break critical patterns.
   *Breaking condition:* In-distribution performance drops >50% at p=0.5 (moderate rate).
   This would suggest high correlation structure in the delta.

2. **OOD-as-noise assumption:** We assume that Delta_W*x on OOD inputs acts as
   unstructured noise. If the adapter delta has systematic structure that BENEFITS
   certain OOD tasks, DARE could remove this benefit.
   *Breaking condition:* Some OOD metrics IMPROVE with composition but worsen with DARE.

3. **Rescaling compatibility:** The 1/(1-p) rescaling amplifies surviving entries.
   At p=0.9, survivors are 10x amplified. Combined with scale s=20, effective
   scale on survivors is 200. This may cause numerical issues in bfloat16.
   *Breaking condition:* NaN or degenerate outputs at p >= 0.9.

## F. Worked Example (d=4, r=2)

Base weight W (4x4, ternary):
```
W = [[ 1, -1,  0,  1],
     [ 0,  1, -1,  0],
     [-1,  0,  1,  1],
     [ 1,  1,  0, -1]]
```

LoRA A (2x4), B (2x4), delta = s * B^T @ A^T with s=2:
```
A = [[ 0.5, -0.3,  0.1,  0.2],
     [-0.1,  0.4,  0.3, -0.5]]

B = [[ 0.2, -0.1,  0.3,  0.4],
     [ 0.1,  0.5, -0.2,  0.1]]

B^T @ A^T (4x4):
= [[ 0.09, -0.02,  0.05,  0.01],
   [-0.04,  0.23, -0.14,  0.07],
   [ 0.13, -0.01,  0.09,  0.16],
   [ 0.03, -0.08,  0.07,  0.13]]

Delta = 2 * B^T @ A^T  (scaled)
```

DARE at p=0.5, mask M (Bernoulli(0.5)):
```
M = [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [1, 1, 0, 0],
     [0, 0, 1, 1]]

Delta_DARE = (1/0.5) * M odot Delta
           = 2 * M odot Delta
```

Non-masked entries are doubled, masked entries are zero.
E[Delta_DARE] = Delta (unbiased).
||Delta_DARE||_0 = 8 (vs 16 for full Delta) = 50% density.

For OOD input x = [1, 0, 1, 0]^T (only 2 non-zero features):
- Full delta perturbation: uses all 4 columns
- DARE delta perturbation: uses only ~2 columns (those with non-zero mask entries in cols 0,2)

Fewer output dimensions are perturbed.

## G. Complexity & Architecture Connection

**FLOPs:** DARE adds one Bernoulli sample and one element-wise multiply per delta
parameter. For rank-16 LoRA on 7 projections x 30 layers: trivial cost.

**Memory:** The materialized delta Delta_W = B^T @ A^T is d_out x d_in = 2560 x 2560
per projection. With 7 projections x 30 layers = 210 matrices. At bfloat16: ~2.75 GB.
DARE mask is the same size but binary (storable as uint8: ~1.37 GB). Total overhead
is modest on 48GB M5 Pro.

**Runtime:** DARE is a pre-merge operation. Applied once before inference.
No inference-time overhead.

**Architecture:** Compatible with any LoRA/adapter composition method.
Works with Grassmannian skeleton structure (shared A, domain-specific B).

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   Sparsifying the delta reduces the number of corrupted output dimensions on OOD
   inputs while the unbiased estimator preserves expected in-distribution effect.

2. Which existing theorem(s) does the proof build on?
   DARE unbiasedness (Yu et al. 2311.03099 Section 3.1), Bernoulli variance.

3. What specific numbers does the proof predict?
   MMLU degradation <= 3pp (vs 5-6pp baseline), GSM8K >= +8pp (vs +10pp baseline),
   in-dist math correctness >= 64%.

4. What would FALSIFY the proof (not just the experiment)?
   The proof is wrong if in-distribution performance drops >50% at p=0.5, which
   would indicate the independence assumption fails for rank-16 structure.

5. How many hyperparameters does this approach add?
   Count: 1 (drop rate p). Cannot be derived from the math alone because the
   correlation structure of Delta_W is data-dependent. This is the guided exploration unknown.

6. Hack check: Am I adding fix #N to an existing stack?
   No. DARE is a single modification to the composition mechanism. It replaces
   the composition step (sum of deltas) with (sum of sparsified deltas). One change.
