# Coverage vs Noise Disentangle: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Value/Range |
|--------|-----------|-------------|
| d | Model embedding dimension | 64 (micro) |
| r | LoRA rank | 8 |
| N | Training examples per expert | 1000 |
| K | Number of experts per condition | 4 |
| S | Number of seeds | 10 |
| T | SGD steps | 500 |
| W* | Ground-truth task matrix | R^{d x d}, rank r |
| A | LoRA input projection (frozen) | R^{r x d} |
| B | LoRA output projection (learned) | R^{d x r} |
| M | Number of input modes (coverage factor) | {5, 20} |
| sigma | Label noise std (noise factor) | {0.05, 0.30} |

## 2. Factorial Design

### 2.1 The Confound

The parent experiment (synthetic_vs_real_data) compared two conditions that
differ on BOTH coverage and noise simultaneously:

    Synthetic: M=5 modes, sigma=0.05  (low coverage, low noise)
    Real:      M=20 modes, sigma=0.30 (high coverage, high noise)

The observed 58% quality gap Q(real) - Q(synthetic) could be decomposed as:

    total_gap = coverage_effect + noise_effect + interaction

Without the 2x2 design, the attribution is ambiguous.

### 2.2 Full 2x2 Factorial

| | Low Noise (sigma=0.05) | High Noise (sigma=0.30) |
|---|---|---|
| Low Coverage (M=5) | Q_LL | Q_LH |
| High Coverage (M=20) | Q_HL | Q_HH |

The two new conditions (Q_LH and Q_HL) break the confound.

### 2.3 Effect Definitions

**Grand mean:**

    mu = (Q_LL + Q_LH + Q_HL + Q_HH) / 4

**Main effect of coverage** (positive = high coverage better):

    E_cov = (Q_HL + Q_HH) / 2 - (Q_LL + Q_LH) / 2

**Main effect of noise** (positive = low noise better):

    E_noise = (Q_LL + Q_HL) / 2 - (Q_LH + Q_HH) / 2

**Interaction** (departure from additivity):

    I = (Q_LL + Q_HH) / 2 - (Q_LH + Q_HL) / 2

**Verification:** Q_HH - Q_LL = E_cov - E_noise + I
(Going from low-cov/low-noise to high-cov/high-noise gains coverage
but loses noise quality.)

### 2.4 Variance Decomposition

Sum of squares (standard two-way ANOVA):

    SS_total = sum_ij (Q_ij - mu)^2

    SS_cov = 2 * [(Q_H. - mu)^2 + (Q_L. - mu)^2]
           = 2 * (E_cov/2)^2 * 2 = E_cov^2

    SS_noise = E_noise^2

    SS_interaction = I^2

Percent variance explained:

    %_cov = SS_cov / SS_total * 100
    %_noise = SS_noise / SS_total * 100
    %_interaction = SS_interaction / SS_total * 100

These sum to 100% by construction (balanced 2x2, no replication within cells
at the aggregated level).

### 2.5 Simple Effects

Conditional effects isolate one factor at a fixed level of the other:

    coverage_at_low_noise  = Q_HL - Q_LL
    coverage_at_high_noise = Q_HH - Q_LH
    noise_at_low_cov       = Q_LL - Q_LH
    noise_at_high_cov      = Q_HL - Q_HH

If interaction is zero: simple effects equal the main effect.
If interaction is non-zero: simple effects differ across levels.

## 3. Statistical Testing

### 3.1 Per-Seed Design

Each seed s in {1, ..., S} produces an independent realization:
- Fresh W*, mode centers, random A
- 4 experts per condition
- Quality averaged across experts: Q_ij(s)

The per-seed ANOVA decomposition yields E_cov(s), E_noise(s), I(s).
We test these distributions against zero using one-sample t-tests.

### 3.2 Paired Tests for Simple Effects

For each simple effect, we compute per-seed differences and test with
a paired t-test:

    H0: E[Q_HL(s) - Q_LL(s)] = 0  (coverage at low noise)
    H0: E[Q_LL(s) - Q_LH(s)] = 0  (noise at low coverage)

Degrees of freedom: S - 1 = 9.

### 3.3 Kill Criteria Mapping

K1: "coverage alone explains <50% of the 58% quality gap"
  -> %_cov < 50 -> KILLED (coverage insufficient)
  -> %_cov >= 50 -> SURVIVES (coverage is the driver)

K2: "noise alone explains >80% of the gap"
  -> %_noise > 80 -> KILLED (coverage is irrelevant)
  -> %_noise <= 80 -> SURVIVES (coverage matters)

## 4. Controlled Variables

To isolate coverage and noise, the following are held constant or
matched across conditions:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Systematic bias | 0.0 | Removed (was 0.25 in parent synthetic) |
| Dirichlet alpha | 0.5 (M=5), 2.0 (M=20) | Matched to parent |
| Input noise scale | 0.3 (M=5), 0.8 (M=20) | Matched to parent |
| Benchmark overlap | Not modeled | Contamination is an input parameter, not a finding |

**Design note:** We keep Dirichlet alpha and input noise scale tied to coverage
level (5 modes -> concentrated, 20 modes -> spread) because these together
define "coverage" as a compound treatment. The alternative (holding alpha and
scale constant while varying only mode count) would test a narrower definition
of coverage. We match the parent experiment's operational definition.

## 5. Worked Example at Micro Scale

d=64, r=8, N=1000, 10 seeds, 4 experts per condition:

**Cell means (quality on uniform eval):**

| | sigma=0.05 | sigma=0.30 |
|---|---|---|
| M=5 | 0.0238 | 0.0234 |
| M=20 | 0.0601 | 0.0584 |

**Effects:**
- E_cov = (0.0601 + 0.0584)/2 - (0.0238 + 0.0234)/2 = 0.0593 - 0.0236 = +0.0357
- E_noise = (0.0238 + 0.0601)/2 - (0.0234 + 0.0584)/2 = 0.0420 - 0.0409 = +0.0010
- I = (0.0238 + 0.0584)/2 - (0.0234 + 0.0601)/2 = 0.0411 - 0.0418 = -0.0006

**Variance explained:**
- Coverage: 96.2% (dominates)
- Noise: 1.6% (negligible)
- Interaction: 2.2% (negligible)

**Interpretation:** Increasing coverage from 5 to 20 modes improves quality
by +0.036 (a 150% relative improvement from the low-coverage baseline).
Reducing noise from 0.30 to 0.05 improves quality by only +0.001 (a 4%
relative improvement). The 6x noise ratio has essentially zero effect.

## 6. Assumptions and Limitations

1. **Coverage and input distribution are coupled.** We define "coverage"
   as the compound of (mode count, Dirichlet alpha, input noise scale).
   A finer-grained ablation could separate these three sub-factors.

2. **d=64 makes effective rank near-maximal.** Both M=5 and M=20 fill
   the 64-dimensional space with N=1000 samples. The coverage difference
   manifests through gradient direction concentration, not dimensionality.
   At d=4096, the dimensionality difference would be much more pronounced.

3. **Linear task only.** W* is rank-r linear. Nonlinear tasks may show
   different sensitivity to noise (e.g., noise near decision boundaries
   may matter more).

4. **Noise is isotropic Gaussian.** Real label noise is structured
   (systematic teacher errors, domain-dependent). Structured noise could
   interact differently with coverage.

5. **Frozen random A.** With Grassmannian-initialized A, the subspace
   placement may change the coverage-noise tradeoff.
