# MATH.md: Scale-Aware Composition at N=15 and N=24

## Type: Guided Exploration (Type 2)

**Proven framework:** Per-domain optimal scales resolve the two-world problem at N=5
(Finding #220). Topology is self-stabilizing under 1/N averaging (Finding #233).

**Unknown:** Do the N=5 optimal scales {math:20, code:20, medical:20, legal:4,
finance:1} remain effective when N grows to 15 and 24? If not, how do they shift?

---

## A. Failure Mode Identification

**Observed success at N=5 (Finding #220):** With per-domain optimal scales and oracle
top-1 routing, all 5 domains improve over base. The "two-world problem" (uniform
s=20 degrades knowledge-dependent domains) is resolved.

**Potential failure at N >> 5:** When composing N adapters simultaneously (not oracle
top-1, but full composition), two mechanisms could degrade per-domain scale selection:

1. **Cross-adapter interference:** The summed perturbation from N-1 "background"
   adapters could overwhelm the signal from the target domain adapter, even at its
   optimal scale. This would make scale selection irrelevant.

2. **Scale drift:** The optimal scale might shift as N grows because the background
   noise floor changes. If s*_N differs dramatically from s*_5, recalibration is
   needed at each N, making deployment impractical.

---

## B. The Right Question

Not: "How do we prevent degradation at large N?"

**Right question:** "Under 1/N averaging composition, does the perturbation
ratio for a single domain adapter remain scale-independent of N, such that
the N=5 optimal scales remain valid at arbitrary N?"

---

## C. Prior Mathematical Foundations

**Weyl's inequality (matrix perturbation):** For the composed perturbation
Delta_composed = (1/N) * sum_{i=1}^{N} s_i * A_i B_i, the spectral norm satisfies:

  ||Delta_composed||_2 <= (1/N) * sum_i s_i * ||A_i B_i||_2

**Finding #233 (self-stabilization):** Under 1/N averaging with incoherent adapters,
the perturbation norm scales as O(1/sqrt(N)) due to cancellation. Spearman rho(N,
d_B) = -1.0 across N in {5, 10, 15, 24, 50}.

**Perturbation ratio (from MATH.md of lora_scale_sweep):** rho(s) = s * ||BA||_2 /
||W||_2. For the code adapter at s=20, rho = 0.034. Even at s=20, the adapter is
in the augmentation regime (rho << 1).

**Key derivation: Single-adapter signal-to-noise ratio under N-composition.**

Consider oracle top-1 routing with composition: the target adapter d has scale s_d,
and N-1 background adapters have uniform scale s_bg. Under 1/N averaging:

  y = Wx + (s_d/N) * A_d B_d x + (1/N) * sum_{j != d} s_bg * A_j B_j x

The "signal" (target adapter contribution) is:
  S = (s_d/N) * ||A_d B_d x||

The "noise" (background adapters) is:
  E[||noise||^2] = (1/N^2) * sum_{j != d} s_bg^2 * ||A_j B_j x||^2

If adapters are incoherent (cross-terms cancel in expectation):
  E[||noise||] ~ (s_bg/N) * sqrt(N-1) * sigma_BA ~ s_bg * sigma_BA / sqrt(N)

The signal-to-noise ratio is:
  SNR = S / noise ~ (s_d * sigma_d) / (s_bg * sigma_BA * sqrt(N-1) / sqrt(N))
      ~ (s_d / s_bg) * (sigma_d / sigma_BA) * sqrt(N / (N-1))

**Critical observation:** Under 1/N averaging, the SNR is approximately independent
of N for large N (sqrt(N/(N-1)) -> 1). The optimal scale s_d does not depend on N.

Under additive composition (no 1/N normalization):
  noise ~ s_bg * sigma_BA * sqrt(N-1)

This grows as sqrt(N), degrading SNR. The optimal scale would need to increase
proportionally to maintain signal strength.

---

## D. Predictions

**For 1/N averaging composition:**

| Prediction | Derived From | Threshold |
|------------|-------------|-----------|
| P1: Optimal scales at N=15 within 2x of N=5 for >= 3/5 domains | SNR ~ independent of N | K637 |
| P2: Per-domain scales at N=15/24 reduce degradation to <= 1/N | Self-stabilization + SNR | K636 |
| P3: Domains degrade <= 1/N with per-domain scales | Finding #233 self-stabilization | K636 |
| P4: Scale values stable: math/code/medical still peak at s=20 | Structured domains: format IS capability | Stability |
| P5: Legal/finance optimal scales may shift modestly (within 2x) | Knowledge domains sensitive to noise | K637 |

**For uniform scale comparison:**
| Prediction | Derived From |
|------------|-------------|
| P6: Uniform s=20 degrades >= 2/5 domains at N=15/24 | Two-world problem persists |
| P7: Per-domain beats uniform by >= 5% on >= 1 domain | Legal/finance benefit |

**Quantitative bound:** Under 1/N averaging, the per-adapter perturbation ratio at
scale s is rho_N(s) = (s/N) * mean(||A_i B_i||_2) / ||W||_2. At N=5, s=20:
rho_5(20) = 0.034/5 = 0.0068. At N=24, s=20: rho_24(20) = 0.034/24 = 0.0014.
The perturbation shrinks with N, so quality should be at least as good.

---

## E. Assumptions & Breaking Conditions

1. **Incoherent synthetic adapters.** Synthetic adapters are sampled from Gaussian
   matching real adapter statistics. If they happen to align with real adapters
   (unlikely for random draws), interference increases. Breaking: synthetic adapters
   have cos > 0.3 with real adapters.

2. **1/N averaging composition.** The analysis assumes 1/N normalization. Pure
   additive composition would show sqrt(N) growth in noise.

3. **Oracle top-1 routing per domain.** Each domain query is routed to the correct
   adapter. If routing fails, scale selection is moot.

4. **Scale sweep covers the true optimum.** Grid {1, 2, 4, 8, 20} may miss the
   optimum, especially if it shifts to intermediate values.

---

## F. Worked Example (N=15, legal domain, s=4)

At N=5 with 1/N averaging, legal adapter at s=4:
  Signal = (4/5) * ||A_legal B_legal x|| = 0.8 * sigma_BA * ||x||
  Noise from 4 background adapters at s=20:
    ~ (20/5) * sqrt(4) * sigma_BA * ||x|| / sqrt(5)
    = 4 * 2 * sigma_BA / sqrt(5) * ||x|| ~ 3.58 * sigma_BA * ||x||
  SNR_5 ~ 0.8 / 3.58 ~ 0.22

At N=15 with 1/N averaging, legal at s=4:
  Signal = (4/15) * sigma_BA * ||x|| = 0.267 * sigma_BA * ||x||
  Noise from 14 background adapters:
    ~ (20/15) * sqrt(14) * sigma_BA * ||x|| / sqrt(15)
    = 1.33 * 3.74 / 3.87 * sigma_BA * ||x|| ~ 1.29 * sigma_BA * ||x||
  SNR_15 ~ 0.267 / 1.29 ~ 0.21

SNR is approximately preserved (0.22 vs 0.21), confirming the N-independence
prediction for 1/N averaging.

**However:** In practice, the experiment uses oracle top-1 routing (only the matched
adapter is active, not all N). With oracle routing, there is NO background noise.
The only effect of N is the 1/N normalization factor on the single active adapter.
This means the effective scale at N=15 is s_eff = s/3 compared to N=5 (since 5/15 = 1/3).
To compensate, we would need s_N = s_5 * N/5.

**This is the key insight:** Under oracle top-1 with 1/N averaging, the adapter output
is scaled by s/N. At N=5, s=20 gives effective scale 4. At N=15, s=20 gives effective
scale 1.33. The optimal s would need to scale linearly with N.

Under oracle top-1 WITHOUT 1/N averaging (just apply the single adapter), N has no
effect at all. This is the deployment-relevant scenario.

**The experiment should test BOTH composition schemes:**
1. Oracle top-1 (no averaging) -- N should have zero effect on optimal scale
2. Full N-adapter 1/N averaging -- optimal scale shifts linearly with N
3. Full N-adapter additive -- noise grows, per-domain scale still helps

---

## G. Complexity & Architecture Connection

**Computational cost per condition:**
- Load model once per (domain, scale, N) configuration
- Generate 10 prompts x 128 tokens ~ 2s per prompt
- 5 domains x 5 scales x 2 N-values x 3 composition schemes = 150 configs
  This is too many. Optimize: test 2 composition schemes (oracle-top1, 1/N-averaging),
  2 N-values, 5 scales on 5 domains = 100 configs at ~25s each = ~40 min.

**Memory:** Same as single adapter loading for oracle top-1. For full composition,
need to load N adapters but apply them as sum, not store separately. Can compute
composed weight matrix once then generate.

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   Under oracle top-1 routing (the deployment case), N adapters have zero effect
   on the active adapter's output. The optimal scale is determined solely by the
   single adapter's perturbation ratio. N is irrelevant.

2. Which existing theorem(s) does the proof build on?
   Weyl's inequality for perturbation bounds. Finding #233 (self-stabilization
   under 1/N averaging). LIMA (2305.11206) for domain-dependent scale sensitivity.

3. What specific numbers does the proof predict?
   Under oracle top-1: optimal scales identical to N=5 (shift = 0x).
   Under 1/N averaging: optimal scale shifts proportionally to N (3x at N=15, 4.8x at N=24).
   Under additive: background noise grows as sqrt(N), per-domain still helps.

4. What would FALSIFY the proof?
   If under oracle top-1, optimal scales shift by > 2x between N=5 and N=15/24.
   This would mean the model loading/composition infrastructure has a bug where
   N affects single-adapter inference.

5. How many hyperparameters does this approach add?
   0. Per-domain scales are the existing mechanism being validated at higher N.

6. Hack check: Am I adding fix #N?
   No. This is a validation experiment, not a new mechanism. Testing whether the
   existing per-domain scale solution generalizes to larger adapter pools.
