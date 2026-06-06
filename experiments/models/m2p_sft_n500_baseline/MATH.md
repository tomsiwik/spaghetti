# MATH.md: SFT n=500 Baseline Measurement

## TYPE: Verification (Type 1)
## PROVEN FRAMEWORK: Wilson (1927) interval and Newcombe (1998) two-proportion z-test
## GOAL: Statistical closure — show current CI_lower=0.773 is biased upward; compute calibrated quality_ratio CI with propagated SFT uncertainty

---

## A. Failure Mode Identification: Upward Bias in quality_ratio CI

### The disease

v4 (exp_m2p_qwen06b_gsm8k_v4) reports:

```
quality_ratio = (M2P_acc - base_acc) / (SFT_acc - base_acc)
             = (0.286 - 0.200) / (0.260 - 0.200)
             = 0.086 / 0.060 = 1.433
```

The 95% CI on quality_ratio was computed as:

```
se_q = se_M2P / (SFT_acc - base_acc)
     = sqrt(0.286 * 0.714 / 500) / 0.060
     = 0.336

CI_lower = 1.433 - 1.96 * 0.336 = 0.773
```

**This CI treats SFT_acc as a known constant.** It is not. SFT_acc = 0.260 was measured
at n_sft = 200 (Wilson CI [0.204, 0.323]). The denominator (SFT_acc - base_acc = 0.060)
carries substantial uncertainty.

The review correctly labeled CI_lower = 0.773 as "optimistic." The actual 95% CI is wider.

### Why this is a root-cause problem, not a symptom

The Fieller method for ratio uncertainty shows that Var(ratio) has two additive terms:

```
Var(quality_ratio) ≈ Var(M2P_acc) / (SFT_acc - base_acc)^2
                   + Var(SFT_acc) * (M2P_acc - base_acc)^2 / (SFT_acc - base_acc)^4
```

With n_sft = 200, Var(SFT_acc) ≈ 0.260 * 0.740 / 200 = 0.000962.
The second term:

```
0.000962 * (0.086)^2 / (0.060)^4 = 0.000962 * 0.00740 / 0.0000130
                                  = 0.000000071 / 0.0000130 ≈ 5.5
```

This second term (≈5.5) dominates the first term (≈0.408/0.0036 = 0.113), increasing
total variance by roughly 50x compared to the current calculation that ignores it.
The current CI_lower = 0.773 is therefore not just slightly optimistic — it is
substantially biased upward due to the small SFT sample size.

**The fix:** Remeasure SFT at n_sft = 500. This reduces Var(SFT_acc) by 2.5x
(from 0.000962 to 0.000385) and produces a calibrated CI.

---

## B. Prior Mathematical Foundations

### Theorem B1 (Wilson Score Interval, Wilson 1927)

For k successes in n independent Bernoulli trials with unknown probability p:

```
CI = [hat_p + z^2/(2n) ± z * sqrt(hat_p*(1-hat_p)/n + z^2/(4n^2))] / (1 + z^2/n)
```

where z = 1.96 for 95% confidence. This interval has near-exact coverage even for
small n and extreme p (unlike the Wald interval which fails near 0 or 1).

**Conditions that apply here:** n_sft = 500 independent GSM8K test problems, each
evaluated pass/fail. The binomial model is appropriate. Wilson CI is preferred over
Wald for n < 1000 or p outside [0.2, 0.8].

### Theorem B2 (Two-Proportion Z-Test, Newcombe 1998)

For two independent proportions p1 (n1 trials) and p2 (n2 trials):

```
z = (p1 - p2) / sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
```

where p_pool = (k1 + k2) / (n1 + n2) is the pooled proportion under H0: p1 = p2.
Under H0, z ~ N(0,1) asymptotically, giving p-value = 2 * (1 - Phi(|z|)).

**Conditions:** n1 = 500 (M2P), n2 = 500 (SFT). Both n*p >= 5 and n*(1-p) >= 5
(since p ≈ 0.27-0.29 and n=500). Central limit theorem applies.

### Theorem B3 (Delta Method for Ratio Variance, Casella & Berger 2002, §5.5.4)

For a ratio R = f(X,Y) = (X - c) / (Y - c) with constants c:

```
Var(R) ≈ [df/dX]^2 * Var(X) + [df/dY]^2 * Var(Y) + 2 * [df/dX][df/dY] * Cov(X,Y)
```

Since M2P accuracy and SFT accuracy are measured on independent test sets,
Cov(M2P_acc, SFT_acc) = 0, giving Fieller's approximation:

```
Var(quality_ratio) ≈ Var(M2P_acc)/(SFT_acc - base)^2
                   + Var(SFT_acc)*(M2P_acc - base)^2/(SFT_acc - base)^4
```

**Conditions:** quality_ratio = (M2P_acc - base_acc)/(SFT_acc - base_acc) with
base_acc = 0.200 treated as fixed (measured at n_base=200 with high precision
via K909 PASS). X = M2P_acc (binomial, n=500, fixed). Y = SFT_acc (binomial, n=500,
to be measured). The delta method is valid when Var(Y) is small relative to
(E[Y] - base)^2, which holds for n=500, SFT_acc ≈ 0.26, base=0.20.

---

## C. Proof of Guarantee

### Theorem 1: Wilson CI for SFT at n=500 is Computable and Calibrated

**Statement.** Given k_sft successes in n_sft = 500 independent GSM8K evaluations
using the v2 SFT adapter (with identical few-shot prefix, generation parameters,
and answer extraction to prior experiments), the Wilson 95% CI for SFT accuracy is:

```
CI_sft = [hat_p + z^2/1000 ± z * sqrt(hat_p*(1-hat_p)/500 + z^2/1000000)] / (1 + z^2/500)
```

where hat_p = k_sft / 500 and z = 1.96.

*Proof.*
1. Each evaluation is an independent Bernoulli trial with unknown probability p_sft.
2. Independence holds because: (a) test examples are drawn i.i.d. from GSM8K test split;
   (b) model state is reset between examples (no KV cache carries over); (c) SEED=42
   ensures a fixed, deterministic test order.
3. By Wilson (1927), the score interval is obtained by inverting the score test
   for H0: p = p0, which gives a quadratic in p0 with closed-form solution as above.
4. Brown, Cai & DasGupta (2001, Statistical Science) prove that the Wilson CI achieves
   coverage within 0.01 of the nominal 0.95 for all n >= 10. At n=500, actual coverage
   is indistinguishable from 0.95.
QED.

### Theorem 2: CI_lower = 0.773 is Biased Upward; n=500 SFT Measurement Reduces Bias

**Statement.** The CI_lower = 0.773 reported in v4 underestimates the true uncertainty
in quality_ratio by ignoring SFT sampling variance. With SFT remeasured at n_sft = 500,
the delta-method CI_lower is strictly lower than 0.773 for any SFT point estimate
in the plausible range [0.22, 0.30].

*Proof.*
Let:
- p_M2P = 0.286, n_M2P = 500 (fixed, not remeasured)
- p_base = 0.200 (fixed)
- p_sft = SFT point estimate at n_sft = 500
- delta = p_sft - p_base (denominator of quality_ratio)

**v4 variance (SFT treated as constant):**
```
Var_v4(quality_ratio) = Var(p_M2P) / delta^2
                      = p_M2P*(1-p_M2P) / (n_M2P * delta^2)
                      = 0.286*0.714 / (500 * delta^2)
                      = 0.2042 / (500 * delta^2)
```

**Correct variance (both sources of uncertainty):**
```
Var_true(quality_ratio) = Var(p_M2P)/delta^2 + Var(p_sft)*(p_M2P - p_base)^2/delta^4
                        = 0.2042/(500*delta^2) + p_sft*(1-p_sft)/(500*delta^4) * (0.086)^2
```

The second term is strictly positive for all p_sft in (0,1). Therefore:
```
Var_true > Var_v4  =>  se_true > se_v4  =>  CI_lower_true < CI_lower_v4 = 0.773
```

The magnitude of the bias depends on p_sft. For p_sft = 0.260 and delta = 0.060:

```
se_true^2 = 0.2042/(500*0.0036) + 0.260*0.740/(500*0.0036^2) * 0.007396
           = 0.1135 + 0.1935*0.007396/0.00001296
           = 0.1135 + 0.1935 * 570.5
           = 0.1135 + 110.4
```

Wait — this gives se_true >> se_v4, which is the Fieller blowup when n_sft is small
relative to the denominator size. Let me restate with correct numerics.

delta = p_sft - p_base = 0.260 - 0.200 = 0.060

```
Term1 = Var(p_M2P)/delta^2 = (0.286*0.714/500) / 0.0036
       = 0.000408 / 0.0036 = 0.1135

Term2 = Var(p_sft)*(p_M2P-p_base)^2/delta^4
       = (0.260*0.740/500) * (0.086)^2 / (0.060)^4
       = 0.000385 * 0.007396 / 1.296e-5
       = 2.847e-6 / 1.296e-5
       = 0.2197
```

So:
```
Var_true = 0.1135 + 0.2197 = 0.3332
se_true = sqrt(0.3332) = 0.577
CI_lower_true = 1.433 - 1.96 * 0.577 = 1.433 - 1.131 = 0.302
```

vs. v4's calculation:
```
Var_v4 = 0.1135  (Term2 omitted)
se_v4 = sqrt(0.1135) = 0.337
CI_lower_v4 = 0.773  (as reported)
```

**Conclusion:** When SFT uncertainty is propagated correctly, CI_lower drops from
0.773 to approximately 0.30 (if SFT at n=500 matches v2's 26.0%).

This proves that 0.773 is upward-biased by approximately 0.47 — a substantial bias
that changes the qualitative conclusion from "M2P robustly beats SFT" to "M2P is
comparable to SFT within uncertainty."

QED.

**Remark.** If SFT accuracy at n=500 is higher than 0.260 (e.g., 0.280), the
denominator delta = 0.080 grows, Term2 shrinks, and CI_lower improves. If SFT is
lower (e.g., 0.240), delta = 0.040 shrinks, Term2 grows, and CI_lower falls further.
The experiment measures which regime holds.

---

## D. Quantitative Predictions (Derived from the Proof)

### Primary predictions (before experiment):

**P1 (Wilson CI for SFT at n=500):**
Based on v2's SFT accuracy of 26.0% at n=200, the SFT model's true accuracy is in
[0.20, 0.32] with 95% confidence. At n=500, the Wilson CI will be roughly:
```
Expected SFT accuracy: ~0.24-0.30 (based on v2 point estimate + binomial noise)
Expected Wilson CI width at n=500: ~2*z*sqrt(0.26*0.74/500) / ~1.008 ≈ ±0.039
Expected Wilson CI: approximately [0.22, 0.30]
```

**P2 (Two-proportion z-test M2P vs SFT at equal n=500):**
With M2P=28.6% and SFT expected ~26.0%, the gap is ~2.6pp.
```
se_test = sqrt(p_pool*(1-p_pool)*(1/500+1/500)) ≈ sqrt(0.273*0.727*0.004) ≈ 0.0281
z_expected = 0.026 / 0.0281 ≈ 0.93
p_expected ≈ 0.35 (not significant)
```
If SFT comes in at 28.0% or higher, the gap closes further and p increases.
If SFT comes in at 24.0% or lower, z increases and may become significant.

**P3 (CI_lower after uncertainty propagation):**
Using Fieller's delta method with SFT at n=500:
```
If p_sft = 0.26:  CI_lower ≈ 0.30  (Theorem 2 worked example above)
If p_sft = 0.28:  delta=0.080, CI_lower ≈ 0.60 (denominator grows, variance shrinks)
If p_sft = 0.24:  delta=0.040, CI_lower ≈ -0.10 (denominator shrinks, variance explodes)
```

**Kill criteria thresholds (derived from above):**

- K919: SFT accuracy measured with Wilson 95% CI — PASS if n_sft=500 and CI computed
  (this is an unconditional measurement, not a threshold test)
- K920: Two-proportion z-test p-value computed — PASS if p < 0.05 (significant),
  INCONCLUSIVE if p >= 0.05 (gap not established)
- K921: quality_ratio CI_lower recomputed with propagated SFT uncertainty (Fieller) —
  PASS if CI is computed; CI_lower value is the finding, not a threshold

---

## E. Assumptions and Breaking Conditions

| Assumption | Content | Consequence if violated |
|------------|---------|------------------------|
| A1 | Same 500 test examples (SEED=42) as v4 | M2P and SFT are not comparable — use same seed |
| A2 | v2 lora_a + sft_b matrices produce the v2 SFT model | If weights are corrupted, SFT eval gives wrong answer |
| A3 | base_acc = 0.200 is fixed (not remeasured) | If base accuracy differs, quality_ratio denominators shift |
| A4 | Independence: M2P and SFT evals on separate model instances | Correct by construction (different phases) |
| A5 | LORA_RANK=4, LORA_SCALE=5.0 must match v2 | Mismatched rank/scale gives wrong LoRA computation |
| A6 | Same FEW_SHOT_PREFIX and generation parameters | Without identical setup, comparison is invalid |

**Breaking condition for Theorem 1:** If lora_a or sft_b weights are corrupted or
loaded with mismatched rank, the SFT model will produce garbage accuracy (near 0%),
which is detectable as a sanity failure.

**Breaking condition for Theorem 2:** If SFT accuracy at n=500 falls below 0.20
(base accuracy), the SFT adapter has failed to generalize, and quality_ratio becomes
negative — Fieller CI diverges. This would require investigation.

---

## F. Worked Example (d_stat = Wilson CI at n=500)

Given k_sft = 130 correct out of n_sft = 500 (hat_p = 0.260, matching v2):

```
z = 1.96, z^2 = 3.8416
hat_p = 0.260

Numerator center  = 0.260 + 3.8416/(2*500) = 0.260 + 0.003842 = 0.263842
Denominator       = 1 + 3.8416/500 = 1.007683

center = 0.263842 / 1.007683 = 0.2619

Numerator half    = 1.96 * sqrt(0.260*0.740/500 + 3.8416/(4*250000))
                  = 1.96 * sqrt(0.000385 + 3.841e-6)
                  = 1.96 * sqrt(0.000389)
                  = 1.96 * 0.01972
                  = 0.03865

half = 0.03865 / 1.007683 = 0.03836

Wilson CI = [0.2619 - 0.0384, 0.2619 + 0.0384] = [0.2236, 0.3002]
```

Then Fieller CI_lower (p_sft = 0.260, same as v2):

```
delta  = 0.260 - 0.200 = 0.060
ratio  = (0.286 - 0.200) / 0.060 = 1.433

Term1 = (0.286*0.714/500) / 0.060^2 = 0.000408 / 0.0036 = 0.1135
Term2 = (0.260*0.740/500) * (0.086)^2 / 0.060^4
       = 0.000385 * 0.007396 / 1.296e-5
       = 2.847e-6 / 1.296e-5 = 0.2197

se_total = sqrt(0.1135 + 0.2197) = sqrt(0.3332) = 0.577

CI_lower = 1.433 - 1.96 * 0.577 = 1.433 - 1.131 = 0.302
CI_upper = 1.433 + 1.96 * 0.577 = 1.433 + 1.131 = 2.564
```

**Compare with v4:** CI_lower was 0.773. Corrected value: 0.302.
This is the calibration improvement the experiment delivers.

---

## G. Complexity and Architecture Connection

This is a pure measurement experiment — no training occurs. Runtime is dominated
by model inference:

- 500 SFT inferences × ~3-4 tokens/s with LoRA on Qwen3-0.6B-4bit
- Estimated: 500 × 384 max tokens / 50 tokens/s ≈ 3800s worst case
- Expected (typical 100-token responses): 500 × 100 / 50 ≈ 1000s (~17 min)

Memory: Qwen3-0.6B-4bit ≈ 0.5GB. LoRA adds ~1MB. Total: well under limit.

Statistical computation (Wilson CI, z-test, Fieller): O(1) numpy operations.

---

## Self-Test (MANDATORY)

**1. What is the ONE mathematical property that makes the current CI_lower biased?**
The ratio estimator Var(quality_ratio) has two terms; v4 included only the numerator
variance (Var(M2P_acc)), not the denominator variance (Var(SFT_acc)). Remeasuring
SFT at n=500 adds the second term, which is derived from the delta method (Casella &
Berger Theorem 5.5.4).

**2. Which existing theorem(s) does the proof build on?**
- Wilson (1927) — score interval for binomial proportion
- Newcombe (1998, Statistics in Medicine) — two-proportion z-test with pooled variance
- Casella & Berger (2002, §5.5.4) — delta method for functions of random variables
- Brown, Cai & DasGupta (2001, Statistical Science) — coverage properties of Wilson CI

**3. What specific numbers does the proof predict?**
- SFT Wilson CI at n=500: approximately [0.22, 0.30] (if SFT ≈ 0.26)
- Two-proportion z-test p-value: ~0.35 (not significant if M2P=28.6%, SFT≈26%)
- Corrected quality_ratio CI_lower: approximately 0.30 (if SFT=0.26), vs reported 0.773
- CI_lower bias = 0.773 - 0.302 = 0.471 (the "optimism" quantified)

**4. What would FALSIFY the proof?**
- If SFT accuracy at n=500 is exactly 0.200 or below (base rate), Theorem 2 breaks
  because the ratio denominator approaches zero and Fieller CI diverges
- If the v2 SFT adapter files (lora_a_matrices.npz, sft_b_matrices.npz) are corrupted,
  the SFT eval gives invalid results — detectable by SFT accuracy < base accuracy

**5. How many hyperparameters does this approach add?**
Zero. This is a measurement experiment. All parameters (n=500, SEED=42, z=1.96,
LORA_RANK=4, LORA_SCALE=5.0) are fixed by prior experiments or by statistical convention.

**6. Hack check: Am I adding fix #N to an existing stack?**
No. This is not a fix — it is a measurement. The single constraint being applied is
"measure the thing that was left unmeasured" (SFT accuracy at n=500 instead of n=200).
No new mechanisms, regularizers, or tricks are added.
