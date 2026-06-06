# Scale Reconciliation: Mathematical Framework

## Type: Guided Exploration

**Proven framework:** LoRA perturbation theory + LIMA hypothesis (2305.11206).
**Unknown:** Whether uniform scale=2.0 matches per-domain optimal behavioral quality.

## A. Failure Mode Identification

**Failure mode:** Using uniform low scale (s=2.0) destroys domain specialization.
At scale s, the composed model computes:

```
y = (W_base + s * B^T A^T) x
```

The adapter perturbation magnitude is ||s * B^T A^T x|| / ||W_base x||.
At s=2.0, this ratio is 10x smaller than at s=20.0.

Finding #217 showed three domain categories:
- **Learnable-task** (math): Peak at s=20, +700% behavioral improvement
- **Structured-output** (code, medical): Peak at s=20, +17-49%
- **Knowledge-dependent** (legal, finance): Peak at s=1-4, degradation at s=20

The risk: s=2.0 may be too low for math/code (losing the 8x math gain) while
being good for legal/finance (avoiding knowledge destruction).

## B. The Right Question

Not: "What is the optimal scale for each domain?"
But: "Is there a scale s* such that the adapter primarily teaches FORMAT
(instruction-following, response structure) without needing to activate
full capability, while preserving base model knowledge across ALL domains?"

The LIMA hypothesis (2305.11206) states: SFT primarily teaches format/style,
not knowledge. If true, low scale suffices because:
1. Format is learned at the model's output distribution boundary
2. Knowledge lives in the base weights W_base
3. High scale overwrites knowledge with adapter-memorized patterns

## C. Prior Mathematical Foundations

**LoRA perturbation theory (Hu et al., 2021, arXiv:2106.09685):**
The effective perturbation ratio rho(s) = s * ||B^T A^T|| / ||W_base||.
When rho >> 1, the adapter dominates base weights.
When rho << 1, the adapter is a small correction to base behavior.

**LIMA (Zhou et al., 2023, arXiv:2305.11206):**
1000 carefully curated examples suffice for high-quality instruction following.
Implication: SFT teaches format at small data scale. By analogy, small
perturbation scale may suffice for format learning.

**Finding #246 (contrastive training, this project):**
At scale=2.0, contrastive-trained adapters showed dramatically different behavior
than at scale=20.0. This suggests scale=2.0 operates in a qualitatively different
regime.

**Finding #238 (behavioral eval, this project):**
Per-domain optimal scales: math s=20 (8/10 correct), code s=20 (8/10 syntax valid),
medical s=20 (0.29 recall), legal s=4 (0.096 recall), finance s=1 (0.156 recall).

## D. Predictions

This is a guided exploration with a proven framework (LoRA perturbation theory)
and an unknown (the scale-behavior mapping at s=2.0).

**Theoretical predictions:**

At s=2.0 vs s=20.0, the perturbation ratio drops by 10x:
rho(2) / rho(20) = 2/20 = 0.1

**P1 (LIMA prediction):** If LIMA hypothesis holds, format quality should be
preserved at s=2.0 (no incoherent output on any domain). Format is a shallow
property of the output distribution.

**P2 (Math degradation prediction):** Math requires activating latent reasoning
chains. At s=2.0, rho is 10x smaller, so the adapter cannot override base model's
weak math patterns. Prediction: math behavioral score drops significantly from
0.80 (s=20) toward base rate (0.10).

**P3 (Knowledge preservation prediction):** For knowledge-dependent domains
(legal, finance), s=2.0 preserves more base knowledge than s=20.
Prediction: legal/finance scores at s=2.0 >= scores at per-domain optimal.

**P4 (Reconciliation prediction):** If s=2.0 is universally better (as Finding
#246 suggests from PPL), it must show behavioral quality within 20% of per-domain
optimal on at least 3/5 domains. Otherwise the contrastive training finding was
measuring a proxy (PPL), not behavior.

**Quantitative predictions table:**

| Domain   | Base    | Per-domain (s=opt) | Predicted s=2.0 | Predicted s=20.0 |
|----------|---------|-------------------|-----------------|------------------|
| math     | 0.10    | 0.80 (s=20)       | 0.10-0.30       | 0.80             |
| code     | 0.42    | 0.62 (s=20)       | 0.42-0.55       | 0.62             |
| medical  | 0.26    | 0.29 (s=20)       | 0.26-0.29       | 0.29             |
| legal    | 0.098   | 0.096 (s=4)       | 0.095-0.100     | <=0.096          |
| finance  | 0.176   | 0.156 (s=1)       | 0.156-0.176     | <=0.156          |

Key prediction: math at s=2.0 will be well below 0.40 (half of s=20's 0.80),
triggering K2. The 8x behavioral gain requires high scale to activate reasoning.

## E. Assumptions and Breaking Conditions

**A1:** Behavioral quality is monotonic in scale for learnable-task domains
(math, code). Breaking: if there is a non-monotonic sweet spot at s=2.0
for math, A1 fails and s=2.0 could outperform expectations.

**A2:** Format quality is robust to scale reduction. Breaking: if s=2.0 is
too small to even teach format, we get incoherent output (triggers K3).

**A3:** The behavioral eval metrics (numerical answer match, syntax parse,
factual recall) are valid proxies for actual quality. Breaking: if metrics
are noisy at n=10, random variation could dominate.

## F. Worked Example (single layer, d=16)

Consider W_base in R^{16x16}, adapter B^T A^T in R^{16x16}.
||W_base||_F ~ sqrt(16*16) * sigma_base ~ 4 * 0.1 = 0.4 (ternary weights)
||B^T A^T||_F ~ sqrt(r) * sigma_adapter ~ 4 * 0.01 = 0.04 (rank-16 LoRA)

rho(s=2) = 2 * 0.04 / 0.4 = 0.2 (adapter is 20% of base)
rho(s=20) = 20 * 0.04 / 0.4 = 2.0 (adapter dominates base by 2x)

At rho=0.2, the model is still primarily W_base with a small correction.
At rho=2.0, the adapter has overridden the base representation.

For math: base gets 1/10 correct. The adapter learned to produce "#### <number>"
format. At rho=0.2, this formatting is weak and the reasoning chain (if learned)
is barely activated. At rho=2.0, the full reasoning pattern fires.

## G. Complexity and Architecture Connection

No additional FLOPs beyond the existing pre-merge composition.
Three generation passes (base, per-domain optimal, uniform s=2.0, uniform s=20.0)
plus evaluation. Memory: one model + one adapter at a time.

Estimated runtime: ~3 x 5 domains x 10 prompts x 128 tokens ~ 15,000 tokens x 3
settings ~ 45,000 tokens total. At ~100 tok/s, ~7.5 min generation + overhead.
Total: ~30-45 min.

## Self-Test

1. **ONE property:** The perturbation ratio rho(s) controls whether the adapter
   overrides base behavior (high rho) or merely adjusts format (low rho).

2. **Existing theorems:** LoRA rank-deficiency (Hu et al., 2021), LIMA
   hypothesis (Zhou et al., 2023, arXiv:2305.11206).

3. **Specific numbers:** Math at s=2.0 predicted to be 0.10-0.30 (vs 0.80 at
   s=20). Legal/finance at s=2.0 predicted >= per-domain optimal.

4. **Falsification:** The framework is wrong if s=2.0 achieves math >= 0.40
   (half of s=20), which would mean reasoning activates at low perturbation
   ratios, contradicting the perturbation dominance model.

5. **Hyperparameters:** 0 added. We are comparing existing scales.

6. **Hack check:** No. This is a measurement experiment, not a fix.
