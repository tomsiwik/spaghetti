# MATH.md: LoRA Scale Sweep — Generation Quality

## Type: Guided Exploration

**Proven framework:** LoRA perturbation theory + LIMA hypothesis.
**Unknown:** The optimal lora_scale value s* that maximizes domain quality
without destroying base capability.

---

## A. Failure Mode Identification

**Observed failure (Finding #209, #212):** At lora_scale s=20, domain SFT
adapters degrade both their own domain (legal -30%, finance -14% generation
quality) and base capabilities (GSM8K -18pp, HumanEval -15pp).

**Root cause hypothesis:** The LoRA perturbation magnitude s*||Delta_W||
exceeds the regime where the adapter augments the base model and enters
the regime where it replaces base representations.

At inference, the model computes:
```
y = (W + s * B * A) * x = W*x + s * (B * A) * x
```

When s*||B*A|| >> ||W||_eff (the effective contribution of the base weight
to the output), the adapter term dominates and the model behaves as if the
base weights don't exist. This is the "overwrite" regime.

When s*||B*A|| << ||W||_eff, the adapter barely affects output. This is
the "negligible" regime.

Between these lies a "sweet spot" where the adapter meaningfully modifies
behavior without destroying base capability.

---

## B. The Right Question

Not: "How do we prevent adapter degradation?"

**Right question:** "At what perturbation magnitude does an additive LoRA
adapter transition from augmenting to overwriting base model behavior, and
can this transition be characterized by a single parameter (the scale)?"

---

## C. Prior Mathematical Foundations

**LIMA (2305.11206):** Demonstrates that SFT with as few as 1000 examples
primarily teaches response format and style, not new knowledge. The base
model's pre-trained knowledge is the primary source of factual content.

**LoRA (2106.09685):** Theorem: rank-r perturbation W + B*A preserves the
top singular vectors of W when ||B*A||_F / ||W||_F is small. Specifically,
by Weyl's inequality, the singular values of W + Delta satisfy:
|sigma_i(W + Delta) - sigma_i(W)| <= ||Delta||_2

**Perturbation bound (from Weyl):** For scale s:
|sigma_i(W + s*B*A) - sigma_i(W)| <= s * ||B*A||_2

The relative perturbation ratio is:
rho(s) = s * ||B*A||_2 / ||W||_2

When rho(s) < 1, the perturbation is smaller than the base weight's
spectral norm — augmentation regime.
When rho(s) > 1, the perturbation dominates — overwrite regime.

---

## D. Framework and Predictions

**Definition.** Let Q(s, d) denote the behavioral quality score (from the
evaluation framework) of adapter for domain d at scale s. Let Q_base(d)
denote the base model quality on domain d.

**Definition.** The advantage ratio is:
alpha(s, d) = Q(s, d) / Q_base(d) - 1

**Guided exploration objective:** Find s* such that alpha(s*, d) > 0 for
own-domain d (adapter improves its own domain) while Q(s*, d') is not
significantly degraded for other domains d'.

**Hypotheses (motivated by perturbation theory + LIMA, not derived from proof):**

1. **Monotonicity of perturbation:** rho(s) is linear in s, so the
   perturbation magnitude grows linearly. There should be a clear
   transition from "negligible" to "augmenting" to "overwriting."

2. **Scale-dependent behavior prediction:**
   - s=1: Small perturbation. Adapter effect minimal. Q(1,d) ~ Q_base(d).
     Prediction: |alpha(1,d)| < 0.05 for all d.
   - s=2: Moderate perturbation (Finding #215 showed PPL improvement).
     Prediction: alpha(2,d) > 0 for own-domain (matches PPL finding).
   - s=4: Larger perturbation, format effects strengthen.
     Prediction: alpha(4,d) peaks or starts declining.
   - s=8: Approaching overwrite. Format improves but knowledge degrades.
     Prediction: alpha(8,d) < alpha(4,d) or alpha(2,d).
   - s=20: Full overwrite (confirmed by Finding #209, #212).
     Prediction: alpha(20,d) < 0 for prose domains (legal, finance).

3. **Domain-dependent transition:** Code/math domains have structured
   outputs where format IS the capability. Prose domains (legal, finance,
   medical) require factual knowledge from the base model.
   Prediction: s* is higher for code/math than for prose domains.

4. **Code adapter dominance at high scale:** At s=20, code adapter was
   universal best (Finding #208). At lower scales, domain adapters should
   narrow this gap because they are no longer overwriting base knowledge.
   Prediction: At s=2, domain adapters beat code adapter on their own
   domain for at least 2/5 domains (vs 1/5 at s=20).

**Quantitative hypotheses (for kill criteria assessment):**

| Prediction | Source | Threshold |
|------------|--------|-----------|
| P1: alpha(s*,d) > 0.10 for at least one domain | K620 | >10% improvement |
| P2: At s*, code adapter is NOT best on all 5 domains | K621 | <5/5 domains |
| P3: At s=1 and s=2, output is coherent (format score > 0.3) | K622 | >0.3 avg |
| P4: alpha(20,d) < 0 for legal, finance (reproduces #209) | Validation | <0 |
| P5: Optimal s varies by domain type | Theory | s*_prose < s*_code |

---

## E. Assumptions & Breaking Conditions

1. **SFT adapters learned useful domain patterns.** If training was
   insufficient (high val loss), no scale will help. Breaking: all scales
   show alpha(s,d) ~ 0. (This would trigger K620 KILL.)

2. **Base model has domain knowledge.** LIMA says SFT reveals existing
   knowledge. If BitNet-2B-4T lacks domain knowledge, adapters cannot
   reveal it. Breaking: base model scores ~0 on all prose domains.

3. **Scale is the primary control variable.** If adapter quality is
   dominated by training data quality or rank, scale won't matter.
   Breaking: all scales show identical quality (K621 KILL).

4. **Behavioral eval framework is reliable.** Finding #210 validated
   Cohen's kappa >= 0.7. If the framework is noisy, we may not detect
   real differences. Mitigation: n=10 per domain per scale, report
   standard errors.

---

## F. Worked Example (Qualitative)

Consider a legal adapter at scale s, evaluated on a legal question about
contract law.

At s=20: The adapter dominates output. It has learned legal formatting
(section headers, "pursuant to") but overwrites the base model's knowledge
of actual contract law principles. Result: well-formatted but factually
empty response. Factual recall ~0.10 (observed in Finding #209: 0.298).

At s=2: The adapter adds mild legal formatting preference but the base
model's knowledge remains intact. The response has the base model's facts
with slight style shift. Factual recall should be near or above base
(0.423 from Finding #209).

At s=1: Nearly identical to base. Minimal adapter effect. Factual recall
~= base (0.423).

The sweet spot is where format + facts combine: adapter adds structure
while base provides substance.

---

## G. Complexity & Architecture Connection

**Computational cost:** Generating with adapter at any scale is identical
FLOPs — the scale s is a scalar multiplier on the LoRA output. No
additional parameters or computation.

**Memory:** Same as single adapter loading (~50MB rank-16 adapter on 2B
model). We sweep scales sequentially, loading once per (adapter, scale)
pair.

**Total evaluations:** 5 domains x 5 scales x 10 prompts = 250 generations
+ 1 base model run (5 domains x 10 prompts = 50 generations).
At ~2s per generation (128 tokens): ~600s = 10 min for generations.
Plus model loading overhead (~30s per load, ~30 loads): ~15 min.
**Total estimated runtime: ~25 min.**

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   **Honest answer: there is none.** This is a guided exploration, not a
   verification experiment. Weyl's inequality bounds singular value shift
   (|sigma_i(W+sDelta) - sigma_i(W)| <= s*||Delta||_2), but the leap from
   singular value bounds to generation quality is not proven. The perturbation
   ratio rho(s) is a useful diagnostic, not an impossibility guarantee.
   Post-experiment measurement showed rho(20) = 0.034 << 1, meaning even at
   s=20 the perturbation is in the "augmentation" regime by Weyl's criterion,
   yet legal/finance still degrade. The mechanism of degradation operates
   through a channel not captured by spectral norm bounds alone.

2. Which existing theorem(s) does the proof build on?
   Weyl's inequality (matrix perturbation theory) + LIMA (2305.11206).

3. What specific numbers does the proof predict?
   P1: alpha > 0.10 for at least one (s,d). P3: format score > 0.3 at s<=2.
   P4: alpha(20, legal) < 0 (reproduces Finding #209).

4. What would FALSIFY the proof?
   If ALL scales produce identical quality (scale has no effect on behavior),
   the perturbation theory framework is wrong for this adapter type.

5. How many hyperparameters does this approach add?
   0 — lora_scale is the existing hyperparameter being swept.

6. Hack check: Am I adding fix #N?
   No. This is a parameter sweep to understand an existing mechanism, not
   adding a new mechanism.

---

## H. Post-Experiment: Measured rho(s) (Fix #5 from adversarial review)

The perturbation ratio rho(s) was measured for the code adapter across all
30 layers x 7 target keys (210 layer-key pairs):

| Scale s | Mean rho(s) |
|---------|-------------|
| 1 | 0.0017 |
| 2 | 0.0034 |
| 4 | 0.0067 |
| 8 | 0.0135 |
| 20 | 0.0337 |

sigma_W ranges [200, 1477], sigma_BA ranges [0.22, 1.43].

**Implication:** At ALL tested scales, rho << 1. The adapter is always in the
augmentation regime by the Weyl criterion. The "overwrite" hypothesis
(Section A) was wrong — legal/finance degradation at s=20 cannot be explained
by spectral dominance. The degradation mechanism likely operates through
output distribution shift (small weight perturbation amplified through
softmax/attention) rather than singular value overwrite. This is an open
question for future work.
