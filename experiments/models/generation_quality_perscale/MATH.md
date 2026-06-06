# MATH.md: Generation Quality with Per-Domain Optimal Scales

## Type: Guided Exploration

**Proven framework:** Domain-dependent LoRA scale (Finding #217) + LIMA hypothesis.
**Unknown:** Does scale-aware routed composition produce better text than base alone?

---

## A. Failure Mode Identification

**Observed failure (exp_generation_quality_test, KILLED):** Routed composition at
uniform scale=20 was worse than base on 3/5 domains. The TWO-WORLD pattern:
structured domains (code +14.4%, math +142.1%) improved but knowledge-dependent
domains (medical -6.9%, legal -8.6%, finance -11.9%) degraded.

**Root cause (Finding #217):** Scale=20 is optimal for learnable-task (math) and
structured-output (code, medical) domains, but catastrophic for knowledge-dependent
domains (legal -31.6%, finance -13.7%). The original test used uniform scale=20
for all adapters, guaranteeing failure on 2/5 domains.

**Hypothesis:** Applying per-domain optimal scales {math:20, code:20, medical:20,
legal:4, finance:1} from Finding #217 should flip the knowledge-dependent domains
from degradation to preservation-or-improvement, changing the overall verdict from
3/5 worse to at most 1/5 worse.

---

## B. The Right Question

Not: "Does composition work?" (already answered: yes for structured domains).

**Right question:** "Does scale-aware composition (each adapter at its empirically
optimal scale) produce better text than the base model across ALL domain types,
including knowledge-dependent domains where uniform scale failed?"

---

## C. Prior Mathematical Foundations

**Finding #217 (domain-dependent scale):** Three categories:
- Learnable-task (math): s*=20, +700% correctness
- Structured-output (code, medical): s*=20, +17-36% quality
- Knowledge-dependent (legal, finance): s*=1-4, degrade at s=20

**Finding #218 (code dominance is scale artifact):** At per-domain optimal scales,
domain adapters significantly win 2/5 (medical, math), tie 3/5.

**LIMA (2305.11206):** SFT teaches format, not knowledge. High scale overwrites
base knowledge in knowledge-dependent domains.

**Perturbation ratio (Finding from scale sweep):** rho(20) = 0.034, meaning even
at s=20 the perturbation is only 3.4% of base spectral norm. Degradation is not
spectral overwrite but output distribution shift amplified through softmax.

---

## D. Hypotheses and Predictions

**Quantitative hypotheses:**

| Prediction | Source | Threshold |
|------------|--------|-----------|
| H1: Scale-aware routing beats base on >= 4/5 domains | Finding #217 optimal scales | >= 4/5 positive alpha |
| H2: Scale-aware routing beats uniform-s=20 on >= 3/5 | Finding #217 shows 2/5 degrade at s=20 | >= 3/5 improvement |
| H3: Legal domain flips from -8.6% to >= 0% | Finding #217: legal s*=4 vs original s=20 | alpha(4, legal) >= 0 |
| H4: Finance domain flips from -11.9% to >= 0% | Finding #217: finance s*=1 vs original s=20 | alpha(1, finance) >= 0 |
| H5: Math remains strong (>= +100%) | Finding #219: math +700% at s=20 | alpha(20, math) > 1.0 |

**Three conditions compared:**
1. Base model (no adapter)
2. Uniform routing: oracle top-1 routing, all adapters at s=20
3. Scale-aware routing: oracle top-1 routing, per-domain optimal scales

---

## E. Assumptions & Breaking Conditions

1. **Scale sweep results transfer to this test.** The scale sweep used the same
   model, adapters, and eval framework. Should transfer directly.
   Breaking: if prompt distribution differs significantly.

2. **Oracle routing is correct.** We use ground-truth domain labels for routing.
   Breaking: if prompt domain is ambiguous (mitigated by using domain-specific prompts).

3. **Behavioral eval framework is reliable.** Finding #210 validated kappa >= 0.7.
   Breaking: if eval is noisy, small improvements may be missed.

4. **Per-domain scale is sufficient.** We use one scale per domain, not per-prompt.
   Breaking: if optimal scale varies within a domain (e.g., easy vs hard questions).

---

## F. Complexity & Architecture Connection

**Computational cost:** Same as original generation quality test. Three conditions
x 5 domains x 10 prompts = 150 generations + 50 base. At ~2s per generation:
~400s = ~7 min for generation + model loading overhead.

**Memory:** Single adapter loaded at a time (oracle top-1). ~2GB base + ~50MB adapter.
Well within M5 Pro 48GB.

**Total estimated runtime: ~20 min** (including model loading).

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   **Honest answer: there is none.** This is a guided exploration that applies an
   empirically-determined scale profile. The hypothesis is that per-domain scale
   selection from Finding #217 will fix the knowledge-dependent domain degradation.
   No impossibility guarantee exists.

2. Which existing theorem(s) does the proof build on?
   LIMA (2305.11206) + empirical Finding #217 (domain-dependent scale).
   No formal theorem is proven.

3. What specific numbers does the proof predict?
   H1: >= 4/5 domains improve. H3: legal flips to >= 0%. H4: finance flips to >= 0%.
   H5: math >= +100%.

4. What would FALSIFY the proof?
   If scale-aware routing is STILL worse on >= 3/5 domains (K1), the problem is not
   scale but something deeper (adapter quality, base model weakness, eval methodology).

5. How many hyperparameters does this approach add?
   0 new. Uses empirically-determined scales from Finding #217.

6. Hack check: Am I adding fix #N?
   No. This is a retest of the existential question with a corrected methodology.
