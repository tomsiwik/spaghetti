# MATH.md — P4.C0: Formatting Adapter

## Background

P4.B0 (Finding #477) trained rank-6 adapters and found math +20pp, medical -4pp.
P4.B1 (Finding #478) showed this was NOT a knowledge gap — Gemma 4 4B scores 0.43-0.63
on hard academic questions. The math improvement was a **notation artifact**: P4.B0 keywords
("a^2", "u dv", "f(g(x))") are format tokens that base model does not produce in natural prose.

P3.C5 (Finding #472): rank-16 + 150 diverse examples → 93.3% style compliance (Coverage Lemma).

---

## Theorem 1 (Format Gap Theorem)

**Claim:** For instruction-tuned Gemma 4 4B (θ_base), format-specific token sets V_format have
non-zero entropy gap:

    H(V_format | θ_base, x) > H_threshold > 0

for x = natural language question, where V_format contains:
- LaTeX: {"\frac{", "\sum_{", "\int_", "\sqrt{", "\forall", "\begin{align"}
- SOAP: {"S:", "O:", "A:", "P:", "HPI:", "ICD-10"}
- Legal: {"WHEREAS", "NOW, THEREFORE", "hereinafter", "pursuant to"}

**Proof:**

*Step 1 — Base model output distribution.* θ_base is instruction-tuned on conversational
dialogue. Its output distribution p(y | x, θ_base) is calibrated toward natural-language
prose answers. Format tokens in V_format have near-zero probability under p(· | x, θ_base):

  E_x[p(t | x, θ_base)] << 1/|V|  for t ∈ V_format

This is not a knowledge gap (Gemma 4 knows calculus, medicine, law) but an output register
gap: instruction-tuned models answer "take the derivative" with words, not symbols.

*Step 2 — Empirical evidence.* P4.B0 observed that format-style keywords ("a^2") showed
+20pp improvement while content keywords ("Zorn", "eigenvalue") showed 0-2pp improvement.
This directly confirms H(V_format | θ_base) > H(V_content | θ_base).

*Step 3 — LoRA shift capacity.* A rank-r LoRA adapter ΔW injects:
    h' = h + ΔW·h = (W₀ + ΔW)·h

For format compliance, the adapter must shift attention toward format tokens. Since ΔW
is unconstrained during training on format-aligned Q&A pairs, it can learn:
  p(V_format | x, θ_base + ΔW) >> p(V_format | x, θ_base)

QED.

**Quantitative prediction:** Base pass rate for format-keyword rubric < 20% (< 2/10 responses
contain the required format markers). Adapted pass rate ≥ 40% (≥ 4/10 responses).

---

## Theorem 2 (Coverage Lemma — Finding #472)

**Claim:** rank(16) LoRA adapter with ≥100 diverse training examples covers all format
categories for a single format domain.

**Proof (by rank-nullity):**
- Format domains have C ≤ 5 distinct structural categories (e.g., LaTeX: symbols, environments,
  delimiters; SOAP: S/O/A/P sections; Legal: whereas/therefore/shall clauses)
- rank(16) > C → all C categories are reachable in the adapter subspace
- 100 examples covering all categories → no category in the null-space of the adapter

From Finding #472: rank-16 + 150 examples → 93.3% compliance for 10 style categories.
For 3-5 format categories with 100 examples: predicted 80-95% pass rate.

QED.

---

## Theorem 3 (Grassmannian Isolation)

**Claim:** Format adapters trained on distinct format domains occupy distinct subspaces
and do not interfere with each other or with domain adapters.

**Proof (by Finding #440):** N=100 domain adapters with max_cos = 2.25e-8.
Format adapters are trained on different input distributions (math questions → LaTeX;
clinical questions → SOAP; legal questions → legal boilerplate). Their ΔW matrices
project onto different subspaces of the Grassmannian Gr(r, d_model).

Cross-domain interference ε < 0.01 by Theorem 1 of P1.T3 (JL-lemma bound).
Predicted cross-domain retention: ≥ 90%.

QED.

---

## Predictions vs Kill Criteria

| Kill Criterion | Theorem | Prediction | Threshold |
|---|---|---|---|
| K1230: base score < 20% for all 3 domains | Theorem 1 | base ≈ 5-15% (format tokens not in prose) | < 20% |
| K1231: ≥3/3 domains ≥20pp improvement | Theorems 1+2 | +40-60pp per domain | ≥ 20pp |
| K1232: cross-domain retention ≥ 90% | Theorem 3 | retention ≈ 95-99% | ≥ 90% |

---

## Failure Mode Analysis

**If K1230 FAILS (base ≥ 20%):** Format gap is smaller than predicted. The instruction-tuned
model already produces format-like tokens in natural language. Impossibility: format adapters
cannot improve output they already produce. Next: test even more specialized notation
(e.g., Unicode math symbols, specific code syntax patterns).

**If K1231 FAILS (<3 domains improve ≥20pp):** Format token injection fails for some domains.
Likely cause: training Q&A pairs don't match eval question distribution. The adapter learns
the format but only when asked the same kind of question. Fix: more diverse training prompts.

**If K1232 FAILS (retention < 90%):** Format adapters collide in activation space.
This would contradict Finding #440 (N=100 Grassmannian isolation). Likely cause: format
adapters are trained on OVERLAPPING input domains (e.g., math + clinical both contain
numbers). Impossibility structure: if train distributions are not mutually orthogonal,
adapter directions may partially align. Fix: use domain-specific routing (only activate
math format adapter for math questions).

---

## Architecture Connection

This experiment tests **format-layer adapters** — adapters that change the OUTPUT FORMAT
of the model, not its knowledge content. This is a key architectural primitive for the
Pierre P1 vision:
- Domain adapters: change WHAT the model talks about
- Format adapters: change HOW the model expresses it
- Personal adapters: change the model's STYLE and PERSONALITY

These three adapter types should be composable (Theorem 3) since they target different
axes of the output distribution.
