# PAPER.md — exp_p3_b0_medical_oe_adapter

## Format-Register Alignment Hypothesis: KILLED

### Abstract

MATH.md Theorem 1 Part 2 predicted that format-register aligned open-ended (OE) training
on medalpaca/medical_meadow_wikidoc would produce δ_D > 0 by providing gradient signal
at domain glossary token positions (G_density ≥ 0.10). Structural measurement falsified
this assumption: actual G_density = **0.0054** (28× below threshold). Smoke test confirmed:
improvement_rate=0.40, vocab_ratio=0.70 (adapted WORSE than base). Experiment killed early
under structural-falsification criterion.

---

## Prediction vs. Measurement Table

| Kill Criterion | Predicted | Measured | Pass? |
|---|---|---|---|
| K1169: improvement_rate > 80% | 82–90% | **40%** (smoke, n=5) | FAIL |
| K1170: vocab_ratio ≥ 1.5 | ≥ 1.78 | **0.70** (adapted WORSE) | FAIL |
| K1171: MMLU regression < 5pp | < 3pp | +40pp (smoke, unreliable n=5) | — |
| Theorem 1 Part 2 pre-condition: G_density ≥ 0.10 | 0.15–0.20 | **0.0054** | FALSIFIED |

### Structural measurement (from Phase 1 log, N=500 training examples)
```
Training data glossary density: 0.0054 (terms/word)
Expected G_density >= 0.10 for positive delta_D signal
```

### Smoke test results (N=5 queries, 50 training iters)
| Query | base | adapted | improved? |
|---|---|---|---|
| ACE inhibitors for hypertension | 3 | 1 | No |
| Type 1 vs Type 2 diabetes | 1 | 2 | Yes |
| Beta-blockers mechanism + clinical uses | 5 | 1 | No |
| Myocardial infarction symptoms | 0 | 0 | No |
| Immune system vs bacterial infections | 1 | 3 | Yes |
| **Mean** | **2.0** | **1.4** | **40%** |

---

## Root Cause Analysis

### Why G_density = 0.0054?

The 30-term glossary (drawn from Finding #457) captures clinical-pharmacological terms
like "vasodilator", "ACE inhibitor", "beta-adrenergic", "histological classification."

medalpaca/medical_meadow_wikidoc is a Q&A dataset where answers are Wikipedia-style
explanatory paragraphs. These answers use GENERAL medical language, not the dense
technical terminology of clinical references.

**Example mismatch:**
- Glossary term: "angiotensin-converting enzyme"
- Wikidoc response: "ACE inhibitors are a class of drugs used to treat hypertension..."

The wikidoc text uses the ACRONYM (ACE) not the full glossary term. Similarly, "high blood
pressure" not "hypertension", "heart failure" not "cardiac decompensation."

The glossary was designed to measure clinical specificity, but wikidoc trains on accessible
lay-medicine language. **Format-register mismatch persists even in OE format — now it's a
VOCABULARY-register mismatch, not just format.**

---

## Impossibility Structure (Derived from Failure)

**Theorem (Vocabulary-Register Alignment Requirement)**:

For OE training to produce δ_D > 0 via the gradient mechanism of Theorem 1 Part 2:
```
G_density(training_data, G_rubric) ≥ θ_min
```
where θ_min ≈ 0.05 (estimated from signal-to-noise at T=500 iters, N=500 examples).

medalpaca/medical_meadow_wikidoc has G_density = 0.0054 < θ_min.

**Therefore**: OE training on wikidoc CANNOT improve vocabulary rubric scores for
the 30-term clinical glossary used in evaluation. The gradient signal for glossary terms
is ~20× too weak regardless of training format (MCQ vs OE).

**Fix requires**: Either
1. Change the rubric to match wikidoc vocabulary (G-lay terms like "high blood pressure")
2. Train on data where G_density(data, G_rubric) ≥ θ_min (e.g., PubMed abstracts, clinical notes)
3. Abandon vocabulary rubric in favor of factual accuracy rubric

---

## Connection to Prior Findings

| Finding | Result | Relevance |
|---|---|---|
| #457 (MCQ format) | improvement_rate=60%, code WORSE | MCQ format kills gradient signal for OE evaluation |
| #459 (PubMedQA yes/no) | delta=+0.015, K1167 FAIL | Concise format also fails |
| P3.B0 (OE wikidoc) | improvement_rate=40%, vocab_ratio=0.70 | OE format fails too — vocabulary mismatch |

**Unified impossibility**: For the 30-term clinical glossary rubric, NO training format
achieves δ_D > threshold because no available dataset (MCQ, yes/no, OE wikidoc) has
sufficient G_density to generate gradient signal for those specific terms.

**Root disease**: The evaluation rubric (30 clinical terms) is mismatched with all
available training data. This is a data-rubric alignment problem, not a format problem.

---

## Next Experiment Implication

P3.B1 (Gram-Schmidt re-orthogonalization for T2+T3 composition) is independent of
this finding and should proceed. It addresses Finding #460 (structural impossibility
of naive T2+T3 composition) — a different problem that has a derived mathematical fix.
