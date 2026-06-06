# LEARNINGS.md — exp_p3_b0_medical_oe_adapter

**Status**: KILLED (Finding #461)
**Date**: 2026-04-11

## Core Finding

Format-register alignment (OE vs MCQ) does not fix the domain adapter improvement problem.
The root disease is **vocabulary-register mismatch between training data and evaluation rubric**,
not training format.

## Why

Theorem 1 Part 2 requires G_density(training_data, rubric) ≥ θ_min ≈ 0.05 for gradient signal.
medalpaca/medical_meadow_wikidoc has G_density = 0.0054 (28× below threshold): it uses lay
vocabulary ("ACE inhibitors", "high blood pressure") while the 30-term clinical rubric expects
technical terms ("angiotensin-converting enzyme", "systolic hypertension").

## Unified Impossibility (All 3 Formats Killed)

| Format | Dataset | G_density | Result |
|--------|---------|-----------|--------|
| MCQ (Finding #457) | MMLU | ~0 | improvement_rate=60%, code WORSE |
| Yes/No (Finding #459) | PubMedQA | ~0 | delta=+0.015 |
| OE (P3.B0) | wikidoc | 0.0054 | vocab_ratio=0.70 (WORSE) |

δ_D > 0 is impossible when G_density(D_train, G_eval) << θ_min, regardless of format.

## Implications for Next Experiment

**P3.B1** (Gram-Schmidt re-orthogonalization for T2+T3 composition) is independent of this
finding — proceed. It addresses a different structural problem (Finding #460: ε_B=0.1607 > 0.1,
power_ratio=2.96×).

If domain adapters are needed: train on PubMed abstracts or clinical notes where G_density ≥ θ_min,
OR redesign the rubric to match wikidoc vocabulary, OR abandon vocabulary rubric for factual accuracy.
