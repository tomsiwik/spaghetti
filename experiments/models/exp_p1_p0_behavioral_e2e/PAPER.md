# PAPER.md — P0: Behavioral E2E Quality

## Prediction vs Measurement

| Metric | Predicted (MATH.md) | Measured | Status |
|--------|---------------------|----------|--------|
| Math improvement rate | ≥80% (Q=ρ·δ≥0.60→80%) | **30%** | **FAIL** |
| Code improvement rate | ≥80% | **20%** (adapted WORSE than base) | **FAIL** |
| Medical improvement rate | ≥70% | **60%** | **FAIL** |
| Legal improvement rate | ≥70% | **20%** (adapted WORSE) | **FAIL** |
| Finance improvement rate | ≥70% | **10%** (adapted WORSE) | **FAIL** |
| Pipeline overhead | ≤10ms | **1.84ms** | **PASS** |
| Overall | ALL PASS | KILLED (5/6 fail) | **KILLED** |

## Vocabulary Scores (Mean Glossary Terms Per Response)

| Domain | Base Mean | Adapted Mean | Delta | Direction |
|--------|-----------|--------------|-------|-----------|
| Math | 1.8 | 2.0 | +0.2 | ↑ slight |
| Code | 2.6 | 2.1 | **−0.5** | ↓ HURT |
| Medical | 1.4 | 2.1 | +0.7 | ↑ positive |
| Legal | 1.9 | 1.7 | **−0.2** | ↓ HURT |
| Finance | 1.9 | 1.5 | **−0.4** | ↓ HURT |

## Why Theorem 1 Was Wrong

Theorem 1 predicted Q_pipeline = ρ_D × δ_D ≥ 0.60 using δ_D ≥ 0.60 from Finding #436.

**Fatal flaw**: Finding #436 (personal adapter) had base compliance = 0%. The base model
never spontaneously adopted the user's specific phrasing. δ = 0.76 because the gap was real.

**Domain adapters trained on MMLU data**: base Gemma 4 already achieves 2.6 code terms/response,
1.9 legal terms/response, 1.9 finance terms/response. The GAP between base and domain is small.

**Worse**: MCQ-trained adapters shift toward CONCISE format. MMLU-style fine-tuning teaches
"select A/B/C/D" rather than "explain in detail". Open-ended queries then get shorter, less
technical responses → vocabulary DECREASES.

Structural impossibility for Theorem 1:
- δ_D = P(adapter better | correct route)
- For capable base + MCQ training: δ_D < 0 possible (negative improvement)
- Q_pipeline = ρ_D × δ_D can be NEGATIVE → routing correctness irrelevant

## What Actually Happened

The P0 gap was: **training format mismatch**. Adapters trained on MMLU (MCQ) were evaluated on
open-ended explanatory queries. MCQ training specializes toward:
- Shorter responses
- "The answer is X because..." patterns (not full explanations)
- Domain signal captured in concise answer tokens, not explanatory vocabulary

Result: vocabulary-based behavioral rubric penalizes MCQ-adapted models that give concise
answers vs base model's verbose explanations.

## Finding

**KILLED**: Domain adapters (rank-4 LoRA, 1000 steps, MMLU MCQ data) do not improve
domain vocabulary usage for open-ended explanatory queries. Adapters trained on MMLU actually
reduce vocabulary richness for code (−0.5 terms/response), legal (−0.2), finance (−0.4).
Medical shows slight improvement (+0.7) but below the 70% improvement threshold.

**Impossibility structure**: The PPL↛behavioral quality gap (r=0.08, established) extends to
vocabulary metrics. MCQ-format training creates a format-register mismatch: the adapter shifts
toward concise Q/A style while the behavioral rubric rewards explanatory vocabulary density.
No amount of routing accuracy compensates when δ_D ≈ 0 or δ_D < 0.

**Next step**: Domain adapters trained on open-ended explanation data (not MMLU MCQ) would
test whether behavioral vocabulary improvement is achievable. Alternatively, accept that
domain adapters serve PPL quality (which is real) but don't add behavioral vocabulary richness
beyond a capable base model.
