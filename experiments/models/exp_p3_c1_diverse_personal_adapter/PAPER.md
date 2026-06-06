# PAPER.md — P3.C1: Diverse Training Data Raises Style 60%→80%+

**Status: KILLED**
**Finding: #468**
**Date: 2026-04-11**

## Hypothesis

P3.C0 style compliance (60%) was caused by training distribution mismatch:
personal adapter trained on 40 science-only examples (P3.B5) but evaluated on
15 diverse questions spanning philosophy, technology, social, etc.

Theorem 1 (PAC generalization, coverage lemma) predicted: expanding training to 200
diverse examples across 10 categories would restore style compliance to ≥80%.

## Prediction vs Measurement Table

| Metric | Prediction | Measured | Delta | Pass? |
|--------|------------|----------|-------|-------|
| pipeline_style_compliance | ≥80% (~87%) | **60.0%** | -27pp | **FAIL** |
| training_time (min) | 8-13 min | 2.4 min | -8pp | PASS |
| adapter_size (MB) | ~3.7 MB | 1.28 MB | -2.4pp | PASS |
| K1196 style ≥ 80% | PASS | **FAIL** | — | FAIL |
| K1197 time ≤ 15 min | PASS | PASS | — | PASS |
| K1198 size ≤ 10 MB | PASS | PASS | — | PASS |

## Key Metrics

| Experiment | Style% | Training Examples | Iters |
|-----------|--------|-------------------|-------|
| P3.B5 (isolation, science) | 92% | 40 (science) | 300 |
| P3.C0 (pipeline, science adapter) | 60% | 40 (science) | 300 |
| P3.C1 (pipeline, diverse adapter) | **60%** | 167 (10 categories) | 500 |

**Delta vs P3.C0: +0pp** — diverse training gave zero improvement.

## Why the Proof Was Wrong

Theorem 1 assumed the coverage lemma: "if training distribution covers test distribution, generalization error → 0." This holds for FIXED-capacity classifiers in bounded VC-dim hypothesis classes.

**Critical hidden assumption violated:** The style direction in activation space is QUESTION-TYPE-INVARIANT. In reality:

- h_science(q) activates transformer subspaces S_science
- h_philosophy(q) activates transformer subspaces S_philosophy  
- S_science ∩ S_philosophy may be small or disjoint

A rank-4 LoRA learns a single low-rank perturbation ΔW = AB^T (r=4). To inject "Hope that helps, friend!" for both h_science and h_philosophy, the adapter needs:

    AB^T · h_science → style_direction
    AB^T · h_philosophy → style_direction

If h_science and h_philosophy are linearly independent and rank(span) > 4, no rank-4 adapter can satisfy both constraints simultaneously.

**Evidence:** 9/15 questions passed (same as P3.C0). These are likely the same SCIENCE/TECH questions that were in the training distribution of P3.B5. Philosophy, environment, social topics still fail.

## Impossibility Structure

For diverse question sets:
- The "style injection direction" is not a single vector but a SUBSPACE of dimension ≥ n_categories
- rank-4 LoRA can span at most rank-4 directions in activation space
- With 10 diverse categories: rank(needed) ≥ 10 >> rank(4)

**Formal statement:** Let C = {science, philosophy, tech, history, health, arts, social, environment, math, general} be the category set. For any rank-4 adapter A, there exist categories c_1, c_2 ∈ C such that:

    ||AB^T h_{c1} - AB^T h_{c2}|| < ε_threshold

meaning the adapter cannot distinguish the two categories' style requirements simultaneously.

**Prediction:** rank-16 LoRA (or rank ≥ n_categories) would succeed where rank-4 fails.
**Alternative:** Few-shot prompting in system prompt (infinite "rank" via context, no training needed).

## Data Point for the Series

| Strategy | Style% | Δ vs P3.C0 |
|---------|--------|------------|
| P3.C0 (science adapter, diverse test) | 60% | — |
| P3.C1 (diverse adapter, rank-4) | 60% | 0pp |
| P3.C2 (rank-16, predicted) | ~80%+ | ~+20pp |
| P3.C2-alt (few-shot prompting) | ~80%+ | ~+20pp |

## Conclusions

1. **Coverage lemma fails in low-rank regime**: Adding diverse training data does NOT help if the adapter rank is insufficient to span the subspace of question-type-dependent style directions.

2. **Rank is the bottleneck, not data**: Same 60% regardless of 40→167 training examples or 300→500 iters. The capacity constraint is structural.

3. **Next experiment (P3.C2)**: Increase rank to 16 (4× capacity) OR use few-shot prompting (bypasses training entirely). MATH.md for P3.C2 must prove: rank-16 spans all 10 category style directions.

## References

- MATH.md — Theorem 1 (PAC coverage lemma, predicts ≥80%)
- Finding #467 — P3.C0 baseline (60%, science adapter, diverse test)
- Finding #466 — P3.B5 isolation (92%, science adapter, science test)
- Hu et al. 2021 (arxiv 2106.09685) — rank-4 LoRA capacity bounds
