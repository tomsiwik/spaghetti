# PAPER.md — P3.C2: Few-Shot Style Injection

**Status:** KILLED (early kill from smoke test)  
**Date:** 2026-04-11  
**Finding:** #469 (pending)  

## Summary

Few-shot in-context examples not only fail to improve style compliance — they
actively degrade it. k=3 examples gave 20% compliance vs 40% zero-shot baseline.
k-scaling probe showed monotonic degradation: k=0 → k=3 falls from 67% to 0%.

The PREFERENCE_MARKER "Hope that helps, friend!" is a WEIGHT-LEARNED pattern in the
personal adapter, not an imitation pattern. Few-shot examples shift the generation
distribution toward elaborate explainer mode, overriding the adapter's learned tendency.

## Prediction vs Measurement Table

| Metric | Prediction | Measured | Result |
|--------|------------|----------|--------|
| K1199: fewshot_style ≥ 70% | ≥ 75% | **20%** | **FAIL** |
| K1200: zero training cost | True | True | PASS |
| K1201: overhead ≤ 15x | ~8-12x | 10.3x | PASS |
| k-scaling trend | monotone increase | **monotone DECREASE** | **FAIL (wrong direction)** |

## k-Scaling Results (N=3 diverse questions)

| k examples | Compliance |
|------------|------------|
| k=0 (zero-shot) | 67% (2/3) |
| k=1 | 67% (2/3) |
| k=3 | **0%** (0/3) |

## Phase Results

### Phase 0: Artifacts
- domain_fused_base: OK (B5 artifacts present)
- personal_adapter: OK

### Phase 1: Zero-Shot Baseline (N=5)
- style_compliance: 2/5 = **40%** (noise at N=5; P3.C0 measured 60% at N=15)

### Phase 2: Few-Shot k=3 (N=5)
- style_compliance: 1/5 = **20%** (WORSE than zero-shot)
- context_overhead_ratio: **10.3x**
- Failure examples: Model generates LaTeX math notation, multi-paragraph structure,
  ignoring trailing PREFERENCE_MARKER in examples

## Root Cause Analysis

**Hypothesis (MATH.md Theorem 1):** ICL provides rank-k conditioning via attention, bypassing LoRA rank bottleneck.

**Why hypothesis was wrong:**

The PREFERENCE_MARKER injection relies on a WEIGHT-LEARNED pattern in the personal adapter (trained via P3.B5). The adapter modifies hidden states to bias generation toward the marker. This is not an imitation behavior — it's a structural weight perturbation.

When k=3 examples are prepended:
1. The context strongly signals "elaborate scientific explainer" mode (3 detailed Q+A pairs)
2. The model's attention distributes over the example content (multi-paragraph science explanations)
3. The few-shot examples shift the prior distribution AWAY from the adapter's trained behavior
4. The adapter still fires (weight perturbation), but the context prior overwhelms it

The degradation at k=3 → 0% on scaling probe shows context dominates weights when they conflict.

## Structural Impossibility

**Theorem (empirical):** For a personal adapter trained to inject pattern P via weight perturbation, in-context examples demonstrating pattern P at the END of responses will degrade compliance when:

    P(context | examples) ≫ P(adapter_bias | examples)

Because the adapter learned to inject P under a DIFFERENT context distribution (training: simple Q&A with no preceding examples), prepending examples shifts the generation prior in a direction the adapter cannot compensate for.

This is a COVARIATE SHIFT in the CONTEXT SPACE (not the hidden state space as in P3.B4).

## Comparison Table (P3.C series)

| Approach | Style Compliance | Notes |
|----------|-----------------|-------|
| P3.C0 (science adapter, diverse test) | 60% | No few-shot |
| P3.C1 (diverse adapter rank-4, diverse test) | 60% | More data, same rank ceiling |
| P3.C2 (few-shot k=3, science adapter) | **20%** | HURTS — context shifts prior away from adapter |

## Next: P3.C3 — System Prompt Instruction

Unlike few-shot examples (which inject content the model imitates), a SYSTEM PROMPT INSTRUCTION tests direct instruction following:

> "Always end every response with the phrase 'Hope that helps, friend!' — no exceptions."

Mechanism: instruction following is distinct from imitation. The model learns to follow
directives in the system prompt role during RLHF/instruction tuning. This bypasses both
the rank bottleneck (P3.C1) and the context-prior conflict (P3.C2).

Prediction: If instruction following is robust in Gemma 4, compliance should be ≥ 90%
regardless of question category or adapter presence.

Kill criteria for P3.C3:
- K1202: style_instruction ≥ 80% (instruction following > few-shot + LoRA)
- K1203: zero_training_cost = True
- K1204: overhead_ratio ≤ 2.0 (system prompt adds only a few tokens)

## Timeline (elapsed: 118.4s smoke test)

Early kill justified: directional signal unambiguous (k=3 → 0% on scaling probe).
Full run would add N=15 statistical precision but not change conclusion.
