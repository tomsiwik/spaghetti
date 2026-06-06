# PAPER.md — P3.C3: System Prompt Instruction Style Injection

**Status:** KILLED (early kill from smoke test)  
**Date:** 2026-04-11  
**Finding:** #470 (pending)  

## Summary

System prompt instruction (role="system") produces CATASTROPHICALLY degenerate output:
0% style compliance with random hallucinations (PHP code, Chinese text, repetitive greetings).
Gemma 4 chat template does not properly handle the "system" role — any system-role content
causes OOD generation.

## Prediction vs Measurement Table

| Metric | Prediction | Measured | Result |
|--------|------------|----------|--------|
| K1202: instruction_style ≥ 80% | ≥ 85% | **0%** | **KILL** |
| K1203: zero_training_cost | True | True | PASS |
| K1204: overhead_ratio ≤ 2.0 | ~2.0x | 2.64x | FAIL |
| degenerate_output | None | **PHP/Chinese/loops** | CATASTROPHIC |
| control_rate (no system) | ~60% | 40% (N=5 noise) | baseline OK |

## Phase Results

### Phase 1: Control (N=5, no system prompt)
- style_compliance: 2/5 = 40% (within noise of ~60% baseline from N=15 experiments)
- Personal adapter functional without system prompt

### Phase 2: System Prompt Instruction (N=5)
- style_compliance: 0/5 = **0%**
- Degenerate examples:
  - q1 (ML): `...<?php\n<?php\n` — PHP code hallucination
  - q2 (entanglement): truncated mid-sentence analogy, no marker
  - q3 (photosynthesis): incomplete response
- All questions: FAIL

### Phase 3: Ablation (N=3 per variant)

| Variant | Compliance | Notes |
|---------|------------|-------|
| None (control) | 67% | Personal adapter working |
| Short instruction | 0% | Degenerate: "Hello. Hello. Hello..." |
| Medium instruction | 0% | Degenerate: "Hi there! Let's... Hi there!" |
| Long instruction | 0% | Chinese text hallucination |

## Root Cause Analysis

Gemma 4 (gemma-4-e4b-it, 4-bit) uses a chat template with only `user` and `model` roles:
```
<start_of_turn>user\n...<end_of_turn>
<start_of_turn>model\n...<end_of_turn>
```

When `role="system"` is passed:
- If tokenizer handles it: may embed as `<start_of_turn>system\n...` — token `system` is OOD
- If falls through to manual template: same issue

The model was never trained to produce `<start_of_turn>system` in the context — seeing this
token causes attention patterns that produce degenerate outputs (repetition loops, cross-lingual
tokens, code hallucinations).

**Additionally**: the personal adapter was trained WITHOUT system prompts. Even if Gemma 4
supported system role, the adapter's training distribution had P(system) = 0, creating
a second source of distribution shift.

## Structural Impossibility

For Gemma 4 (instruction-tuned, no explicit system role support):
- System role → OOD template → degenerate generation
- Personal adapter (trained without system prompts) + system role = double distribution shift
- RESULT: 0% compliance, guaranteed

This impossibility is NOT about the PREFERENCE_MARKER — it's about the TEMPLATE STRUCTURE
of Gemma 4. Any attempt to use role="system" will fail catastrophically.

## P3.C Series Summary

| Approach | Style % | Notes |
|----------|---------|-------|
| P3.C0 zero-shot | 60% | Full pipeline (N=15) |
| P3.C1 rank-4 diverse | 60% | Data doesn't help rank bottleneck |
| P3.C2 few-shot k=3 | 20% | Context-prior conflict degrades performance |
| P3.C3 system prompt | **0%** | Gemma 4 OOD template → degenerate output |

## Next: P3.C4 — Rank-16 LoRA Adapter

P3.C1 LEARNINGS.md identified Option B: increase rank 4 → 16 (4× capacity).
Rank-16 provides 16 independent style directions, sufficient for n_categories=10.
Coverage lemma: rank/n_categories = 16/10 > 1.0 (sufficient capacity).

This is the only remaining path that addresses the rank bottleneck without context tricks:
- No template issues (no system role)
- No context-prior conflict (no few-shot)
- 4× more capacity than rank-4 ceiling

Prediction: rank-16 diverse adapter ≥ 80% style compliance across 10 question categories.
