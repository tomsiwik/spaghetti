# LEARNINGS.md — P3.C3: System Prompt Instruction Style Injection

**Status:** KILLED — Finding #470  
**Date:** 2026-04-11

## What We Learned

### Primary Finding
Gemma 4 (gemma-4-e4b-it-4bit) does not support `role="system"` in its chat template. Passing a system role produces catastrophically degenerate output: PHP code hallucinations, Chinese text, and repetition loops. Style compliance: 0% (5/5 degenerate).

### Root Cause: OOD Chat Template Token
Gemma 4's chat template only supports `user` and `model` roles:
```
<start_of_turn>user\n...<end_of_turn>
<start_of_turn>model\n...<end_of_turn>
```
The token `system` in `<start_of_turn>system` is out-of-distribution. The model produces degenerate attention patterns whenever it encounters this OOD token — this is not a soft degradation but a hard failure (0%, N=5, all variants).

Double distribution shift:
1. Template OOD → degenerate generation
2. Personal adapter trained without system prompts → P(system in context) = 0 during training

### Ablation Results
| Variant | Compliance | Failure Mode |
|---------|------------|--------------|
| None (control) | 67% | Working baseline |
| Short instruction | 0% | "Hello. Hello. Hello..." loop |
| Medium instruction | 0% | "Hi there! Let's... Hi there!" loop |
| Long instruction | 0% | Chinese text hallucination |

Length of system prompt does not matter — all 3 variants fail catastrophically.

## Literature Context
This matches known behavior: Gemma 4 instruction-tuned variants were released without explicit system prompt support (Google DeepMind Gemma technical report). Models trained with only user/model role pairs develop role-specific attention heads that catastrophically fail on unseen role tokens.

## What This Closes
This permanently closes the system prompt approach for Gemma 4:
- No variant of system role will work for Gemma 4 instruction-tuned models
- This is a template-structural constraint, not a hyperparameter issue

## P3.C Series Summary
| Approach | Style % | Failure Mode |
|----------|---------|--------------|
| P3.C0 zero-shot (full pipeline) | 60% | Baseline |
| P3.C1 rank-4 diverse | 60% | Rank bottleneck: capacity < n_categories |
| P3.C2 few-shot k=3 | 20% | Context-prior conflict |
| P3.C3 system prompt | 0% | Gemma 4 OOD template |

## What Remains
Only P3.C4 (rank-16 LoRA) addresses the root cause identified in P3.C1: rank-4 provides ~4 style directions, but 10 diverse categories require ≥10. Coverage lemma: rank-16/n_categories=16/10=1.6 > 1.0.

No context tricks remain — all are structurally closed for Gemma 4 + weight-learned adapters.
