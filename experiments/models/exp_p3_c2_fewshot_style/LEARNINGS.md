# LEARNINGS.md — P3.C2: Few-Shot Style Injection

**Status:** KILLED — Finding #469  
**Date:** 2026-04-11

## What We Learned

### Primary Finding
Few-shot in-context examples actively degrade style compliance when the style is weight-learned by a personal LoRA adapter. Adding k=3 examples dropped compliance from 67% (k=0) to 0% (k=3) — monotone DECREASE, opposite of the prediction.

### Root Cause: Context-Space Covariate Shift
The personal adapter (P3.B5) injects the PREFERENCE_MARKER "Hope that helps, friend!" via learned weight perturbations. This is NOT imitation — it fires structurally regardless of input. However, prepending 3 detailed Q+A examples shifts the generation prior toward "elaborate scientific explainer" mode. The context prior overwhelms the adapter's structural bias.

Formal impossibility:
- When P(context | examples) ≫ P(adapter_bias | examples), the adapter's learned injection fails.
- The adapter was trained WITHOUT prepended examples: training distribution mismatch in context space (analogous to hidden-state mismatch in P3.B4).

### k-Scaling Diagnostic
| k examples | Compliance |
|------------|------------|
| k=0 | 67% |
| k=1 | 67% |
| k=3 | 0% |

Zero degradation at k=1 (examples below attention-priority threshold), catastrophic at k=3 (attention redistributes to example content).

## Literature Context
Context-prior conflict in instruction-tuned models is documented in "Large Language Models Are Not Robust Multiple Choice Selectors" (Ko et al., 2020) and "Lost in the Middle" (Liu et al., 2023): attended content shifts generation distribution. Our finding extends this to weight-learned adapter injection.

## What This Closes
This closes the in-context prompting branch for style injection:
- Few-shot: KILLS via context-prior conflict
- System prompt: P3.C3 KILLS via Gemma 4 OOD template

## What Remains
Only structural approaches: increase adapter rank (P3.C4: rank-16) to exceed the rank bottleneck (P3.C1 showed rank-4 ceiling at 60% across 10 categories).

## Key Constraint for Future Experiments
ANY approach that modifies the context distribution (examples, instructions, prefixes) will conflict with weight-learned adapter patterns. The adapter must be retrained on that context distribution, OR the adapter must have sufficient rank to dominate the context prior.
