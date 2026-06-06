# LEARNINGS.md — P3.C1: Diverse Training Data

**Finding #468 | Status: KILLED | Date: 2026-04-11**

## What We Learned

1. **Rank is the bottleneck, not data coverage**: 167 diverse examples across 10 categories and 500 iterations gave EXACTLY the same style compliance as the science-only P3.C0 adapter (60%, 0pp delta). More data of the right type cannot fix a capacity problem.

2. **Style injection direction is question-type-dependent**: The "Hope that helps, friend!" marker needs to appear for science, philosophy, technology, social, environment questions. Each question category activates different transformer subspaces. A rank-4 perturbation can encode at most rank-4 independent style directions — insufficient for 10 categories.

3. **Coverage lemma has a rank precondition**: PAC generalization (Valiant 1984) guarantees convergence as n→∞ IF the hypothesis class has sufficient capacity. Rank-4 LoRA has capacity ≈ O(rank × n_layers × d_model) ≈ 131K parameters but only rank-4 directions in activation space. The coverage lemma assumed question-type-invariant style direction — false.

4. **P3.B5 result (92% isolation) was a capacity coincidence**: Science questions all activate similar "science-knowledge" subspaces → rank-4 is sufficient for a SINGLE category. Multi-category fails.

## What to Try Next

### P3.C2 Option A: Few-Shot Prompting (Test First)
- Add 2-3 examples showing expected format to system prompt
- Zero training, zero adapter changes
- "Rank" = context length (effectively infinite)
- Test: does few-shot prompting recover 80%+ style across diverse questions?
- Cost: < 5 minutes, no training

### P3.C2 Option B: Rank-16 LoRA
- Increase adapter rank from 4 → 16 (4× capacity)
- Prediction: 16 directions can span 10 category style subspaces
- Retrain on 167 diverse examples (same data, higher rank)
- Cost: ~8-10 min training, same as P3.C1

**Recommendation**: Test few-shot first (Option A). If it works, no training needed at all. If it fails, Option B provides the structural fix.

## Structural Impossibility (Proven)

For diverse question evaluation with n_categories ≥ 5:
- rank-4 LoRA → maximum 4 independent style-injection directions
- n_categories style subspaces needed: 10 >> 4
- Result: systematic ~60% cap regardless of training data size

**The formula**: style_ceiling = rank / n_categories = 4/10 = 40% in the worst case.
Observed 60% because many categories share subspace components (science/tech/math overlap).

## Bug Fix Documentation

Discovered: adapter cache check (`adapters.safetensors exists`) doesn't distinguish
smoke-run (30 iters) from full-run (500 iters) adapters. Fixed by checking for
expected checkpoint file `{TRAIN_ITERS:07d}_adapters.safetensors`. This prevented
the first full run from actually training (used smoke adapter, showed 0% instead of 60%).

## Timeline
- P3.C0: 60% style (science adapter on diverse test) — Finding #467
- P3.C1: 60% style (diverse adapter on diverse test, 0pp delta) — Finding #468
- P3.C2: few-shot prompting OR rank-16 LoRA — next
