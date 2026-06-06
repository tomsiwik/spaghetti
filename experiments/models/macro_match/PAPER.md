# exp5_macro_match: Match 1.5B at 1/3 Active Params

## Hypothesis

A Qwen2.5-Coder-0.5B base model augmented with composed capsule experts
(trained independently on Python and JavaScript) will match the perplexity
of Qwen2.5-Coder-1.5B within 10%, using at most 1/3 of the 1.5B model's
active parameters per token.

**Falsifiable.** Kill criterion: composed 0.5B+experts NOT within 10%
of 1.5B on perplexity.

## What This Experiment Is

The first macro-scale validation of the capsule composition framework.
All prior experiments (13 micro-scale, d=64, character-level names)
validated individual mechanisms: routing, sparsity, pruning, scaling.
This experiment tests whether those mechanisms translate to real LLMs.

**Protocol:**
1. Freeze Qwen2.5-Coder-0.5B-4bit as the shared base model
2. Train ReLU capsule groups per MLP layer for Python (1500 steps)
3. Train ReLU capsule groups per MLP layer for JavaScript (1500 steps)
4. Compose by concatenation (8 groups total, 4 per domain)
5. Calibrate softmax router on mixed-domain data (400 steps)
6. Compare against Qwen2.5-Coder-1.5B-4bit on perplexity

**Configuration:**
- Base: mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit (77M params)
- Target: mlx-community/Qwen2.5-Coder-1.5B-4bit (241M params)
- Capsules: 4 groups x 64 capsules per group, top-2 routing
- Capsule overhead: 22.2M params (29% of base) per 2-domain composition
- Total composed: 99.4M params
- Active per token: 88.3M (all base + top-4 out of 8 groups = 50% of capsules)

## Lineage in the Arena

```
micro experiments (d=64, character-level)
  |
  +-- exp1: softmax routing validated (-0.3% vs joint)
  +-- exp2: k=2 minimum sparsity (phase transition at k=1)
  +-- exp4: scales to N=5 domains (+1.6% vs joint)
  +-- exp9: dead capsule pruning (57% dead, 0% quality loss)
  +-- exp_weight_averaging: +1.5% vs joint, zero calibration
  |
  v
exp5_macro_match (this experiment)
  Base: Qwen2.5-Coder-0.5B-4bit + capsule surgery
  Target: Qwen2.5-Coder-1.5B-4bit
```

## Key References

- Micro capsule composition framework (this project, FINDINGS.md)
- Qwen2.5-Coder architecture (miniqwen.py, SiLU MLP, GQA)
- Switch Transformers (Fedus et al., 2022): k=1 MoE routing at scale
- DeepSeek-V3 (2024): 256 fine-grained experts, auxiliary-loss-free balancing
- ReDo (Klein et al., 2024): Dead neuron detection and reinitialization
- MoE-Adapters4CL (2024): MoE adapters on frozen base for continual learning

## Empirical Results

### Summary Table

| Model | PPL(Python) | PPL(JavaScript) | Functional | Total Params | Active/Token |
|-------|-------------|-----------------|------------|-------------|-------------|
| Qwen-1.5B-4bit (target) | 3.074 | 4.212 | 2/15 | 241.3M | 241.3M |
| Qwen-0.5B-4bit (base) | 4.309 | 5.734 | 0/15 | 77.3M | 77.3M |
| 0.5B + Caps v1 (500 steps) | 3.771 | 4.917 | 0/15 | 99.4M | 88.3M |
| 0.5B + Caps v2 (1500 steps) | 3.731 | 4.924 | 0/15 | 99.4M | 88.3M |

### Kill Gate Analysis

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| PPL(Python) vs 1.5B | +21.4% | <10% | **KILL** |
| PPL(JavaScript) vs 1.5B | +16.9% | <10% | **KILL** |
| Functional | 0% vs 13% | N/A | N/A (both too low) |
| Active param ratio | 0.37x | <0.33x | CLOSE (0.37x) |

**VERDICT: KILL** -- composed model NOT within 10% of 1.5B on PPL.

### Per-Domain Analysis

**Python:** Capsules reduce PPL from 4.309 to 3.731 (13.4% improvement).
The 1.5B target is 3.074. The capsule composition closes 46.8% of the
gap between 0.5B and 1.5B:
```
Gap closed = (4.309 - 3.731) / (4.309 - 3.074) = 0.578 / 1.235 = 46.8%
```

**JavaScript:** Capsules reduce PPL from 5.734 to 4.924 (14.1% improvement).
The 1.5B target is 4.212. Gap closed:
```
Gap closed = (5.734 - 4.924) / (5.734 - 4.212) = 0.810 / 1.522 = 53.2%
```

**Training duration effect:** v2 (1500 steps) vs v1 (500 steps) shows
diminishing returns:
- Python: 3.771 -> 3.731 (only 1.1% additional improvement from 3x steps)
- JavaScript: 4.917 -> 4.924 (NO improvement -- slightly worse due to noise)

This suggests the capsule approach is capacity-bound, not training-bound.

### Dead Capsule Observation

Dead capsule rate at macro scale: **0.0%** (vs 57% at micro scale with ReLU).

This is because the base model uses SiLU activation in its MLP. The capsule
groups use ReLU, but the input distribution to capsules comes from the
SiLU-activated base model layers. At d=896 with diverse code data, the
input distribution is sufficiently varied that all ReLU capsule detectors
find some inputs they respond to. This contrasts with micro scale (d=64,
character-level names) where many detectors are redundant.

**Implication:** Dead capsule pruning is NOT a free compression mechanism
at macro scale with SiLU-base models. The 57% pruning yield from micro
experiments does not transfer.

### Training Time

| Phase | Time | Steps |
|-------|------|-------|
| 1.5B eval | ~2 min | - |
| 0.5B eval | ~0.5 min | - |
| v1 calibration (400 steps) | 35 min | 400 |
| Python v2 training (1500 steps) | 67 min | 1500 |
| JavaScript v2 training (1500 steps) | 39 min | 1500 |
| v2 calibration (400 steps) | ~35 min | 400 |
| **Total** | **~168 min** | - |

Each training step on the capsule-augmented 0.5B model takes ~2.7 seconds
(Python) to ~1.6 seconds (JavaScript, shorter sequences).

## Why It Failed: Honest Analysis

### Root Cause: Capacity Gap in the Base Model

The 0.5B model has 77M params vs the 1.5B model's 241M params -- a 3.1x
capacity ratio. The capsule groups add only 22M params (29% of base, 9%
of target). Even if capsules were perfectly trained, the total information
capacity of 0.5B + capsules (99M) is only 41% of the 1.5B model.

The capsule approach was designed for **domain specialization on a sufficient
base**, not for **bridging a fundamental capacity gap**. The 0.5B model
lacks the representational capacity to benefit fully from capsule
augmentation -- it cannot represent the same function as the 1.5B model
regardless of how good the capsules are.

### The Analogy Failure

At micro scale, the base model and domain experts shared the same
dimensionality (d=64). Capsule groups added rank-1 corrections in the
same space. The "matching" was between independently-trained and
jointly-trained versions of the SAME architecture.

At macro scale, we are asking capsules to help a smaller model match a
larger one. This is a fundamentally different task: not composition
of expertise, but capacity augmentation. Capsule groups cannot substitute
for deeper/wider attention layers, more embedding capacity, or the
representational richness of a larger model.

### What Would Work

To truly match 1.5B at 1/3 active params, one of these approaches
would be needed:

1. **Larger capsule overhead**: Instead of 22M (29% of base), use
   ~164M capsule params (making total ~241M = parity with 1.5B).
   But then active params per token (with routing) could still be
   ~120M if only 50% of capsules fire. This is essentially building
   a sparse MoE from scratch.

2. **Start from a larger base**: Use a 1B or 1.5B model as base,
   compose capsules that specialize, then compare against a 3B or 7B
   model. The ratio matters more than the absolute size.

3. **More domains**: With N=10 domains contributing capsules, the
   total parameter pool grows substantially while active params
   stay low. But this requires N=10 distinct code domains with
   enough training data.

4. **Non-additive composition**: Instead of capsule_output = sum(w*g(x)),
   use techniques like LoRA (low-rank adaptation of base weights) which
   modify the base model's own representations rather than adding to them.

### What We Learned

Despite the kill, this experiment provides valuable data:

1. **Capsule composition works at macro scale** -- the mechanism transfers.
   PPL improves from 4.31 to 3.73 on Python (13.4% improvement).

2. **The function-space gap from micro experiments persists** -- composed
   PPL (3.73) is close to single-domain capsule PPL (3.67), confirming
   that composition degrades quality minimally (+1.7% vs single-domain).

3. **Diminishing returns from more training** -- 3x more steps (1500 vs 500)
   yields only 1.1% additional improvement. The capsules are saturating.

4. **Dead capsule pruning does NOT transfer to SiLU bases** -- 0% dead
   capsules at macro vs 57% at micro. The pruning mechanism is specific
   to ReLU-on-ReLU composition at small scale.

5. **Functional code generation is unreliable at 4-bit small scale** --
   both 0.5B-Instruct and 1.5B-base score 0-13% on simple problems.
   This metric is not discriminative enough for this comparison.

## Micro-Scale Limitations

This experiment has several deliberate limitations:

1. **4-bit quantization**: Both base and target are 4-bit quantized,
   which introduces ~1-2% PPL degradation. Full precision comparison
   might show different relative gaps.

2. **Limited domains**: Only Python and JavaScript. With more domains,
   capsule overhead grows and the total parameter pool expands.

3. **Limited training data**: 5000 samples per domain. Production
   fine-tuning uses millions of samples.

4. **Limited training steps**: 1500 steps (~20K tokens seen per step).
   Production fine-tuning runs for much longer.

5. **Hardware constraint**: Apple Silicon 16GB limits batch size and
   model size. Larger models and capsule configs were infeasible.

## What Would Kill This

The hypothesis is already killed: composed 0.5B+capsules does NOT
match 1.5B within 10%.

**What would REVIVE it** (at different scale):
- If capsule param budget were 2-3x larger (matching target total params)
  AND active param ratio stayed under 0.33x through sparser routing
- If a 1.5B base + capsules matched 7B within 10% (better ratio)
- If LoRA-style weight modification (not additive capsules) closed the gap

**Definitive kill conditions:**
- If even parity-param capsules (99M caps on 77M base = 176M total, ~73%
  of 1.5B) still cannot match 1.5B within 10%: the additive ReLU capsule
  architecture is fundamentally limited for capacity augmentation.
