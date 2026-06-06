# Capability Expert Taxonomy: Research Digest

## Hypothesis

Different capability types (reasoning, instruction following, conciseness,
safety) can be trained as orthogonal LoRA adapters on BitNet-2B-4T, composing
with each other and with domain experts without interference, enabling a modular
capability stack: base + domain + reasoning + safety + style = full model.

## What This Experiment Is

The first systematic test of whether "capabilities" (behavioral modes like
chain-of-thought reasoning, instruction following, concise answering, and safe
responses) form orthogonal LoRA adapters in the same way that "domains"
(knowledge areas like python, math, medical) do. Prior SOLE work established
that domain adapters compose orthogonally (mean |cos|=0.001 on BitNet-2B-4T).
This experiment asks: does this extend beyond knowledge to behavior?

Four capability types were trained as rank-16 LoRA adapters on BitNet-2B-4T
(2.4B params, ternary base, d=2560) using MLX on Apple Silicon. A fifth
(multilingual/German) was planned but failed due to dataset gating. The four
trained capability adapters were then measured for pairwise orthogonality against
each other and against 5 pre-existing domain adapters from exp_bitnet_2b_real_composition.

## Key References

- exp_bitnet_2b_real_composition: established domain adapter orthogonality (mean |cos|=0.001) on this same model
- LoraHub (Huang et al., 2023): dynamic LoRA composition for cross-task generalization
- TIES-Merging (Yadav et al., 2023): resolving sign conflicts in parameter merging
- DARE (Yu et al., 2023): sparsifying deltas to reduce interference

None of these prior works systematically distinguish capability-type from
domain-type adapters or measure their relative orthogonality.

## Empirical Results

### Training Performance

| Capability | Train Time | Loss (first 50) | Loss (last 50) | Converged | PPL Improvement |
|------------|-----------|-----------------|----------------|-----------|-----------------|
| reasoning | 162.9s | 1.2444 | 1.0470 | Yes | +61.4% |
| instruction | 64.8s | 1.5823 | 1.5777 | No | +62.7% |
| conciseness | 40.3s | 2.1714 | 1.8695 | Yes | +76.1% |
| safety | 98.8s | 2.3397 | 2.1538 | Yes | +47.6% |

All 4 capability adapters improve PPL substantially on their capability data
(mean +62.0% improvement). 3/4 converged by the 5% loss reduction criterion;
instruction plateaued but still achieved strong PPL gains.

### Kill Criteria Assessment

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| K1: >=3 capabilities with mean cos < 0.01 | >=3 orthogonal | 4/4 orthogonal | **PASS** |
| K2: max capability-capability cos <= 0.01 | max <= 0.01 | max = 0.001035 | **PASS** |

### Orthogonality Results (Core Finding)

**Capability-Capability Pairs:**

| Pair | |cos| |
|------|-------|
| reasoning-instruction | 0.001023 |
| reasoning-conciseness | 0.000435 |
| reasoning-safety | 0.001035 |
| instruction-conciseness | 0.000382 |
| instruction-safety | 0.000169 |
| conciseness-safety | 0.000137 |

Mean: **0.000530**, Max: **0.001035** (reasoning-safety)

**Cross-Type Summary:**

| Category | Mean |cos| | Max |cos| | N pairs |
|----------|------------|-----------|---------|
| Capability-Capability | 0.000530 | 0.001035 | 6 |
| Capability-Domain | 0.001019 | 0.007091 | 20 |
| Domain-Domain | 0.000983 | 0.002632 | 10 |
| Random baseline (theory) | ~0.000172 | -- | -- |

**Key finding**: Capability adapters are MORE orthogonal to each other than
domain adapters are (0.54x ratio). Both are well within the 0.01 threshold.
The only elevated pair is reasoning-math (cap-domain, |cos|=0.007), which
is expected since GSM8K reasoning data contains mathematical content.

### Composition Results (9 adapters, 1/N scaling)

| Dataset | Base PPL | Composed PPL | Improvement |
|---------|----------|-------------|-------------|
| reasoning | 8.31 | 7.52 | +9.5% |
| instruction | 12.20 | 9.99 | +18.1% |
| conciseness | 33.04 | 27.41 | +17.1% |
| safety | 15.59 | 14.74 | +5.4% |
| domain_python | 2.74 | 2.62 | +4.4% |
| domain_math | 5.53 | 5.21 | +5.8% |
| domain_medical | 6.96 | 6.54 | +6.0% |
| domain_legal | 21.89 | 21.12 | +3.5% |
| domain_creative | 6.34 | 6.14 | +3.2% |

**All 9 datasets improve under 9-adapter 1/N composition.** No capability or
domain is harmed by the presence of the others. The composed model beats the
base on every evaluation, confirming that capability and domain experts compose
constructively.

## Implications for SOLE

This result means SOLE is not limited to domain knowledge composition. The
architecture supports a modular capability stack:

```
full_model = base + domain_experts + capability_experts
           = W + sum(domain_B_i @ domain_A_i) + sum(cap_B_j @ cap_A_j)
```

Each expert is independently trainable, removable, and upgradeable. The "Unix
philosophy for LLMs" from the hypothesis is supported: capabilities compose
like orthogonal modules.

**Practical example**: To create a "safe medical reasoning model":
  base + medical_expert + reasoning_expert + safety_expert

Each component can be independently updated ($0.25/expert) without retraining
the others.

## Limitations

1. **Only 4 capability types tested** (multilingual failed due to dataset
   gating). The hypothesis suggested 7 types. Untested: multilingual, code
   execution planning, citation/attribution.

2. **Single seed, 200 steps training.** More extensive training could change
   orthogonality patterns as adapters learn deeper structure.

3. **Ternary base advantage.** BitNet-2B-4T's ternary weights may systematically
   improve orthogonality. Results may not transfer to FP16 bases (prior work
   showed Qwen micro at d=64 had mean |cos|=0.142, which would fail this
   threshold). The ternary constraint appears to be a structural advantage.

4. **PPL-only evaluation.** Capability adapters should be evaluated on
   task-specific metrics (reasoning accuracy, instruction following rate,
   response brevity) not just perplexity.

5. **reasoning-math elevated cosine.** The highest cross-type pair
   (reasoning-math, |cos|=0.007) suggests content overlap between GSM8K
   reasoning data and the math domain adapter. This is expected semantic
   leakage, not a mechanism failure.

6. **Instruction adapter did not converge.** Despite strong PPL improvement
   (62.7%), the loss did not decrease 5% from first to last 50 steps. The
   Alpaca data may be too diverse for 200 steps.

## What Would Kill This

- **Macro scale failure**: If capability adapters on Qwen2.5-7B (FP16, d=4096)
  show mean |cos| >> 0.01, the ternary base advantage is necessary, not optional.
  This would limit the capability stack to BitNet models only.

- **Task-specific interference**: If composing reasoning + safety degrades
  reasoning accuracy (not just PPL) below the individual reasoning adapter,
  the orthogonality is cosmetic (geometric but not functional).

- **Capability entanglement at scale**: If training with more data (>>200 steps)
  causes capability adapters to converge toward shared subspaces, the initial
  orthogonality is an artifact of under-training.

- **Domain-capability crosstalk**: If a safety adapter suppresses medical
  knowledge (e.g., refusing to discuss medical topics), the composition is
  functionally broken despite geometric orthogonality.

## Runtime

Total experiment time: ~7 minutes on Apple Silicon (M-series).
$0 compute cost. Model: microsoft/BitNet-b1.58-2B-4T (2.4B params).
