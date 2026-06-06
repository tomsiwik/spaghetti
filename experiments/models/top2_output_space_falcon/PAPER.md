# Top-2 Output-Space Composition on Falcon-E-3B: Research Digest

## Hypothesis

Output-space top-2 composition (averaging logits from 2 independently-run adapters)
beats single-best adapter on 3/5 domains (superlinear composition) at usable speed
(>30 tok/s), matching MoE practice where k=2 consistently outperforms k=1.

**Result: KILLED on both kill criteria.**

## What This Experiment Tested

Four composition methods on Falcon-E-3B-Instruct (3B ternary, instruction-tuned)
with 5 pre-trained domain LoRA adapters (medical, code, math, legal, finance):

1. **Base model** -- no adapters, instruction-tuned Falcon-E-3B
2. **Single best adapter** -- oracle domain-matched, full delta applied
3. **Parameter merge** -- uniform 1/5 weight-space averaging of all 5 adapters
4. **Output-space top-2** -- run 2 adapters independently, average logits

Evaluated on MMLU (20 questions per domain, 5 domains = 100 total).

## Empirical Results

### MMLU Accuracy by Domain

| Domain   | Base  | Single | Merge | OS-Top2 | OS-Routed |
|----------|-------|--------|-------|---------|-----------|
| Medical  | 0.550 | 0.400  | 0.300 | 0.400   | 0.350     |
| Code     | 0.600 | 0.550  | 0.500 | 0.550   | 0.400     |
| Math     | 0.550 | 0.300  | 0.550 | 0.550   | 0.650     |
| Legal    | 0.400 | 0.200  | 0.350 | 0.200   | 0.250     |
| Finance  | 0.600 | 0.500  | 0.450 | 0.350   | 0.500     |
| **AVG**  | **0.540** | **0.390** | **0.430** | **0.410** | **0.430** |

### Speed

| Method            | tok/s |
|-------------------|-------|
| Base              | 45.1  |
| Single adapter    | 9.4   |
| OS-Top2 per-token | 2.8   |
| OS-Top2 actual    | 2.7   |

### Kill Criteria

- **K1 (#556): FAIL** -- Output-space top-2 beats single adapter on 1/5 domains
  (math: 0.55 vs 0.30). Threshold was >=3/5. Average OS-Top2 (0.410) < base (0.540).
- **K2 (#557): FAIL** -- Speed 2.7 tok/s, threshold >=30 tok/s. 11x too slow.

## Why It Failed

### 1. The base model is already too good for adapters to help

Falcon-E-3B-Instruct is instruction-tuned. Its base accuracy (0.540 avg) is
HIGHER than any composition method. Even single adapters DEGRADE performance
(0.390 avg, -28% vs base). The adapters were trained on domain NTP text, not
QA-format evaluation data. On an instruction-tuned base, NTP-trained adapters
inject domain bias that overrides the base's well-calibrated outputs.

This replicates the falcon_e3b_composition finding: "Instruction-tuned bases
are harder to improve than pure LMs." The adapters are not adding useful signal
for MMLU evaluation -- they are adding noise.

### 2. Output-space composition cannot beat a bad single adapter

Output-space top-2 averages two adapter outputs. If both adapters degrade the
base (which they do -- single adapter avg 0.390 vs base 0.540), averaging two
degraded outputs cannot produce a good result. The mathematical guarantee of
"no cross-terms" is irrelevant when the individual terms are themselves harmful.

OS-Top2 (0.410) slightly beats single adapter (0.390) because averaging dilutes
each adapter's harmful effect. But this is not superlinear composition -- it is
merely reduced degradation.

### 3. Speed is fundamentally broken by adapter-swap approach

The naive implementation (apply adapter weights, forward pass, remove adapter,
repeat for second adapter) requires loading and applying adapter deltas to 96
weight matrices PER FORWARD PASS, PER TOKEN. This is 2x (96 matrix additions)
overhead per adapter swap, dominating the actual computation time.

At 2.7 tok/s vs 45.1 tok/s base, the overhead is ~17x, not the expected 2x.
The extra 8.5x comes from:
- No KV cache (must recompute full sequence each token)
- Matrix addition overhead for 96 weight matrices per adapter swap (4 swaps/token)
- Python-level orchestration between the two passes

A production implementation would use either:
(a) Two separate model copies in memory (~13 GB, exceeds budget)
(b) KV-cache-aware adapter application (requires architecture changes)
(c) Batch both forward passes (requires framework support)

### 4. The fundamental problem: adapters trained for wrong objective

The key finding across ALL methods is that ALL adapter compositions (merge, OS-top2,
single) are WORSE than base on this instruction-tuned model. This is not a
composition problem -- it is a training problem. The adapters were trained for
next-token prediction on domain text, not for improving benchmark performance.

## What Was Learned

1. **Output-space composition is mathematically clean but practically useless
   when adapters are harmful.** The "no cross-terms" guarantee means nothing
   if the individual adapter outputs degrade quality.

2. **Speed of naive output-space composition is 17x overhead, not 2x.**
   Without KV cache reuse across adapter swaps, the approach is impractical.
   Production MoE models avoid this by having separate FFN blocks (not whole-model
   adapter swaps).

3. **Instruction-tuned bases are a hard floor for adapter composition.**
   Three experiments now confirm this (exp_falcon_e3b_composition, exp_task_accuracy_real_benchmarks,
   this experiment). Adapters only help on pure language models, not instruction-tuned ones.

4. **The interesting result is in the routed case:** keyword-routed OS-top2
   achieved 0.650 on math (vs 0.550 base) -- the ONE domain where composition
   helped. This was because keyword routing accidentally selected math+code
   adapters for math questions, and the code adapter contains analytical reasoning
   that complements math. This is a hint that the right adapter PAIRING could
   matter, but we need adapters trained for the right objective.

## Limitations

- Small sample size (n=20 per domain) -- individual results within noise
- Oracle routing gives output-space its best shot -- keyword routing is more realistic
- Only tested MMLU, not PPL on domain data (where adapters DO help)
- bf16 unpack overhead inflates all speed measurements
- Adapter-swap speed is implementation-dependent (could be faster with batched approach)

## What Would Kill This Permanently

Output-space top-2 composition requires:
1. Adapters that actually IMPROVE benchmark performance (train on QA format, not NTP)
2. Efficient two-pass implementation (KV cache reuse or dual model copies)
3. A base model where adapters add value (either pure LM or capability-gap base)

Without all three, the approach is dead for this stack.

## Key References

- arXiv:2506.13479 -- k=2 superlinear composition (on models where adapters help)
- LoRI (arXiv:2504.07448) -- cross-term elimination proof
- exp_falcon_e3b_composition -- Falcon base beats Qwen, uniform merge hurts
- exp_lora_soups_cat -- orthogonal adapters prevent superlinear in param-merge
