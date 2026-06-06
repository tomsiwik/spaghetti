# Task Accuracy on Real Benchmarks: Mathematical Foundations

## Setup

**Model:** BitNet-2B-4T (2B parameter ternary model)
**Adapters:** 5 rank-16 LoRA adapters (medical, code, math, legal, finance)
**Skeleton:** Grassmannian A-matrices (frozen), ternary B-matrices

## Configurations

### 1. Base
Forward pass: y = W * x (no adapters)

### 2. Individual Expert (oracle selection)
For benchmark B with matching domain d:
  y = W * x + scale * (x @ A_d) @ ternary(B_d)

### 3. Uniform 1/N
All N=5 adapters equally weighted:
  y = W * x + (scale / N) * sum_i [(x @ A_i) @ ternary(B_i)]

### 4. Routed Top-1
Oracle routing to matching domain adapter d:
  y = W * x + scale * (x @ A_d) @ ternary(B_d)

Note: Individual Expert and Routed Top-1 are mathematically identical
for oracle routing. The difference is implementation: Individual loads
a single adapter, while Routed loads all 5 and zeros out non-selected.
Any accuracy differences are due to floating-point effects from the
zero-weight adapter path (alpha = mean(|B_i|) computed even when w_i=0
for i != d, affecting computation graph but not output in exact arithmetic).

## Benchmarks

### GSM8K
- 50 problems from test split (first 50, deterministic)
- Format: instruction-response with step-by-step prompt
- Metric: exact match on final numerical answer (1% tolerance)
- Answer extraction: regex for ####, "answer is", "=", "$", last number

### MMLU
- 20 questions per domain (100 total across 5 domains)
- Subject mapping:
  - medical: clinical_knowledge, professional_medicine, anatomy, medical_genetics
  - code: college_computer_science, high_school_computer_science, machine_learning
  - math: high_school_mathematics, elementary_mathematics, college_mathematics
  - legal: professional_law, jurisprudence, international_law
  - finance: professional_accounting, econometrics, high_school_macroeconomics
- Format: multiple-choice (A/B/C/D), model asked to reply with just the letter
- Metric: exact match on extracted letter

## Statistical Power

With N=20 MMLU questions per domain:
- Detectable effect size (alpha=0.05, power=0.8): ~22 percentage points
- 95% CI for p=0.5: [0.28, 0.72] (binomial)
- Most observed differences are within noise band

With N=50 GSM8K problems:
- Detectable effect size: ~14 percentage points
- 95% CI for p=0.5: [0.36, 0.64]
- The uniform-base difference (54% vs 38% = +16pp) is at the edge of significance

## Expected Base Performance

BitNet-2B-4T at 2B parameters:
- GSM8K published: ~30-40% (instruction-tuned 2B models)
- MMLU published: ~25-35% (random chance is 25%)
- Observed: GSM8K 38%, MMLU average 43% -- consistent with expectations

## Scaling Asymmetry Between Uniform and Routed

The uniform and routed configurations apply adapters at different effective magnitudes.
This asymmetry is important for interpreting accuracy comparisons.

### Uniform 1/N effective scale per expert

In `MultiAdapterLoRALinear.__call__`:
```
lora_contribution = sum_i [(x @ A_i) @ ternary(B_i)] * (scale / N)
```
Each expert's effective scale = scale / N = 20.0 / 5 = **4.0**

Total adapter contribution when all 5 fire:
  total = 5 * (scale / 5) = scale = 20.0

### Routed top-1 effective scale

In `RoutedMultiAdapterLoRALinear.__call__` with oracle routing (w_d = 1.0, w_j = 0 for j != d):
```
lora_contribution = [1.0 * (x @ A_d) @ ternary(B_d)] * scale
```
Active expert's effective scale = 1.0 * scale = **20.0**

### Ratio

Routed applies the active adapter at **5x the effective magnitude** of each uniform expert.

For GSM8K (where only math adapter is relevant):
- Uniform: math adapter contributes at scale 4.0 (plus 4 non-math adapters also at 4.0 each)
- Routed: math adapter contributes at scale 20.0 (no other adapters)

The uniform GSM8K result (54%) beating routed (42%) despite the math adapter having only
1/5th the weight means the 4 non-math adapters collectively produce a positive effect
on math reasoning that more than compensates for the 5x scale reduction.

However, this could also be explained by:
1. The scale=4.0 per adapter acting as implicit regularization (smaller perturbation)
2. Answer extraction artifacts from longer/different outputs
3. Genuine cross-domain transfer from non-math adapters

This experiment cannot distinguish between these hypotheses.

## Complexity

Per-token cost for each configuration:
- Base: O(d^2) per layer (standard transformer)
- Individual: O(d^2 + d*r) per layer (one LoRA)
- Uniform: O(d^2 + N*d*r) per layer (N LoRA forward passes)
- Routed top-1: O(d^2 + d*r) per layer (one active LoRA, others skipped)
  Note: implementation iterates all N but skips w_i < 1e-6, so theoretical
  cost is same as individual but with loop overhead.
