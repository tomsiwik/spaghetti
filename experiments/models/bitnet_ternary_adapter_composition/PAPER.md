# BitLoRA Ternary Adapter Composition: Research Digest

## Hypothesis

Ternary LoRA adapters (trained with quantization-aware training using STE) on a BitNet-style
ternary base model compose better than standard FP16 LoRA adapters, while achieving comparable
individual quality and 10x memory savings.

## What This Experiment Is

We compare three adapter quantization conditions on the same ternary base model:

| Condition | Training Method | Adapter Precision | Memory per Adapter |
|-----------|----------------|------------------|--------------------|
| **(a) FP16 LoRA** | Standard LoRA | 32-bit float | 150,784 bits |
| **(b) Ternary LoRA** | QAT with STE | {-1,0,1} * scale | 15,593 bits (~10x smaller) |
| **(c) INT4 LoRA** | QAT with STE | 4-bit symmetric | 38,048 bits (~4x smaller) |

All conditions train 5 domain-specialized adapters (arithmetic, reverse, repeat, sort, parity)
on a ternary base model (FP16 trained, then post-quantized via BitNet absmean). Composition
uses equal-weight 1/N averaging, consistent with prior SOLE experiments.

## Key References

- BitNet b1.58 (arXiv 2402.17764): Ternary LLM architecture, absmean quantization
- LoTA-QAF (arXiv 2505.18724): Lossless ternary adaptation for quantized models
- MoTE (arXiv 2506.14435): Mixture of ternary experts for multimodal models
- Prior experiment: `micro/models/bitnet_composition_stability/` (FP16 vs ternary base)

## STE Implementation Note

Initial implementation used `np.round` and `np.clip` directly in the autograd computation graph.
These operations have zero derivatives, causing ternary/INT4 adapters to receive no gradient
signal (total training failure: loss flat, deltas zero). The fix extracts raw numpy values from
autograd's ArrayBox, computes quantized values as constants, then returns `W + (W_q - W_const)`.
Autograd differentiates through `W` (identity gradient = STE) while the forward pass uses the
correct quantized value. This is a pedagogically important result: **naive QAT in autograd fails
silently** -- the model appears to train but learns nothing.

## Empirical Results

### Kill Criteria Assessment (3 seeds: 42, 123, 314)

| Criterion | Metric | Value | Threshold | Verdict |
|-----------|--------|-------|-----------|---------|
| **K1** | Ternary/FP16 individual PPL ratio | 1.026 +/- 0.004 | < 1.05 | **PASS** (3/3 seeds) |
| **K2** | Ternary mean composed PPL | 4.159 | < FP16 (4.348) | **PASS** (3/3 seeds) |
| **K3** | Ternary/FP16 training loss ratio | 1.178 +/- 0.010 | < 2.0 | **PASS** (3/3 seeds) |

All three kill criteria pass on all three seeds.

### Summary Table (mean across 3 seeds)

| Metric | FP16 LoRA | Ternary LoRA | INT4 LoRA |
|--------|-----------|-------------|-----------|
| Mean single-adapter PPL | **2.36** | 2.42 (+2.6%) | 2.38 (+0.9%) |
| Mean composed PPL | 4.35 | **4.16** (-4.4%) | 4.30 (-1.1%) |
| Mean composed/base ratio | 0.635 | **0.609** | 0.631 |
| Mean pairwise \|cos\| | 0.276 | **0.222** (-19.3%) | 0.262 (-4.8%) |
| Delta norm CV | 0.199 | 0.253 (+27%) | **0.186** (-6.6%) |
| Adapter storage (bits) | 150,784 | **15,593** (9.7x smaller) | 38,048 (4.0x smaller) |

### Per-Domain Composed PPL (mean across seeds)

| Domain | FP16 | Ternary | INT4 | Base (no adapter) |
|--------|------|---------|------|-------------------|
| arithmetic | 7.16 | 7.20 | 7.30 | 8.66 |
| reverse | 4.58 | **4.12** | 4.46 | 8.63 |
| repeat | 3.77 | 3.43 | **3.37** | 6.25 |
| sort | 3.73 | **3.57** | 3.70 | 7.75 |
| parity | 2.51 | **2.47** | 2.66 | 3.45 |

Ternary LoRA wins 3/5 domains in composed PPL. FP16 wins arithmetic. INT4 wins repeat.

### Key Findings

**1. Ternary adapters compose BETTER than FP16 (-4.4% composed PPL)**

This is the central result. Despite ~2.6% worse individual quality, ternary adapters
produce a 4.4% improvement when composed. The composition improvement more than
compensates for the individual quality loss. This is consistent across all 3 seeds.

**2. Lower interference explains the composition advantage**

Ternary adapters have 19.3% lower mean pairwise |cosine| similarity (0.222 vs 0.276).
The quantization discretization acts as implicit regularization, reducing cross-adapter
interference. This aligns with the theoretical prediction from Johnson-Lindenstrauss:
random ternary projections preserve distances while inducing sparser representations.

**3. INT4 is the Goldilocks condition**

INT4 adapters achieve nearly FP16-level individual quality (+0.9%) with slight
composition improvement (-1.1% composed PPL) and 4x memory savings. They represent
the practical middle ground: less aggressive than ternary but still beneficial.

**4. Memory savings are substantial**

| | FP16 | Ternary | INT4 |
|--|------|---------|------|
| Storage per adapter | 18.4 KB | **1.9 KB** | 4.6 KB |
| 500 adapters | 9.2 MB | **0.95 MB** | 2.3 MB |
| 50K adapters | 920 MB | **95 MB** | 230 MB |

At SOLE scale (thousands of experts), ternary adapters reduce memory by ~10x.

**5. Delta norm CV is HIGHER for ternary (contrary to MATH.md prediction)**

The mathematical analysis predicted ternary would bound magnitude variance. Instead,
delta norm CV is 27% higher for ternary (0.253 vs 0.199). The quantization creates
more discrete magnitude buckets, increasing variance. However, this does NOT hurt
composition -- the lower cosine similarity (reduced directional interference) matters
more than magnitude uniformity for equal-weight composition.

## What Killed the Hypothesis "Magnitude Bounding"

The MATH.md predicted ternary adapters would have more uniform magnitudes due to
the discrete nature of {-1, 0, 1} weights. This is wrong: the scale factors
(alpha_A * alpha_B) vary MORE across domains for ternary than FP16. The actual
mechanism for better composition is **directional decorrelation** from quantization
noise, not magnitude bounding.

## Limitations

1. **Micro scale only** (d=64, r=4, toy domains). Must validate at d=4096+ on real tasks.
2. **Post-quantized ternary base** (not natively trained BitNet). Natively-trained BitNet
   may show different adapter dynamics.
3. **Equal-weight 1/N composition only**. PPL-probe or top-k routing may change the
   relative advantage.
4. **Composition improvement is modest** (-4.4%). At macro scale with more adapters (N>5),
   the reduced interference could compound -- or noise could dominate.
5. **STE fidelity**: The quantization error during training (mean ~15-20% for ternary)
   may behave differently at higher ranks where more parameters provide redundancy.
6. **Ternary composed PPL (4.16) still worse than FP16 single-adapter PPL (2.36)**.
   The advantage is relative to FP16 *composition*, not FP16 *single*.

## What Would Kill This

At macro scale (Qwen2.5-7B or BitNet-2B-4T):
- If ternary QAT fails to converge on real tasks (STE insufficient at larger scale)
- If composition advantage disappears at N>5 (micro-scale artifact)
- If ternary individual quality drops >5% vs FP16 on real benchmarks (MMLU, HumanEval)
- If INT4 achieves same composition benefit with less quality loss (ternary unnecessary)

## Verdict

**SUPPORTED.** Ternary LoRA adapters trained with QAT (STE) compose 4.4% better than
FP16 LoRA at only 2.6% individual quality cost and 10x memory savings. The mechanism
is directional decorrelation (lower cosine similarity) from quantization-induced
regularization, not the hypothesized magnitude bounding. All 3 kill criteria pass on
all 3 seeds. The result is directionally promising but must be validated at macro scale.
