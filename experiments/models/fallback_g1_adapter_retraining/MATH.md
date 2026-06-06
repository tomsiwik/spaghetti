# MATH.md: Adapter Retraining + Smart Composition

## Problem Statement

Prior experiment (warmstart_scale_validation) produced catastrophically divergent
adapters: rank-16 LoRA on ALL projections of a d=1024 model gave 34.6M trainable
params (17% of 204M base). PPL exploded from 84 to 1415 on domain data (17x WORSE).
Composition ratio was 27.68x.

The root cause was NOT "adapter deltas too small" — it was massive overfitting from
too many trainable parameters relative to domain data (34.6M params on 500K tokens).

This experiment fixes adapter training AND tests smarter composition methods.

## Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| W | (d_out, d_in) | Base model weight matrix |
| A_i | (r, d_in) | LoRA down-projection for adapter i |
| B_i | (d_out, r) | LoRA up-projection for adapter i |
| r | LoRA rank | scalar |
| alpha | LoRA scaling factor | scalar |
| s = alpha/r | Effective LoRA scale | scalar |
| tau_i = s * B_i @ A_i | Task vector for adapter i | (d_out, d_in) |
| lambda_i | Per-task scaling coefficient | scalar |
| N | Number of adapters | scalar |
| d | Model hidden dimension | scalar |

## Fix 1: Conservative LoRA Sizing

**Prior failure**: 34.6M trainable on 204M base = 17% ratio. Even at d=256 with
rank-32 on ALL projections, this gives ~590K trainable on ~13M base = 4.5%.

**Rule of thumb**: trainable params should be < 1% of base for stable LoRA at
small data scales (500K tokens per domain).

For d=256, 4 layers, rank=4, ATTENTION ONLY (QKVO):
- Per-projection: A is (4, 256) + B is (256, 4) = 2,048 params
- Per layer: 4 projections * 2,048 = 8,192 params
- Total: 4 layers * 8,192 = 32,768 params
- Base params: ~13M
- Ratio: 32K / 13M = 0.25% (well under 1%)

**Why attention only**: At small scale (d=256), MLP LoRA tends to overfit because
the MLP is 4d wide — adding LoRA there quadruples the parameter impact per layer.

## Fix 2: Early Stopping

Monitor validation PPL every 500 steps. Stop if no improvement for 2500 steps
(5 consecutive checks). This prevents the training-past-optimum failure that caused
the prior 17x PPL blowup.

## Fix 3: Composition Methods

### Method A: 1/N Averaging (baseline)

    W_composed = W + (1/N) * sum_i(tau_i)

Known weakness: dilutes each adapter by 1/N.

### Method B: Task Arithmetic (Ilharco et al., 2023)

    W_composed = W + sum_i(lambda_i * tau_i)

Each lambda_i controls per-task contribution. Test lambda in {0.3, 0.5, 0.7, 1.0}.
At lambda=1.0, equivalent to simple summation (full strength, no dilution).

### Method C: TIES-Merging (Yadav et al., 2023)

Three steps on flattened task vectors:

1. **Trim**: Zero out bottom 80% of delta_i entries by magnitude
2. **Elect Sign**: Majority vote on sign per parameter position
3. **Disjoint Merge**: Average surviving entries matching elected sign

    W_composed = W + lambda * tau_merged

This resolves sign conflicts and removes low-magnitude noise.

## Evaluation

Primary metric: **domain PPL improvement ratio**

    ratio_i = PPL_base(domain_i) / PPL_adapted(domain_i)

A ratio > 1.0 means the adapter helped. Target: > 1.10 (10% improvement in PPL).

For composition:
    composition_ratio = PPL_composed(general_val) / PPL_base(general_val)

Near 1.0 = safe. < 1.0 = helps general too. > 1.5 = concerning.

## Delta Magnitude Diagnostic

Before composition, measure:
    relative_delta = ||s * B_i @ A_i||_F / ||W||_F

If < 0.001: adapter is effectively a no-op (what prior work called "vacuous")
If > 0.1: adapter may be overfitting
Target: 0.005 - 0.05

## Worked Example (d=256, r=4, N=3)

Base weight W: shape (256, 256), ||W||_F ~ 16 (Xavier init, sqrt(2/d)*sqrt(d*d) = sqrt(2*d))
LoRA: A (4, 256), B (256, 4), alpha=8, s = 8/4 = 2.0

After 3000 steps with lr=3e-4, cosine decay, zero-init B:
- ||B||_F ~ 0.5-2.0 (depends on gradient signal)
- ||A||_F ~ 1.0 (Kaiming init, scale = 1/sqrt(d))
- ||delta||_F = s * ||B @ A||_F ~ 2.0 * ||B||_F * ||A||_F / sqrt(r)
- If ||B||_F ~ 1.0: delta ~ 2.0 * 1.0 * 1.0 / 2.0 = 1.0
- relative_delta = 1.0 / 16.0 = 0.063 (in target range)

For 3 adapters with 1/N averaging:
- Effective delta per adapter: 0.063 / 3 = 0.021 (still detectable)

For Task Arithmetic lambda=1.0:
- Effective delta: 3 * 0.063 = 0.19 (sum, may be too aggressive)

For Task Arithmetic lambda=0.3:
- Effective delta: 0.3 * 3 * 0.063 = 0.057 (moderate)

## Assumptions

1. FineWeb-Edu provides sufficiently distinct domain distributions
2. d=256 model can learn enough structure for domain adaptation to be meaningful
3. 500K tokens per domain is sufficient for rank-4 LoRA (very few params to learn)
4. Early stopping prevents overfitting that destroyed prior adapters
5. The ternary base's quantization noise is not the bottleneck for adapter learning
