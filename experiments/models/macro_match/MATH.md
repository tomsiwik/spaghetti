# exp5_macro_match: Mathematical Foundations

## Problem Statement

Given a frozen base model M_0 with parameter count P_base and a target dense
model M_target with parameter count P_target > P_base, can we compose domain
expert capsules onto M_0 such that:

1. Quality: PPL(M_composed) <= 1.10 * PPL(M_target) on target domains
2. Efficiency: ActiveParams(M_composed) <= P_target / 3

## Notation

| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| M_0 | Frozen base model (Qwen2.5-Coder-0.5B-4bit) | P_base = 77M |
| M_target | Dense target (Qwen2.5-Coder-1.5B-4bit) | P_target = 241M |
| D | Number of domains | D = 2 (Python, JavaScript) |
| G | Capsule groups per domain | G = 4 (actual experiment) |
| C | Capsules per group | C = 64 (actual experiment) |
| L | Number of transformer layers | L = 24 |
| d | Model embedding dimension | d = 896 |
| k | Top-k groups selected per token | k = 2 per domain, 4 total |
| N_total | Total groups in composed model | N_total = G * D = 8 |

## Capsule Parameter Count (Actual Experiment: G=4, C=64)

Each capsule group per layer:
- A matrix (detector): C x d = 64 x 896 = 57,344 params
- B matrix (expansion): d x C = 896 x 64 = 57,344 params
- Per group per layer: 114,688 params

Router per layer:
- W_router: d x N_groups = 896 x 4 = 3,584 params (per domain training)
- W_router: d x N_total = 896 x 8 = 7,168 params (composed)

Total capsule params per domain:
```
P_capsule_domain = L * (G * 2 * C * d + d * G)
                 = 24 * (4 * 2 * 64 * 896 + 896 * 4)
                 = 24 * (4 * 114,688 + 3,584)
                 = 24 * (458,752 + 3,584)
                 = 24 * 462,336
                 = 11,096,064
                 ~ 11.1M params per domain
```

Composed model (2 domains):
```
P_total = P_base + D * P_capsule_domain  (+ router expansion)
        = 77.3M + 22.2M
        = 99.4M total params (measured)
```

## Active Parameters Per Token

With top-k routing (k=4 out of 8 total groups):
```
P_capsule_active = L * (k_total * 2 * C * d + d * N_total)
                 = 24 * (4 * 114,688 + 896 * 8)
                 = 24 * (458,752 + 7,168)
                 = 24 * 465,920
                 = 11,182,080
                 ~ 11.1M active capsule params
```

Total active per token:
```
P_active = P_base + P_capsule_active
         = 77.3M + 11.1M
         = 88.3M (measured)
```

Ratio to target:
```
P_active / P_target = 88.3M / 241.3M = 0.37x
```

Close to the 1/3 target. The capsule overhead is small (14.4% of base)
because G=4 and C=64 are conservative. The active param ratio is dominated
by the base model itself.

## Empirical Verification of Gap Analysis

Measured PPL values:
```
PPL(0.5B, Python) = 4.309
PPL(1.5B, Python) = 3.074
PPL(composed, Python) = 3.731

H_gap = ln(4.309) - ln(3.074) = 1.461 - 1.123 = 0.338 nats
H_closed = ln(4.309) - ln(3.731) = 1.461 - 1.316 = 0.145 nats
Gap fraction closed = 0.145 / 0.338 = 42.9%

For JavaScript:
PPL(0.5B, JS) = 5.734
PPL(1.5B, JS) = 4.212
PPL(composed, JS) = 4.924

H_gap = ln(5.734) - ln(4.212) = 1.746 - 1.437 = 0.309 nats
H_closed = ln(5.734) - ln(4.924) = 1.746 - 1.594 = 0.152 nats
Gap fraction closed = 0.152 / 0.309 = 49.2%
```

Capsules close 43-49% of the cross-entropy gap between 0.5B and 1.5B.
To close 90% (matching within 10%), capsules would need to deliver
~2x their current impact, which would require either:
- 2x more capsule capacity (G=8, C=128) -- tested but infeasible on hardware
- More efficient capsule architecture (e.g., LoRA instead of additive ReLU)
- Different base model ratio (1.5B base vs 3B target)

## Quality Model

The composition quality model from micro experiments:

At micro scale (d=64, character-level names):
- N=2 domains: +0.3% vs joint (with 100 calibration steps)
- N=5 domains: +1.6% vs joint (with 200 calibration steps)

The composition gap comes from the function-space mismatch:
```
f_composed(x) = M_0(x) + sum_{i in selected} w_i * Group_i(x)
```

vs the ideal:
```
f_joint(x) = M_0(x) + Delta_joint(x)
```

where Delta_joint is learned on all domains simultaneously.

The calibration (router training on mixed data) partially closes this gap.

## Scaling Analysis: 0.5B+Capsules vs 1.5B

The 1.5B model has ~3x the capacity of 0.5B. Our capsule groups add
~88M params (full precision, ~1.15x the base). The question is whether
domain-specific fine-tuning can close the gap that comes from having
a smaller base model.

Key insight: the 0.5B model already encodes substantial Python/JavaScript
knowledge (it is Qwen2.5-Coder, specifically trained for code). The
capsule groups refine this knowledge on domain-specific data, effectively
providing a form of domain adaptation that the 1.5B model achieves through
raw capacity.

The perplexity relationship:
```
PPL(0.5B) / PPL(1.5B) = exp(H(0.5B) - H(1.5B))
```

where H is the cross-entropy. If PPL(0.5B) = 4.31 and PPL(1.5B) ~ 3.0
(estimated), then:
```
H_gap = ln(4.31) - ln(3.0) = 1.461 - 1.099 = 0.362 nats
```

The capsule groups need to close this 0.362-nat gap through domain
specialization. At micro scale, capsules reduced PPL from 4.31 to 3.72
on Python, closing 0.148 nats of the gap. This represents 41% of the
estimated gap.

## Kill Criteria Formalization

For domain d, let:
- PPL_t(d) = perplexity of 1.5B target on domain d
- PPL_c(d) = perplexity of composed model on domain d

Kill if:
```
exists d: |PPL_c(d) - PPL_t(d)| / PPL_t(d) > 0.10
```

Or for functional evaluation:
```
Score_c / Score_t < 0.90
```

## Assumptions

1. **Base model quality**: The 0.5B-4bit model retains sufficient
   representational capacity for code understanding. 4-bit quantization
   introduces ~1-2% PPL degradation vs full precision.

2. **Capsule additivity**: Capsule outputs add to the base MLP output.
   This is exact by construction (surgery.py zero-initializes B weights).

3. **Domain independence**: Python and JavaScript capsule groups are
   approximately orthogonal in weight space (validated at micro scale,
   cos ~ 0.000 for distinct domains).

4. **SiLU vs ReLU**: The base model uses SiLU activation in its MLP,
   but our capsule groups use ReLU. This means:
   - Capsule dead neuron detection is exact (f=0 for dead)
   - Base MLP has no dead neurons (SiLU has no hard zeros)
   - The additive combination SiLU(base) + ReLU(capsule) is valid

5. **Scale transfer**: Composition mechanisms validated at d=64 transfer
   to d=896. The key mechanism (softmax routing + calibration) is
   architecture-independent.

## Worked Example (Actual Config: G=4/domain, C=64, k=2/domain)

For a single token at layer 0 in the composed model:
- Input x has shape (1, d) = (1, 896)
- Router scores: s = x @ W_r^T, shape (1, 8) for 8 groups
- Top-4 selection: indices of 4 highest scores (k=2 per domain * 2 domains)
- For each selected group i:
  - a_i = ReLU(A_i @ x), shape (1, 64) -- detector activations
  - out_i = B_i @ a_i, shape (1, 896) -- expansion
  - weighted: w_i * out_i where w_i = softmax(s)[i]
- Capsule output: sum of 4 weighted outputs, shape (1, 896)
- Layer output: original_mlp(x) + capsule_output

At d=896 with C=64:
- A_i matmul: 896 * 64 = 57,344 MADs
- B_i matmul: 64 * 896 = 57,344 MADs
- Per selected group: 114,688 MADs
- 4 groups: 458,752 MADs per layer
- Router: 896 * 8 = 7,168 MADs per layer
- Capsule total per layer: ~466K MADs

Base MLP per layer (SiLU-gated, Qwen2.5-Coder-0.5B):
- gate_proj: 896 * 4864 = 4,358,144 MADs
- up_proj: 896 * 4864 = 4,358,144 MADs
- down_proj: 4864 * 896 = 4,358,144 MADs
- Total: ~13.1M MADs

Capsule overhead: 466K / 13.1M = 3.6% per layer.
This is very cheap, but also explains the limited capacity: the capsule
contribution is only 3.6% of the base MLP compute per layer.
