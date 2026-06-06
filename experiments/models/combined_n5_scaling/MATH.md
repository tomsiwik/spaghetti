# Combined Parallel + Pure-Linear at N=5: Mathematical Foundations

## Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| x_l | (B, T, d) | Input to layer l |
| d | scalar | Model dimension (64 at micro) |
| h | scalar | Number of attention heads (4) |
| d_h | scalar | Head dimension = d/h (16) |
| L | scalar | Number of layers (4) |
| N | scalar | Number of composed domains (5) |
| G | scalar | Capsule groups per domain (4) |
| k | scalar | Top-k groups selected per token (2 per domain, 10 composed) |
| C | scalar | Capsules per group (64) |
| D | scalar | Total composed groups = N * G (20) |

## Architecture Under Test

The combined parallel+pure-linear architecture per layer:

```
n_l = Norm(x_l)
x_{l+1} = x_l + GDN(n_l) + CapsulePool(n_l)
```

where GDN is full GatedDeltaNet linear attention (L2 norm + delta rule +
conv1d + per-dim beta + SiLU gate + exponential decay) and CapsulePool
is the routed capsule MoE.

## N=5 Composition Protocol

Given N=5 independently fine-tuned domain models, each with G=4 groups:

1. **Concatenation**: Stack all domain groups into D = N*G = 20 composed groups
2. **Top-k scaling**: k_composed = N * k_single = 5 * 2 = 10
3. **Router calibration**: Train only router weights W_r in R^{20 x d} on
   mixed-domain data (200 steps, round-robin across 5 domains)

## Composition Gap Analysis

### Definition

Let L_joint be the mean validation loss of a jointly-trained model on all
N=5 domains. Let L_composed be the mean validation loss of the composed
model after calibration. The composition gap is:

```
gap% = (L_composed - L_joint) / L_joint * 100
```

### N=2 Baseline (from exp_parallel_pure_linear_combined)

At N=2, the 2x2 factorial showed:
- seq_hybrid gap: -0.50% (composed slightly better than joint)
- par_pure_linear gap: +0.96%
- Cross-architecture degradation: +1.48%
- Effects approximately additive (interaction +0.31%)

### Scaling from N=2 to N=5

Two effects increase with N:

**1. Router capacity pressure**: With D=20 groups (vs 8 at N=2), the
softmax router must discriminate among more candidates. Router W_r has
shape (D, d) = (20, 64). At N=2 with D=8, each router weight vector
partitions a lower-dimensional space. At D=20, 20 weight vectors in
R^64 remain well-separated (20 << 64), so linear separability should
hold.

**2. Interference accumulation**: Each composed group adds its output
to the residual stream. With k=10 active groups (vs k=4 at N=2), the
MLP branch output is a weighted sum of 10 group outputs instead of 4.
The per-group contribution is still gated by softmax routing weights,
but the residual magnitude scales as:

```
||MLP_out|| ~ sqrt(k) * sigma_group    (if group outputs uncorrelated)
             = sqrt(10/4) * ||MLP_out_N2||
             ~ 1.58x
```

The RMSNorm in GatedDeltaNet output gating partially controls this.

**3. Calibration budget**: At N=2, 100 steps with alternating domains
gives each domain ~50 calibration batches. At N=5 with 200 steps
round-robin, each domain gets ~40 batches. Per-domain calibration
budget is slightly reduced but within the validated range (the N=5
prune_compose experiment showed 100 steps sufficed at N=5 for the
relu_router architecture).

### Expected Gap at N=5

From prior experiments:
- N=5 composition with standard capsule_moe: +1.6% vs joint (validated)
- N=2 par_pure_linear: +0.96% vs joint
- N=2 seq_hybrid: -0.50% vs joint

The parallel+pure-linear architecture adds ~1.46pp to the composition gap
at N=2 (compared to seq_hybrid). If this architectural penalty is
independent of N (additive), the expected N=5 gap would be:

```
gap_N5_par_pl = gap_N5_seq_hybrid + delta_architecture
              ~ (baseline_N5_gap) + 1.46pp
```

The baseline_N5_gap for seq_hybrid at N=5 is the primary unknown.
At N=2 it was -0.50%. If it degrades at the same rate as the standard
capsule_moe (from ~-0.3% at N=2 to ~+1.6% at N=5, a ~2pp increase),
then:

```
estimated_gap_N5_par_pl ~ (-0.50% + 2pp) + 1.46pp ~ +3.0%
```

This is well within the 8% kill threshold.

### Kill Criterion Formalization

Kill if:

```
mean_over_seeds(gap%(par_pure_linear, N=5)) > 8%
```

where:

```
gap% = (L_composed(par_pure_linear) - L_joint(par_pure_linear)) / L_joint(par_pure_linear) * 100
```

The 8% threshold is generous because:
- N=2 gap was only +0.96% (8x margin)
- Standard capsule_moe at N=5 was +1.6% (5x margin)
- Even if degradation is super-additive, 8% allows substantial room

## Computational Complexity

### Per-Domain Fine-Tuning (Phase 2)

Each domain model trains G=4 groups, each with capsules of shape:
- A: (C, d) = (64, 64) -- down-projection
- B: (d, C) = (64, 64) -- up-projection

Per-group FLOPs per token: 2 * d * C = 2 * 64 * 64 = 8192
Per-layer MLP FLOPs: G * 2 * d * C = 4 * 8192 = 32768
Total MLP FLOPs: L * 32768 = 131072

Attention (GatedDeltaNet) FLOPs per layer: ~6 * d^2 = 24576
Total attention: L * 24576 = 98304

Total per-token: ~229K FLOPs (unchanged from N=2; each domain trains independently)

### Composed Model (Phase 3-4)

Composed groups: D = 20
Active groups per token: k = 10

Per-layer MLP FLOPs: k * 2 * d * C = 10 * 8192 = 81920
Router FLOPs per layer: D * d = 20 * 64 = 1280

Total per-token composed: L * (81920 + 1280 + 24576) = 431K FLOPs

### Joint Training Baseline

Same architecture (D=4, k=2) but trained on all data:
Total per-token: L * (2 * 8192 + 4*64 + 24576) = 132K FLOPs

### Experiment Wall-Clock Estimate

Per seed, per condition:
- Joint training: 5*300 = 1500 steps * 32 batch * 32 seq = 1.54M tokens
- Pretrain: 300 steps = 0.31M tokens
- Fine-tune: 5 * 300 = 1500 steps = 1.54M tokens
- Calibration: 200 steps = 0.20M tokens
- Total: ~3.6M tokens per seed

At ~30K tok/s (parallel+pure-linear micro throughput): ~120s per seed.
2 conditions * 3 seeds = 6 runs * ~120s = ~720s total (12 minutes).

## Worked Example (d=64, h=4, L=4, N=5, G=4, k=2)

1. Input tokens: (B=32, T=32)
2. Embeddings: (32, 32, 64)
3. Norm: RMSNorm -> (32, 32, 64)
4. GDN attention branch:
   - Q, K, V: each (32, 32, 64) via linear + conv1d
   - L2 normalize Q, K
   - Recurrence over T=32 timesteps: S state (32, 4, 16, 16)
   - Output gated: SiLU(z) * RMSNorm(o) -> (32, 32, 64)
5. CapsulePool branch (composed, D=20, k=10):
   - Router: (32, 32, 64) @ (20, 64)^T -> (32, 32, 20)
   - Softmax + top-10 selection
   - For each of 10 active groups: A @ ReLU(B @ norm_x)
     B: (64, 64), A: (64, 64) -> (32, 32, 64) per group
   - Weighted sum of 10 outputs -> (32, 32, 64)
6. Residual: x + attn_out + mlp_out = (32, 32, 64)
7. Repeat for 4 layers
8. lm_head: (32, 32, 64) -> (32, 32, 28)

## Assumptions

1. **Architectural penalty is N-independent**: The +1.46pp degradation
   from parallel+pure-linear vs sequential+hybrid at N=2 is assumed to
   be a fixed penalty that does not grow with N. Justified by: the
   parallel block topology and attention type are per-layer properties
   that do not change with the number of composed groups.

2. **Router capacity sufficient at D=20**: With 20 groups in R^64, the
   softmax router can separate them (D << d). Justified by: at N=5 with
   the standard capsule_moe, routing works (+1.6% gap, not a router
   failure mode).

3. **Calibration budget scales sub-linearly**: 200 steps at N=5 (vs 100
   at N=2) is sufficient. Each domain sees ~40 batches vs ~50 at N=2.
   Justified by: the prune_compose_n5 experiment used 100 steps at N=5
   and achieved +0.02% delta.

4. **Micro-scale validity**: Results at d=64, L=4, T=32 are directionally
   indicative. The GatedDeltaNet state capacity concern (16x16 at micro)
   is an acknowledged limitation.
