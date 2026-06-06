# Linear State Capacity Scaling: Mathematical Foundations

## Notation

| Symbol | Shape | Definition |
|--------|-------|------------|
| x_t | (d,) | Hidden state at position t |
| d | scalar | Model dimension (64, 128, or 256) |
| h | scalar | Number of attention heads (h=4, fixed) |
| d_h | scalar | Head dimension (d_h = d/h: 16, 32, or 64) |
| T | scalar | Sequence length (T=32) |
| S_t^j | (d_h, d_h) | Recurrent state for head j at position t |
| C_S | scalar | State capacity: d_h^2 elements per head |
| G | scalar | Number of capsule groups per layer (G=4) |
| P | scalar | Capsules per group (P=64) |
| k | scalar | Top-k group routing (k=2) |
| L | scalar | Number of transformer layers (L=4) |
| N | scalar | Number of composed domain pools (N=2) |

## The State Capacity Hypothesis

GatedDeltaNet's linear attention uses a recurrent state S of shape (d_h, d_h)
per head per layer. This state stores key-value associations accumulated over
the sequence:

    S_t = g_t * S_{t-1} + k_t @ delta_t^T

where g_t in (0,1) is a decay gate, k_t in R^{d_h} is the key, and
delta_t in R^{d_h} is the retrieval-corrected value.

The state has C_S = d_h^2 elements to store information about up to T positions.
When T << C_S, capacity is not binding. When T approaches C_S, the state must
compress or overwrite older information.

### Capacity at each scale

| Config | d_h | C_S = d_h^2 | T | T/C_S | Capacity regime |
|--------|-----|-------------|---|-------|-----------------|
| d=64 | 16 | 256 | 32 | 0.125 | Abundant (8x headroom) |
| d=128 | 32 | 1024 | 32 | 0.031 | Very abundant (32x headroom) |
| d=256 | 64 | 4096 | 32 | 0.008 | Extremely abundant (128x headroom) |

At T=32, ALL configurations have abundant state capacity. The ratio T/C_S
decreases as d_h grows, meaning larger models have MORE capacity headroom
relative to sequence length. State capacity saturation requires T >> d_h^2,
which at d_h=256 (macro) means T >> 65,536 -- well beyond typical sequence
lengths.

This analysis predicts that state capacity should NOT be the bottleneck at
our test dimensions. Any composition degradation at larger d must come from
a different source.

## The Undertraining Confound

The more dangerous confound is parameter-to-data ratio. At fixed data and
training steps, increasing d changes the optimization landscape:

| Config | Params (PL) | Params (FA) | Steps*BatchSize | Tokens/Param (PL) |
|--------|-------------|-------------|-----------------|-------------------|
| d=64 | 240K | 203K | 600*32=19.2K | 2.56 |
| d=128 | 677K | 538K | 900*32=28.8K | 1.36 |
| d=256 | 2,140K | 1,599K | 1200*32=38.4K | 0.57 |

At d=256, each parameter sees only 0.57 tokens on average -- far below the
typical requirement of 10-100 tokens per parameter for convergence. The model
is severely undertrained.

We can detect undertraining by checking the JOINT (non-composed) baseline
loss. If joint loss degrades with d, the model capacity exceeds data capacity,
and composition results are uninformative.

### Expected joint loss behavior

If capacity is not binding but training is insufficient:
- Joint loss should degrade at larger d (model too big for data)
- Composition gap should grow because the base model is undertrained
- This growth is NOT specific to linear attention

If state capacity is binding:
- Joint loss should be similar across d (more params = more capacity)
- Composition gap should grow specifically for linear attention
- Full attention (no finite state) should NOT show the same degradation

## The Discriminating Test

The experiment compares pure-linear (GatedDeltaNet 4:0) vs full attention
(4:0) at each dimension. The kill criterion is on the RATIO of gap growth:

    ratio(d) = |gap_PL(d)| / |gap_PL(d=64)|

If ratio(d=128) > 3 or ratio(d=256) > 3, the hypothesis is killed.

But the discriminating power depends on the CONTROL condition. If full
attention shows the SAME gap growth pattern, the effect is general (likely
undertraining), not specific to linear state capacity.

The linear-specific gap is:

    delta(d) = gap_PL(d) - gap_FA(d)

If delta(d) grows with d, linear attention has a dimension-specific problem.
If delta(d) is stable or decreasing, the gap growth is shared.

## Computational Cost

Per layer, per token at each dimension:

| d | GatedDeltaNet per layer | Full Attention per layer | Ratio |
|---|------------------------|-------------------------|-------|
| 64 | ~28K MADs | ~20K MADs | 1.4x |
| 128 | ~98K MADs | ~73K MADs | 1.3x |
| 256 | ~360K MADs | ~280K MADs | 1.3x |

The relative cost ratio is nearly constant because both architectures
are dominated by the d^2 projection terms at T=32.

## Worked Example

d=128, h=4, d_h=32, T=32, G=4, P=64, L=4, N=2

State capacity: C_S = 32^2 = 1024 elements per head.
Sequence length: T=32, giving T/C_S = 0.031 (32x headroom).

Composition protocol:
1. Pretrain: 450 steps on all data, all params trainable
2. Domain A: 450 steps, only capsule groups trainable
3. Domain B: 450 steps, only capsule groups trainable
4. Compose: concatenate (4+4=8 groups), double top-k (2+2=4)
5. Calibrate: 150 steps, only router trainable

Parameters: 676,896 (pure-linear), 537,600 (full attention)
Tokens per param: 1.36 (borderline adequate)

Expected result: if state capacity is not binding, pure-linear gap
should be comparable to d=64 (+0.61%). If undertraining dominates,
both pure-linear and full attention gaps should grow.

Observed: pure-linear gap = +1.30%, full attention gap = +3.10%.
Pure-linear outperforms full attention at this dimension, opposite to
the state capacity prediction.

## Assumptions

1. Head count fixed at h=4 to isolate d_h scaling from head-count effects.
   At macro scale, h scales with d (e.g., Qwen3.5: d=896, h=14, d_h=64).

2. Capsule groups and routing are identical across dimensions (G=4, P=64, k=2).
   This means capsule count does not scale with d. At macro scale, capsule
   count would scale to maintain representation capacity.

3. Training steps scale 1.5x and 2x for d=128 and d=256 respectively.
   This is insufficient to compensate for the quadratic growth in parameters
   (2.8x and 8.9x). A proper tokens-per-parameter match would require
   >5000 steps at d=256.

4. The dataset (32K names, character-level) has limited information content.
   At d=256, the model has more capacity than the dataset can fill, making
   the optimization landscape degenerate (many equally good minima).

5. The sequential Python-level recurrence at d=256 is extremely slow (~650
   tok/s vs ~23K at d=64). This practical constraint limits the feasible
   number of seeds and training steps.
