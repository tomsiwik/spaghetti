# Tiny Routing Heads: Mathematical Foundations

## Notation

| Symbol | Meaning | Shape/Type |
|--------|---------|------------|
| d | Base model hidden dimension | scalar (2560 for BitNet-2B) |
| r | LoRA rank | scalar (16) |
| N | Number of domain adapters | scalar (5) |
| h_head | Routing head hidden dimension | scalar (32) |
| x | Input token sequence | (1, L, d) |
| h | Hidden state from base model | (1, L, d) |
| h_pool | Mean-pooled hidden state | (d,) |
| W1_i | Head i first layer weights | (d, h_head) |
| b1_i | Head i first layer bias | (h_head,) |
| W2_i | Head i second layer weights | (h_head, 1) |
| b2_i | Head i second layer bias | (1,) |
| s_i(x) | Score from head i for input x | scalar in [0, 1] |
| A_i | LoRA A matrix for adapter i | (d_in, r) |
| B_i | LoRA B matrix for adapter i | (r, d_out) |

## Per-Adapter Routing Head

Each adapter i has its own binary classification head f_i: R^d -> [0, 1]:

```
s_i(x) = sigma(W2_i * ReLU(W1_i * h_pool(x) + b1_i) + b2_i)
```

where h_pool(x) = mean(h(x), dim=seq_len) is the mean-pooled hidden state from
the base model's last layer, and sigma is the sigmoid function.

### Parameter Count Per Head

```
params_i = d * h_head + h_head + h_head * 1 + 1
         = d * h_head + h_head + h_head + 1
         = h_head * (d + 2) + 1
```

For d=2560, h_head=32:
```
params_i = 32 * (2560 + 2) + 1 = 32 * 2562 + 1 = 81,985
```

That is ~82K per head. For truly tiny heads (~5K), use h_head=2:
```
params_i = 2 * (2560 + 2) + 1 = 5,125
```

However, h_head=2 may be too small for discrimination. We explore h_head in {2, 8, 32}.

Total head params (N=5, h_head=32): 5 * 81,985 = 409,925 ~ 410K

### S2 Check: Head params < 1% of adapter params

Adapter params per adapter (7 projections, rank 16):
```
params_adapter = 7 * (d_in * r + r * d_out) per layer * 30 layers
```
For BitNet-2B with d=2560:
- q_proj, k_proj, v_proj, o_proj: d_in=d_out=2560, each = 2*2560*16 = 81,920
- gate_proj, up_proj: d_in=2560, d_out=6912, each = (2560+6912)*16 = 151,552
- down_proj: d_in=6912, d_out=2560, each = (6912+2560)*16 = 151,552
- Per layer: 4*81,920 + 3*151,552 = 327,680 + 454,656 = 782,336
- Total: 30 * 782,336 = 23,470,080 ~ 23.5M per adapter
- All 5 adapters: 117.4M

Head params (N=5, h_head=32): 410K / 117.4M = 0.35% < 1% [S2 PASS]
Head params (N=5, h_head=2): 25.6K / 117.4M = 0.02% < 1% [S2 PASS easily]

## Training: Binary Classification

Each head i is trained as a binary classifier:
- Positive examples: sequences from adapter i's own domain
- Negative examples: sequences sampled uniformly from all other domains

Loss for head i:
```
L_i = -E[y * log(s_i(x)) + (1-y) * log(1 - s_i(x))]
```
where y=1 for own-domain, y=0 for other-domain samples.

### S3: Independence Property

Each head is trained independently using only:
1. The frozen base model (for hidden state extraction)
2. Its own domain's data (positive examples)
3. Random data from other domains (negative examples)

Adding adapter N+1 requires only training head N+1. No existing head needs
retraining because their training data and labels are unchanged.

## Inference: Top-k Selection and Pre-Merge

At inference time for input x:
1. Compute h_pool(x) from base model (one forward pass through embedding + layers + norm)
2. All N heads score in parallel: s_i = f_i(h_pool(x)) for i=1..N
3. Select top-k adapters by score: I = argtop_k({s_i})
4. Pre-merge: W_composed = W_base + sum_{i in I} (s_i / sum_I s_j) * B_i @ A_i
5. Forward pass with composed weights

### Overhead Analysis (K2)

Head inference cost: N parallel MLPs of size (d, h_head, 1).
FLOPs per head: 2 * d * h_head + 2 * h_head = 2 * h_head * (d + 1)

For h_head=32, d=2560: FLOPs = 2 * 32 * 2561 = 163,904
All 5 heads: 819,520 FLOPs

Base forward pass FLOPs (approximate, 30 layers):
- Per layer: ~4 * d^2 (attention) + 3 * d * d_ff (MLP) ~ 4*2560^2 + 3*2560*6912
- Per layer: 26.2M + 53.1M = 79.3M
- Total: 30 * 79.3M = 2.38B FLOPs

Head overhead: 819K / 2.38B = 0.034% << 5% [K2 expected PASS by wide margin]

The real question is the hidden state extraction cost. If we need a full forward
pass through the base model just to get h for routing, the overhead is 100%
(double the compute). Key insight: we can use an EARLY layer's hidden state
(e.g., layer 4) for routing, and the computation up to that layer is shared
with the actual forward pass. The routing decision is then made early and the
remaining layers use the composed weights.

For this experiment, we use the simpler approach: full base model hidden states
from a single forward pass, then route and do a second pass with composed weights.
The overhead is dominated by the second pass, not the heads themselves.
K2 measures head inference overhead specifically (the MLP computation), not the
hidden state extraction.

## Worked Example (Micro Scale)

d=64, r=4, N=4 adapters, h_head=8

Head params per adapter: 8 * (64 + 2) + 1 = 529
Total head params: 4 * 529 = 2,116
Adapter params per adapter (2 layers, 7 projections): 2 * 7 * 2 * 64 * 4 = 7,168
Ratio: 2,116 / (4 * 7,168) = 7.4% -- higher at toy scale (expected)

Input: x with h_pool shape (64,)
Head i output: sigma(W2 @ ReLU(W1 @ h_pool + b1) + b2) = scalar in [0,1]

If scores = [0.92, 0.15, 0.87, 0.03]:
- Top-2: adapters 0 and 2
- Weights: [0.92/(0.92+0.87), 0.87/(0.92+0.87)] = [0.514, 0.486]
- Composed: 0.514 * adapter_0 + 0.486 * adapter_2
