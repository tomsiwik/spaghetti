# BitNet + llama.cpp LoRA Serving: Mathematical Analysis

## Notation

| Symbol | Definition | Value |
|--------|-----------|-------|
| d | Hidden dimension | 2560 |
| r | LoRA rank | 16 |
| N | Number of active adapters | 1-5 |
| L | Number of transformer layers | 30 |
| P | Number of projection types per layer | 7 (q, k, v, o, gate, up, down) |

## LoRA Overhead Model

### Per-Token LoRA Cost

For a single adapter on one projection, the LoRA delta computation is:

$$\Delta y = x A B$$

where $A \in \mathbb{R}^{d_{in} \times r}$, $B \in \mathbb{R}^{r \times d_{out}}$.

FLOPs for one projection:
$$F_{lora} = 2 \cdot d_{in} \cdot r + 2 \cdot r \cdot d_{out}$$

### Total LoRA FLOPs per Token

With N active adapters across L layers and P projections:

$$F_{total\_lora} = N \cdot L \cdot P \cdot (2 d_{in} r + 2 r d_{out})$$

For BitNet-2B-4T:
- Attention projections (q, k, v, o): $d_{in} = d_{out} = 2560$ (q, o), $d_{out} = 640$ (k, v)
- MLP projections (gate, up): $d_{in} = 2560, d_{out} = 6912$
- MLP down: $d_{in} = 6912, d_{out} = 2560$

Per-layer LoRA FLOPs (one adapter):
$$F_{layer} = 2r(2 \cdot 2560^2 + 2 \cdot 2560 \cdot 640 + 2 \cdot 2560 \cdot 6912 + 6912 \cdot 2560)$$
$$= 2 \cdot 16 \cdot (13107200 + 3276800 + 35389440 + 17694720)$$
$$= 32 \cdot 69468160 = 2.22 \times 10^9$$

Wait -- that seems too high. Let me recalculate more carefully.

Per-layer LoRA FLOPs for one adapter:
- q_proj: $2(2560 \cdot 16 + 16 \cdot 2560) = 2 \cdot 81920 = 163840$
- k_proj: $2(2560 \cdot 16 + 16 \cdot 640) = 2 \cdot 51200 = 102400$
- v_proj: same as k = $102400$
- o_proj: same as q = $163840$
- gate_proj: $2(2560 \cdot 16 + 16 \cdot 6912) = 2 \cdot 151552 = 303104$
- up_proj: same as gate = $303104$
- down_proj: $2(6912 \cdot 16 + 16 \cdot 2560) = 2 \cdot 151552 = 303104$

Total per layer per adapter: $163840 + 102400 + 102400 + 163840 + 303104 + 303104 + 303104 = 1441792$ FLOPs

Total per token per adapter (30 layers):
$$F_{lora,1} = 30 \cdot 1441792 = 43253760 \approx 43.3 \text{M FLOPs}$$

### Base Model FLOPs per Token

For a 2.4B parameter model, approximate FLOPs per token:
$$F_{base} \approx 2 \cdot 2.4 \times 10^9 = 4.8 \times 10^9 \text{ FLOPs}$$

### Predicted Overhead

$$\text{overhead}(N) = \frac{N \cdot F_{lora,1}}{F_{base}} = \frac{N \cdot 43.3 \times 10^6}{4.8 \times 10^9} = 0.9\% \cdot N$$

Predicted: 0.9% per adapter. Measured: ~9.4% per adapter.

### Why 10x Higher Than Predicted?

The discrepancy comes from:
1. **Memory bandwidth**: TQ2_0 base model weights are 2-bit packed, achieving high arithmetic intensity. LoRA weights are float32, much lower arithmetic intensity.
2. **LoRA is bandwidth-bound**: The small rank-16 matrices don't amortize memory access. Each LoRA matmul loads A and B but does very little compute relative to bytes transferred.
3. **Graph scheduling**: Each LoRA adapter adds nodes to the compute graph, increasing scheduling overhead.

The 10x gap between FLOP-predicted and measured overhead confirms that LoRA serving is memory-bandwidth-limited, not compute-limited.

## Scaling Predictions

Fitting an affine model $\text{overhead}(N) = C + m \cdot N$ to 3 measured data points:

| Active adapters | Affine predicted | Measured |
|----------------|--------------------|----------|
| 1 | 17.0% | 17.0% |
| 3 | 32.0% | 34.4% |
| 5 | 47.0% | 47.1% |

Fitted parameters: $C = 9.5\%$ (fixed setup cost), $m = 7.5\%$ per adapter.

The fixed component $C$ includes graph scheduling overhead and per-adapter memory setup. The marginal per-adapter cost is ~7.5% at rank-16.

### K2 Confidence Interval

With $n = 3$ runs, $\text{df} = 2$, $t_{0.025} = 4.303$:
- 5x throughput: $33.8 \pm 0.6$ t/s → $SE = 0.6 / \sqrt{3} = 0.35$ t/s
- True base: $63.9 \pm 2.3$ t/s → $SE = 2.3 / \sqrt{3} = 1.33$ t/s
- Overhead 95% CI via delta method: $47.1\% \pm 5.3\%$ → range $[41.8\%, 52.4\%]$

The 50% threshold falls within the 95% CI. The K2 PASS is marginal.

### Maximum Adapters Within 50% Budget

$$N_{max,50\%} = \lfloor (50\% - 9.5\%) / 7.5\% \rfloor = 5 \text{ (at rank-16)}$$

At rank-8 (halving marginal cost, speculative):
$$N_{max,50\%} \approx \lfloor (50\% - 9.5\%) / 3.75\% \rfloor = 10 \text{ (at rank-8, unverified)}$$

## Memory Model

Per-adapter GGUF size (rank-16, float32):
$$M_{adapter} = 2 \cdot r \cdot L \cdot P \cdot \bar{d} \cdot 4 \text{ bytes}$$

where $\bar{d}$ is the average dimension across all projections.

$$\bar{d} = (2560 + 640 + 640 + 2560 + 6912 + 6912 + 2560) / 7 \approx 3255$$

$$M_{adapter} = 2 \cdot 16 \cdot 30 \cdot 7 \cdot 3255 \cdot 4 = 87,494,400 \approx 83.4 \text{ MB}$$

Measured: 82.5 MB per adapter (consistent with calculation).

Maximum adapters in 1 GB memory budget: $\lfloor 1024 / 82.5 \rfloor = 12$ adapters.
