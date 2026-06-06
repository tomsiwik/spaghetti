# Batched LoRA Latency: Mathematical Analysis

## Notation

| Symbol | Definition | Value (Qwen2.5-0.5B) |
|--------|-----------|----------------------|
| d | Hidden dimension | 896 |
| L | Number of transformer layers | 24 |
| T | Number of LoRA target modules per layer | 7 (q,k,v,o,up,gate,down) |
| r | LoRA rank | 16 |
| k | Number of selected experts (top-k) | 1, 2, or 4 |
| N | Total number of experts | 4 |
| P | Total base model parameters | ~494M |
| s | LoRA scaling factor | alpha/rank = 1.0 |

## LoRA Delta Computation

For a single LoRA adapter applied to a linear layer W of shape (d_out, d_in):

```
Y_base = x @ W.T                           -- shape: (batch, seq, d_out)
Y_lora = x @ W.T + s * (x @ A.T @ B.T)    -- LoRA-adapted output
delta  = s * (x @ A.T @ B.T)               -- LoRA delta only
```

Where:
- A has shape (r, d_in): the down-projection
- B has shape (d_out, r): the up-projection
- x has shape (batch, seq, d_in)

FLOPs per delta computation:
- x @ A.T: batch * seq * d_in * r
- result @ B.T: batch * seq * r * d_out
- Total: batch * seq * r * (d_in + d_out)

For self-attention targets (d_in = d_out = d = 896):
- FLOPs per target = 2 * r * d = 2 * 16 * 896 = 28,672 per token

For MLP targets, dimensions vary but we approximate with d for simplicity.

## Total LoRA FLOPs

Per token, for k experts:

```
F_lora = k * L * T * 2 * r * d
       = k * 24 * 7 * 2 * 16 * 896
       = k * 4,816,896
```

| k | F_lora | Overhead vs base (F_base = 988M) |
|---|--------|----------------------------------|
| 1 | 4,816,896 | 0.49% |
| 2 | 9,633,792 | 0.98% |
| 4 | 19,267,584 | 1.95% |

## Why Sequential Implementation is Slow

The sequential approach performs:
1. Zero LoRA weights via `load_state_dict()` -- O(336 tensor copies)
2. Base forward pass (without LoRA) -- F_base FLOPs
3. For each expert i in top-k:
   a. `load_state_dict()` to set expert i's weights -- O(336 tensor copies)
   b. Full forward pass (base + LoRA) -- F_base + F_lora/k FLOPs

Total FLOPs: (1 + k) * F_base + F_lora
Total state_dict operations: (1 + k) * 336 tensor copies

The state_dict overhead dominates because:
- Each `load_state_dict()` call validates keys, copies 336 tensors
- Python dict iteration and tensor.copy_() calls are not GPU-parallelizable
- For k=2: 3 * 336 = 1,008 tensor copy operations per inference

## Why Batched Approaches are Faster

### Approach A: Direct Parameter Copy (k forward passes, skip state_dict)

Instead of `model.load_state_dict(state, strict=False)`, directly write:
```python
lora_layer.lora_A["default"].weight.data.copy_(A)
lora_layer.lora_B["default"].weight.data.copy_(B)
```

This eliminates:
- Key validation overhead
- Dict iteration
- Strict mode checking

Still requires k forward passes but each parameter swap is O(1) instead of O(336).

### Approach B: Hook-Based (1 forward pass)

Register forward hooks on each LoRA target that add the weighted delta:
```python
def hook(module, input, output):
    x = input[0]
    for each expert i with weight w_i:
        delta = s * (x @ A_i.T @ B_i.T)
        output += w_i * delta
    return output
```

FLOPs: F_base + F_lora (exactly the theoretical minimum)
Overhead: 168 Python hook function calls per forward pass

### Approach C: Stacked Batch (1 forward pass, k * batch_size)

Stack the input k times, apply expert-specific deltas per slice:
```python
stacked_input = input.repeat(k, 1)  # (k*batch, seq)
# hooks apply expert i's delta to slice [i*batch : (i+1)*batch]
output = model(stacked_input)
# unstack and weighted average
```

FLOPs: k * F_base + F_lora (k forward passes worth of base computation)
But in ONE kernel launch, better GPU utilization.

## Memory Overhead

Per expert, pre-extracted LoRA matrices:
```
M_expert = L * T * 2 * r * d * sizeof(float32)
         = 24 * 7 * 2 * 16 * 896 * 4 bytes
         = 18.4 MB
```

For N=4 experts: 73.5 MB. For N=100 experts: 1.8 GB. Fits in GPU memory.

## Gap Between Theory and Practice

Theoretical overhead: 0.98% at k=2
Best measured overhead: 61% at k=2 (hook-based)

The gap is due to:
1. **Python hook overhead**: 168 Python function calls per forward, each with GIL acquisition, argument unpacking, and return value processing. At ~0.1ms per hook call, this adds ~17ms
2. **Small matmul inefficiency**: Each hook computes r * d matmuls, which are too small for GPU saturation (16 * 896 = 14K elements). The GPU is underutilized
3. **Memory allocation**: Each delta tensor requires allocation and deallocation
4. **Framework overhead**: PyTorch dispatcher, autograd graph (even with no_grad), CUDA kernel launch latency

To achieve theoretical overhead, one needs:
- Fused CUDA kernel that applies all LoRA deltas inline during the linear layer computation
- No Python in the hot path
- Batched matmul across all 168 targets (single kernel launch)

This is the approach used by production MoE systems (e.g., vLLM's LoRA serving, S-LoRA).
