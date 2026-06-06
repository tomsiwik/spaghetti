# Training Speed Optimization: Mathematical Foundations

## 1. Mechanism Definition

### Training Loop Anatomy

Each LoRA training step on BitNet-2B-4T consists of:

1. **Forward pass**: x in R^{1 x L} -> logits in R^{1 x L x V}
   - 30 transformer layers, each with 7 LoRA-wrapped projections (q,k,v,o,gate,up,down)
   - Each LoRA projection: y = W_base @ x + (A @ x) @ B * scale
   - Total FLOPs per forward: ~2 * P * L where P = 2.4B params, L = seq_len

2. **Loss computation**: cross_entropy(logits, targets) -> scalar
   - O(L * V) where V = 151,936 (Qwen tokenizer)

3. **Backward pass**: value_and_grad computes dL/d(lora_a), dL/d(lora_b)
   - FLOPs roughly 2x forward (chain rule through same graph)
   - Only LoRA params get gradients (base frozen), but full activations stored

4. **Optimizer update**: Adam step on trainable params
   - For each param p: m = beta1*m + (1-beta1)*g, v = beta2*v + (1-beta2)*g^2
   - p = p - lr * m_hat / (sqrt(v_hat) + eps)
   - Memory: 2x trainable params for m, v states

5. **Evaluation barrier**: mx.eval(model.parameters(), optimizer.state)
   - Forces graph execution, synchronizes GPU

### Bottleneck Analysis Framework

Total wall time per step:
  T_step = T_python + T_dispatch + T_gpu + T_sync

Where:
- T_python: Python overhead (data prep, GC, interpreter)
- T_dispatch: MLX graph construction and kernel dispatch
- T_gpu: Actual Metal compute time
- T_sync: mx.eval synchronization overhead

Key insight from exp_mx_compile_full_pipeline: for GENERATION, T_dispatch is
hidden by async_eval double-buffering. For TRAINING, mx.eval is SYNCHRONOUS
at end of each step, so T_dispatch is NOT hidden -- it adds to total time.

This means mx.compile CAN help training even though it doesn't help generation.

## 2. Why Each Optimization Works

### 2a. Disabling Python GC During Training

Python's cyclic GC runs every ~700 allocations (default threshold). In a training
loop with thousands of tensor operations per step, GC triggers multiple times per
step. Each GC pause: ~1-10ms depending on object count.

Expected savings: if GC triggers K times per step with average pause P_gc:
  T_saved = K * P_gc per step
  Typical: K=2-5, P_gc=1-5ms, so 2-25ms per step

Reference: PyTorch training best practices (gc.disable during tight loops).
This is already in CODING_GUIDELINES.md.

### 2b. Pre-tokenization

Current code tokenizes on-the-fly each step:
  tokens = tokenizer.encode(text)  # Python call, ~0.1-1ms per sample
  x = mx.array(tokens[:-1])[None, :]  # array creation, ~0.01ms

With 200 steps cycling through 500 samples, tokenization happens 200 times total.
Pre-tokenizing converts this to array indexing: O(1) instead of O(L).

Expected savings: ~0.1-1ms per step * 200 steps = 20-200ms total per domain.
Small but free.

### 2c. mx.compile on loss+grad Function

The loss_and_grad function builds a new computation graph each step. With
mx.compile, the graph is traced once and cached. Subsequent calls skip Python
dispatch entirely.

For TRAINING (unlike generation):
- mx.eval is synchronous, so dispatch overhead is NOT hidden
- The training graph is large (forward + backward through 30 layers)
- Graph construction: O(num_ops) Python calls -> single compiled call
- Expected: 30-50% of T_dispatch eliminated

Key constraint: mx.compile with nn.value_and_grad requires careful handling.
The compiled function must not capture mutable state. Use mx.compile on the
step function that includes both the forward/backward AND optimizer update.

From fast-mlx guide: "Compiling graphs with mx.compile can make them run a
lot faster" and "recompilation is relatively expensive and should only be done
if there is sufficient work over which to amortize the cost." A full training
step has AMPLE work to amortize compilation.

### 2d. Batch Size > 1

Current: batch_size=1, seq_len=256. GPU utilization likely <50%.
Matrix multiplications at batch=1 are bandwidth-bound (vector-matrix multiply).
At batch=4-8, they become compute-bound (matrix-matrix multiply).

Throughput scaling: T_batch_B / T_batch_1 < B (sub-linear due to better GPU
utilization). Expected: batch=4 gives ~2-3x throughput per sample.

Memory cost: activations scale linearly with batch size.
Peak memory per step: ~model_size + batch * seq_len * d * num_layers * 2
At batch=1: ~4GB (model) + ~0.2GB (activations) = ~4.2GB
At batch=4: ~4GB + ~0.8GB = ~4.8GB (well within 48GB)

### 2e. Wrapping step in a function (grad release)

From fast-mlx guide: when grads are held as a reference during mx.eval,
that memory cannot be reused. Wrapping loss_and_grad + optimizer.update in
a function releases grads before eval, reducing peak memory.

## 3. What Breaks It

### mx.compile Risks
- Recompilation on shape change: if seq_len varies per sample, each unique
  shape triggers recompilation (~50-60ms each, from mx_compile experiment).
  Mitigation: pad to fixed seq_len or use shapeless=True.
- Closure capture: if model weights are captured as constants, recompilation
  happens after every optimizer update. Must pass model as input, not closure.

### Batch Size Risks
- At very large batch sizes, memory pressure can trigger swap -> catastrophic slowdown
- Loss landscape changes: larger batch = smoother gradients = may need LR adjustment
- For 200 steps with batch=4, we consume 4x more data per step -> 800 effective
  samples. With only 500 training samples, we cycle faster.

### GC Disable Risks
- If MLX creates Python-side reference cycles (unlikely for tensor ops), disabling
  GC could leak memory. Mitigated by gc.enable() + gc.collect() after loop.

## 4. Complexity Analysis

Current per-step cost (batch=1, seq_len=256, d=2560, 30 layers):
- Forward FLOPs: ~2 * 2.4B * 256 = ~1.2T FLOPs
- Backward FLOPs: ~2.4T FLOPs
- Optimizer: ~3 * trainable_params * 4 ops = negligible vs forward/backward
- Total: ~3.6T FLOPs per step

M5 Pro theoretical compute: ~27 TFLOPS (bfloat16)
Theoretical step time: 3.6T / 27T = 133ms (lower bound)

If observed step time >> 133ms, significant overhead exists.
If observed step time ~ 133ms, already near-optimal (K1 kill).

## 5. Worked Example

Baseline (200 steps, batch=1, domain=medical):
- Observed: ~30-60s per domain (150-300ms per step)
- Theoretical minimum: ~133ms per step -> 26.6s per domain
- Overhead ratio: 1.1-2.3x -> meaningful optimization headroom

Optimized (200 steps, batch=4, compiled, no GC):
- If batch=4 gives 2x throughput: ~75ms per effective sample
- 200 steps * 4 samples/step = 800 samples, but same 200 grad updates
- Expected: ~15-20s per domain (2-3x speedup)

## 6. Connection to Architecture

Training speed directly impacts the Evolve cycle. With 50+ adapters needing
periodic retraining, a 2x speedup saves ~25 minutes per full evolution cycle
(50 adapters * 30s each). This enables tighter iteration loops and more
experimental throughput for the research program.
