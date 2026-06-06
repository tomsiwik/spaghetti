# MLX Optimization Guide for Ternary Models on Apple M5 Pro

## Hypothesis

There exist at least 3 concrete, paper-backed optimization techniques that can
improve throughput, memory efficiency, or serving quality for our BitNet-2B-4T
+ SOLE architecture on M5 Pro 48GB.

## Key Finding: The System is Bandwidth-Bound; Optimizations Must Reduce Bytes, Not Compute

Our prior experiments (Findings #76, #176) established that BitNet-2B-4T on
M5 Pro achieves 165.6 tok/s at 73% of measured peak bandwidth (268.6 GB/s).
The system is unconditionally memory-bandwidth-bound at batch=1 (Theorem 1 in
MATH.md). The M5 Pro's Neural Accelerators provide 4x AI compute over M4 Pro,
but this extra compute is irrelevant for single-batch generation -- the
bottleneck is reading 1.18 GB of weights per token, not computing with them.

All actionable optimizations must therefore target one of:
1. **Reduce bytes read** (smaller model, sparser access patterns)
2. **Increase bandwidth utilization** (fewer round-trips, memory wiring)
3. **Amortize fixed overhead** (batching, speculative decoding)
4. **Exploit latent hardware** (Neural Accelerators via batching, ANE for offload)

---

## M5 Pro Hardware Profile

| Component | Specification | Source |
|-----------|--------------|--------|
| CPU | 18 cores (6 super + 12 performance) | Apple Newsroom [1] |
| GPU | 20 cores with Neural Accelerator per core | Apple Newsroom [1] |
| Neural Engine | 16 cores (dual-die) | Apple Specs [14] |
| Memory | 48 GB unified (our config) | Apple Specs [14] |
| Memory Bandwidth | 307 GB/s (official) | Apple Newsroom [1,2] |
| Memory Bandwidth | 268.6 GB/s (measured peak) | Finding #176 |
| Process | 3nm N3P, Fusion Architecture (dual-die) | Wikipedia [4] |
| AI Compute | >4x M4 Pro peak GPU AI compute | Apple Newsroom [1,3] |

**Key architectural difference from M4 Pro:** The M5 Pro integrates Neural
Accelerators directly into each GPU core. This is distinct from the Apple Neural
Engine (ANE), which remains as a separate 16-core unit. The per-GPU-core
accelerators target tensor operations and provide the claimed 4x speedup for
matrix-heavy AI workloads [1,3,5].

---

## Recommendation 1: Memory Wiring for Consistent Bandwidth

### Problem
macOS manages GPU memory lazily. Under memory pressure, model weights can be
paged out, causing catastrophic throughput drops. Even without explicit pressure,
the OS may not "wire" (pin) all GPU memory by default.

### Solution
Use `sysctl iogpu.wired_limit_mb` to increase the wired memory limit, ensuring
model weights and KV cache remain resident in GPU-accessible memory.

```bash
# Set wired memory limit to 38 GB (leaves 10 GB for OS + apps)
sudo sysctl iogpu.wired_limit_mb=38912
```

MLX also provides programmatic control:
```python
import mlx.core as mx
device = mx.device_info()
total = device["memory_size"]
mx.set_memory_limit(total - 8 * 1024**3)  # Leave 8 GB for system
mx.set_cache_limit(2 * 1024**3)            # Cap cache at 2 GB
```

### Evidence
- mlx-lm documentation states: "If you see degraded generation speeds [...]
  increase the system wired memory limit" [mlx-lm README, ref 15]
- The M5 Pro's unified memory architecture means CPU and GPU share the same
  physical pool; wiring prevents the VM subsystem from evicting hot pages [1,9]
- Our measured peak bandwidth (268.6 GB/s) is 87.5% of official 307 GB/s;
  memory wiring may close some of this gap

### Prediction
| Metric | Current | With Wiring | Basis |
|--------|---------|-------------|-------|
| Peak bandwidth | 268.6 GB/s | 280-307 GB/s | Eliminate VM interference |
| tok/s (base) | 165.6 | 172-190 | Proportional to BW |
| Consistency (std) | Unknown | Lower variance | No page faults |

### References
- MLX framework documentation: memory management [mx.set_memory_limit]
- mlx-lm README: wired memory recommendations
- Apple Silicon unified memory architecture [1,2,9]

---

## Recommendation 2: Batched Pre-fill via GPU Neural Accelerators

### Problem
At batch=1, the system is bandwidth-bound and the M5 Pro's 4x AI compute
advantage over M4 Pro is wasted. Prompt processing (prefill) is compute-bound
for longer contexts and CAN benefit from the Neural Accelerators, but only if
batched properly.

### Solution
For serving scenarios with multiple concurrent requests, batch prompts together
to increase arithmetic intensity past the ridge point (~batch 13, see MATH.md
Section C). This transforms the workload from bandwidth-bound to compute-bound,
where the Neural Accelerators provide genuine speedup.

For single-user scenarios, implement speculative decoding: generate k candidate
tokens with a small draft model, then verify all k in a single batched forward
pass through the main model. The verification step has batch=k, increasing
arithmetic intensity.

### Implementation
```python
# Batched generation with mlx-lm
from mlx_lm import batch_generate

# Multiple prompts in a single call
prompts = ["What is ML?", "Explain quantum computing", "Write a poem"]
responses = batch_generate(model, tokenizer, prompts, max_tokens=200)
```

For speculative decoding on MLX:
- Draft model: Use a smaller ternary model (e.g., BitNet 0.7B) or n-gram model
- Verification: Single batched forward pass through BitNet-2B-4T
- Expected gain: k-1 "free" tokens per verification step at ~1.5x overhead

### Evidence
- vllm-mlx achieves 525 tok/s on 0.6B model (M4 Max), representing 21-87%
  improvement over llama.cpp via batching and native Metal [arxiv 2601.19139]
- The roofline model (MATH.md Theorem 1) proves batch>13 crosses the ridge point
- Our inference_speed_10x experiment showed batch_generate API exists in mlx-lm [ref 10]
- bitnet.cpp achieves 1.37-5.07x speedup on ARM CPUs via tile optimization [ref 24]

### Prediction
| Scenario | Current tok/s | Predicted tok/s | Basis |
|----------|-------------|-----------------|-------|
| Batch=1 gen | 165.6 | 165.6 (no change) | BW-bound, no benefit |
| Batch=4 prefill | ~40 tok/s/req | ~120 tok/s/req | 3x from shared weight read |
| Speculative (k=4) | 165.6 | ~230-280 | ~1.5-1.7x from draft acceptance |
| Batch=4 gen | N/A | ~400 aggregate | 4 parallel weight amortizations |

### References
- vllm-mlx: Native LLM Inference at Scale on Apple Silicon (arxiv 2601.19139)
- Speculative Decoding (arxiv 2302.01318, Leviathan et al.)
- MLX batch_generate API (mlx-lm documentation)
- bitnet.cpp tiling strategies (microsoft/BitNet [ref 24])

---

## Recommendation 3: mx.compile with Shapeless Mode for Fixed-Architecture Generation

### Problem
mx.compile gives 21% speedup at seq=1 (Finding #176), but recompiles whenever
input shapes change. During generation, each token has the same shape (batch=1,
seq=1), making this stable. However, the first-token prefill has variable length,
causing a recompile each time.

### Solution
Compile the generation step (NOT prefill) with `shapeless=True` to avoid any
recompilation during the decode phase. This is safe because:
1. The decode step always processes exactly one token
2. KV cache length changes but is handled via in-place updates, not shape changes
3. The compiled graph fuses RMSNorm + projection + RoPE + attention + SiLU into
   fewer Metal dispatches

```python
import mlx.core as mx

@mx.compile
def generate_step(model, tokens, cache):
    """Compiled single-token generation. Shape is always (1, 1)."""
    logits = model(tokens, cache=cache)
    return logits[:, -1, :]

# The prefill step is NOT compiled (variable length):
prefill_logits = model(prompt_tokens, cache=cache)
mx.eval(prefill_logits)

# Decode loop uses compiled step:
for i in range(max_tokens):
    logits = generate_step(model, next_token, cache)
    next_token = mx.argmax(logits, axis=-1, keepdims=True)
    mx.eval(next_token)
```

### Evidence
- Finding #176: mx.compile gives 21% speedup at seq=1, 0% at seq=256
  (compile helps only when dispatch overhead is significant relative to compute)
- Finding #76: async_eval already hides most dispatch overhead in mlx-lm's
  default generation loop; compile gives marginal additional benefit ON TOP OF
  async_eval, but eliminates overhead that async_eval can't hide (graph
  construction, kernel launch latency)
- MLX compilation docs: "fusing multiple unary and binary element-wise operations
  into a single kernel" -- this directly reduces memory round-trips [ref 6,7,8]
- GELU fusion achieves 5x speedup via compile [MLX compile docs]

### Prediction
| Metric | Uncompiled | Compiled | Evidence |
|--------|-----------|---------|----------|
| Single token forward | 7.29 ms | 6.03 ms | Finding #176 (measured) |
| tok/s (decode) | 137 | 166 | Finding #176 (measured) |
| Recompilations/session | Variable | 1 (prefill) + 0 (decode) | Architecture |

### References
- MLX compilation documentation (ml-explore.github.io/mlx/compile)
- Finding #176: Metal kernel profiling results
- Finding #76: mx.compile redundancy analysis

---

## Recommendation 4: Custom Fused BitLinear Metal Kernel

### Problem
The current MLX BitLinear implementation performs ternary unpacking, matmul, and
activation normalization as separate Metal kernel launches. Each launch reads
from and writes to device memory. For 30 layers * 7 projections = 210 kernel
launches per token, the overhead accumulates.

### Solution
Implement a fused Metal kernel that combines:
1. Ternary weight unpacking (from packed uint8)
2. Matmul via addition/subtraction (no FP multiply needed)
3. RMSNorm on the output

This eliminates intermediate memory writes between these stages, keeping data
in threadgroup (tile) memory.

### Implementation Approach
Use `mlx.core.fast.metal_kernel` to JIT-compile a custom kernel:

```python
import mlx.core as mx

fused_bitlinear = mx.fast.metal_kernel(
    name="fused_bitlinear",
    input_names=["packed_weights", "activations", "norm_weight"],
    output_names=["output"],
    source="""
    // Per-thread: unpack ternary weight, accumulate via add/sub, normalize
    uint row = thread_position_in_grid.x;
    uint col = thread_position_in_grid.y;

    // 1. Unpack ternary weight from packed byte (5 trits per byte)
    uint byte_idx = (row * cols + col) / 5;
    uint trit_pos = (row * cols + col) % 5;
    uchar packed = packed_weights[byte_idx];
    int weight = extract_trit(packed, trit_pos);  // -1, 0, or 1

    // 2. Multiply-free matmul: just add, subtract, or skip
    float acc = 0.0;
    for (uint k = 0; k < K; k++) {
        int w = unpack_weight(packed_weights, row, k);
        if (w == 1) acc += activations[col * K + k];
        else if (w == -1) acc -= activations[col * K + k];
    }

    // 3. Fused RMSNorm
    // ... (in threadgroup shared memory)
    output[row * out_cols + col] = normalized_acc;
    """,
)
```

### Evidence
- Microsoft bitnet.cpp achieves 1.37-5.07x speedup over llama.cpp on ARM CPUs
  through LUT-based ternary matmul + configurable tiling [ref 24]
- T-MAC framework (underlying bitnet.cpp) replaces multiply-accumulate with
  lookup tables, eliminating FP multiply entirely [ref 10 in NLM research]
- MLX custom Metal kernel API supports JIT compilation from source strings
  [mlx.core.fast.metal_kernel, ref 25,29]
- mlx-bitnet roadmap lists "Optimized Metal kernels for 1.58-bit weights"
  as "In Progress" [ref 8 in NLM research]
- Sparse-BitNet (arxiv 2603.05168) combines 1.58-bit with N:M sparsity
  for 1.30x additional speedup
- Our profiling shows ternary unpacking costs 124.5ms for full prefill
  (Finding #176) -- fusing with matmul eliminates the intermediate write

### Prediction
| Metric | Current (separate) | Fused (predicted) | Basis |
|--------|-------------------|-------------------|-------|
| Kernels per layer | 7+ | 1 | Fusion |
| Intermediate memory writes | 7 per layer | 0 | Kept in tile memory |
| Prefill overhead | 124.5 ms | ~50-80 ms | Eliminate unpack stage |
| tok/s improvement | -- | 5-15% | Reduced dispatch + BW |

### References
- bitnet.cpp: Official inference framework for 1-bit LLMs (github.com/microsoft/BitNet)
- T-MAC: Table-based multiply-accumulate (underlying bitnet.cpp kernels)
- Sparse-BitNet (arxiv 2603.05168): N:M sparsity for ternary models
- MLX custom Metal kernels docs (ml-explore.github.io/mlx/dev/custom_metal_kernels)
- exo-explore/mlx-bitnet: existing MLX ternary implementation

---

## Recommendation 5: ANE Offload for Routing via Orion-Style Direct Access

### Problem
Our SOLE architecture includes a softmax routing head (330K params) that selects
top-k experts per token. This routing computation runs on the GPU, occupying
0.46% of inference time (0.166ms/36ms, Finding from routing optimization
experiment). While negligible now, at higher adapter counts or with more complex
routing, it could become a bottleneck -- and more importantly, any GPU cycle
spent on routing is a cycle NOT spent on weight-reading.

### Solution
Offload the routing head to the Apple Neural Engine (ANE) via Core ML or
direct ANE access (Orion-style). The ANE processes the routing decision in
parallel with GPU weight-reading, achieving true overlap.

### Evidence
- Orion project (arxiv 2603.06728) demonstrated 170+ tok/s for 124M model
  on M4 Max ANE by bypassing Core ML and using _ANEClient directly [ref 19,33,34]
- The ANE has a dedicated high-bandwidth memory connection in M5 Pro,
  operating independently of the GPU memory bus [ref 1,5]
- Routing head is ~330K params = 660KB in bf16 -- trivially fits on ANE
- ANE excels at small, fixed-shape tensor operations (exactly what routing is)
- Apple Intelligence Foundation Models use ANE for on-device inference [ref 32]

### Prediction
| Metric | Current (GPU) | ANE Offload | Basis |
|--------|-------------|-------------|-------|
| Routing latency | 0.166 ms | ~0.1 ms | ANE optimized for small models |
| GPU availability | 99.54% | 100% | Routing off GPU |
| Net throughput gain | -- | <1% | Routing was never the bottleneck |

**Honest assessment:** This is a micro-optimization with <1% throughput impact.
Its value is architectural -- it establishes the pattern for CPU/GPU/ANE
heterogeneous serving that scales better as the system grows more complex.

### References
- Orion: Characterizing and Programming Apple's Neural Engine (arxiv 2603.06728)
- Apple Intelligence Foundation Language Models Tech Report 2025 [ref 32]
- Core ML optimization guide: palettization and W8A8 paths [ref 28]

---

## Recommendation 6: Adapter Compression via Ternary B-Matrices

### Problem
Each adapter occupies 45.2 MB in bf16 (Finding #77). At N=25, this is 1.13 GB;
at N=100, it's 4.52 GB. For bandwidth-bound inference, every byte of adapter
weights that must be read costs throughput.

### Solution
Use ternary B-matrices (already proven in Finding: "Ternary B-matrix: 15.8x
adapter compression"). Since our A-matrices are frozen Grassmannian (orthonormal),
and B-matrices are learned, quantizing B to ternary reduces per-adapter storage
from 45.2 MB to ~2.86 MB (15.8x). This directly reduces the bytes-per-token
when adapters are composed at runtime.

### Evidence
- Ternary adapters compose better than FP16 adapters: -4.4% PPL, -19.3% cosine
  (VISION.md, conclusive finding)
- Ternary B-matrix achieves 15.8x compression with pure-addition composition
  (VISION.md, Serve readiness)
- Runtime LoRA with ternary B: h_out = h + scale * (h @ A) @ B becomes pure
  additions when B is ternary, matching the base model's compute paradigm
- At N=25 with ternary B: 25 * 2.86 MB = 71.5 MB vs 25 * 45.2 MB = 1.13 GB
  (saving 1.06 GB of bandwidth per token)

### Prediction
| Metric | bf16 B (N=25) | Ternary B (N=25) | Ratio |
|--------|-------------|-----------------|-------|
| Total adapter memory | 1,130 MB | 71.5 MB | 15.8x smaller |
| Bytes read per token | +1,130 MB | +71.5 MB | 15.8x less |
| tok/s (addmm, N=1) | 97 | ~130-150 | Less adapter data to read |
| Composition compute | matmul | add/sub only | Eliminates FP multiply |

### References
- VISION.md: Ternary adapter composition results
- BitNet b1.58 (arxiv 2402.17764): ternary weight paradigm
- LoRI (arxiv 2504.07448): Frozen A + sparse B, 17.3% better merge
- FlyLoRA (arxiv 2510.08396): Frozen random A as implicit router

---

## Recommendation 7: vllm-mlx for Production Serving

### Problem
mlx-lm's default generation pipeline is designed for research (single-user,
synchronous). For production serving with multiple concurrent users, it lacks
request scheduling, continuous batching, and efficient KV cache management.

### Solution
Use vllm-mlx for production deployment. It provides:
- Continuous batching (dynamic batch formation from concurrent requests)
- PagedAttention-style KV cache management on unified memory
- Native Metal kernel path (no translation layer overhead)
- Up to 87% throughput improvement over llama.cpp on M4 Max [ref 10,27]

### Evidence
- vllm-mlx (arxiv 2601.19139): 525 tok/s on 0.6B model (M4 Max), 340 tok/s
  on Llama 3.2 1B, representing 21-87% over llama.cpp
- Two implementations exist: vllm-metal (Metal compute shaders) and vllm-mlx
  (MLX framework). vllm-mlx is more mature for LLM serving [ref 27]
- Continuous batching is the key technique: it dynamically forms batches from
  incoming requests, pushing past the batch=1 bandwidth bound
- For our SOLE architecture, this requires custom integration with the routing
  head and adapter composition pipeline

### Prediction
| Metric | mlx-lm (current) | vllm-mlx | Basis |
|--------|------------------|----------|-------|
| Single-user tok/s | 165.6 | ~165 | Same BW-bound baseline |
| 4-user aggregate tok/s | ~165 (sequential) | ~500+ | Continuous batching |
| Request scheduling | None | Priority queues | Production feature |
| KV cache efficiency | Simple rotate | Paged | Lower memory waste |

### References
- vllm-mlx: Native LLM and MLLM Inference at Scale on Apple Silicon (arxiv 2601.19139)
- Blog comparison: vllm-metal vs vllm-mlx (blog.labs.purplemaia.org [ref 27])

---

## What We Ruled Out (Negative Findings)

### N1: GPU Neural Accelerators Cannot Help Single-Batch Generation
The M5 Pro's 4x AI compute claim is real but irrelevant for batch=1 generation.
Theorem 1 proves ternary inference at batch=1 is unconditionally bandwidth-bound.
The Neural Accelerators help for prefill and batched serving only.

### N2: Sparse-BitNet (N:M Sparsity) Already Killed
Finding from exp_sparse_bitnet_pruning: "sparse matmul 7% slower than packed
BitLinear kernel. Inference bandwidth-bound; uint8 packing already optimal.
Metal SIMD handles zeros for free." The Sparse-BitNet paper (arxiv 2603.05168)
claims 1.30x speedup, but this requires custom sparse tensor cores not available
in Metal's standard API.

### N3: KV Cache Quantization Hurts at Short Contexts
Finding from inference_speed_10x: "At 100 tokens, the quant/dequant overhead
exceeds bandwidth savings: 172 -> 160 tok/s (-7%)." Only beneficial for very
long contexts (>1K tokens) where KV cache size dominates weight size.

### N4: Pre-Merge is Worse for Ternary
Finding: "Pre-merge WORSE for ternary (-36%, destroys BW advantage)." Unpacking
ternary to bf16 increases model from 1.18 GB to 4.83 GB, losing the entire
bandwidth advantage. Runtime LoRA (factored form) must be used instead.

---

## Summary: Ranked Recommendations

| # | Recommendation | Expected Impact | Effort | Priority |
|---|---------------|-----------------|--------|----------|
| 1 | Memory wiring (`iogpu.wired_limit_mb`) | 5-15% BW improvement | Low (1 line) | **P0** |
| 2 | Batched prefill/serving | 2-4x multi-user throughput | Medium | **P0** |
| 3 | mx.compile for decode step | 21% decode speedup (confirmed) | Low (done) | **P0 (done)** |
| 4 | Fused BitLinear Metal kernel | 5-15% from fewer dispatches | High (custom kernel) | **P1** |
| 5 | ANE routing offload | <1% throughput, architectural value | High (Orion-style) | **P3** |
| 6 | Ternary B-matrices | 15.8x adapter compression | Medium (already proven) | **P1** |
| 7 | vllm-mlx for production | 2-4x multi-user throughput | High (integration) | **P2** |

## Kill Criteria Assessment

- **K1 (id 254): "Research produces no actionable recommendations"** -> **FAIL** (7 actionable recommendations produced, each with paper references and quantitative predictions)

## Success Criteria Assessment

- **S1 (id 25): "Produces at least 3 concrete, implementable recommendations with paper references"** -> **PASS** (7 recommendations, each citing arxiv papers or official documentation)

## Limitations

1. **No implementation** -- this is a research survey, not an implementation experiment.
   All predictions are derived from roofline analysis and prior findings, not new measurements.
2. **M5 Pro Neural Accelerator specifics are sparse** -- Apple has not published detailed
   ISA or programming guides for the per-GPU-core accelerators. The 4x claim is from
   marketing materials, not technical documentation.
3. **vllm-mlx integration with SOLE is untested** -- continuous batching with per-token
   routing and dynamic adapter composition is a novel serving challenge.
4. **Fused kernel performance is predicted, not measured** -- the 5-15% estimate is
   based on dispatch overhead analysis; actual fusion gains depend on Metal compiler behavior.

## Key References

1. Apple Newsroom: M5 Pro/M5 Max announcement (2026/03)
2. AppleInsider: M5 Pro/M5 Max specs and performance analysis
3. MacSales: Inside M5 Pro Fusion Architecture
4. Wikipedia: Apple M5
5. Apple Newsroom: M5 base chip (2025/10)
6. llama.cpp Issue #5761: BitNet b1.58 ternary model support
7. BitNet b1.58 2B4T Technical Report (arxiv 2504.12285)
8. BitNet b1.58 (arxiv 2402.17764)
9. MLX framework (github.com/ml-explore/mlx)
10. vllm-mlx (arxiv 2601.19139)
11. TernaryLLM overview (emergentmind.com)
12. MLX Custom Metal Kernels documentation
13. Popular Science: M5 Pro/M5 Max overview
14. Apple Support: MacBook Pro M5 Pro Tech Specs
15. SitePoint: Local LLMs on Apple Silicon 2026
16. Reddit: M5 Neural Accelerator discussion
17. Reddit: 1-bit model CPU benchmarks
18. Apple Newsroom: MacBook Pro M5 Pro/M5 Max
19. Reddit/arXiv: Orion ANE programming (arxiv 2603.06728)
20. ERIC KIM: M5 chip analysis
21. HuggingFace: microsoft/bitnet-b1.58-2B-4T model card
22. bitnet-rust: High-performance Rust BitNet implementation
23. Hacker News: BitNet 2B4T technical discussion
24. microsoft/BitNet: Official 1-bit LLM inference framework
25. MLX mlx.core.fast.metal_kernel API reference
26. MLX Issue #3263: metal_kernel buffer binding
27. Blog: vllm-metal vs vllm-mlx comparison
28. Core ML Tools: Optimization overview
29. MLX docs: Custom Metal Kernels guide
30. MLX Discussion #1977: Custom primitives
31. Ultra-low-bit quantization overview (emergentmind.com)
32. Apple Intelligence Foundation Models Tech Report 2025
33. Reddit: Orion ANE training discussion
34. Orion: Characterizing and Programming Apple's Neural Engine (arxiv 2603.06728)
35. Apple Support: MacBook Pro M5 Tech Specs
36. exo-explore/mlx-bitnet: MLX BitNet implementation
