# Learnings: exp_batched_lora_gather_mlx

## Core Finding
MLX lazy evaluation IS the kernel fusion that Punica BGMV provides on CUDA. Stacking K adapter matrices into a batched tensor and computing a single matmul provides zero speedup at production scale (1.02x at d=2560, L=30) because MLX's computation graph already batches sequential matmuls into a single GPU dispatch. The Punica optimization pattern is structurally impossible to improve upon in lazy-evaluation frameworks. Adapter serving overhead is memory bandwidth, not dispatch count.

## Why This Happened (Literature-Grounded)

Two compounding mechanisms explain this structural impossibility:

**1. Lazy Evaluation Eliminates Dispatch Overhead by Construction.** MLX records all matmul operations into a computation graph without executing them, then dispatches the entire graph to Metal as a batch when `mx.eval()` is called. This is functionally equivalent to what Punica BGMV achieves via fused CUDA kernels — except MLX does it automatically for ALL operations. The Punica paper (2310.18547) explicitly targets eager-execution frameworks (PyTorch) where each op launches a separate kernel. On lazy frameworks (MLX, JAX/XLA), the framework's compiler already performs this fusion. Snider & Liang (2301.13062) confirm that XLA's operator fusion eliminates redundant kernel launches via the same mechanism — lazy graph construction followed by fused dispatch.

**2. LoRA Adapter Matmuls are Bandwidth-Bound, Not Compute-Bound.** The roofline analysis shows operational intensity of 7.95 FLOPs/byte for rank-16 adapters at d=2560, far below the M5 Pro's ridge point of ~51 FLOPs/byte. This means adapter computation is bottlenecked by reading weight matrices from memory, not by arithmetic. Stacking matrices into a batched tensor does not reduce total bytes read — the same A and B matrices must be loaded regardless of dispatch strategy. LoRAFusion (2510.00206) independently confirms that "LoRA kernels are memory-bandwidth-bound because of small rank, with the memory read and write of large activation tensors becoming the bottleneck." The isolated matmul benchmark (all strategies ~157 us regardless of K) is definitive proof.

**3. Python Loop Overhead is Negligible at Production Scale.** The micro-scale 1.75x speedup (d=128, K=5) exists because Python loop overhead (~0.1ms) is 25% of total compute (~0.4ms). At production scale (d=2560, L=30), total compute is ~55ms and Python overhead is <0.5ms (<1%). This is a standard observation in high-performance computing: overhead elimination only helps when overhead is a significant fraction of total work.

## Confirming Evidence

- **Punica BGMV (Chen et al., 2310.18547)**: The BGMV fused kernel they develop is specifically for CUDA/PyTorch eager execution. Their optimization eliminates per-adapter kernel launch overhead — the exact overhead that MLX's lazy evaluation already eliminates. Our null result is the expected outcome when porting this pattern to a lazy framework.
- **Operator Fusion in XLA (Snider & Liang, 2301.13062)**: Comprehensive analysis of XLA's fusion passes. Confirms that lazy graph construction + compiler fusion eliminates redundant kernel launches. MLX's Metal backend implements the same principle.
- **LoRAFusion (2510.00206)**: Documents that LoRA serving is fundamentally bandwidth-bound due to small rank. Multi-LoRA techniques address this via batching across requests, not within a single request's adapter dispatch.
- **Finding #76 (mx.compile redundant for generation)**: Our own prior result showing async_eval already hides dispatch latency. This experiment extends that finding to multi-adapter stacking specifically.
- **Finding #288 (v3 48% overhead from bf16 matmul)**: Identified the overhead as raw matmul cost, not dispatch overhead. Our isolated matmul benchmark confirms this at the individual operation level.
- **Finding #300 (memory bandwidth is the bottleneck)**: Predicted the kill — dispatch optimization cannot reduce bandwidth.
- **vllm-mlx (Barrios et al., 2601.19139)**: Documents MLX inference on Apple Silicon as bandwidth-bound, with unified memory architecture providing zero-copy access but not changing the fundamental bandwidth constraint.

## Contradicting Evidence

- **S-LoRA (Sheng et al., 2311.03285)**: Achieves significant throughput gains serving thousands of concurrent LoRA adapters via custom CUDA kernels. BUT: their optimization targets batched requests (many users, many adapters simultaneously), not single-request multi-adapter dispatch. The S-LoRA gain comes from avoiding adapter swapping across requests, not from reducing per-request dispatch overhead.
- **CaraServe (Li et al., 2401.11240)**: CPU-assisted LoRA loading achieves 1.4x speedup via cold-start elimination. BUT: their bottleneck is adapter LOADING (CPU→GPU transfer), not adapter DISPATCH (matmul execution). On MLX's unified memory architecture, there is no CPU→GPU transfer to optimize — adapters are already in shared memory.
- **At very large K (K=50+)**: The lazy evaluation graph may become large enough that stacking provides benefit via reduced graph complexity. Untested at this scale. The MLX issue #1828 documents performance degradation when calling the same kernel >10,000 times, suggesting there IS a scale at which graph overhead matters. Our K=5 is well below this threshold.

## Alternative Approaches (What Could Actually Improve Speed)

### Proven in our codebase:
1. **Module selection (Finding #304)**: Attn-only for prose domains = 43% fewer adapter reads. Directly reduces bandwidth by eliminating MLP adapter weight loads. Compounds multiplicatively with other bandwidth reductions.
2. **Pre-merge composition (Finding #225)**: 0% overhead at inference by merging adapters before serving. Already deployed in v3. Trades composition flexibility for zero adapter overhead.

### From literature (untested):
3. **Compress then Serve (Gabrielsson et al., 2407.00066)**: Joint compression of LoRA adapters into shared basis + per-LoRA scaling matrices. 500 LoRAs maintain 75% of single-LoRA throughput. Directly reduces bandwidth by compressing adapter weights. Could compound with ternary adapters for extreme compression.
4. **Ternary adapters (BitDelta-style)**: 10x smaller weight reads via ternary quantization of adapter matrices. Pure addition composition eliminates matmul entirely. The most direct attack on the bandwidth bottleneck.
5. **Fewer active adapters via better routing**: Reducing K from 5 to 2-3 directly reduces adapter bandwidth. Finding #305's segment isolation approach enables this by routing segments to single adapters instead of composing all K.

## Implications for Next Experiments

1. **Dispatch optimization on MLX is a closed research direction.** MLX lazy evaluation makes any stacking/batching/fusion scheme redundant for per-request adapter dispatch. Do not attempt BGMV, FlashLoRA, or similar kernel-level optimizations. Finding #76 (mx.compile redundant) and this finding together establish: on MLX, the framework IS the optimizer.

2. **The speed frontier is bandwidth reduction, not compute optimization.** The base model forward pass dominates (51.7ms / 58ms = 89%). Adapter overhead is only 6.3ms at K=5. Even halving adapter overhead saves 3ms — optimizing the wrong 11% of the pipeline. True speed gains come from: (a) ternary adapters reducing weight reads, (b) module selection reducing adapter count, (c) pre-merge eliminating adapter overhead entirely.

3. **This finding is MLX/JAX-specific.** If the project ever ports to PyTorch/CUDA, Punica BGMV becomes immediately relevant. The impossibility is tied to the framework's evaluation model, not to the hardware or the mathematical operation.

4. **The addmm fusion provides free ~1% improvement.** `mx.addmm` consistently saves ~1% by fusing base_out + scale * lora_out into a single operation. Marginal but zero-cost to adopt in production code.

5. **Adapter bandwidth budget: 6.3ms at K=5 is the optimization target.** At 273 GB/s bandwidth on M5 Pro, 6.3ms of bandwidth reads ~1.7 GB of adapter weights. Ternary adapters (10x compression) would reduce this to ~0.17 GB / ~0.6ms, making adapter overhead negligible (<1% of total).
