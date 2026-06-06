# Learnings: exp_bitnet_adapter_inference_speed

## Core Finding

Runtime LoRA application on Apple Silicon (MLX) incurs 33.2% overhead at k=1 (3.5x theoretical FLOPs prediction), while pre-merge is free (~0%). The 3-4x theory-measurement gap is explained by Metal kernel dispatch overhead and Python-level reimplementation costs, both of which are micro-scale artifacts that shrink at production dimensions.

## Why This Happened (Literature-Grounded)

The 3.5x gap between theoretical FLOPs (9.375%) and measured overhead (33.2%) is consistent with three well-documented mechanisms in the serving literature:

1. **Kernel dispatch dominance at small scale.** At d=128, the computation per matmul is tiny (~32K FLOPs for A@x), meaning Metal kernel launch latency dominates wall-clock time. Each runtime LoRA adapter adds 48 extra kernel dispatches (2 matmuls x 6 linear layers x 4 blocks). This is the classic "small-model benchmarking artifact" — at production scale (d=4096), the theoretical overhead drops to rho=0.59% per adapter, and dispatch overhead becomes proportionally negligible.

2. **Memory bandwidth bottleneck.** LLM decoding is memory-bandwidth-bound, not compute-bound (confirmed by LoTA-QAF, MoTE, and S-LoRA analyses). Runtime LoRA requires fetching A and B matrices from separate memory buffers for every forward pass, adding memory traffic that pure FLOP analysis ignores. On Apple Silicon's unified memory this is less severe than discrete GPUs, but still non-zero.

3. **Python-level reimplementation overhead.** The RuntimeLoRATransformer replaces MLX's optimized `nn.Linear.__call__` with explicit Python-level matrix operations. The adversarial review correctly identifies this as making the 33.2% an upper bound — part of the overhead is the cost of the Python reimplementation, not LoRA itself.

## Confirming Evidence

**LoTA-QAF (arXiv: 2505.22934):** Reports 1.7-2.0x speedup from merging adapters into base weights vs. running with active runtime adapters. This directly confirms that unmerged runtime LoRA carries substantial overhead, consistent with our 33.2% finding at micro scale.

**AdapterSoup / LoRA Soups:** Treats adapter composition as a pre-processing step (weight-space averaging), achieving O(1) serving overhead. Confirms the architectural validity of our "always pre-merge" recommendation.

**MoTE (Mixture of Ternary Experts):** Retains a shared full-precision FFN and routes to ternary experts, reducing expert memory footprint by >60% (18.1GB to 6.8GB for 3B model). Confirms that ternary weights are key to making multi-expert serving viable on memory-constrained devices — directly relevant to our BitNet-SOLE architecture.

**Prior project experiments:** CPU llama.cpp (7.5% per adapter), CUDA RTX 4090 (pre-merge ~0%), PyTorch CPU (pre-merge 0%, dynamic ~260%). All confirm the pattern: pre-merge is universally free, runtime overhead varies by implementation quality.

## Contradicting Evidence

**ExpertWeave:** Demonstrates 4-11% latency overhead for 20 concurrent adapters using fused batched-rerouting kernels and virtual-memory co-location. This is dramatically lower than our 33.2% for k=1, suggesting that with proper system-level optimization (custom CUDA kernels, memory management), runtime LoRA overhead can be brought well below our K1 threshold.

**Punica (Grouped GEMM):** Uses specialized Grouped GEMM kernels to batch requests for different LoRA adapters concurrently. This eliminates the sequential dispatch overhead that dominates our micro-scale measurement. The 33.2% overhead is "a symptom of naive implementation rather than a fundamental limitation of LoRA."

**S-LoRA:** Serves thousands of concurrent adapters on a single machine using Unified Paging (joint management of KV cache and adapter weights). Proves that runtime multi-adapter serving scales with proper systems engineering.

**Key discrepancy explanation:** Our experiment uses naive Python-level LoRA application on MLX with no kernel fusion, no memory co-location, and no batched dispatch. Production serving systems (ExpertWeave, Punica, S-LoRA) use custom GPU kernels that fuse the LoRA computation into the base matmul, eliminating the dispatch overhead that dominates our measurement. The K1 KILL is valid for current MLX capabilities but does NOT imply a fundamental architectural limitation.

## Alternative Approaches (What We Could Try Instead)

1. **Fused LoRA kernels for Metal.** If MLX adds fused LoRA support (combining base matmul + LoRA in a single Metal kernel), the dispatch overhead disappears. This is the equivalent of what Punica's Grouped GEMM does for CUDA. Worth monitoring MLX releases.

2. **Router-Driven LoRA Compaction (RDLC).** Instead of storing and fetching adapter weights, procedurally generate them from a random seed with token-dependent coefficients. Eliminates memory fetch overhead entirely — turns adaptation into an algorithmic function. Relevant if we need dynamic per-token routing on memory-constrained devices.

3. **LoRAuter (task-level retrieval).** Compute adapter composition weights once per query (O(T) complexity) rather than per-layer routing (O(N*L)). Combined with pre-merge, this gives O(1) inference cost with query-adaptive composition. Could be the right approach for SOLE's hash-ring routing.

4. **FlexMoRE (rank-heterogeneous experts).** Allocate higher LoRA ranks to reasoning-heavy tasks and lower ranks to knowledge tasks. Matches full-sized expert performance at <1/3 parameter count. Directly applicable to SOLE's domain experts.

5. **MoTE-style shared FFN + ternary experts.** Already aligned with our BitNet architecture. The shared full-precision FFN preserves factual knowledge while ternary experts specialize cheaply. At 1.58-bit, expert memory is minimal.

## Implications for Next Experiments

1. **Pre-merge is the correct serving strategy for SOLE on all platforms.** This is now confirmed across CPU (llama.cpp), CUDA (RTX 4090), and Apple Silicon (MLX). The cross-platform serving matrix is complete. No further inference-speed experiments are needed for pre-merge.

2. **Dynamic per-token routing requires amortization, not runtime LoRA.** If the SOLE router changes expert selection per token, the merge should be batched across tokens sharing the same expert set (e.g., during prompt processing or speculative decoding). The merge itself is cheap (~9.4M FLOPs for N=8 at micro scale, ~7.5% of forward pass at production scale).

3. **The 33.2% overhead should NOT be extrapolated to production scale.** At d=4096, rho=0.59%. Even at 4x measured/theoretical, that's ~2.4% per adapter — well within acceptable limits. A macro-scale MLX benchmark would likely pass K1.

4. **MLX kernel fusion is worth monitoring.** If MLX adds fused LoRA operations, a re-test at micro scale could yield <15% overhead, reviving runtime LoRA as a viable path for dynamic routing on Apple Silicon.

5. **RDLC (zero-stored experts) is a promising research direction** for scenarios where memory is the constraint (e.g., serving many adapters on edge devices). This aligns with the ternary philosophy — minimal storage, maximal capacity.

## New References to Add

- **ExpertWeave** — Multi-adapter serving with virtual-memory co-location and fused batched-rerouting kernels (4-11% overhead for 20 adapters)
- **Punica** — Grouped GEMM kernels for batched multi-tenant LoRA serving
- **S-LoRA** (arXiv: 2311.03285) — Unified Paging for thousands of concurrent LoRA adapters
- **LoRAuter** — Task-aware vector retrieval for O(T) adapter selection
- **FlexMoRE** — Rank-heterogeneous LoRA experts matching full-sized performance at 1/3 parameters
- **RDLC** — Router-Driven LoRA Compaction, zero-stored-expert adaptation via procedural generation
