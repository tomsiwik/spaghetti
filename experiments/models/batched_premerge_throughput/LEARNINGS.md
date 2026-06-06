# Learnings: exp_batched_premerge_throughput

## Core Finding

Runtime LoRA (factored x@A@B application) is **4-87x faster than batched pre-merge** for per-token routed composition on Apple Silicon, because it avoids materializing the O(d·r·d) outer product entirely. The architectural recommendation is definitive: **pre-merge for always-on adapters at load time, runtime LoRA for per-token routed domain experts at inference time.** This resolves the ternary-to-dense latency problem identified in exp_e2e_demo_pipeline_mlx without requiring re-quantization.

## Why This Happened (Literature-Grounded)

### The O(d/r) complexity gap is fundamental

Pre-merge materializes the full rank-r update: `delta = B^T @ A^T`, a (d_out, d_in) matrix costing O(d·r·d) = 104.9M FLOPs at d=2560, r=16. Runtime LoRA applies the factored form: `x @ A` then `@ B`, costing O(d·r + r·d) = 81.9K FLOPs per token per expert. The ratio is O(d/r) = 160x per operation. Empirically, the advantage ranges from 4x (high N, small T where grouping overhead matters) to 87x (favorable T/M ratios).

This is not a new observation in isolation — LoRA (Hu et al., arXiv 2106.09685) originally described both application modes. But the **interaction with ternary bases** is what makes this load-bearing for our architecture: pre-merge destroys ternary structure (exp_e2e_demo_pipeline_mlx showed 2.33x sustained generation penalty), while runtime LoRA keeps base weights untouched. Runtime LoRA simultaneously solves the FLOP problem AND the weight-structure problem.

### Batched pre-merge speedup is real but inflated by straw-man baseline

The "13.1x mean speedup" over naive per-token merge conflates two effects: (1) algorithmic merge reduction (T/M ratio, the genuine contribution) and (2) Python loop overhead elimination (the confound). The reviewer correctly identified that measured 56x exceeds theoretical 15.1x at T=512, N=4, k=1 — the ~3.7x excess is Python iteration overhead. A vectorized naive baseline would narrow the gap to approximately T/M.

The honest framing: batched pre-merge achieves speedup ≈ T/M over a properly vectorized baseline, where M = min(T, C(N,k)) unique expert sets. This ranges from trivial (1.04x at M/T=0.91) to substantial (T/M at low M/T). But it is irrelevant in practice because runtime LoRA dominates both strategies.

### Grouping overhead is negligible — MLX gather is fast

Token grouping via `mx.sort` + `mx.gather` costs 0.18-0.23ms regardless of batch size (T=32 to T=512). This is <1.5% of merge savings at all tested configurations. MLX's Metal-backed gather operations are highly optimized for Apple Silicon's unified memory architecture. This validates the assumption from MATH.md and means the batching machinery itself is viable infrastructure for per-token routing dispatch.

## Confirming Evidence

1. **exp_e2e_demo_pipeline_mlx (our own)**: Predicted "runtime LoRA at 0.58% per-token overhead may be cheaper than pre-merge at 133% on ternary bases." This experiment confirms that prediction quantitatively — runtime LoRA is 4-87x faster than even the optimized batched pre-merge, let alone the 133% penalty from naive merge on ternary.

2. **MoLoRA (arXiv 2603.15965)**: Achieves per-token routing with 1.7B base + 4 adapters outperforming monolithic 8B, using runtime adapter application (not pre-merge). Their success at scale confirms runtime LoRA as the production-viable strategy for routed composition.

3. **QA-LoRA (arXiv 2309.14717)**: Documents that merging into quantized weights requires re-quantization to preserve inference speed. Our finding that runtime LoRA sidesteps this entirely is a stronger solution — no approximation error from re-quantization.

4. **DeepSeek-V3 (arXiv 2412.19437)**: Uses expert parallelism for MoE serving, where each expert is applied independently (analogous to runtime LoRA) rather than merged into a single weight matrix. The factored application pattern is the production standard for mixture-of-experts.

## Contradicting Evidence

1. **Pre-merge was "0% overhead" in exp_adapter_inference_speed_mlx.** This is not contradicting — that experiment measured single-pass forward time with uniform routing (all tokens same expert set). The 0% overhead applies to the merge-then-forward path when amortized across all tokens. The issue arises with heterogeneous per-token routing, where merge must happen per unique expert set. The reconciliation: pre-merge IS free when the expert set is constant (always-on adapters); it is expensive when it varies per token.

2. **The naive baseline criticism weakens the "batched > naive" headline.** The reviewer correctly notes that no production system would use a Python per-token loop. The batched approach IS the obvious implementation. However, this does not invalidate the experiment — it correctly identifies runtime LoRA as superior to both merge strategies, which is the actionable finding.

## Alternative Approaches

### 1. Hybrid pre-merge + runtime LoRA pipeline
Pre-merge always-on adapters (instruction tuning, safety) once at load time. Apply domain-specific experts via runtime LoRA at inference. This is the recommended architecture from this experiment. Estimated overhead: ~0.58% per routed expert per token (proven in exp_adapter_inference_speed_mlx) + 0% for always-on (amortized pre-merge).

### 2. Expert parallelism via MLX stream scheduling
MLX supports multiple compute streams. Domain experts could run on separate streams in parallel, similar to DeepSeek-V3's expert parallelism but on a single device. This would reduce the k*O(d*r) serial overhead of applying multiple experts per token. Reference: MLX async evaluation documentation; DeepSeek-V3 (arXiv 2412.19437) expert parallelism.

### 3. Compiled runtime LoRA kernels via mx.compile
The experiment showed 1.79x speedup from mx.compile on batched merge. Applying compilation to the runtime LoRA path could further reduce the already-low overhead. The risk: variable group sizes may trigger recompilation (flagged in MATH.md but untested). Mitigation: pad groups to power-of-2 sizes.

### 4. Fused adapter kernels
Instead of separate x@A and result@B operations, fuse into a single kernel that streams through A and B without materializing the intermediate (1, r) vector. At r=16, the intermediate is tiny but the kernel dispatch overhead may be significant for T=1 autoregressive steps. Reference: custom Metal kernels in mlx-bitnet (exo-explore/mlx-bitnet).

## Implications for Next Experiments

1. **Runtime LoRA is the serving strategy for routed composition.** All future per-token routing experiments should use runtime LoRA, not pre-merge. Pre-merge is reserved for always-on adapters only. This simplifies the serving pipeline and eliminates the ternary-to-dense conversion problem.

2. **The "pre-merge is free" claim in VISION.md and FINDINGS.md needs a second qualification.** exp_e2e_demo_pipeline_mlx already added "not free for sustained generation on ternary." This experiment adds: "not the right strategy for per-token routing regardless of base type." Pre-merge is free only for constant expert sets.

3. **Routed top-k composition is unblocked.** The competitive benchmark (exp_competitive_benchmark) identified uniform composition as a failure mode — SOLE lost to Qwen2.5-3B on 4/6 benchmarks partly because uniform composition hurts factual recall. Routed top-k with runtime LoRA is now the clear path forward: route per-token, apply top-k experts via factored form, keep base weights in ternary.

4. **Memory overhead from runtime LoRA is minimal but needs tracking.** Each adapter requires A (d_in × r) + B (r × d_out) in memory. At d=2560, r=16, bf16: 2 × 2560 × 16 × 2 bytes = 160KB per adapter per layer. For N=24 adapters × 28 layers = 107.5MB total adapter memory. This is negligible vs the 1.7GB base model.

5. **mx.compile recompilation with variable group sizes is an open risk.** This experiment did not test it. If routed top-k produces highly variable group sizes across batches, recompilation overhead could negate the runtime LoRA advantage. The batched premerge experiment's grouping infrastructure (sort + gather + scatter) transfers directly and should be reused.

## Recommended Follow-Up

**exp_routed_topk_composition** — Per-token top-k routing with runtime LoRA on the 7 genuine domain adapters from exp_real_data_25_domain_adapters. Motivation: (1) competitive benchmark showed uniform composition hurts factual recall, (2) this experiment proved runtime LoRA is 4-87x faster than pre-merge for routed workloads, (3) N=24 experiment showed genuine domain routing heads achieve 98.5% val accuracy with >96% recall. Use MoLoRA (arXiv 2603.15965) architecture as reference — they achieved 1.7B+4 adapters > 8B monolithic. Pre-registered K1: routed composition PPL < uniform composition PPL on held-out domain data. K2: runtime LoRA overhead < 5% vs base model generation speed.
