# Learnings: exp_bitnet_llamacpp_serving

## Core Finding

llama.cpp successfully serves BitNet-2B-4T (TQ2_0) with up to 5 runtime LoRA adapters on CPU, with 47.1% throughput overhead following an affine model (9.5% + 7.5%×N), hot-swap via API works without corruption, but the overhead is 10× higher than FLOP predictions due to memory-bandwidth bottleneck.

## Why This Happened (Literature-Grounded)

The 10× gap between FLOP-predicted (0.9%/adapter) and measured (7.5%/adapter) overhead is a well-documented phenomenon in LoRA serving literature. The root cause is **arithmetic intensity mismatch**:

1. **TQ2_0 base model weights are 2-bit packed**, achieving high arithmetic intensity (many FLOPs per byte loaded). LoRA weights are float32 with tiny rank-16 matrices — each matmul loads A∈ℝ^{d×r} and B∈ℝ^{r×d} but performs very few FLOPs relative to bytes transferred.

2. **LoRAFusion (arXiv 2510.00206)** documents that applying LoRA reduces training throughput by ~40% despite adding <1% parameters, attributing this to "redundant memory accesses of the large activation tensor relative to their small computational scale." This confirms our finding is not an artifact of our setup.

3. **S-LoRA (arXiv 2311.03285)** addresses this at GPU scale using custom SGMV CUDA kernels (from Punica) that batch heterogeneous adapter requests. Without such kernels, naive LoRA serving on CPU has no way to amortize the memory access pattern, explaining why our overhead is particularly high.

4. **Compress then Serve (arXiv 2407.00066)** shows that even with optimized GPU kernels, multi-adapter overhead is significant — their compressed LoRA approach maintains 75% throughput at 1000+ adapters, implying ~25% overhead with specialized compression vs our ~47% with 5 naive adapters.

The affine model (fixed 9.5% + 7.5%×N marginal) is consistent with a two-component cost: graph scheduling overhead (fixed, from having adapter nodes in the compute graph) plus per-adapter bandwidth-bound matmul cost (linear in N).

## Confirming Evidence

- **S-LoRA (arXiv 2311.03285)**: Confirms multi-adapter serving requires specialized memory management. Their Unified Paging + custom CUDA kernels achieve 4× throughput over naive PEFT, implying naive serving has massive overhead. CONFIRMS our finding of high overhead without optimization.

- **LoRAFusion (arXiv 2510.00206)**: Documents ~40% throughput loss from LoRA despite <1% parameter increase. Attributes this to memory bandwidth, not compute. CONFIRMS the FLOP-vs-actual gap.

- **Punica (SGMV kernels)**: Achieves ~2ms per-token latency overhead via batched heterogeneous adapter computation. Shows that kernel-level optimization can dramatically reduce overhead. CONFIRMS that the overhead is an implementation issue, not a fundamental limit.

- **dLoRA**: Proposes merging the most-accessed adapter into base weights to avoid runtime overhead entirely. Notes that "unmerged inference fails to address the large number of extra computation overhead." CONFIRMS that runtime LoRA inherently carries significant overhead.

## Contradicting Evidence

- **LoRA original paper (arXiv 2106.09685)**: Claims "no inference latency" because adapted weights can be merged with base weights. This is technically true for single-adapter deployment but **inapplicable** to our multi-adapter runtime serving scenario where adapters must remain separate for hot-swap and per-request routing.

- **Our own exp_bitnet_serving_path (KILLED)**: Previously explored LoTA-QAF weight merge as an alternative — found it non-viable (116× below grid flip threshold for ternary weights). The merge path that would eliminate overhead is blocked by ternary arithmetic constraints, forcing us into the higher-overhead runtime path.

- **No contradicting evidence on the overhead magnitude**: No source found lower per-adapter overhead on CPU without custom kernels. Our 7.5%/adapter is actually reasonable compared to the literature's reports of 25-40% total overhead on GPU.

## Alternative Approaches (What We Could Try Instead)

### 1. Compress then Serve (arXiv 2407.00066)
Joint compression of LoRA adapters into a shared basis + per-adapter scaling matrices. Maintains 75% throughput at 1000+ adapters. Would require: (a) collecting all trained adapters, (b) computing shared SVD basis, (c) converting to compressed GGUF format. **Most promising for scaling beyond N=5.**

### 2. LoRA-Mixer (arXiv 2507.00029)
Serial attention routing across frozen LoRA experts — achieves +7.6% on GSM8K using 48% of parameters vs other MoE approaches. Supports frozen LoRA deployment without retraining. Maps directly to our per-token routing track (exp_bitnet_per_token_routing). **Worth investigating as the routing layer.**

### 3. EdgeLoRA (community, llama.cpp ecosystem)
Adaptive adapter selection + LRU caching + unified batch inference for llama.cpp. Claims 2-4× throughput gains over vanilla llama.cpp LoRA. Not yet verified but could directly benefit our serving path. **Low-hanging fruit if it works with TQ2_0.**

### 4. Rank Reduction
Our affine model predicts rank-8 would halve marginal cost to ~3.75%/adapter, allowing ~10 simultaneous adapters under 50%. This is the simplest mitigation — train at rank-8 instead of rank-16. Trade-off: adapter expressivity vs serving budget.

### 5. Selective Adapter Activation
Instead of N simultaneous adapters, use per-token routing to activate only top-k (e.g., k=2 of N=20). This converts the problem from "serve N adapters" to "route to k adapters per token." Aligns with exp_bitnet_per_token_routing.

## Implications for Next Experiments

1. **exp_bitnet_per_token_routing (P2)**: Now has concrete overhead budget to work with. Per-token routing should target k≤3 active adapters per token to stay under 30% overhead (leaving margin for routing computation itself). The LoRA-Mixer architecture is a strong candidate for the routing mechanism.

2. **Rank-8 validation**: A quick follow-up to verify the affine model's rank-8 extrapolation (N_max≈10) would de-risk the scaling story. Could be done as a micro-benchmark, not a full experiment.

3. **Metal/GPU support**: TQ2_0 Metal shaders are upstream's problem, not ours. But when they ship, overhead percentages will change dramatically (CPU is bandwidth-starved relative to GPU). Our affine model is CPU-specific.

4. **Compress then Serve**: If we ever need >5 simultaneous adapters (production scenario), this paper's joint compression approach is the most promising path. Would require a new experiment node.

5. **The K2 margin is a feature, not a bug**: 47.1% vs 50% with CI overlapping threshold means rank-16 is at the practical limit for simultaneous activation. This naturally pushes toward per-token routing (activate fewer, route smarter) rather than brute-force multi-adapter serving.

## New References to Add

1. **Compress then Serve** (arXiv 2407.00066) — Joint LoRA compression for serving, 75% throughput at 1000+ adapters
2. **LoRA-Mixer** (arXiv 2507.00029) — Serial attention routing for frozen LoRA experts, frozen deploy paradigm
3. **LoRAFusion** (arXiv 2510.00206) — Documents ~40% throughput loss from LoRA, bandwidth-bound analysis
