# Deep Research: MoE vs LoRA Composition, Memory, Speed, Routing, Falcon-Edge
**Date:** 2026-03-27
**NotebookLM ID:** 24299486-cf34-4df2-a4a9-b4401ae84f61

---

## 1. MoE vs LoRA Composition: The Fundamental Mathematical Difference

### 1.1 MoE: Multiplicative Expert Selection

In standard MoE, the output for token x at an MoE layer is:

```
y = sum_{i in TopK} g_i(x) * E_i(x)
```

where `g_i(x) = softmax(W_g * x)_i` is the gate score, `E_i(x)` is the i-th expert FFN output, and TopK selects the top-k experts by gate score. The key mathematical property: **each expert is a separate, complete FFN** with its own `W_up`, `W_down`, `W_gate` matrices. Experts compose through **output-space weighted averaging**, not parameter-space addition.

**Switch Transformer** (arXiv:2101.03961, Fedus et al. 2022):
- Simplified to k=1 (single expert per token)
- Capacity factor C controls max tokens per expert: `buffer_size = C * (N_tokens / N_experts)`
- C=1.0-1.25 optimal; lower C better in large model regime (memory-constrained)
- Achieves 7x pre-training speedup at same compute
- Load balancing via auxiliary loss: `L_aux = alpha * N * sum_i(f_i * P_i)` where f_i = fraction of tokens routed to expert i, P_i = mean gate probability for expert i

**Mixtral 8x7B** (arXiv:2401.04088, Mistral AI 2024):
- 8 expert FFNs per layer, top-2 routing
- 47B total params, 13B active per token
- Gate: `G(x) = softmax(TopK(x * W_g))` with k=2
- Output: `y = sum_{i=0}^{n-1} G(x)_i * SwiGLU_i(x)` where SwiGLU_i is the i-th expert
- Different experts selected per token per layer -- massive combinatorial expressivity

**DeepSeek-V3** (arXiv:2412.19437, DeepSeek-AI 2024):
- 671B total params, 37B active per token
- Fine-grained experts: more experts (N -> mN), smaller hidden dim (d -> d/m), more activated (k -> mk)
- **Auxiliary-loss-free load balancing**: bias term b_i added to gate logits for routing only
  - `routing_score_i = W_g * x + b_i` (for expert selection)
  - `gate_value_i = softmax(W_g * x)_i` (for weighting, NO bias)
  - b_i dynamically adjusted based on batch-level load statistics
  - Avoids quality degradation from auxiliary loss while maintaining balance
- **Multi-Head Latent Attention (MLA)**: compresses KV cache via low-rank projection
  - `c_t = W_DKV * h_t` (compressed latent, much smaller than full KV)
  - K, V reconstructed from c_t at inference time

### 1.2 LoRA Composition: Additive Low-Rank Weight Updates

Standard LoRA composes as:

```
W_composed = W_base + sum_i alpha_i * B_i * A_i
```

where A_i in R^{r x d}, B_i in R^{d x r}, rank r << d. The key mathematical property: **all adapters modify the same weight matrix additively in parameter space**.

### 1.3 Why Additive Composition Fails Where Multiplicative Selection Succeeds

**The interference problem is fundamental:**

1. **Subspace collision**: When adapter deltas Delta_i = B_i * A_i occupy overlapping subspaces, addition creates destructive interference. At rank r with d-dimensional weights, random LoRA subspaces have expected |cos(Delta_i, Delta_j)| ~ sqrt(r/d). For r=16, d=2560 (BitNet-2B scale), this is ~0.079. Small but non-zero, and **cumulative** over N adapters.

2. **No input-dependent gating**: Uniform composition applies ALL adapter deltas regardless of input. MoE selects 1-2 of 8+ experts per token. This is the critical difference -- MoE has O(k) interference per token, LoRA composition has O(N) interference.

3. **Scale ambiguity**: sum of N rank-r updates has effective rank <= Nr, but the magnitude is also N*alpha times the base weight perturbation. 1/N scaling resolves magnitude but destroys per-adapter signal.

**From project findings (FINDINGS.md):** Equal-weight composition yields PPL in trillions at N=5 without 1/N scaling. With 1/N scaling, composed PPL ~ base PPL (adapters cancel out). Top-2 routing wins 8/15 domains with avg -25.7% PPL improvement. **Routing is mandatory.**

### 1.4 What Would Make LoRA Composition as Effective as MoE

**Key papers on bridging the gap:**

- **X-LoRA** (arXiv:2402.07148): Dynamic layer-wise, token-level scaling of LoRA adapters via a learned scaling head. Scalings S in R^{T x L x N} (tokens x layers x experts). Softmax ensures sum-to-one per layer per token. Essentially turns LoRA adapters into a soft MoE.

- **LoRA-MoE / MoLA** (NAACL 2025 Findings): MoE-style routing over LoRA experts with layer-wise expert allocation. Each LoRA becomes an "expert" in an MoE framework.

- **Ortho-LoRA** (arXiv:2601.09684): Projects conflicting task gradients onto orthogonal complements within LoRA subspace. Addresses the root cause: gradient conflict in low-rank optimization landscape.

- **OSRM** (arXiv:2505.22934): Constrains LoRA subspaces to be orthogonal BEFORE fine-tuning, preventing interference at training time rather than trying to fix it at merge time.

- **LoRI** (arXiv:2504.07448): Sparse B matrices reduce cross-task interference. Global mask with ~90% sparsity. **Project finding: null result on ternary base** -- B-sparsity neutral on BitNet-2B (|cos| already ~0.008).

**The fundamental requirement:** LoRA composition needs input-dependent routing (soft or hard) to match MoE. Pure parameter-space addition cannot achieve this. The project's SOLE architecture with hash-ring routing is one valid approach; X-LoRA/MoLA-style learned routing is another.

---

## 2. Reducing Inference Memory for Multi-Adapter LoRA Systems

### 2.1 S-LoRA: Unified Paging (arXiv:2311.03285)

**Key innovation: Unified Memory Pool** that manages both KV cache tensors and LoRA adapter weights in a single paged memory system.

- Adapter weights stored in host memory, loaded on-demand
- Custom CUDA kernels for heterogeneous batching (different adapters in same batch)
- Tensor parallelism strategy for multi-GPU serving
- **Results:** Up to 4x throughput improvement vs HuggingFace PEFT / naive vLLM
- Serves thousands of concurrent LoRA adapters on single GPU
- Memory overhead: only active adapter weights + shared KV cache

### 2.2 Compress then Serve (arXiv:2407.00066)

**Joint compression of LoRA collections:**

1. **Independent compression**: Lower rank per LoRA via SVD -- `Delta_i = B_i * A_i approx U_i * S_i * V_i^T` truncated to rank r' < r
2. **Joint diagonalization**: Find shared basis U_shared, per-LoRA scaling D_i
   - `Delta_i approx U_shared * D_i * V_shared^T`
   - Shared basis amortizes across all adapters
   - Clustering algorithm groups similar LoRAs for joint compression
3. **Results:** 1.6x throughput increase serving 1000+ LoRAs, maintaining 80% of single-model throughput

### 2.3 KV Cache Compression via Cross-Layer Attention

**CLA** (arXiv:2405.12981, Brandon et al. 2024):
- Adjacent layers share K, V activations: layer group {l, l+1, ..., l+s-1} all use KV from layer l
- CLA2 (sharing factor 2): **2x KV cache reduction** on top of MQA
- 1B model: 0.21-0.48 perplexity improvement at same KV budget
- 3B model: only 0.04 perplexity degradation with 2x smaller cache
- CLA3/CLA4 provide more reduction but worse accuracy

**Related approaches:**
- **YOCO** (You Only Cache Once): Only top layer computes KV, bottom half uses efficient attention -- constant cache size
- **SimLayerKV** (arXiv:2410.13846): Identifies which layers have similar KV patterns, only caches unique layers
- **Cross-Model KV Reuse** (arXiv:2512.17910): Shares KV cache between base and LoRA-adapted models via base-aligned block hashing

### 2.4 vLLM Multi-LoRA Memory Management

- **LRU cache** for adapter weights: keeps `max_cpu_loras` in memory
- **Punica kernel**: manages LoRA loading and batched computation
- **Hash-based KV cache**: integrates LoRA IDs into hash for cache reuse across adapted models
- **Tradeoff**: More simultaneous LoRAs = less KV cache space = more preemption = lower throughput at high concurrency
- `--max-lora-rank` controls memory allocation per adapter slot

### 2.5 Achieving <3GB for 2-3B Model with 25+ Adapters

**Architecture stack for minimal memory:**

| Component | Memory | Technique |
|-----------|--------|-----------|
| Base model (BitNet-2B, ternary) | 0.6-1.0 GB | 1.58-bit native weights |
| 25 LoRA adapters (r=16, d=2560) | ~50 MB total | Each adapter: 2 * r * d * 2B * L_adapted ~ 2MB |
| KV cache (seq=2048) | 0.3-0.8 GB | CLA2 + MQA halves this |
| Activations | 0.2-0.4 GB | Gradient checkpointing not needed for inference |
| **Total** | **~1.5-2.5 GB** | **Fits in 3GB budget** |

The ternary base model is the key enabler. FP16 2B model would be ~4GB for weights alone.

**Joint compression** (Compress then Serve) can further reduce 25 adapters by ~4x via shared basis factorization, bringing adapter overhead to ~12MB.

---

## 3. Inference Speed on Apple Silicon for LoRA-Composed Models

### 3.1 MLX Compilation and Kernel Fusion

**mx.compile** fuses multiple GPU kernel launches into a single kernel:
- Reduces memory bandwidth requirements (fewer intermediate reads/writes)
- Reduces kernel launch overhead
- Lazy evaluation: computation graph built, executed when result needed
- Custom Metal kernels via JIT compilation for specialized operations

### 3.2 M5 Performance Characteristics

From Apple ML Research ("Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU"):

| Metric | M4 | M5 | Improvement |
|--------|----|----|-------------|
| Memory bandwidth | 120 GB/s | 153 GB/s | 28% |
| Generation speed | baseline | 1.19-1.27x | 19-27% |
| TTFT (14B dense) | -- | <10s | -- |
| TTFT (30B MoE) | -- | <3s | -- |
| Neural Accelerator TTFT | M4 baseline | up to 4x | via TensorOps/Metal 4 |

**Key insight:** Token generation is memory-bandwidth-bound, not compute-bound. M5's 153 GB/s is the bottleneck.

### 3.3 Speculative Decoding on Apple Silicon

**Recurrent Drafter** (Apple ML Research):
- Up to 2.3x speedup on Metal GPUs
- Small draft model (48.77M params for 3.18B base) predicts next tokens
- Verification batch: check N draft tokens in parallel

**Apple Intelligence Foundation Model**:
- 3.18B base + 48.77M drafter + rank-32 LoRA adapters
- Joint LoRA + draft model training (Eagle_LoRA framework, arXiv:2512.22455)
- Higher accepted token rate than naive speculative decoding

### 3.4 Ternary MatMul Optimization

**BitNet b1.58 inference** (arXiv:2504.12285, arXiv:2502.11880):
- Ternary weights {-1, 0, 1} packed: 4 values per int8
- MatMul becomes addition/subtraction only (no actual multiplication)
- **bitnet.cpp**: On Apple M2 CPU: 45 tok/s for BitNet-2B-4T
- 15x more energy-efficient than FP16 LLaMA-3-8B on same hardware
- Full model fits in 1.2 GB RAM

### 3.5 Theoretical Maximum Throughput for BitNet-2B on M5 Pro

**Calculation:**

```
Model size: ~0.6 GB (ternary packed)
M5 Pro memory bandwidth: ~200 GB/s (estimated, Apple hasn't published exact M5 Pro specs)
Per-token memory read: ~0.6 GB (must read all weights for each generated token)

Theoretical max tok/s = bandwidth / model_size
                      = 200 GB/s / 0.6 GB
                      = ~333 tok/s (memory-bandwidth ceiling)
```

**Practical estimates:**
- bitnet.cpp on M2: 45 tok/s (CPU only, no GPU acceleration)
- MLX with mx.compile + Metal: likely 150-250 tok/s on M5 Pro (based on 1.19-1.27x M5/M4 scaling and M4 benchmarks showing ~200 tok/s for similar model sizes)
- With speculative decoding (2.3x): 345-575 tok/s theoretical
- **>100 tok/s is achievable** on M-series for BitNet-2B

**vllm-mlx benchmarks:**
- 525 tok/s on Qwen3-0.6B (smaller model)
- 21-87% higher throughput than llama.cpp across 0.6B-30B range

### 3.6 Continuous Batching with Adapters

Not yet benchmarked specifically for multi-LoRA on MLX. vLLM supports continuous batching with LoRA but runs on CUDA. vllm-mlx (arXiv:2601.19139) brings vLLM's scheduling to Apple Silicon but multi-LoRA support on MLX is nascent.

---

## 4. Pointer-Based / Reference-Based Routing for Expert Selection

### 4.1 Hash Layers (arXiv:2106.04426, Roller & Sukhbaatar 2021)

**The foundational zero-merge approach:**
- Tokens routed to fixed expert modules via hash function
- No routing parameters, no load balancing loss, no learned gates
- Hash computed from token ID (most local feature works best)
- **Balanced hash**: pre-computed lookup table built from training data distribution
  - Greedily assign most frequent tokens to emptiest buckets
  - O(1) routing at inference -- simple table lookup
- **Results:** Competitive with or outperforms Switch Transformers and BASE Layers
- **Key finding:** Random balanced hashes > learned clusters for routing

### 4.2 Approaches That Avoid Weight Merging

**The SOLE architecture (from this project's hash-ring routing):**
- Hash-ring assigns each input to exactly one pre-computed adapter
- No weight merging at inference time -- pure index lookup
- 5.3% displacement at N=20 (from macro/hash_routing_scale/)
- O(1) routing decision, zero gradient computation for routing

**SiDA-MoE:**
- Offline-trained hash function reduces expert selection overhead
- Significant perplexity cost vs learned routing
- Tradeoff: deterministic O(1) routing vs quality

**Grouter** (arXiv:2603.06626):
- Decouples routing from representation
- Routes in a separate embedding space, not from token hidden states
- Accelerates MoE training by avoiding routing-representation entanglement

**Expert-Token Resonance (ETR)**:
- Bidirectional: tokens choose experts AND experts choose tokens
- Still uses learned routing, but the selection is pointer-like (hard assignment)

### 4.3 Zero-Merge Composition Taxonomy

| Approach | Routing | Merge? | Overhead | Quality |
|----------|---------|--------|----------|---------|
| Hash Layers | Hash table lookup | No | O(1) | Competitive with learned |
| SOLE Hash-Ring | Consistent hash | No | O(1) | Supported at N=50 |
| Switch Transformer | Learned top-1 | No (output avg) | O(d * N_experts) | Best sparse scaling |
| MoE top-k | Learned top-k | No (output avg) | O(d * N_experts) | Standard |
| Task Arithmetic | None (static) | Yes | O(d^2 * L) | Degrades with N |
| LoRA Soup | None (static) | Yes | O(d^2 * L) | OK for similar tasks |

**The critical distinction:** MoE/hash routing composes in **output space** (weighted average of function outputs), while LoRA merging composes in **parameter space** (weighted average of weight deltas). Output-space composition preserves each expert's internal nonlinear computation. Parameter-space composition destroys it.

---

## 5. Falcon-Edge 3B (Falcon-E-3B) Architecture and LoRA Support

### 5.1 Architecture

| Property | Falcon-E-3B | BitNet-2B-4T |
|----------|-------------|--------------|
| Parameters | 3B | 2B |
| Weight precision | Ternary (1.58-bit) | Ternary (1.58-bit) |
| Architecture basis | Llama (modified BitNet) | Custom BitNet |
| Training data | ~1.5T tokens (TII internal) | 4T tokens |
| Vocab size | 32,678 | 32,000 |
| Memory footprint | 999 MB | 1.2 GB (with bitnet.cpp) |
| Normalization | RMSNorm (pre-attention, pre-MLP; **removed** within BitNet layers) | LayerNorm within BitNet |
| Attention | GQA (inferred from Falcon3-3B: 12 query heads, 4 KV heads, head_dim=256) | Standard MHA |
| MLP | SwiGLU (BitNet-ternary variant) | BitNet FFN |
| Quantization formula | `scale = 1.0 / w.abs().mean()`, clamp to [-1,1], divide by scale | Standard BitNet STE |

**Key architectural difference from standard BitNet:** Falcon-Edge eliminates LayerNorm layers WITHIN BitNet linear layers while retaining pre-attention and pre-MLP norms. This ensures Llama-compatible architecture for tooling interoperability.

### 5.2 LoRA Fine-tuning Support

**Current status: Full fine-tuning only.**

From the official blog: "Currently, only full-finetuning is supported through this framework." PEFT methods for BitNet remain "an exciting and impactful open question."

**Fine-tuning workflow:**
1. Load prequantized checkpoint: `revision="prequantized"`
2. Replace linear layers: `replace_linear_with_bitnet_linear(model)`
3. Full fine-tune with standard HF/trl

**Can LoRA work on Falcon-E-3B?** In principle yes -- the architecture is Llama-compatible, so standard LoRA can be applied to the bfloat16 variant. For the ternary variant, the same challenges apply as with BitNet-2B-4T: LoRA's A and B matrices are FP16/BF16, while base weights are ternary. The project has demonstrated this works on BitNet-2B (see FINDINGS.md: N=50 ternary composition experiments). **Falcon-E-3B should be equally viable for LoRA composition.**

### 5.3 Benchmarks vs BitNet-2B-4T

| Benchmark Suite | Falcon-E-3B-Instruct | BitNet-b1.58-2B-4T |
|-----------------|---------------------|---------------------|
| Leaderboard v1 avg | **53.17** | 51.54 |
| Memory | **999 MB** | 1.2 GB |
| Training tokens | 1.5T | **4T** |

Falcon-E-3B scores higher on aggregate benchmarks despite 2.7x fewer training tokens, likely due to the 50% more parameters (3B vs 2B).

### 5.4 Composition Experiment Implications

**For adapter composition, Falcon-E-3B has several advantages over BitNet-2B-4T:**
1. **Smaller memory footprint** (999MB vs 1.2GB) -- more room for adapters and KV cache
2. **Llama-compatible architecture** -- better tooling support (HF, mlx_lm, vLLM)
3. **GQA** -- 3x fewer KV heads than MHA, smaller KV cache
4. **Larger model** -- more capacity for absorbing adapter deltas without interference

**Disadvantages:**
1. **Less tested** for LoRA composition (BitNet-2B has N=50 proven in this project)
2. **No PEFT support documented** (would need custom implementation)
3. **Fewer training tokens** (might have weaker base knowledge)

---

## Paper Index

| Paper | arXiv ID | Key Contribution |
|-------|----------|-----------------|
| Switch Transformers | 2101.03961 | k=1 sparse MoE, capacity factor |
| Hash Layers | 2106.04426 | Deterministic hash routing, no learned gates |
| S-LoRA | 2311.03285 | Unified paging for thousands of concurrent LoRAs |
| Mixtral 8x7B | 2401.04088 | Top-2 routing, 8 expert FFNs per layer |
| X-LoRA | 2402.07148 | Dynamic layer/token-level LoRA scaling |
| CLA | 2405.12981 | Cross-layer KV cache sharing, 2x reduction |
| Compress then Serve | 2407.00066 | Joint LoRA compression via shared basis |
| Auxiliary-Loss-Free Balancing | 2408.15664 | Bias-term load balancing (used in DeepSeek-V3) |
| LoRA Soups | 2410.13025 | Empirical LoRA merging for skill composition |
| BitNet b1.58 2B4T | 2504.12285 | First open 1.58-bit 2B LLM |
| bitnet.cpp | 2502.11880 | Efficient ternary inference on CPUs |
| DeepSeek-V3 | 2412.19437 | Fine-grained MoE, MLA, auxiliary-loss-free |
| LoRI | 2504.07448 | Sparse B matrices for interference reduction |
| OSRM | 2505.22934 | Orthogonal subspaces for robust merging |
| Ortho-LoRA | 2601.09684 | Orthogonal gradient projection for multi-task LoRA |
| Cross-Model KV Reuse | 2512.17910 | KV cache sharing between base/adapted models |
| vllm-mlx | 2601.19139 | vLLM on Apple Silicon |
| Recurrent Drafter | 2403.09919 | Speculative decoding draft model |
| Grouter | 2603.06626 | Decoupled routing from representation |
| Apple Intelligence FM | 2507.13575 | LoRA + speculative decoding on-device |
| DR-LoRA | 2601.04823 | Dynamic rank LoRA for MoE adaptation |
| CoMoL | 2603.00573 | Core space merging for efficient MoE LoRA |
