# MATH.md — T4.3v2: MLX Adapter Serving (Loophole Fix)

## Background

T4.3 (Finding in exp_p1_t4_vllm_adapter_serving) had three critical loopholes identified
by LOOPHOLE_AUDIT:
1. **Swap latency** excluded MLX graph recompilation (first forward pass after weight swap)
2. **Throughput** mixed prefill+decode, biased by out-of-domain adapter
3. **Routing latency** used dict lookup instead of real TF-IDF router

This experiment fixes all three to produce valid serving metrics.

## Theorem 1 (MLX Lazy Evaluation and Graph Recompilation)

**Statement:** On MLX, the true adapter hot-swap latency includes:
  (a) Weight loading: `model.load_weights(path, strict=False)` — O(adapter_size)
  (b) Weight materialization: `mx.eval(model.parameters())` — device transfer
  (c) First forward pass: graph trace + execute with new weights

**Derivation:**
MLX uses deferred execution (lazy evaluation). Operations build a computation graph
that is not executed until `mx.eval()` or an implicit trigger (print, item, etc.).

After `model.load_weights()`:
- The new weight tensors are in the computation graph but NOT materialized
- `mx.eval(model.parameters())` forces weight transfer to device (~adapter_size bytes)
- First `generate()` call builds and executes the full computation graph with new weights

For LoRA adapters on Gemma 4 4B at rank=16:
- Adapter size ≈ 2 × d_model × r × n_lora_layers × 2 (v_proj + o_proj)
  = 2 × 5120 × 16 × 28 × 2 ≈ 23M parameters ≈ 46MB at float16
- Weight transfer: 46MB / (memory bandwidth ≈ 540 GB/s) ≈ 0.1ms
- Graph trace overhead: O(n_layers) ≈ 5-20ms for 28 layers
- Total expected: 10-50ms well under 100ms threshold

**Claim:** True hot-swap latency (weight load + first forward pass, 1 token) < 100ms.

## Theorem 2 (Decode-Only Throughput Isolation)

**Statement:** Decode-only throughput (tokens/sec after prefill) is the correct metric
for LoRA serving overhead, because:
1. Prefill is a parallelizable batch operation; decode is sequential
2. LoRA adds: for each layer, 2 extra matmuls of shape [batch, seq, r] → [batch, seq, d]
   where r << d_model (rank-16 << 5120)
3. Decode overhead per token: O(r × d_model) per LoRA layer = O(r/d_model) relative cost

**Expected overhead:**
LoRA adds per decode step per layer: 2 matmuls of (1 × 5120 × 16) = 163,840 FLOPs
Base model per decode step per layer: ~2 × 5120 × 5120 = 52M FLOPs (attention + FFN)
Relative overhead: 2 × 163K / 52M ≈ 0.6% per layer ≈ 0.6% total for 28 layers

Therefore: decode throughput degradation << 15% threshold.

**Measurement:** Use `stream_generate()` final response's `generation_tps` field (decode-only tps).
Compare base model vs LoRA model on same in-domain prompt.

## Theorem 3 (TF-IDF Routing Latency)

**Statement:** A TF-IDF + Ridge classifier over N=5 domains runs in < 5ms per query on CPU.

**Derivation:**
TF-IDF feature extraction: O(V × D) where V = vocabulary size (5000), D = document length
Ridge prediction: O(C × V) where C = classes (5), V = 5000

For a 50-word prompt:
- TF-IDF vectorize: ~50 word lookups in a 5000-word vocabulary ≈ O(0.1ms)
- Ridge predict: 5 × 5000 dot product ≈ O(0.1ms)
- Total: << 5ms threshold

Prior evidence: Finding #474 (P4.A0) measured TF-IDF+Ridge at 0.247ms p99.
K1242 threshold (5ms) is 20× above prior measurement — conservative threshold.

## Kill Criteria Derivations

**K1240 (< 100ms):** From Theorem 1, expected 10-50ms. PASS expected.
**K1241 (< 15% degradation):** From Theorem 2, expected ~0.6% overhead per layer. PASS expected.
**K1242 (< 5ms routing):** From Theorem 3, expected ~0.25ms. PASS expected.

## What Would KILL This Experiment

**If K1240 FAILS (swap+first-forward > 100ms):**
MLX graph recompilation is O(model_size) not O(adapter_size). Fix: keep base graph frozen,
only recompile adapter subgraph (requires mlx.compile with adapter weights as state).

**If K1241 FAILS (throughput degradation > 15%):**
Memory bandwidth bottleneck: loading LoRA weights into cache for each decode step causes
cache pressure beyond the 15% threshold. Fix: fuse LoRA into base weights at serve time
(adapter merging, as in Finding #425 / Room Model).

**If K1242 FAILS (routing > 5ms):**
Sklearn TF-IDF on CPU is slower than expected for our prompts. Fix: precompute TF-IDF
matrix in numpy, use dot product directly (bypasses sklearn overhead).

## References

1. LOOPHOLE_CODE.md (exp_p1_t4_vllm_adapter_serving): Three original loopholes identified
2. Finding #474 (P4.A0): TF-IDF+Ridge routing 0.247ms p99 at N=5
3. MLX lazy evaluation model: https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
4. Hu et al. (2021) arxiv 2106.09685 LoRA: rank-r overhead analysis Section 4
5. Finding #432 (T4.3): Original serving experiment — swap 3.6ms p99 (without first forward)
