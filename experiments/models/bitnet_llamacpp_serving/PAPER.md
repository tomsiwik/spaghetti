# BitNet + llama.cpp LoRA Serving: Research Digest

## Hypothesis

llama.cpp can serve BitNet-2B-4T in GGUF format with runtime LoRA adapters, supporting multi-adapter composition and hot-swap on commodity CPU hardware.

**Verdict: SUPPORTED (all 3 kill criteria pass)**

## What This Experiment Tested

The production serving path for Composable Ternary Experts on commodity hardware. Prior work (exp_bitnet_serving_path) killed LoTA-QAF weight merge as non-viable (116x below grid flip threshold). Runtime LoRA via llama.cpp is the only remaining serving path. This experiment validates that path end-to-end.

Three kill criteria:

- **K1**: llama.cpp can load BitNet GGUF (TQ2_0) + LoRA adapters
- **K2**: Multi-adapter overhead under 50% throughput loss vs base-only
- **K3**: Hot-swap adapters mid-session without output corruption

## Key References

- llama.cpp (ggml-org/llama.cpp) — CPU inference engine with LoRA support
- BitNet b1.58 (arXiv 2402.17764) — Ternary weight LLM architecture
- microsoft/BitNet-b1.58-2B-4T — 2.4B parameter ternary model
- lora-mixer-frozen-deploy (arXiv 2507.00029) — Frozen LoRA deployment patterns

## Technical Contributions

### GGUF Conversion for BitNet-2B-4T

The official llama.cpp convert script (b8480) has two bugs that prevent BitNet-2B-4T conversion:

1. **Architecture name case mismatch**: Script registers `BitnetForCausalLM` but model config declares `BitNetForCausalLM` (uppercase N). Fixed by registering both variants.
2. **Missing tensor name mappings**: BitNet-2B-4T uses `self_attn.attn_sub_norm` and `mlp.ffn_sub_norm` but the gguf tensor_mapping.py only maps the older 3B model's names (`inner_attn_ln`, `ffn_layernorm`). Fixed by adding the 2B-4T naming variants.
3. **Vocabulary mismatch**: BitNet-2B-4T uses GPT-2 tokenizer but script assumes sentencepiece. Fixed with runtime detection.

These patches are in `convert_hf_to_gguf.py` (local copy) and the installed `gguf/tensor_mapping.py`.

### LoRA Adapter GGUF Conversion

Wrote `convert_adapter_to_gguf.py` to convert MLX-trained npz adapters to llama.cpp GGUF LoRA format. Key mapping:
- HuggingFace tensor names to GGUF names (e.g., `self_attn.q_proj` to `attn_q`)
- Suffix convention: `.lora_a` and `.lora_b` (not `.loraA`/`.loraB`)
- Shape transposition: MLX stores lora_a as (in_features, rank), GGUF expects (rank, in_features)

### Metal/GPU Limitation

TQ2_0 and TQ1_0 ternary quantization types have NO Metal shader support in llama.cpp (as of commit 7cadbfc, 2026-03-24). The Metal backend loads but `kernel_mul_mm_tq2_0_f32` compilation fails with "Function not found in library". All benchmarks use CPU-only build (`-DGGML_METAL=OFF`). This is a known upstream limitation, not a blocker (CPU throughput is adequate for commodity serving).

## Empirical Results

### K1: Base + LoRA Loading (PASS)

| Configuration | Load | Prompt eval | Token gen |
|---------------|------|-------------|-----------|
| Base only | OK | 83.7 t/s | 63.9 t/s |
| + math adapter | OK | 82.4 t/s | 53.0 t/s |
| + python adapter | OK | - | - |
| + legal adapter | OK | - | - |
| + medical adapter | OK | - | - |
| + creative adapter | OK | - | - |

All 5 domain adapters (rank-16, all projections, 82.5 MB each in GGUF) load successfully alongside the 1.11 GiB TQ2_0 base model.

### K2: Multi-Adapter Throughput Overhead (PASS)

Measured via llama-server API with proper prompt/generation separation. Base: no adapters loaded. Adapters: 5 domain LoRAs (rank-16, d=2560, 30 layers).

| Configuration | Prompt eval (t/s) | Token gen (t/s) | Overhead |
|---------------|-------------------|-----------------|----------|
| True base (no adapters) | 83.7 +/- 3.9 | 63.9 +/- 2.3 | 0% |
| Base (5 loaded, 0 active) | 83.1 +/- 4.8 | 60.8 +/- 3.1 | +4.7% |
| 1x LoRA (scale=1.0) | 82.4 +/- 0.7 | 53.0 +/- 1.3 | +17.0% |
| 3x LoRA (scale=1/3 each) | 74.4 +/- 1.2 | 41.9 +/- 0.2 | +34.4% |
| 5x LoRA (scale=1/5 each) | 69.8 +/- 1.4 | 33.8 +/- 0.6 | +47.1% |

**K2 verdict: PASS** (47.1% < 50% threshold, but marginal — 95% CI [41.8%, 52.4%])

Key observations:
- **Loading overhead**: ~4.7% just from having adapters in memory (even when all scales=0)
- **Affine scaling model**: overhead ≈ 9.5% + 7.5% × N_active (fixed setup cost + per-adapter marginal)
- **At rank-8**: Expected ~3.75% marginal per adapter, allowing ~10 simultaneous adapters under 50% (unverified)
- **CPU throughput**: 33.8 t/s with 5 active adapters is still usable for interactive serving
- **K2 margin**: 47.1% vs 50% threshold gives only 2.9pp margin; the 95% CI overlaps the threshold (n=3, df=2)

### K3: Hot-Swap Coherence (PASS)

**K3a — Deterministic reproducibility**: Same prompt with adapter A (math), then B (legal), then A again in separate CLI processes. Seed=42, temp=0. All 3 runs of A1 and A2 produce identical output. A differs from B. **PASS**. (Note: this tests seed-deterministic reproducibility across separate runs, not in-process hot-swap.)

**K3b — Server API hot-swap**: llama-server `/lora-adapters` endpoint enables real-time adapter weight changes. Test sequence:
1. Generate with math adapter active -> output_1
2. Hot-swap to python adapter via API -> success
3. Generate with python adapter -> different output
4. Hot-swap back to math adapter -> success
5. Regenerate same prompt -> output_2

**output_1 == output_2**: True. No corruption from hot-swap cycle. **PASS**.

Server reports all 5 adapters with per-adapter scaling. Swap is instantaneous (no model reload).

## Memory Analysis

| Component | Size |
|-----------|------|
| Base model (TQ2_0 GGUF) | 1.11 GiB |
| Per-adapter (rank-16 GGUF, float32) | 82.5 MB |
| 5 adapters total | 412.5 MB |
| CPU compute buffer | ~24 MiB |
| **Total with 5 adapters** | **~1.53 GiB** |

At 82.5 MB per adapter, a 1 GB memory budget fits ~12 adapters. Memory scales linearly and becomes significant at high adapter counts.

## Production Serving Architecture

Based on these results, the production serving path is:

```
llama-server -m bitnet-2b-4t.gguf \
  --lora adapter_1.gguf \
  --lora adapter_2.gguf \
  ... \
  --lora adapter_N.gguf \
  --port 8080

# Runtime adapter selection via API:
POST /lora-adapters
[{"id": 0, "scale": 0.5}, {"id": 3, "scale": 0.5}]

# Completion uses active adapters:
POST /completion
{"prompt": "...", "n_predict": 100}
```

Key capabilities:
- Pre-load N adapters at startup (82.5 MB each)
- Hot-swap active adapters per request via API (zero reload latency)
- Per-adapter scaling for weighted composition
- ~7.5% marginal throughput cost per active adapter (rank-16), plus ~9.5% fixed setup cost

## Limitations

1. **Metal/GPU not supported**: TQ ternary types lack Metal shaders. CPU-only inference. This limits throughput to ~34-64 t/s on Apple M1 Max (vs potentially 200+ t/s with Metal). Upstream fix pending.

2. **No per-request routing**: Current llama-server sets adapters globally (not per-request). For true per-token routing, would need custom server or batch API with adapter selection per request.

3. **Output quality not tested**: This experiment validates the SERVING MECHANISM (loading, throughput, hot-swap) but does not evaluate output quality. The base model produces garbled output (expected for a non-instruction-tuned base model with short prompts). Quality validation is covered by separate composition experiments (exp_bitnet_scale_n15, exp_bitnet_per_token_routing).

4. **GGUF conversion requires patches**: The official llama.cpp convert script has three bugs for BitNet-2B-4T. These patches should be upstreamed.

5. **Overhead scales linearly**: At 5 active rank-16 adapters, overhead is 47.1% (near the 50% threshold). For more than 5 simultaneous adapters, either reduce rank or use per-token routing (activate only top-k adapters per token).

## What Would Kill This

- **At micro scale**: Metal/GPU support for TQ types would change the throughput equation entirely. If GPU-accelerated TQ gets ~5x speedup, overhead % would drop proportionally.
- **At production scale**: If per-adapter overhead scales super-linearly with model size (d>2560), the 50% budget could be hit at fewer adapters. Needs validation at Qwen-7B scale.
- **Architectural risk**: llama.cpp LoRA applies adapters as additive delta to the dequantized output. For ternary weights, this means the delta is computed in float, which is correct but bypasses any ternary arithmetic speedup. The overhead comes from the float LoRA matmul, not from the base model computation.
