# SHINE M2P → MLX Port Plan

## What SHINE Actually Is

SHINE (arXiv:2602.06358) is a **context-to-parameter** hypernetwork. It does NOT replace
fine-tuning — it converts a document/context into temporary LoRA weights so the LLM
can answer questions about that context WITHOUT the context in the prompt.

**Pierre use case:** Tier 4 session adapters. User provides a document → SHINE generates
LoRA weights in one forward pass → LLM answers questions about that document.

We may have been using M2P wrong: treating it as a replacement for SFT (domain adapter
generation) when it's designed for context-conditioning (session adapters).

## Architecture (from source code)

### 1. Memory Extraction (LoraQwen.py)
- M learnable memory embeddings appended to context tokens
- Fed through frozen LLM with "Meta LoRA" (rank 128) 
- At each layer, memory token hidden states extracted
- Output: `(batch, num_layers, num_mem_tokens, hidden_size)`

### 2. M2P Transformer (metanetwork_family.py: MetanetworkTransformer)
- Input: memory tensor + layer PE + token PE
- Alternating attention:
  - Even layers: cross-layer (column) attention — each memory token attends across all layers
  - Odd layers: within-layer (row) attention — tokens within each layer attend to each other
- Couple layers: optional secondary attention block
- Output: flat tensor reshaped into LoRA A, B matrices

### 3. LoRA Generation (LoraQwen.py: LoraLinear.generate_lora_dict)
- Flat output from M2P split into per-layer, per-module chunks
- Each chunk reshaped into A (d_in, r) and B (r, d_out) matrices
- Applied to ALL linear layers: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Scale: sqrt(scale) applied to both A and B

### 4. Training (meta_train_parallel.py)
- Reconstruction loss: LLM with generated LoRA must reproduce the context
- Completion loss: LLM with generated LoRA must complete truncated context
- Meta LoRA (rank 128) is always active during context encoding
- Generated LoRA (rank 4-8) is used for reconstruction/QA

## Port Strategy: PyTorch → MLX

### Step 1: LoraLinear → MLX LoraLinear
- Replace `nn.Linear` with `nn.Module` wrapping MLX QuantizedLinear
- Replace `torch.matmul` with `@` operator
- Handle batched LoRA dict application
- Key shapes: A=(batch, d_in, r), B=(batch, r, d_out)

### Step 2: MetanetworkTransformer → MLX
- Replace `nn.TransformerEncoderLayer` with custom MLX multi-head attention + FFN
- Replace `nn.Parameter` with `mx.array` tracked by nn.Module
- Port alternating row/column attention pattern
- Port couple layers (secondary attention)

### Step 3: LoraQwen3Model → MLX Gemma4 wrapper
- Replace Qwen3 decoder layers with Gemma 4 decoder layers
- Adapt for Gemma 4 specifics: K=V on global layers, p-RoPE, QK-norm
- Port memory token injection and extraction
- Handle 35 local + 7 global layer pattern

### Step 4: Metanetwork wrapper (metanetwork_family.py: Metanetwork)
- Forward: encode context → extract memory → M2P transform → generate LoRA dicts
- Training: reconstruction + completion loss

### Step 5: Training loop
- Port self-supervised pretraining (reconstruction + completion)
- Adapt for Gemma 4 chat template
- MLX optimizer and gradient computation

## Key Differences: Qwen3 → Gemma 4

| Aspect | Qwen3 (SHINE) | Gemma 4 (Pierre) |
|--------|---------------|-------------------|
| Layers | 36 (uniform) | 42 (35 local + 7 global) |
| Attention | Standard MHA | QK-norm, V-norm, K=V on global |
| Head dim | Uniform | 256 local, 512 global |
| RoPE | Standard | Proportional (25%) on global |
| Quantization | BF16 | 4-bit (QuantizedLinear) |

## Files Copied

- `reference_pytorch/metanetwork_family.py` — M2P Transformer
- `reference_pytorch/LoraQwen.py` — LoRA-wrapped Qwen3 model  
- `reference_pytorch/meta_train_parallel.py` — Training loop
- `reference_pytorch/utils/` — Utilities
- `reference_pytorch/*.yaml` — Configs

## Critical Realization

The centroid trap (Finding #345) killed M2P when we used it for **multi-domain adapter
generation** (generate different B-matrices per domain from a single M2P). But SHINE
is designed for **single-context conditioning** — each context gets its own unique
LoRA weights. There is no multi-domain collapse because there's only one context per
forward pass.

**This means M2P might work perfectly for its intended use case (session adapters)
even though it fails for domain adapter generation.** We need to test it correctly:
give it a document, generate LoRA weights, test if the LLM can answer questions about
that document without the document in the prompt.
