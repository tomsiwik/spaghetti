# MATH.md: SHINE S1 — Memory Token Extraction on Gemma 4 E4B

## Grounding

**Paper:** SHINE (arXiv:2602.06358), Section 3.2 — Memory Token Extraction.
Appending M learnable token embeddings to the input sequence and collecting
per-layer hidden states is the standard context-encoding step in hypernetwork
architectures (also used in Perceiver, Set Transformer).

**Prior finding:** PORT_PLAN.md documents the Qwen3 reference implementation.
Gemma 4 E4B has architectural differences (per-layer input gating, shared KV,
sliding window) that require adaptation.

## Mechanism (Atomic Level)

### Setup

Let the frozen Gemma 4 E4B model have:
- L = 42 layers (35 sliding-window + 7 full-attention)
- d = 2560 (hidden dimension)
- Vocabulary embedding: E: V -> R^d (quantized)

We introduce M learnable memory token embeddings:

  Z in R^{M x d}   (initialized from N(0, 0.02))

### Forward Pass

Given input token IDs x = [x_1, ..., x_T]:

1. **Embed:** h^0 = [E(x_1), ..., E(x_T), z_1, ..., z_M] * sqrt(d)
   - Shape: (T+M, d)

2. **Per-layer input:** Gemma 4 E4B uses per-layer embeddings E_pl: V -> R^{42*256}.
   Memory tokens have no token IDs, so we derive per-layer inputs by projecting
   through the per_layer_model_projection path (operates on embeddings, not IDs).
   For memory tokens: per_layer_input_i = proj(z_i * sqrt(d)), same path as when
   input_embeddings are provided without input_ids.

3. **Layer processing:** For each layer l in [0, 41]:
   - h^l = DecoderLayer_l(h^{l-1}, mask, cache, per_layer_input_l)
   - Extract: M_l = h^l[-M:]  (last M positions)

4. **Output:** Memory states tensor S in R^{L x M x d}
   - S[l, :, :] = M_l for each layer l

### Attention Pattern for Memory Tokens

- **Sliding-window layers (35/42):** Window size W=512. Memory tokens at positions
  [T, T+M-1] attend to positions [max(0, T+j-W+1), T+j] for token j.
  As long as T+M <= T+W (i.e., M <= 512), all memory tokens see context tokens
  within their window. With M=32 and typical T~512-1024, memory tokens see
  the last ~480 context tokens.

- **Full-attention layers (7/42):** No window restriction. Memory tokens attend
  to ALL context tokens. These layers provide the global context signal.

### Non-Degeneracy Condition

**Theorem:** If the frozen model layers apply distinct transformations per layer
(different weight matrices), and the input sequence contains information (not
all-zero or constant), then the memory states across layers will NOT be
degenerate (not all identical).

**Proof sketch:** Each DecoderLayer applies self-attention (position-dependent)
followed by MLP with different learned weights. The residual stream accumulates
different features at each layer. Since memory tokens attend to different context
windows at different layers and the attention/MLP weights differ, the extracted
states S[l] != S[l'] for l != l' in general.

**Quantitative prediction:** For non-trivial input, the mean pairwise cosine
similarity between layer representations should satisfy:

  mean_l!=l' cos(S[l], S[l']) < 0.95

This threshold (K2) ensures the memory states carry layer-specific information,
not just a copy of the input embedding.

## Predictions

| ID | Prediction | Kill threshold |
|----|-----------|---------------|
| K1252 | Memory states shape = (42, M, 2560) | Exact match required |
| K1253 | Mean cross-layer cosine < 0.95 | >= 0.95 means degenerate |
| K1254 | Extraction latency < 500ms for T=1024 on M5 Pro | >= 500ms too slow for session use |

### Additional quantitative predictions:
- Memory norms should be O(1) after RMSNorm residual stream (expect 1.0-10.0 range)
- Cross-layer cosine should show structure: nearby layers more similar than distant
  (expect cosine gradient from ~0.8 at distance=1 to ~0.5 at distance=40)
- Full-attention layers (indices 5,11,17,23,29,35,41) may show lower cosine with
  sliding layers due to global vs. local context aggregation
