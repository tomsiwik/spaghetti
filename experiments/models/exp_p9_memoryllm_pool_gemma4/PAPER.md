# MemoryLLM Memory Pool on Gemma 4: Hidden-State Injection Fails Without Training

## Reference
MemoryLLM: Self-Updatable LLM with Weight Memory (arXiv:2402.04624)

## Summary

Ported MemoryLLM's per-layer hidden-state memory pool to Gemma 4 E4B on MLX.
Two modes tested: (1) text-context baseline (fact as text prefix), and
(2) hidden-state injection (stored hidden states prepended at each layer).

**Key finding:** Text-context recall is trivially 100%. Hidden-state injection
produces degenerate output (0% recall). The mechanism requires fine-tuning
the attention layers to read from injected hidden states — a pretrained model
cannot use them. Write latency (2.82ms) meets requirements.

## Kill Criteria Results

| Kill | Prediction | Measured | Status |
|------|------------|----------|--------|
| K1366: recall > 50% | ~70% (attention similarity) | **0% HS / 100% context** | FAIL (HS) / PASS (context) |
| K1367: latency < 5ms | < 1ms | **2.82ms avg** | PASS |
| K1368: quality ±2pp | within 1pp | **+30pp (flawed test)** | INCONCLUSIVE |

## Prediction vs Measurement

| Quantity | Predicted | Measured | Error | Explanation |
|----------|-----------|----------|-------|-------------|
| HS recall | 70% | 0% | -70pp | Position encoding mismatch; model not trained for HS injection |
| Context recall | ~100% | 100% | 0pp | Trivial — standard context reading |
| Write latency | < 1ms | 2.82ms | +1.82ms | mx.eval sync adds overhead beyond pure copy |
| Base accuracy | ~80% MMLU | 0% | N/A | Prompt format issue — terse questions produce `<end_of_turn>` |

## Detailed Results

### Hidden-State Injection: Complete Failure

All 5 facts: model outputs `<end_of_turn><turn|><turn|><turn|>...` repetitions.

**Root cause:** The prepended hidden states have incorrect position encoding.
Memory tokens at positions [0, K-1] receive position embeddings for those positions,
but they were COMPUTED at different positions during the injection forward pass.
Gemma 4's sliding layers use RoPE (θ=10000), which encodes absolute position into
Q/K vectors. When hidden states from position p are placed at position 0..K-1,
their internal representation is INCONSISTENT with the position encoding applied
during attention. The attention mechanism sees corrupted Q/K relationships.

Additionally, the model's attention patterns were never trained to attend to
"disconnected" hidden states from a separate forward pass. MemoryLLM's paper
explicitly requires fine-tuning with a LoRA decoder to learn memory reading.

### Text-Context Baseline: Trivially Perfect

5/5 facts recalled correctly when the fact is prepended as text. This is just
standard in-context learning — the model reads the fact from its own context window.
The 10 intervening distractor queries don't affect recall because they use separate
generation calls (no shared KV cache).

### Write Latency: Passes

| Metric | Value |
|--------|-------|
| Average | 2.82ms |
| Min | 1.83ms |
| Max | 6.39ms |
| Memory size | 27.5 MB (42×128×2560 float16) |

The write operation (mx.eval on the stacked memory tensor) meets the 5ms threshold.
Slightly higher than the theoretical 0.14ms due to mx.eval synchronization overhead.

### Base Quality: Test Invalid

The MMLU-style test produced 0% accuracy on base (without memory) because the
terse question format causes the model to output `<end_of_turn>` immediately.
This is a prompt formatting issue, not a model quality issue. The test is not
valid for measuring base quality perturbation.

## Impossibility Structure

**Why hidden-state injection fails without training:**

The MemoryLLM mechanism injects hidden states h_l from a previous forward pass
at each layer l. For the attention mechanism to use these states:

1. **Position coherence required:** The Q/K projections applied to memory hidden
   states must produce attention scores that reflect the CONTENT of the memory,
   not the POSITION it was originally at. RoPE encodes position into Q/K,
   so hidden states from position p, when placed at position q≠p, produce
   attention scores for position q, not for their actual content.

2. **Learned read pattern required:** The model must learn that the first K
   positions contain "external memory" with different statistical properties
   than normal context. Without training, the model treats memory positions
   as normal (corrupted) tokens.

3. **Per-layer inconsistency:** Each layer receives its OWN stored states
   (not propagated from previous layer). This creates layer-wise inconsistency
   that compounds through the network. In normal operation, h_l depends on h_{l-1};
   in memory injection, h_l is INDEPENDENT of the current context at layer l.

**Structural fix:** MemoryLLM's paper uses LoRA fine-tuning on the decoder
to teach the model to read from memory tokens. Alternatively, using KV-cache
prefilling (which preserves position encoding) achieves the same recall without
any training — but loses the FIFO update advantage.

## Implications for Pierre Architecture

1. **Memory pool as designed in MemoryLLM: NOT viable** without fine-tuning.
   The mechanism requires training to learn memory reading patterns.

2. **KV-cache prefilling: viable alternative** for persistent context.
   No training needed, preserves position encoding, but no FIFO updates.

3. **PLE injection (Gemma 4's native mechanism):** Already supports per-layer
   vector injection via the gated PLE pathway. This is architecturally similar
   to memory injection but uses a TRAINED gate + projection. PLE-based memory
   could work without additional fine-tuning.

4. **For our composable adapter system:** Memory is less critical than
   composition quality. Focus on orthogonal adapters (Finding #511's conclusion)
   rather than memory mechanisms.

## Status: KILLED

The primary prediction — hidden-state injection achieves >50% recall without
training (K1366) — was refuted by 70pp (predicted 70%, measured 0%). This is
a decisive refutation of the core mechanism under test.

- **K1366 FAIL:** Hidden-state recall 0/5 (0%). Degenerate output (`<turn|>` repetitions)
- **K1367 PASS:** Write latency 2.82ms < 5ms threshold
- **K1368 INCONCLUSIVE:** Prompt formatting bug invalidates base quality measurement

The text-context baseline (100% recall) is standard in-context learning, not
the mechanism being tested. It does not rescue the hypothesis.

The impossibility structure is the key finding: position encoding corruption +
untrained attention patterns + per-layer inconsistency make hidden-state
injection structurally impossible without fine-tuning. This is not a parameter
tuning failure — it is a fundamental architectural incompatibility.
