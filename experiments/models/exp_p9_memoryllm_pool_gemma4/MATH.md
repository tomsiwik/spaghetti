# Memory Pool for Gemma 4: Mathematical Foundation

## Reference
MemoryLLM: Self-Updatable LLM with Weight Memory (arXiv:2402.04624)

## Setup

Gemma 4 E4B: L=42 layers, d=2560 hidden dim, sliding_window W=512,
7 global layers (full attention), 35 sliding layers (windowed attention).

Memory pool M ∈ ℝ^{L × K × d} stores hidden states from injection text,
where K = number of memory tokens per layer.

## Mechanism

**Injection phase:** Given injection text x of T_inj tokens, run forward pass.
At each layer l, the hidden state h_l ∈ ℝ^{T_inj × d} is captured.
Store: M[l] = h_l[-K:] (last K hidden states from layer l's output).

**Inference phase:** Given query text of T_q tokens, at each layer l:
1. Prepend memory: h̃_l = [M[l]; h_l] ∈ ℝ^{(K+T_q) × d}
2. Apply attention + FFN normally to h̃_l
3. Extract output: h_{l+1} = output[:, K:, :] (discard memory positions)

Memory tokens provide attention context but their outputs are discarded
per-layer. Each layer reads from its OWN stored states, not propagated states.

## Theorem 1: Memory Visibility Under Sliding Window

**Statement.** For a sliding attention layer with window size W, memory tokens
prepended at positions [0, K-1] are visible to ALL input tokens at positions
[K, K+T_q-1] if and only if K + T_q ≤ W.

**Proof.** Input token at position K+j attends to positions [K+j-W+1, K+j].
Memory token at position 0 is visible when K+j-W+1 ≤ 0, i.e., j ≤ W-K-1.
For the LAST input token (j = T_q-1): visible iff T_q-1 ≤ W-K-1,
i.e., K + T_q ≤ W. ∎

**Prediction:** For K=128, W=512: queries up to T_q=384 tokens see all memory.
For K=256, W=512: queries up to T_q=256 tokens see all memory.
Global layers (full attention) have no window constraint — memory always visible.

We use K=128 to maximize the query length budget within the sliding window.

## Theorem 2: Memory Write Latency Bound

**Statement.** The memory write operation (storing L×K hidden states of
dimension d) requires exactly L·K·d scalar copies. No gradient computation,
no matmul, no synchronization beyond the injection forward pass.

**Proof.** The write operation is:
  M[l] ← h_l[-K:]  for l = 0, ..., L-1

This is a contiguous memory copy from the forward pass output buffer.
Total data: L·K·d·sizeof(float16) bytes.

For E4B: 42 × 128 × 2560 × 2 = 27.5 MB.
At MLX unified memory bandwidth (~200 GB/s): 27.5MB / 200 GB/s = 0.14 ms. ∎

**Prediction:** Write latency < 1ms (well under K1367's 5ms threshold).
The injection FORWARD PASS is separate and costs ~50-100ms.

## Theorem 3: Attention-Based Recall Quality

**Statement.** Let q ∈ ℝ^{d_h} be the query vector for a recall question and
k_fact ∈ ℝ^{d_h} be the key vector of the stored fact in memory. Under softmax
attention with temperature 1/√d_h, the probability of attending to the fact
token is:

  p(fact) = exp(q·k_fact/√d_h) / [exp(q·k_fact/√d_h) + Σ_j exp(q·k_j/√d_h)]

For a fact with cosine similarity cos(q, k_fact) = s and K-1 unrelated memory
tokens with expected similarity ~0 (random directions in d_h-dimensional space):

  p(fact) ≈ exp(s·√d_h) / [exp(s·√d_h) + (K-1)]

For d_h=256 (Gemma 4 head dim) and s=0.3 (moderate similarity):
  p(fact) ≈ exp(0.3·16) / [exp(4.8) + 127] ≈ 121.5 / 248.5 ≈ 0.49

For s=0.4: p(fact) ≈ exp(6.4) / [exp(6.4) + 127] ≈ 602 / 729 ≈ 0.83

**Prediction:** Recall depends on semantic similarity between query and stored
fact. For well-formed factual queries with moderate similarity (s≥0.3), recall
probability per attention head ≥ 49%. With 8 heads and multiple layers (7 global),
ensemble recall should exceed 70%.

**Failure mode:** If the model's pretrained attention patterns don't naturally
match query tokens to prepended hidden states from a different context (position
encoding mismatch), recall could be near zero. This is the primary risk.

## Theorem 4: Base Quality Perturbation Bound

**Statement.** Adding K memory tokens to a sequence of T_q tokens changes the
attention distribution by at most:

  ||α' - α||_1 ≤ 2K/(K+T_q)

where α is the original attention distribution and α' is the memory-augmented one.

**Proof.** By the sensitivity of softmax to additional elements. In the worst
case, all K memory tokens have maximum attention score, stealing probability
mass from the original T_q tokens. The total probability mass shifted is at
most K/(K+T_q) (by softmax normalization), giving L1 distance ≤ 2K/(K+T_q). ∎

For K=128, T_q=200: ||Δα||_1 ≤ 2·128/328 ≈ 0.78.

This is a LOOSE upper bound. In practice, memory tokens for unrelated content
will have low attention scores, causing much less perturbation.

**Prediction:** For general knowledge questions (MMLU-style), memory tokens from
unrelated injection text will have low cosine similarity with query keys.
Expected perturbation < 5% of attention mass. MMLU accuracy within 2pp of base.

**Risk:** If memory tokens happen to be semantically close to MMLU question
content, they could introduce spurious context and degrade accuracy by > 2pp.

## Kill Criteria Predictions

| Kill | Prediction | Threshold | Risk Level |
|------|------------|-----------|------------|
| K1366: Recall after 10 queries | ~70% (no memory update during queries) | > 50% | Medium — position encoding mismatch could kill this |
| K1367: Write latency | < 1ms (pure array copy) | < 5ms | Low |
| K1368: Base quality | MMLU within 1pp (low attention to unrelated memory) | within 2pp | Medium — attention mass shift could exceed bound |

## Behavioral Predictions

1. Recall will be HIGHER at global layers (full attention, 75% NoPE) than sliding layers
2. Recall degrades gracefully with query length (window constraint)
3. Write latency is dominated by the injection forward pass, not the memory copy
4. Base quality degradation scales with K — smaller K = less perturbation
