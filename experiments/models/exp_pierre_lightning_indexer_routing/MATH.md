# MATH.md — Lightning Indexer for Adapter Routing (DSv4 CSA-derived)

## Architecture (DSv4 §2.3.1, Eqs 13-17 adapted)
For query token t with hidden state h_t ∈ ℝ^d:

```
c_t^Q   = h_t · W^DQ              (low-rank query, ℝ^{d_c}, d_c=128)
[q_t,h] = c_t^Q · W^IUQ            (n_h^I=6 indexer query heads, ℝ^{c^I=128})
w_t^I   = h_t · W^w               (per-head indexer weights, ℝ^{n_h^I})

Per adapter a ∈ {1..7}:
I_t,a = Σ_h w_t,h^I · ReLU(q_t,h^I · K_a^IComp)   # K_a^IComp ∈ ℝ^{c^I=128}, learned

selection = top_k(I_t,:, k=2)
soft_weights = softmax(top_k_scores)
```

## Why this is fast on M5 Pro
- 7 adapters × 6 heads × 128 dims × 4 bytes = 21KB indexer keys → fits L1 (128KB)
- Per-query compute: ~50K MACs total → <10µs single-core NEON
- FP4 path (DSv4 reports 99.7% recall): another 2× speedup

## Predictions (K2154-K2157)
- Top-2 accuracy ≥80% on benchmark prompts (vs 78.3% prior hidden-state router)
- Latency ≤50µs FP32, ≤10µs FP4 (vs ~50µs M2P-gate softmax)
- Composed accuracy ≥ M2P-gated (parity at higher speed)
- Indexer keys ≤5MB total

## Talos integration
- Indexer keys + query projection live in L1 throughout decode
- Per-token routing decision: 1 dot product + ReLU + top-k → ~5µs in optimized C
- Replaces M2P-gate's MLX softmax-over-7 with hand-tuned NEON kernel

## References
- DeepSeek V4 §2.3.1 (CSA Lightning Indexer)
- F#58 (top-2 wins)
- ../talos-vs-macbook (NEON kernel pattern)
