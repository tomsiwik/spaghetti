# MATH.md — exp_memento_cross_session_persistence

**Claim:** A persisted memento buffer keyed by user-id enables cross-session handoff with ≥90% of full-context task accuracy, bounded buffer size (sub-linear growth under compaction), and KV-channel preservation across serialization round-trip.

---

## 1. Failure mode

Primary degenerate behavior: "The KV channel is a transient in-memory artifact — serializing to disk destroys the implicit information that mementos carried via KV representations. On rehydration, accuracy collapses to text-only memento-replay quality (paper's 15pp AIME24 ablation). K2 multi-turn accuracy falls to <80% of full-context."

Secondary: "Buffer compaction (LRU or relevance-weighted) discards mementos that reference earlier context the user returns to. Cross-session continuity degrades with session count — K3 sub-linear growth fails or K2 degrades over sessions."

## 2. Cited prior math / findings

- **Kontonis arxiv:2604.09852:** mementos carry dual text+KV channel; KV channel is load-bearing (15pp AIME24 ablation)
- **Pierre F#614:** thinking-mode required on Gemma 4 reasoning — mementos are thinking-mode artifacts
- **Pierre F#666:** target-gated — rehydration latency is proxy; multi-turn accuracy is target

## 3. Theorem (informal)

Let `M_i` be the memento set produced in session `i` for a user, `H_i` be the raw history (user turns + assistant turns), `buffer(i) = compact(M_1, ..., M_i)` under policy π.

**Theorem.** For a stream of sessions `s_1, s_2, ...`, running session `s_{n+1}` with prefix `buffer(n)` (instead of prefix `[H_1, ..., H_n]`):

1. **Latency:** rehydrate time < 50ms at |buffer| ≤ 1k tokens (K1 proxy)
2. **Accuracy:** task accuracy on session s_{n+1} ≥ 0.9 × accuracy with full [H_1..H_n] prefix (K2 target, pair K1)
3. **Compaction bound:** `|buffer(n)| ≤ 2k tokens` for all `n` under LRU-compaction policy with budget 2k (K3)
4. **Serialization preservation:** KV-cache round-trip (save → disk → load → continue) yields accuracy within 2pp of in-memory continuation (K4 target serving)

**Proof sketch.**
1. *Latency.* 1k tokens ≈ 2MB of 4-bit quantized KV state. Disk read on SSD ≈ 10ms; prepend to cache ≈ 30ms on M5 Pro. 50ms budget realistic.
2. *Accuracy preservation.* Mementos are designed to capture "reasoning state needed for subsequent blocks" (paper §3). Cross-session user turns are structurally similar to intra-session block transitions: both require carrying semantic state without carrying verbatim tokens. The 15pp AIME24 finding says KV channel matters — if we preserve it across disk, we preserve the mechanism. 10% budget (90% retention) is generous — expect 94-98%.
3. *Compaction bound.* Standard LRU or relevance-weighted compaction with fixed budget guarantees `|buffer| ≤ budget`. Sub-linear growth is trivial (constant-bounded).
4. *Serialization.* MLX `mx.savez` is bit-exact for float16/4-bit. Round-trip of KV tensors is lossless at the tensor level. Accuracy preservation within 2pp is a property of deterministic inference on identical state.

**Weak link:** (2) — the assumption that cross-session state preservation matches intra-session block transition. If user sessions are topically disjoint, mementos from session 1 may be actively misleading in session 2. Mitigation: relevance-weighted compaction with embedding similarity to current query.

## 4. Kill-criterion map

| KC | Measured quantity | Threshold | Type |
|---|---|---|---|
| K1 | rehydration latency at |buffer|=1k tokens, 100-memento buffer on M5 Pro | < 50ms | proxy |
| K2 | multi-turn task accuracy: memento-only handoff vs full-context, 30-turn benchmark | ≥ 90% | target (pair K1) |
| K3 | |buffer(n)| for n ∈ {10, 50, 100} under LRU budget=2k | all ≤ 2k tokens | target compaction |
| K4 | accuracy drop: round-tripped vs in-memory buffer continuation | < 2pp | target serialization |

Multi-turn bench: synthetic user-simulator over 30 turns of mixed-topic queries; measure task-completion rate. Full-context control replays all prior turns verbatim. Memento-only control uses compacted buffer only.

## 5. Predicted measurements

- K1: rehydrate ≈ 25-40ms
- K2: 92-96% of full-context (allowing for some topic-disjoint degradation)
- K3: buffer plateau at ~1.8k tokens after n=20 sessions (compaction stabilizes)
- K4: < 1pp round-trip drop (essentially bit-exact)

Risk: if K2 falls below 85%, the mechanism works *within session* but not *across*. That's a finding about the temporal locality of mementos.
