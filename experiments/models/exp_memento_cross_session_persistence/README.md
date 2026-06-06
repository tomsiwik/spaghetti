# exp_memento_cross_session_persistence

## Paper
[arXiv:2604.09852](https://arxiv.org/abs/2604.09852) — **within-session** context compression. This experiment extends it to **cross-session** handoff: novel contribution (not in paper).

## Dependency
- `exp_memento_gemma4_replication` must reach status=supported first (provides the memento-SFT'd Gemma 4 E4B checkpoint).

## Mechanism
Session N generates `M_N = {m_1, ..., m_k}` (mementos = dense summaries with associated KV states). Persist `M_N` keyed by `user_id` to disk. Session N+1 rehydrates `M_N` as a prefix instead of replaying raw user history.

Under LRU-or-relevance-weighted compaction, the buffer is bounded: `|buffer(user)| ≤ 2k tokens` regardless of session count.

## Reference repo pattern
`github.com/microsoft/memento` does not cover cross-session. We extend the serving layer:
- `memento/serving/user_buffer.py` — per-user buffer with compaction policy
- `memento/serving/hydrate.py` — load buffer at session-start, prepend to KV cache

## MLX serialization caveat
The **KV channel** (implicit information in KV states surviving block eviction — paper's 15pp AIME24 finding) must survive serialization. MLX arrays serialize via `mx.save` / `mx.savez` to `.npz`. Verify bit-exact round-trip before claiming K4 PASS.

## Quick start
```bash
# After exp_memento_gemma4_replication reaches supported:
experiment claim <worker-id> --id exp_memento_cross_session_persistence
experiment run exp_memento_cross_session_persistence
```
