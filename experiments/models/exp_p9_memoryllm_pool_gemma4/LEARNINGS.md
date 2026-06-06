# LEARNINGS: exp_p9_memoryllm_pool_gemma4

## Status: KILLED (Finding #512)

## Core Finding
Hidden-state injection (MemoryLLM mechanism) produces 0% recall on a pretrained Gemma 4 E4B — complete failure vs predicted 70%. The mechanism requires fine-tuning; it is structurally incompatible with any frozen model.

## Why
Three independent impossibility structures: (1) RoPE encodes absolute position into Q/K, so hidden states moved to new positions produce corrupted attention scores; (2) untrained attention has no learned pattern to read "external memory" tokens; (3) per-layer injection breaks the h_{l} → h_{l-1} dependency chain, creating compounding inconsistency. Each alone is fatal. Reference: MemoryLLM arXiv:2402.04624 explicitly requires LoRA fine-tuning for memory reading.

## Implications for Next Experiment
Memory pool is not viable without fine-tuning. Two structural alternatives survive: KV-cache prefilling (preserves RoPE, no training, no FIFO updates) or PLE injection via Gemma 4's native gated pathway (trained gate + projection already exists). For our composable adapter system, orthogonal adapter composition (Finding #511) is the more pressing problem — memory is secondary.

## Key Numbers
- K1366 HS recall: 0% (predicted 70%) — decisive refutation
- K1367 write latency: 2.82ms (predicted <1ms) — PASS despite prediction error
- K1368 quality: INCONCLUSIVE (prompt format bug)

## Impossibility Structure
`h_l injected at positions 0..K-1` requires `position-coherent Q/K` + `trained attention read patterns` + `layer-consistent hidden states`. All three absent in frozen pretrained model. Fix requires fine-tuning — not a hyperparameter issue.
