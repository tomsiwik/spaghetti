# LEARNINGS — `exp_memento_gemma4_replication_impl`

**Date:** 2026-04-25 (analyst iter ~106, drain-window).
**Verdict ratified:** PROVISIONAL. **Finding:** F#799 (canonical reviewer-attributed).

## Core Finding

Phase A inspect+extend executes cleanly on `mlx-community/gemma-4-e4b-it-4bit` in 3.0s wall-clock. Tokenizer mutates 262144→262148; embed layer is `nn.QuantizedEmbedding[262144,320]` (= hidden=2560 at 4-bit packing); `tie_word_embeddings=True` so a single embed-layer mutation suffices for vocab extension (no separate `lm_head`). Phase A.v2 quantized resize, Phase B 2-stage SFT, Phase C custom inference loop, Phase D KC eval all `NotImplementedError`. All 4 KCs (#1829-#1832) `untested` — F#666 target-gating preserved.

## Why

Marginal Phase A executable slice over parent design-only F#682-cluster precedent. Quantized-resize round-trip (dequantize → mean-init concat → re-quantize) is a non-trivial research subtask, not pure plumbing — extracting it into Phase A.v2 isolates the mlx-quant correctness bound from MEMENTO mechanism claims. `/mlx-dev` attestation in MATH.md §0 cites concrete Gemma 4 quantized-embedding sources (`mlx_lm/utils.py L869`, `gemma4_text.py L222/L594`). `/fast-mlx` deferred — defensible because no training-loop code lands this iter.

## Implications for Next Experiment

1. **Memento _impl continuation is a 6-10h follow-up on the same dir** (Phase A.v2 → B → C → D), NOT a new experiment ID. Mirrors politeness_full / refactor_full / conciseness_full / formality_full precedent. Out of scope for the current single-iter drain budget.
2. **P≤2 backlog has 1 entry left:** `exp_g4_adapter_class_composition_full_impl` (P=1 macro, last drain pick). Recommend as TOP next claim to reach `RESEARCH_BACKLOG_DRAINED`. Macro-scale feasibility check first: if it requires a 26B teacher or multi-hour training, smoke-port the F#790 / F#797 / F#798 enable_thinking=False harness pattern and stop at smoke-PROVISIONAL.
3. **Reusable Gemma 4 E4B 4-bit architectural facts** worth promoting to a context memory for any future Gemma 4 IMPL (vocab/adapter-token extension, custom-token routing, MEMENTO-style buffers): packed embed shape `[262144, 320]` ⇒ hidden=2560; `tie_word_embeddings=True` ⇒ single embed-resize via `embed_tokens.as_linear`; 24 baseline special tokens; `add_special_tokens` returns clean `n_added=4` matching `len(MEMENTO_TOKENS)`.
4. **Antipattern `mem-antipattern-researcher-prefiles-finding-before-review` is now MITIGATED** (3 consecutive post-promotion observances: refactor_full iter ~100 + formality_full iter ~102 + memento_replication_impl iter ~104 — researcher honored finding-add gate in all three, reviewer filed canonical finding in all three). Memory updated to record mitigation.

## Antipattern capture

No new antipatterns this iter. `mem-antipattern-researcher-prefiles-finding-before-review` upgraded to MITIGATED status (3 consecutive observances).

## References

No external paper added — F#682-cluster precedent + parent `exp_memento_gemma4_replication` cite the original MEMENTO paper already.
