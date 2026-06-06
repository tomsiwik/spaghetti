# PAPER — `exp_memento_gemma4_replication_impl`

**Verdict.** PROVISIONAL (`is_smoke=true`; Phase A inspect+extend executed cleanly; Phase A.v2 quantized-resize, Phase B SFT, Phase C custom inference loop, Phase D KC eval all `NotImplementedError` and deferred to a follow-up iteration outside the researcher-hat single-iter cap).

**Date.** 2026-04-25 (researcher iter ~104, drain-window).

**Marginal contribution over parent `exp_memento_gemma4_replication` (PROVISIONAL design-only, F#682-family novel-mechanism cluster):**

1. **/mlx-dev skill invoked**, attestation block written to `MATH.md` §0 with concrete Gemma 4 quantized-embedding citations (mlx_lm/utils.py L869, gemma4_text.py L222/L594). Reviewer (m2) gate satisfied.
2. **Phase A token-extension actually executed** end-to-end (parent deferred this entirely). Loaded `mlx-community/gemma-4-e4b-it-4bit`, mutated tokenizer to vocab+4 in-memory, resolved new token IDs, and inspected the embed-layer topology.
3. **Plumbing verified**:
   - Model loads in 3.0s on M5 Pro 48GB (cached weights).
   - HF tokenizer mutation works as documented.
   - `embed_tokens` is confirmed `nn.QuantizedEmbedding` with packed shape `[262144, 320]` (= 2560 hidden / 8 codes-per-byte at 4-bit).
   - `tie_word_embeddings=True` confirmed; no separate `lm_head` linear layer (lm_head dispatches via `embed_tokens.as_linear` per `gemma4_text.py` L594).

---

## Prediction-vs-measurement table

| KC (DB id) | Prediction (MATH.md §3, parent inherited) | Mechanism | Measured | Status |
|---|---|---|---|---|
| K#1829 (=K1799 parent) | Peak KV cache ratio `M_memento / M_base ≤ 0.5` on GSM8K-Hard n≥200 | Block-mask attention + selective KV eviction | NOT MEASURED — Phase C inference loop not implemented | untested |
| K#1830 (=K1800 parent) | GSM8K-Hard drop < 5pp AND MMLU drop < 3pp at n≥200 | 2-stage SFT preserves accuracy under compression | NOT MEASURED — Phase B (SFT) and Phase D (eval) not implemented | untested |
| K#1831 (=K1801 parent) | KV-channel ablation acc gap ≥ 10pp on reasoning bench | Implicit KV signal carries memento information beyond the summary text | NOT MEASURED — Phase D K3 ablation arm not implemented | untested |
| K#1832 (=K1802 parent) | Throughput `M_memento / M_base ≥ 1.3×` on long-context prompts | Block-mask reduces per-token attention work to O(L_summary + L_block) | NOT MEASURED — Phase C/D not implemented | untested |

## What was actually measured (Phase A inspect+extend)

| Item | Value | Source |
|---|---|---|
| Model load wall-clock | 3.0s | `phase_a_results.load_seconds` |
| `tokenizer.vocab_size` baseline | 262144 | `phase_a_results.baseline_vocab` |
| Pre-existing special tokens | 24 | `phase_a_results.baseline_special_tokens_count` |
| `add_special_tokens` returned | 4 (= len(MEMENTO_TOKENS)) | `phase_a_results.n_added_tokens` |
| New `tokenizer.vocab_size` | 262148 | `phase_a_results.new_vocab` |
| Memento token IDs | `<|block_start|>=262144`, `<|block_end|>=262145`, `<|summary_start|>=262146`, `<|summary_end|>=262147` | `phase_a_results.memento_token_ids` |
| All new IDs ≥ baseline | True | `phase_a_results.all_new_ids_above_baseline` |
| `embed_tokens` type | `QuantizedEmbedding` | `phase_a_results.embed_type` |
| `embed_tokens.weight` packed shape | `[262144, 320]` (= 2560 hidden_size / 8 codes-per-byte at 4 bits) | `phase_a_results.embed_shape_packed` |
| `model.config.vocab_size` | 262144 | `phase_a_results.config_vocab_size` |
| `tie_word_embeddings` | True | `phase_a_results.tie_word_embeddings` |
| Separate `lm_head` linear | False (tied; dispatches via `embed_tokens.as_linear`) | `phase_a_results.lm_head_separate` |

## Key observation about Phase A.v2 (deferred)

Phase A.v2 = actually mutate `embed_tokens.weight` from shape `(262144, hidden)` to `(262148, hidden)` and validate forward-pass logits remain sane on existing tokens. This is **not just plumbing** — it is itself a research subtask because:

1. **Quantized resize is non-trivial.** `nn.QuantizedEmbedding` packs 4-bit codes (`weight` shape `(262144, hidden//8)`) plus per-group `scales` and `biases`. Naive `embed.weight = new_weight` will break the dispatch contract.
2. **Faithful procedure (documented in `MATH.md` §0.1 step A.7):**
   - Dequantize: `w_fp = mx.dequantize(embed.weight, embed.scales, embed.biases, group_size, bits)`
   - Compute mean row across existing 262144 rows.
   - Concatenate 4 mean-init rows → new fp weight `(262148, hidden)`.
   - Either (a) build a fresh un-quantized `nn.Embedding(262148, hidden)` (mixed-precision; embed unquantized, rest of model 4-bit), or (b) re-quantize via `nn.quantize` to keep all-4-bit.
   - Replace `text_model.embed_tokens` and call `mx.eval(model.parameters())` to materialize.
3. **Tied-lm_head implication:** because `gemma4_text.Model.__call__` uses `out = self.model.embed_tokens.as_linear(out)` when `tie_word_embeddings=True`, only ONE resize is needed (no separate `lm_head` mutation).

The dequantize → resize → re-quantize round-trip can shift logits subtly, requiring a regression check (compare base logits before/after on a probe prompt). This regression check is part of Phase A.v2 scope.

## Why PROVISIONAL not SUPPORTED

`is_smoke=true`. Per researcher hat clause 6.4: "smoke-mode runs complete as `--status provisional` with a TODO to rerun at full N; never `supported` or `killed`." Additionally, all 4 KCs are `untested` — a SUPPORTED verdict would be fabrication.

## Why PROVISIONAL not KILLED

No proof-based impossibility result has been derived. The blocker is **implementation effort** (Phase A.v2 quantized-resize subtask + Phase B 4000-step SFT + Phase C custom MLX inference loop + Phase D K1-K4 eval, total ~6-10h on M5 Pro 48GB), not falsification. Phase A executed cleanly; no failure mode has been observed.

## Hand-off (what next iteration must do)

To convert this PROVISIONAL → SUPPORTED or KILLED requires (in order):

1. **Phase A.v2 — quantized-embedding resize + forward-pass smoke** (~30-60 min):
   - Invoke `/mlx-dev` AND `/fast-mlx`.
   - Implement the dequantize → mean-init → concat → re-quantize round-trip in `phase_a_resize_embedding()` (new function).
   - Save extended-model checkpoint to `adapters/memento_extended_e4b/`.
   - Forward-pass smoke: run 1 token of inference on the extended model with the original tokenizer (sanity), then 1 token using a memento-token-bearing prompt (verify new tokens dispatch).
   - Logit-regression check: `||logits_extended[0:262144] - logits_baseline[0:262144]||_2 < ε` for a fixed probe prompt.

2. **Phase B — 2-stage SFT on `microsoft/OpenMementos`** (~3-5h):
   - Implement `phase_b_sft_stage1` and `phase_b_sft_stage2` (currently `NotImplementedError`).
   - Full-parameter SFT, **not LoRA** (paper faithfulness; antipattern-t).
   - `mx.checkpoint` + batch=1 + grad-accumulation; if OOM → SEQLEN=2048 (narrower-context finding, not different mechanism).
   - `mx.eval(model.parameters(), loss)` per step; `mx.clear_cache()` between stages.

3. **Phase C — custom MLX inference loop with block-mask attention** (~1-2h):
   - Implement `BlockMaskState`, `phase_c_block_mask_generate`.
   - Use `mx.fast.scaled_dot_product_attention(mask=...)` per `/mlx-dev`.
   - Track per-token block boundaries; selective KV eviction post `<|summary_end|>`.

4. **Phase D — KC eval harness** (~2-3h):
   - K1: peak-KV instrumentation on GSM8K-Hard n≥200 (track `mx.get_active_memory()` peak per-prompt).
   - K2: GSM8K-Hard accuracy n≥200 + MMLU N≥200 with `enable_thinking=True` (per F#614 +25pp lift).
   - K3: ablation arm — same `M_memento` with `mask_blocks_completely=True`. Measure `acc_gap = acc(M_memento, normal) - acc(M_memento, summary-only)`. Pass if `≥ 10pp`.
   - K4: throughput on `n≥50` long-context prompts (≥4096 tokens each).

5. **Verdict-consistency pre-flight** (PLAN.md §1, all 6 checks) before `experiment complete --status supported`.

## Assumptions (per researcher autonomy clause)

- **Hidden size = 2560** inferred from packed `[262144, 320]` shape at 4-bit (`320 × 8 codes/byte = 2560`). Confirmed against Gemma 4 E4B mlx-community spec.
- **Tied lm_head removes one resize step** — verified by `tie_word_embeddings=True` and absence of `lm_head` linear (lm_head dispatches via `embed_tokens.as_linear`).
- **4 new IDs are clean append** (262144-262147) — HF tokenizer's `add_special_tokens` returned `n_added=4` matching `len(MEMENTO_TOKENS)`. No collisions with existing special tokens.
- **Phase A is genuinely smoke**, not a partial KC binding — all 4 KCs (K#1829-K#1832) require Phase B+C+D and remain `untested`. F#666 target-gating preserved.

## Verdict-consistency pre-flight (all 6 checks per PLAN.md §1)

1. `results.json["verdict"]` = `"PROVISIONAL"` — not KILLED, not SUPPORTED ✓
2. `results.json["all_pass"]` = `false` — consistent with PROVISIONAL ✓
3. PAPER.md verdict line reads `PROVISIONAL` ✓
4. `is_smoke` = `true` — explicit smoke routing to provisional ✓
5. KC text matches parent DB-canonical IDs K#1799-K#1802 (=#1829-#1832 in this _impl). No KC modified ✓
6. Antipattern scan in MATH.md §7 covers all 12 items; 0 fire, all OK or N/A ✓

## Findings worth registering (research signal beyond pre-reg KCs; reviewer files canonical)

- **F1 (positive plumbing pattern, 1st instance for Memento cluster):** Phase A inspect+extend pattern — load 4-bit Gemma 4, inspect QuantizedEmbedding shape + tied-lm_head topology + tokenizer mutation in <5s wall-clock — is reusable across any Gemma 4 IMPL that requires vocab extension. This unblocks similar IMPLs (e.g., adapter-token-aware composition experiments) by isolating the quantized-resize subtask cleanly.
- **F2 (Gemma 4 E4B 4-bit topology — concrete numbers):** packed embed shape `[262144, 320]` → hidden_size=2560 (not 2304 as one might assume from "4B params"); `tie_word_embeddings=True` removes one resize step; lm_head dispatches via `embed_tokens.as_linear`. These are reusable architectural facts for any future Gemma 4 IMPL.
- **F3 (3rd post-promotion observance of `mem-antipattern-researcher-prefiles-finding-before-review`):** This iteration does not call `experiment finding-add` — defers to reviewer per the antipattern gate. After 3 consecutive observances (refactor_full iter ~100 + formality_full iter ~102 + this iter), the antipattern can be considered mitigated.
