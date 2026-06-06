# MATH — `exp_memento_gemma4_replication_impl`

## §0. Platform skill attestation (PLAN.md §1011/§1012, reviewer (m2) gate)

Skills invoked this iteration (researcher iter ~104, 2026-04-25):
- **`/mlx-dev`** — INVOKED. Knowledge cited in this scaffold:
  - `mx.eval(model.parameters(), loss)` at step boundaries (Phase B requirement).
  - `mx.clear_cache()` between phase boundaries (F#673).
  - `nn.value_and_grad(model, loss_fn)`, not `.backward()`.
  - `mlx.optimizers.AdamW`, not `mlx.optim.AdamW`.
  - Lazy graph: implicit eval via `print/.item()/np.array`; structure step boundaries explicitly.
  - Gemma 4 E4B 4-bit: `embed_tokens` is `nn.QuantizedEmbedding` (mlx_lm/utils.py L869); `lm_head` is **tied** to `embed_tokens` (gemma4_text.py L594 `out = self.model.embed_tokens.as_linear(out)` when `tie_word_embeddings=True`). Resizing therefore happens once (on `embed_tokens`); `lm_head` reuses the resized weight via `as_linear`.
  - `mlx_lm.utils.load()` returns `(model, tokenizer)` with tokenizer = HF `TokenizerWrapper`. `add_special_tokens` mutates the underlying HF tokenizer in-memory.
- **`/fast-mlx`** — DEFERRED. Will be invoked at the start of Phase B implementation iteration (custom MLX SFT training loop). Phase A executed in this iteration is plumbing-only inspection (no training, no inference); /fast-mlx optimization patterns are not yet load-bearing.

**Inheritance.** Sections §1-§5 are inherited **verbatim** from parent `exp_memento_gemma4_replication/MATH.md`. KC IDs (#1829, #1830, #1831, #1832 in this _impl, mapping to parent K#1799, K#1800, K#1801, K#1802) are pre-registered in DB and have not been modified between parent and this _impl iteration. KC text is canonical (DB-anchored).

**This _impl iteration's marginal contribution over parent design-only PROVISIONAL (parent F#682-family novel-mechanism cluster):**
1. **/mlx-dev attestation** with concrete Gemma 4 quantized-embedding citations (parent had skill list, no specific code citations).
2. **Phase A inspect-and-extend (executable)** — actually loads the 4-bit model, mutates the tokenizer to vocab+4, inspects `embed_tokens` layer type and shape, and writes a `phase_a_results` block to `results.json`. Parent deferred this entirely.
3. **Phase B/C/D scaffolds** — `NotImplementedError` with concrete docstrings + line-level hand-off TODOs (where the custom training loop and block-mask inference loop go).
4. **Verdict honesty** — `is_smoke=true`, verdict=`PROVISIONAL`, all 4 KCs `untested`. The _impl is structurally a `_impl-Phase-A` slice; full pipeline (Phase B 2-stage SFT 4000 steps + Phase C custom inference + Phase D K1-K4 eval) remains a 6-10h follow-up iteration.

## §0.1. Phase A scope (executable in this iteration)

Phase A = tokenizer extension and embedding-layout inspection. Concretely:

| Step | Action | Output |
|---|---|---|
| A.1 | `mlx_lm.utils.load(BASE_MODEL)` | loaded model + tokenizer |
| A.2 | Inspect `tokenizer.vocab_size`, special tokens | baseline vocab=262144 (Gemma 4 E4B) |
| A.3 | `tokenizer._tokenizer.add_special_tokens({"additional_special_tokens": MEMENTO_TOKENS})` | new vocab=262148 |
| A.4 | Resolve token IDs for the 4 new tokens via `tokenizer.encode("<|block_start|>", add_special_tokens=False)` | 4 integer IDs, all ≥ 262144 |
| A.5 | Inspect `model.language_model.model.embed_tokens` type + shape | `nn.QuantizedEmbedding` confirmed; shape=(262144, hidden) |
| A.6 | Inspect `model.language_model.tie_word_embeddings` + `lm_head` presence | `tie=True`; no separate `lm_head` linear layer; lm_head reuses embed via `as_linear` |
| A.7 | Document resize procedure for Phase A.v2 (deferred): dequantize embed → expand to vocab+4 → mean-init last 4 rows → leave un-quantized OR re-quantize to 4-bit → save extended-model checkpoint | recorded in `phase_a_results.resize_plan` |

**What Phase A does NOT do this iteration** (deferred to Phase A.v2 in a follow-up iteration outside researcher cap):
- Actually mutate `model.language_model.model.embed_tokens` weights (quantized resize is non-trivial; needs `mx.dequantize` + careful re-init + optional `nn.quantize` round-trip).
- Save extended-model checkpoint to disk.
- Run any forward pass with the new tokens.

The reason Phase A is split: *quantized-embedding resize on a 4-bit MLX model is itself a research subtask* — naive `model.embed_tokens.weight = new_weight` will not work because `QuantizedEmbedding` stores packed 4-bit codes plus per-group scales/biases, and the `as_linear` lm_head dispatch requires the same packing. The split surfaces this honestly: tokenizer extension (cheap, in-memory) is done; quantized-embedding resize (needs MLX research + forward-pass smoke) is a follow-up.

## §0.2. KC binding under split Phase A

All 4 KCs (#1829, #1830, #1831, #1832) require Phase B (SFT) AND Phase C (block-mask inference) AND Phase D (eval harness). None are decidable from Phase A alone. Therefore:

- `is_smoke=true` (per researcher hat clause 6.4: smoke runs route to PROVISIONAL).
- All KCs `untested` with `reason="Depends on Phase B+C+D; Phase A inspect-only this iter"`.
- Verdict `PROVISIONAL`, **not** `SUPPORTED` and **not** `KILLED`. F#666 target-gating is preserved (no proxy KC stand-alone PASS asserted; no spurious smoke-N MMLU drop reported because no inference run).

---

# MATH — `exp_memento_gemma4_replication` (parent inherited verbatim)

**Claim:** MEMENTO's 2-stage SFT + block-mask attention replicates at Gemma 4 E4B 4-bit MLX: ≥2× KV cache reduction on GSM8K-Hard with < 5pp accuracy drop; ablating the KV channel drops accuracy ≥ 10pp (replicates paper's 15pp AIME24 finding at our scale).

See parent `micro/models/exp_memento_gemma4_replication/MATH.md` §1-§5 for full theorem statement, proof sketch, KC pre-registration, and predicted measurements. Inheritance is verbatim — KC IDs map 1:1 (parent K#1799 → impl K#1829, K#1800 → K#1830, K#1801 → K#1831, K#1802 → K#1832).

## §6. Phase B/C/D handoff (deferred to follow-up iteration)

**Phase B — 2-stage SFT on `microsoft/OpenMementos`:**
- `stage1`: standard next-token CE on 228K traces, full-parameter SFT (no LoRA substitution per antipattern-t — paper's mechanism is FT-on-block-boundary-aware-data, not adapter).
- `stage2`: attend-only-to-mementos forward (mask all block-content KV after `<|summary_end|>`), 2000 steps each.
- Optimizer: `mlx.optimizers.AdamW(lr=2e-5)`. `mx.eval` at each step boundary; `mx.clear_cache()` between stage-1/stage-2.
- Memory: `mx.checkpoint` + `batch=1` + grad-accumulation; if OOM at SEQLEN=4096, drop SEQLEN=2048 (notes a *narrower-context* finding, NOT a different mechanism).
- Wall-clock: ~3-5h on M5 Pro 48GB.

**Phase C — custom MLX inference loop with block-mask attention:**
- Implements `BlockMaskState`: tracks per-token block boundaries from emitted `<|block_start|>`/`<|block_end|>` tokens.
- Per-token attention mask surgery: when generating after `<|summary_end|>`, mask all block-content KV (keep only summary-text KV + current block KV).
- Selective KV eviction: free GPU memory for evicted block KV (this is what produces K1's KV reduction).
- Uses `mx.fast.scaled_dot_product_attention(mask=...)` per `/mlx-dev` skill.
- Wall-clock: ~1-2h to implement + smoke test.

**Phase D — eval harness:**
- K1: peak-KV instrumentation on GSM8K-Hard n≥200 (greedy generation; track `mx.get_active_memory()` peak).
- K2: GSM8K-Hard accuracy + MMLU N=200/cat with thinking-mode-on (per F#614 +25pp lift).
- K3: ablation arm — same M_memento with `mask_blocks_completely=True` (summary-text-only baseline). Measure acc gap.
- K4: throughput on long-context prompts (n≥50 prompts, ≥4096 tokens each).

**Total wall-clock budget:** 6-10h (Phase B 3-5h + Phase C 1-2h + Phase D 2-3h). This is what makes the IMPL exceed the researcher-hat single-iteration cap, justifying the Phase A split adopted here.

## §7. Antipattern scan

| Antipattern | This iteration | Status |
|---|---|---|
| composition-bug (`Σ B_i @ A_i` not `(ΣB)(ΣA)`) | N/A — no LoRA composition this iter | OK |
| LORA_SCALE>8 | N/A — no LoRA this iter | OK |
| tautological routing | N/A — no routing this iter | OK |
| shutil.copy as new adapter | N/A — no adapter saved this iter | OK |
| hardcoded `"pass": True` | All KCs `"untested"` (no premature pass) | OK |
| eval-template truncation | N/A — no eval this iter | OK |
| proxy-model substitution | Loads exact `mlx-community/gemma-4-e4b-it-4bit` per parent §0 | OK |
| KC measures wrong object | All KCs unchanged from parent DB | OK |
| smoke reported as full | `is_smoke=true` flag set; verdict=PROVISIONAL routed | OK |
| novel-mechanism-single-iter scope (`mem-antipattern-novel-mechanism-single-iteration-scope`) | Phase A split honored; Phase B/C/D explicitly deferred | OK |
| antipattern-t (silent scope swap) | Full-parameter SFT preserved; LoRA substitution forbidden in §6 hand-off | OK |
| researcher pre-files finding before review (`mem-antipattern-researcher-prefiles-finding-before-review`) | No `experiment finding-add` this iter; defer to reviewer | OK |

---

## §8. Verdict pre-registration

If Phase A executes cleanly (model loads, tokenizer extends, embed inspected): verdict=`PROVISIONAL`, KCs all `untested`, `is_smoke=true`.

If Phase A errors (load fails, tokenizer mutation rejected, etc.): verdict=`PROVISIONAL` with `phase_a_status=error`, KCs all `untested`, error logged in `results.json["phase_a_results"]["error"]`. **Never** silently upgrade to SUPPORTED; never KILL on a smoke plumbing error (which would reflect tooling, not mechanism).
