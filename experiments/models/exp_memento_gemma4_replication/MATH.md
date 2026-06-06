# MATH.md — exp_memento_gemma4_replication

**Claim:** MEMENTO's 2-stage SFT + block-mask attention replicates at Gemma 4 E4B 4-bit MLX: ≥2× KV cache reduction on GSM8K-Hard with < 5pp accuracy drop; ablating the KV channel drops accuracy ≥ 10pp (replicates paper's 15pp AIME24 finding at our scale).

---

## 0. Scope & skill invocation

**Status:** PROVISIONAL (design-only). This experiment is a **novel-mechanism replication** — MEMENTO's block-mask attention with dynamic KV eviction is not executable via the `mlx_lm.lora` CLI. A faithful implementation requires a custom MLX generation loop (per-token `BlockMaskState` update + attention-mask surgery + KV-tensor eviction calls between blocks), plus 2-stage SFT (stage 1 standard next-token CE on OpenMementos; stage 2 attend-only-to-mementos which itself needs the mask-surgery path). Full pipeline exceeds the researcher-hat 30-min / 40-tool-call cap (`researcher.md` §"Context discipline"; `mem-antipattern-novel-mechanism-single-iteration-scope`). Per `reviewer.md` §5 canonical PROVISIONAL-as-design clause (3-precedent threshold: F#682, F#683, F#684), the filing is honest design-only: pre-registered MATH.md with all 4 KCs + graceful-failure `run_experiment.py` that writes `results.json` verdict=PROVISIONAL with all KCs `"untested"`. The full implementation is filed as `exp_memento_gemma4_replication_impl` at P3, inheriting this MATH.md verbatim.

**Platform skills required before IMPL writes any MLX code:**
- `/mlx-dev` — MLX array/nn/training/inference discipline. Load the official `mlx-lm` module, confirm `mx.fast.scaled_dot_product_attention` signature (pass `mask=` kwarg), use `nn.value_and_grad(model, loss_fn)` not torch-style `.backward()`, `mx.eval(model.parameters(), loss)` at each SFT step boundary, `mx.clear_cache()` between stage-1/stage-2 and between SFT and KC-eval phases per F#673.
- `/fast-mlx` — generation-loop performance. Enforce `mx.compile` on the per-token generation step where the mask does not shape-change mid-sequence; use paged KV-cache structure compatible with selective eviction (block-index metadata kept alongside K/V).

**Model & base configuration (fixed in pre-reg):**
- Base: `mlx-community/gemma-4-e4b-it-4bit` (`PLAN.md` Part 2 dev target).
- Tokenizer extension: 4 special tokens `<|block_start|>`, `<|block_end|>`, `<|summary_start|>`, `<|summary_end|>`. Resize `model.model.embed_tokens` + `model.lm_head` to `vocab+4`; new rows init from the mean of existing rows (standard, preserves logit distribution). `mlx-lm >= 0.22` required for tokenizer-extend path.
- SFT approach: full-parameter SFT per paper faithfulness (LoRA-adapter variant would change what K2 measures — the memory burden the mechanism is meant to reduce — so **no scope-swap to LoRA**; IMPL will use `mx.checkpoint` + batch=1 + grad-accumulation to fit 4-bit Gemma 4 E4B SFT on M5 Pro 48GB, not a silent LoRA substitution per antipattern-t).
- Dataset: `microsoft/OpenMementos` (228K traces; MIT; already SFT-formatted with boundary tokens).
- Baseline: a fresh `mlx-lm.load(BASE_MODEL)` instance kept in separate memory region; K1/K2/K4 are head-to-head `M_memento` vs `M_base`.

**Scope-preservation note (antipattern-t):** If IMPL encounters OOM on 48GB at `SEQLEN=4096 batch=1`, the fixes in priority order are: (i) `mx.checkpoint` recomputation, (ii) grad-accumulation with `batch=1` effective-batch unchanged, (iii) reduce `SFT_STEPS` (weaker claim, not different mechanism), (iv) escalate to 26B-A4B with a new experiment. Silent swap to LoRA-adapter or shorter `SEQLEN` is forbidden — those change what the KCs measure.

---

## 1. Failure mode

Primary degenerate behavior: "Gemma 4 E4B at ~4B active params is below the minimum scale for memento-format reasoning. The model cannot produce well-scoped summaries of its own thinking blocks; summaries are either (a) near-empty (model defaults to 0-token mementos), (b) copy-paste of the block (no compression), or (c) hallucinated state that doesn't match the block content." Under (a)/(b) K1 compression fails; under (c) K2 accuracy fails.

Secondary: "The implicit KV channel that carries information across block boundaries (paper's 15pp AIME24 finding) requires a minimum model capacity. At 4B active, KV-cache representations are too compressed to carry that implicit signal. K3 ablation test falls within noise (< 5pp gap), invalidating the mechanism claim." This is itself a **publishable finding** ("MEMENTO KV channel requires ≥8B active").

## 2. Cited prior math / findings

- **Kontonis et al. arxiv:2604.09852:**
  - 2.5× KV reduction, 1.75× throughput at 8B-32B
  - 228K OpenMementos traces, 15-25% compression target
  - 15pp AIME24 drop when KV channel removed (ablation keeps only summary text)
  - Dual-channel hypothesis: mementos carry both explicit text AND implicit KV state
- **Pierre F#614 (supported):** CoT/thinking-mode gives +25pp on MMLU-Pro for Gemma 4 E4B reasoning — thinking is load-bearing
- **Pierre F#536 (supported):** Gemma 4 E4B thinking-mode baseline 62.1% MMLU-Pro (N=20/cat)
- **Pierre F#666:** target-gated — KV-reduction proxy paired with accuracy target

## 3. Theorem (informal)

Let `M_memento` = SFT'd Gemma 4 E4B with block-mask inference, `M_base` = unmodified Gemma 4 E4B. Let `KV(x)` denote peak KV-cache size during generation on prompt `x`.

**Theorem (replication).** After 2-stage SFT on OpenMementos:

1. **KV reduction:** `E[KV(M_memento; x)] / E[KV(M_base; x)] ≤ 0.5` on GSM8K-Hard (K1)
2. **Accuracy preservation:** `acc(M_memento, GSM8K-Hard) ≥ acc(M_base) − 5pp` at n≥200 (K2, pair K1 per F#666)
3. **KV-channel necessity:** running `M_memento` with full block-eviction (summary-text-only, mask all block KV) drops accuracy by ≥ 10pp vs `M_memento` with blocks masked only after summary emission (K3 replicates paper's 15pp finding at our scale)
4. **Throughput:** `M_memento` throughput ≥ 1.3× `M_base` on long-context prompts (K4 — weaker than paper's 1.75× due to MLX vs vLLM infra gap)

**Proof sketch.**
1. *KV reduction (1).* Paper demonstrates 2.5× at 8B; 2× is a weaker claim expected at 4B (smaller model produces proportionally smaller blocks, compression ratio may differ but direction holds).
2. *Accuracy bound (2).* Standard SFT-from-supervised-trace-distillation preserves accuracy within a model-capacity-dependent margin. At 4B, 5pp tolerance is reasonable (paper reports < 3pp at 8B).
3. *KV-channel (3).* Paper's core empirical claim. We bet on direction-preserving at 4B: the KV channel is a property of the attention mechanism (not model size), so ablating it should cause measurable drop. Magnitude may be smaller (5-15pp range instead of 15pp). If drop < 10pp, we REVISE to a lower threshold and note the scale-dependence as a finding.
4. *Throughput (4).* Block-masked attention reduces per-token work from O(L) to O(L_summary + L_current_block) ≈ O(L/3) on average. Gives ~1.5× throughput in theory; MLX overhead (not vLLM) reduces to ~1.3× in practice.

**Failure-gate:** If K2 fails (accuracy drop > 5pp) or K3 fails (< 10pp ablation drop), the experiment status is `killed` with finding: "MEMENTO mechanism fails at sub-8B scale on Gemma 4 architecture." This unlocks a pivot experiment at 26B-A4B.

## 4. Kill-criterion map

| KC | Measured quantity | Threshold | Type |
|---|---|---|---|
| K1 | peak KV cache reduction ratio on GSM8K-Hard n≥200 | ≥ 2.0× | proxy |
| K2 | GSM8K-Hard accuracy drop vs base, MMLU drop vs base | GSM8K < 5pp AND MMLU < 3pp | target (pair K1) |
| K3 | KV-channel ablation drop on reasoning bench (GSM8K-Hard or AIME subset) | ≥ 10pp | target replication |
| K4 | end-to-end throughput vs base on long-context prompts | ≥ 1.3× | target serving |

## 5. Predicted measurements

- K1: 2.0-2.3× (lower than paper's 2.5× because 4B blocks are smaller → less compressible proportionally)
- K2: GSM8K-Hard drop ∈ [2, 6]pp (may fail boundary — watch)
- K3: ablation drop ∈ [8, 15]pp (paper found 15pp at 8B; expect weaker at 4B)
- K4: 1.3-1.5× throughput on M5 Pro

If K3 drops < 10pp, that's the scale-dependence finding. If K2 fails, we pivot the program to 26B-A4B.
