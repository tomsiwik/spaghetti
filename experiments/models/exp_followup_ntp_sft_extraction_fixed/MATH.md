# MATH: exp_followup_ntp_sft_extraction_fixed — Qwen2.5-3B GSM8K with proper extraction

## 1. Failure mode

Prior experiment `exp_competitive_benchmark` reported Qwen2.5-3B-Instruct-4bit GSM8K = 36%
(50 samples). Published Qwen2.5-3B-Instruct GSM8K ≈ 68.7% (Qwen2.5 Technical Report, Table 8).
A ~32 pp gap is too large to be 4-bit quantization alone (typical degradation: ≤3 pp on GSM8K
per MLX-community QA reports). Candidate causes: (a) wrong prompt format (missing chat
template), (b) extraction regex rejecting valid answers, (c) MLX-4bit quality cliff, (d) n=50
sampling noise.

This experiment re-runs Qwen baseline with **n=200**, **proper `tokenizer.apply_chat_template`
(chatml)**, and **canonical lm-evaluation-harness GSM8K extraction** to separate (a)+(b) from
(c). The original `exp_competitive_benchmark` parent is KILLED on K1/K2/K3; this followup is
purely a baseline-validity cleanup.

## 2. Prior results cited

- **`exp_competitive_benchmark` LEARNINGS** (our finding #??): "Qwen GSM8K score is
  suspiciously low (36% vs published 65-70%) ... Likely a prompt format or answer extraction
  bug."
- **Qwen2.5 Technical Report** (arxiv:2412.15115): Qwen2.5-3B-Instruct GSM8K = 68.7%
  (greedy, 8-shot chat template).
- **lm-evaluation-harness** (Gao et al.): canonical GSM8K extraction is `#### <num>`
  first, else last numeric token in generation.
- **MLX-community model card** for `mlx-community/Qwen2.5-3B-Instruct-4bit`: quantized from
  Qwen/Qwen2.5-3B-Instruct with group_size=64, bits=4 (typically ≤3 pp degradation on
  GSM8K).

## 3. Theorem (proof sketch)

**Claim:** If (i) the prompt uses `tokenizer.apply_chat_template(messages,
add_generation_prompt=True)` per the Qwen2.5 chat spec, AND (ii) extraction prefers the
`#### <num>` anchor before fallback to the last number, AND (iii) n ≥ 200 (sampling σ
≤ √(p(1-p)/n) ≈ 3.3 pp at p=0.65), then the measured GSM8K accuracy matches the Qwen2.5
Technical Report baseline within one σ budget plus a 3 pp 4-bit quant penalty.

**Proof:**
- Qwen's reported 68.7% is measured with chatml + `#### <num>` CoT target (Qwen2.5 paper
  appendix). Using identical prompting + extraction isolates quantization as the only
  source of degradation.
- MLX `group_size=64, bits=4` quantization preserves per-group scales/biases; typical
  GSM8K degradation on 3B-class models is ≤3 pp (MLX-community reports, 2025 QA).
- Binomial sampling at n=200, p=0.65: σ = √(0.65·0.35/200) ≈ 0.034. 95% CI ≈ ±6.7 pp.
  At n=200 the signal substantially separates the extraction-bug hypothesis (36% vs 68.7%
  gap = 32.7 pp >> 6.7 pp CI) from the quant-degradation hypothesis (0-3 pp expected).
- Greedy decoding (temp=0) eliminates decoding noise.

**Predicted outcomes:**
| Outcome | Accuracy band | Interpretation |
|---------|---------------|----------------|
| Extraction/prompt was culprit | 62-70% | Fix recovers baseline. Original 36% was artifact. |
| Quantization-native gap | 50-62% | Real but smaller than original 36% suggested. |
| Quality cliff | 35-45% | MLX 4-bit Qwen genuinely underperforms published fp16. |
| Pathological | <35% | Further bug in this rerun. Investigate. |

## 4. Kill criteria (pre-registered, DB IDs)

- **K1547** (DB): Fixed-extraction GSM8K reproduces Qwen baseline within 2 pp at n≥200
  (else extraction was culprit).
  **Operationalization:** Let `acc_new` = our measurement, `acc_baseline = 0.687` (Qwen2.5
  Technical Report Table 8, Qwen2.5-3B-Instruct GSM8K fp16). Pass if `acc_new ≥ 0.667`
  (i.e., within 2 pp of baseline, accepting 4-bit quant penalty as ≤2 pp for strictest
  reading of K1547). Fail otherwise (extraction was the culprit OR 4-bit quality cliff OR
  still-broken prompt).

**Exit branches:**
- Pass (acc_new ≥ 66.7%): supported. Original 36% is retroactively attributable to
  extraction/prompt bug in `exp_competitive_benchmark`.
- Fail (acc_new < 66.7%): killed on K1547. Original 36% is closer to real MLX-4bit
  quality OR rerun still has a hidden bug. Either way, we cannot use original comparison.

## 5. Success criteria

None registered. This is a cleanup; the research question is "was extraction the
culprit?" — answered by PASS vs FAIL on K1547.

## 6. Antipattern self-check

- **ap-001 (composition math bug):** N/A — no composition, single-model eval.
- **ap-002 (tautological routing):** N/A — no router.
- **ap-003 (LORA_SCALE=20 unsafe):** N/A — no LoRA.
- **ap-008 (thinking-mode truncation):** N/A — Qwen2.5 has no thinking mode.
- **ap-017 (stub adapter):** N/A — no adapter.
- **ap-018 (SFT format mismatch):** N/A — inference-only eval.
- **proxy-model-substituted-for-target:** Experiment is explicitly ABOUT the proxy's own
  baseline validity, not using proxy to stand in for target Gemma 4. Claim scope is
  limited to Qwen2.5-3B-Instruct. Does not apply.
- **KC-swap:** MATH.md committed before first run; KC comes from DB (#1547) and is
  quoted verbatim. No post-data editing.
- **smoke-as-full:** n=200 is full (per KC); no smoke reduction.

## 7. Platform / required skills

- Target platform: Apple M5 Pro 48GB, MLX (PLAN.md Part 2).
- Skills invoked: `/mlx-dev` (before coding — inference patterns, lazy eval, memory).
  `/fast-mlx` not needed — this is a one-off eval, not a performance-sensitive training
  loop, and `mlx_lm.generate` already uses mx.compile internally for the prefill/decode
  kernels.

## 8. Baseline anchor

Qwen2.5 Technical Report Table 8 (Qwen2.5-3B-Instruct):
- GSM8K (greedy, 8-shot chat): 68.7%

Anchor used in KC check: **0.687**. Pass threshold: **0.667** (within 2 pp).
