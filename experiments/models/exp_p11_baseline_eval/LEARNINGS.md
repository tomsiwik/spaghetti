# LEARNINGS: exp_p11_baseline_eval (P11.E0)

**Status: KILLED** — 2/3 pre-registered KCs falsified (K1505, K1506).

## Core Finding
The "registry-fill" design is sound, but the registry is unfillable today: every P1 adapter directory contains only `adapter_config.json` — `adapters.safetensors` was never persisted. Of the secondary measurement that did run, base Gemma-4 4-bit MMLU-Pro+thinking is **40.7%** (114/280), **−21.4pp** below the 62.1% claimed by Finding #530 — and **−4.3pp below thinking=OFF** on the same model. avg_thinking_chars=2931, so this is a real measurement, not antipattern (n) truncation.

## Why
1. **K1505 (data integrity, not code).** P1 single-domain and multi-domain training experiments were closed without saving weights. Pre-flight check verified directory existence but not the weight file. Cannot be repaired by editing this experiment.
2. **K1506 (cited baseline drift).** Finding #530's 62.1% was inherited as a prediction without re-deriving the protocol. Differences in prompt template, max_tokens, MCQ parser regex, or the 4-bit quant level are all candidates — but none were controlled here. The cited number is not load-bearing evidence in our setup.
3. **Bonus failure.** Datasets-server returned HTTP 422 on GSM8K → eval pipeline brittle to external service availability.

## Implications for Next Experiment
- **Block all P11.A0/A1 adapter comparisons** until P1 training is re-run with a `save_weights` step that asserts `adapters.safetensors` exists post-train. New experiment: `exp_p1_retrain_with_persist` (priority ≥ P=1).
- **Do not cite Finding #530's 62.1% again** without a re-measured base under the same prompt/parse/quant. Treat 40.7% as the current empirical anchor for 4-bit Gemma-4 + thinking + plain-MCQ prompt.
- **Localize GSM8K test set** to remove datasets-server dependency (small unblocker).
- The next baseline-eval re-run can be much smaller: only adapters whose weights actually exist need scoring; the base condition does not need to be repeated unless the prompt format changes.
