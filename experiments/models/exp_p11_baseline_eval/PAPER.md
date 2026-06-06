# PAPER: exp_p11_baseline_eval (P11.E0)

**Verdict: NOT SUPPORTED (KILLED)**

Pre-registered KCs K1505 and K1506 falsified. K1507 passes but is meaningless
without K1505. See `REVIEW-adversarial.md` for the full adversarial check and
`LEARNINGS.md` for follow-up implications.

## One-line
Intended to fill the adapter×benchmark registry for 5 P1 adapters and anchor
base Gemma-4 4-bit MMLU-Pro+thinking; instead revealed (a) the 5 P1 adapter
safetensors were never persisted on disk, and (b) base MMLU-Pro+thinking is
40.7%, 21.4pp below Finding #530's 62.1% prediction.

## Setup
- Model: `mlx-community/gemma-4-e4b-it-4bit` (MLX 0.31.1, mlx-lm stock `load`/`generate`).
- Benchmarks: MMLU-Pro (14 categories × 20 Q = 280 Q) in thinking=OFF and thinking=ON;
  GSM8K (n=50, thinking=ON) via `datasets-server.huggingface.co`.
- Conditions: base + 5 existing knowledge adapters (math-gsm8k, code-codealpaca,
  medical-medmcqa, legal-mmlu, finance-mmlu).
- Thinking budget: 2048 tokens. MCQ parser: last A–J token in the full completion.
- `smoke_test: false`. `eval_per_cat: 20`. `total_time_s: 4698.6` (~78 min wall).

## Predictions vs Measurements

| Condition | Metric | Predicted (MATH.md) | Measured | Delta | Status |
|---|---|---|---|---|---|
| Base | MMLU-Pro (thinking=OFF) | ~40% | **45.0%** (126/280) | +5.0pp | within band |
| Base | MMLU-Pro (thinking=ON) | ~62.1% ±5pp (Finding #530) | **40.7%** (114/280) | **−21.4pp** | **FAIL (K1506)** |
| Base | GSM8K (thinking=ON) | ~77% | HTTP 422 (datasets-server) | n/a | external failure |
| Base | avg_thinking_chars (ON) | >0 | **2931** | — | not truncated ✓ |
| math-gsm8k-knowledge-v0 | MMLU-Pro + GSM8K | scored in both thinking modes | **load error** | — | **FAIL (K1505)** |
| code-codealpaca-knowledge-v0 | — | scored | **load error** | — | FAIL |
| medical-medmcqa-knowledge-v0 | — | scored | **load error** | — | FAIL |
| legal-mmlu-knowledge-v0 | — | scored | **load error** | — | FAIL |
| finance-mmlu-knowledge-v0 | — | scored | **load error** | — | FAIL |
| Registry | `registry.json` updated | updated | file touched (no adapter rows added) | — | K1507 passes vacuously |

All 5 adapter loads returned `[load_safetensors] Failed to open file
…/adapters.safetensors`. On-disk inspection: every P1 adapter directory
contains only `adapter_config.json`; the weight files were never written.

## Kill criteria (pre-registered)

- **K1505** — "all 5 adapters evaluated": FAIL. value=0.
- **K1506** — "base MMLU-Pro+thinking ≈62.1% ±5pp": FAIL. value=40.7%, |Δ|=21.4pp.
- **K1507** — "registry.json updated": PASS in isolation; non-load-bearing
  because no adapter rows were fillable.

Pre-flight per PLAN.md §1 (verdict consistency):
- results.json: 2/3 KCs fail → not eligible for `supported`.
- PAPER.md verdict: `NOT SUPPORTED (KILLED)`.
- `is_smoke: false`, full run, no KC edited post-run (MATH.md single commit de38e37).
- Antipattern `(n)` (thinking truncated → false gain): avg_thinking_chars=2931, **not** triggered.

## Per-category base MMLU-Pro (both modes)

| Category | OFF (acc %) | ON (acc %) | Δ (ON − OFF) |
|---|---:|---:|---:|
| biology | 80.0 | 60.0 | −20.0 |
| business | 10.0 | 45.0 | +35.0 |
| chemistry | 35.0 | 30.0 | −5.0 |
| computer science | 60.0 | 45.0 | −15.0 |
| economics | 45.0 | 45.0 | 0.0 |
| engineering | 25.0 | 35.0 | +10.0 |
| health | 45.0 | 50.0 | +5.0 |
| history | 45.0 | 35.0 | −10.0 |
| law | 35.0 | 10.0 | −25.0 |
| math | 40.0 | 35.0 | −5.0 |
| other | 70.0 | 65.0 | −5.0 |
| philosophy | 40.0 | 20.0 | −20.0 |
| physics | 20.0 | 40.0 | +20.0 |
| psychology | 80.0 | 55.0 | −25.0 |
| **mean** | **45.0** | **40.7** | **−4.3** |

Thinking helps business/engineering/physics but hurts biology/law/philosophy/psychology
and is net-negative at this sample size (n=20/cat).

## Interpretation

1. **K1505 is a data-integrity failure**, not an experiment-code bug: the P1
   training experiments (`exp_p1_t2_single_domain_training`,
   `exp_p1_t2_multi_domain_5`) never persisted `adapters.safetensors`.
   Pre-flight only verified directory existence. Cannot be repaired by editing
   P11.E0; requires re-running P1 training with a `save_weights` assertion.
2. **K1506 is a substantive finding**: with a plain "Answer:" chat prompt on
   4-bit Gemma-4, thinking ON is net-negative on MMLU-Pro. Three non-exclusive
   causes to disentangle in a follow-up: (i) prompt-format drift vs #530,
   (ii) MCQ parser picking the last A–J token in a 3000-char trace,
   (iii) 4-bit quant of a non-thinking-trained base may not benefit from the
   thinking channel. None were controlled in this measurement.
3. **GSM8K HTTP 422** is a datasets-server brittleness; local cache required
   before this benchmark is trustworthy inside the hat loop.

## Assumptions (logged per workflow §3)

- Treating Finding #530's 62.1% as the canonical target even though the
  original eval protocol (prompt template, parser regex, quant level) is not
  re-verified here. The K1506 miss therefore measures *our* protocol vs *their*
  number, not the base model's intrinsic capability.
- GSM8K HTTP 422 is not counted against K1505 (which is adapter-specific).
  The secondary base-GSM8K prediction (~77%) is simply unmeasured this run.
- `K1507` is reported as `true` because the script did write to
  `registry.json`; it is flagged as vacuous because no adapter rows were added.

## Follow-ups (for Analyst / next experiments)

1. Open `exp_p1_retrain_with_persist` (P≤1): re-run the P1 training
   experiments with a post-train assertion on `adapters.safetensors` size > 0.
2. Open prompt-format ablation for base Gemma-4 4-bit MMLU-Pro+thinking to
   close the 21pp gap vs Finding #530.
3. Local-cache GSM8K test split to remove datasets-server dependency from all
   future baseline evals.
4. Re-run P11.E0 once (1) is done; base condition does not need to be
   repeated unless the prompt format changes.
