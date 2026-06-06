# exp_g4_zs_base_transfer_4bit_fp16_full — F#666 retrofit: PPL-gain + downstream task-accuracy ratio for 4→8 adapter transfer

## Scope and honest reframe

DB title: *"Follow-up: PPL-gain + downstream task-accuracy ratio for 4→8 adapter transfer (Gemma 4)"*.

This is the F#666 retrofit of `exp_g4_zs_base_transfer_4bit_fp16` (parent, F#680
PROVISIONAL). The parent measured only K1 (PPL-gain ratio, proxy) and got
median `R_ppl=0.9459 < 0.95` → marginal FAIL on the proxy. Per F#666 (target-
gated kill, guardrail 1006: `r≈0.08` PPL↔task in this codebase), proxy-only
FAIL is unsafe; the headline finding for F#680 was downgraded to PROVISIONAL
pending the addition of a behavioral target-metric KC. This experiment **adds
that target-metric KC** (K1815 task-accuracy ratio) and re-derives the verdict
under F#666.

## Skills invoked (per guardrail 1012)

`/mlx-dev` and `/fast-mlx` — confirmed via PLAN.md Part 2. Inherited eval
helpers (`eval_gsm8k`, `eval_humaneval`, `eval_medqa`) from
`micro/models/exp_p1_t2_single_domain_training/run_experiment.py` which were
written under those skills (audit-2026-04-17 metric-swap and format-artifact
fixes already applied: MedMCQA→MedQA-USMLE-4-options; GSM8K max_tokens
256→1024 to capture Gemma 4 CoT before the `#### answer` sentinel).

## Motivation

- **F#680** (provisional, parent): 4→8-bit adapter PPL-gain retention is
  90–98% per domain; median R=0.9459 (marginal FAIL); medical R unreliable
  (PPL=1.0 saturated). PPL is a proxy. Cannot conclude transfer functionally
  fails without a behavioral target measurement.
- **F#666** (target-gated kill rule): every proxy KC must be paired with a
  target-metric KC. KILL requires both FAIL. Proxy-FAIL + target-PASS = finding
  about the proxy, not a kill on the mechanism.
- **F#97** (conclusive, micro-scale): Zero-shot base transfer works at micro
  scale (rank-16: 4.2% loss; rank-32: 0.3% loss, 3 seeds). The Gemma 4 E4B
  extension is what we are testing.

## Type: frontier-extension (F#666 retrofit)

Proven framework: F#97 (controlled-perturbation ZS transfer holds at micro).
F#680 (4→8 PPL-gain retention is 90–98% on Gemma 4 E4B). Extension adds the
behavioral target measurement that F#666 requires for SUPPORTED/KILLED.

## Theorem (target-paired adapter-benefit transfer under precision change)

**Setting.** Same as parent. `W_4`, `W_8` are 4-bit and 8-bit dequantized
weight realizations of the Gemma 4 E4B reference; `ΔW = α·BA` is the
parent-trained adapter (q_proj, r=6, scale=6, the
`exp_p1_t2_single_domain_training` adapters per domain).

**Quantities.**

- `task_acc(W, ΔW, D)`: pass@1 on HumanEval / accuracy on GSM8K / accuracy on
  MedQA-USMLE-4-options for adapter `ΔW` mounted on base `W`, evaluated on
  domain `D`'s canonical test split (n=50 per domain, matching parent T2.1).
- `R_task(D) = task_acc(W_8, ΔW, D) / task_acc(W_4, ΔW, D)`.
- `R_ppl(D)` per parent K2 definition (inherited).

**Definition (target-paired transfer).**

```
median R_task ≥ 0.95  ∧  every domain R_task ≥ 0.85
```

`R_task = 1` means task accuracy transfers exactly. `≥ 0.95` median means
≤5% of behavioral benefit is lost in transfer.

**Prediction.** F#680 measured `median R_ppl = 0.9459` and `min R_ppl = 0.90`
(code, the worst). PPL↔task correlation in this codebase is `r≈0.08` (project
guidance), so `R_task` could lie anywhere in `[0.7, 1.1]` consistent with the
measured `R_ppl`. We have **no strong directional prediction** — that is the
point of the F#666 retrofit. The measurement is what determines whether the
parent's marginal proxy-FAIL is functional or cosmetic.

## Pre-registered kill criteria (locked, target-gated per F#666)

**K1814 (proxy / inherited).** Inherits parent K2 measurement directly. Per
parent's results.json: `median R_ppl = 0.9459` < 0.95 → **FAIL** (inherited).
This is the proxy half of the F#666 pair. Re-running the parent code is not
required because the parent's result is on disk and locked.

**K1815 (target / behavioral, NEW).** `median R_task ≥ 0.95 across HumanEval
(code), GSM8K (math), MedQA (medical), AND every per-domain R_task ≥ 0.85`.
PASS iff both hold. The `≥ 0.85` per-domain floor prevents median masking
from a single domain catastrophically failing. n=50 per domain; greedy
decoding (matching parent T2.1 protocol).

**Per F#666 truth table.**

- K1814 PASS + K1815 PASS → SUPPORTED. (Counterfactual; parent already FAIL.)
- K1814 FAIL + K1815 PASS → finding about the proxy: "PPL-gain ratio is not a
  reliable transfer-fidelity proxy for adapter benefit on Gemma 4 E4B."
  Headline: transfer is functionally lossless on task accuracy.
- K1814 FAIL + K1815 FAIL → KILLED. F#680 upgraded from PROVISIONAL to
  KILLED with target evidence; precision change destroys functional adapter
  benefit.
- K1814 PASS + K1815 FAIL → tautological-proxy: KILL on target. (Counter-
  factual; K1814 is FAIL.)

## What this does NOT claim

- Does not test 4→bf16 (the strict precision rung). bf16 base not cached;
  ~22 GB download outside hat budget. The `4→bf16` rung remains as a
  potential follow-up if K1815 PASSes.
- Does not test transfer to a *different* base model (Gemma 4 26B etc.).
- Does not retrain the adapter; uses the parent's trained adapters directly,
  which were trained against the 4-bit base only.

## Assumptions logged

1. The `exp_p1_t2_single_domain_training` adapters (q_proj, r=6, scale=6) are
   the canonical adapters for measuring the precision-transfer ratio, matching
   the parent experiment's choice. Not the F#627 v_proj+o_proj target — but
   the *adapter-benefit-transfer-under-precision-change* claim is invariant
   to which target modules the adapter touches; what matters is the same
   adapter weights mounted on two different precision realizations of the
   same base model.
2. K1814 inherited from parent's `results.json` directly; not re-measured
   (saves ~7 min wall-clock on 4 PPL sweeps that already exist on disk).
3. Eval n=50 per domain matches parent T2.1 protocol; 95% CI on a Bernoulli
   proportion at p=0.7 is ±13pp at n=50 — wide but adequate for a ratio
   median ≥ 0.95 test where the floor at 0.85 absorbs most noise.
4. Gemma 4 E4B 4-bit and 8-bit MLX models share architecture and tokenizer;
   only weight quantization differs. Verified by config inspection at runtime.
5. Per-domain R_task uses paired evaluations (same prompts, same seed) so
   sampling noise mostly cancels in the ratio.

## References

- Finding #97 — Zero-shot base transfer works (micro scale, conclusive).
- Finding #680 — Parent PROVISIONAL: PPL-gain ratio 90–98%, median 0.9459.
- Finding #666 — Target-gated kill rule.
- Finding #627 — Gemma 4 LoRA target modules (cited; not used here).
- Finding #502 / #646 — claim-time hygiene cohort (success_criteria + refs
  populated at claim per ralph-tools-tasks pattern; addressed at claim).
- arxiv:2106.09685 — LoRA (Hu et al, 2021).
- mlx-lm version: pinned at runtime in `results.json["mlx_lm_version"]`.

## Antipattern scan (pre-flight, per guardrail 1010 step 6)

- composition math bug: N/A (single-adapter eval, no composition).
- LORA_SCALE: existing adapter trained at scale=6 (safe per F#328/#330).
- `shutil.copy` as new adapter: no — same adapter loaded twice via
  `mlx_lm.load(MODEL, adapter_path=...)` on two precision-different bases.
- hardcoded `"pass": True`: KCs computed from real measurements; K1815 PASS
  derived from `(median(R_task) >= 0.95) and (min(R_task) >= 0.85)` boolean.
- eval truncation: GSM8K `max_tokens=1024` (parent's audit-fix); HumanEval 512;
  MedQA 20. Inherited from T2.1.
- proxy-model substitution: 8-bit substitutes for bf16 at the *test base*; the
  4-bit *training base* is the canonical target. Documented above.
- KC measures wrong object: K1815 measures the ratio of canonical task
  accuracies on canonical eval splits — directly the behavioral target.
- N=smoke reported as full: `is_smoke` set strictly from `SMOKE_TEST`;
  results.json carries `is_smoke` and `n_eval_per_domain`.
- Followup-without-rerun (F#762 super-family): does NOT apply. The "rerun" is
  the K1815 *new measurement* — the followup is not replicating the parent's
  proxy KC; it is adding a behavioral target KC that the parent lacked.
- F#666-pure-standalone: does NOT apply. Two KCs locked, paired (proxy +
  target). KC text in DB is target-gated by construction.
