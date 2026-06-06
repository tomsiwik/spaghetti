# MATH: r_95 SNR predictor hits ≥90% within-2x across 5 Gemma 4 domains

## Theorem (pre-registered claim, KC #1586, KC #1587)

Let `{Σ^(d)}_{d=1..5}` be the per-domain gradient-spectrum sequences
emitted during Gemma 4 E4B LoRA training (Adam, PoLAR r=6 on
`v_proj+o_proj`) across five disjoint corpora (math, code, medical,
finance, legal). For each domain `d`, let `r_95^(d)` be the rank that
retains 95% of the signal-to-noise-weighted singular-value energy of
`Σ^(d)` (Finding #154 definition), and let `r*^(d)` be the minimal rank
at which the fine-tuned adapter reaches ≥95% of its own max-rank
behavioral score on a held-out benchmark (GSM8K/HumanEval/MedMCQA/
FinQA/CaseHOLD-dev depending on `d`).

**KC #1586 (within-2x accuracy):** across the 5 domains,
`⟨1[½ ≤ r_95^(d)/r*^(d) ≤ 2]⟩_d ≥ 0.90`.

**KC #1587 (beats null by 20pp):** the within-2x rate of `r_95` beats a
null predictor that outputs a constant rank (e.g. the median r*) by
≥20 percentage points.

## Why (prior-theorem citation)

Finding #154 (`adaptive_rank_snr_fallback`, proven, micro scale) shows
`compound_r95_t2.0` achieves 95.0% mean within-2x across 12 synthetic
conditions (d={64,128,256}, r=8, synthetic spectra), vs 83.3% for r_99
and 57.2% for null. The fallback fires when `r_99/r_95 > 2.0`, adding
+23.3pp mean improvement at SNR≤10.

Mechanism: LoRA fine-tunes the base via a low-rank residual whose
dominant singular directions carry the task-specific update. The
training-gradient SNR spectrum (per-step mean over per-step variance
of the gradient singular values) forecasts which singular directions
survive noise averaging; the r_95 cutoff captures the directions whose
expected SNR exceeds the spectral-MDL threshold of Gavish–Donoho
(2014, `arxiv:1305.5870`). For a well-conditioned optimization, `r*`
sits close to the SNR-predicted knee.

Caveat explicit in Finding #154: "Macro risk: if real training always
produces SNR ≥ 10, the fallback is correct but vacuous." Transferring
to a real 4-bit Gemma 4 base stresses exactly this: 4-bit quantization
elevates the quantization-noise floor, which may shift r_95 away from
r*.

## Failure mode this makes impossible

A null (constant-rank) heuristic over-provisions compute on "narrow"
domains (whose r* is small) and under-provisions on "wide" domains
(whose r* is large). On Gemma 4 E4B 4-bit, getting the per-domain rank
wrong by >2× is predicted to inflate adapter size by 2–4× without
matching behavioral gain (Finding #154 caveat: behavioral loss per
rank mis-set ≈ 1–3pp at SNR=5). A SNR-guided predictor that holds
within-2x on ≥90% of domains makes that over/under-provisioning
failure statistically impossible on this platform.

## Platform

- Base: `mlx-community/gemma-3-4b-it-4bit` (Gemma 4 E4B 4-bit, via MLX-LM)
- Target adapter: PoLAR r=6 on `v_proj+o_proj`, LORA_SCALE=5
  (Finding #586 scale-safety bound; audit-2026-04-17 cohort standard)
- Domains: math, code, medical, finance, legal (disjoint corpora)
- Per-domain rank sweep required for r*: {r=2, r=4, r=6, r=12, r=24}
- Eval harness: GSM8K (math), HumanEval (code), MedMCQA (medical),
  FinQA-dev (finance), CaseHOLD-dev (legal) — all at max_tokens ≥ 512
  (Finding from exp_p1_t2: max_tokens=256 truncates Gemma 4 CoT and
  gives base=0% — format artifact, not measurement).

## Pre-registered preconditions (KC structure)

The r_95 predictor needs three measurable things. Each is pre-registered
as a precondition probe — if any FAIL, KC #1586 and KC #1587 are
**unmeasurable** on current platform state and the experiment is
`KILLED` with blocker recorded (not "inconclusive" / "deferred").

- **P1: five-domain adapter corpus with rank sweep.** The per-domain
  `r*` endpoint requires the adapters at ranks {2,4,6,12,24} on disk
  (25 adapter safetensors) — or at minimum, the five r=6 adapters plus
  per-domain training logs sufficient to recompute a rank-ablation
  behavioral score. Required: ≥5 `adapter_model.safetensors` /
  `adapters.safetensors` weight files with non-zero bytes across the
  five domain directories.

- **P2: training-gradient SNR spectra per domain.** The r_95 predictor
  consumes `Σ^(d)` — per-step gradient singular-value mean / variance.
  Upstream training must log these spectra or save sufficient state
  (per-step gradient L2 plus step count) to reconstruct them.
  Required: a `grad_snr.json` / `training_log.jsonl` under the
  upstream adapter directory with ≥100 steps of SNR records.

- **P3: rank-ablation behavioral baseline per domain.** Computing `r*`
  requires each domain's behavioral score at each rank in the sweep.
  At a minimum the `r=6` behavioral score must be supported (not
  KILLED) so the within-2x window has one measured endpoint.
  Required: upstream training experiment `all_pass=True` with
  non-zero behavioral deltas at r=6 for each domain, not a reconstructed
  format-artifact number.

## Kill criteria (canonical)

- **K1586 PASS:** P1 ∧ P2 ∧ P3 hold AND r_95 within-2x rate ≥ 0.90
  across the five Gemma 4 domains.
- **K1586 FAIL:** P1 ∧ P2 ∧ P3 hold AND r_95 within-2x rate < 0.90 —
  the predictor does not transfer from micro synthetic spectra to real
  Gemma 4 4-bit training.
- **K1586 UNMEASURABLE → KILLED:** any of P1/P2/P3 FAIL — the main
  measurement cannot be evaluated; experiment is KILLED on the probe
  without retries; recovery path documented for v2.
- **K1587 PASS:** K1586 PASS AND r_95 within-2x rate exceeds the null
  predictor's by ≥20pp.
- **K1587 FAIL:** K1586 PASS AND r_95 improvement over null < 20pp.
- **K1587 UNMEASURABLE → KILLED:** inherits K1586's measurability.

## Predicted numbers (registered pre-run)

If preconditions hold:
- Null (constant median r*) within-2x rate ≈ 0.60
- r_99 within-2x rate ≈ 0.75 (transferring from the synthetic finding
  with some 4-bit quantization-noise penalty)
- r_95 within-2x rate ≈ 0.85 (below the synthetic 0.95 due to
  quantization-noise floor; still beats null by ≥25pp if true)

The predicted r_95 hitting 0.85 would FAIL KC #1586 (threshold 0.90).
The registration accepts this as a possible outcome — the predictor
may not transfer — which is itself informative.

## Assumptions

- `audit-2026-04-17` cohort standing rule: heavy retraining
  (≥30min compute per adapter, × 25 adapters = ~12h) is not in scope
  for this hat iteration; precondition-probe KILL is the honest
  outcome when the upstream training hasn't been redone at
  LORA_SCALE=5 with the expanded domain set.
- KC #1586, KC #1587 are locked at pre-registration. Relaxation
  (e.g. to within-4x, or threshold 0.80) invalidates the probe and
  requires a v2 experiment.
- `mlx-lm` version pinned to the version used by
  `exp_p1_t2_single_domain_training` when that upstream eventually
  reruns; version drift invalidates gradient-SNR spectra (API shift
  between mlx-lm 0.21 and 0.31 changes optimizer state layout).
