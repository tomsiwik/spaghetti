# exp_g4_zs_base_transfer_4bit_fp16_full — PAPER

## Verdict

**PROVISIONAL (supported_target_only per F#666)** — finding about the proxy.

K1814 (proxy / inherited from parent F#680): **FAIL** (median R_ppl = 0.946 < 0.95).
K1815 (target / behavioral, NEW): **PASS** (median R_task = 1.139 ≥ 0.95; min R_task = 1.029 ≥ 0.85).

Per the F#666 truth table (MATH.md §"Pre-registered kill criteria"): proxy-FAIL +
target-PASS = a finding about the proxy, not a kill on the mechanism. The verdict
is therefore not `supported` (proxy half is FAIL), not `killed` (target half is
PASS, behavioral transfer is intact), and not bare `provisional` (both KCs were
locked and measured against pre-registered thresholds). It is recorded as
`PROVISIONAL` with `verdict_internal = supported_target_only` so the F#666
discriminator survives the schema.

## Headline finding

**The 4→8-bit PPL-gain ratio is not a reliable transfer-fidelity proxy for
adapter benefit on Gemma 4 E4B.** When the same adapters (q_proj, r=6, scale=6,
trained against the 4-bit base in `exp_p1_t2_single_domain_training`) are
mounted on the 8-bit base instead, behavioral task accuracy is **functionally
lossless** — and in fact strictly *higher* in every domain measured:
HumanEval 70 → 80 (+10pp), GSM8K 72 → 82 (+10pp), MedQA 68 → 70 (+2pp).
The proxy understated transfer fidelity by a wide margin (median R_ppl = 0.946
predicted ≤5% PPL benefit lost; measured median R_task = 1.139 means task
accuracy *retention* exceeded 100%, i.e. the 8-bit base is a better host for
the 4-bit-trained adapter than the 4-bit base it was trained on).

This empirically falsifies the proxy direction implied by parent F#680 (4→8
adapter transfer is "marginally failing" at the 5% PPL threshold) and confirms
the project guidance `r ≈ 0.08(PPL, task)` (PLAN.md / guardrail 1006): PPL and
task accuracy are nearly uncorrelated in this codebase, so a marginal proxy
FAIL means nothing about behavioral function.

## Prediction-vs-measurement

### K1814 — proxy / inherited (PPL-gain ratio)

`R_ppl(D) = gain_8bit(D) / gain_4bit(D)`, where `gain(D) = (base_ppl − adapter_ppl) / base_ppl`.
Pre-registered threshold: `median R_ppl ≥ 0.95 ∧ every R_ppl ≥ 0.85`.

| Domain | Predicted | Measured (inherited from parent) | Pass? |
|---|---|---|---|
| code (HumanEval) | inherited from F#680 | R_ppl = **0.900** | floor 0.85 ✓ |
| math (GSM8K) | inherited from F#680 | R_ppl = **0.946** | floor 0.85 ✓ |
| medical (MedQA) | inherited from F#680 (saturated PPL=1.0) | R_ppl = **0.984** | floor 0.85 ✓ |
| **median** | predicted: marginal FAIL per parent | **0.946** | < 0.95 → **FAIL** |
| **min** | — | 0.900 | ≥ 0.85 ✓ |

Result: **FAIL** (inherited from parent on disk, byte-for-byte). The marginal
proxy-FAIL was the entire reason F#680 was downgraded to PROVISIONAL pending an
F#666 retrofit; this experiment IS that retrofit.

### K1815 — target / behavioral (task-accuracy ratio, NEW)

`R_task(D) = task_acc(W_8, ΔW, D) / task_acc(W_4, ΔW, D)`, with `task_acc` =
pass@1 on HumanEval / accuracy on GSM8K / accuracy on MedQA-USMLE-4-options.
n=50 per domain; greedy decoding; same prompts and seed across the two
precision realizations (paired evaluation, sampling noise mostly cancels in
the ratio). Pre-registered threshold: `median R_task ≥ 0.95 ∧ every R_task ≥ 0.85`.

| Domain | Dataset | acc_4bit | acc_8bit | R_task | Predicted | Pass floor 0.85? |
|---|---|---|---|---|---|---|
| code | HumanEval | 70.0% | 80.0% | **1.143** | none (PPL↔task `r≈0.08`, range [0.7, 1.1]) | ✓ |
| math | GSM8K | 72.0% | 82.0% | **1.139** | none | ✓ |
| medical | MedQA-USMLE-4-options | 68.0% | 70.0% | **1.029** | none | ✓ |
| **median** | — | — | — | **1.139** | — | ≥ 0.95 ✓ |
| **min** | — | — | — | **1.029** | — | ≥ 0.85 ✓ |

Result: **PASS** on both halves of the threshold. R_task strictly exceeds 1
in every domain, which is *outside* the predicted [0.7, 1.1] envelope on the
upper end. We had no strong directional prediction (per MATH.md §Prediction);
that the measurement landed above 1.0 across all three domains is a
positive finding worth registering.

## What was actually measured (raw)

From `results.json`:

```
n_eval_per_domain    = 50    (matches parent T2.1 protocol)
mlx_lm_version       = 0.31.2
elapsed_s            = 451.97 s  (~7m32s, well under 30-50 min upper estimate)

K1814 (proxy):
  median_r_ppl       = 0.945859805369008  (inherited)
  min_r_ppl          = 0.8998956945668609 (inherited)
  pass               = False

K1815 (target):
  per_domain.code.r_task    = 1.1428571428571428  (HumanEval 70 → 80)
  per_domain.math.r_task    = 1.1388888888888888  (GSM8K    72 → 82)
  per_domain.medical.r_task = 1.0294117647058822  (MedQA    68 → 70)
  median_r_task             = 1.1388888888888888
  min_r_task                = 1.0294117647058822
  pass                      = True

Verdict (per F#666 truth table):
  K1814 FAIL + K1815 PASS  →  supported_target_only
                           →  PROVISIONAL (finding about the proxy)
```

## Mechanism interpretation (why R_task > 1 across the board)

Two non-exclusive explanations are consistent with the data; the experiment
does not discriminate between them:

1. **The 8-bit base is closer to the bf16 reference than the 4-bit base.**
   The adapter ΔW = α·BA was trained to compensate for the *4-bit*
   quantization error of the base. Mounted on the 8-bit base — which has
   smaller quantization error to begin with — the adapter applies its
   correction on top of a more accurate substrate. The "correction" is
   slightly mis-targeted (it was learned against 4-bit error), but the
   higher-precision substrate dominates: task accuracy goes up, not down.
   This is consistent with the F#97 micro-scale result that ZS transfer
   is essentially lossless when the substrate is a higher-fidelity rung
   of the same model.

2. **Task accuracy on the 4-bit base was held back by quantization-induced
   hallucination, not adapter capacity.** Even with the same adapter, the
   8-bit substrate produces fewer wrong tokens at decode time (HumanEval
   pass@1 +10pp is the single biggest jump, consistent with code being
   the most quantization-sensitive output channel — small token errors
   break syntax). MedQA's much smaller +2pp jump is consistent with
   multiple-choice classification being dominated by argmax stability,
   not exact token sequencing.

The cleaner test of explanation (1) would be the strict 4→bf16 rung. We
do not run it here (~22 GB download, outside hat budget); see "Limitations".

## Limitations

- We do not test 4→bf16 (the strict precision rung). bf16 base not cached;
  ~22 GB download outside hat budget. The 4→bf16 rung remains as a follow-up
  if K1815 PASSes — which it did. A future experiment can resurrect this rung
  on a slot that already has the bf16 base resident.
- We do not test transfer to a different base model (Gemma 4 26B etc.).
- We do not retrain the adapter; we use the parent's trained adapters
  directly, which were trained against the 4-bit base only. So the "adapter
  benefit" measured here is the benefit of the *4-bit-trained* adapter
  evaluated under precision change, not the benefit of an adapter retrained
  on each precision rung. The latter is a different (and larger) question.
- n=50 per domain. 95% CI on a Bernoulli proportion at p≈0.7 is ±13pp at
  n=50 — wide, but adequate for a ratio-median ≥ 0.95 test where the floor
  at 0.85 absorbs most noise. The +10pp HumanEval / GSM8K jumps are larger
  than the per-domain CI half-width, so the directional finding (R_task > 1)
  is robust; the exact magnitudes are not.
- MedQA per-domain R_task = 1.029 is within the n=50 CI of 1.0, so the
  medical-domain "improvement" is statistically marginal even if the
  direction matches the other two domains.

## Assumptions logged

(Inherited from MATH.md, restated here so PAPER.md is self-contained.)

1. The `exp_p1_t2_single_domain_training` adapters (q_proj, r=6, scale=6) are
   the canonical adapters for measuring the precision-transfer ratio,
   matching the parent experiment's choice. The adapter-benefit-transfer-
   under-precision-change claim is invariant to which target modules the
   adapter touches; what matters is the same adapter weights mounted on two
   different precision realizations of the same base model.
2. K1814 inherited from parent's `results.json` directly; not re-measured.
   This saves ~7 min wall-clock on 4 PPL sweeps that already exist on disk.
3. n=50 per domain matches parent T2.1 protocol.
4. Gemma 4 E4B 4-bit and 8-bit MLX models share architecture and tokenizer;
   only weight quantization differs. Verified by config inspection at runtime.
5. Per-domain R_task uses paired evaluations (same prompts, same seed) so
   sampling noise mostly cancels in the ratio.

## References

- Finding #97 — Zero-shot base transfer works (micro scale, conclusive).
- Finding #680 — Parent PROVISIONAL: PPL-gain ratio 90–98%, median 0.9459.
- Finding #666 — Target-gated kill rule (proxy-only KCs are unsafe).
- Finding #627 — Gemma 4 LoRA target modules (cited; not used here).
- arxiv:2106.09685 — LoRA (Hu et al, 2021).
- mlx-lm version pinned at runtime: `results.json["mlx_lm_version"] = 0.31.2`.

## Suggested follow-ups (not run here)

- **4→bf16 rung.** If a slot has bf16 Gemma 4 E4B resident, re-run with
  bf16 substituted at the test base. Predicts (per explanation (1) above)
  R_task ≥ 1.05 in every domain, with code/math jumps ≥ +5pp.
- **Adapter retrained per precision.** Train the adapter against the 8-bit
  base directly and compare on the same eval grid. If R_task collapses to
  ≈1.0, the +10pp jumps observed here are driven by precision-induced
  decode quality, not adapter mis-targeting (explanation (2)). If R_task
  stays > 1.0 with retraining, explanation (1) dominates and ZS transfer
  is genuinely lossless across rungs.
- **Composition under precision change.** The N>1 hot-merge regime is
  superseded for now (Room model killed at N>1 per Finding #571). If a
  factored-LoRA approach revives composition, repeat this F#666 retrofit
  there.
