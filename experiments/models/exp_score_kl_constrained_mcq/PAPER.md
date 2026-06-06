# PAPER — exp_score_kl_constrained_mcq

## Verdict: **KILLED** (full scale, K1726 FAIL)

Full-scale rerun completed (2026-04-18, 94 min wallclock). K1724 PASS and
K1725 PASS confirm the **behavioural** SCoRe claim: rank-8 LoRA + β=1.0 KL
constraint preserves base MMLU-Pro+thinking accuracy (adapter 60.0% vs base
58.3%, Δ=+1.7 pp, within the 2 pp bound) **while also** matching or
exceeding the plain-SFT MCQ baseline (60.0% vs 50.4% reference, +9.6 pp).
However, **K1726 FAILS**: max step-wise `KL(π_0‖π_θ) = 0.176 nats` > 0.1 nat
bound (mean KL across 1000 steps = 0.081). Per PLAN.md §1 KC-discipline, any
KC fail at full scale ⇒ `killed`. The KC cannot be relaxed post-hoc.

Structural reading: the **mechanism works** (thinking preserved, MCQ
competitive) but the pre-registered β=1.0 is too weak to clamp per-step KL
within the 0.1 bound at 1000 iters on 27 s1K traces. The load-bearing
behavioural prediction of MATH §3 (H1: "the −11.7 pp Finding #536 gap is
*driven* by out-of-support logit drift") lands on the predicted side —
adapter+thinking is within 2 pp of base, which is the first Gemma-4 adapter
in this repo to pass K1725 since Finding #536. Kill is on the KL-trust-region
numerical bound, not on the mechanism.

## Prediction vs measurement (full scale, n_eval = 60 per arm)

| KC    | Prediction                                               | Measurement                                          | Status |
| ----- | -------------------------------------------------------- | ---------------------------------------------------- | ------ |
| K1724 | Adapter MMLU-Pro+thinking ≥ 50.4 % (plain-SFT baseline)  | 60.0 % (36/60) — **+9.6 pp over baseline**           | ✓ PASS |
| K1725 | Δ accuracy vs base ≤ 2.0 pp                              | Δ = **+1.7 pp** (base 58.3 % → adapter 60.0 %)       | ✓ PASS |
| K1726 | max step-wise `KL(π_0‖π_θ) ≤ 0.1` nats                   | **max_kl = 0.1756 nats** (mean 0.0808) over 1000 steps | ✗ FAIL |

Verdict: KILLED (K1726 FAIL). `all_pass = false`. `is_smoke = false`.

## Behavioural breakdown (full scale)

Per-category (n=20 each):

| Category          | Base acc     | Adapter acc  | Δ     | Base think chars | Adapter think chars |
| ----------------- | ------------ | ------------ | ----- | ---------------- | ------------------- |
| math              | 80.0 % (16/20) | 65.0 % (13/20) | −15 pp | 2350             | 2358                |
| computer science  | 40.0 % (8/20)  | 55.0 % (11/20) | +15 pp | 2570             | 2213                |
| health            | 55.0 % (11/20) | 60.0 % (12/20) | +5 pp  | 3744             | 3054                |
| **overall**       | **58.3 %**   | **60.0 %**   | **+1.7 pp** | **2888**       | **2542**            |

Observations:
- **Thinking preserved.** avg_thinking_chars 2542 (adapter) vs 2888 (base) —
  12 % shorter but not collapsed. Finding #536's −11.7 pp regime was
  associated with near-zero thinking chars; here thinking chains are intact.
- **Asymmetric per-category shifts** (−15, +15, +5) suggest redistribution,
  not uniform improvement. Sample size per category (20) is too small for
  these to be significant individually (± 22 pp 95 % CI on p=0.5, Wilson).
  The overall +1.7 pp at n=60 is also not statistically distinct from zero
  (±12 pp 95 % CI), which is the K1725 intent — overall should be **close to**
  base, not better.
- **β = 1.0 is numerically under-constraining.** K1726 smoke predicted the
  opposite at n_steps=20 (max_kl=0.00257, 40× under bound). Scaling to 1000
  steps pushed max_kl to 0.176, 1.76× over bound. Smoke did **not** linearly
  extrapolate; KL accumulation on LoRA-bounded updates is **super-linear**
  in step count (`arxiv:2409.12917 §4.3`).

## Why this is a genuine kill, not a KC-swap candidate

Pre-registration in MATH.md §KC: *"max KL divergence … is ≤ 0.1 nats at all
logged steps. Binary pass/fail."* This was locked pre-smoke
(`git log micro/models/exp_score_kl_constrained_mcq/MATH.md` shows no edits
since first write). Observed max_kl = 0.176 violates the bound.

The `kc_swap_after_failure` antipattern would apply if we now said "0.176 is
close enough, mechanism works, call it supported." It doesn't. K1724 and
K1725 being clean PASS is **positive downstream evidence** that a follow-up
experiment with β chosen to clamp KL ≤ 0.1 should both (a) still preserve
thinking and (b) stay competitive on MCQ — but that's a v2 experiment, not
this one.

## Structural fix (v2 design note — not for this experiment)

The natural v2 re-derives β from the observed KL curve:
- Observed `mean_kl ≈ 0.081` at β=1.0 over 1000 steps.
- Linearised (SCoRe § Theorem 3.1): `KL ∝ 1/β` for small updates, so raising
  `β → 2.0` is predicted to halve mean_kl to ≈ 0.04, keeping max_kl safely
  under 0.1 (observed max / mean = 2.17, so max_at_β=2 ≈ 0.087).
- K1725 risk at β=2.0: stronger trust-region can under-fit MCQ; v2 must
  re-check that K1724/K1725 margins survive.

Follow-up hypothesis ID candidate: `exp_score_kl_beta2_fullscale` — rerun at
β = 2.0, keep n_steps=1000 and eval_per_cat=20. Do not resurrect this exp.

## Verdict-consistency pre-flight (PLAN.md §1)

1. `results.json["verdict"] = "KILLED"` — consistent with killed.
2. `results.json["all_pass"] = false` — consistent.
3. PAPER.md verdict line reads **KILLED** (not supported).
4. `is_smoke = false` — full-scale run.
5. `git diff MATH.md` shows no KC edits between smoke and full run.
6. Antipattern scan:
   - composition bug: N/A (no composition).
   - tautological routing: N/A (single adapter, no routing).
   - lora-scale: N/A (LORA_SCALE=1.0, safe).
   - thinking-mode truncation: does **not** apply — max_tokens=2048 in
     eval, avg_thinking_chars 2542–2888 (non-zero, chains intact).
   - hardcoded `pass: True`: K-flags computed from measurements.
   - smoke-as-full: `is_smoke=false`, n_eval=60 > smoke n_eval=6.
   - KC-swap-after-failure: K1726 was NOT relaxed despite K1724/K1725
     passing.
   - file-existence cache: run regenerated s1K tokenisation; a1_pass=true.

Pre-flight holds for **killed** verdict (all six items consistent with
KILLED; no antipattern triggers a verdict upgrade).

---

## Smoke archive (superseded)

Below retained for audit trail; superseded by the full-scale rerun above.

### Verdict: **PROVISIONAL** (smoke scale)

Smoke run succeeded mechanically. At smoke scale K1725/K1726 pass, but K1724
smoke-FAILs (adapter 50.0% < 50.4% plain-SFT baseline). `is_smoke=True` ⇒ per
PLAN.md §1.4, verdict is **PROVISIONAL** and the experiment must be rerun at
full scale (`SMOKE_TEST=0`, N_STEPS=1000, EVAL_PER_CAT=20) before
`supported`/`killed` can be declared.

## Prediction vs measurement

| KC    | Prediction                                         | Measurement (smoke, n=6 eval)               | Status (smoke) |
| ----- | -------------------------------------------------- | ------------------------------------------- | -------------- |
| K1724 | Adapter MMLU-Pro+thinking ≥ 50.4% (plain-SFT baseline) | 50.0% (3/6)                           | ✗ FAIL (smoke)  |
| K1725 | |Δ accuracy vs base| ≤ 2.0 pp                      | Δ = +0.0 pp (base 50.0%, adapter 50.0%)     | ✓ PASS (provisional) |
| K1726 | max step-wise `KL(π_0‖π_θ) ≤ 0.1 nats`             | max_kl = 0.00257 nats (all 20 steps logged) | ✓ PASS (provisional) |

## What smoke validates

1. Custom MLX training loop with KL-regularised loss runs end-to-end without
   compile/trace errors. The `nn.value_and_grad(model, kl_loss)` closure
   correctly backprops through the LoRA parameters only — base model
   (`mx.stop_gradient(base_model(…))`) does not receive gradients.
2. Two Gemma-4-E4B-it-4bit instances (base + trainable) fit in memory on M5
   Pro 48GB: peak 8.55 GB active after both loads; no OOM during the 20-step
   run with grad-checkpointing disabled on this path (we save only LoRA).
3. KL magnitude at `β=1.0` remains orders of magnitude below the K1726 bound
   (0.00222 max vs 0.1 bound). This is consistent with SCoRe's theoretical
   prediction (`arxiv:2409.12917` Theorem 3.1): KL concentrates around β-scaled
   Fisher-geometry updates, and for LoRA with rank 8 on v/o projections the
   rank-bounded update produces exceedingly small information-theoretic drift.
4. `adapter_config.json + adapters.safetensors` round-trip cleanly through
   `mlx_lm.load(MODEL_ID, adapter_path=…)`, producing a numerically distinct
   adapted model at eval time (the per-category split differs from base:
   base math 100% / cs 50% / health 50% vs adapter math 100% / cs 100% /
   health 0%).

## What smoke does NOT validate

- **K1724/K1725 are informative-only at n_eval=6.** With 6 total eval samples
  per arm, a ±16.7 pp change is within binomial noise. Base 66.7% and adapter
  66.7% could be drawn from very different underlying distributions.
- **K1726 at n=20 steps tells us nothing about longer training.** The paper
  reports full runs at 1000 iters; KL accumulation scales superlinearly if
  β is too small (arxiv:2409.12917 §4.3). We cannot infer from smoke that
  `max KL ≤ 0.1` will still hold at step 1000.
- **Thinking-mode preservation (the load-bearing behavioral claim).** The
  per-category health 50% → 0% swing on 2 samples is consistent with
  random, but could also indicate the adapter is shifting behavior in a
  thinking-mode-relevant way. N=20 per category is the minimum needed for
  a ±5pp CI.
- **`avg_thinking_chars = 0` on both arms.** Neither base nor adapter
  produced matches against `<think>` or `<|channel|>thought` regex. This is
  either (a) Gemma 4 E4B 4-bit at 4-bit not emitting visible thinking tokens
  at all, or (b) a regex/format mismatch — needs investigation before the
  full run. MATH §3 of parent experiment noted this; the full rerun must
  confirm the right channel pattern is captured.

## Assumptions logged

- `β = 1.0` chosen by literature, not swept. If full-scale rerun shows
  K1726 failing (KL > 0.1), the fix is β ↑ — but that may break K1724/K1725
  by over-constraining learning. That would be a **structural** result (rank-8
  LoRA on v/o cannot simultaneously learn s1K *and* preserve base), not a
  hyperparameter bug.
- KL computed over **all** answer-span tokens (broader than SCoRe's
  first-token variant), so our bound is stricter than the paper's.
- Base model double-loaded — this doubles 4-bit weight memory (~4.3 GB → 8.55 GB
  observed). On M5 Pro 48 GB this is fine; on smaller Macs the experiment
  would need weight sharing or swapping.

## Full-scale rerun plan

```
SMOKE_TEST=0 uv run python micro/models/exp_score_kl_constrained_mcq/run_experiment.py
```
- `N_STEPS = 1000`
- `EVAL_PER_CAT = 20` → 60 total MMLU-Pro evals per arm
- `BETA_KL = 1.0` (unchanged)
- Expected wallclock: ≈ 50 min (train) + 10 min (2×eval) ≈ 1 h on M5 Pro.

Pre-flight before full-scale `--status supported`: verify all six items of
PLAN.md §1 (verdict field not "KILLED"; all_pass true; no PROVISIONAL in
this doc; is_smoke=false; KC unchanged per git diff MATH.md; no antipattern
applies).

## References

- Kumar et al., `arxiv:2409.12917` (SCoRe stage-I KL-constrained SFT).
- Finding #536 — MCQ adapter suppresses thinking, −11.7 pp.
- `exp_model_thinking_preservation_training` — parent (killed at smoke without KL).
- `mem-antipattern-008` — thinking truncation; `max_tokens = 2048` in eval.
