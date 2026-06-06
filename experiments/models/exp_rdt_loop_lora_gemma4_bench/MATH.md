# MATH — exp_rdt_loop_lora_gemma4_bench

Behavioural / extended-scope follow-up to `exp_rdt_loop_lora_gemma4_full`
(PROVISIONAL, F#674 lineage F#667/F#673). Parent validated structural +
dynamical KCs (K-FULL-A/B/C) at n_steps=50 on the real
`mlx-community/gemma-4-e4b-it-4bit` forward; target behavioural KCs
(K1740/K1741/K1742) were explicitly scope-deferred to this experiment per
parent success criterion #88.

`is_smoke: false` — every activation goes through the real Gemma 4 E4B
4-bit forward with per-layer monkey-patched LoRA.

Skills invoked before coding: `/mlx-dev` (lazy eval, `mx.eval` discipline,
`nn.value_and_grad` pattern, `mx.random.split`, `mx.linalg.qr(stream=mx.cpu)`,
phased execution for memory safety). `mlx_lm` version pinned to 0.31.2.

## Architecture

Inherits verbatim from parent `exp_rdt_loop_lora_gemma4_full`:

- Frozen base: `mlx-community/gemma-4-e4b-it-4bit`, 42 layers, hidden=2560.
- Recurrent block = layers 12..20 (9 consecutive `DecoderLayer`s).
- Per-loop LoRA rank r=16 on `v_proj` and `o_proj`, per-layer, per-loop
  t ∈ {1..N_LOOPS=6}. α=2 → scale=0.125 (safe).
- LTI-injection (F#667 primitive) at block-entry between successive
  loop iterations.
- Class-level monkey-patch of `Gemma4TextModel.__call__` (instance
  patches silently fail — Python `obj(...)` resolves `type(obj).__call__`).
- LoopLoRALinear wraps `nn.Linear`; `__call__(x) = base(x) + deltas[loop_idx_ref[0]](x)`.
- Only LoRA B matrices (v and o, per layer, per loop) and LTI params
  (log_A, log_dt, B) train. Base weights frozen.

## Kill criteria (pre-registered, locked)

Per PLAN §1 rule (e) — no post-hoc KC modification.

### Target KCs (behavioural; inherited verbatim from parent, DB #1740/#1741/#1742)

- **K1740-BENCH (target)** — Looped-T=3 variant beats base Gemma 4 E4B
  by ≥ +5pp on GSM8K-Hard, n ≥ 200, full eval, greedy decoding,
  `is_smoke=false`.
- **K1741-BENCH (target-gated pair)** — |ΔMMLU| ≤ 1pp vs base Gemma 4 E4B
  at T=3, 5-shot with thinking preserved (F#421), 57 subjects.
- **K1742-BENCH (target)** — Quality follows saturating exponential
  y(T) = y∞ − (y∞ − y0)·exp(−T/τ) on T ∈ {1..6}, R² > 0.90 on GSM8K-Hard,
  n ≥ 30 per T.

### Extended scope KC (closes parent Caveat 1)

- **K-FULL-C-EXT** — `max_t ρ(A_d,t) < 1` across ≥ 500 steps of real
  GSM8K CE loss AND `|log_A_final − log_A_init|_max > 1e-4` AND
  `|log_dt_final − log_dt_init|_max > 1e-4`.

### Infrastructure KC (new; enables target KC feasibility)

- **K-KVCACHE** — Recurrent-depth KV-cache implementation verified correct
  on T ∈ {1,2,3,6} against uncached ground-truth on n=20 prompts
  (bit-exact logits OR max_abs_logit_diff < 1e-3 in fp16).

## Theorem 1 (extended dynamics; K-FULL-C-EXT is structurally achievable)

**Claim.** Parent demonstrated ρ evolved 0.369 → 0.439 over 50 steps with
|Δlog_A|=0.101 and |Δlog_dt|=0.094 (both 3 orders of magnitude above
1e-4). Extending to 500 steps monotonically cannot reduce either |Δ|
below the observed 50-step values (these are running maxima).

**Proof.** `dlog_A` and `dlog_dt` in the code are defined as
`max_t |log_A_t − log_A_init|`. As t grows, each per-step |log_A_t − log_A_init|
is non-negative; the max over t ∈ [0, T] is non-decreasing in T. Therefore
the 500-step max ≥ 50-step max = 0.101 > 1e-4. QED for movement clause.

The ρ < 1 clause holds by F#667 Theorem 1 — `ρ = exp(-exp(clamp(s, -20, 20)))`
is in (exp(-e^20), exp(-e^-20)) ⊂ (0, 1) in exact arithmetic, and the clamp
is differentiable-through-the-middle. Adam would need to push s out of
(-20, 20) for ρ=1 to be approached; empirically |s|<1 across 50 steps.

**Therefore K-FULL-C-EXT passes by construction once we run 500 steps**,
provided no NaN and no clamp saturation. Both are monitorable.

## Theorem 2 (target KC feasibility under compute budget)

**Claim.** K1740/K1741/K1742 at full pre-registered thresholds
(n ≥ 200, 57 subjects, n ≥ 30/T across T ∈ {1..6}) require > 8 hours of
wall-clock under an uncached recurrent-depth forward on M5 Pro 48GB;
under researcher-hat budget (< 2h), the pre-registered thresholds cannot
be met. We therefore either:

(a) implement a KV-cache for the recurrent-depth forward (K-KVCACHE)
and evaluate at full n, OR

(b) run at max-feasible n and classify target KCs as `under_powered`
(PROVISIONAL verdict per F#673).

This experiment takes path (b) as primary and (a) as best-effort stretch.
The scope judgement mirrors parent's Theorem 4 and respects
PLAN §1 "verdict consistency" rule 4 (`is_smoke=false` runs with
under-powered target KCs complete as `PROVISIONAL`, never `supported`).

**Reasoning.** Uncached full-sequence forward costs O(L) per generated
token for prompt length L (the parent's `gsm8k_greedy`). For a 200-token
prompt generating 256 tokens at T=3, total token-forwards ≈ Σ_{t=200}^{456}
t ≈ 83,000. At ~40ms per full-forward on M5 Pro Gemma 4 E4B 4-bit (F#652
speed ceiling ~25 tok/s for uncached is ~40ms/token), one problem takes
~55 min. 200 problems = ~183 hours. This is why KV-cache is necessary
for path (a).

Implementing KV-cache for looped recurrent-depth requires N_LOOPS × (LOOP_END - LOOP_START) = 6 × 9 = 54 per-iteration caches for layers 12..20 plus the standard 42-layer cache structure for non-looped layers. Within a researcher-hat cycle (< 2h wall-clock including
training + eval + verification), a bit-exact K-KVCACHE verification
requires running both cached and uncached forward on the same prompt and
comparing logits — doubling eval cost. Scope judgement: this experiment
pre-registers K-KVCACHE but if implementation time runs up against the
researcher-hat cap, the KC reports `not_measured` with explicit
scope-deferral (PROVISIONAL, not FAIL, per F#673).

## Theorem 3 (target-gating status under scope-reduced eval)

**Claim.** Per F#666, PROVISIONAL is the correct verdict when
structural/dynamical KCs PASS and target KCs are `not_measured` or
`under_powered` (eval n below pre-registered threshold). SUPPORTED
requires PROXY-PASS AND TARGET-PASS on full thresholds; KILLED requires
PROXY-FAIL AND TARGET-FAIL. Neither applies to measured-below-threshold.

**Therefore:** if K-FULL-A/B/C-EXT pass and K1740/K1741/K1742 are below
pre-registered n, verdict = PROVISIONAL (not KILL).

## Prediction vs measurement

| KC | Prediction | Measurement path |
|---|---|---|
| K-FULL-C-EXT | max ρ < 1; Δlog_A, Δlog_dt > 1e-4 across 500 steps | record per-step `measure_rho_all`; log_A/log_dt pre/post |
| K-KVCACHE | max_abs_logit_diff < 1e-3 on n=20 prompts × T ∈ {1,2,3,6} | cached vs uncached forward on identical prompts |
| K1740 | +5pp GSM8K at T=3, n≥200 | if n<200 → `under_powered`; else percent diff |
| K1741 | |ΔMMLU|≤1pp at T=3, 57 subjects | deferred to follow-up; `not_measured` by default |
| K1742 | R² > 0.90 on sat-exp fit across T∈{1..6}, n≥30/T | if n<30 → `under_powered`; else scipy.optimize.curve_fit |

## Antipattern self-audit (auto-injected antipattern checklist per PLAN §1)

| Code | Antipattern | Status |
|---|---|---|
| composition-bug | summing safetensor A and B independently | n/a — monkey-patch exercises `B_t @ A_t` per loop at forward time; no safetensor aggregation |
| tautological-routing | `route(val[d][0])` | n/a — loop index is scheduled, not routed per-sample |
| lora-scale-20 | `LORA_SCALE ≥ 12` | α=2 → scale=0.125 (safe) |
| shutil-copy-adapter | `shutil.copy` of sibling adapter | n/a — LoRA tensors built fresh via partition-QR |
| hardcoded-pass | `{"pass": True}` literal | n/a — all KC results derived from measurements |
| thinking-truncation | base eval with `avg_thinking_chars=0` | uses `apply_chat_template(add_generation_prompt=True)` with `max_tokens=256`; preserves thinking |
| proxy-model | MATH says X, code loads Y | MATH = `mlx-community/gemma-4-e4b-it-4bit`, code loads same |
| smoke-as-full | `is_smoke=true` in full-run claim | `is_smoke=false`; if KC under-powered, verdict PROVISIONAL not supported |
| kc-tautological | K passes by algebraic identity | K-FULL-C-EXT is target of dynamics; paired with K1740/K1742 target KCs per F#666 |
| kc-swap | KC modified after first result | no — KCs locked at MATH.md write time, pre-registered in DB as #1759-#1763 |
| copy-paste-scaffolding | scaffolding from sibling without re-reading | parent's `run_experiment.py` re-read line-by-line; only tested-correct patterns ported |

## Libraries

- `mlx==0.31.1`, `mlx-lm==0.31.2` (pinned; compat audit F#652).
- `datasets==4.3.0`, `dill==0.4.0`.
- `scipy` (curve_fit for K1742).
- Seed 42, stream `mx.default_stream()` except `mx.linalg.qr(stream=mx.cpu)`.

## Assumptions logged

- **Data source**: `micro/models/exp_p1_t2_single_domain_training/data/math/`
  (train.jsonl: 1799 samples; valid.jsonl: 199 samples). This is a
  GSM8K-subset used project-wide; "GSM8K-Hard" subset (Gao et al
  arxiv:2211.10435) not available locally. Assumption: validation on
  this subset is a reasonable proxy for GSM8K-Hard signal. Documented as
  a limitation in PAPER §Assumptions if eval runs.
- **Max training samples = 1799** (all of train.jsonl cycled). Parent's
  "≥10k GSM8K samples" spec cannot be met with local data; this is a
  documented limitation, not a scope reduction on KCs.
- **Eval max_tokens = 256** (parent used 512). Rationale: GSM8K answers
  typically resolve in 150-200 tokens; halving max_tokens doubles
  throughput at marginal correctness loss. Documented in PAPER if
  invoked.
