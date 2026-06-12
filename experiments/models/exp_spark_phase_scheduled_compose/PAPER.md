# PAPER — Decode-step phase-scheduled adapter mixing recovers most of the static-merge gap

`exp_spark_phase_scheduled_compose` · verdict: **SUPPORTED** (pre-registered gates) · `all_pass: true` · `is_smoke: false`

## Claim

On a reason-then-emit-strict-JSON task, the optimal math/code adapter mix is non-stationary in
decode time. A live phase schedule (math adapter dominant during chain-of-thought, code adapter
dominant during JSON emission), holding total injected weight constant at 1 every step
(`w_m(t)+w_c(t)=1`), **recovers most of the interference gap that a static 0.5/0.5 merge opens up**
(scheduled 0.817 vs static 0.633, +18.3pp) purely by *when* it spends each unit of weight — not by
injecting more signal.

**Scheduling does NOT beat the best single adapter.** Math-only (0.850) is strictly above scheduled
(0.817). The pre-registered gates (kill 2308) are about the static merge, and they pass; but the
honest headline is narrower than "composition wins": decode-step timing recovers most of the
static-merge interference gap, yet does **not** reach the single-adapter ceiling. See the
head-to-head below.

## Setup (real, frozen)

- Base: `mlx-community/gemma-4-e4b-it-4bit`, frozen. mlx-lm 0.31.2.
- Adapters: `adapter_math.safetensors` (GSM8K/CoT), `adapter_code.safetensors` (HumanEval/JSON),
  r=6 q_proj-only, from `exp_composition_residual_analysis/`.
- Composition `Σ_i (B_i A_i)`, `LORA_SCALE = 6.0`, ε-leak = 0.1.
- n = 60 constructed arithmetic word problems; answer must be strict JSON `{"answer": <int>}`.
- Phase detector is the LIVE `emitted_open_brace` flag from generated tokens (NOT oracle).
- Combined score = CoT-correct AND JSON-valid (the kill metric).

## Prediction vs. measurement

### Per-arm results (n=60)

| Arm              | CoT-correct | JSON-valid | Combined |
|------------------|-------------|------------|----------|
| math_only        | 0.850       | 0.850      | **0.850** |
| code_only        | 0.433       | 1.000      | **0.433** |
| static_0.5/0.5   | 0.633       | 0.983      | **0.633** |
| scheduled        | 0.817       | 0.817      | **0.817** |

The static merge dilutes both skills: CoT drops to 0.633 (below math-only's 0.850) and the combined
score collapses to 0.633. The scheduled arm recovers CoT to 0.817 while keeping JSON parseable on the
same items. The scheduled arm is the only composed arm whose `brace_rate` is non-zero (0.817) — i.e.
it actually entered the EMIT phase on those items via the live detector, confirming the schedule
fired on model-generated braces, not an oracle.

### Head-to-head: scheduled vs. best single (math-only)

Per-item combined outcomes (n=60):

- math-only wins (math correct, scheduled wrong): **4 items** — ids 33, 38, 42, 58.
- scheduled wins (scheduled correct, math wrong): **2 items** — ids 11, 13.
- net: **math −2 items ahead of scheduled** = the 0.850 − 0.817 = 0.033 combined gap.

So scheduling beats the *diluted static merge* by +18.3pp, but loses to the *trivial best-single
baseline* by 2 items. The single-adapter ceiling is not reached.

### JSON-valid is NOT an independent axis in the scheduled arm

`score()` computes `cot_correct` only inside `if json_valid:`, so `cot_correct ⟹ json_valid` in
every arm. In the scheduled arm this collapses fully: every brace-emitting item both parses AND is
correct, and every failure is a no-brace runaway, so `combined == cot_correct == json_valid == 0.817`
identically. The three scheduled sub-scores being equal is structural, not a copy bug — but it also
means JSON-validity carries no independent signal here.

### Scheduled failure mode = non-termination, not malformed JSON

All 11 scheduled failures (ids 3, 8, 28, 32, 33, 38, 42, 43, 48, 53, 58) share `saw_brace:false` and
`ntok:512`: the model never emitted `{`, so the schedule never entered the EMIT phase and decode ran
to the 512-token cap. There are **zero** malformed-JSON failures in the scheduled arm. The
malformed-JSON failure mode that MATH.md predicted (code adapter must dominate "or the model emits
malformed JSON") never actually occurs under the schedule.

### P1 — Underpower guard (tested first)

- `gap_underpower = best_single − combined(static) = 0.850 − 0.633 = **0.21667**`
- Threshold: ≥ 0.15 → **PASS** (regime is NOT underpowered; static genuinely sits ≥15pp below the
  best single adapter, so there is a real gap for timing to recover).

### P2 — Timing lift (the hypothesis)

- `lift = combined(scheduled) − combined(static) = 0.817 − 0.633 = **0.18333**`
- Threshold: ≥ 0.15 → **PASS** (scheduled exceeds static by 18.33pp, clearing the +15pp bar).

## Kill criterion 2308

> "scheduled composition combined-score (CoT-correct AND JSON-valid) does NOT exceed static 0.5/0.5
> merge by >=15pp on the constructed phase task, OR static merge fails to show a >=15pp gap below
> best-single-adapter (regime underpowered)"

- underpowered: **false** (gap 0.21667 ≥ 0.15)
- lift: **0.18333 ≥ 0.15**
- result: **pass** — kill 2308 does NOT fire.

## Verdict

**SUPPORTED on the pre-registered gates (kill 2308), with an explicit ceiling caveat.** Both gates
pass: static sits ≥15pp below best-single (underpower gap +0.21667) and scheduled exceeds static by
≥15pp (lift +0.18333). The magnitude-match invariant (`w_m(t)+w_c(t)=1` at every step in both static
and scheduled arms) means total injected adapter signal is identical between static and scheduled;
the only difference is *when* each adapter's unit of weight is spent. The +18.33pp scheduled-over-
static lift therefore isolates a TIME-axis composition gain: spending the math weight early and the
code weight late beats spending half of each throughout.

**Caveat (does not change the gate result, but bounds the claim):** scheduling does NOT beat the
best single adapter. Math-only 0.850 > scheduled 0.817; head-to-head, math wins 4 items (33,38,42,58)
and scheduled wins 2 (11,13), net math +2. The honest reading is: decode-step timing recovers most
of the static-merge interference gap but does **not** reach the single-adapter ceiling. The scheduled
arm's only failure mode is non-termination (no brace, 512-tok cap; ids 3,8,28,32,33,38,42,43,48,53,58),
not malformed JSON — the predicted malformed-JSON failure mode never occurs, and JSON-validity is not
an independent axis in this arm (combined == cot_correct == 0.817 identically).

Measured: math 0.850 · code 0.433 · static 0.633 · scheduled 0.817 · underpower gap +0.21667
(≥0.15) · scheduled−static lift +0.18333 (≥0.15) · scheduled−best_single −0.033 (math ahead by 2
items). Wall clock 1400.9 s. `is_smoke: false`.
