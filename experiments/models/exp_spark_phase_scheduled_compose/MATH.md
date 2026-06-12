# MATH — Decode-step phase-scheduled adapter mixing vs. static merge

`exp_spark_phase_scheduled_compose`

## 0. What failure is being prevented

Static scalar adapter merge `y = Wh + s(w_m·δ_math + w_c·δ_code)` applies ONE mix weight for
the entire generation. On a **reason-then-emit-strict-JSON** task the optimal weight is
*non-stationary in decode time*: during the chain-of-thought phase the math adapter must dominate
(or the model mis-reasons), and during the JSON-emit phase the code adapter must dominate. A static
0.5/0.5 merge dilutes BOTH skills at the moment each is needed and collapses the combined score.
This is a TIME-axis interference, invisible to every prior static composition finding (F#830/836/841:
static FR/TIES/uniform-1/N), which all treat the mix as a single weight for the whole sequence.

> **POST-HOC CORRECTION (after measurement).** The pre-registered premise above hypothesised that
> insufficient code-adapter weight during emit would cause **malformed JSON**. The data refute that
> failure-mode story for the scheduled arm: it has ZERO malformed-JSON failures. Its only failure
> mode is **non-termination** — on the 11 failing items the model never emits `{`, so the schedule
> never enters the EMIT phase and decode runs to the 512-token cap (`saw_brace:false, ntok:512`).
> See §4 for why JSON-validity is therefore not an independent axis in the scheduled arm.

## 1. Setup (REAL, frozen)

- Base: `mlx-community/gemma-4-e4b-it-4bit`, frozen.
- Adapters: `adapter_math.safetensors` (GSM8K, CoT), `adapter_code.safetensors` (HumanEval, code/JSON),
  both r=6 q_proj-only, 42 layers, from `exp_composition_residual_analysis/` (F#627 recipe).
- Composition is `Σ_i (B_i A_i)` — two independent deltas summed in activation space, NEVER `(ΣB)(ΣA)`.
- `LORA_SCALE = s = 6.0 ≤ 8`.

Per-layer q_proj output with decode-step weights `w_m(t), w_c(t)`:

    y_t = W h_t + s·[ w_m(t)·(h_t A_m) B_m  +  w_c(t)·(h_t A_c) B_c ]

## 2. The MAGNITUDE-MATCH invariant (F#863 lesson)

F#863 (strobe) died as a pure injection-magnitude confound: an arm that simply injected *more*
total adapter signal "won" for the wrong reason. To isolate TIMING from total magnitude we hold the
total injected weight constant across ALL composed arms and across ALL decode steps:

    INVARIANT:   w_m(t) + w_c(t) = 1     for every step t, in every composed arm.

The static arm uses `w_m = w_c = 0.5` (sum 1). The scheduled arm uses `w_m(t), w_c(t)` that also sum
to 1 at every t. Hence ∫ total magnitude is IDENTICAL between static and scheduled; the only
difference is *when* each adapter's unit of weight is spent. A scheduled win therefore isolates
timing, not magnitude. (Best-single arms are the ceiling reference, not magnitude-matched by
construction — they carry total weight 1 on a single adapter.)

## 3. The decode-time phase detector (NOT oracle)

Phase is defined from the ACTUALLY-GENERATED tokens during greedy decode, not from an oracle
position. We maintain a boolean `emitted_open_brace`, initialised False. At each decode step, after
the token is sampled, if the decoded token text contains the first `{` of the answer object we flip
`emitted_open_brace = True` and it stays True. The schedule reads this flag BEFORE producing the
next step's logits:

    REASON phase  (emitted_open_brace == False):  w_m = 1 - ε,  w_c = ε
    EMIT   phase  (emitted_open_brace == True ):   w_m = ε,      w_c = 1 - ε

with a small ε = 0.1 leak so neither skill is fully starved (still sums to 1). The flip happens at
the brace the model itself produced; if the model never emits `{`, the arm simply never enters EMIT
phase — no oracle correction.

Static arm: `w_m(t) = w_c(t) = 0.5 ∀t` regardless of phase.

## 4. Scoring (behavioral, two sub-scores + combined)

Per prompt (n = 60 constructed reason-then-emit items, each a small arithmetic word problem whose
answer must be returned as strict JSON `{"answer": <int>}`):

- **CoT-correct**: the integer in the emitted JSON's `answer` field equals the ground-truth integer.
- **JSON-valid**: the emitted answer block parses as JSON via `json.loads` AND has key `answer`.
- **combined = CoT-correct AND JSON-valid** (both true). This is the kill metric.

Report all three per arm.

> **POST-HOC NOTE on sub-score (non-)independence.** In the implementation, `cot_correct` is computed
> only *inside* the `if json_valid:` branch (an integer can only be read from the `answer` field once
> the block parses). Hence `cot_correct ⟹ json_valid` in EVERY arm, and `combined == cot_correct`
> always. In the scheduled arm this collapses completely: every brace-emitting item both parses and
> is correct, and every failure is a no-brace runaway, so `cot_correct == json_valid == combined ==
> 0.817` identically. The equality of the three scheduled sub-scores is therefore structural (not a
> copy bug), and JSON-validity carries no independent signal in the scheduled arm — the predicted
> "code adapter underweighted → malformed JSON" axis (§0) is not observed.

## 5. Predictions and pre-registered thresholds

Let `best_single = max(combined(math-only), combined(code-only))`.

**P1 (UNDERPOWER GUARD — tested FIRST, before the schedule):**
> static 0.5/0.5 combined must be ≥ 15pp BELOW best_single.
> `gap_underpower = best_single − combined(static) ≥ 0.15`.
> If this fails, the regime is underpowered → verdict **killed** (regime underpowered), reported as
> the first result. We do NOT proceed to judge the schedule on an underpowered task.

**P2 (the hypothesis):**
> scheduled combined must EXCEED static 0.5/0.5 combined by ≥ 15pp.
> `lift = combined(scheduled) − combined(static) ≥ 0.15`.

**Verdict logic (pre-registered, = kill 2308 verbatim):**

> KILL 2308: "scheduled composition combined-score (CoT-correct AND JSON-valid) does NOT exceed
> static 0.5/0.5 merge by >=15pp on the constructed phase task, OR static merge fails to show a
> >=15pp gap below best-single-adapter (regime underpowered)"

- if `gap_underpower < 0.15` → **killed** (underpowered; reported first).
- else if `lift < 0.15` → **killed** (timing does not recover the gap).
- else → **supported** (`gap_underpower ≥ 0.15` AND `lift ≥ 0.15`).

## 6. Why non-obvious / falsifiable

The magnitude-match invariant means the scheduled arm cannot win by injecting more signal — if it
wins it is purely because spending the math unit-of-weight early and the code unit-of-weight late
beats spending half of each throughout. A flat or negative `lift` despite a real underpower gap
falsifies the time-axis hypothesis cleanly. `is_smoke:false`.
