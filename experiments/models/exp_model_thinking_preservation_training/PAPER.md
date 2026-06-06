# PAPER: Training recipe that preserves thinking mode (smoke)

Experiment: `exp_model_thinking_preservation_training`
Status: `provisional` (smoke; `is_smoke=true`, N=20 steps, EVAL_PER_CAT=2).
Verdict line: **PROVISIONAL — full-scale rerun required to rule on K1685/K1686/K1687.**

## 1 Hypothesis (from MATH.md §3)

Training a LoRA adapter on reasoning traces whose **targets contain
`<think>...</think>` tokens** (A1 `mean_think_frac ≥ 0.4`), with ≥3
non-collinear eval domains, should keep `|acc(π_Δ) − acc(π_base)| ≤ 2pp` on
MMLU-Pro+thinking and preserve `avg_thinking_chars ≥ 1500`.

## 2 What we ran

| Item | Value |
|---|---|
| Base model | `mlx-community/gemma-4-e4b-it-4bit` |
| Adapter | LoRA rank 8, scale 1.0, targets `v_proj + o_proj` |
| Training data | s1K (olympiad math with `<think>` traces), **27 examples** after filter (smoke) |
| Training steps | 20 (smoke); full would be 1000 |
| Eval | MMLU-Pro: math, computer science, health; 2 questions per category (smoke) |
| Thinking | `enable_thinking=True` on both base and adapter runs; `max_tokens=2048` |
| Total runtime | 265s |

A1 validation: 27/27 examples contain `<think>`; mean think fraction of
assistant target = **0.70** — well above the 0.40 lower bound.

## 3 Prediction vs measurement

| KC | Prediction (proof) | Measurement (smoke) | Δ | Verdict |
|---|---|---|---|---|
| K1685 | `|adapt − base| ≤ 2pp` | base 66.7%, adapt 33.3% → Δ = −33.4pp | 33x over bound | **FAIL** (smoke) |
| K1686 | avg thinking ≥ 1500 chars | **0 chars** on BOTH base AND adapter | — | **FAIL / inconclusive** |
| K1687 | 3 categories each within 5pp | math −50pp, cs 0pp, health −50pp | 2/3 fail | **FAIL** (smoke) |

Per-category (base → adapter):
- math: 100% → 50% (Δ = −50pp)
- computer science: 50% → 50% (Δ = 0pp)
- health: 50% → 0% (Δ = −50pp)

Training loss: val 1.207 → 1.169 (20 steps, not converged).

## 4 Interpretation

**Smoke does NOT falsify the proof; it exposes the lower bound of what the
proof predicts.** Assumption (A3) `η·N·√r·max‖∇L‖ ≤ 0.5` holds only *at
convergence*; at N=20 the adapter has taken ~1% of the way into the full
trajectory, and loss is still decreasing. The MMLU-Pro drop at N=20
matches the F#538 pattern: partially trained adapter is worse than base
because it's drifted without reaching the SFT-residual equilibrium.

**The K1686 = 0 result is the mem-antipattern-008 footprint** (thinking-mode
truncation / template mismatch): *both* base and adapter produced 0
`<think>` chars. Prior experiments (F#536, exp_p11_baseline_eval) measured
`avg_thinking_chars ≥ 2900` on the same base model with a similar MCQ
prompt. The most likely explanation is that the chat template applied here
omits the `<start_of_turn>thought\n` block that triggers Gemma 4's native
thinking channel; our regex doesn't match whatever format emerged. This is a
**measurement artifact**, not a recipe falsification. Fix in next run:
inspect raw `response[:400]` on a single eval question and repair the parser
before trusting K1686.

## 5 Findings (provisional)

- **F(provisional-a)** A1 can be cleanly enforced: copying s1K traces with
  `<think>` blocks gives 100% A1 satisfaction and mean think-fraction 0.70 — the
  training-data side of the recipe is implementable.
- **F(provisional-b)** LoRA r=8 on `v_proj+o_proj` at N=20, LR=1e-5 is
  **under-trained**: adapter underperforms base by 33pp. The (A3) bound is
  violated from the opposite direction — not too much drift, but a
  half-baked drift that's worse than either extreme. Confirms F#538 intuition
  that partially-trained reasoning SFT is in the worst regime.
- **F(provisional-c)** Thinking-char measurement failed on BOTH base and
  adapter — parser / template mismatch — this is `mem-antipattern-008`
  triggering, not a new finding.

## 6 Assumptions & limits (pre-registered in MATH.md §5)

- Smoke mode: cannot conclude K1685/K1686/K1687 — completed as `provisional`
  per PLAN.md §1 verdict-consistency rule 4.
- Single training domain (s1K math); the 3-domain mixture (A2) is not
  tested here. Full rerun needs real `<think>`-annotated code + medical
  traces.
- SFT-residual architecture (F#403) is **not** implemented here; full run
  requires custom head on top of mlx-lm.

## 7 Next steps (blocking full verdict)

1. Fix thinking-char parser: probe raw response, find the actual delimiter
   Gemma 4 E4B 4-bit emits under the `enable_thinking` template. Without
   this, K1686 cannot be measured.
2. Assemble 3-domain `<think>`-annotated training set: s1K (math) +
   thinking-augmented code traces + thinking-augmented medical traces.
   Target ~500 examples per domain.
3. Implement SFT-residual head `B_applied = B_sft + s · head(z)` with
   zero-init (F#403 recipe) in MLX — custom training loop instead of
   mlx-lm.lora CLI.
4. Full rerun: `SMOKE_TEST=0`, N=1000 steps, EVAL_PER_CAT=20, full MMLU-Pro
   categories.
5. Only after (1)–(4) can K1685/K1686/K1687 be resolved.

## 8 Antipattern audit (PLAN.md §1 rule 6)

- **mem-antipattern-001** (composition math): N/A — single-adapter eval.
- **mem-antipattern-002** (tautological routing): N/A — no routing.
- **mem-antipattern-003** (unsafe LORA_SCALE): PASS — scale=1.0.
- **mem-antipattern-004** (KC-swap-after-failure): PASS — KCs locked in
  MATH.md §4 pre-run.
- **mem-antipattern-005** (smoke as full): PASS — `is_smoke=true` in
  results, verdict is PROVISIONAL not SUPPORTED.
- **mem-antipattern-008** (thinking-mode truncation): **TRIGGERED** on K1686
  — base and adapter both 0 thinking chars despite prior experiments
  measuring ≥2900 on the same model. Documented as a measurement artifact
  above; **not** silently treated as a recipe failure.
- **mem-antipattern-011** (wrong-model proxy): PASS — Gemma 4 E4B 4-bit, the
  actual target.
- **mem-antipattern-013** (hardcoded pass=True): PASS — KC evaluation uses
  measured values.

Verdict consistency (PLAN.md §1 checklist):
1. `verdict = "PROVISIONAL"` (not KILLED/SUPPORTED).
2. `all_pass = false`.
3. PAPER verdict line contains `PROVISIONAL`.
4. `is_smoke = true` ⇒ `provisional`. ✓
5. KCs unchanged between MATH.md §4 and results.
6. mem-antipattern-008 flagged, not ignored.
