# PAPER — exp_spark_intra_rank_phase_gate

## Verdict: **KILLED** (kill 2309, both clauses fired)

## Claim under test
Within ONE frozen rank-6 math LoRA on `q_proj`, splitting the per-layer delta's SVD into a
**head** half (top-σ, domain identity) and **tail** half (low-σ, generic reasoning), then
**timing which half fires** — tail during chain-of-thought, head during answer-emit, with
per-token injected-delta L2 renormalized to equal the uniform-math delta L2 — beats applying
the same adapter uniformly. Composition is claimed to be *intra-adapter / which-rank-when*,
not magnitude.

## Setup (real, not smoke)
- Model: `mlx-community/gemma-4-e4b-it-4bit`; adapter: `data/adapters/math/adapters.safetensors`,
  rank 6, scale 6.0, on `q_proj`, 42 layers.
- GSM8K n=80, greedy, `enable_thinking=True`, 1024 max tokens.
- Phase boundary detected at decode time from the model's own emitted `#### ` delimiter
  (not oracle). `is_smoke: false`. Wall: 7990 s.
- Magnitude-match invariant verified: 12,294,828 checks, 0 violations, max rel err 1.9e-06
  (tol 1e-3). Every arm writes the **same per-token delta energy** as uniform-math, so any
  difference isolates direction/timing from magnitude.

## Prediction vs measurement

| Arm | GSM8K EM | note |
|---|---|---|
| base (no adapter) | **0.8125** | reference |
| uniform-math (full δ every token) | **0.7125** | best-single ceiling |
| head-only-always | **0.7625** | static rank-half |
| tail-only-always | **0.6000** | static rank-half |
| schedule (tail@reason, head@answer) — the hypothesis | **0.6875** | |
| swap (head@reason, tail@answer) — control | **0.7625** | |

best_schedule = max(schedule, swap) = **0.7625** (the swap control, not the hypothesis).

- **Predicted:** best_schedule − uniform-math ≥ +5pp AND no static rank-half matches uniform
  (point estimate +6–10pp).
- **Measured:** best_schedule − uniform-math = +5.0pp (the swap control); the *hypothesis*
  arm `schedule` is −2.5pp **below** uniform. max static rank-half = 0.7625 ≥ uniform 0.7125.

## Underpower guard (F#866 / F#871)
uniform-math − base = **−10.0pp** (0.7125 vs 0.8125). The math `q_proj` adapter is
**self-sabotaging**: applied uniformly it scores *below* base on GSM8K, exactly the F#866
failure mode. Consequence: any arm "beating" uniform-math here is recovering self-inflicted
damage, not adding capability. The single arm that ties the best (swap, 0.7625) is still
**−5.0pp below base** — it does not even recover to the no-adapter line. There is no real
win available to claw back; uniform-math is below its own floor.

## Which kill clause fired
Kill 2309 is a disjunction; **both** disjuncts independently fire (KILLED if either holds):

- **Clause A (timing win)** — FAILED. `best_schedule − uniform = +5.0pp`, which does not
  clear the strict `>= +5pp` margin (computed 4.999...e-2 < 0.05). And the win is carried
  entirely by the `swap` *control*, not the predicted `schedule` arm — `schedule` is −2.5pp.
  Timing in the predicted direction does not help.
- **Clause B (timing necessity)** — FAILED, decisively. A **static** rank-half already
  matches/beats uniform-math: head-only-always = 0.7625 ≥ uniform 0.7125 (and swap = 0.7625).
  Since a no-timing static subspace selection equals the best timed arm, **timing is
  irrelevant** — the entire phase-gate mechanism is unnecessary. This is the dispositive clause.

## Reading
The only thing that helped relative to uniform was *dropping the tail directions* (head-only
static = swap = 0.7625): keeping the top-σ subspace and discarding the low-σ "reasoning" half
recovers some of the adapter's self-inflicted damage. That is a static rank-truncation effect,
not a temporal schedule. The hypothesized "tail during reasoning, head during answer" schedule
is the *worst* adapter arm except tail-only. Intra-adapter phase timing is refuted on q_proj
math under magnitude-match.

## Verdict line
**KILLED — kill 2309 fired on BOTH clauses (A: +5.0pp < +5pp strict margin and carried by the
swap control, not the schedule hypothesis; B: static head-only 0.7625 ≥ uniform 0.7125, so a
rank-half matches uniform and timing is irrelevant). Underpower guard: uniform-math −10pp below
base — the adapter self-sabotages (F#866), and the best arm is still −5pp below base.**
