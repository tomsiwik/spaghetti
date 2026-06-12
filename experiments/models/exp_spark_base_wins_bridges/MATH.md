# MATH — exp_spark_base_wins_bridges

## Claim (REFRAMED per REVIEW — matches the data, not the original hunch)
Composition need not live in parameter-merge space. It can live on the **decode-step (time) axis**,
with the **frozen base as a first-class third routing option** in an entropy-argmin per-token router
over {frozen-base, math-adapter, code-adapter}. At each decode step we run all three forward passes,
each producing a next-token distribution, and emit the token from whichever distribution has the
**lowest Shannon entropy** (argmin-entropy = most-confident expert).

The claim is: **entropy-argmin 3-way routing where the frozen base usually wins** beats the best of
ALL three single arms — including base-alone evaluated on the EXACT same task/scorer — by ≥ +3.0pp
exact-match on a task that genuinely requires both domains, while the frozen base is selected on
≥ 15% of emitted tokens.

**The claim is NOT "the base wins the bridge tokens."** The original hunch was that the base would win
specifically the high-entropy inter-domain pivots (`def`, `return`, `=`, `:`, newlines). The data
refute that as the *locus*: the base wins ~76.6% of ALL tokens, of which the predicted bridge pivots
are only ~17.4%. Bridge tokens are therefore NOT the mechanism — the base simply wins the bulk of the
ordinary scaffold/prose tokens, and the adapters fire only on their confident domain spans. The
falsifiable test is the router-vs-best-single lift and the base-win fraction, NOT a bridge-locus claim.

## What is DIFFERENT from the killed exp_spark_entropy_gated_lora (and why it escapes that kill)
The prior experiment was **killed** because its substrate — F#827's −12pp off-domain interference /
+22pp on-domain lift — did not reproduce under the only in-repo adapter (q_proj r6 scale6). Its gate
was a *continuous attenuation* of a *single* adapter (`scale·(1−p_top1_base)`); the base was only a
**reference** used to compute the gate, never an emitter. The gate fired correctly (mean ≈ 0.08) but
had nothing to act on, so both kill criteria were undefined-by-construction.

Three structural differences here:
1. **Discrete 3-way argmin, not continuous attenuation.** The base is a *first-class emitter* that can
   WIN a token outright, not a multiplier on one adapter. We never reference F#827's interference
   magnitudes; the prediction is a *router-vs-best-single* delta measured fresh in THIS harness.
2. **Two opinionated adapters compete (math + code), not one.** The killed run had a single adapter, so
   "routing" was just gate-scheduling that adapter. Here the novel quantity is the *transition token*
   where two confident-but-wrong domain priors collide and the **uncommitted base** has lower entropy.
3. **Genuine two-domain task** where best-single is provably below ceiling (insufficiency gap measured
   in the pre-gate). The killed run scored each domain in isolation, so there were no bridge tokens to
   win. Here every problem requires code structure AND arithmetic, manufacturing real bridge tokens.

The prior LEARNINGS mandate: *do not cite F#827 magnitudes as substrate without reproducing them.*
We comply by **not depending on F#827 at all** — the substrate is the F#874 no-thinking high-headroom
harness in which we *first verify, in this very run,* that each adapter is net-positive on its own
domain (pre-gate). If that verification fails the verdict is KILLED (pre-gate), exactly as the
LEARNINGS demand.

## The task (genuine two-domain, bridge tokens guaranteed)
GSM8K word problems, answered by **writing a Python function `solve()` that returns the numeric
answer**, which we then execute. Correct iff the executed return value equals the gold `#### N`.
- The **code** domain is load-bearing: output must be syntactically valid, runnable Python with
  `def solve():` / `return` (transition/bridge tokens), else execution fails → wrong.
- The **math** domain is load-bearing: the arithmetic inside the function must be correct, else the
  returned number is wrong.
Best-single is insufficient by construction: the code adapter writes runnable code but its arithmetic
priors are weak; the math adapter does arithmetic but in `#### N` prose, not a runnable function; and
base-alone (now an explicit arm) has no domain specialization. Bridge tokens (`def`, `return`, `=`,
`:`, `\n`) are tracked descriptively as a share of base wins, but — per the REVIEW reframe — they are
NOT claimed to be the mechanism; the base wins the bulk of ordinary scaffold tokens, not just pivots.

## Harness (must match F#874 no-thinking high-headroom regime)
- Model: `mlx-community/gemma-4-e4b-it-4bit`, frozen 4-bit.
- **thinking OFF** (plain `apply_chat_template`, no `enable_thinking=True`).
- **Weak prompt** (terse, no few-shot, no chain-of-thought instruction).
- **~800 new tokens** headroom per problem (no thinking truncation; ample for a `solve()` function).
- Adapters: math = `exp_p1_t2_single_domain_training/adapters/math/adapters.safetensors`,
  code = `.../adapters/code/adapters.safetensors`. Both q_proj-only, r=6, scale=6.0 (≤ 8 guard).
  Same recipe ⇒ injection magnitude comparable across the two adapters where applied (F#863).
- Composition rule: each expert is `base + Σ_layers (scale · (x@Aᵢ)@Bᵢ)` applied independently
  (NEVER `(ΣB)(ΣA)`); the router picks ONE expert's logits per step. scale = 6.0 ≤ 8.
- Greedy decode. Seeds fixed: SEED = 42. n = 60 GSM8K problems. is_smoke = false.

## Router
At decode step t, with shared KV context, compute logits from each expert e ∈ {base, math, code}:
H_e(t) = −Σ_v softmax(logits_e)_v · log softmax(logits_e)_v (Shannon entropy, nats).
Winner w(t) = argmin_e H_e(t). Emit argmax of logits_{w(t)}. Advance all three caches with that token.

## Prediction (pre-registered)
- **Pre-gate**: math-adapter EM_on-math > base EM_on-math AND code-adapter EM_on-code > base EM_on-code,
  both in this harness. Predicted: each adapter ≥ +3pp on its own domain. (If either ≤ 0 ⇒ KILLED.)
- **Best-single now includes base-alone on the solve() task** (REVIEW fix 1):
  `best_single = max(base_solve_EM, math_solve_EM, code_solve_EM)`, all three measured on the IDENTICAL
  solve() prompt + executed scorer (NOT the prose pre-gate prompt). If the router cannot beat THIS
  best_single by ≥ +3.0pp, the adapters add nothing and the verdict is KILLED.
- **Router**: router EM − best-single EM ≥ +3.0pp.
- **Base-win fraction**: frozen base selected on ≥ 15% of emitted tokens (NOT a bridge-locus claim —
  bridge share is reported descriptively only).
- **Non-collapse check**: router per-item outputs must genuinely differ from base-alone on ≥ 1 item
  (reported as `router_items_differ_from_base`); if the router collapses to base-alone the lift is an
  artifact.

## Pre-registered kill criteria — kill 2311 (verbatim, all three clauses)
> On a genuine two-domain task in the no-thinking high-headroom harness (thinking off, weak prompt,
> ~800 tok), the entropy-argmin per-token router over {math-adapter, code-adapter, FROZEN-BASE} fails
> to beat the best single adapter by >=+3.0pp EM; OR the frozen-base option is selected on <15% of
> tokens (no real bridge regime); OR either adapter is not net-positive on its own domain in THIS
> harness (pre-gate fails).

Refutation thresholds (numeric, UNCHANGED from the pre-registration):
- Clause 1 (router lift): KILLED if `router_EM − best_single_EM < +0.030`.
- Clause 2 (bridge regime): KILLED if `base_win_fraction < 0.15`.
- Clause 3 (pre-gate): KILLED if `math_adapter_EM_math ≤ base_EM_math` OR `code_adapter_EM_code ≤ base_EM_code`.
SUPPORTED only if all three clauses pass. Pre-gate (clause 3) is evaluated and reported FIRST.

REVIEW clarification (no threshold change): `best_single` in clause 1 is now
`max(base_solve_EM, math_solve_EM, code_solve_EM)` measured on the identical solve() task. Because the
base-alone arm is inside the max, clause 1 implicitly enforces `router_EM − base_solve_EM ≥ +0.030`:
if the router merely ties or loses to base-alone, the adapters add nothing and clause 1 fires → KILLED.
