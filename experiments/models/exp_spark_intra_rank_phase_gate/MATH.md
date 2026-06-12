# MATH — exp_spark_intra_rank_phase_gate

## Claim
Composition is **intra-adapter**, not inter-adapter. Within ONE frozen rank-r math
LoRA adapter on `q_proj`, the per-layer delta `Δ^l = s · B^l A^l` (shape out×in) has an
SVD `Δ^l = U^l Σ^l V^lᵀ`. We split the r singular directions into a **head** half
(top ⌈r/2⌉ by σ — domain/knowledge identity) and a **tail** half (bottom ⌊r/2⌋ —
generic reasoning). The hypothesis: firing ONLY tail directions during chain-of-thought
tokens and ONLY head directions during answer-emit tokens, with the **per-token injected
delta L2 renormalized to equal the uniform-math delta L2 at every position**, beats the
same adapter applied uniformly.

## Construction (per layer l, per token t)
Let `δ_full(x_t) = s · (B^l A^l) x_t` be the uniform-math injected delta (the per-token
vector added to `q_proj` output). Its low-rank SVD gives projector onto a rank subset S:
`δ_S(x_t) = U^l_S (U^l_Sᵀ δ_full(x_t))`, i.e. the component of the full delta lying in the
span of the chosen singular directions. Here `U^l` are the left singular vectors of `Δ^l`
(out-space), and `δ_full ∈ span(U^l)` exactly (rank ≤ r), so `δ_head + δ_tail = δ_full`.

**Magnitude-match invariant (F#863).** For any arm that injects a sub-rank component
`δ_S`, we renormalize per token:
`δ_inject(x_t) = δ_S(x_t) · (‖δ_full(x_t)‖₂ / (‖δ_S(x_t)‖₂ + ε))`.
Thus `‖δ_inject(x_t)‖₂ = ‖δ_full(x_t)‖₂` at EVERY layer and EVERY token (asserted in code,
tol 1e-3 relative, skipping only tokens where ‖δ_full‖≈0). Any win therefore isolates
**which-rank-when** from magnitude — every arm writes the same per-token energy as
uniform-math; only the DIRECTION (head vs tail subspace) and TIMING differ.

## Phase boundary (decode-time, not oracle)
Prompt is built with `enable_thinking=True`. EMPIRICALLY (verified in smoke on this
tokenizer/template) gemma-4 does NOT open a `<|channel>thought` block for GSM8K — it emits
its chain-of-thought directly and is prompted to end with `#### ` before the numeric answer.
So the operative reasoning→answer-emit delimiter the model itself emits is `#### `. We detect
the transition by string-matching the cumulative DECODED text for `#### ` (and we also honor
the thinking-channel closes `<channel|>` / `</think>` / `<|think|>` if a think block ever
appears). Tokens generated BEFORE the delimiter = reasoning phase; tokens AFTER = answer-emit
phase. This is computed from generated tokens at decode time — it is the model's own emitted
delimiter, never the gold label.

## Arms (all delta-norm-matched to uniform-math per token)
- `base`            : no adapter (REFERENCE; F#866 — math q_proj can score BELOW base).
- `uniform-math`    : full δ_full every token (best-single ceiling).
- `head-only-always`: δ_head renormalized, every token.
- `tail-only-always`: δ_tail renormalized, every token.
- `schedule`        : tail during reasoning, head during answer-emit (the hypothesis).
- `swap`            : head during reasoning, tail during answer-emit (control).

## Underpower guard (F#871)
We FIRST establish uniform-math-solo's position vs `base`. If uniform-math is at/above
base AND there is no headroom (it is already near its own achievable ceiling), we say so.
We report `base` so a "win" that merely recovers self-inflicted damage is visible — a
schedule arm beating a self-sabotaged uniform-math but still below base is NOT a real win.

## Prediction
`best_schedule = max(schedule, swap)` EM beats `uniform-math` EM by ≥ +5pp on GSM8K
(n=80, real exact-match), AND no static rank-half arm (head-only / tail-only) already
matches uniform-math. Predicted point estimate: schedule ≈ uniform-math + 6–10pp.

## Kill criterion 2309 (pre-registered, verbatim, both clauses)
> Best of {head@answer/tail@reason, its swap} fails to beat uniform-math-solo by >=5pp
> GSM8K EM (n=80), OR any static rank-half arm already matches it (timing irrelevant)

Operationalized:
- clause A (timing win): `best_schedule_EM - uniform_math_EM >= +0.05` (5pp).
- clause B (timing necessity): no static rank-half arm matches uniform-math, i.e.
  `max(head_only_EM, tail_only_EM) < uniform_math_EM` (strict; a tie/match kills).
- SUPPORTED iff clause A holds AND clause B holds. KILLED otherwise.

## Refutation threshold (numeric)
KILLED if `best_schedule_EM - uniform_math_EM < 0.05`
OR `max(head_only_EM, tail_only_EM) >= uniform_math_EM`.
