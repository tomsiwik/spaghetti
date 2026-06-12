# MATH — exp_spark_phantom_router

## The question (frame-break attempt on F#875)
F#875 (`exp_spark_base_wins_bridges`) reported a per-token **entropy-argmin router** over
{frozen-base, math-adapter, code-adapter} scoring **EM 0.6167** on GSM8K-as-`solve()` vs **base-alone
greedy 0.4333** — a **+18.33pp** win — with the **frozen base selected on 76.6% of emitted tokens**.

That last fact is the threat. If the base wins the overwhelming majority of tokens, the adapters
contribute on only ~23% of tokens. The hypothesis here is that the lift is **not** routing-into-adapter
knowledge but the **argmin-entropy decode rule itself**: at each step, choosing the *most-confident*
next-token distribution from an ensemble of forward passes is a self-correcting decoding heuristic that
helps even when **every** ensemble member is the SAME frozen base with ZERO adapters.

## Theorem (pre-registered prediction)
Let `R = EM(F#875 3-arm entropy-argmin router)`, `B = EM(base-alone greedy)`, both re-measured IN THIS
RUN on the identical task/scorer/seeds. Let `F = EM(adapter-FREE self-entropy-argmin)` over a
self-ensemble of the SAME frozen base, ZERO adapters. Define the **recovery fraction**

    recovery_frac = (F − B) / (R − B)        (denominator = the IN-RUN router−base gap, not 18.33)

**Frame-break hypothesis (to be tested, NOT assumed):** an adapter-free argmin-entropy control recovers
**most** of the gain → `F − B ≥ +9.0pp` (≥ half of F#875's pre-registered +18.33pp).

**Pre-registered refutation threshold (kill 2312, verbatim):** if `F − B < +9.0pp` → **KILLED**.

### Both outcomes are informative (stated up front, no goalpost-moving)
- **KILLED (F−B < 9.0pp):** the adapter-free control CANNOT recover the win. The argmin-entropy decode
  rule alone is insufficient — the adapters are doing real, load-bearing work. This **VINDICATES F#875**:
  routing is REAL, the frame-break is FALSE.
- **SUPPORTED (F−B ≥ 9.0pp):** an adapter-free self-ensemble recovers most of the gain. F#875's
  composition-beats-single milestone **COLLAPSES to a decoding artifact** ("argmin-entropy decoding
  helps"), invalidating the program's first positive composition result.

The +9.0pp absolute threshold is the pre-registered kill line. `recovery_frac` is reported
descriptively against the matched in-run `R − B` denominator (per-seed and mean).

## The adapter-free self-ensemble (defined precisely — guardrail 2)

### Why pure temperature scaling is a NO-OP (and is therefore rejected as the mechanism)
For logits `z` and temperature `t>0`, `softmax(z/t)`:
1. **argmax invariance:** `argmax_i z_i/t = argmax_i z_i` for all `t>0`. Temperature never changes the
   emitted token of any single member.
2. **entropy monotonicity:** Shannon entropy `H(softmax(z/t))` is monotonically increasing in `t` (it
   limits to 0 as `t→0⁺` and to `log V` as `t→∞`). So across an all-base temperature ensemble the
   argmin-entropy member is ALWAYS the lowest-`t` member, at EVERY token.

Consequences: argmin-entropy over `{base@t₁, base@t₂, …}` deterministically selects the smallest `t` at
every step, whose argmax equals plain greedy. The arm would be **identical to base-alone greedy** — a
degenerate no-op that tests nothing. Temperature scaling cannot be the adapter-free perturbation.

### The defensible perturbation: a base-only PROMPT-FRAMING self-ensemble (real logit diversity)
We perturb the **conditioning context**, not the logits. The F#875 router gets genuine per-token
distribution diversity because base/math/code are *different functions* of the same emitted prefix. The
adapter-free analogue must also produce genuinely different next-token distributions from the SAME
frozen base. We obtain that with a **K=3 prompt-framing self-ensemble**: the identical frozen base is run
on three semantically-equivalent framings of the SAME problem,

  - F0: the EXACT F#875 `solve()` prompt (so member 0 == base-alone greedy by construction),
  - F1: same task with an added neutral system-style preamble ("You are a careful Python programmer.
        Think about edge cases."),
  - F2: same task with a different but equivalent instruction phrasing ("Implement `solve()` returning
        the final number; use a code block.").

All three are **frozen base, ZERO adapters**. Like F#875 they share the **emitted-token stream** but keep
**independent KV caches** (each member conditions on its own prompt prefix + the shared committed tokens).
At each decode step we emit the argmax token of whichever member has the **lowest next-token Shannon
entropy** (argmin-entropy = most-confident framing). Because the three prefixes differ, the three
next-token distributions genuinely differ → the argmin-entropy selection is **non-degenerate** (different
framings win at different tokens) and the arm is **not** a relabeled base-alone greedy.

**Falsifiable non-degeneracy guard (verified in-run, not assumed):** we record, per seed, the fraction of
emitted items whose adapter-free output differs from base-alone greedy (`af_frac_differ_from_base`) and
the per-member win counts. If the adapter-free arm collapses to base-alone (`af_frac_differ_from_base ==
0`, i.e. member F0 wins every token) the arm is a no-op and the result is reported `provisional`
(degenerate control), NOT a clean kill/support — exactly the failure guardrail 2 warns against. This makes
the "genuinely differs from base" requirement an explicit, measured pre-condition of any verdict.

This is the **invert-assumption** perturbation: F#875 assumes you must run the adapters' forward passes to
get the routing signal; here the routing signal is manufactured from the base's OWN logits under
context perturbation, with no adapter ever loaded.

## Matched anchors re-measured IN THIS RUN (guardrail 1)
On GSM8K-as-`solve()`, `n=60`, seeds `{42, 1, 2}`, for EACH seed we run all three arms on the SAME items:
  (a) **base-alone greedy** `B`  — F#875 single-BASE arm, identical prompt/scorer.
  (b) **F#875 3-arm entropy-argmin router** `R` over {base, math, code} — identical to F#875 router_decode.
  (c) **adapter-FREE self-entropy-argmin** `F` — K=3 base prompt-framing self-ensemble, zero adapters.
We do NOT compare against F#875's stale single-seed 0.4333/0.6167. The recovery fraction uses the matched
in-run `R − B`. Report per-seed `B,R,F`, `recovery_frac`, and the mean across seeds. Verdict on MEAN
`F − B`.

### Seed semantics
GSM8K item selection is deterministic (first 60 test items) → identical across seeds. Greedy decode and
argmin-entropy selection are deterministic given the model. `mx.random.seed(seed)` is set per seed for
full reproducibility / any nondeterminism in MLX kernels; the three arms within a seed see the same items.

## Harness (must match F#874/F#875 no-thinking high-headroom regime)
- Model: `mlx-community/gemma-4-e4b-it-4bit`, frozen 4-bit. mlx-lm == 0.31.2.
- Thinking OFF (plain chat template, no few-shot), MAX_NEW = 800, greedy argmax decode.
- Adapters (router arm only): `exp_p1_t2_single_domain_training/adapters/{math,code}/adapters.safetensors`,
  q_proj r6 scale6, installed exactly as F#875 (`Σ_layers scale*(x@A)@B`, never `(ΣB)(ΣA)`; 42 wrappers).
- Scorer: the EXACT F#875 executable `score_twodomain` — extract ```python```, run `solve()`, compare
  return to gold `#### N` within 1e-2.
- n=60 two-domain items per seed; seeds {42,1,2}. is_smoke = false. No mocks.

## Verdict logic
- mean(`F − B`) ≥ +9.0pp  → **supported** (frame-break: F#875 collapses to a decoding artifact).
- mean(`F − B`) <  +9.0pp  → **killed** (kill 2312: F#875 routing is REAL, frame-break false).
- adapter-free arm degenerate to base-alone (mean `af_frac_differ_from_base == 0`) → **provisional**.
- Sanity: if in-run `R − B` ≤ 0 (router fails to reproduce its own anchor), report `provisional` —
  recovery_frac is undefined / the comparison is not meaningful that seed.
