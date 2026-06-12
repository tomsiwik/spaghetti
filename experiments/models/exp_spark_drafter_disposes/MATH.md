# MATH.md — Drafter disposes, verifier composes

Experiment: `exp_spark_drafter_disposes`
Base (verifier): `mlx-community/gemma-4-e4b-it-4bit`, frozen 4-bit.
Drafters: q_proj LoRA r6 scale6 — `data/adapters/math`, `data/adapters/medical`, plus a
magnitude-matched random-logit NULL drafter. Decoding: GREEDY lossless speculative decoding.

## Frame-break

The 845-finding interference program treats multi-skill composition as a weight-space ACCURACY
problem (merge / route / gate / suppress). We relocate the axis: **never place off-domain adapter
weight in the output path.** The frozen base is the lossless speculative *verifier*; a domain LoRA is
only the cheap *drafter*. Under the exact-match greedy acceptance rule, the verified output is
provably the base-greedy sequence regardless of which drafter proposes. Composition then cannot
change accuracy — it can only change acceptance length / wall-clock speed.

## Notation

- Base model `p`, frozen. At a prefix `x`, define base-greedy token
  `g(x) = argmax_v logits_p(v | x)`. (`argmax` ties broken by lowest index — deterministic.)
- Base-only greedy sequence from prompt `P`, length `T`:
  `G(P) = (g_1, g_2, ..., g_T)`, where `g_{t} = g(P, g_1, ..., g_{t-1})`. This is the reference.
- Drafter `q_d` (base with LoRA adapter `d` set on `q_proj` via submodule replacement). It proposes a
  block of `k` tokens by its own greedy rollout.

## Theorem 1 (Accuracy-invariance — K1 holds by construction)

**Claim.** Greedy lossless speculative decoding with verifier `p`, using the acceptance rule

> propose draft block `(d_1,...,d_k)`; run ONE verifier forward over the block to get
> `g_i = argmax logits_p(· | prefix, d_1..d_{i-1})` for `i=1..k` and the bonus position `i=k+1`;
> scan left-to-right: accept `d_i` iff `d_i = g_i`; at the first mismatch (or after accepting all
> `k`) emit the verifier's own `g_i` at that position and discard the remaining draft;

produces an output sequence **identical to `G(P)`** for ANY drafter `q_d`, given identical decoding
budget `T`.

**Proof.** By induction on output position `t`.
Base case `t=1`: prefix is `P`. Either `d_1 = g_1` (accepted, emitted token `= g_1`) or `d_1 ≠ g_1`
(rejected, verifier emits `g_1`). Either way emitted token `= g_1 = G(P)_1`.
Inductive step: assume emitted prefix `= (g_1,...,g_{t-1}) = G(P)_{1:t-1}`. The verifier evaluates
`g_t = argmax logits_p(· | P, g_1..g_{t-1})`, which is exactly `G(P)_t` by definition. The accept/
reject rule emits at position `t` either the drafted token *only when it equals* `g_t`, or `g_t`
itself on rejection. In both branches the emitted token is `g_t = G(P)_t`. □

**Corollary (K1).** Exact token-sequence match of verified output to base-only greedy `= 100.0%`
for the math drafter AND the medical drafter AND the null drafter. Any per-position divergence is a
**decoding bug** (wrong acceptance rule / cache mis-rewind), not a property of the drafter. K1 is
therefore a *measured correctness check of the implementation*, never hardcoded. Predicted: 100.0%
match in all drafter conditions. **Refutation: any condition < 100.000%.**

## Theorem 2 (Discrimination — K2)

Let acceptance length `L_d` = number of draft tokens accepted before the first rejection in a block
(so emitted-per-verifier-call `= L_d + 1`). The expected accepted run length is governed by the
per-token agreement probability `a_d = P_x[ argmax q_d(·|x) = argmax p(·|x) ]` under the *base-greedy*
trajectory distribution. For geometric acceptance, `E[L_d] = a_d / (1 - a_d)` (truncated at block
size `k`), strictly increasing in `a_d`.

A drafter agrees with base-greedy more often exactly when its logit ordering at the top is closer to
the base's. Define the math adapter's **mean top-logit-shift magnitude**
`σ* = mean_x || logits_{q_math}(x) − logits_p(x) ||` (measured in-run on the base-greedy trajectory).

- **On-domain (math):** the LoRA was trained on math; on GSM8K prefixes it perturbs logits in a
  direction that *preserves or sharpens* the base's top choice on tokens it has learned, so
  `a_math` is high → long runs.
- **Off-domain (medical):** the medical LoRA shifts q_proj toward medical structure; on math
  prefixes those shifts are uncorrelated with the base top token, flipping the argmax more often →
  `a_medical < a_math` → shorter runs.
- **Null (magnitude-matched random):** perturb base logits by Gaussian noise with std calibrated so
  the mean perturbation magnitude equals `σ*` (matched in-run, not eyeballed). This isolates "any
  perturbation of the right size" from "learned on-domain structure." Random noise of magnitude `σ*`
  flips the argmax with probability driven only by the base logit gap, with no learned alignment, so
  `a_null < a_math`.

**Prediction (K2).** `E[L_math] ≥ 1.5 · E[L_medical]` AND `E[L_math] ≥ 1.3 · E[L_null]`.
**Refutation:** either ratio below its threshold.

## Theorem 3 (Net speedup — K3)

Let `c` = cost of one base verifier forward over a `k`-token block (≈ one decode step, batched),
`δ` = cost of the drafter's `k`-token greedy rollout (`k` small forwards with LoRA). Tokens emitted
per verifier call `= E[L_math] + 1`. Wall-clock tok/s ratio vs base-only greedy:

`speedup ≈ (E[L_math] + 1) / (1 + δ/c)`.

With `δ/c` modest (drafter is the same base + tiny r6 LoRA, `k` small) and `E[L_math]+1 > 1.25·(1+δ/c)`,
net speedup exceeds 1.25×.

**Prediction (K3).** on-domain math-drafter wall-clock tok/s `≥ 1.25 ×` base-only greedy tok/s.
**Refutation:** ratio < 1.25.

## Pre-registered kill clauses (DB id 2297) — KILLED unless ALL THREE hold (n=200 GSM8K)

- **K1 (accuracy-invariance):** exact base-greedy token-match = 100.0% for math AND medical drafters
  (null also reported). Any delta > 0pp in either → KILL.
- **K2 (discrimination):** `mean L_math ≥ 1.5 · mean L_medical` AND `mean L_math ≥ 1.3 · mean L_null`.
- **K3 (net speedup):** math-drafter tok/s `≥ 1.25 ×` base-only greedy tok/s.

`all_pass = K1 ∧ K2 ∧ K3`. Verdict `supported` iff `all_pass`, else `killed`. `is_smoke:false`.
