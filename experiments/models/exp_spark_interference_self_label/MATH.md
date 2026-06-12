# MATH — interference as a self-label (zero-training domain detector)

## Failure being prevented
Every prior interference arc tried to *suppress* off-domain damage (routing, merge, orthogonalization,
KL-health) and all died. This experiment asks the inverted question: is the damage itself a *free label*?
If an adapter, applied off-domain, votes *against the base model's own next-token prediction*, then the
**sign of that vote** is a router-free, training-free, calibration-free domain detector. The failure we are
trying to prevent is "you need a trained router / calibrated threshold to know which adapter owns a prompt."

## Setup
- Frozen base `B` = `mlx-community/gemma-4-e4b-it-4bit`.
- Adapters `A_i`, i ∈ {code, math, medical}: LoRA on `self_attn.q_proj`, rank 6, **scale 6.0 ≤ 8** (guard OK).
  Composition is the single low-rank delta `scale·(x@a)@b` per projection — never `(ΣB)(ΣA)`.
- Prompts `D_j`, j ∈ {code, math, medical}: held-out `valid.jsonl` user turns from the *same* training
  domains the adapters were trained on. ~30 prompts/domain.

## The signal
For a prompt, run greedy generation with the **base model** to get the base trajectory and, at each of the
first `K=8` generated positions `t`, the base top-1 token `ŷ_t = argmax_v logit^B_t(v)`. The base defines
both the token *and the position context* (teacher forcing on the base's own greedy continuation).

For adapter `i`, recompute the logits at those same positions/contexts with `B + A_i` applied, and read off
the logit assigned to the base's own pick `ŷ_t`. Define the per-(prompt, adapter) score

    s_i = (1/K) · Σ_{t=1..K}  [ logit^{B+A_i}_t(ŷ_t) − logit^B_t(ŷ_t) ].

`s_i > 0` ⇒ adapter raises the margin on the base's own prediction (compatible / on-domain).
`s_i < 0` ⇒ adapter lowers it (it disagrees with the base — the interference signature).

## Theorem (prediction)
A LoRA trained on domain `d` minimizes cross-entropy on `d`'s tokens, i.e. it pushes probability mass toward
the *correct* continuations of `d`-prompts. On a `d`-prompt the base's greedy `ŷ_t` is highly correlated
with that correct continuation, so `A_d` reinforces `ŷ_t` ⇒ E[s_d | domain=d] ≥ 0. On an off-domain prompt
`A_d` perturbs `q_proj` in directions tuned for *other* token statistics, which on average *reduce* the
margin of the base's pick ⇒ E[s_i | domain≠i] < 0. Therefore the scalar `s_i` separates
on-domain (label 1) from off-domain (label 0), and its **ranking** yields

    AUROC( score = s_i ; positive = on-domain ) > 0.5,

with the registered claim that it clears a *useful* bar of **0.70** (no threshold tuning — AUROC is
threshold-free; the "sign" framing is just the natural 0-crossing operating point inside that ranking).

## Pre-registered numbers
- **Predicted:** mean AUROC ≥ 0.80 across the 3 adapters × on/off split; sign-at-zero accuracy > 0.60.
- **Refutation threshold (KILL 2300):** mean AUROC of `s_i` (on-domain vs off-domain), averaged over the
  3 adapters, **< 0.70** ⇒ `killed`.
- Reported but not gating: per-adapter AUROC, sign-at-0 accuracy, mean s on/off per adapter.

## AUROC definition (no sklearn)
For adapter `i`: positives P = {s_i on its own domain's prompts}, negatives N = {s_i on the other two
domains' prompts}. AUROC = (1/|P||N|) Σ_{p∈P,n∈N} [ 1·(p>n) + 0.5·(p==n) ] — the Mann–Whitney U statistic,
exact and dependency-free.

## Real, not mock
Real frozen 4-bit Gemma, real trained adapters, real held-out prompts, real forward passes. No training, no
merging, no router. `is_smoke:false`.
