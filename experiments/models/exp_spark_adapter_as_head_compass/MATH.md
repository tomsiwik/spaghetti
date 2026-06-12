# MATH — adapter delta as a read-only head compass

## Frame
A LoRA math adapter on `self_attn.q_proj` (rank r=6, scale 6.0) is **net-negative when applied**
to gemma-4-e4b-it-4bit (F#866/873): injecting `Δ = scale · (A @ B)` into the q_proj output degrades
GSM8K. HYPOTHESIS: the delta's **output column space still points at the heads that matter**. So we
NEVER apply Δ to the forward pass. We use it only as a *direction-finder* to select which **frozen base
q_proj heads** to amplify.

## Setup / dimensions
- hidden H = 2560, q_proj output Dq = 2048 = `n_heads (8) × head_dim (256)`.
- adapter per layer ℓ: `A_ℓ ∈ R^{2560×6}`, `B_ℓ ∈ R^{6×2048}`. Applied delta on q output would be
  `Δq = scale · (x @ A_ℓ) @ B_ℓ` — a vector in the 2048-dim q-output space whose reachable directions
  are exactly `rowspace(B_ℓ)` (a ≤6-dim subspace of the 2048-dim output).
- The q-output space splits into 8 contiguous 256-dim head blocks: head h owns coords `[256h : 256h+256]`.

## Compass score (read-only)
For layer ℓ and head h, score = energy the adapter delta would write into that head's output block:

    score(ℓ, h) = mean( B_ℓ[:, hd·h : hd·(h+1)]^2 )      (mean squared energy per coordinate in block h)

where `hd` = that layer's head_dim (256 sliding, 512 full-attention). We use the **per-coordinate mean**
(not the Frobenius sum) so wider full-attention head blocks do not win merely by having more coordinates —
the compass ranks by direction strength, not block size. This is a pure function of the **frozen adapter
weights** — no forward pass through the adapter, no x.
Sum over all layers gives a global per-head importance; we also keep per-layer scores. We select the
top-K (ℓ,h) head-slots by score as the "compass-selected" set.

## Intervention (amplify base, NOT apply delta)
Replace the base q_proj output `q ∈ R^{...×2048}` by `q' = q ⊙ m`, where `m` is a fixed diagonal mask:
`m[256h:256h+256] = γ` for selected heads (factor γ ∈ {1.1,1.2,1.3,1.5}), else 1.0. This is a
coordinate-wise rescaling of the **base** head's own query — `Δ` never enters the residual stream.

ASSERTION enforced in code: the amplified output equals `q_base ⊙ m` exactly; no additive `B@A` term.
The masking and the adapter weights are kept on disjoint code paths; we assert `m` is built only from
the *selection set* (integers), never from delta values.

## Prediction
- Compass-amplify EM exceeds base EM by ≥ +4pp AND exceeds random-amplify (same head count, same γ,
  fixed seed 1234) by ≥ +4pp, at the best γ in the sweep.
- Predicted compass-amplify EM ≈ base + 5-8pp at γ≈1.2-1.3 (small enough to not break attention).

## Refutation (pre-registered kill 2310, verbatim)
"GSM8K EM (n>=60) for amplifying adapter-compass-selected base q_proj heads does NOT exceed BOTH
(a) no-intervention base AND (b) amplifying random-selected heads (same count/factor) by >=+4pp each;
OR best result requires applying the adapter delta to logits at all"

Numeric threshold: let `c*` = best compass-amplify EM over γ sweep, `b` = base EM, `r*` = best
random-amplify EM (matched γ). SUPPORTED iff `c* - b >= 4.0` AND `c* - r* >= 4.0`. Otherwise KILLED.
A delta-applied arm is included ONLY as a labeled refuting context arm; if it is the best arm, that is
itself a refutation by the second clause.
