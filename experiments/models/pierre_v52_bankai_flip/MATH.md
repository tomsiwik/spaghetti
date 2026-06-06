# MATH.md — Pierre v5.2 Bankai Row Flip

## Theorem (Ternary Row-Flip Impossibility, corollary of F#291)

Let a ternary BitLinear row be `r ∈ {-1, 0, +1}^d`. Define the Bankai-style
row flip as

    flip(r, s) := clip(r + s·1, -1, +1),   s ∈ {-1, +1}.

Then at least `n_sat(r, s)` of the `d` entries are mapped to a fixed point
(no semantic change) where

    n_sat(r, +1) = #{i : r_i = +1}   (all +1s saturate under increment)
    n_sat(r, -1) = #{i : r_i = -1}   (all -1s saturate under decrement)

For a base with i.i.d. sign-symmetric distribution over {-1,0,+1}, the expected
fixed-point fraction is exactly `1/3` regardless of direction.

## Proof

Direct computation on the three cases:
  r_i=+1, s=+1: clip(2, -1, +1) = +1  (fixed)
  r_i= 0, s=+1: clip(1, -1, +1) = +1  (shift, non-fixed)
  r_i=-1, s=+1: clip(0, -1, +1) =  0  (shift, non-fixed)
  r_i=+1, s=-1: clip(0, -1, +1) =  0  (shift, non-fixed)
  r_i= 0, s=-1: clip(-1,-1, +1) = -1  (shift, non-fixed)
  r_i=-1, s=-1: clip(-2,-1, +1) = -1  (fixed)

Under the sign-symmetric prior P(r_i=+1)=P(r_i=-1)=P(r_i=0)=1/3,
E[n_sat/d] = 1/3 for either direction. QED.

## Corollary (Reduction to F#291)

F#291 established: for a ternary base with K=3 levels, lossless integer merge
requires K ≥ 2·max_delta+1 = 3 only when max_delta ≤ 1 AND the operation is
applied only to non-boundary entries. Row-flip applies ±1 to every entry
uniformly, hitting boundaries on exactly one third of the row in expectation.

Greedy search does NOT circumvent this: each "beneficial" flip trades the
destruction of n_sat entries for the directional change of the remainder.
At out_features × layers × modules ≈ 210 × 3000 rows, the saturated fraction
produces O(10⁸) boundary clips per domain patch — the identical failure
regime v5.1 empirically measured (62M–515M boundary clips across ω∈{0,1,2,4}).

## Pre-registered Predictions

P1. Behavioral score ≤ 0.05 on the 5-domain suite (v5.1 measured 0.003).
P2. PPL ratio vs. base ≥ 10× at any non-trivial flip count.
P3. Greedy search cannot discover a flip set that improves PPL more than
    noise, because flipping any row destroys ~d/3 of its existing signed entries.

## Kill Criteria

K733: "Zero domain signal (PPL same as base after all flips)" → FAIL expected
K734: "Search > 30 min per domain" — inconclusive (not run)
K735: "Speed < 120 tok/s after apply" — inconclusive (not run; reference
       v5.1 measured 138.3 tok/s, so this one likely passes if run)

## Verdict

KILLED without empirical run. Impossibility-structure cited: F#291.
Antipattern-match: composition-bug / ternary-saturation family — same
boundary-clip failure v5.1 ran at cost of 629.7s. No new empirical
evidence needed; v5.1's PPL ∈ [10³, 10⁹] and behavioral=0.003 transfers
because the row-flip operation reduces to the same clip(r + δ, -1, +1)
computation v5.1 executed, only with fewer total deltas.
