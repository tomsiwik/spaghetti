# MATH — conflict-subspace deflation beats uniform-1/N

## Setup

Frozen base `mlx-community/gemma-4-e4b-it-4bit`. Two real LoRA adapters on `self_attn.q_proj`
(rank r=6, native train scale s=6.0, 42 layers, identical key set verified):
`data/adapters/math` (GSM8K) and `data/adapters/medical` (MedQA-domain flashcards).

Per layer ℓ the LoRA delta-WEIGHT (applied as `y = W x + (Δ x)` with row-vector x, so Δ acts as a
right-multiply `x @ (A B)`):

    Δ_math^ℓ = s · (A_math^ℓ B_math^ℓ)        shape (d_in=2560, d_out∈{2048,4096})
    Δ_med^ℓ  = s · (A_med^ℓ  B_med^ℓ)

The **summed** delta is

    D^ℓ = Δ_math^ℓ + Δ_med^ℓ ,   rank(D^ℓ) ≤ 12.

## Arms (matched total scale, same merged model on the same mixed eval)

Uniform-1/N merge (N=2) applies coefficient c = 1/N = 1/2 to the summed delta:

- **(i) uniform-1/N** (standing baseline, keeps winning F#863/867):
      y = W x + c · D^ℓ x .
- **(ii) conflict-deflated**: SVD `D^ℓ = U Σ Vᵀ`; build projector that nulls the top-k≤4
  RIGHT-singular directions (the output-space modes that carry the largest summed energy):
      D̃^ℓ = D^ℓ (I − V_k V_kᵀ)   where V_k = top-k right-singular vectors (d_out × k),
      y = W x + c · D̃^ℓ x .
  This removes ONLY the k over-amplified shared modes; all other directions keep the SAME c=1/2.
- **(iii) base** (context): y = W x.

Both (i) and (ii) carry **identical** total Frobenius budget minus exactly the top-k modes —
matched scale c=1/2 everywhere else. The ONLY structural difference is the rank-k null.

## Why uniform-1/N cannot do this (non-obvious lever)

1/N scales ALL singular directions of D by the same c. It shrinks the clash mode and the useful
modes together; it can never *selectively* remove the direction where Δ_math and Δ_med
constructively over-amplify. Deflating the top-k right-singular modes of the summed D is exactly
the move uniform scaling is structurally incapable of. If composition damage concentrates in a
tiny shared subspace, σ₁..σ_k ≫ σ_{k+1}, and removing them should recover behavioral accuracy
that uniform 1/N leaves on the table.

## Prediction (theorem-as-hunch)

If the damage lives in a tiny GLOBAL shared subspace, then on a **mixed off-domain eval**
(GSM8K math exact-match #### + MedQA-USMLE 4-option letter exact-match), aggregate behavioral
accuracy of arm (ii) exceeds arm (i):

    acc(ii) − acc(i) ≥ +3.0 pp   (aggregate, exact-match).

Auxiliary evidence the subspace is tiny: the per-layer spectrum of D should show
σ₁/σ₁₂ large (a few dominant modes), reported in `results.json`.

## Pre-registered refutation (kill 2307, verbatim)

> Deflating the top-k<=4 SVD directions of the SUMMED delta (BA_math+BA_med) per layer recovers
> <+3pp behavioral accuracy on a mixed off-domain eval vs the uniform-1/N merge baseline -> KILLED

Numeric threshold: let Δacc = acc_aggregate(ii) − acc_aggregate(i) in percentage points.

    Δacc < +3.0 pp  ⇒  verdict = KILLED.
    Δacc ≥ +3.0 pp  ⇒  verdict = SUPPORTED.

k is swept over {1,2,3,4}; the arm-(ii) score reported for the kill is the **best k** (a fair test
of "does deflating a tiny subspace help at all"), with all four k reported. Scale s=6.0, merge
coefficient c=1/2 (matched). N_math = N_med = 80 problems each (160 total), greedy decode, real
exact-match. is_smoke=false.
