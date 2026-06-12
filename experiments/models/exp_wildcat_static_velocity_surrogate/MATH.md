# MATH — exp_wildcat_static_velocity_surrogate

## Question (the disease, not the symptom)
F#862 showed the compose-safe **velocity core** of a thinking adapter (entries whose effective delta
reached ≥80% of final magnitude by step 200, with stable sign) recovers +30pp GSM8K when composed with a
math adapter (0.44 → 0.74, beating the 0.70 math-solo ceiling). But the mask needs **training checkpoints**,
and the deployed adapter bank only has final weights. The disease: velocity is treated as a *trajectory-only*
observable. The question: **is the velocity core a latent variable already imprinted in the final weight
geometry**, recoverable checkpoint-free?

## Theorem (sketch)
Under linearized LoRA dynamics, gradient flow on ΔW = AB fits the largest spectral components of the
target correlation first (spectral bias / greedy low-rank learning, cf. Saxe et al. 2014; Arora et al. 2019
deep matrix factorization). Therefore entries learned **early with stable sign** are dominated by the
**top singular directions** of the final ΔW₁₀₀₀ and have **large final magnitude**; late-moving entries are
small-magnitude residue spread across the trailing singular directions. Hence a static surrogate built from
ΔW₁₀₀₀ alone — at the matched global sparsity f* = 0.4335 (the measured F#862 core fraction, a single scalar
hyperparameter) — should substantially overlap the trajectory-defined core and reproduce its behavioral
recovery.

Three pre-registered surrogates (per (layer, proj), keep top-f* entries by score):
- **S1 magnitude**: score = |ΔW₁₀₀₀|ᵢⱼ.
- **S2 top-SVD agreement**: ΔW_top = rank-4 SVD truncation of ΔW₁₀₀₀ (half of rank 8, via the
  QR-of-factors trick — exact, no large SVD); score = (ΔW_top ⊙ ΔW) / (ΔW² + ε), the signed fraction of
  each entry explained by the leading spectral components.
- **S3 factor-energy**: score = ‖A_row(i)‖ · ‖B_col(j)‖ (rank-1 envelope of where the adapter
  concentrates capacity; the "B-column norm" surrogate).

Null: **random mask** at the same per-(layer,proj) fraction f*. Anchor: the trajectory **ground-truth core**
(step 200 vs 1000, thresh 0.80, sign match — the exact F#862 mask), re-run in the same harness.

## Predicted numbers
- Jaccard(best surrogate, ground-truth core) ≥ 0.45 (vs E[Jaccard] ≈ f*/(2−f*) ≈ 0.277 for a random mask
  at f* = 0.4335).
- Behavioral, GSM8K n=50 (exact F#862 setting: math q_proj scale 6.0 + masked dense thinking v/o_proj
  scale 1.0 on frozen `mlx-community/gemma-4-e4b-it-4bit`, greedy, max 1024 new tokens):
  best surrogate EM ≥ **0.68** (≥80% of the 0.44→0.74 gap) and ≥ random null + 6pp.
- Expected ordering: S1 ≈ S2 > S3 > random; full-thinking baseline B ≈ 0.44.

## Refutation threshold (KILL 2334, pre-registered)
KILL if **best static-surrogate EM ≤ random-mask null + 2pp** OR **best EM < 0.59**
(< 50% recovery of the 0.44→0.74 gap), n=50, same harness as F#862.
If killed: velocity is genuinely a trajectory observable — final-weight geometry does not encode it, and
checkpoint-free deployment of the F#862 fix is impossible; adapters would need trajectory logging at
training time.

## Why this could fail (honest risk)
The velocity criterion is a *ratio* test (m₂₀₀ ≥ 0.8·m₁₀₀₀), not a magnitude test — small entries that
converged early are in the core, large entries still growing are out. If the core is magnitude-balanced
rather than magnitude-biased, S1/S3 collapse to ~random Jaccard and the kill fires. That is exactly what
the experiment measures.
