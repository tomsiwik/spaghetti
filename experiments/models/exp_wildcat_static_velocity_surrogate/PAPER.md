# PAPER — exp_wildcat_static_velocity_surrogate

## Claim under test
The F#862 trajectory-defined "velocity core" of a thinking adapter (entries at ≥80% of final
magnitude by step 200, stable sign) is a latent variable recoverable from the **final weights alone**
(checkpoint-free), via one of three pre-registered static surrogates at matched sparsity
f* = 0.4335: S1 magnitude, S2 top-SVD agreement (rank 4), S3 factor-energy.

Setup (exact F#862 harness): frozen `mlx-community/gemma-4-e4b-it-4bit`, math adapter on q_proj
scale 6.0 + masked dense thinking delta on v_proj/o_proj scale 1.0, GSM8K n=50, greedy,
max 1024 new tokens, seed 42, mlx_lm 0.31.2.

## Prediction vs measurement

| Quantity | Predicted | Measured | Pass? |
|---|---|---|---|
| Full-thinking baseline B | ~0.44 | 0.44 | yes (harness reproduced) |
| Ground-truth core anchor EM | ~0.74 | 0.74 | yes (F#862 reproduced exactly) |
| Best surrogate EM | ≥ 0.68 | 0.68 (S2 top-SVD) | yes (boundary) |
| Best surrogate − random null | ≥ +6pp | **+2.0pp** (0.68 vs 0.66) | **no** |
| Jaccard(best surrogate, GT core) | ≥ 0.45 (random ≈ 0.277) | 0.442 (S2); random 0.277 | no (marginal) |
| Ordering | S1 ≈ S2 > S3 > random | S2 (0.68) > random (0.66) > S3 (0.56) > S1 (0.50) | no |

Per-condition EM: B_full_thinking 0.44, s1_mag 0.50, s3_energy 0.56, **random 0.66**,
s2_svd 0.68, gt 0.74.

Jaccard vs GT core: s1_mag 0.361, s2_svd 0.442, s3_energy 0.357, random 0.277.

## Kill criterion (pre-registered, #2334)
KILL if best surrogate EM ≤ random null + 2pp OR best EM < 0.59.
Measured: 0.68 ≤ 0.66 + 0.02 — the null clause fires (absolute clause passes).
Gate (best ≥ 0.68 AND ≥ null + 6pp): fail.

## VERDICT: KILLED — best static surrogate (S2 top-SVD, EM 0.68) beats the random-mask null (0.66) by only +2.0pp, inside the pre-registered ≤+2pp kill band; the velocity core is not recoverable from final-weight geometry.

## Interpretation
1. **The strongest result is the null itself.** A *random* mask at f* = 0.4335 recovers 0.44 → 0.66
   (+22pp of the +30pp gap) despite chance-level core overlap (Jaccard 0.277). Most of the F#862
   benefit comes from *sparsifying the thinking delta at all* — diluting the destructive interference
   with the math adapter — not from selecting the velocity-core entries specifically.
2. The trajectory ground truth still carries real signal: GT (0.74) beats random (0.66) by +8pp at
   identical sparsity. The kill says final weights don't encode *which* entries those are: even the
   best surrogate's mask overlap (0.442) sat below the 0.45 prediction, and its +2pp over random is
   consistent with that weak overlap.
3. The "honest risk" in MATH.md materialized: the velocity criterion is a ratio test, and the core is
   magnitude-balanced — S1 (magnitude) actually *underperformed* random (0.50 vs 0.66).
4. Consequence: checkpoint-free deployment of the F#862 fix is impossible; capturing the velocity core
   requires trajectory logging at training time. Follow-up worth a spec: characterize the random-mask
   dilution effect (EM vs sparsity curve), since it is cheap, checkpoint-free, and delivers ~73% of
   the GT recovery.

Artifacts: `results.json` (verdict: killed, all_pass: false, is_smoke: false,
wall clock 7818 s), `MATH.md`, `run_experiment.py`.
