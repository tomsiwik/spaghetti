# PAPER — DFA R1: disjoint-frame B-projection does NOT cut N=2 behavioral interference

**Experiment:** `exp_bet_dfa_r1_n2_composition` · **Bet:** dfa-init (R1) · **Verdict: KILLED (K2314)**

## Setup (as pre-registered in MATH.md)

Frozen `mlx-community/gemma-4-e4b-it-4bit`, r=6 q_proj LoRA adapters python(code) + math
(`data/adapters/{python,math}`), scale 6.0, 42 layers. Per layer, joint QR of
`[B_py^T | B_math^T]` yields frozen disjoint orthonormal frames (measured
`max|Q_py^T Q_math| = 1.95e-16` — Theorem 1's zero output-overlap held exactly). Each adapter's
delta-output projected onto its own frame; composition `Σ P_i (B_i A_i)`. Behavioral harness,
no-thinking, greedy: GSM8K n=200 (math, the F#827 interference axis), HumanEval n=100 (code,
solo-preservation probe). Real unit-test / exact-match execution, `is_smoke:false`, 7328 s.

## Prediction vs measurement

| Quantity | Predicted | Measured | Met? |
|---|---|---|---|
| Interference gap `B−C` (math-solo vs naive sum) | ≥ 0.10 | **0.370** (0.705 → 0.335) | yes — interference real, 2.6× F#827's −14pp |
| Recovery `D−C` ≥ 0.5·gap | ≥ 0.185 | **0.065** (17.6% of gap) | **no** |
| Residual drag `B−D` | ≤ 0.07 | **0.305** | **no** |
| Solo drop `E−F` (code projected vs not) | ≤ 0.02 predicted, kill at >0.05 | **0.050** (0.38 → 0.33) | kill not triggered (at threshold), but ≈0pp prediction missed |

GSM8K: A(base)=0.140 · B(math-solo)=0.705 · C(naive sum)=0.335 · D(DFA sum)=0.400.
HumanEval code-solo: E(unprojected)=0.380 · F(projected)=0.330.

## Kill criteria (pre-registered)

- **K2313** (frame destroys skill, >5pp solo drop): **pass** — drop exactly 0.050, at but not over
  the threshold. Note the prediction was ≈0pp (projection is a numerical no-op on python's own
  delta, rel-err ~1e-3); a 5pp behavioral swing from a 1e-3 weight-space perturbation is itself
  informative noise/sensitivity.
- **K2314** (interference not cut ≥50%): **FAIL** — recovered only 17.6% of the gap
  (0.065 / 0.370); residual drag 0.305 ≫ 0.07. KILLED.

## Verdict line

**VERDICT: KILLED — exact param-space output-orthogonality (Q_py^T Q_math = 0 to 2e-16, every
layer) recovered only 17.6% of a 37pp behavioral interference gap; the ≥50% gate required 18.5pp
recovered, we measured 6.5pp.**

## What this kills, and what it doesn't

This is precisely the "honest risk" the dfa-init ladder pre-registered: *param-space disjointness
does not buy behavioral non-interference* — the same gap that killed A-orthogonality now kills
B-orthogonality. Even with mathematically exact zero overlap between the two adapters' delta-output
subspaces at every layer, the math skill stays 30.5pp below its solo ceiling. Interference is
therefore not a linear-algebraic collision in any single layer's output space; it propagates
through the nonlinear residual stream (each adapter's delta changes the *inputs* to all later
layers, so the other adapter's A sees off-distribution activations — an effect no output-side
frame can remove).

Per the ladder: **R2 (train with the frame) is now skipped**; the surviving bet is **R3** — the
function-space (JEPA shared-predictor) objective, since only train-time alignment of *behavior*,
not geometry, addresses the mechanism this result exposes. The D−C = +6.5pp crumb suggests
output-side deflection is weakly directionally right but an order of magnitude too small.

Caveat for cross-experiment comparison: per the harness-relative-EM finding, these EMs are only
comparable within this run's no-thinking harness; the B−C gap (37pp) is larger than F#827's −14pp
partly for that reason.
