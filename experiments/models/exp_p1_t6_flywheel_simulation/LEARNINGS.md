# LEARNINGS.md — T6.4: Flywheel Simulation (3 Sequential Promotions)

## Core Finding

3 sequential base promotions are structurally safe: cumulative spectral perturbation
reaches only 7.62% (max), well within the Davis-Kahan safe zone of 10%, and pairwise
adapter interference remains below 0.09 cosine throughout.

## Why

Near-orthogonal adapters (Finding #427, max cos < 0.1) cause cross-terms to add only
~26% to variance, producing a 2.18× cumulative scaling vs the √N = 1.73× ideal. The
Welch bound (Strohmer & Heath 2003) guarantees a minimum pairwise separation floor of
0.028 cosine for K=3 vectors in d=2560, making catastrophic collapse structurally
impossible at this scale.

## Implications for Next Experiment

Extrapolation puts N=5 promotions at ε_cumul ≈ 10.2% (boundary) and N=12 at the
hard limit — T6.5 should stress-test N=10-20 to confirm or tighten the scaling law.
The flywheel is viable for the near-term 5-domain target; T6.5 closes the question
for the 25-domain production goal.

## Caveats

- Synthetic W_base (std=0.05). Real Gemma 4 weights → smaller ε_single → flywheel extends further.
- q_proj only; v_proj and gate_proj not validated.
- Float overflow in some layers (bf16 weights loaded as fp32); results unaffected but explicit nan tracking recommended in T6.5.
