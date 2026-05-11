# PAPER — Composition behavior of joint-Stiefel-trained adapters

## Verdict: KILLED (dependency unmet)

This experiment requires adapter weights from `exp_pierre_joint_stiefel_b_train`,
which was itself KILLED (B-only Stiefel insufficient for composition — K3/K4 failed).
No weights were saved to disk from that run.

Additionally, the joint training experiment already demonstrated that B-orthogonality
alone does not deliver composition guarantees (cross-contribution 3.4%–501k% vs 1%
threshold). Testing different composition operators on those weights would not change
the fundamental finding: A-matrix coupling breaks composition regardless of how B's
are merged.

## Kill criteria outcomes

- **K1**: UNTESTED — no weights to evaluate
- **K2**: UNTESTED — no weights to evaluate
- **K3**: UNTESTED — no weights to evaluate
- **K4**: UNTESTED — no weights to evaluate

## Why this is the right outcome

The upstream experiment proved that joint-Stiefel on B alone is necessary but not
sufficient. Smarter composition operators cannot fix the A-matrix coupling problem.
The path forward is double-Stiefel (A+B) or factored composition that accounts
for the full A@B contribution, not better averaging of B matrices.
