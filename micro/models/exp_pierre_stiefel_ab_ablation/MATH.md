# MATH.md — Stiefel-A vs Stiefel-B vs both (ablation)

## Position in the arc

Last experiment of the Stiefel arc. Decomposes the wins observed in the
prior experiments into per-constraint contributions.

Pierre currently has:
- **Stiefel-A**: Grassmannian retraction on A (already in `polar_train.py`).
  Shared across K adapter siblings.
- **Stiefel-B**: new (being tested in `exp_pierre_joint_stiefel_b_train`).
  Joint constraint across K adapters.

Question: **does Stiefel-A alone (current) suffice, or does Stiefel-B
add real value when Stiefel-A is also active?**

If Stiefel-A + standard B already gives clean composition, we don't need
the much-more-expensive joint Stiefel-B training. If Stiefel-B alone
matches Stiefel-A + Stiefel-B, we could drop the A constraint and use
only B (simpler Pierre).

## Four-way ablation

Train K=7 adapters under each of 4 conditions:

| Condition | A constraint | B constraint |
|---|---|---|
| **(a) none** | unconstrained A (re-init from QR but no retraction) | unconstrained |
| **(b) A only** | Stiefel-A retraction (Pierre current default) | unconstrained |
| **(c) B only** | unconstrained | joint Stiefel-B retraction |
| **(d) both** | Stiefel-A | joint Stiefel-B |

Evaluate each under fixed composition method (simple_mean) AND under each
adapter's own native task.

## Pre-registered Kill Criteria

- **K1 (CURRENT IS BEST)** Condition (b) — Pierre's current Stiefel-A —
  beats (a) by ≥ 2pp avg.
  PASS → Stiefel-A alone is doing useful work (validates Pierre's
  existing design).
  FAIL → Stiefel-A retraction is performative; remove it.

- **K2 (B ADDS VALUE)** Condition (d) > (b) by ≥ 2pp on composed avg.
  PASS → joint Stiefel-B is worth the training cost.
  FAIL → Stiefel-A is doing all the work; don't refactor for B.

- **K3 (B ALONE)** Condition (c) ≥ (b) (B-only ≥ A-only).
  PASS → could drop A constraint, simplify Pierre.
  Probably FAIL given Pierre's design assumes A is the shared invariant.

- **K4 (DIMINISHING RETURNS)** `(d) - (a) > (b) + (c) - 2(a)`.
  Tests whether A and B constraints are independent (additive gain) or
  redundant (joint < sum).

## Verdict implications

| Condition wins (simplified) | Implication |
|---|---|
| (d) best, (b) and (c) intermediate | Both constraints help; ship both. |
| (d) ≈ (b) | Stiefel-B doesn't add value over Stiefel-A. Drop the joint-B arc. |
| (d) ≈ (c) | Stiefel-A doesn't add value over Stiefel-B. Simplify Pierre. |
| (a) best | Both constraints hurt training. Revert to unconstrained. (Unlikely given prior art.) |

## Implementation status

**SPEC — heaviest of the arc. Combines training infrastructure of all
prior experiments.**

Required:
1. Multi-task training pipeline (from `exp_pierre_joint_stiefel_b_train`).
2. Configurable A and B retraction toggles (each can be off/Stiefel).
3. Four full training runs (4 × 3.5h = 14h training).
4. Eval rig: each condition × {single, composed} = 8 eval passes.

Total: **~16h compute + the MLX infrastructure debt of all prior experiments.**

## Why this is the experiment that wraps the arc

If we run only the joint-Stiefel-B training and it works, we have a
SUPPORTED but **not attributed** result — we don't know if the A
constraint is contributing or not. This ablation closes that
attribution question, which matters for:
- Documentation: which guarantees does Pierre have?
- Future research: which constraint axis is the next frontier?
- Engineering cost: can we drop one constraint and simplify training?

## References

- All siblings in Stiefel arc
- PoLAR paper for Stiefel-A justification
- arxiv 2508.17901 / 2510.01938 for Stiefel-B prior art
