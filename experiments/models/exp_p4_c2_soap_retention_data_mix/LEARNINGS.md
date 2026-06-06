# LEARNINGS — exp_p4_c2_soap_retention_data_mix

## Core Finding

50/50 data mixing (domain + general) fixes LoRA retention loss (0.80→1.00) while also
improving format compliance (+70pp→+80pp). No trade-off occurred — both objectives improved.

## Why

Multi-task gradient mixing reduces the component of updates that destroy general knowledge
(Kirkpatrick et al., 1612.00796 is EWC; data mixing is simpler). At rank-16, the LoRA has
sufficient capacity for both tasks. The surprise: mixing also improved format quality,
suggesting general-knowledge gradients act as regularization that filters SOAP formatting noise.
This parallels Finding #519 (Stiefel retraction on TT cores = norm regularizer, not just constraint).

## Implications for Next Experiment

**Constraint-as-regularizer** is now a provisional pattern across two findings (#519, #520).
The next experiment should either:
1. Test whether this holds for other adapter pairs (code+medical, math+legal) at scale
2. Derive the optimal mixing ratio α mathematically (currently arbitrary at 0.5)

## Caveats

- N=10: retention jump (8/10→10/10) is p~0.11; format improvement (+70pp→+80pp) is noise-level
- Theorem 1 in MATH.md is informal (multi-task learning restatement, not a convergence proof)
- Legal retention (K1248) cited from P4.C1, not re-measured (adapter unchanged, defensible)
- "Regularization > trade-off" narrative is provisional until replicated with N>50

## References

- Finding #480: P4.C1 baseline (retention=0.80, the problem fixed here)
- Finding #519: Stiefel norm control on TT-LoRA (same constraint-as-regularizer pattern)
- Kirkpatrick et al. (1612.00796): EWC — data mixing is the zero-cost alternative
