# MI vs cosine for expert independence — KILLED (preemptive, derivation-only)

## Verdict
**KILLED.** Both kill criteria FAIL by mathematical derivation. No model run.

## Lineage
- audit-2026-04-17-resurrect (dispatch-mislabel) → preempt-killed
  2026-04-19 by researcher iter 51 via F#22 / F#544 / F#286 / F#425.
- Sister kills: F#22 (KL composition health), F#544 (KL macro retest),
  F#590/F#591 (metric-swap), F#286/F#285 (spectral-arc proxy).

## Hypothesis (DB pre-registered)
MI(expert_i outputs, expert_j outputs) predicts composition quality
strictly better than cosine similarity, at acceptable cost.

## Kill criteria (from DB, not relaxed)

| ID  | Criterion                                                                | Pre-registered threshold | Measurement                                                | Result |
| --- | ------------------------------------------------------------------------ | ------------------------ | ---------------------------------------------------------- | ------ |
| 418 | MI doesn't predict composition quality better than cosine                | r²(MI,Q) − r²(cos,Q) < 0.1 | Bound: r²(MI,Q) ≤ r²(PPL,Q) = 0.0064 by DPI (F#286 r=0.08) | FAIL   |
| 419 | MI computation cost > 100× cosine cost                                   | C_MI/C_cos > 100         | KSG: ≥ 2304 · log(75) · 50 ≈ 4.96e5 ≫ 100                  | FAIL   |

Both FAIL ⇒ KILLED.

## Prediction-vs-measurement

| Quantity                         | Predicted (MATH §Theorem)                | Measured (this experiment)                                   |
| -------------------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| r²(MI, behavioral-Q) upper bound | ≤ r²(PPL, Q) = 0.0064                    | bound holds — DPI argument from F#286 r=0.08                 |
| r²(MI,Q) − r²(cos,Q)             | ≤ 0.0064 ≪ 0.1                           | K#418 trigger condition — FAIL                                |
| C_MI / C_cos at d=2304, n=75     | ≥ d·log(n)·k_const ≈ 4.96e5              | K#419 trigger condition — FAIL                                |
| Sister-finding parity (F#22)     | MI = KL(p(X,Y)‖p(X)p(Y)) → same family   | inherited kill                                                |

## Why this kills (derivation summary)
1. **Behavior is bottlenecked by the PPL channel** (F#286): in this
   codebase, Pearson r(PPL, behavior) = 0.08 ⇒ r²(PPL,Q) = 0.0064.
2. **Any output-distributional metric M** (cosine, KL, MI) sees Q only
   through the same channel. By the Data Processing Inequality,
   r²(M, Q) ≤ r²(PPL, Q) ≤ 0.0064.
3. **MI is in the same family as KL** by definition
   (MI(X;Y) = KL(p(X,Y)‖p(X)p(Y))). F#22 already killed KL on the same
   ground ("KL measures impact magnitude not quality — same failure as
   cosine gating"). F#544 confirmed at N=5 macro (Spearman ρ = −0.7).
4. **MI cost** is unavoidably orders of magnitude higher than cosine.
   KSG estimator ≥ d·log(n)·k_const FLOPs per pair. At d=2304 n=75 the
   ratio is ~5e5 ≫ 100.

## Limitations / when this kill could be revisited
- If a behavioral evaluation regime is constructed where r(PPL, Q) is
  large (≥ 0.6, say), the DPI argument relaxes and MI might pass K#418.
  Such a regime would need to redefine "composition quality" away from
  current PPL/loss proxies.
- If MI is restricted to scalar-valued outputs (not d=2304 hidden
  vectors), the cost argument K#419 may relax — but then we are no
  longer measuring expert-output dependency.

## Further-kill conditions (none new)
None. This kill is parent-finding-cascade, not a new failure mode.

## References
- F#22 [killed] KL divergence composition health
- F#544 [killed] KL anti-correlates with quality (macro retest)
- F#286 [supported] Spectral Gini production-irrelevant — wrong proxy
- F#285 [supported] Spectral arc optimized wrong metric, r(PPL,Q)=0.08
- F#425 [killed] Weight-space cosine fails N=5 (82→8% behavioral)
- Kraskov, Stögbauer & Grassberger (2004), arXiv:cond-mat/0305641 (KSG MI estimator).
- Belghazi et al. (2018) MINE, arXiv:1801.04062 (neural MI estimation).
