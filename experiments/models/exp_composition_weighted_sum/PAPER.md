# exp_composition_weighted_sum — PAPER.md

## Verdict
**KILLED (preempt-structural, triple-fire — method-dependent-redundancy 3rd instance post-promotion + F#666-pure 21st + §5 13th; bonus F#664 preempt-category 2nd reuse + F#643 tautological-duplicate)**

## Prediction-vs-measurement table

| KC | MATH.md prediction | Measurement | Match |
|---|---|---|---|
| K1896 "weighted < 2pp improvement over uniform at N=3" | **FAIL** (all branches bounded) | Not run; derived from F#664 (fixed ≤7pp loss), F#164 (learned diverges), F#137 (data-cond = +9.34pp ≫ 2pp but is verbatim F#137 per F#643) | ✓ preempt-structural |
| K1897 "learned weights overfit OOD" | **INCONCLUSIVE** | Not run; learned branch un-evaluable per F#164 landscape divergence; target-unbound per F#666-pure | ✓ preempt-structural |

## Finding ledger (references cited)
| Finding | Relevance |
|---|---|
| **F#664** | Fixed-algebraic-blend preempt category — covers uniform, TIES, Task-Arithmetic λ, RS-parity. "Any new variant proposing fixed algebraic weights is preempt-killable by reduction to F#157." 2nd reuse. |
| **F#157** | ≥7pp loss at rank-matched budget regardless of structured-vs-equal coefficient choice (hier_equal=-7.29%). |
| **F#164** | CAT learned-weight impossibility: orthogonal adapters (|cos|~0.001) → flat landscape → LRs diverge across {1e-4..1e-1}. Task Arithmetic λ=0.5 beats uniform by 8.1% but merge quality is about SCALING not WEIGHTING. |
| **F#137** | Relevance-weighted (PPL-probe n=10) gives +9.34pp over equal-weight (r=0.990). Data-CONDITIONED weights escape F#664's kill family but this experiment would be verbatim re-run. |
| **F#496** | Null-space weighted composition beats exclusive routing +32.7% on mixed-domain. |
| **F#22/F#544** | Linear composition anti-correlates with quality (ρ=-0.7, r=-0.46). |
| **F#543** | Uniform additive composition at N=5 → 2.57× degradation. By monotonicity N=3 still degrades. |
| **F#643** | Tautological-duplicate KC antipattern: K1896 ≡ F#137's finding at 2pp/9pp threshold class. |
| **F#731** | 1st method-dependent-redundancy instance (exp_composition_n5_scaling). |
| **F#732** | 2nd method-dependent-redundancy instance (exp_composition_runtime_vs_merge_n10) → PROMOTION fired. |

## Branch enumeration
| α_i class | Covered by | K1896 outcome | Redundancy note |
|---|---|---|---|
| Fixed uniform 1/N | F#543, F#157 | FAIL (0 improvement by construction) | baseline is uniform itself |
| Fixed Task-Arithmetic λ | F#664, F#164 | FAIL (within ≤7pp Δ=0.24pp noise) | F#164 gave +8.1% SCALING but merge quality is scaling not weighting |
| Fixed TIES | F#664 | FAIL | explicit in F#664 impossibility |
| Learned (CAT-style) | F#164 | un-evaluable (divergence) | learned branch structurally blocked |
| Data-conditioned (PPL-probe, relevance) | F#137 | PASS (+9.34pp) but = F#137 verbatim | F#643 tautological-duplicate |
| Null-space weighted | F#496 | PASS (+32.7%) but orthogonal mechanism, out of scope for "α_i * ΔW_i" | F#643 |

Every admissible branch collapses.

## Antipattern audit results
| Antipattern | Fires? | Evidence |
|---|---|---|
| method-dependent-redundancy | **YES** (3rd instance post-promotion, anchor appended) | Branch table above |
| F#666-pure target-unbound | **YES** (21st reuse, 2 KCs: metric + OOD) | K1896 "2pp" on what metric; K1897 "overfit" no OOD task |
| §5 tautological-inter-variant-delta | **YES** (13th reuse) | K1896 literally weighted-minus-uniform |
| F#664 preempt-category | **YES** (2nd reuse) | Direct application to fixed α branch |
| F#643 tautological-duplicate KC | **YES** | K1896 ≡ F#137 at 2pp/9pp threshold |
| composition-bug / LORA_SCALE / shutil.copy / hardcoded pass / eval truncation / proxy model | N/A | no code executes empirical path |

## F#702 hygiene-patch compliance
- `platform`: `local-apple` set ✓
- `experiment_dir`: `micro/models/exp_composition_weighted_sum/` set ✓
- `references`: 9 findings cited inline (F#137/F#157/F#164/F#496/F#543/F#643/F#664/F#22/F#544) + F#731/F#732 anchors ✓
- `evidence`: populated via `--evidence` flag on complete ✓
- `success_criteria`: N/A (KILLED; flag unsupported per drain precedent)

## Post-promotion stability note
Per the `mem-antipattern-method-dependent-redundancy` memory (promoted 2026-04-24 after F#732): 3rd instance is an **anchor append, not a re-promotion**. Pattern definition stands. This instance confirms the canonization under post-promotion drain protocol.

## Assumptions (autonomy log per guardrail 1008)
1. `d_model = 3584` for Gemma 4 E4B — per PLAN.md Part 2 / F#627. Used in Thm 2 flat-landscape justification.
2. "Learned weights" in the experiment notes means CAT-style gradient optimization of α_i (F#164 case). Hypernetwork-/TTT-based formulations are unclaimed by this experiment's framing (`α_i * ΔW_i` is scalar-per-adapter) and would need a distinct experiment.
3. N=3 inherits F#543's N=5 degradation by additive-composition monotonicity (more adapters → more delta-stacking; N=3 bounded by N=5's 2.57× floor and F#157's -7.29pp floor).
