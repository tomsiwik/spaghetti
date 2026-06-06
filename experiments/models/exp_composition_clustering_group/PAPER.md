# exp_composition_clustering_group — PAPER.md

## Verdict
**KILLED (preempt-structural, QUADRUPLE-FIRE — first in drain history)**

Fires: method-dependent-redundancy 4th (anchor-append) + F#666-pure 22nd + §5 14th + **infrastructure-benchmark bucket F#715 3rd (STANDALONE PROMOTION TRIGGER)**. Plus F#157 direct-reduction preempt and F#643 tautological-duplicate KC 3rd drain-window reuse (standalone memory warranted).

## Prediction-vs-measurement table

| KC | MATH.md prediction | Measurement | Match |
|---|---|---|---|
| K1898 "cluster-based ≈ flat" | **FAIL** (indistinguishable-or-worse across 5 branches) | Not run; F#157 directly: hier_ppl=−13.58% vs flat_ppl=−16.57% (delta=−2.99pp wrong direction). F#41: within-cluster interchangeable to 0.12%. F#498: subspace projection destroys composition. | ✓ preempt-structural |
| K1899 "clustering overhead > 10ms" | **INADMISSIBLE** (F#715 bucket) | Not run; F#157 already reports 0.3% overhead at micro scale; behavioral threshold uncalibrated | ✓ preempt-structural |

## Finding ledger (references cited)
| Finding | Relevance |
|---|---|
| **F#157** | Hierarchical composition KILLED at delta −2.99pp (wrong direction). "Domain hierarchy is data organization, not composition architecture." Direct preempt-source. |
| **F#41** | Within-cluster adapters interchangeable to 0.12% range. Clustering composition is informationless above representative selection. |
| **F#298** | N=24 composition: routing bottleneck is semantic domain overlap; within-cluster misrouting is PPL-benign. |
| **F#498** | Universal subspace compression destroys composition orthogonality (max_cos 0.60→0.96). Any subspace-projection interpretation of clustering composition is KILLED. |
| **F#543** | Uniform additive composition at N=5 → 2.57× degradation. Additive-over-cluster-means remains additive. |
| **F#66** | Bf16 merge / runtime LoRA equivalent modulo bandwidth. Composition algebra baseline. |
| **F#643** | Tautological-duplicate KC antipattern. K1898 ≡ F#157 verbatim → 3rd drain-window reuse = standalone memory trigger. |
| **F#664** | Fixed-algebraic-blend preempt category. Cluster-mean-combine (additive across K cluster-reps) falls in family. |
| **F#666** | Target-gated KILL. K1898 "not significantly different" on no target metric. K1899 threshold behaviorally uncalibrated. |
| **F#715** | Infrastructure-benchmark preempt bucket. K1899 "10ms overhead" is pure wall-clock measurement with no behavioral calibration. **3rd instance = STANDALONE PROMOTION TRIGGER.** |
| **F#731/F#732/F#733** | method-dependent-redundancy anchors 1–3. This (F#734) is 4th anchor-append. |

## Branch enumeration (5-way partition)

| Branch interpretation | Mechanism | Prior finding | K1898 outcome |
|---|---|---|---|
| Hierarchical composition | F#157's original (SVD foundation + per-cluster residual) | F#157 KILLED | −2.99pp wrong direction |
| Similarity-k-means clustering | k-means on A/B cosine; compose within group | F#41 SUPPORTED | interchangeable 0.12% → composition informationless |
| Routing-only clustering | Cluster→route, use representative | F#298 PROVISIONAL | within-cluster misrouting PPL-benign (not a composition mechanism) |
| Subspace-projection clustering | Per-cluster shared-subspace projection | F#498 SUPPORTED | projection destroys composition |
| Ordinal ensemble | Cluster-mean × K, additive combine | F#543 / F#66 / F#664 | additive over means = F#543 degradation + F#664 preempt |

∀ branch ∃ prior finding → outcome derivable without running. **method-dependent-redundancy fires (4th instance).**

## Antipattern audit results
| Antipattern | Fires? | Evidence |
|---|---|---|
| method-dependent-redundancy | **YES** (4th instance, anchor-append per post-promotion escalation) | 5-branch enumeration above |
| F#666-pure target-unbound | **YES** (22nd reuse, 2 KCs) | K1898 no metric; K1899 threshold-uncalibrated |
| §5 tautological-inter-variant-delta | **YES** (14th reuse) | K1898 literally cluster-minus-flat |
| Infrastructure-benchmark bucket (F#715) | **YES (3rd instance — STANDALONE PROMOTION TRIGGER)** | K1899 wall-clock-only, threshold uncalibrated |
| F#157 direct-reduction preempt | **YES** (1st drain-window reuse as preempt-source) | Hierarchical composition directly killed |
| F#643 tautological-duplicate KC | **YES** (3rd drain-window reuse → standalone memory warranted) | K1898 ≡ F#157 |
| composition-bug / LORA_SCALE / shutil.copy / hardcoded pass / eval truncation / proxy model | N/A | no code executes empirical path |

## F#702 hygiene-patch compliance
- `platform`: `local-apple` set ✓
- `experiment_dir`: `micro/models/exp_composition_clustering_group/` set ✓
- `references`: 11 findings cited inline (F#41/F#66/F#137/F#157/F#298/F#498/F#543/F#643/F#664/F#666/F#715) + F#731/F#732/F#733 anchors ✓
- `evidence`: populated via `--evidence` flag on complete ✓
- `success_criteria`: N/A (KILLED; flag unsupported per drain precedent)

## Quadruple-fire classification note
First drain-history quadruple-fire. Prior max = triple-fire (9 instances; `mem-pattern-triple-fire-hierarchy-axis-invariant` canonical). The 4th fire here is **infrastructure-benchmark bucket (F#715)**, which reaches its own standalone-promotion threshold at 3rd instance — previously the axis was sub-threshold and tracked inline via `mem-antipattern-method-dependent-redundancy` watchlist. K1899 triggers split-out.

Per triple-fire-hierarchy memory, the hierarchy is axis-invariant: F#666-pure (KC class) > hygiene-multi-defect (metadata) > ... — infrastructure-benchmark is an F#666 sub-bucket (metric-class axis), not a hierarchy-reordering fire. The quadruple classification is additive, not hierarchical.

## Post-promotion stability note (method-dep-redundancy 4th instance)
Per `mem-antipattern-method-dependent-redundancy` escalation rule: "4th instance → still anchor-append (this memory canonical; no tier-2 promotion planned)." F#734 appended as 4th anchor alongside F#731/F#732/F#733. Watchlist entry for `exp_composition_clustering_group` was marked "uncertain, candidate for PROVISIONAL design-lock, NOT method-dependent redundancy" — **prediction falsified**, per F#733's lesson reinforced: watchlist must enumerate branches, not argue from "novel mechanism" at the top level.

## Assumptions (autonomy log per guardrail 1008)
1. "Similarity" is under-specified in DB notes. 5 natural interpretations enumerated; each covered. Any 6th interpretation (e.g., learned-embedding clusters) routes through F#41 or F#498.
2. N ≥ 3 (clustering vacuous at N ≤ 2). F#543's N=5 result and F#298's N=24 result both exceed threshold.
3. "Flat composition" baseline = uniform additive per F#543 / F#66. Any alternative baseline (PPL-probe F#137, null-space F#496) strengthens the KILL.
4. "Clustering overhead" (K1899) = clustering assignment + per-cluster composition aggregation. F#157's 0.3% bounds the runtime at micro scale.
