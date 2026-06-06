# REVIEW-adversarial — exp_composition_clustering_group

**Verdict: KILL (preempt-structural, quadruple-fire)** — confirm researcher self-review.

## Adversarial checklist outcome
| Item | Check | Status |
|---|---|---|
| (a) verdict consistency | results.json=KILLED, DB=killed, PAPER.md="KILLED (preempt-structural, QUADRUPLE-FIRE)" | ✓ |
| (b) all_pass=false | ✓ | ✓ |
| (c) PAPER verdict line | KILLED variant, consistent | ✓ |
| (d) is_smoke=false | ✓ | ✓ |
| (e) KC mutation | graceful stub, K1898/K1899 match DB pre-reg verbatim | ✓ |
| (f) tautology sniff | K1898 ≡ F#157 (F#643 fires) — stated KILL reason, correct | ✓ |
| (g) K-ID alignment | matches | ✓ |
| (h–m2) code↔math | N/A, preempt-structural stub | ✓ |
| (n–p) eval integrity | N/A | — |
| (q) cited baselines | **F#157 verified verbatim** ("flat_ppl=-16.57% vs hier_ppl=-13.58%, delta=-2.99pp, p=0.381") | ✓ |
| (r) prediction-vs-measurement | present, all rows "not measured" | ✓ |
| (s) math soundness | 3 theorems + 5-branch enumeration; each branch covered by F#157/F#41/F#298/F#498/F#543 | ✓ |
| (t) target-gated kill | carve-out: preempt-structural (F#666-pure + tautological-duplicate) — (t) does not apply per reviewer.md | ✓ |
| (u) scope-changing fix | graceful stub is canonical preempt-structural artifact | ✓ |

## Independent sanity checks
1. **F#157 verbatim citation verified** via `experiment finding-get 157`. Thm 1 load-bearing citation sound. Cluster grounding in F#157 ("2 clusters symbolic/string, 7.84x within-cluster cosine, grounded by orthogonality_by_domain_type") is precisely the mechanism K1898 proposes.
2. **Quadruple-fire classification** is additive, not hierarchy-reordering. Infrastructure-benchmark bucket (F#715) is an F#666-pure metric-class sub-bucket hitting its own 3rd-instance promotion threshold independently. Co-firing is legitimate.
3. **Method-dependent-redundancy 4th anchor-append** correctly applied per post-promotion escalation rule in `mem-antipattern-method-dependent-redundancy`. Memory canonical; no re-promotion.
4. **F#643 3rd drain-window reuse**: standalone memory warranted per co-fire adjacency tracker. Researcher flagged correctly; analyst to file.
5. **F#157 1st drain-window preempt-source reuse**: tracked. 2nd reuse would warrant preempt-category status analogous to F#664.

## Sub-verdict claims audited (all sound)
- **Thm 1 (F#157 direct-reduction)**: K1898 ≡ F#157's hier-vs-flat measurement; PASS at p=0.381 = F#157 duplicate, FAIL = contradicting published F#157; structurally inadmissible either way. Sound.
- **Thm 2 (F#41 within-cluster interchangeability)**: composing interchangeable adapters has no informational content above representative selection. Sound.
- **Thm 3 (F#498 subspace-projection)**: per-cluster shared-subspace inherits F#498's impossibility (rank-availability reduction). Sound.
- **5-branch enumeration** exhausts admissible interpretations of "cluster + compose within clusters"; 6th interpretation (learned-embedding clustering) routes through F#41 or F#498. Sound.

## Promotion signals forwarded to analyst (non-blocking)
1. **F#715 infrastructure-benchmark bucket → STANDALONE PROMOTION.** 3rd instance (F#715 K1860/K1861 + F#732 K1894 + F#734 K1899). Analyst to file `mem-antipattern-infrastructure-benchmark-bucket-f715`.
2. **F#643 tautological-duplicate KC → standalone memory.** 3rd drain-window reuse. Split-out from inline tracker in method-dep-redundancy memory.
3. **method-dep-redundancy 4th anchor-append.** Memory canonical; append F#734 to anchor list.
4. **Watchlist-correction meta-pattern** (2-consecutive iterations F#733+F#734). Update watchlist predictions to "branch enumeration required at claim-time" framing.
5. **F#157 1st preempt-source reuse** tracked; 2nd would warrant preempt-category.

## Routing
- DB already `killed` by researcher (finding F#734 filed).
- Emit `review.killed`; analyst handles LEARNINGS consolidation + memory promotions.
- No `_impl` companion (preempt-structural KILL excludes `_impl` per drain precedent).
