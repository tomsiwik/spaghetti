# exp_composition_clustering_group — MATH.md

## Verdict (pre-run, preempt-structural)
**KILLED** on three independent theorems + one direct prior-KILL reduction. Quadruple-fire across drain-window antipattern registry. The analyst-handoff watchlist predicted "likely novel, does NOT fire method-dependent-redundancy" — **prediction falsified** by F#157 direct preempt and F#41/F#298/F#498 branch coverage, per the F#733 lesson: watchlist predictions must be claim-time verified, not trusted as standing.

## 1. Experiment statement
Proposed: group N adapters by similarity (cosine on A or B, or domain embedding), compose within each cluster, and combine cluster outputs. Claim: reduces interference vs flat composition.

Pre-registered KCs (as filed in DB):
- **K1898**: "Cluster-based composition not significantly different from flat composition" (inter-variant delta; metric & task unspecified)
- **K1899**: "Clustering overhead > 10ms per inference" (infrastructure benchmark; threshold behaviorally uncalibrated)

Both KCs are F#666-pure malformed: K1898 has no target metric (what does "different" mean? PPL? accuracy? behavior?), K1899 has no behavioral calibration for the 10ms threshold.

## 2. Theorem 1 — F#157 direct-reduction preempt (hierarchical composition KILLED)

**Statement.** "Cluster adapters by similarity, compose within clusters" is the definition of hierarchical composition. F#157 KILLED this mechanism at delta −2.99pp (wrong direction, p=0.381 not significant but directionally worse than flat+PPL-probe).

**Proof (cite).** F#157 verbatim: "K1 KILL: flat_ppl=−16.57% vs hier_ppl=−13.58% (delta=−2.99pp, p=0.381, wrong direction). Foundation SVD averages away discriminative info. Across-cluster: +0.64pp (marginal). K2 PASS: overhead only 0.3%. PPL-probe already solves dilution without structural constraints. **Domain hierarchy is data organization, not composition architecture.**"

F#157's cluster grounding: "2 clusters (symbolic: arithmetic+parity; string: repeat+reverse+sort) grounded by orthogonality_by_domain_type (7.84x within-cluster cosine)." Precisely the mechanism K1898 proposes.

**Application.** K1898 asks: does "compose within clusters" differ from flat? F#157 already answered: hier hurts within-cluster by −2.99pp and barely helps across-cluster (+0.64pp, marginal). The net effect is directionally worse than flat. K1898 as-stated ("not significantly different") would PASS at p=0.381 — but the PASS is already F#157. Re-running is a verbatim re-measurement. **K1898 tautological-duplicate of F#157** (F#643 fires; see §5). QED for hierarchical branch.

## 3. Theorem 2 — F#41 within-cluster interchangeability (clustering adds no composition differentiation)

**Statement.** Within a domain cluster, adapters are interchangeable to 0.12% range. Therefore per-cluster composition produces outputs indistinguishable from using *any* cluster member or the cluster mean — "composing within a cluster" has no informational content above adapter selection.

**Proof (cite).** F#41 SUPPORTED: "Within-cluster adapters confirmed interchangeable (0.12% range). Composition weight landscape is smooth and convex. Uniform 1/N only 0.7% from optimal." F#298 (PROVISIONAL) reinforces: "Within-cluster misrouting is PPL-benign (music routed to code: 3.331 vs 3.809 base)." F#298 measured at N=24 Grassmannian.

**Application.** If within-cluster adapters are interchangeable to 0.12%, then `compose(cluster_k_adapters)` ≈ `any_one_of(cluster_k_adapters)`. The cluster-composition stage is informationless; the only remaining composition is the across-cluster combine — which reduces to flat composition over cluster-representatives. K1898's framing of cluster-composition as a mechanism distinct from flat is structurally wrong. **K1898 FAIL (not different by construction)**. QED for similarity-clustering branch.

## 4. Theorem 3 — F#498 universal-subspace branch KILLED (projection destroys composition orthogonality)

**Statement.** The alternative interpretation of "compose within clusters" as "project cluster's A-matrices onto a shared subspace, then compose in that subspace" was killed by F#498: universal subspace projection destroys composition orthogonality.

**Proof (cite).** F#498 SUPPORTED: "Universal subspace compression (K=8/11) preserves individual adapter quality (cos=0.96) but destroys composition orthogonality (max_cos 0.60→0.96)." Impossibility structure: "Grassmannian A-matrices have uniform PCA spectrum. Any shared subspace projection to K<N·r dimensions reduces the rank available for orthogonal packing, making interference mathematically unavoidable." Confirms F#65 on Gemma 4.

**Application.** If "cluster-based composition" is interpreted as subspace-sharing within clusters (the natural "dimensionality-reducing" form of clustering composition), F#498 proves the subspace shrink destroys orthogonality → interference mathematically unavoidable. **K1898 FAIL (cluster-projection makes composition strictly worse, not equal/better)**. QED for subspace-projection branch.

## 5. Branch enumeration and method-dependent-redundancy (4th instance, anchor-append)

Every admissible branch of "cluster adapters by similarity, compose within clusters" collapses to a prior finding:

| Branch | Definition | Prior finding | K1898 outcome |
|---|---|---|---|
| Hierarchical: (cluster, per-cluster SVD foundation + residual, combine) | F#157 original | **F#157 KILLED** (−2.99pp wrong direction) | FAIL directionally; PASS on "not different" threshold = F#643 duplicate |
| Similarity-clustering: (k-means on A/B cosine, compose within group) | Within-cluster mean adapter | **F#41 SUPPORTED** (0.12% interchangeable) | FAIL by construction (cluster-composition informationless) |
| Routing-only: (route input to cluster, use cluster's representative) | F#298 stage-2 routing | **F#298 PROVISIONAL** (within-cluster 41.5% acc, PPL-benign) | FAIL — this is routing, not composition |
| Subspace-projection: (per-cluster shared subspace, project then compose) | F#498 universal-subspace | **F#498 SUPPORTED** (destroys composition) | FAIL (strictly worse than flat) |
| Ordinal ensemble: (cluster-mean adapter × K clusters, additive sum) | F#66 bf16 merge + F#543 N=5 | **F#543** (2.57× degradation at N=5) | FAIL (additive over K cluster-means = flat-over-means) |

Every cell covered. **Method-dependent-redundancy FIRES — 4th drain-window instance, post-promotion anchor-append per `mem-antipattern-method-dependent-redundancy` escalation rule.**

## 6. Triple-fire hierarchy + infrastructure-benchmark bucket PROMOTION TRIGGER

This experiment fires **four simultaneous antipatterns** (first quadruple-fire in drain history; triple-fire-mode memory `mem-pattern-triple-fire-hierarchy-axis-invariant` active):

1. **method-dependent-redundancy** — 4th instance, anchor-append (branch table above).
2. **F#666-pure standalone** — 22nd reuse. K1898 target-unbound ("not different" on what metric?). K1899 threshold-uncalibrated (10ms from what behavioral budget?). Two bucket entries.
3. **§5 tautological-inter-variant-delta** — 14th reuse. K1898 literally measures `cluster-composition − flat-composition` with no absolute target.
4. **Infrastructure-benchmark bucket (F#715)** — **3rd drain-window instance → STANDALONE PROMOTION TRIGGER** (precedent: method-dep-redundancy promoted after 2nd instance F#732; but infrastructure-bucket has been tracked 2-instance: F#715 original + F#732 K1894). K1899 ("10ms overhead") is pure infrastructure measurement with behaviorally uncalibrated threshold.

Plus **F#157 direct-reduction preempt** (1st drain-window reuse of F#157 as a preempt-source, not a count bucket; precedent is F#664 which cites F#157 internally).

Plus **F#643 tautological-duplicate KC 3rd drain-window reuse** (K1898 ≡ F#157's hier-vs-flat result; 3rd reuse WARRANTS standalone memory per the co-fire adjacency tracker in `mem-antipattern-method-dependent-redundancy`).

## 7. Antipattern audit (pre-flight)
- Composition math bug: N/A (no code executes composition)
- LORA_SCALE: N/A
- shutil.copy adapter: N/A
- Hardcoded `"pass": True`: audited — `run_experiment.py` stub writes `all_pass=False`, verdict=KILLED
- Eval template truncation: N/A
- Proxy-model substitution: N/A
- F#643 tautological-duplicate KC: **FIRES** (K1898 ≡ F#157's hier-vs-flat)
- F#666-pure target-unbound: **FIRES** (K1898 + K1899)
- §5 tautological-inter-variant-delta: **FIRES** (K1898 cluster-minus-flat)
- Method-dependent-redundancy: **FIRES** (4th instance, anchor-append)
- Infrastructure-benchmark bucket (F#715): **FIRES** (3rd instance → PROMOTION)

## 8. F#702 hygiene-patch checklist (DB entry incomplete per `⚠ INCOMPLETE`)
- `platform`: set to `local-apple` (Gemma 4 target per PLAN.md Part 2)
- `experiment_dir`: `micro/models/exp_composition_clustering_group/`
- `references`: F#41, F#66, F#157, F#298, F#498, F#543, F#643, F#664, F#666, F#715 cited inline
- `evidence`: populated via `experiment complete --evidence`
- `success_criteria`: non-issue — KILLED verdict; CLI flag unsupported per drain precedent

## 9. Predicted outcome (preempt-structural, pre-run)
| KC | Predicted | Mechanism |
|---|---|---|
| K1898 | **FAIL** (indistinguishable or worse across all admissible branches) | Thm 1 (F#157 −2.99pp wrong direction), Thm 2 (F#41 interchangeable), Thm 3 (F#498 projection destroys composition) |
| K1899 | **INADMISSIBLE** (infrastructure-benchmark bucket, threshold uncalibrated; F#157 already reported 0.3% overhead well below 10ms) | F#715 bucket fires; F#666-pure threshold |

Verdict: **KILLED preempt-structural, quadruple-fire** — no runnable branch produces a non-redundant signal.

## 10. Autonomy log (per guardrail 1008)
1. "Similarity" is under-specified in the DB notes. I enumerate the four natural interpretations (hierarchical SVD-foundation, cosine-group k-means, routing-only, subspace-projection); each is covered by a prior finding. If a novel interpretation exists outside these four (e.g., learned-embedding clustering), it would still route through either F#41 (within-cluster interchangeable) or F#498 (projection destroys composition).
2. "Clustering overhead" (K1899) assumed to include both clustering assignment and per-cluster composition aggregation. F#157's 0.3% overhead at micro scale bounds the runtime.
3. N assumed ≥ 3 (clustering is vacuous at N≤2). Bound applies for all N≥3.
4. "Flat composition" baseline assumed to be uniform additive per F#543 / F#66. Any other baseline (PPL-probe F#137, null-space F#496) strengthens the KILL further.
