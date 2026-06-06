# PAPER.md — exp_g4_tfidf_ridge_n25_clean

## Verdict: KILLED_PREEMPTIVE (5-theorem, defense-in-depth)

K1624 (">=90% weighted accuracy with disjoint splits and hard negatives")
extends F#474 (N=5, 97.3% weighted accuracy on code/math/medical/finance/
legal) to N=25 on MMLU-Pro. The extension crosses three scope axes
un-covered by F#474: N-scale (10 → 300 cross-pairs), subcategory-
tautology (MMLU-Pro only has 14 top-level categories; N=25 requires
subcategory TF-IDF = label-embedded), and hard-negative protocol
(un-pre-registered, circular if mined by TF-IDF).

## Prediction vs Measurement

| Theorem | Predicted (blocks SUPPORTED?) | Measured | Verdict |
|---|---|---|---|
| T1 infrastructure | shortfall ≥ 4 artefacts | 0 local pipeline artefacts; 0 `*tfidf*` / `n25*` dirs in repo | BLOCK |
| T2 budget | protocol-missing blocks (compute fine) | compute negligible; hard-negative protocol un-registered | BLOCK |
| T3 framework | `Success Criteria: NONE` + ⚠ INCOMPLETE | `experiment get` confirms both LITERAL | BLOCK |
| T4 KC pins | ≤2/5 pins | 1/5 (only "pooled" hit via "weighted") | BLOCK |
| T5 scope caveat | ≥1 F#474 breach literal | 3/3 (A N-scale, B subcategory-tautology, C hard-negative circularity) | BLOCK |

`all_block = True` (5/5). Defense-in-depth: T1 ∨ T3 ∨ T5 alone blocks.

## §T1 — Infrastructure

K1624 requires a full pipeline: disjoint N=25 MMLU-Pro splits, hard-
negative mining, TF-IDF fit over 25-domain corpus, ridge classifier,
weighted-accuracy eval. `micro/models/exp_g4_tfidf_ridge_n25_clean/`
contains only MATH/PAPER/REVIEW/run_experiment/results. No routers,
no splits, no eval fixtures. `find micro/models -name '*tfidf*'` = 0
and `-name 'n25*'` = 0. Shortfall = full pipeline.

## §T3 — Framework incomplete (LITERAL)

```
Success Criteria: NONE — add with: experiment success-add exp_g4_tfidf_ridge_n25_clean ...
⚠ INCOMPLETE: success_criteria, references, experiment_dir, kill_results
```
ap-framework-incomplete blocks SUPPORTED regardless of measurement.

## §T4 — KC pin count (1/5)

K1624 = ">=90% weighted accuracy with disjoint splits and hard negatives":
- baseline ✗ (no baseline comparator)
- delta ✗ (threshold, no delta)
- pooled ✓ ("weighted")
- ε / methodology-ε ✗ (no p<, CI, ±, significance)
- enum ✗ (N=25 only in title, not KC; seeds absent)

1/5. The 90% threshold with un-specified hard-negative distribution
is the ap-017 (c) F#44 pattern (raw-threshold without methodology-ε).

## §T5 — Scope caveat literal (ap-017 tautological-routing branch)

### §T5.A — N-scale non-transfer

F#474 source N=5 (C(5,2)=10 cross-pairs) → K1624 target N=25
(C(25,2)=300 cross-pairs, **30× multiplier**). F#474 already underperformed
its own prediction at N=5 (accuracy 97.3% vs ≥98%; max_cosine 0.237 vs
<0.15). F#474's impossibility structure LITERAL: *"K1214 fails only if
math_vs_legal confusion rate >10%, requiring |V_math_legal_shared|/|V_math|
> 0.3"* — covers exactly 1 pair, says nothing about the 299 new pairs.
Under independent-errors assumption, expected misroute rate grows
super-linearly with pair count for non-orthogonal label vocabularies.

### §T5.B — Subcategory-tautology

MMLU-Pro has 14 top-level categories. N=25 requires subcategory
splitting (e.g. `math.algebra` vs `math.calculus`, `physics.mechanics`
vs `physics.thermodynamics`). TF-IDF over subcategory vocabulary is
label-embedded: the feature ≡ the label. F#474 succeeded at N=5 with
top-level disjoint-vocab categories (code/math/medical/finance/legal).
Subcategory vocab is not disjoint — it shares the parent-category vocab
by construction. Result: either high accuracy tautologically (label
leakage via subcategory name) or low accuracy from parent-vocab
collision. The `tautological-routing` tag applies directly.

### §T5.C — Hard-negative circularity

K1624 requires "hard negatives" without pre-registering mining protocol.
Hard-negative selection is itself a routing decision. If negatives are
mined by TF-IDF similarity, K1624 is self-referential (router defines
its own adversaries). If mined externally (e.g. by a different
embedding), the external method's choices define K1624's outcome, not
TF-IDF ridge generalization. Protocol absence ⇒ PASS/FAIL is a function
of the omitted specification, not of the router under test.

## Conclusion

K1624 is structurally un-runnable (T1), framework-incomplete (T3),
non-discriminating at ε-level (T4), and scope-non-transferable from
F#474 along three independent axes (T5A/B/C). Any re-run either
yields PASS tautologically (easy negatives + subcategory-label
leakage) or FAIL from cross-pair explosion — in both cases
uninformative. Cohort-drain preemptive-kill ratified.

## Assumptions

- Independent-errors decomposition for pair-confusion lower bound at
  N=25. Empirically may be lower (some pairs orthogonal) or higher
  (confusion clusters), but F#474's single-pair impossibility proof
  does not generalize either way.
- MMLU-Pro's 14 top-level category count is current (HF dataset
  TIGER-Lab/MMLU-Pro). N=25 subcategory requirement derives directly.
- "Weighted accuracy" assumed to mean per-domain-weighted per F#474
  convention; K1624 does not specify.

## References

- F#474 exp_p4_a0_5domain_routing (SUPPORTED, 97.3% weighted @ N=5)
- ap-017 tautological-routing + scope-non-transfer branches
- mem-antipattern-framework-incomplete (ap-framework-incomplete)
