# PAPER.md — exp_g4_tfidf_routing_no_alias

## Verdict: KILLED_PREEMPTIVE (5-theorem, defense-in-depth)

K1625 (">=88% weighted acc (no aliasing)") raises F#502's N=25 TF-IDF
routing bar from 84.2% to ≥88% (+3.8pp) by removing exactly one aliased
pair (medical↔clinical_knowledge). F#502's design constraint is
*schema-complete* ("domain labels must map to genuinely different data"),
not single-pair. The extension crosses three scope axes un-covered by
F#502: vacuous-bound inheritance (no new bound for the raised threshold),
proof-sketch hard-negative theorem (un-reproved), and single-alias vs
schema (un-audited residual aliases).

## Prediction vs Measurement

| Theorem | Predicted (blocks SUPPORTED?) | Measured | Verdict |
|---|---|---|---|
| T1 infrastructure | shortfall ≥ 4 artefacts | 0 local pipeline artefacts; 0 `*no_alias*` pipeline dirs; tfidf dirs exist but none with alias-audited N=25 splits | BLOCK |
| T2 budget | protocol-missing blocks (compute fine) | compute 0.006 min (negligible); alias-audit + hard-negative protocols un-registered | BLOCK |
| T3 framework | `Success Criteria: NONE` + ⚠ INCOMPLETE | `experiment get` confirms both LITERAL | BLOCK |
| T4 KC pins | ≤2/5 pins | 1/5 (only "pooled" hit via "weighted") | BLOCK |
| T5 scope caveat | ≥1 F#502 breach literal | 3/3 (A vacuous-bound, B proof-sketch, C single-alias-vs-schema) | BLOCK |

`all_block = True` (5/5). Defense-in-depth: T1 ∨ T3 ∨ T5 alone blocks.

## §T1 — Infrastructure

K1625 requires a full pipeline: alias-audited MMLU-Pro N=25 splits,
hard-negative mining, TF-IDF fit, ridge (or equivalent) classifier,
weighted-accuracy eval. `micro/models/exp_g4_tfidf_routing_no_alias/`
contains only MATH/PAPER/REVIEW/run_experiment/results. Repo glob
`*no_alias*` matches only this preempt dir itself. `*tfidf*` returns
source pipelines (exp_p1_t4_tfidf_routing_v2, exp_g4_tfidf_ridge_n25_clean,
m2p_tfidf_routing_n5, tfidf_routing_real_text, exp_p1_t4_tfidf_routing_gemma4)
— none carry alias-audited N=25 MMLU-Pro splits. Shortfall = full
pipeline + schema audit.

## §T3 — Framework incomplete (LITERAL)

```
Success Criteria: NONE — add with: experiment success-add exp_g4_tfidf_routing_no_alias ...
⚠ INCOMPLETE: success_criteria, references, experiment_dir, kill_results
```
ap-framework-incomplete blocks SUPPORTED regardless of measurement.

## §T4 — KC pin count (1/5)

K1625 = ">=88% weighted acc (no aliasing)":
- baseline ✗ (no baseline comparator; F#502 84.2% only implicit)
- delta ✗ (threshold 88%, no delta-to-baseline)
- pooled ✓ ("weighted acc")
- ε / methodology-ε ✗ (no p<, CI, ±, significance, seed spread)
- enum ✗ (N=25 only in title, not KC; seeds absent; which alias pair
  removed not enumerated in KC)

1/5. Matches ap-017 (c) F#44 raw-threshold-without-methodology-ε pattern.

## §T5 — Scope caveat literal (F#502 source)

### §T5.A — Vacuous-bound inheritance

F#502 caveat LITERAL: *"Ridge classification bound vacuous at both K=5
and K=25; result is empirical, not tight-bound derived"*. K1625 raises
the empirical threshold from 84.2% → 88.0% (+3.8pp) with no new bound
derivation. The source caveat transfers directly: there is no theoretical
guarantee of the +3.8pp gain, only conjecture that one-alias removal is
sufficient. ap-017 (a) post-hoc-threshold-without-theory.

### §T5.B — Proof-sketch hard-negative theorem

F#502 caveat LITERAL: *"Theorem 2 (hard-neg stress test) is proof sketch
only"*. K1625 inherits F#502's hard-negative protocol without
re-proving or pre-registering a full theorem. Hard-negative mining is
the free parameter that determines PASS/FAIL; its absence from K1625's
spec means the outcome is under-determined.

### §T5.C — Single-alias fix vs schema constraint

F#502 failure mode LITERAL: *"Dataset aliasing creates irresolvable
confusion — Bayes-optimal is 50% for equal priors. **Design constraint:
domain labels must map to genuinely different data.**"* The constraint
is schema-complete (quantifier: ∀ labels). K1625 removes exactly ONE
pair: medical↔clinical_knowledge. MMLU-Pro at N=25 contains multiple
candidate residual aliases (biology↔health, chemistry↔physics,
business↔economics, engineering↔math.calculus, psychology↔health).
No schema audit is pre-registered. If even one residual alias survives,
its pair is Bayes-capped near 50%, which under uniform weighting
contributes a 1/N = 4% accuracy loss per capped pair. K1625's +3.8pp
budget (84.2 → 88) is consumed by a single residual alias; ≥2 residual
aliases make the threshold unachievable by construction without changing
the schema further (not sanctioned by K1625).

## Conclusion

K1625 is structurally un-runnable (T1), framework-incomplete (T3),
non-discriminating at ε-level (T4), and scope-non-transferable from
F#502 along three independent axes (T5A vacuous-bound, T5B proof-sketch,
T5C single-alias-vs-schema). Any re-run either (a) tautologically passes
by over-pruning the schema (post-hoc label engineering) or (b) fails by
reproducing F#502's 84.2% empirical floor if residual aliases survive.
Both outcomes uninformative. Cohort-drain preemptive-kill ratified.

## Assumptions

- MMLU-Pro's 14 top-level category structure and candidate alias list
  (biology↔health etc.) are current per TIGER-Lab/MMLU-Pro HF dataset;
  specific alias identities may shift with dataset revision but the
  schema-completeness argument is quantifier-level, not instance-level.
- "Weighted accuracy" assumed to follow F#502's per-domain weighting
  convention; K1625 does not specify.
- Single-alias 4% contribution is under uniform N=25 weighting; domain-
  weighted schemes may shift absolute loss but not the schema-
  completeness argument.

## References

- F#502 exp_p1_t4_tfidf_routing_v2 (SUPPORTED, N=25 84.2% weighted acc
  with disjoint splits + hard negatives; medical/clinical 78.8%
  confusion documented as dataset alias)
- ap-017 tautological-routing branch (cohort scope)
- ap-framework-incomplete (mem-antipattern-framework-incomplete)
