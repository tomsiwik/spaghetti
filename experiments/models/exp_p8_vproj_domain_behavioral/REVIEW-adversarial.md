# REVIEW — exp_p8_vproj_domain_behavioral

## Round 2 (audit-rerun, 2026-04-18) — **Verdict: KILLED (supersedes Round 1)**

This experiment was tagged `audit-2026-04-17-rerun` + `tautological-routing`
in the 2026-04-17 repo-wide audit. Re-review findings:

1. **K1315 is tautological (antipattern #6 — KC measures wrong object).**
   Pre-registered KC: "5-adapter **Grassmannian composition** retains ≥80% of
   solo behavioral quality". Code (`phase_composition_test`, lines 536–578):
   each adapter loaded independently via hot-swap, evaluated against its own
   domain queries at temperature 0.0. `comp_rate == solo_rate` by construction,
   retention=1.00 is a mechanical artifact. Round 1 of this review already
   flagged this ("K1315 composition test is trivially satisfied"), but
   Round 1 labelled it *non-blocking* and verdict remained PROCEED. Under
   PLAN.md §1 pre-flight item 6 (antipattern check), this is blocking.

2. **Re-classified KC tally**: K1312 FAIL, K1313 FAIL, K1314 PASS,
   K1315 FAIL on pre-reg KC → 1/4 pass → verdict **KILLED**.

3. **No re-execution.** The antipattern is structural (KC-vs-measurement
   mismatch), not a transient bug — re-running the same code produces the
   same tautological number. Following the pattern used for
   `exp_p7_null_space_adapter_quality` and `exp_p6_lingering_adapter_online`,
   we reconstruct `results.json` from the existing measurements in PAPER.md
   with `verdict=KILLED`, `all_pass=false`, and preserve behavioral findings
   in LEARNINGS.md.

4. **MATH.md is git-clean** since pre-reg commit 78538d2 — no KC swap.
   KC stands as pre-registered; the measurement is what diverges.

5. **Behavioral finding preserved** (not credited as supported KC closure):
   v_proj+o_proj adapters strictly dominate q_proj adapters on vocabulary
   improvement across 5 domains. Directional claim holds; absolute 60%
   thresholds were not met for math/code and K1315 did not test composition.

**V2 path.** `exp_p8_vproj_vs_qproj_v2` should (a) drop or reformulate K1315
to a true parameter-merge composition test (ΔW = Σ B_i A_i^T, single forward
per query, Grassmannian-orthogonal A via QR on random Gaussian, per-layer
cross-talk measurement max |cos(A_vi·x, A_vj·x)| ≤ 0.30 as KC), (b) pre-measure
base model vocabulary baseline per domain and pre-register per-domain
thresholds at base + Δ rather than a flat 60%, (c) train on >100 unique
examples per domain to avoid the 80-cycle ceiling effect acknowledged in
Round 1.

---

## Round 1 (2026-04-12) — historical, verdict SUPERSEDED

**Verdict: PROCEED** (superseded by Round 2 above)

## Summary

Core finding is sound: v_proj+o_proj adapters improve behavioral text quality vs q_proj
across all 5 domains. Data is consistent, per-query results verify aggregate rates,
and MATH.md reasoning about output-path vs query-path modification is mechanistically correct.
Status SUPPORTED is appropriate (2/4 kill criteria pass, directional finding validated).

## Issues

### 1. K1315 composition test is trivially satisfied (non-blocking)

K1315 reports 100% retention under "composition" — but this is sequential serving where
each adapter runs independently. By construction, solo and composition rates are identical
because no parameter merging occurs. Theorem 3 (MATH.md) predicts retention under actual
Grassmannian parameter composition (merged ΔW). The test validates serving infrastructure
(already confirmed by Finding #503) but does not test the theorem's prediction.

**Impact on finding:** Does not invalidate the v_proj+o_proj mechanism finding. But K1315
should be reframed in PAPER.md as "sequential serving confirmed" rather than "composition
retention validated." Actual composition testing is deferred to an experiment that merges
adapter weights.

### 2. Legal at 35% undercuts "all domains improve" narrative (non-blocking)

Domain improvement rates: medical 70%, math 55%, code 50%, finance 50%, legal 35%.
Only 1/5 domains exceeds 60%. Legal's 35% is notably weak — worse than finance and code
despite legal having rich domain-specific vocabulary. PAPER.md discusses math/code ceiling
effects but doesn't address legal's underperformance.

**Note:** Legal still improved vs q_proj (20% → 35%), so the directional claim holds.
But the magnitude is concerning and worth acknowledging.

### 3. "Ceiling effect" explanation is post-hoc (non-blocking)

MATH.md predictions for math (70-80%) and code (65-75%) substantially overestimate
measured values (55%, 50%). The post-hoc explanation (base model already competent,
limited training data) is plausible but unfalsifiable. A stronger experiment would
predict which domains face ceiling effects based on base model competence scores.

**Note:** The 80-example training set (8-10 unique, cycled) is genuinely small. This
is a reasonable limitation to acknowledge, not a finding flaw.

## Data Integrity Check

- Per-query counts match aggregate rates: math 11/20=55% ✓, code 10/20=50% ✓,
  medical 14/20=70% ✓, legal 7/20=35% ✓, finance 10/20=50% ✓
- results.json kill_criteria flags match computed rates ✓
- Training times consistent (2.1-2.8 min per domain, 12.2 min total) ✓
- q_proj baseline values match behavioral E2E killed experiment claims ✓

## Verdict Rationale

PROCEED. The finding that v_proj+o_proj is the correct projection target for behavioral
quality is well-supported by data across 5 domains and correctly explained by the
output-path mechanism. Kill criteria results are honestly reported. The non-blocking
issues are caveats for the analyst to capture in LEARNINGS.md, not blocking revisions.
