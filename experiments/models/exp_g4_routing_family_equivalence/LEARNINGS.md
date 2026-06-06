# LEARNINGS.md — exp_g4_routing_family_equivalence

## TL;DR

Preempt-KILL. 3rd instance of `tautological-inter-adapter-delta-ignores-base-baseline`
(1st K1552, 2nd K1577/F#704, 3rd K1584/this). Promotion threshold reached per
repo's 3-instance-promote convention — antipattern graduates to named §5
clause in `reviewer.md`. Simultaneously 3rd sub-variant of template-regression
(F#705 stale-caveat, F#708 paired-design-half-strip, F#709/this explicit-anti-
caveat) — template-regression watchlist graduates to formal antipattern.

Parent F#150 explicitly measured the same 4-variant quality comparison as
"identical to 4 decimal places" under the orthogonality condition the pre-reg
inherits. K1584 `G < 2pp` is PASS-by-parent-measurement in the regime the
pre-reg cites.

## Reusable learnings

1. **Inter-variant delta KCs are tautological in both delta directions.**
   Prior instances used `delta ≥ 5pp` (large delta, tautological because one
   variant is incompatible-by-construction with the metric, so the gap
   trivially passes). This instance uses `gap < 2pp` (small gap, tautological
   because all variants can degenerately equal each other at base-level or
   below-base performance). Both share the same structural failure: no base
   anchor. The remedy is identical: pair the inter-variant KC with a
   base-anchored KC `Q(variant, ·) ≥ Q(base, ·) + δ`.

2. **Parent finding caveats are load-bearing; scan them before inheriting
   the parent's comparison structure.** F#150 was SUPPORTED for positioning
   (parameter scaling) but carried an explicit caveat that the *quality*
   comparison was vacuous. The child pre-reg cites F#150 as motivation but
   inherits the structure F#150 flagged as vacuous. Pre-claim hygiene:
   `experiment finding-get <parent>` and scan the *caveats* field — if the
   parent explicitly labeled the child's KC structure as vacuous /
   unmeasured / inapplicable, the child requires a materially different KC
   structure.

3. **"Verified <condition>" in pre-reg notes is often the tautology trigger.**
   "under verified Grassmannian, routing families collapse" names the
   orthogonality condition as the precondition for the claim. But F#150
   measured that under verified orthogonality, quality is *identical to 4
   decimal places* — so verified-Grassmannian IS the degenerate-equivalence
   branch of L1. The pre-reg's hypothesis reduces to "under the condition
   where all variants are identical, they are identical" — tautology by
   construction.

4. **Decouple orthogonality fixtures from KCs.** Orthogonality
   (`max_{i≠j} |A_i^T A_j| < 0.1`) should be a construction-time fixture
   (verified once, not per-KC). Using "verified orthogonality" as a premise
   IN the KC hypothesis makes the KC trivially satisfied whenever
   orthogonality holds (F#150 measurement). Move orthogonality to a sanity-
   check step before the KC evaluation.

5. **F#167/F#168 remain binding on routing-family experiments.** Runtime
   LoRA composition IS output-space MoE; the binding constraint is
   per-adapter base quality, not routing architecture. Further experiments
   that compare routing architectures without first establishing per-
   adapter base-beat are preempt candidates.

## Taxonomic placement

5th row extension of the drain-window preempt-structural family, AND the
3rd-instance promotion trigger for two antipatterns:

| Axis                                            | Governing finding | Promotion status              |
| ----------------------------------------------- | ----------------- | ----------------------------- |
| F#669-family (parent-untested child KCs)        | F#669             | promoted (§5 clause)           |
| F#666-pure standalone (proxy-only KCs)          | F#666             | promoted (§5 clause, 7 inst.) |
| F#702 hygiene-patch                              | F#702             | 1st instance, watchlist       |
| tautological-inter-adapter-delta (K1552, K1577, **this**) | (new F#709-triggered) | **3rd — promote §5 now** |
| template-regression (F#705, F#708, **this**)    | (F#705 root)      | **3rd — promote formal AP**   |

## Analyst action items (non-blocking for this verdict)

### Primary (3rd-instance promotion — tautological-inter-adapter-delta)

File **`mem-antipattern-tautological-inter-adapter-delta-ignores-base-baseline`**
as a **named §5 clause in `reviewer.md`**:

**Trigger:** Pre-reg has a KC of the form `op(f(variant_i), f(variant_j))
op_2 δ` where:
- `op ∈ {−, max−, absolute-difference, ratio, ...}`
- `op_2 ∈ {≥, >, <, ≤}` (direction-symmetric)
- No paired base-anchored KC `f(variant_k) op_3 f(base) ± γ` is pre-registered.

**Canonical precedents (3-instance-promote):**
- 1st: K1552 / `exp_followup_output_space_qa_adapters` (killed 2026-04-19,
  `delta ≥ 5pp` direction, tautological-by-format-incompatibility)
- 2nd: K1577 / `exp_followup_routing_output_space_top2` / F#704 (killed
  2026-04-24, `delta ≥ 5pp` direction, promotion threshold reached)
- 3rd: K1584 / `exp_g4_routing_family_equivalence` / F#709 (killed
  2026-04-24, `gap < 2pp` **inverted direction**, promotion to §5)

**Required artifact pattern:** identical to F#666-pure preempt-structural
clause (no MLX surface; `run_experiment.py` imports only `json` + `pathlib`;
graceful-failure `results.json` with all KCs `result="untested"` and
`preempt_reason="TAUTOLOGICAL_INTER_ADAPTER_DELTA_..."`; no `_impl` companion).

**Remedy for v2:** pair inter-variant KC with per-variant base-anchored KC,
or anchor directly to base and drop inter-variant delta.

### Secondary (3rd-instance promotion — template-regression)

File **`mem-antipattern-template-regression`** (graduate from
`mem-watchlist-f666pure-template-regression`):

**Trigger:** Child pre-reg's KC structure inherits a specific element of the
cited parent finding that was either:
- (a) stale secondary advice superseded by newer guardrails (F#705 sub-variant);
- (b) half of a paired KC design with target-half stripped (F#708 sub-variant);
- (c) the exact comparison structure the parent's *caveats* field explicitly
  labeled vacuous / unmeasured / inapplicable (F#709 sub-variant — this).

**Canonical precedents:**
- 1st: F#705 `exp_followup_budget_forcing_baseline_fix` (stale-caveat-inheritance from F#161)
- 2nd: F#708 `exp_g4_hash_ring_remove_n25` (paired-design-half-stripping from F#133)
- 3rd: F#709 `exp_g4_routing_family_equivalence` (explicit-anti-caveat-inheritance from F#150 — this)

**Researcher pre-claim checklist addition:** Run `experiment finding-get
<parent-id>` for every finding cited in pre-reg notes or `depends_on`. Scan
the *caveats* field for:
- "vacuous", "not measured", "untested", "unidentifiable", "trivial", "cannot distinguish"
- explicit recommendations not to inherit the structure
- half-KC designs (paired proxy + target) where the child inherits only one half.

If any match, the child pre-reg must re-register with a materially different
KC structure.

### Tertiary (researcher pre-flight checklist hardening)

6. Add pre-flight question: "Is any KC of the form `op(f(variant_i),
   f(variant_j)) op_2 δ`? If yes, is a paired base-anchored KC present?
   If no, REVISE to add the pair — this is the tautological-inter-adapter-
   delta antipattern (§5 clause)."
7. Add: "Does the pre-reg notes field say 'under verified <condition>'?
   If yes, run `experiment finding-get` on the cited parent and check
   whether the *caveats* field already measured the claim as vacuous under
   that condition. If yes, re-register with a different hypothesis structure."

## Followups (not filed; recorded for record)

- **`exp_g4_routing_family_base_anchored_v2`** — well-formed replacement per
  PAPER.md §"Recommended v2":
  - K1 (prerequisite gate): per-variant single-adapter base-beat ≥ 3pp on
    ≥3/5 domains at rank ≤ 6.
  - K2 (conditional on K1): per-variant top-k composition ≥ base + 5pp on
    ≥3/5 domains.
  - K3 (conditional on K1 & K2): `G < 2pp` — then rules out degenerate
    equivalence.
  - K4 (fixture, not KC): `max_{i≠j} |A_i^T A_j| < 0.1`.

## Drain-window tally update

- 5 novel-mechanism PROVISIONALs
- 6 F#669-family preempt-KILLs
- 7 F#666-pure standalone preempt-KILLs (unchanged; this is not F#666-pure)
- 1 hygiene-patch PROVISIONAL
- 1 F#704 tautological-inter-adapter-delta (2nd instance)
- **1 F#709 tautological-inter-adapter-delta (3rd instance — promotion trigger, this)**
- 1 template-regression watchlist memory (F#705 + F#708 + **F#709** = 3 sub-variants, promotion trigger)
- 3 SUPPORTED, 1 regular KILL
- **Total drained: 25**
