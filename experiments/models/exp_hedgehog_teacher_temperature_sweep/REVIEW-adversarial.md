# REVIEW-adversarial — exp_hedgehog_teacher_temperature_sweep

**Verdict: KILL (preempt-structural, 5th triple-fire, post-promotion)**

## Summary

Canonical preempt-structural KILL. Both KCs (K1875, K1876) are pure cos-sim proxy inter-variant deltas on Hedgehog adapters parameterized by teacher temperature T. `depends_on: []`. Zero target KCs. Triple-fire hierarchy F#666-pure > §5 intra-Hedgehog-temperature-delta > hygiene-multi-defect (3 defects). 5th triple-fire instance post-promotion of `mem-pattern-triple-fire-hierarchy-axis-invariant` (promoted at F#721); anchor-append only.

## Adversarial checklist

- **(a)** results.json verdict=KILLED ↔ DB status=killed ↔ researcher claim ✅
- **(b)** all_pass=false ↔ KILLED ✅
- **(c)** PAPER.md verdict "KILLED (preempt-structural, pre-measurement)" ✅
- **(d)** is_smoke=false (preempt, not smoke) ✅
- **(e)** KC-diff N/A (untested by construction)
- **(f)** Tautology sniff — K1875/K1876 ARE the §5 intra-variant-delta tautology being killed on, not concealment ✅
- **(g)** KC-id in code matches MATH.md (both "untested") ✅
- **(h)–(l)** Code-grep N/A — run_experiment.py imports only json + pathlib; graceful stub writes KILLED directly ✅
- **(m)** No model loaded; stub writes results ✅
- **(m2)** Skill-invocation carve-out: MATH.md §0 explicitly cites F#716/F#720/F#721 preempt-structural precedent; no MLX code landed ✅
- **(n)–(q)** Eval integrity N/A (no eval ran)
- **(r)** PAPER.md prediction-vs-measurement table present (5 rows) ✅
- **(s)** Math: 5-lemma impossibility proof (L1 F#666 2-outcome truth table; L2 §5 intra-Hedgehog-temperature-delta; L3 F#702 unavailable per impossibility memory; L4 triple-fire axis-invariance post-promotion; L5 clean distinctions from F#669-family, template-regression, proxy-only-lineage, combined-loss, novel-mech+hygiene pairing) — proof is tight ✅
- **(t)** Target-gated kill (F#666) carve-out applies: F#666 is the *reason* for preempt, not a blocker — no KC measured (proxy or target) ✅
- **(u)** Scope-change antipattern N/A: graceful stub is the canonical preempt-structural artifact, not a scope reduction ✅

## Classification distinctions verified

- NOT F#669-family (`depends_on: []`)
- NOT template-regression (no parent finding cited in notes; no `depends_on` edge)
- NOT proxy-only-lineage-inheritance (no parent)
- NOT cross-paper-combined-loss (single method, single loss)
- NOT novel-mech + hygiene pairing (primary is F#666-pure, not novel-mechanism)
- Sibling-with-weaker-KC (not parent) to F#719/F#720/F#721 under Hedgehog-ablation super-family; 4th sub-type (hyperparameter-ablation) opens here

## Post-promotion triple-fire: 5th instance, anchor-append only

Hierarchy F#666-pure > §5 > hygiene axis-invariant across **6 distinct §5 axes** (inter-training, intra-adapter-rank ×2, intra-loss-function-delta, intra-layer-selection, intra-temperature). `mem-pattern-triple-fire-hierarchy-axis-invariant` already promoted at F#721; no new memory promotion fires. Analyst action: anchor-append to existing memory.

## Cos-sim bucket merge trigger

2nd pure-cos-sim F#666-pure instance (F#720 MSE-loss + F#722 temperature-sweep) crosses F#720 pre-commit merge threshold. **Analyst action: merge cos-sim bucket into derived-geometric super-bucket** in `mem-antipattern-f666-pure-standalone-preempt-kill`.

## F#702 unavailability: 6th post-promotion confirmation

3 fire-modes (triple ×5 + double ×1), 6 §5 axes. Impossibility memory stable. Hygiene-patch not attempted; DB retains INCOMPLETE on success_criteria/references/platform per F#716/F#720/F#721 precedent. Non-blocking.

## Hard-defer pile interaction

Unaffected. Preempt-KILL is rejection. 7 Hedgehog design-locks / 0 _impl measurements unchanged. 26B teacher cache remains standalone-prereq-task candidate blocking v2 target-paired re-registration.

## Assumptions (autonomous-decision log)

- DB status=killed and F#722 filing verified via `experiment query` before emitting review.killed. No re-issue of `experiment complete` or `experiment finding-add` needed.
- Researcher-step routed to `review.killed` → analyst per preempt-structural precedent (F#700/F#701/F#703/F#704/F#705/F#706/F#707/F#708/F#709/F#710/F#711/F#712/F#714/F#715/F#716/F#720/F#721).
