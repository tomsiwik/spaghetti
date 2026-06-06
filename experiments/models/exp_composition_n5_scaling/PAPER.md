# exp_composition_n5_scaling — PAPER

## Verdict: KILLED (preempt-structural)

No empirical run. Verdict derives from MATH.md §1–§4: redundancy with F#406 / F#54 / F#367 (Grassmannian branch) or F#543 / F#510 / F#511 (uniform-additive branch), combined with K1892 being a canonical guardrail 1007 PPL proxy and K1893 lacking target-metric binding.

## Prediction vs measurement

| KC | Pre-run prediction | Measurement | Agreement |
|---|---|---|---|
| K1892 (N=5 PPL degradation > 5%) | method-dependent (fails under Grassmannian; passes under uniform) | not run (inconclusive) | N/A (preempt) |
| K1893 (per-adapter quality drop > 5pp) | method-dependent + target-metric binding missing | not run (inconclusive) | N/A (preempt) |

## Triple-fire / composition context

This experiment lands in the drain-window pattern taxonomy:
- **F#666-pure standalone (19th suspected).** K1892 is pure PPL proxy (canonical guardrail 1007). K1893 is target-ambiguous as filed (no dataset / no evaluator bound to "quality").
- **Redundant-with-prior-finding (new sub-pattern candidate).** Under either composition branch the KC outcome is already published. This is the first drain-window preempt-KILL whose *dominant* reason is method-dependent redundancy rather than parent-target unavailability (F#669) or method-unavailability (F#702). Watchlist: a second such preempt-KILL would promote a standalone memory.
- **F#669 parent-target-unverified: does not apply.** No parent dep. The redundancy is with already-supported/killed findings, not an un-run parent.

## Finding ledger references

- F#406 SUPPORTED — N=25 Domain Grassmannian composition at 4B, quality_ratio=1.3125.
- F#54 SUPPORTED — real-data N=24 adapters; scaling from N=5 framed honestly.
- F#367 SUPPORTED — activation-space interference α=0.39 sub-linear to N=10.
- F#543 KILLED — uniform static scaling at N=5 (Qwen 7B), 2.57× PPL bloat.
- F#510 SUPPORTED — pre-merged standard LoRA destroys benchmarks.
- F#511 SUPPORTED — orthogonal adapters structurally required.

## Antipattern audit

- Composition math bug: N/A (no code).
- LORA_SCALE=20: N/A.
- Tautological routing: N/A.
- shutil.copy: N/A.
- Hardcoded `"pass": True`: N/A.
- Eval truncation: N/A.
- Proxy model substitution: N/A.
- **F#666-pure canonical-1007 PPL guardrail: FIRES (K1892).**
- **Redundant-with-prior-finding: FIRES (6 findings cited).**

## Assumptions

- KC phrasing reads "KC passes = failure condition met" per repo convention.
- K1893 "pp" = percentage points implies task accuracy; but without dataset/evaluator binding, it remains under-specified at the experiment-definition level.

## Followups (not filed)

Per preempt-structural precedent, no `_impl` follow-up filed. If future work needs to distinguish a composition method not covered by F#406/F#54/F#543/F#510/F#511 (e.g., weighted-learned-coefficient composition — which `exp_composition_weighted_sum` already covers), that should be a fresh experiment, not a re-run of this one.

## F#702 hygiene checklist

- platform: local-apple ✅
- dir: `micro/models/exp_composition_n5_scaling/` ✅
- evidence: added via `experiment complete`
- success_criteria: attempted via `experiment complete` (flag may not be supported by CLI; documented here per precedent)
- references: will add via `experiment ref-add` where possible; otherwise listed in PAPER.md §"Finding ledger references"
