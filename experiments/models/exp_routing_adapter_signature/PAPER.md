# PAPER.md — exp_routing_adapter_signature

**Verdict:** KILLED (preempt-structural).

## Prediction vs measurement

| KC | Prediction | Measurement | Status |
|----|------------|-------------|--------|
| K1902 (routing-acc < TF-IDF baseline) | n/a — structural preempt | no run | fail (structural) |
| K1903 (sig compute > 1ms) | n/a — structural preempt | no run | fail (structural) |

all_pass = false. is_smoke = false.

## Verdict justification

Experiment structurally disqualified under Finding #666 / guardrail 1007 (target-gated kill): both registered KCs (routing-accuracy and wall-clock compute time) are proxy-only metrics; zero target behavioral KC. Under the target-gated kill rule, the admissible verdict set is {structural-KILL}; running the experiment cannot produce a SUPPORTED or meaningful KILLED outcome because any measurement would be a claim about the proxy, not the phenomenon.

## Triple-fire classification

1. **F#666-pure 24th drain-window instance.** Routing-accuracy sub-flavor confirmed-recurrent (F#706→F#707→F#710 lineage).
2. **F#715 infrastructure-benchmark bucket 6th drain-window instance, wall-clock sub-flavor.** Post-promotion anchor-append. 5th wall-clock instance counting F#715/F#732/F#734/F#735/THIS.
3. **F#706/F#707/F#710-lineage routing-accuracy-as-proxy, 3rd explicit drain-window instance.** Analyst signal for potential split out of `mem-antipattern-f666-pure-standalone` into dedicated `mem-antipattern-routing-accuracy-as-proxy` if 4th instance fires.

Non-promoting: F#702 hygiene-patch unavailable (derived-lemma, vacuous under structural saturation); prior-art redundancy (F#137 / F#269 / F#427 / F#453 / F#498).

## Recommended rework path

Cheapest admissible branch (per MATH.md Theorem 4): **Branch 2 — de-register K1902/K1903, re-file with target behavioral KC** such as "signature-routed N=25 adapter serving produces task-accuracy within ε of TF-IDF-routed serving on domain-matched prompts." This converts the question from "does the proxy move?" to "does the serving behavior preserve target accuracy?" — which is the actual engineering claim.

## References

- Finding #666 (target-gated kill rule / F#666-pure standalone pattern; promoted canonical memory anchored at F#721).
- Finding #706 (1st FNR / classification-accuracy-as-proxy canonical guardrail 1007).
- Finding #707 (canonical routing-match-rate dual).
- Finding #710 (2nd routing-accuracy confirmed-recurrent sub-flavor).
- Finding #715 (infrastructure-benchmark bucket, promoted at F#734 QUADRUPLE-FIRE).
- Finding #702 (hygiene-patch derived-lemma).
- Finding #137 (direction-probe r≈0.990 — adapter subspace encodes direction).
- Finding #269 (direction-interference between adapters).
- Finding #427 (TF-IDF routing power-law α=0.145).
- Finding #453 (max cos=0.0861 adapter pair separation).
- Finding #498 (subspace destroys composition).
- Finding #734 (watchlist-correction meta-pattern — branch enumeration at claim-time).
- Finding #735 (F#715 variance-bound sub-flavor, tool-as-experiment 1st instance inline).
