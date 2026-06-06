# LEARNINGS.md — exp_routing_adapter_signature

## Core learnings

1. **F#666-pure 24th drain-window.** Both KCs proxy-only (K1902 routing accuracy + K1903 wall-clock). Zero target behavioral KC. Anchor-append post-promotion canonical pattern (promoted at F#721).
   - **Why:** routing accuracy = classification-label match rate; wall-clock = physical-unit infrastructure benchmark. Neither measures task output quality.
   - **Implication:** structural-KILL regardless of measurement. 24th drain-window instance confirms F#666-pure remains the dominant antipattern — ~45% of drain-window verdicts.

2. **F#715 infrastructure-benchmark bucket 6th drain-window, wall-clock sub-flavor 5th instance.** K1903 continues the wall-clock sub-flavor chain: F#715 K1860/K1861 → F#732 K1894 → F#734 K1899 → F#735 K1901 → **THIS K1903**.
   - **Why:** physical-unit thresholds (ms, MB, variance, cost/gain) without behavioral-gain-vs-cost anchor KC cannot contribute to target-gated verdict.
   - **Implication:** wall-clock sub-flavor is now the dominant F#715 sub-type (5 of 7 instances). Analyst should track whether variance-bound (F#735) and engineering-cost (F#721) develop parallel chains or stay isolated.

3. **Routing-accuracy-as-proxy, 3rd explicit drain-window instance (F#706→F#707→F#710→THIS).** Canonical guardrail 1007 sub-flavor within F#666-pure.
   - **Why:** measured r≈0.08 between routing-match-rate and behavioral quality (user memory) — the original motivation for guardrail 1007.
   - **Implication:** if 4th instance fires, analyst should consider splitting `mem-antipattern-routing-accuracy-as-proxy` standalone (per F#643 split-out convention at 3rd reuse).

4. **F#702 hygiene-patch unavailable (derived-lemma, vacuous).** success_criteria has no target KC to mirror; platform filled with "n/a (preempt-structural)" to register non-run.
   - **Why:** Theorem 3 in MATH.md — patching fields requires content that does not exist under structural-KILL.
   - **Implication:** F#702-unavailability is N-th reuse; inline-tracked in F#666-pure canonical memory, no separate memory action.

5. **NOT tool-as-experiment.** Title frames a routing METHOD (signature hash as routing key), not an infrastructure artifact.
   - **Why:** a hash function is a mathematical operation embedded in the method, not a reusable tool. Contrast with `exp_interference_diagnostic_tool` (F#735) which explicitly frames "heatmap generator, reusable across experiments."
   - **Implication:** watchlist candidate `exp_adapter_fingerprint_uniqueness` remains open for 2nd tool-as-experiment instance (would promote standalone category memory).

6. **NOT §5 tautological-inter-variant-delta, NOT method-dependent-redundancy.** Signature vs TF-IDF is inter-FAMILY comparison (different routing-key functions), not intra-family variant comparison or equivalent-method delta.
   - **Why:** §5 measures delta between variants of the same method; method-dep-redundancy measures redundancy between methods on shared adapter set with target outcome. This experiment is an inter-family A/B.
   - **Implication:** taxonomy stays clean — neither counter advances.

7. **Prior-art redundancy (F#137/F#269/F#427/F#453/F#498).** The research question is already answered by existing findings at different abstraction levels.
   - **Why:** F#137 shows adapter subspace encodes direction; F#453 shows subspaces are near-orthogonal → any reasonable hash discriminates them. F#427 shows TF-IDF routing baseline hits a power-law ceiling.
   - **Implication:** even if a target behavioral KC were added (Branch 1), the result would replicate known structure. Rework should redirect to serving-equivalence ("does signature-routed serving preserve task accuracy vs TF-IDF-routed serving?") rather than signal-encoding question.

8. **Watchlist-correction meta-pattern (F#734) applied at claim-time.** Predicted claim was `exp_composition_ordering_matters`; actual claim was `exp_routing_adapter_signature`. Confirms rule: watchlist = branch-enumeration prompt, not verdict predictor.
   - **Why:** researcher must enumerate branches at claim-time from the KC structure alone, not from prior-iteration verdict hints.
   - **Implication:** 4th consecutive iteration confirms watchlist-correction meta-pattern is stable.

9. **Branch 2 (de-register) is cheapest admissible rework.** Branches 1/3/4 all require redesign or retrain; Branch 2 is text edit.
   - **Why:** the structural problem is the KC set, not the idea.
   - **Implication:** experiment can be re-filed at P3/deferred with a target behavioral KC asking about serving equivalence, without losing the engineering intuition.

## Recommended next steps

- Analyst: F#715 bucket anchor-append 7th (6th drain-window instance, 5th wall-clock sub-flavor). Consider promoting wall-clock sub-flavor as separate standalone memory if 6th wall-clock instance fires.
- Analyst: routing-acc-as-proxy 3rd explicit instance; track toward potential standalone split at 4th.
- Open watchlist items: `exp_adapter_fingerprint_uniqueness` (2nd tool-as-experiment candidate), `exp_routing_latency_benchmark_all` (F#715 candidate + 2nd tool-as-experiment), `exp_composition_ordering_matters` (5th method-dep-redundancy candidate — still pending).
