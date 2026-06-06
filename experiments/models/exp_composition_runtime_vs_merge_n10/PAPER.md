# exp_composition_runtime_vs_merge_n10 — PAPER

## Verdict: KILLED (preempt-structural, method-dependent redundancy 2nd instance — PROMOTION)

No empirical run. Verdict derives from MATH.md §1–§4:
- **Thm 1**: K1894 (runtime > 2× merge latency) is structurally FALSE by F#399's closed-form BW-bound theorem. At N=10 and rank=8, effective rank-units = 80 ≪ d_model, so merge speedup ceiling < 1.1× — nowhere near the 2.0× required for K1894 to pass. Runtime is BW-bound optimal; merge cannot beat it by 2× at practical ranks.
- **Thm 2**: K1895 (merge quality > 5pp worse than runtime) is determinable per branch across every plausible composition × precision fill-in (F#66 / F#510 / F#511 / F#543 / F#406 / F#54). The experiment does not specify a branch; under any fill-in the answer is already published. **2nd drain-window instance of method-dependent redundancy** (F#731 was 1st) → promotion trigger.
- **Thm 3**: K1895 has no dataset / evaluator / task binding → F#666-pure canonical guardrail 1007 (20th reuse). K1894 is pure latency ratio → infrastructure-benchmark bucket per F#715 (2nd instance).

## Prediction vs measurement

| KC | Pre-run prediction | Measurement | Agreement |
|---|---|---|---|
| K1894 (runtime > 2× merge latency) | FAIL (runtime ≤ 1.1× slower per F#399 BW-bound) | not run (inconclusive) | N/A (preempt) |
| K1895 (merge quality > 5pp worse) | branch-redundant; every plausible branch covered | not run (inconclusive) | N/A (preempt) |

## Composition / drain-window taxonomy

This experiment lands on the following patterns:
- **Method-dependent redundancy — 2nd instance (PROMOTION).** F#731 was 1st. Per the scratchpad watchlist, a 2nd instance promotes a standalone memory. Filed here. The pattern:
  - Distinct from F#669 (parent-target-unverified): no un-run parent dependency.
  - Distinct from F#702 (method-unavailable): the method *is* runnable in principle.
  - Distinct from F#666-pure standalone: K1895 *does* co-fire F#666-pure but the redundancy stands independently — even with a bound target metric, Thm 2 would still preempt-KILL per branch enumeration.
- **F#666-pure canonical guardrail 1007 — 20th reuse.** K1895's unbound "quality" (|Target|=0).
- **Infrastructure-benchmark bucket (F#715) — 2nd instance.** K1894 latency-ratio without behavioral binding.
- **F#399-derivable structural impossibility (K1894).** Novel sub-pattern? F#399 is a closed-form theorem; this is the first drain-window preempt-KILL to *derive FAIL from an inequality in an existing theorem*. 1st instance — watchlist.

## Finding ledger references

- **F#399 SUPPORTED** — Pre-merge vs runtime LoRA: runtime optimal for practical ranks; 1.5× pre-merge speedup requires rank ≥ 380 (structural impossibility). K1894 directly FAIL-derivable.
- **F#66 SUPPORTED** — Float Merge: fp32 KILLED on K3; bf16 merge loses ~50% of adapter delta (ULP limit) but 39% faster tok/s; dual-mode serving SUPPORTED.
- **F#406 SUPPORTED** — N=25 Domain Grassmannian composition at 4B, quality_ratio=1.3125 (subsumes N ≤ 25 runtime).
- **F#54 SUPPORTED** — Real-data N=24 composition under Grassmannian.
- **F#510 SUPPORTED** — Pre-merged standard LoRA destroys benchmarks (GSM8K 0 vs 73; HumanEval 0 vs 63).
- **F#511 SUPPORTED** — Orthogonal adapters structurally required; naive overlap destructive.
- **F#543 KILLED** — Uniform additive N=5 on Qwen 7B: 2.57× PPL bloat.
- **F#715 KILLED** — Infrastructure-benchmark bucket (serialization-format latency ratios without behavioral binding).
- **F#731 KILLED** — 1st drain-window instance of method-dependent redundancy (exp_composition_n5_scaling).

## Antipattern audit

- Composition math bug: N/A (no code).
- LORA_SCALE=20: N/A.
- Tautological routing: N/A.
- shutil.copy: N/A.
- Hardcoded `"pass": True`: N/A.
- Eval truncation: N/A.
- Proxy model substitution: N/A.
- **F#666-pure canonical-1007 guardrail: FIRES** (K1895 unbound "quality"; K1894 infrastructure).
- **Method-dependent redundancy: FIRES (2nd instance — PROMOTION).**
- **F#399 BW-bound structural: FIRES** (K1894 derivable FAIL).
- **Infrastructure-benchmark bucket: FIRES (2nd instance after F#715).**

## Assumptions

- "Practical rank" interpreted as rank ≤ 64 (repo convention per PLAN.md Part 2 Pierre architecture).
- K1895 phrasing "quality > 5pp worse" reads as percentage-point degradation on *some* task-accuracy metric; target-metric binding is definitionally missing (no dataset / evaluator specified in the KC text or notes).
- "N=10" interpreted as 10 concurrently-applied adapters in runtime composition vs 10 pre-merged into base weights.

## Followups (not filed)

Per preempt-structural precedent, no `_impl` follow-up filed. The serving tradeoff *in production* is already captured by the dual-mode serving architecture documented in F#66 (bf16 merge = 16.7 tok/s; runtime = 12.0 tok/s on the same model), which this experiment would have re-measured on a different base.

Analyst guidance: remaining composition siblings (`exp_composition_weighted_sum`, `exp_composition_clustering_group`, `exp_composition_residual_analysis`, `exp_composition_ordering_matters`) should be triaged next. Some will NOT fire method-dependent redundancy (weighted-learned coefficients test a mechanism not yet covered); others may.

## F#702 hygiene checklist

- platform: `local-apple` ✅ (set via `experiment update`)
- dir: `micro/models/exp_composition_runtime_vs_merge_n10/` ✅ (set via `experiment update`)
- evidence: populated via `experiment complete`
- success_criteria: CLI flag not supported; documented here per precedent (non-blocking)
- references: cited inline in MATH.md §8 and PAPER.md "Finding ledger references"
