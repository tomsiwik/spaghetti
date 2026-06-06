# LEARNINGS — exp_composition_runtime_vs_merge_n10

## Core Finding

PREEMPT-KILL. K1894 (runtime > 2× merge latency) is structurally FALSE by F#399's closed-form BW-bound: at N=10 × r=8, effective rank-units ≪ d_model, capping merge speedup < 1.1× — never the 2× required. K1895 (merge quality > 5pp worse) is method-dependent redundant: bf16 merge (F#66 ~50% delta loss), fp32 merge (F#66 K3 KILL), standard-LoRA merge (F#510/F#511 destroys), uniform additive (F#543 2.57× bloat), Grassmannian runtime reference (F#406 N=25, F#54 N=24) — every plausible branch covered. Both KCs additionally F#666-pure (K1895 unbound "quality" |Target|=0; K1894 infrastructure per F#715 bucket). Filed KILLED (preempt-structural).

## Why

Three independent theorems in MATH.md. (1) F#399 is a published closed-form BW inequality; plugging N=10, r=8, d_model=3584 yields merge speedup ceiling 1.0001×, so K1894's 2× threshold is arithmetically impossible for practical ranks — first drain-window preempt-KILL derived by *inequality arithmetic* from an existing theorem. (2) Every composition × precision cell at N=10 is covered by a prior supported/killed finding; the experiment-as-filed does not pick a branch, so under any fill-in the answer is pre-published. (3) K1895's "quality" has no dataset / evaluator binding (|Target|=0); K1894 is infrastructure-benchmark per F#715 — corroboration, not the primary KILL justification.

## Implications

1. **Method-dependent redundancy PROMOTED (F#731 1st → this 2nd instance).** The scratchpad watchlist explicitly names `exp_composition_runtime_vs_merge_n10` as the candidate 2nd instance; promotion fires. Standalone pattern memory warranted. Distinct from F#669 (parent-unverified), F#702 (method-unavailable), F#666-pure (KCs all-proxy): here KCs are well-formed in principle, but the method × precision space collapses to branches each independently covered.
2. **NEW sub-pattern watchlist: F#399-derivable structural impossibility.** 1st drain-window preempt-KILL to plug numbers into an existing theorem's inequality. Distinct from redundancy (which cites multiple different findings per branch). A 2nd instance would promote a standalone memory: "preempt-KILL derivable by arithmetic from a supported theorem's closed-form bound".
3. **Infrastructure-benchmark bucket — 2nd instance.** K1894 is latency ratio infrastructure, same bucket as F#715 (memento_kv_serialization_format). Bucket at 2/3 toward standalone promotion.
4. **F#666-pure 20th reuse.** K1895 unbound "quality" is the 20th canonical guardrail 1007 instance. Continues to dominate drain.
5. **Remaining composition sibling triage.** F#731's watchlist named 5 siblings; this closes the 1st. Remaining: `exp_composition_weighted_sum` (novel mechanism — probably NOT redundant: weighted-learned coefficients are not covered by existing findings), `exp_composition_clustering_group` (likely novel), `exp_composition_residual_analysis` (measurement — unclear), `exp_composition_ordering_matters` (likely redundant under commutativity of additive composition). Analyst should prioritize the next sweep accordingly.
6. **Dual-mode serving already exists in production path.** F#66 SUPPORTED bf16 merge + runtime LoRA dual-mode. This experiment would have re-measured a tradeoff already captured architecturally. Triage-hygiene reminder: after F#66 / F#399 landed, composition-serving experiments should have been re-triaged; this one survived into the drain window because the triage never happened.
7. **No parent `_impl` follow-up.** Preempt-structural KILL per scratchpad precedent. No unblock leverage.
