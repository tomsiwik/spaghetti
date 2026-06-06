# LEARNINGS — exp_composition_n5_scaling

## Core Finding

PREEMPT-KILL. N=5 LoRA-composition scaling interference already has published answers in both composition branches: Grassmannian-routing (F#406 N=25 SUPPORTED, F#54 N=24 SUPPORTED, F#367 α=0.39 sub-linear) and uniform-additive (F#543 KILLED at N=5 on Qwen 7B, F#510/F#511 standard-LoRA pre-merge destroys benchmarks). K1892 is additionally F#666-pure canonical guardrail 1007 (PPL proxy; r≈0.08 ↔ behaviour). K1893 target-metric binding missing (dataset/evaluator unspecified). Filed KILLED (preempt-structural).

## Why

Three independent theorems in MATH.md. (1) K1892 violates guardrail 1007; K1893 as filed does not carry a target-metric binding, so the pair degenerates toward F#666-pure. (2) Under Grassmannian-routing, monotonicity of sub-linear interference (F#367) means any bound passing at N=25 (F#406) also passes at N=5 — KC outcome derivable without running. (3) Under uniform-additive, F#543 already settled N=5 as a 2.57× PPL bloat; F#510/F#511 show standard-LoRA pre-merge is destructive regardless of N. Either branch → no new finding.

## Implications

1. **New preempt-KILL sub-pattern candidate: `method-dependent redundancy`.** Distinct from F#669 (parent-target-unverified) and F#702 (method unavailable): here the method exists but the KC outcome under *any* plausible method is already published. Watchlist: a 2nd such preempt-KILL would promote a standalone memory.
2. **F#666-pure 19th reuse (suspected).** K1892 is PPL proxy; K1893 under-bound. Confirms canonical guardrail 1007 remains high-traffic.
3. **Composition-scaling experiment backlog needs triage.** Several remaining P=2 open composition experiments (`exp_composition_runtime_vs_merge_n10`, `exp_composition_weighted_sum`, `exp_composition_clustering_group`, `exp_composition_residual_analysis`, `exp_composition_ordering_matters`) should each be checked against F#406 / F#54 / F#367 / F#510 / F#511 / F#543 before running. Some may share the same redundancy and warrant preempt-KILL; others (weighted learned coefficients, clustering-group composition) test orthogonal mechanisms not yet covered.
4. **Finding taxonomy reminder.** `exp_composition_n5_scaling` was filed pre-F#406/F#54. After a key finding lands, prior experiments targeting the same question should be re-triaged; this one survived into the drain window because the triage never happened. Consider: every time a composition-related finding is SUPPORTED/KILLED, sweep open composition experiments for subsumption.
5. **No parent `_impl` follow-up.** Preempt-structural KILL per scratchpad precedent. No unblock leverage.
