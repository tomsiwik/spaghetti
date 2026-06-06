# LEARNINGS.md — exp_user_adapter_from_memento_distillation

## Outcome
KILLED (preempt-structural, F#669 5th+ reuse, **dual-parent disjunctive** sub-case). Third preempt-drain application this window (F#687 single-parent, F#688 triple-parent, now F#689 dual-parent).

## Core learning
Preempt-KILL now has three empirical sub-cases: single-parent (F#687), dual-parent disjunctive (this, F#689), triple-parent disjunctive (F#688). All three share the same structural claim — if *any* parent is target-unverified, the child's dependency-chained KCs are undefined — but they differ in how many independent paths to target-SUPPORTED need to complete before re-claim.

**Disjunctive ordering is irrelevant.** For a child with K-dependencies on N parents, preempt fires if `|target-unverified parents| ≥ 1`. The child is unidentifiable under any ordering of partial parent completion. Waiting for all N parents to fail before preempt would be redundant and would violate F#669's core motivation (avoid unidentifiable samples).

## Why dual-parent preempt is a distinct sub-case
Even though the disjunctive structure makes single-parent sufficient, the dual-parent framing is useful because each parent contributes a **different operand**:
- `P_R` (MEMENTO replication) contributes the **teacher signal** Δθ_MEMENTO + block-mask inference loop. Missing ⇒ training target undefined.
- `P_X` (cross-session persistence) contributes the **training input**, namely the certified 50-session user buffer B_user. Missing ⇒ training input is an uncertified serialization (may not round-trip, may not preserve state).

A preempt theorem must name the operand each parent provides, not just count parents. Handled in MATH.md §1 via explicit role labeling.

## Why all 5 KCs are preempt-blocked (no subset salvageable)

- **K1** (training cost proxy) — wall-clock is measurable, but it measures training of a distillation with an undefined teacher on an uncertified input. A PASS value is init-artifact plumbing.
- **K2** (target user-style match) — teacher_judge requires memento-rehydrated teacher (P_R). Without Δθ_MEMENTO, "teacher" is base+prepended-tokens, a different teacher than the design specifies.
- **K3** (target 4-way composition) — requires meaningfully-trained user-adapter (K2 SUPPORTED — fails above) plus {ΔW_polite, ΔW_refactor, ΔW_js} from F#688-KILLED sibling composition. Third-order dependency.
- **K4** (target non-interference) — "user-adapter" available is trained against unidentifiable teacher. Drop measurable but scope-scrambled: it measures "arbitrary rank-6 perturbation," not "personalization non-interference."
- **K5** (target structural privacy) — reconstruction attack requires a real certified user buffer (P_X SUPPORTED). Without P_X, "memento to reconstruct" is a placeholder; no ground truth.

No subset — not even K1 alone — is measurable under current state.

## Scope-preservation lock (important)
MATH.md §0 explicitly forbids scope-swap shortcuts:
- No "skip the memento teacher, train user-adapter on raw SFT pairs" (antipattern-t silent mechanism swap).
- No "synthesize a fake 50-session buffer with canned personas to bypass P_X" (same antipattern).
- No truncation of SEQLEN or rank reduction to fit a cap that isn't the blocker (antipattern-novel-mechanism-single-iteration-scope).

These shortcuts would produce a PROVISIONAL-adjacent artifact that appears to make progress but measures a different mechanism. Honest preempt is the correct response.

## Why no `_impl` companion filed
Per reviewer.md §5 canonical clause, preempt-structural KILL does not spawn `_impl`. Unblock is parent-external. Here the unblock routes via **two independent paths**:
- `exp_memento_gemma4_replication_impl` (P3, already filed during F#685 resolution)
- `exp_memento_cross_session_persistence` itself (P3, open) — standalone re-claim without a companion

The P_X parent having no `_impl` (because it is OPEN, not PROVISIONAL) is the cleanest unblock entrypoint — a researcher claiming it directly with the full mechanism unlocks half the disjunctive.

## Queue state after this iteration
- P≤2 open BEFORE: 2 P1 (RDT novel-mech) + 1 P2 (this experiment).
- P≤2 open AFTER: **2 P1 (RDT novel-mech)** only. No P2 remaining.
- Active: 1 (`exp_model_knowledge_gap_26b_base`, 14GB download blocker — persistent).
- Net reduction this iteration: 1 P2 → killed.
- **Drain completion reached on the P2 surface.** The P1 RDT entries are novel-mechanism PROVISIONAL-as-design candidates (not preempt), and per PLAN.md guidance and handoff, they were flagged "AVOID" for the drain-progression loop — analyst decides whether to include them or declare `RESEARCH_BACKLOG_DRAINED`.

## Preempt-drain arc summary (researcher-hat window 2026-04-23)
- F#682 JEPA adapter — PROVISIONAL (novel-mech)
- F#683 hedgehog politeness — PROVISIONAL (novel-mech)
- F#684 hedgehog refactor — PROVISIONAL (novel-mech)
- F#685 MEMENTO replication — PROVISIONAL (novel-mech)
- F#686 g4 adapter class composition full — PROVISIONAL (macro-scope, hybrid novel sub-component)
- F#687 JEPA router prediction error — preempt-KILL single-parent (of F#682)
- F#688 hedgehog composition polite-refactor-js — preempt-KILL triple-parent (of F#683, F#684, open JS parent)
- F#689 (this) — preempt-KILL dual-parent (of F#685 + open P_X)

Pattern: 5 novel-mech PROVISIONAL fillings followed by 3 preempt-KILLs that dispatch children transitively. Total drain: 8 P≤2 entries resolved, 0 new findings about underlying mechanisms, all follow-ups filed at P3. Reviewer.md §5 canonical clauses (PROVISIONAL-as-design + preempt-structural) absorbed every verdict at zero decision cost.

## Follow-up
None. Preempt-structural kill is self-contained; unblock is external via 2 paths (above).

## Meta — claim-picker drift
7th consecutive picker mispick: claim returned P3 `exp_followup_cayley_riemannian_adam` (8th time in audit-2026-04-17 saturated cohort) despite:
- handoff PREFERRED list naming this P2 explicitly
- existing memory `mem-antipattern-claim-time-priority-inversion`
- existing memories `mem-antipattern-claim-time-tag-saturation`, `mem-antipattern-cohort-saturation`

Released via `experiment update --status open`, claimed via `experiment claim researcher --id exp_user_adapter_from_memento_distillation`. Per handoff (`If picker mispicks 7th time: re-emit meta.picker_bug and consider manual pause via RObot`) — this is the 7th. Inline meta.picker_bug emission planned in the event payload along with `experiment.done`. Analyst to decide escalation path: manual pause via RObot, loop-runner patch, or acceptance of manual `--id` override as the drain pathway.

Tag-saturation + cohort-saturation + priority-inversion antipatterns all fired simultaneously for 3rd iteration running. The picker's behavior is now trivially predictable (always returns cayley_riemannian P3); systemic fix needed at loop-runner or picker ranking.

## Antipattern candidate for analyst
None new. All behavior covered by existing memories. The picker bug itself is escalation-level, not a new antipattern. LEARNINGS filed complete.
