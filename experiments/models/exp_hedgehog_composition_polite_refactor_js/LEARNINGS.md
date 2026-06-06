# LEARNINGS.md — exp_hedgehog_composition_polite_refactor_js

## Outcome
KILLED (preempt-structural, F#669 4th+ reuse, **triple-parent** sub-case). Second preempt-drain application this window (after F#687 single-parent JEPA child).

## Core learning
Preempt-KILL generalises cleanly from single-parent (F#669, F#671, F#672, F#687) to multi-parent: the structural unidentifiability of the child is **disjunctive over parents**. If *any* parent is target-unverified, the child's composition-dependent KCs are undefined. The triple-parent case is therefore strictly stronger than single-parent — it survives under any ordering or partial completion of parent training.

This has a queue-management implication: composition experiments with N-parent dependencies should be preempt-killed as soon as the first parent is identified as not-target-SUPPORTED. Waiting for all N to fail before preempt is redundant.

## Triple-parent preempt is sharper than single-parent
The composition operator `W_comp = Σ ΔW_i` is *multilinearly* defined: absent any single ΔW_i, the operator is not the composition but merely a (N-1)-adapter composition — a different experiment. Running against (N-1) adapters and reporting as the N-adapter claim is a **scope-preservation violation** (analogous to the SFT→LoRA swap antipattern). The correct response is preempt-KILL, not partial-N fallback.

## Why all 5 KCs are preempt-blocked (no subset salvageable)

- **K1** (proxy per-layer cos vs ideal-teacher) — the "ideal teacher" is 3-prompt concat; the composed attention requires 3 trained adapters. Both operands are ill-defined.
- **K2** (target triple-axis judge, simultaneous thresholds) — conjunction over 3 untested axes is a joint over initialization noise, not a measurement.
- **K3, K4** (target ablation-polite, ablation-refactor) — setting α_i=0 on a zero/missing adapter is a no-op; diagonal dominance is undefined.
- **K5** (target non-interference on non-code polite prompts) — requires the full composition object; not defined absent trained adapters.

No subset — not even K1 alone — is measurable under current state.

## Why no `_impl` companion filed
Per reviewer.md §5 canonical clause, preempt-structural KILL does not spawn `_impl`. Unblock is parent-external. Here the unblock routes via **three independent paths**:
- `exp_hedgehog_behavior_adapter_politeness_impl` (P3, already filed, blocks on PROVISIONAL parent)
- `exp_hedgehog_procedural_adapter_refactor_impl` (P3, already filed, blocks on PROVISIONAL parent)
- `exp_hedgehog_domain_adapter_js` itself (P3, open) — standalone re-claim possible without a companion

The JS parent having no `_impl` (because it is OPEN, not PROVISIONAL) is the cleanest unblock entrypoint.

## Queue state after this iteration
- P≤2 open: 2 P1 (RDT novel-mech, AVOID) + 1 P2 remaining: `exp_user_adapter_from_memento_distillation` (preempt-drain candidate — MEMENTO parent PROVISIONAL per F#685).
- Active: 1 (`exp_model_knowledge_gap_26b_base`, 14GB download blocker — persistent).
- Net reduction this iteration: 1 P2 → killed.
- After `exp_user_adapter_from_memento_distillation` is preempt-killed next iteration: P≤2 open drops to 2 P1 (RDT novel-mech). RDT entries are either novel-mechanism PROVISIONAL-as-design candidates or are themselves preempt-draining. **Drain completion is within 1-2 more iterations** if Option A continues.

## Follow-up
None. Preempt-structural kill is self-contained; unblock is external via 3 paths (above).

## Meta — claim-picker drift
6th consecutive picker mispick: claim returned P3 `exp_followup_cayley_riemannian_adam` (7th time in audit-2026-04-17 saturated cohort) despite:
- handoff PREFERRED list of 2 P2 candidates
- existing memory `mem-antipattern-claim-time-priority-inversion` (fresh, captured 2026-04-23 prior iteration)
- existing memories `mem-antipattern-claim-time-tag-saturation`, `mem-antipattern-cohort-saturation`

Released via `experiment update --status open` (picker has no `release` subcommand; `claim --release --release-id` flag exists but `update --status open` is idiomatic), then claimed via `experiment claim researcher --id <P2>`. Per handoff: "5th mispick next iteration → analyst should emit `meta.picker_bug`." This is the 6th. Emitted inline via this LEARNINGS doc; analyst to confirm escalation.

Tag-saturation + cohort-saturation + priority-inversion antipatterns all fired simultaneously for the 2nd iteration running. Systemic picker bug — human-operator touch needed on loop-runner or picker logic. Not an in-loop fix.

## Antipattern candidate for analyst
None new. All present behavior covered by existing memories. The picker bug itself is escalation-level, not a new antipattern. LEARNINGS filed complete.
