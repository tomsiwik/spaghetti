# PAPER.md — exp_user_adapter_from_memento_distillation

## Verdict: KILLED (preempt-structural, F#669 5th+ reuse, dual-parent disjunctive)

This experiment was preempt-killed before any MLX code was written. The kill is structural, not empirical: all 5 kill criteria transitively require two parent artifacts — a memento-rehydrated teacher (from `exp_memento_gemma4_replication`) and a certified 50-session user memento buffer (from `exp_memento_cross_session_persistence`). Neither parent is target-SUPPORTED:

- `exp_memento_gemma4_replication` → PROVISIONAL (F#685, design-only scaffold; novel-mechanism 2-stage SFT + block-mask inference loop never executed)
- `exp_memento_cross_session_persistence` → OPEN (P3, never claimed, never run)
- Sibling `exp_hedgehog_composition_polite_refactor_js` (needed for K3 4-way composition) → KILLED (preempt, F#688, 3/3 Hedgehog parents unverified)

Testing K1–K5 against an undefined teacher signal and/or an uncertified buffer produces either vacuous PASS (init-artifact co-occurrence) or vacuous FAIL (pipeline plumbing noise), i.e. an unidentifiable sample per F#669.

## Prediction vs measurement

| KC  | Prediction                                                                        | Measurement   | Verdict  |
| --- | --------------------------------------------------------------------------------- | ------------- | -------- |
| K1  | training wall-clock ≈ 22-28 min on M5 Pro                                         | not measured  | untested |
| K2  | \|Δjudge\| ≈ 2-4 pp (student matches teacher closely)                             | not measured  | untested |
| K3  | axis drops ≤ 2pp on each of {user, polite, refactor, JS}                          | not measured  | untested |
| K4  | MMLU drop ≤ 1pp, HumanEval drop ≤ 1pp with user-adapter attached                  | not measured  | untested |
| K5  | reconstruction L2 error ≫ threshold (information-theoretic argument)              | not measured  | untested |

**All KC rows are "not measured" because neither parent artifact exists.** The dual-parent preempt is disjunctive-over-parents: either one missing ⇒ child unidentifiable. This positioning is strictly sharper than single-parent (F#687) and weaker than triple-parent (F#688) only in parent count — structural stability under any parent-completion ordering is identical.

## Assumptions

- Both parents will eventually be re-run to full scale:
  - `P_R` has `_impl` companion at P3 (`exp_memento_gemma4_replication_impl`, filed during F#685 resolution).
  - `P_X` has no `_impl` because its own status is `open`, not `provisional` — it can be claimed directly when the queue reorders.
- When both parents reach `supported` with target KCs verified, this child becomes re-claimable with the original design (MATH.md §6) intact.
- No redesign attempted this iteration to avoid the dual-parent dependency (e.g. synthetic 50-session buffer that bypasses `P_X`, or pair-composition that drops K3 to N=2). The alternative-unblock path is mentioned in MATH.md §4 but is explicitly out of scope for the drain-progression iteration (Option A per analyst handoff).
- Scope-preservation lock (MATH.md §0): this preempt does **not** silently swap the mechanism. The original design calls for Hedgehog per-layer cos-sim distillation over a memento-rehydrated teacher. Under preempt, no substitution is attempted — any "skip the memento teacher, train user-adapter on raw SFT pairs" shortcut would be antipattern-t (silent mechanism swap) and antipattern-novel-mechanism-single-iteration-scope. Forbidden.

## Related findings

- **Finding #669** — defining precedent for preempt-KILL on target-unverified parent (single-parent case).
- **Finding #671, #672, #687, #688** — prior reuse of F#669 (4 prior applications; this is the 5th+). F#688 is the closest structural analog (multi-parent composition-dependent child, all parents unverified).
- **Finding #685** — parent `P_R` PROVISIONAL (novel-mechanism 2-stage SFT + block-mask attention).
- **Finding #683, #684** — relevant because sibling `exp_hedgehog_composition_polite_refactor_js` (KILLED F#688, the K3 dependency) required Hedgehog-distilled adapters from these PROVISIONAL politeness + refactor parents.
- **Finding #666** — target-gated KC discipline; per reviewer.md §5 canonical clause, does NOT gate preempt-KILL.
- **Finding #627** — N=24 SFT-LoRA runtime composition SUPPORTED on Gemma 4 E4B; design precedent for §6 theorem.
- **Finding #562** — structural orthogonality at native dims supports N-way composition under Grassmannian A-init.
- **Finding #571** — pre-merge composition killed 4× (motivates runtime-only composition).

## Unblock path

Re-claim this experiment when **both conditions** simultaneously hold:

1. `exp_memento_gemma4_replication` → `status=supported` with K2 (GSM8K drop < 5pp ∧ MMLU drop < 3pp target) and K3 (KV-channel ablation ≥ 10pp replicating paper's 15pp AIME24 finding) SUPPORTED at full scale. Routes through `exp_memento_gemma4_replication_impl` (P3, filed).
2. `exp_memento_cross_session_persistence` → `status=supported` with K1 (rehydration latency < 50ms at N=100 mementos), K2 (multi-turn task accuracy ≥ 90% of full-context on 30-turn user-simulator), K3 (sub-linear compaction bounded at < 2k tokens), and K4 (pickle/disk round-trip within 2pp in-memory accuracy) SUPPORTED. No `_impl` needed — directly claimable.
3. **For K3 specifically** (4-way composition): resolution of F#688's triple-Hedgehog-parent preempt (via `exp_hedgehog_behavior_adapter_politeness_impl`, `exp_hedgehog_procedural_adapter_refactor_impl`, `exp_hedgehog_domain_adapter_js` all target-SUPPORTED). Partial unblock on K1/K2/K4/K5 is possible once P_R + P_X SUPPORT without K3 resolution — K3 can be dropped or re-scoped to pair composition.

Then the memento-rehydrated teacher exists as a target-validated operator, the 50-session buffer exists as a certified target, and the original design (MATH.md §6) becomes executable at full scale; the 5 KCs become measurable against predictable thresholds.

**Alternative unblock (out of scope now):** redesign child with fewer parent dependencies — e.g. synthetic 50-session buffer generated from canned user personas (bypasses `P_X`), or pair composition N=2 (bypasses full composition dependency). Would require new experiment id.

## Follow-up filed

None. Preempt-structural kill does not spawn an `_impl` companion (per reviewer.md §5 canonical clause). Unblock is parent-external via two independent paths:

- P_R unblock → `exp_memento_gemma4_replication_impl` (P3, filed during F#685 resolution)
- P_X unblock → `exp_memento_cross_session_persistence` itself (P3, open, no companion needed)
