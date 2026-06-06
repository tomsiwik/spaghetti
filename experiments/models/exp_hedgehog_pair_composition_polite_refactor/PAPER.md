# PAPER.md — exp_hedgehog_pair_composition_polite_refactor

**Verdict:** KILLED (preempt-structural; no measurement performed).
**Clause:** F#669-family (17th reuse; 2-parent F#669 cardinality 1st observation) + F#666-pure schema-defect (both KCs proxy-only, compound).
**Date:** 2026-04-25 · drain-window iter ~55 · researcher hat.
**Iterations to reach kill:** 0 measurement iters (preempt before any experiment code is executed).

## §1 Summary

`exp_hedgehog_pair_composition_polite_refactor` was claimed-and-identified as a
structural-blocker before measurement. Both DB-listed parents are PROVISIONAL
with all 4 KCs each untested:

- `exp_hedgehog_behavior_adapter_politeness` — Phase 0 only, K#1782-K#1785 untested.
- `exp_hedgehog_procedural_adapter_refactor` — design only, K#1786-K#1789 untested.

Therefore neither trained polite-adapter nor trained refactor-adapter weights
exist. Both KCs of this experiment require the trained weights to be composed
and evaluated:

- K#1846 ("pair-composed drops either axis > 5pp vs isolated") needs four
  benchmark runs across two trained adapters + their pair composition.
- K#1847 ("per-layer cos of pair vs 2-prompt teacher < 0.65") needs the trained
  pair adapter to compute per-layer cos-sim against a teacher.

Measurement is impossible by construction. **2-parent F#669 cardinality** is the
1st observation of this sub-form within the F#669 family.

## §2 Prediction-vs-measurement table

| Quantity | Predicted | Measured | Note |
|----------|-----------|----------|------|
| K#1846 PASS (drop ≤ 5pp on either axis) | N/A — preempt | N/A — no measurement | Adapters do not exist |
| K#1847 PASS (cos ≥ 0.65) | N/A — preempt | N/A — no measurement | Adapters do not exist |
| `verdict` (results.json) | `KILLED` | `KILLED` | Match |
| `all_pass` | `false` | `false` | Match |
| `is_smoke` | `false` | `false` | Match |
| F#669 reuse count | 17 (17th reuse) | 17 | Match |
| 2-parent F#669 cardinality | 1st obs | 1st obs | NEW sub-form |

No PROVISIONAL, PARTIALLY-SUPPORTED, NOT-SUPPORTED, INCONCLUSIVE, or DEGENERATE
language appears anywhere in the verdict line. KILL is unambiguous.

## §3 Why this is not a doom-loop (4th consecutive researcher preempt)

| Iter | Experiment | Mechanism | Cluster | Distinguishing factor |
|------|-----------|-----------|---------|----------------------|
| ~47/~48 | rank_ablation | post-F#770-repair F#669 | Hedgehog | 1st post-F#770-reveals-F#669 |
| ~49/~50 | jepa_scale_sweep | post-F#770-repair F#669 | JEPA | 2nd post-F#770-reveals-F#669, cross-cluster |
| ~52/~53 | cross_axis_interference | pre-F#770-repair compound F#666+F#669 | Hedgehog | 1st pre-F#770-repair compound, 1-parent F#669 cardinality |
| ~55 (this) | pair_composition_polite_refactor | pre-F#770-repair compound F#666+F#669 | Hedgehog | 2nd pre-F#770-repair compound, **1st 2-parent F#669 cardinality** |

Each iter is substantively distinct per `mem-pattern-triple-fire` criteria:
different cluster OR different mechanism OR different finding-index pair.
However, the **pattern of preempt-KILL itself** has canonicalized at 4 consecutive
instances. Per doom-loop guidance ("STOP the repetition and take a structurally
different action"), the structurally-different action is the HALT_ESCALATION
addendum requested at the analyst pass — not yet another preempt.

## §4 Drain-stall reality

Verified 2026-04-25 via `experiment list -s open`: 9 P≤2 entries.

- 6 are P=1 macro `_impl` with multi-hour budgets (politeness_impl 4-6h,
  formality_impl 4-6h, conciseness_impl 4-6h, memento_replication_impl, class_composition_full_impl).
- 1 is P=1 micro `_impl` with 4-6h budget (procedural_adapter_refactor_impl) —
  exceeds researcher cap.
- 1 is P=1 micro `_impl` with 2h budget (rdt_loop_kv_cache_impl) — parent legitimately
  scope-deferred to child; preempting blocks legit measurement path; budget exceeds
  researcher 90-min rule of thumb.
- 1 is THIS entry (pair_composition, P=1 micro cascade).
- 1 is triple_composition (P=2 micro cascade, 3 PROVISIONAL parents).

After this entry preempted: **1 in-cap progress path remains** (triple_composition).
After both: **zero** in-cap P≤2 paths exist. Orchestrator must promote at least
politeness_impl + refactor_impl to release the drain.

## §5 Recommended HALT_ESCALATION addendum (analyst-scope, next iter)

The existing HALT_ESCALATION.md (2026-04-19) addresses a different blocker stack
(Python 3.14 + analyst-cap). Current drain-stall is **macro-budget cap + cascade
preemption saturation** — distinct cause, requires new addendum.

Recommended addendum content:

1. **Promote `exp_hedgehog_behavior_adapter_politeness_impl` (P=1 macro 4-6h)** —
   highest-leverage; unblocks F#683-cluster cascade (~5+ children including this entry's
   parent #1).
2. **Promote `exp_hedgehog_procedural_adapter_refactor_impl` (P=1 micro 4-6h)** —
   unblocks this entry's parent #2 + procedural-cluster.
3. Defer non-Hedgehog macros (memento, class_composition) until F#683-cluster lands.
4. Separate orchestration for `rdt_loop_kv_cache_impl` (2h, parent scope-deferred).

## §6 Antipattern scan

- Composition math: N/A (no composition computed). Secondary: F#752 τ≈0.48
  ceiling not preregistered against K#1846's 5pp threshold — would need re-derivation
  if parents reached SUPPORTED.
- LORA_SCALE: N/A (no LoRA initialized).
- shutil.copy adapter: N/A (no adapter swap).
- Hardcoded pass: carved out (kill_results = fail; F#669-family clause).
- Eval truncation: N/A (no eval).
- Proxy-model substitution: N/A (no model loaded).
- KC-modification post-hoc: DID NOT OCCUR — K#1846 + K#1847 byte-for-byte
  identical to 2026-04-23 DB record.

## §7 Carve-out attestation

Per F#669-family carve-out (precedents: F#775 rank_ablation iter ~48, F#777
jepa_scale_sweep iter ~50, F#779 cross_axis_interference iter ~53):

- No MLX code executed.
- No model loaded.
- No composition computed.
- No benchmark eval.
- `/mlx-dev` and `/fast-mlx` not invoked because `run_experiment.py` has no
  MLX surface (it writes a diagnostic JSON and exits rc=0).

## §8 Cross-references

- F#669 (governing — 17th reuse).
- F#666 (compound schema-defect; F#666-pure both-proxy variant).
- F#770/F#771 (cohort + audit-correction; this entry NOT in cohort).
- F#775/F#777 (post-F#770-repair F#669 reveals).
- F#776/F#778 (schema-repair-reveals-F#669 cross-cluster meta-pattern; 2/3 toward canonicalization).
- F#779/F#780 (pre-F#770-repair compound F#666+F#669; this entry = 2nd same-cluster instance).
- F#683 (Hedgehog parent finding).
- F#752 (composition residual τ≈0.48; secondary antipattern §6).
- HALT_ESCALATION.md (2026-04-19, distinct blocker; new addendum requested for current macro-budget stall).
