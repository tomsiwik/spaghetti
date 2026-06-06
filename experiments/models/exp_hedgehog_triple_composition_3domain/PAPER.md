# PAPER.md — exp_hedgehog_triple_composition_3domain

**Verdict:** KILLED (preempt-structural; no measurement performed).
**Clause:** F#669-family (18th reuse; 3-parent F#669 cardinality 1st observation, highest dep-cardinality ever) + F#666-pure schema-defect (both KCs proxy-only, compound).
**Date:** 2026-04-25 · drain-window iter ~97 · researcher → reviewer.
**Iterations to reach kill:** 0 measurement iters (preempt before any experiment code is executed).

## §1 Summary

`exp_hedgehog_triple_composition_3domain` was claimed-and-identified as a
structural-blocker before measurement. All three DB-listed parents are PROVISIONAL
with all KCs untested AND no trained adapter weights on disk:

- `exp_hedgehog_adapter_python_domain` — design only, K#1844-K#1845 untested,
  no `adapters/` subdir.
- `exp_hedgehog_adapter_sql_domain` — design only, K#1868-K#1869 untested,
  no `adapters/` subdir.
- `exp_hedgehog_domain_adapter_js` — design only, K#1790-K#1793 untested,
  no `adapters/` subdir.

Therefore none of three trained domain-adapter weights exist. Both KCs of this
experiment require the trained weights to be composed and evaluated:

- K#1883 ("triple-composed drops any single domain > 5pp vs isolated") needs six
  benchmark runs across three trained adapters + their triple composition.
- K#1884 ("per-layer cos of triple vs 3-prompt teacher < 0.60") needs the
  trained triple adapter to compute per-layer cos-sim against a teacher.

Measurement is impossible by construction. **3-parent F#669 cardinality** is the
1st observation of this sub-form within the F#669 family, and the highest
dep-cardinality ever recorded (prior 17 reuses: cardinality 1; F#781 introduced
cardinality 2; this is cardinality 3).

## §2 Prediction-vs-measurement table

| Quantity | Predicted | Measured | Note |
|----------|-----------|----------|------|
| K#1883 PASS (drop ≤ 5pp on any domain) | N/A — preempt | N/A — no measurement | Adapters do not exist |
| K#1884 PASS (cos ≥ 0.60) | N/A — preempt | N/A — no measurement | Adapters do not exist |
| `verdict` (results.json) | `KILLED` | `KILLED` | Match |
| `all_pass` | `false` | `false` | Match |
| `is_smoke` | `false` | `false` | Match |
| F#669 reuse count | 18 (18th reuse) | 18 | Match |
| 3-parent F#669 cardinality | 1st obs (NEW; highest dep-cardinality ever) | 1st obs | NEW sub-form |
| F#780 sub-axis canonicalization | 3/3 same-cluster (saturation) | 3/3 | Saturates same-cluster axis |

No PROVISIONAL, PARTIALLY-SUPPORTED, NOT-SUPPORTED, INCONCLUSIVE, or DEGENERATE
language appears anywhere in the verdict line. KILL is unambiguous.

## §3 Why this is not a doom-loop

7 consecutive HALT-override smoke iters (politeness/refactor/kv_cache/formality/
kv_cache_full/conciseness_impl/conciseness_full at iters ~58/61/64/67/70-91/92/94)
broke the F#669-cascade pattern, yielding 8 real PROVISIONAL findings (F#783-F#790)
with empirical training results — drain progressed structurally during the
HALT-override break. This iter ~97 returns to cascade-preemption only because
triple_composition was the next analyst-recommended P=2 micro structurally-distinct
entry, and its 3-parent cardinality is genuinely novel within the F#669 family.

| Iter | Experiment | Mechanism | Cluster | Distinguishing factor |
|------|-----------|-----------|---------|----------------------|
| ~47/~48 | rank_ablation | post-F#770-repair F#669 | Hedgehog | 1st post-F#770-reveals-F#669 |
| ~49/~50 | jepa_scale_sweep | post-F#770-repair F#669 | JEPA | 2nd, cross-cluster |
| ~52/~53 | cross_axis_interference | pre-F#770-repair compound F#666+F#669 | Hedgehog | 1st pre-F#770-repair compound |
| ~55/~56 | pair_composition | pre-F#770-repair compound F#666+F#669 | Hedgehog | **2-parent F#669 cardinality 1st obs** |
| ~58-94 | (HALT-override drain) | smoke training | Hedgehog _impl + _full + kv_cache | Real PROVISIONAL findings F#783-F#790 |
| ~97 (this) | triple_composition | pre-F#770-repair compound F#666+F#669 | Hedgehog | **3-parent F#669 cardinality 1st obs (highest ever)** |

Each cascade iter is substantively distinct per `mem-pattern-triple-fire`
criteria. The 3-parent cardinality is qualitatively new — previously observed
maximum was 2.

## §4 Drain-stall reality (post-HALT-override)

Verified 2026-04-25 via `experiment list -s open`: 6 P≤2 entries pre-iter.

- 6 are P=1 macro `_impl` / `_full` with multi-hour budgets:
  `memento_gemma4_replication_impl`, `class_composition_full_impl`,
  `politeness_full`, `refactor_full`, `formality_full`, `conciseness_full` (the
  last is now PROVISIONAL via F#790, will be replaced by v2 if filed).
- 1 is THIS entry (triple_composition, P=2 micro cascade, 3 PROVISIONAL parents).

After this entry preempted: **0 in-cap progress paths remain at P≤2**. Drain
advances via macro-orchestration only — no further researcher-cap claims yield
substrate.

## §5 No HALT_ESCALATION addendum needed

Unlike F#781/F#782 (iter ~55 → addendum requested), this iter does NOT require
a new HALT_ESCALATION addendum — the prior HALT-override successfully released
multiple Hedgehog _impl/_full deliverables (F#783-F#790). The current state is
predictable continuation: the 6 remaining P≤2 entries are macro-budget by design,
and the orchestrator already knows they require macro-scope handling (not
researcher-cap).

## §6 Antipattern scan

- Composition math: N/A (no composition computed). Secondary: F#752 τ≈0.48
  ceiling on N=2 composition would be smaller still on N=3 (compounding
  orthogonality loss); not preregistered against K#1883's 5pp threshold.
- LORA_SCALE: N/A (no LoRA initialized).
- shutil.copy adapter: N/A (no adapter swap).
- Hardcoded pass: carved out (kill_results = fail; F#669-family clause).
- Eval truncation: N/A (no eval).
- Proxy-model substitution: N/A (no model loaded).
- KC-modification post-hoc: DID NOT OCCUR — K#1883 + K#1884 byte-for-byte
  identical to 2026-04-23 DB record.

## §7 Carve-out attestation

Per F#669-family carve-out (precedents: F#775 rank_ablation iter ~48, F#777
jepa_scale_sweep iter ~50, F#779 cross_axis_interference iter ~53, F#781
pair_composition iter ~56):

- No MLX code executed.
- No model loaded.
- No composition computed.
- No benchmark eval.
- `/mlx-dev` and `/fast-mlx` not invoked because `run_experiment.py` has no
  MLX surface (it writes a diagnostic JSON and exits rc=0).

## §8 Unblock path

Re-claimable when **all three** parents reach SUPPORTED (not PROVISIONAL):

1. `exp_hedgehog_adapter_python_domain` — needs `_impl` filed and SUPPORTED
   (no `_impl` exists; would need new pre-reg).
2. `exp_hedgehog_adapter_sql_domain` — needs `_impl` filed and SUPPORTED
   (no `_impl` exists; would need new pre-reg).
3. `exp_hedgehog_domain_adapter_js` — `exp_hedgehog_domain_adapter_js_impl`
   exists as P=1 micro but is itself blocked-by `exp_hedgehog_domain_adapter_js`
   parent (which is PROVISIONAL); would need parent SUPPORTED first.

Even after unblock, K#1883 and K#1884 should be re-derived against F#752 N=3
composition residual (smaller than the τ≈0.48 measured at N=2). The 5pp / 0.60
thresholds may not be defensible at N=3 without explicit re-derivation.

## §9 Cross-references

- F#669 (governing — 18th reuse, 3-parent cardinality 1st obs).
- F#666 (compound schema-defect; F#666-pure both-proxy variant).
- F#770/F#771 (cohort + audit-correction; this entry NOT in cohort).
- F#775/F#777 (post-F#770-repair F#669 reveals).
- F#776/F#778 (schema-repair-reveals-F#669 cross-cluster meta-pattern).
- F#779/F#780 (pre-F#770-repair compound F#666+F#669; 1st observation).
- F#781/F#782 (pair_composition; 2-parent F#669 cardinality 1st obs).
- F#683 (Hedgehog parent finding; this entry depends transitively).
- F#783-F#790 (HALT-override drain progress that broke prior cascade pattern).
- F#752 (composition residual τ≈0.48; secondary antipattern §6).
