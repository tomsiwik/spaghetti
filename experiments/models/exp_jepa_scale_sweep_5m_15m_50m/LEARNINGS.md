# LEARNINGS — exp_jepa_scale_sweep_5m_15m_50m

**Date:** 2026-04-25 · **Hat:** Analyst · **Drain-window iter ~51**
**Verdict ratified:** KILLED (preempt-structural, F#669-family clause)

## Core Finding

Preempt-KILL: scale-sweep child cannot evaluate K1988/K1989 because the parent measurements they reference (parent K1768 token-space r=16 LoRA baseline at n≥100; trained JEPA residual-stream adapters at any scale) do not exist. **Findings ratified:** F#777 (F#669 15th reuse, 4th F#682-child, 1st post-F#770 F#682-child) + F#778 (schema-repair-reveals-F#669 meta-pattern, 2nd cross-cluster obs Hedgehog→JEPA, 2/3 to canonicalization).

## Why

1. **F#669-family transitivity.** K1988 RHS = parent K1768 (untested; parent _impl F#772 Phase A only n=10 baseline=40.0% < n≥100 minimum). K1989 requires trained JEPA adapters at s∈{5M,15M,50M} on a Phase B custom MLX training loop that returns `NotImplementedError`. Both comparisons evaluate against `NaN` → unidentifiable.
2. **Schema-repair-reveals-F#669 (meta-pattern).** Iter ~38 F#770 repair added paired target KCs satisfying F#666; this *migrates* the diagnosis from F#666-pure-standalone (KC-design-defect) to F#669-cascade (parent-cascade-defect) without unblocking the child. Net: identical preempt-KILL disposition; different unblock path (parent-completion vs re-pre-reg).
3. **Cluster-invariance attested.** 1st obs F#776 = Hedgehog (rank_ablation iter ~48); 2nd obs F#778 = JEPA (this scale_sweep iter ~50). Same F#770 cohort, two distinct clusters, same migration. The pattern is structural, not cluster-specific.
4. **F#682 cluster transition.** F#727/F#728/F#729 fired pre-F#770 on F#666-pure-standalone clause. F#777 is the cluster's first post-F#770 F#669 instance — KC-design-defect era → schema-repair-revealed-F#669 era.

## Implications for Next Experiment

1. **Predicted 3rd obs (canonicalization trigger):** `exp_hedgehog_cross_axis_interference` (P=2, parent politeness PROVISIONAL, currently F#666-pure-standalone with K1859 against parent K1784 untested). If F#770-repaired with paired target referencing K1784 → F#669 cascade fires → 3rd cross-cluster obs → meta-pattern canonicalizes per `mem-pattern-triple-fire`. If left F#666-pure-standalone → preempt under F#666-pure clause (no schema-repair-reveals-F#669 advancement).
2. **Pre-claim audit rule (operational):** before claiming any F#770-cohort child, run parent-status check first. If parent is PROVISIONAL not SUPPORTED, route preempt-KILL via F#669-family clause without measurement. Saves ~30-min-per-iter scaffold writing.
3. **Drain progress:** P≤2 open queue 12 → 11; active 1 → 0; finding-ledger 37 → 39. No P=1 macro `_impl` claims (4-10h budget exceeds 90-min cap). Best next P=2: `exp_hedgehog_cross_axis_interference` (locks 3rd obs IF F#770-repaired, else clean F#666-pure-standalone preempt). AVOID `exp_hedgehog_triple_composition_3domain` (3-parent F#669 cascade — JS+Python+SQL `_impl`s all open).
4. **Unblock path** (for re-claim): parent F#682 must reach SUPPORTED via _impl F#772 with K1768 measured at n≥100 + executable trained JEPA adapter at s∈{5M,15M,50M}. K1988/K1989 stay byte-stable; no KC-augmentation needed.

## Cross-refs

F#669 (governing, 14 prior reuses → F#777 = 15th) · F#666 (KC structure, satisfied post-repair) · F#770/F#771 (cohort enumeration + bulk repair iter ~38) · F#682 (parent PROVISIONAL) · F#772 (parent _impl PROVISIONAL Phase A only) · F#775+F#776 (Hedgehog rank_ablation iter ~48 — 1st obs of meta-pattern; F#778 advances to 2nd) · LeWorldModel arxiv:2603.19312 + LeJEPA arxiv:2511.08544 (parent foundational refs).
