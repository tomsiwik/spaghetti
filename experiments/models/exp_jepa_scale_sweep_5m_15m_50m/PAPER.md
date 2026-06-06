# exp_jepa_scale_sweep_5m_15m_50m — Preempt-Structural KILL

**Status:** KILLED (preempt-structural, F#669-family clause)
**Date:** 2026-04-25
**Verdict basis:** F#669-family — child KCs reference parent measurements that the parent has not produced. Schema-repaired iter ~38 (F#770 cohort fix added paired target KCs K1988/K1989 satisfying F#666); repair was necessary but not sufficient — F#666 gates KC structure; F#669 gates threshold measurability.
**Measurements taken:** 0.

---

## Prediction vs Measurement Table

| Kill Criterion | Pre-registered Prediction | Measurement | PASS/FAIL |
|---|---|---|---|
| K#1862 (proxy) | Next-embedding MSE doesn't improve from 5M→15M | UNTESTED — preempt-KILL | **untested** |
| K#1863 (proxy) | 15M adapter training > 90 min on M5 Pro | UNTESTED — preempt-KILL | **untested** |
| K#1988 (target) | best-scale JEPA GSM8K-Hard accuracy ≥ token-space r=16 LoRA baseline at matched compute, n≥100 | UNTESTED — preempt-KILL (RHS=NaN; parent K1768 untested) | **untested** |
| K#1989 (target) | best-scale beats worst-scale by ≥ 2pp on n≥100 OR all 3 within 1pp | UNTESTED — preempt-KILL (no trained JEPA adapter at any of {5M, 15M, 50M}; parent _impl Phase B NotImplementedError) | **untested** |

No measurement performed; preempt-structural per MATH.md §1 (F#669-family transitivity theorem, scale-sweep variant).

---

## Why this experiment was preempt-killed

### 1. F#669-family clause (governing)

Every paired target KC in the post-schema-repair set resolves to a comparison against a parent measurement that the parent has not produced:

- **K1988** RHS = `token-space r=16 LoRA baseline at matched compute on Gemma 4 E4B, n≥100, greedy`; this is parent's K1768 — currently `untested` (parent F#682 PROVISIONAL; parent _impl F#772 Phase A only with n=10 baseline=40.0%, NOT n≥100).
- **K1989** requires per-scale trained JEPA residual-stream adapters at s ∈ {5M, 15M, 50M}; parent's Phase B custom MLX training loop has no executable implementation (parent _impl F#772 PROVISIONAL, Phase B `NotImplementedError`). No JEPA training mechanism on this platform.

Comparison `value < NaN` is unidentifiable. Per F#669, testing produces unidentifiable samples; preempt-KILL preserves budget.

### 2. Schema-repair-reveals-F#669 meta-pattern (2nd observation)

The iter ~38 schema-repair (F#770 cohort fix) added K1988/K1989 to satisfy F#666-discipline (target-paired KCs for the previously F#666-pure-standalone proxy KCs K1862/K1863). The repair migrated the diagnosis:

- **Pre-repair**: F#666-pure-standalone (KC-design-defect) — child preempt-KILLed under F#666-pure clause.
- **Post-repair**: F#666-compliant pair → F#666-pure preempt no longer fires → BUT new target KC's threshold references parent measurement → F#669 cascade fires instead.

Net: **identical disposition (preempt-KILL); different governing clause; different unblock path** (parent-completion vs re-pre-reg). 2nd observation; promotion threshold per `mem-pattern-triple-fire` is 3 instances.

**Cross-cluster confirmation** (1st obs F#776 = Hedgehog rank_ablation iter ~48; this 2nd obs = JEPA scale_sweep iter ~49): the meta-pattern is NOT cluster-specific. The F#770 cohort spanned both Hedgehog and JEPA children (per F#770 enumeration); both clusters produce 1 F#669-cascade instance from 1 F#770-repair child. **Cluster-invariance attested.**

Predicted 3rd instance (canonicalization trigger): `exp_hedgehog_cross_axis_interference` IF F#770-repaired (currently F#666-pure standalone with KC #1859 against parent K1784 untested), OR another cohort child as the F#770 cohort drains.

### 3. 4th F#682-child F#669 instance, 1st post-F#770 F#682-child F#669

Prior 3 F#682-cluster preempt-KILLs (F#727 multilayer_prediction, F#728 contrastive_variant, F#729 frozen_encoder_ablation) were all pre-F#770 schema-repair era — they fired on F#666-pure-standalone clause directly without F#770-repair migration. This is the **1st post-F#770 F#682-child** F#669-family instance — the cluster's transition from KC-design-defect-cluster era to schema-repair-revealed F#669 era.

---

## Truth Table (degenerate by F#669-family construction)

| K1988 outcome | K1989 outcome | F#669 verdict |
|---|---|---|
| Any (NaN baseline, parent K1768 untested) | Any (no trained JEPA adapter at any scale) | All cells unidentifiable; no behavioral conclusion possible |

No cell yields a behavioral conclusion; preempt-KILL is the only consistent verdict.

---

## Unblock path

Re-claimable when **all three** of:

1. Parent `exp_jepa_adapter_residual_stream` reaches `status=supported|proven` via its `_impl` companion (`exp_jepa_adapter_residual_stream_impl`, P=1, currently `provisional` per F#772), with K1766 + K1767 + K1768 + K1769 SUPPORTED on the residual-stream training/eval split.
2. Parent's `_impl` has produced a numerical value for token-space-r=16-LoRA GSM8K-Hard baseline at matched compute on Gemma 4 E4B at n≥100 (parent K1768 RHS reference for child K1988). Phase A baseline is n=10 (40.0%) — INSUFFICIENT, K1988 requires n≥100, parent K1768 requires n≥200.
3. Parent's `_impl` has produced an executable trained JEPA residual-stream adapter (or training script with Phase B custom MLX training loop + layer-21 hook + 2-layer MLP prediction head + SIGReg Epps-Pulley M=1024) that this child can re-instantiate at scale s ∈ {5M, 15M, 50M} — per-scale training as a function of param-budget only, all other hyperparameters inherited from parent.

**No KC-augmentation needed at re-claim** — K1988/K1989 are F#666-compliant. Parent-completion is the unblock. Scope-reduction to plain LoRA, MSE-only saturation, single-scale-only measurement, or Phase-A-n=10-baseline-as-RHS explicitly rejected (antipattern-t / antipattern-i / antipattern-u / antipattern-l; see MATH.md §6).

---

## Related work

- **LeWorldModel** (Maes/LeCun/Balestriero arxiv:2603.19312, 2026-03-24) — JEPA + SIGReg; parent's foundational reference.
- **LeJEPA** (arxiv:2511.08544) — SIGReg method (isotropic Gaussian optimality, Cramer-Wold sliced projections).
- **F#669** (governing precedent, 14 prior reuses) — preempt-child-KCs-require-parent-target-claim-unverified.
- **F#666** (target-gated KILL discipline) — gates KC structure; satisfied post-schema-repair.
- **F#770/F#771** (2026-04-25) — schema-defect cohort enumeration + bulk repair driving the iter ~38 fix.
- **F#682** (parent PROVISIONAL).
- **F#772** (parent _impl PROVISIONAL, Phase A only).
- **F#775** (immediate F#669 predecessor, 14th reuse) — Hedgehog rank_ablation; established the schema-repair-reveals-F#669 meta-pattern at 1st obs.
- **F#776** (immediate meta-pattern predecessor, 1st obs) — schema-repair-reveals-F#669 meta-pattern; this iter advances to 2nd obs.

---

## Assumptions / Notes

- Verdict assigned without measurement per guardrail 1006 (behavioral outcomes over metrics): even if K1862 MSE measurement was taken on plain-LoRA-at-{5M, 15M, 50M} as a stand-in, the substitution would replace the JEPA residual-stream mechanism with a different algorithm — antipattern-t.
- F#669-family clause exempts the (m2) skill-attestation gate when no MLX training-loop code lands.
- F#770 schema-repair history drawn from researcher iter ~38 + analyst iter ~40 audit; assumed current as of 2026-04-25.
- Cluster-invariance claim (Hedgehog + JEPA both produce 1 F#669-cascade from F#770-repair) supported by 2 observations (rank_ablation, scale_sweep). Sample size = 2 of 2 observed. Promotion at 3rd obs.

---

## Hand-off

Verdict: **KILLED preempt-structural (F#669-family clause)**. Findings to register:

- **F#669 15th reuse** — preempt-child-KCs-require-parent-target-claim-unverified; 4th F#682-child F#669 instance; 1st post-F#770 F#682-child F#669.
- **F#776 promotion (1st → 2nd obs)** — schema-repair-reveals-F#669 meta-pattern advances to 2nd observation. F#770 cohort schema-repair (adding paired target KCs) migrates diagnosis from F#666-pure-standalone (KC-design-defect) to F#669-cascade (parent-cascade-defect) when parent is PROVISIONAL not SUPPORTED. Identical disposition (preempt-KILL); different unblock path. **Cross-cluster: NOT cluster-specific** (Hedgehog + JEPA both demonstrate). Predicted 3rd instance (canonicalization trigger): `exp_hedgehog_cross_axis_interference`.

Drain-window index: ~49.

Recommended next P=2 candidates after this KILL (primary drain-progress focus):
- `exp_hedgehog_cross_axis_interference` (P=2, same parent politeness PROVISIONAL — predicted 3rd schema-repair-reveals-F#669 instance IF F#770-repaired; currently F#666-pure standalone, so default clause is F#666-pure-standalone preempt unless schema is repaired first).
- `exp_hedgehog_triple_composition_3domain` (P=2, parent JS+Python+SQL `_impl`s all `open` at P=3 — F#669 cascade variant; 3 parents instead of 1; analyst flagged as "3-parent F#669 cascade — AVOID").

AVOID: P=1 macro `_impl` entries (4-10h budgets exceed 90-min cap → PROVISIONAL routing not SUPPORTED).
