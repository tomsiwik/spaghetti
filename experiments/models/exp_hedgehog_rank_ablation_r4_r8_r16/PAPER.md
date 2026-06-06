# exp_hedgehog_rank_ablation_r4_r8_r16 — Preempt-Structural KILL

**Status:** KILLED (preempt-structural, F#669-family clause)
**Date:** 2026-04-25
**Verdict basis:** F#669-family — child KCs reference parent measurements that the parent has not produced. Schema-repaired iter ~36 (F#770 cohort fix added paired target KCs K1980/K1981/K1982 satisfying F#666); repair was necessary but not sufficient — F#666 gates KC structure; F#669 gates threshold measurability.
**Measurements taken:** 0.

---

## Prediction vs Measurement Table

| Kill Criterion | Pre-registered Prediction | Measurement | PASS/FAIL |
|---|---|---|---|
| K#1852 (proxy) | r=4 cos-sim < 80% of r=8 cos-sim | UNTESTED — preempt-KILL | **untested** |
| K#1853 (proxy) | r=16 < 5pp cos-sim improvement over r=8 | UNTESTED — preempt-KILL | **untested** |
| K#1980 (target) | r=4 ΔJudge < (r=8 ΔJudge − 5pp) on n=100 neutral prompts | UNTESTED — preempt-KILL (RHS=NaN; parent K1783 untested) | **untested** |
| K#1981 (target) | r=16 ΔJudge < (r=8 ΔJudge + 3pp) on n=100 neutral prompts | UNTESTED — preempt-KILL (RHS=NaN; parent K1783 untested) | **untested** |
| K#1982 (target) | For ANY r ∈ {4,8,16}: MMLU drop > 3pp OR HumanEval drop > 3pp | UNTESTED — preempt-KILL (no trained Hedgehog adapter at any r; parent F#683 PROVISIONAL) | **untested** |

No measurement performed; preempt-structural per MATH.md §1 (F#669-family transitivity theorem).

---

## Why this experiment was preempt-killed

### 1. F#669-family clause (governing)

Every paired target KC in the post-schema-repair set resolves to a comparison against a parent measurement that the parent has not produced:

- **K1980** RHS = `(r=8 ΔJudge − 5pp)`; r=8 ΔJudge is parent's K1783 — currently `untested`.
- **K1981** RHS = `(r=8 ΔJudge + 3pp)`; same NaN-RHS structure.
- **K1982** requires per-rank trained Hedgehog adapters at r ∈ {4, 8, 16}; parent's Hedgehog distillation loop has no executable MLX implementation (parent F#683 PROVISIONAL, Phase B `NotImplementedError`).

Comparison `value < NaN` is unidentifiable. Per F#669, testing produces unidentifiable samples; preempt-KILL preserves budget.

### 2. Schema-repair-reveals-F#669 meta-pattern (1st observation)

The iter ~36 schema-repair (F#770 cohort fix) added K1980/K1981/K1982 to satisfy F#666-discipline (target-paired KCs for the previously F#666-pure-standalone proxy KCs K1852/K1853). The repair migrated the diagnosis:

- **Pre-repair**: F#666-pure-standalone (KC-design-defect) — child preempt-KILLed under F#666-pure clause.
- **Post-repair**: F#666-compliant pair → F#666-pure preempt no longer fires → BUT new target KC's threshold references parent measurement → F#669 cascade fires instead.

Net: **identical disposition (preempt-KILL); different governing clause; different unblock path** (parent-completion vs re-pre-reg). 1st observation; promotion threshold per `mem-pattern-triple-fire` is 3 instances. Predicted next 2 (already constructed in DB awaiting claim): `exp_jepa_scale_sweep_5m_15m_50m` (K1988/K1989 already F#770-repair-added against parent residual-stream PROVISIONAL K1768), `exp_hedgehog_cross_axis_interference` (same parent politeness PROVISIONAL; KC #1859 currently F#666-pure — would migrate if F#770-repaired).

### 3. 1st Hedgehog-cluster F#669 instance

Prior 8 Hedgehog-cluster preempt-KILLs (F#714, F#716, F#720, F#721, F#722, F#723, F#755, F#756) were all F#666-pure-standalone or §5 sub-types. This is the 1st Hedgehog-cluster F#669-family instance — the cluster's transition from KC-design-defect-cluster to parent-cascade-defect-cluster. Caused by F#770 cohort schema-repair retroactively elevating the Hedgehog children's diagnosis tier.

---

## Truth Table (degenerate by F#669-family construction)

| K1980/K1981 outcome | K1982 outcome | F#669 verdict |
|---|---|---|
| Any (NaN RHS) | Any (no trained adapter) | All cells unidentifiable; no behavioral conclusion possible |

No cell yields a behavioral conclusion; preempt-KILL is the only consistent verdict.

---

## Unblock path

Re-claimable when **all three** of:

1. Parent `exp_hedgehog_behavior_adapter_politeness` reaches `status=supported|proven` via its `_impl` companion (`exp_hedgehog_behavior_adapter_politeness_impl`, P=1, currently `open`), with K1782 + K1783 + K1784 SUPPORTED.
2. Parent's `_impl` has produced a numerical value for `r=8 ΔJudge` (parent K1783) that can serve as the RHS reference for K1980/K1981.
3. Parent's `_impl` has produced an executable trained Hedgehog adapter (or training script) that this child can re-instantiate at r ∈ {4, 16} — per-rank training as a function of rank only, all other hyperparameters inherited from parent.

**No KC-augmentation needed at re-claim** — K1980/K1981/K1982 are F#666-compliant. Parent-completion is the unblock. Scope-reduction to plain LoRA, PPL parity, or single-rank measurement explicitly rejected (antipattern-t / antipattern-i / antipattern-u; see MATH.md §6).

---

## Related work

- **Hedgehog Stage-1** (Zhang 2024 arxiv:2402.04347; Moudgil/Apple+MILA arxiv:2604.14191 Apr 2026) — per-layer attention cos-sim distillation; mechanism whose rank dependence this experiment would have measured.
- **F#669** (governing precedent, 13 prior reuses) — preempt-child-KCs-require-parent-target-claim-unverified.
- **F#666** (target-gated KILL discipline) — gates KC structure; satisfied post-schema-repair.
- **F#770/F#771** (2026-04-25) — schema-defect cohort enumeration + bulk repair driving the iter ~36 fix.
- **F#683** (parent PROVISIONAL).

---

## Assumptions / Notes

- Verdict assigned without measurement per guardrail 1006 (behavioral outcomes over metrics): even if K1852/K1853 cos-sim measurements were taken on plain-LoRA-at-{4,8,16} as a stand-in, the substitution would replace the Hedgehog distillation mechanism with a different algorithm — antipattern-t.
- F#669-family clause exempts the (m2) skill-attestation gate when no MLX training-loop code lands.
- F#770 schema-repair history drawn from researcher iter ~38 + analyst iter ~40 audit; assumed current as of 2026-04-25.

---

## Hand-off

Verdict: **KILLED preempt-structural (F#669-family clause)**. Findings to register:

- **F#669 14th reuse** — preempt-child-KCs-require-parent-target-claim-unverified; 1st Hedgehog-cluster F#669 instance.
- **F#NEW (1st observation, candidate for promotion)** — schema-repair-reveals-F#669 meta-pattern: F#770 cohort schema-repair (adding paired target KCs) migrates diagnosis from F#666-pure-standalone (KC-design-defect) to F#669-cascade (parent-cascade-defect) when parent is PROVISIONAL not SUPPORTED. Identical disposition (preempt-KILL); different unblock path. Predicted 2nd/3rd instances: jepa_scale_sweep, hedgehog_cross_axis_interference.

Drain-window index: ~47.

Recommended next P=2 candidates after this KILL (primary drain-progress focus):
- `exp_hedgehog_cross_axis_interference` (P=2, same parent politeness PROVISIONAL — predicted 2nd schema-repair-reveals-F#669 instance if F#770-repaired; currently F#666-pure standalone clause applies).
- `exp_jepa_scale_sweep_5m_15m_50m` (P=2, parent residual-stream PROVISIONAL — already F#770-repaired with K1988/K1989 against parent K1768; ready-to-claim 2nd schema-repair-reveals-F#669 instance).
- `exp_hedgehog_triple_composition_3domain` (P=2, parent JS+Python+SQL _impls all `open` at P=3 — F#669 cascade variant; 3 parents instead of 1).

AVOID: P=1 macro `_impl` entries (4-10h budgets exceed 90-min cap → PROVISIONAL routing).
