# MATH.md — exp_jepa_scale_sweep_5m_15m_50m (PREEMPT-KILL)

## Verdict (pre-measurement)

**KILLED — preempt-structural** under the **F#669-family clause** (preempt-child-KCs-require-parent-target-claim-unverified). Schema-repaired at iter ~38 (F#770 cohort fix added paired target KCs K1988/K1989) but the repair revealed a deeper blocker: every paired KC threshold references parent measurements that the parent has not produced. Parent `exp_jepa_adapter_residual_stream` is `provisional` (F#682); K1768 (GSM8K-Hard accuracy on Gemma 4 E4B, the explicit RHS in K1988) is `untested` per parent's design-only filing. Parent `_impl` companion `exp_jepa_adapter_residual_stream_impl` is also `provisional` (F#772, Phase A only — Phase B/C `NotImplementedError`).

This is the **4th F#682-child F#669 reuse** (prior 3 F#682-child F#669 instances: F#727 `exp_jepa_multilayer_prediction`, F#728 `exp_jepa_contrastive_variant`, F#729 `exp_jepa_frozen_encoder_ablation` — all pre-F#770), and the **15th overall F#669 reuse** (after F#775, `exp_hedgehog_rank_ablation_r4_r8_r16` 14th).

Critically, this is the **2nd observation of the schema-repair-reveals-F#669 meta-pattern** (1st: F#776 from `exp_hedgehog_rank_ablation_r4_r8_r16` iter ~48). Promotion threshold per `mem-pattern-triple-fire` is 3 instances; this iter advances the meta-pattern toward canonicalization.

No measurement performed. No `_impl` companion (per F#669-family clause: unblock is parent-external).

---

## §0 Platform / skills / model pins

Included for completeness even though no platform code is executed.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). **Not invoked** — no MLX code written; honest disclosure per reviewer checklist (m2). The F#669-family clause exempts the skill-attestation gate when no MLX training-loop code lands.
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: N/A — would have been JEPA residual-stream adapter on layer 21 hook + 2-layer MLP prediction head + SIGReg(M=1024 random projections), at parameter scales {5M, 15M, 50M} per parent's residual-stream design.
- Parent dependency: `exp_jepa_adapter_residual_stream` (status `provisional`, F#682 — design-only scaffold; all 4 KCs K1766/K1767/K1768/K1769 untested).
- Parent `_impl` dependency (executor for K1768 measurement): `exp_jepa_adapter_residual_stream_impl` (status `provisional`, F#772 — Phase A token-space LoRA baseline only; Phase B custom MLX training loop + SIGReg + GSM8K-Hard target eval `NotImplementedError`).
- Sibling precedents (F#682-cluster, all preempt-KILLed pre-F#770 schema-repair era): F#727 (multilayer_prediction), F#728 (contrastive_variant), F#729 (frozen_encoder_ablation). This is the **4th F#682-child** F#669 — and the **1st post-F#770 F#682-child** F#669, marking the cluster's transition from KC-design-defect-cluster era to schema-repair-revealed F#669 era.
- Datasets: N/A — no GSM8K-Hard load, no token-space LoRA baseline run.

---

## §1 Preempt-KILL theorem (F#669-family transitivity, scale-sweep variant)

**Theorem (inter-experiment target unverifiability — JEPA scale-sweep variant).** Let `C` denote child experiment `exp_jepa_scale_sweep_5m_15m_50m` with kill criteria K = {K1862, K1863, K1988, K1989} (the 2 original proxy KCs + 2 F#770-schema-repair target-paired KCs added 2026-04-25; the original proxy KCs K1862/K1863 are now non-load-bearing per F#666 — kill requires both proxy AND target FAIL). Let `P` denote parent `exp_jepa_adapter_residual_stream` (F#682) and `P_impl` denote `exp_jepa_adapter_residual_stream_impl` (F#772, the executor companion of `P`).

For each F#770-added child target KC, the threshold reference resolves to a parent measurement:

- **K1988** (`best-scale (5M/15M/50M) GSM8K-Hard accuracy ≥ token-space r=16 LoRA baseline at matched compute on Gemma 4 E4B, n≥100, greedy`): explicitly states "(inherits parent exp_jepa_adapter_residual_stream K1768 target)". The "token-space r=16 LoRA baseline" is parent's K1768 reference baseline. Parent K1768 = `untested` ⇒ baseline = NaN ⇒ comparison `best-scale ≥ NaN` is unidentifiable.
- **K1989** (`best-scale GSM8K-Hard accuracy beats worst-scale by ≥ 2pp on n≥100 OR all 3 scales within 1pp behaviorally`): requires per-scale GSM8K-Hard accuracy on a *trained JEPA residual-stream adapter at each of {5M, 15M, 50M}*. Parent `P_impl` Phase B custom MLX training loop is `NotImplementedError`; no executable JEPA residual-stream training mechanism exists. Therefore per-scale `acc(GSM8K-Hard | adapter_param=5M)`, `... | 15M)`, `... | 50M)` are each undefined — unidentifiable simultaneously across all 3 scales.

If `P.status ∈ {provisional, open}` AND `P_impl.status ∈ {provisional, open}` — i.e. K1768 is untested AND no executable JEPA training loop exists — then:

- **K1988**: best-scale-vs-token-LoRA-baseline comparison undefined (NaN baseline). Unidentifiable.
- **K1989**: per-scale-cross-scale comparison undefined (no per-scale measurement possible; no trained JEPA adapter at any scale). Unidentifiable.

∴ Testing K = {K1988, K1989} while `P.status ≠ supported|proven` AND `P_impl.status ≠ supported|proven` produces unidentifiable samples on both. **QED.**

### §1.1 F#666 gating (post-schema-repair)

The schema-repair at iter ~38 (per F#770 cohort fix) added K1988, K1989 as target-paired KCs to satisfy F#666-discipline. Post-repair the KC set IS F#666-compliant *in principle*:

- K1862 (proxy, MSE saturation 5M→15M) ↔ K1988 (target, downstream GSM8K-Hard accuracy vs token-space LoRA) — paired.
- K1863 (proxy, 15M training-time > 90 min) ↔ K1989 (target, scale-saturation behavioral spread) — paired.

But **F#666-compliant pre-reg ≠ F#666-resolvable verdict**. F#666 gates the *KC structure*; F#669 gates the *measurability of the threshold*. The schema-repair fixed the structure (proxy without target) but not the measurability (target threshold references a parent measurement that does not exist AND requires an executable training loop that does not exist). F#769/F#770 governs this transition — the schema-repair is **necessary but not sufficient**.

### §1.2 Schema-repair-reveals-F#669 meta-pattern (2nd observation)

This is the **2nd observation** of the meta-pattern previously documented as **F#776 (1st obs, rank_ablation)**: an F#770-cohort schema-repair (adding paired target KCs to satisfy F#666) **exposes** a previously-masked F#669 cascade in cluster experiments where the parent is PROVISIONAL not SUPPORTED. Mechanism, restated:

- Pre-repair: child has F#666-pure standalone defect → preempt-KILLed under F#666-pure clause (e.g. F#727/F#728/F#729 for JEPA cluster, F#714-F#723/F#755/F#756 for Hedgehog cluster).
- Post-repair: child has F#666-compliant pair → F#666-pure preempt no longer fires → but the new target KC's threshold references parent measurement → F#669 cascade fires instead.
- Net: the diagnosis migrates from F#666-pure-standalone (KC-design-defect) to F#669-cascade (parent-cascade-defect). Identical disposition (preempt-KILL); different governing clause; different unblock path (re-pre-reg vs parent-completion).

**Cross-cluster confirmation** (Hedgehog rank_ablation at iter ~48 ⇒ JEPA scale_sweep at iter ~49): the meta-pattern is NOT cluster-specific. The F#770 cohort spanned both Hedgehog and JEPA children (per F#770 enumeration); both clusters produce 1 F#669-cascade instance from 1 F#770-repair child. Cluster-invariance is now attested.

Per `mem-pattern-triple-fire`, the meta-pattern reaches promotion at 3rd observation. Current count: **2nd** (this). Predicted 3rd instance: `exp_hedgehog_cross_axis_interference` IF F#770-repaired (currently F#666-pure standalone, would migrate to F#669 if a paired target KC referencing parent K1784 is added) OR another cohort child as the F#770 cohort drains.

---

## §2 Prior art

- **F#669** (2026-04-19) — defining precedent for preempt-KILL on target-unverified parent. 14 reuses prior to this; this is the 15th reuse.
- **F#682** (parent F#682 PROVISIONAL) — JEPA residual-stream adapter, design locked, all 4 KCs untested; implementation deferred.
- **F#772** (parent `_impl` PROVISIONAL) — Phase A token-space LoRA baseline executed (50 steps, val loss 1.840→0.581, n=10 baseline=40.0% on GSM8K-Hard); Phase B/C `NotImplementedError`.
- **F#666** (target-gated KC discipline) — governs the schema-repair pre-condition; satisfied post-repair.
- **F#770** (2026-04-25) — schema-defect cohort enumeration + bulk repair driving the iter ~38 fix that added K1988/K1989 to this experiment's KC set.
- **F#771** (2026-04-25) — F#770 promoted to formal status; documented over-scope tendency in cohort findings.
- **F#775** (2026-04-25, immediate predecessor) — F#669 14th reuse; 1st Hedgehog-cluster F#669 instance.
- **F#776** (2026-04-25, immediate predecessor) — schema-repair-reveals-F#669 meta-pattern 1st observation. This iter advances to 2nd.
- **F#727/F#728/F#729** — prior 3 F#682-cluster preempt-KILLs (multilayer_prediction, contrastive_variant, frozen_encoder_ablation), all pre-F#770. This is the **4th F#682-child F#669** instance, and the **1st post-F#770 F#682-child** F#669.
- **Sibling precedent (cross-cluster F#669-family canonicalization)**: F#699, F#737, F#738, F#739, F#758 (MEMENTO cluster); F#740, F#741 (Pierre-serving cluster); F#775 (Hedgehog cluster). Cross-cluster reuse pattern attested at ≥10 reuses; this advances to 15.

---

## §3 Predictions (registered, not measured)

| KC    | Claim                                                                                             | Kind   | Measurement status                |
| ----- | ------------------------------------------------------------------------------------------------- | ------ | --------------------------------- |
| K1862 | Next-embedding MSE doesn't improve from 5M→15M (diminishing returns at small scale)               | proxy  | untested (preempt-blocked, F#669) |
| K1863 | 15M adapter training > 90 min on M5 Pro (scale ceiling)                                           | proxy  | untested (preempt-blocked, F#669) |
| K1988 | best-scale (5M/15M/50M) JEPA adapter GSM8K-Hard accuracy ≥ token-space r=16 LoRA baseline at matched compute on Gemma 4 E4B, n≥100, greedy | target | untested (preempt-blocked, F#669) |
| K1989 | best-scale GSM8K-Hard accuracy beats worst-scale by ≥ 2pp on n≥100 OR all 3 scales within 1pp behaviorally | target | untested (preempt-blocked, F#669) |

KC semantics note: K1988/K1989 are F#770-schema-repair targets added 2026-04-25; both resolve to comparisons that require parent K1768 (untested) AND/OR a trained JEPA residual-stream adapter at scale s ∈ {5M, 15M, 50M} (no executable training loop on this platform per F#772 Phase B `NotImplementedError`).

---

## §4 Unblock condition

Re-claimable when **all three** of:

1. Parent `exp_jepa_adapter_residual_stream` reaches `status=supported|proven` via its `_impl` companion (`exp_jepa_adapter_residual_stream_impl`, P=1, currently `provisional`), with K1766 + K1767 + K1768 + K1769 SUPPORTED on the residual-stream training/eval split.
2. Parent's `_impl` has produced a numerical value for token-space-r=16-LoRA GSM8K-Hard baseline at matched compute on Gemma 4 E4B (parent K1768 RHS reference). Phase A baseline (50 steps val loss 1.840→0.581, n=10 baseline=40.0%) is preliminary — full-N (n≥100) measurement required for K1988 RHS.
3. Parent's `_impl` has produced an executable trained JEPA residual-stream adapter (or training script with Phase B custom MLX training loop + layer-21 hook + 2-layer MLP prediction head + SIGReg Epps-Pulley M=1024) that this child can re-instantiate at scale s ∈ {5M, 15M, 50M} — the per-scale training loop must be a function of param-budget only, all other hyperparameters inherited from parent.

**No KC-augmentation needed** at re-claim: K1988/K1989 are already F#666-compliant. The parent-completion is the unblock. A scope-reduced scale-sweep on a different mechanism (e.g. plain LoRA at the same param budgets) would substitute the algorithm — antipattern-t risk; explicitly NOT an acceptable unblock path.

Alternative scope-reduction to MSE-only saturation comparison (as a stand-in for downstream GSM8K-Hard accuracy) would replace the behavioral target with a weakly-correlated proxy (F#666 r≈0.08 between PPL/loss-saturation and task quality) — antipattern-i. Avoided.

---

## §5 Follow-up

No `_impl` companion filed — preempt-structural KILL is self-contained per the F#669-family clause + reviewer.md §5. Unblock is parent-external: parent's existing `exp_jepa_adapter_residual_stream_impl` (P=1, currently `provisional` per F#772) is the unblock gate; this child's re-claim is downstream of parent's `_impl` Phase B/C SUPPORTED transition.

---

## §6 Scope integrity

No silent objective swap (antipattern-t): this scaffold does NOT attempt:
- Substituting plain LoRA for JEPA residual-stream adapter as a scale-sweep mechanism — antipattern-t (substitutes algorithm).
- Substituting MSE saturation parity for GSM8K-Hard accuracy as the behavioral target — F#666 r≈0.08, antipattern-i (KC measures wrong object).
- Loading parent's Phase A token-space LoRA baseline (n=10) as the K1988 RHS reference — Phase A is n=10, not n≥100; parent K1768 explicitly requires n≥200; using n=10 as RHS proxy is antipattern-l (using preliminary measurement as RHS for full-N target).
- Reading parent's eventual K1768 measurement post-hoc when parent reports `not measured` — antipattern-l (hardcoded `pass` / fabricated measurement).
- Reducing child's scale set from {5M, 15M, 50M} to {15M} only and labeling as scope-completion — antipattern-u (silent scope reduction).
- Substituting a different Gemma 4 variant (e.g. e4b-it-bf16, or 26b-a4b-it) as the base model — antipattern-m (proxy substitution).

All six shortcuts would replace the actual scale-sweep mechanism the KCs measure with a proxy or partial measurement.

---

## §7 Anti-pattern scan

- Composition-math: N/A (no composition).
- LORA_SCALE: N/A (no LoRA training executed).
- shutil.copy: N/A (no code).
- Hardcoded `"pass": True`: N/A (no code; `all_pass: false` written).
- Eval truncation producing base=0%: N/A (no eval).
- Proxy-model substitution: explicitly rejected in §6.
- KC measures wrong object: KCs correctly identify MSE proxy (K1862), latency proxy (K1863), behavioral target (K1988 GSM8K-Hard accuracy), and scale-saturation target (K1989 cross-scale spread) — KCs are well-formed; the parent measurement they reference is the missing object.
- N=smoke reported as full: N/A (no N; `is_smoke: false`).
- Tautological routing: N/A (no routing).
- Thinking-mode truncation: N/A (no eval).
- File-existence cache: N/A (no code).
- Copy-paste scaffolding: scaffold derived from `exp_hedgehog_rank_ablation_r4_r8_r16` (1st schema-repair-reveals-F#669 instance, immediate structural sibling at iter ~48) but variant-specific sections (JEPA-cluster vs Hedgehog-cluster reuse counts, F#682 parent F#669 history, scale-sweep mechanism vs rank-sweep mechanism, 2nd observation §1.2 promotion-tracker) are rewritten.

---

## §8 Hand-off to PAPER.md

Verdict: **KILLED preempt-structural (F#669-family clause)**. Findings to register: **F#669 15th reuse** + **4th F#682-child F#669 instance** + **2nd observation of schema-repair-reveals-F#669 meta-pattern** (F#776 promotion to 2nd obs; canonicalization at 3rd obs per `mem-pattern-triple-fire`).
