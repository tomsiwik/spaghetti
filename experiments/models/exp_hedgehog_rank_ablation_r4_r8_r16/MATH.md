# MATH.md — exp_hedgehog_rank_ablation_r4_r8_r16 (PREEMPT-KILL)

## Verdict (pre-measurement)

**KILLED — preempt-structural** under the **F#669-family clause** (preempt-child-KCs-require-parent-target-claim-unverified). Schema-repaired at iter ~36 (F#666 paired target KCs #1980/#1981/#1982 added) but the repair revealed a deeper blocker: every paired KC threshold references parent measurements that the parent has not produced. Parent `exp_hedgehog_behavior_adapter_politeness` is `provisional` (F#683); Phase B / target KCs (K1782 cos, K1783 ΔJudge, K1784 non-interference, K1785 ablation) are all `untested` per parent's design-only filing.

This is the **1st Hedgehog-cluster F#669 reuse** (prior 7 Hedgehog preempts — F#714, F#716, F#720, F#721, F#722, F#723, F#755, F#756 — were F#666-pure-standalone or §5 sub-types; F#770 schema-repair upgraded the diagnosis), and the **14th overall F#669 reuse** (after F#741).

No measurement performed. No `_impl` companion (per F#669-family clause: unblock is parent-external).

---

## §0 Platform / skills / model pins

Included for completeness even though no platform code is executed.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). **Not invoked** — no MLX code written; honest disclosure per reviewer checklist (m2). The F#669-family clause exempts the skill-attestation gate when no MLX training-loop code lands.
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: N/A — would have been LoRA on `v_proj`/`o_proj` per parent's Hedgehog Stage-1 recipe at ranks {4, 8, 16}.
- Parent dependency: `exp_hedgehog_behavior_adapter_politeness` (status `provisional`, F#683 — design-only scaffold, Phase B `NotImplementedError`).
- Sibling precedents (Hedgehog cluster, all preempt-KILLed but on different clauses): F#714, F#716, F#720, F#721, F#722, F#723, F#755, F#756 (all F#666-pure standalone or §5 sub-types). This is the **1st Hedgehog F#669 reuse** — a different super-family.
- Datasets: N/A — no neutral-prompt politeness eval, no MMLU subset, no HumanEval load.

---

## §1 Preempt-KILL theorem (F#669-family transitivity)

**Theorem (inter-experiment target unverifiability — rank-sweep variant).** Let `C` denote child experiment `exp_hedgehog_rank_ablation_r4_r8_r16` with kill criteria K = {K1980, K1981, K1982} (the 3 F#770-schema-repair target-paired KCs; the original proxy KCs K1852/K1853 are now non-load-bearing per F#666 — kill requires both proxy AND target FAIL). Let `P` denote parent `exp_hedgehog_behavior_adapter_politeness`.

For each child KC, the threshold reference resolves to a parent measurement:

- **K1980** (`r=4 ΔJudge < (r=8 ΔJudge − 5pp absolute)`): `r=8 ΔJudge` is parent's K1783 measurement on neutral prompts. Parent K1783 = `untested` ⇒ RHS = `(NaN − 5pp)` = NaN. Child K1980 evaluates `r=4 ΔJudge < NaN` — unidentifiable.
- **K1981** (`r=16 ΔJudge < (r=8 ΔJudge + 3pp absolute)`): same RHS structure. Parent K1783 untested ⇒ RHS = NaN. Unidentifiable.
- **K1982** (cross-rank non-interference: factual QA / HumanEval drop > 3pp vs base Gemma 4 E4B for ANY r ∈ {4, 8, 16}): mirrors parent K1784 design but each rank's child trained-adapter would itself be a forward of the parent's untrained Hedgehog-distillation mechanism. Parent's Hedgehog distillation loop has no executable MLX implementation (parent F#683 PROVISIONAL, Phase B NotImplementedError). Therefore `acc_r(eval) − acc_base(eval)` for each `r` requires a *trained Hedgehog adapter* that does not exist on this platform. Without such an adapter, the per-rank drop is undefined for r ∈ {4, 8, 16} — unidentifiable.

If `P.status ∈ {provisional, open}` — i.e. K1783 and K1784 are untested — then:

- **K1980**: r=4 ΔJudge comparison undefined (NaN RHS). Cannot resolve to PASS or FAIL. Unidentifiable.
- **K1981**: r=16 ΔJudge comparison undefined (NaN RHS). Unidentifiable.
- **K1982**: per-rank trained-Hedgehog adapter does not exist; per-rank acc drop undefined. Unidentifiable for all 3 ranks simultaneously.

∴ Testing K = {K1980, K1981, K1982} while `P.status ≠ supported|proven` produces unidentifiable samples on all three. **QED.**

### §1.1 F#666 gating (post-schema-repair)

The schema-repair at iter ~36 (per F#770 cohort fix) added K1980, K1981, K1982 as target-paired KCs to satisfy F#666-discipline. Post-repair the KC set IS F#666-compliant *in principle*:

- K1852 (proxy, cos-sim) ↔ K1980 (target, ΔJudge) — paired.
- K1853 (proxy, cos-sim improvement) ↔ K1981 (target, ΔJudge improvement) — paired.
- K1982 — standalone target (non-interference) mirroring parent K1784.

But **F#666-compliant pre-reg ≠ F#666-resolvable verdict**. F#666 gates the *KC structure*; F#669 gates the *measurability of the threshold*. The schema-repair fixed the structure (proxy without target) but not the measurability (target threshold references a parent measurement that does not exist). F#769/F#770 governs this transition — the schema-repair is **necessary but not sufficient**.

### §1.2 Schema-repair-reveals-F#669 meta-pattern (1st instance)

This is the **1st observation** of the meta-pattern: an F#770-cohort schema-repair (adding paired target KCs to satisfy F#666) **exposes** a previously-masked F#669 cascade in cluster experiments where the parent is PROVISIONAL not SUPPORTED. Mechanism:

- Pre-repair: child has F#666-pure standalone defect → preempt-KILLed under F#666-pure clause (e.g. F#714, F#716 for Hedgehog cluster).
- Post-repair: child has F#666-compliant pair → F#666-pure preempt no longer fires → but the new target KC's threshold references parent measurement → F#669 cascade fires instead.
- Net: the diagnosis migrates from F#666-pure-standalone (KC-design-defect) to F#669-cascade (parent-cascade-defect). Identical disposition (preempt-KILL); different governing clause; different unblock path (re-pre-reg vs parent-completion).

Predicted next instances: `exp_hedgehog_cross_axis_interference` (same parent politeness PROVISIONAL; KC #1859 is currently F#666-pure but if F#770 schema-repairs add a paired target, F#669 will fire), `exp_jepa_scale_sweep_5m_15m_50m` (parent residual-stream PROVISIONAL; F#770-repair already added K1988/K1989 target-paired KCs referencing parent K1768 — the 2nd instance is already constructed in the DB awaiting claim).

Per `mem-pattern-triple-fire`, the meta-pattern reaches promotion at 3rd observation. Current count: **1st** (this).

---

## §2 Prior art

- **F#669** (2026-04-19) — defining precedent for preempt-KILL on target-unverified parent. 13 reuses prior to this; this is the 14th reuse.
- **F#683** (2026-04-23) — parent PROVISIONAL: Hedgehog politeness adapter, design-only, Phase B `NotImplementedError`, all 4 KCs `untested`.
- **F#666** (target-gated KC discipline) — governs the schema-repair pre-condition; satisfied post-repair.
- **F#770** (2026-04-25) — schema-defect cohort enumeration + bulk repair driving the iter ~36 fix that added K1980/K1981/K1982 to this experiment's KC set.
- **F#771** (2026-04-25) — F#770 promoted to formal status; documented the over-scope tendency in cohort findings.
- **F#714/F#716/F#720/F#721/F#722/F#723/F#755/F#756** — prior 8 Hedgehog-cluster preempt-KILLs, all F#666-pure-standalone or §5 sub-types. This is the **1st Hedgehog-cluster F#669** instance.
- **Sibling precedent (cross-cluster F#669-family canonicalization)**: F#699, F#737, F#738, F#739, F#758 (MEMENTO cluster); F#740, F#741 (Pierre-serving cluster). Cross-cluster reuse pattern attested at ≥9 reuses; this advances to 14.

---

## §3 Predictions (registered, not measured)

| KC    | Claim                                                                                                | Kind   | Measurement status                |
| ----- | ---------------------------------------------------------------------------------------------------- | ------ | --------------------------------- |
| K1852 | r=4 cos-sim < 80% of r=8 cos-sim (rank too low for routing capture)                                  | proxy  | untested (preempt-blocked, F#669) |
| K1853 | r=16 achieves < 5pp improvement over r=8 (diminishing returns)                                       | proxy  | untested (preempt-blocked, F#669) |
| K1980 | r=4 ΔJudge < (r=8 ΔJudge − 5pp) on n=100 neutral prompts using parent K2 rubric                      | target | untested (preempt-blocked, F#669) |
| K1981 | r=16 ΔJudge < (r=8 ΔJudge + 3pp) on n=100 neutral prompts using parent K2 rubric                     | target | untested (preempt-blocked, F#669) |
| K1982 | For ANY r ∈ {4,8,16}: MMLU-subset acc drop > 3pp OR HumanEval pass@1 drop > 3pp vs base Gemma 4 E4B  | target | untested (preempt-blocked, F#669) |

KC semantics note: K1980/K1981 are F#770-schema-repair targets added 2026-04-25; K1982 is the F#770-schema-repair non-interference target. All three resolve to comparisons against parent measurements (K1783, K1784) that the parent has not produced.

---

## §4 Unblock condition

Re-claimable when **all three** of:

1. Parent `exp_hedgehog_behavior_adapter_politeness` reaches `status=supported|proven` via its `_impl` companion (`exp_hedgehog_behavior_adapter_politeness_impl`, P=1, currently `open`), with K1782 + K1783 + K1784 SUPPORTED on the politeness training/eval split.
2. Parent's `_impl` has produced a numerical value for `r=8 ΔJudge` (parent K1783) that can serve as the RHS reference for child K1980/K1981.
3. Parent's `_impl` has produced an executable trained Hedgehog adapter (or training script) that this child can re-instantiate at r ∈ {4, 16} — the per-rank training loop must be a function of rank only, all other hyperparameters inherited from parent.

**No KC-augmentation needed** at re-claim: K1980/K1981/K1982 are already F#666-compliant. The parent-completion is the unblock. A scope-reduced rank-sweep on a non-Hedgehog mechanism (e.g. plain LoRA at the same ranks) would substitute the algorithm — antipattern-t risk; explicitly NOT an acceptable unblock path.

Alternative scope-reduction to PPL-only rank comparison (as a stand-in for ΔJudge) would replace the behavioral target with a weakly-correlated proxy (F#666 r≈0.08 between PPL and task quality) — antipattern-i. Avoided.

---

## §5 Follow-up

No `_impl` companion filed — preempt-structural KILL is self-contained per the F#669-family clause + reviewer.md §5. Unblock is parent-external: parent's existing `exp_hedgehog_behavior_adapter_politeness_impl` (P=1, `open`) is the unblock gate; this child's re-claim is downstream of parent's `_impl` SUPPORTED transition.

---

## §6 Scope integrity

No silent objective swap (antipattern-t): this scaffold does NOT attempt:
- Substituting plain LoRA for Hedgehog distillation as a rank-sweep mechanism — antipattern-t (substitutes algorithm).
- Substituting PPL parity for ΔJudge as the behavioral target — F#666 r≈0.08, antipattern-i (KC measures wrong object).
- Loading a prior Hedgehog checkpoint from a different base or domain as a Gemma-4 stand-in — antipattern-m (proxy substitution).
- Reading parent's eventual K1783 measurement post-hoc when parent reports `not measured` — antipattern-l (hardcoded `pass` / fabricated measurement).
- Reducing child's rank set from {4, 8, 16} to {8} only and labeling as scope-completion — antipattern-u (silent scope reduction).

All five shortcuts would replace the actual rank-sweep mechanism the KCs measure with a proxy or partial measurement.

---

## §7 Anti-pattern scan

- Composition-math: N/A (no composition).
- LORA_SCALE: N/A (no LoRA training executed).
- shutil.copy: N/A (no code).
- Hardcoded `"pass": True`: N/A (no code; `all_pass: false` written).
- Eval truncation producing base=0%: N/A (no eval).
- Proxy-model substitution: explicitly rejected in §6.
- KC measures wrong object: KCs correctly identify cos-sim proxy (K1852/K1853), behavioral target (K1980/K1981 ΔJudge), and non-interference target (K1982 MMLU/HumanEval drop) — KCs are well-formed; the parent measurement they reference is the missing object.
- N=smoke reported as full: N/A (no N; `is_smoke: false`).
- Tautological routing: N/A (no routing).
- Thinking-mode truncation: N/A (no eval).
- File-existence cache: N/A (no code).
- Copy-paste scaffolding: scaffold derived from `exp_memento_streaming_inference` (5th MEMENTO F#669 child, closest structural sibling) but variant-specific sections (rank-sweep mechanism, schema-repair-reveals-F#669 meta-pattern §1.2, 1st-Hedgehog-cluster F#669 sub-classification) are rewritten.

---

## §8 Hand-off to PAPER.md

Verdict: **KILLED preempt-structural (F#669-family clause)**. Finding to register: **F#669 14th reuse** + **1st Hedgehog-cluster F#669 instance** + **1st observation of schema-repair-reveals-F#669 meta-pattern** (candidate for promotion at 3rd obs per `mem-pattern-triple-fire`).
