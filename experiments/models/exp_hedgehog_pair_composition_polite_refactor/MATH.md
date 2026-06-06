# MATH.md — exp_hedgehog_pair_composition_polite_refactor

**Disposition:** PREEMPT-KILL (no measurement) under F#669-family clause (17th F#669 reuse).
**Compound failure mode:** F#669-cascade (2-parent cardinality, NEW within F#780 sub-axis) + F#666-pure schema-defect (both KCs proxy-only, never F#770-repaired).
**Date:** 2026-04-25 · drain-window iter ~55 · researcher hat
**Doom-loop note:** 4th consecutive researcher preempt-KILL (~47/~49/~52/~55). Substantively distinct sub-mechanism: 2-parent F#669 cardinality NEVER previously observed; F#780 sub-axis advances 1/3 → 2/3 toward canonicalization. HALT_ESCALATION addendum requested at analyst pass.

## §0 Skill attestation (carve-out)

Per F#669-family carve-out: no MLX code is executed, no model is loaded, no adapter
is trained, no composition is performed. Skills `/mlx-dev` and `/fast-mlx`
(PLAN.md Part 2) are **not** invoked because there is no MLX surface in
`run_experiment.py` — the script writes a diagnostic `results.json` and exits with
rc=0. Identical pattern to F#775 (rank_ablation iter ~48), F#777 (jepa_scale_sweep
iter ~50), F#779 (cross_axis_interference iter ~53).

## §1 Hypothesis (verbatim from DB)

> Simpler N=2 version of exp_hedgehog_composition_polite_refactor_js. Tests
> pairwise additivity before tackling N=3/4.

KCs pre-registered (both, untouched since 2026-04-23):

- K#1846: Pair-composed polite+refactor drops either axis > 5pp vs isolated.
- K#1847: Per-layer cos of pair composition vs 2-prompt concatenated teacher < 0.65.

## §2 Why this experiment cannot run — F#669 cascade (2-parent cardinality)

`exp_hedgehog_pair_composition_polite_refactor` `depends_on:`
- `exp_hedgehog_behavior_adapter_politeness`
- `exp_hedgehog_procedural_adapter_refactor`

**Both** parents must be SUPPORTED for K#1846 to be measurable: the kill criterion
explicitly compares "pair-composed polite+refactor" against "either axis isolated."
"Isolated" = trained-and-evaluated single-axis adapter accuracy. The pair is the
sum-of-deltas (or DARE-equivalent) of the two trained adapters.

Parent state (verified 2026-04-25 via `experiment get`):

| Parent | Status | Phases done | KC results |
|--------|--------|-------------|-----------|
| `exp_hedgehog_behavior_adapter_politeness` | provisional | Phase 0 only | K#1782-K#1785 all untested |
| `exp_hedgehog_procedural_adapter_refactor` | provisional | design only | K#1786-K#1789 all untested |

**Therefore neither trained polite-adapter weights nor trained refactor-adapter
weights exist.** K#1846 requires:
1. Apply isolated polite adapter to politeness prompts → `acc_polite_iso`.
2. Apply isolated refactor adapter to refactor prompts → `acc_refactor_iso`.
3. Compose both adapters; apply pair to politeness prompts → `acc_polite_pair`.
4. Compose both adapters; apply pair to refactor prompts → `acc_refactor_pair`.
5. Test `(acc_polite_iso − acc_polite_pair) > 5pp OR (acc_refactor_iso − acc_refactor_pair) > 5pp`.

Steps 1-4 are unconstructable: the adapters being composed do not exist as trained
artifacts.

K#1847 requires per-layer cos-sim measurement between the pair adapter's attention
output and a teacher driven by 2-prompt concatenation. Same dependency on trained
weights — unconstructable.

This is the canonical F#669 structure (Finding #669): **child KC measurement
transitively requires a parent target that is target-unverified**. F#669 has been
reused 16 times prior; this is the 17th reuse. Notable structural distinction:

- **2-parent F#669 cardinality** — first observation. Prior F#669 reuses had
  cardinality 1 (single PROVISIONAL parent) or were post-F#770-repair where the
  parent was the impl-counterpart of the child. This is the first child whose
  measurement is gated on TWO independent PROVISIONAL parents simultaneously.
- 3rd Hedgehog-cluster F#669 (after F#775 post-F#770 + F#779 pre-F#770).
- 2nd Hedgehog-cluster pre-F#770-repair F#669 (after F#779).

## §3 Why this experiment cannot run — F#666-pure schema-defect (compound)

K#1846 and K#1847 are **both proxy KCs** with no target-pair counterpart:

- K#1846: "drops either axis > 5pp vs isolated" — proxy for behavioral retention,
  but the "axis accuracy" is itself a benchmark proxy. Without a behavioral judge
  (e.g. "does the user perceive the response as polite AND refactored"), K#1846
  is doubly proxy. No target KC pairs accuracy-delta with behavioral fidelity.
- K#1847: "per-layer cos of pair vs 2-prompt teacher < 0.65" — pure structural
  proxy. No paired target KC for behavioral parity (e.g. judge agreement on
  polite-and-refactored output between pair-composed and 2-prompt concat).

This is F#666-pure (proxy-only KC set, no target pair) layered on top of F#669,
identical to the cross_axis_interference compound mode (F#779/F#780) but with
**2 KCs** instead of 1. F#780 sub-axis 1st observation had 1 KC (cross_axis);
this is 2nd observation with 2 KCs and 2 parents.

## §4 Composition antipattern (relevant but secondary)

Even if both parents reached SUPPORTED, the pair composition itself would face the
F#302/F#334 composition-residual ceiling (`tau ≈ 0.48` per F#752 quantitative
replication). K#1846's 5pp threshold and K#1847's 0.65 cos threshold are not
preregistered against that ceiling — they would need re-derivation post-F#752.
This is a **secondary** structural concern (composition-math antipattern adjacent),
not the immediate blocker. The immediate blocker is F#669+F#666 above.

## §5 Why no data was collected (carve-out justification)

Surrogate adapter (random-init) measures noise, not pair-composition behavior.
Identity-adapter substitution measures the base model, not the pair. Either
shortcut would violate F#666 (proxy substitution as target). Skipping K#1846 and
returning hardcoded `pass:True` would violate the F#666 hardcoded-pass antipattern.
Reducing the 5pp / 0.65 thresholds to make measurement meaningful would violate
KC-modification-post-hoc (guardrail 1010 verdict-consistency #5).

The correct path: wait for both parents to reach SUPPORTED via their `_impl`
counterparts (`exp_hedgehog_behavior_adapter_politeness_impl` 4-6h macro;
`exp_hedgehog_procedural_adapter_refactor_impl` 4-6h micro-macro). Both exceed
the 90-min researcher-hat cap → orchestrator-scope unblock required (see §7).

## §6 Predicted findings

- F#NEW1 — **F#669 17th reuse, 2-parent F#669 cardinality 1st observation, 3rd
  Hedgehog-cluster F#669, 2nd Hedgehog-cluster pre-F#770-repair compound F#666+F#669.**
- F#NEW2 — **F#780 sub-axis 2nd-instance (1/3 → 2/3 toward canonicalization, BUT
  same-cluster Hedgehog→Hedgehog).** Cross-cluster canonicalization (3-cluster)
  remains pending; same-cluster 2-of-3 is partial confirmation only. Distinct from
  F#776/F#778 schema-repair-reveals-F#669 path (different diagnosis trajectory).

## §7 Drain-stall escalation (requires HALT_ESCALATION addendum)

This is the **4th consecutive researcher preempt-KILL** (iters ~47, ~49, ~52, ~55).
Per `mem-pattern-triple-fire` 3-instance promotion threshold, the preempt-KILL
pattern itself has now CANONICALIZED as the dominant researcher activity in the
current drain window. The drain has not advanced via measurement in 4 iterations.

Drain reality (verified 2026-04-25 via `experiment list -s open`):

| Group | Count | In researcher-cap? | Cascade-preemptable? |
|-------|-------|--------------------|----------------------|
| P=1 macro `_impl` | 5 | NO (>90 min) | NO (legit measurement path) |
| P=1 micro `_impl` (refactor) | 1 | NO (4-6h budget) | NO (legit measurement path) |
| P=1 micro `_impl` (rdt KV-cache) | 1 | NO (2h budget) | NO (parent scope-deferred to child) |
| P=1 micro composition (cascade) | 1 (this) | YES | YES (preempt-KILL) |
| P=2 micro composition (cascade) | 1 (triple_composition) | YES | YES (preempt-KILL) |

**Total in-cap progress paths: 2** (this entry + triple_composition).
Both are F#669-cascade. Once both preempted, **zero** in-cap progress paths remain
at P≤2. The remaining 7 entries are all macro-budget — orchestrator must promote
to release the drain.

Recommended HALT_ESCALATION addendum content (analyst-scope, next iter):

1. **Promote `exp_hedgehog_behavior_adapter_politeness_impl`** (P=1 macro,
   highest-leverage: unblocks F#683-cluster cascade including this entry's first
   parent).
2. **Promote `exp_hedgehog_procedural_adapter_refactor_impl`** (P=1 micro 4-6h:
   unblocks this entry's second parent + procedural-cluster).
3. Defer `_memento`, `_class_composition`, `formality_impl`, `conciseness_impl`
   until politeness+refactor land (F#683 cluster precedence).
4. `rdt_loop_kv_cache_impl` — separate macro-budget escalation (parent
   scope-deferred to child, 2h budget; orchestrator must extend researcher-hat
   cap or assign macro-hat).

## §8 Cross-references

- F#669 (governing — 17th reuse).
- F#666 (compound schema-defect; F#666-pure both-proxy variant).
- F#770/F#771 (~13 P≤2 schema-defect cohort; this entry NOT in cohort per F#771
  audit-correction).
- F#775 (1st Hedgehog-cluster F#669, post-F#770).
- F#776 (1st schema-repair-reveals-F#669 obs).
- F#777 (15th F#669, 4th F#682-child).
- F#778 (2nd schema-repair-reveals-F#669 cross-cluster Hedgehog→JEPA, 2/3).
- F#779 (16th F#669, 1st Hedgehog pre-F#770-repair compound).
- F#780 (compound F#666+F#669 pre-F#770-repair sub-axis 1st obs; THIS entry = 2nd obs same-cluster).
- F#683 (Hedgehog parent finding; this entry's first parent).
- F#752 (composition residual τ≈0.48; secondary antipattern concern §4).

## §9 No code, no measurement, no model load

Per F#669 carve-out: `run_experiment.py` writes `results.json` with KILLED verdict
and exits rc=0. No `mlx_lm.load`, no `nn.value_and_grad`, no `mx.eval`, no
adapter composition, no benchmark eval. KCs #1846 and #1847 byte-for-byte
identical to 2026-04-23 DB record (no post-hoc modification).
