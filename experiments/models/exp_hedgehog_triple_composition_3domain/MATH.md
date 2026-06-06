# MATH.md — exp_hedgehog_triple_composition_3domain

**Disposition:** PREEMPT-KILL (no measurement) under F#669-family clause.
**Compound failure mode:** F#669-cascade (3-parent cardinality, NEW within F#669 family — highest dep-cardinality observed) + F#666-pure schema-defect (K#1883 + K#1884 proxy-only, no target-metric KC pair).
**Date:** 2026-04-25 · drain-window iter ~97 · researcher → reviewer (combined).
**Doom-loop note:** 5th preempt-KILL in cluster (post-HALT-override smoke break ~58/61/64/67/70/92/94 inserted real research progress). NEW substantive sub-form: 3-deps-all-PROVISIONAL is highest dep-cardinality F#669 instance observed (prior 17 reuses had cardinality 1; F#781 introduced cardinality 2; this is 1st cardinality 3).

## §0 Skill attestation (carve-out)

Per F#669-family carve-out: no MLX code is executed, no model is loaded, no adapter
is trained, no composition is performed. Skills `/mlx-dev` and `/fast-mlx`
(PLAN.md Part 2) are **not** invoked because there is no MLX surface in
`run_experiment.py` — the script writes a diagnostic `results.json` and exits
rc=0. Identical pattern to F#775 (rank_ablation iter ~48), F#777 (jepa_scale_sweep
iter ~50), F#779 (cross_axis_interference iter ~53), F#781 (pair_composition iter ~56).

## §1 Hypothesis (verbatim from DB)

> Hedgehog triple domain composition: JS + Python + SQL N=3.
> Simpler N=3 version testing 3-domain additivity before tackling N=4+.

KCs pre-registered (both, untouched since 2026-04-23):

- K#1883: Triple domain composition drops any single domain > 5pp vs isolated.
- K#1884: Per-layer cos of triple composition < 0.60.

## §2 Why this experiment cannot run — F#669 cascade (3-parent cardinality NEW)

`exp_hedgehog_triple_composition_3domain` `depends_on:`
- `exp_hedgehog_adapter_python_domain` (P=1 micro)
- `exp_hedgehog_adapter_sql_domain` (P=2 micro)
- `exp_hedgehog_domain_adapter_js` (P=1 micro)

**All three** parents must be SUPPORTED for K#1883 to be measurable: the kill
criterion explicitly compares "triple-composed JS+Python+SQL" against "any single
domain isolated." "Isolated" = trained-and-evaluated single-domain adapter
accuracy. The triple is the sum-of-deltas (or DARE-equivalent) of the three
trained adapters.

Parent state (verified 2026-04-25 via `experiment get` + filesystem `ls`):

| Parent | Status | Phases done | KC results | adapters/ subdir |
|--------|--------|-------------|-----------|------------------|
| `exp_hedgehog_adapter_python_domain` | provisional | design only | K#1844-K#1845 untested ([·]) | NOT FOUND |
| `exp_hedgehog_adapter_sql_domain` | provisional | design only | K#1868-K#1869 untested ([?]) | NOT FOUND |
| `exp_hedgehog_domain_adapter_js` | provisional | design only | K#1790-K#1793 untested ([?]) | NOT FOUND |

**Therefore none of the three trained domain-adapter weights exist.** K#1883 requires:
1. Apply isolated python adapter to python prompts → `acc_py_iso`.
2. Apply isolated sql adapter to sql prompts → `acc_sql_iso`.
3. Apply isolated js adapter to js prompts → `acc_js_iso`.
4. Compose all three; apply triple to python prompts → `acc_py_triple`.
5. Compose all three; apply triple to sql prompts → `acc_sql_triple`.
6. Compose all three; apply triple to js prompts → `acc_js_triple`.
7. Test `(acc_py_iso − acc_py_triple) > 5pp OR (acc_sql_iso − acc_sql_triple) > 5pp OR (acc_js_iso − acc_js_triple) > 5pp`.

Steps 1-6 are unconstructable: the adapters being composed do not exist as trained
artifacts. Confirmed by `ls`:
- `micro/models/exp_hedgehog_adapter_python_domain/adapters/` — No such file or directory.
- `micro/models/exp_hedgehog_adapter_sql_domain/adapters/` — No such file or directory.
- `micro/models/exp_hedgehog_domain_adapter_js/adapters/` — No such file or directory.

K#1884 requires per-layer cos-sim of the triple adapter against a teacher driven
by 3-prompt concatenation. Same dependency on trained weights — unconstructable.

This is the canonical F#669 structure (Finding #669): **child KC measurement
transitively requires a parent target that is target-unverified**. F#669 has been
reused 17 times prior; this is the 18th reuse. Notable structural distinction:

- **3-parent F#669 cardinality** — first observation, highest dep-cardinality
  ever recorded. Prior 17 F#669 reuses had cardinality 1 (single PROVISIONAL
  parent); F#781 (pair_composition) introduced cardinality 2; this is cardinality 3.
- 4th Hedgehog-cluster F#669 (after F#775 rank_ablation post-F#770; F#779
  cross_axis pre-F#770; F#781 pair_composition pre-F#770).
- 3rd Hedgehog-cluster pre-F#770-repair F#669 (after F#779, F#781).

## §3 Why this experiment cannot run — F#666-pure schema-defect (compound)

K#1883 and K#1884 are **both proxy KCs** with no target-pair counterpart:

- K#1883: "drops any single domain > 5pp vs isolated" — proxy for behavioral
  retention, but the "domain accuracy" is itself a benchmark proxy. Without a
  behavioral judge ("does the user perceive the response as JS-correct AND
  Python-correct AND SQL-correct"), K#1883 is doubly proxy. No target KC pairs
  accuracy-delta with behavioral fidelity.
- K#1884: "per-layer cos of triple vs 3-prompt teacher < 0.60" — pure structural
  proxy. No paired target KC for behavioral parity.

This is F#666-pure (proxy-only KC set, no target pair) layered on top of F#669,
identical to the cross_axis_interference / pair_composition compound mode
(F#779/F#780/F#781/F#782) but with **3 parents** instead of 1 or 2. F#780 sub-axis
2nd same-cluster instance (F#782) was pair_composition; this advances the count
to **3rd same-cluster instance (Hedgehog→Hedgehog→Hedgehog)** — canonicalization
threshold reached at 3/3 same-cluster but cross-cluster canonicalization remains
pending.

## §4 Composition antipattern (relevant but secondary)

Even if all three parents reached SUPPORTED, the triple composition itself would
face the F#302/F#334 composition-residual ceiling (`tau ≈ 0.48` per F#752
quantitative replication). K#1883's 5pp threshold and K#1884's 0.60 cos threshold
are not preregistered against that ceiling — they would need re-derivation
post-F#752. Worse, F#752 measured tau on N=2 composition; N=3 composition residual
would be smaller still (compounding orthogonality loss). This is a **secondary**
structural concern (composition-math antipattern adjacent), not the immediate
blocker. The immediate blocker is F#669+F#666 above.

## §5 Why no data was collected (carve-out justification)

Surrogate adapters (random-init all three) measure noise, not triple-composition
behavior. Identity-adapter substitution measures the base model, not the triple.
Either shortcut would violate F#666 (proxy substitution as target). Skipping
K#1883 and returning hardcoded `pass:True` would violate the F#666 hardcoded-pass
antipattern. Reducing the 5pp / 0.60 thresholds to make measurement meaningful
would violate KC-modification-post-hoc (guardrail 1010 verdict-consistency #5).

The correct path: wait for all three parents to reach SUPPORTED via their `_impl`
counterparts. Only `exp_hedgehog_domain_adapter_js_impl` exists as a follow-up
(P=1 micro, blocked-by `exp_hedgehog_domain_adapter_js`). No `_impl` for python
or sql exists. Even after `_impl` lands for one domain, the other two remain
PROVISIONAL — measurement unconstructable.

## §6 Predicted findings

- **F#NEW1 — F#669 18th reuse, 3-parent F#669 cardinality 1st observation
  (highest dep-cardinality ever), 4th Hedgehog-cluster F#669, 3rd Hedgehog-cluster
  pre-F#770-repair compound F#666+F#669.**
- **F#NEW2 — F#780 sub-axis 3rd-instance same-cluster** (Hedgehog→Hedgehog→Hedgehog,
  3/3 same-cluster canonicalization threshold reached). Cross-cluster
  canonicalization (3-cluster diversity) remains pending — same-cluster 3-of-3
  is full saturation within Hedgehog cluster only.

## §7 Drain-stall reality (continuation)

Per F#781/F#782 reviewer iter ~56 + analyst iter ~57 HALT_ESCALATION addendum
recommendations: politeness_impl, refactor_impl, conciseness_impl have all been
processed via HALT-override smoke (researcher iter ~58/61/64/67/70/92/94),
yielding F#783, F#784, F#785, F#786, F#787, F#788, F#789, F#790. These provided
genuine drain progress (real PROVISIONAL findings, not preempt-KILL).

This iter ~97 returns to the cascade-preemption track because triple_composition
was the next analyst-recommended P=2 micro structurally-distinct entry. Pre-flight
filesystem check confirms F#669-family blocker holds (no adapter checkpoints in
any of 3 deps).

After this preempt: P≤2 open=5 remaining (memento_replication, class_composition_full_impl,
politeness_full, refactor_full, formality_full). All P=1 macro `_impl` follow-ups
or replication; all exceed researcher cap. Drain advances via macro-orchestration,
not researcher-cap claims.

## §8 Cross-references

- F#669 (governing — 18th reuse, 3-parent cardinality 1st obs).
- F#666 (compound schema-defect; F#666-pure both-proxy variant).
- F#770/F#771 (~13 P≤2 schema-defect cohort; this entry NOT in cohort per F#771).
- F#775/F#777 (post-F#770-repair F#669 reveals).
- F#776/F#778 (schema-repair-reveals-F#669 cross-cluster meta-pattern).
- F#779/F#780 (pre-F#770-repair compound F#666+F#669; 1st observation).
- F#781/F#782 (pair_composition; 2-parent F#669 cardinality 1st obs; 2nd
  same-cluster F#780 sub-axis instance).
- F#683 (Hedgehog parent finding; this entry depends transitively).
- F#752 (composition residual τ≈0.48; secondary antipattern §4).

## §9 No code, no measurement, no model load

Per F#669 carve-out: `run_experiment.py` writes `results.json` with KILLED verdict
and exits rc=0. No `mlx_lm.load`, no `nn.value_and_grad`, no `mx.eval`, no
adapter composition, no benchmark eval. KCs #1883 and #1884 byte-for-byte
identical to 2026-04-23 DB record (no post-hoc modification).
