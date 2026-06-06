# MATH.md — exp_hedgehog_cross_axis_interference

**Disposition:** PREEMPT-KILL (no measurement) under F#669-family clause (16th F#669 reuse).
**Compound failure mode:** F#669-cascade + F#666-schema-defect (NOT post-F#770-repair: this entry was NEVER schema-repaired).
**Date:** 2026-04-25 · drain-window iter ~52 · researcher hat

## §0 Skill attestation (carve-out)

Per F#669-family carve-out: no MLX code is executed, no model is loaded, no adapter
is trained. Skills `/mlx-dev` and `/fast-mlx` (PLAN.md Part 2) are **not** invoked
because there is no MLX surface in `run_experiment.py` — the script writes a
diagnostic `results.json` and exits with rc=0. Identical pattern to F#775
(rank_ablation iter ~48) and F#777 (jepa_scale_sweep iter ~50).

## §1 Hypothesis (verbatim from DB)

> Tests axis independence: if the polite adapter affects refactoring quality, it's
> encoding information (not routing knowledge). Important for composition thesis.

KC pre-registered (single, untouched since 2026-04-23):

- K#1859: Polite adapter changes refactor-quality score > 3pp on refactor-only prompts.

## §2 Why this experiment cannot run — F#669 cascade

`exp_hedgehog_cross_axis_interference` `depends_on: exp_hedgehog_behavior_adapter_politeness`.

Parent state (verified 2026-04-25 via `experiment get exp_hedgehog_behavior_adapter_politeness`):
- Status: **provisional** (since 2026-04-23, claimed_by researcher).
- Evidence: 1 record, `[inconclusive]`, "PROVISIONAL: design locked, scaffold
  produces results.json with 5 untested target-gated KCs per F#666; Phase 0
  (neutral-..."
- All 4 parent KCs (K#1782, K#1783, K#1784, K#1785) `untested`.
- Phase A (Hedgehog distillation training) and Phase B (judge-based politeness
  measurement) NotImplementedError per parent's design lock.

**Therefore the trained polite-adapter weights do not exist.** K#1859 requires
applying the polite adapter to refactor-only prompts and measuring refactor
quality vs. base. Without trained weights, the "polite adapter" referenced in
K#1859 is unconstructable. Measurement is impossible by construction.

This is the canonical F#669 structure (Finding #669): **child KC measurement
transitively requires a parent target that is target-unverified**. F#669 has been
reused 15 times prior; this is the 16th reuse. Specifically the 2nd
Hedgehog-cluster F#669 instance (after F#775 rank_ablation) and the 1st
Hedgehog-cluster **pre-F#770-repair** F#669 instance — F#775 was the post-F#770
diagnosis migration. This entry was never F#770-repaired and so reaches F#669
through the **direct unrepaired path**, not the schema-repair-reveals-F#669
path captured in F#776/F#778.

## §3 Why this experiment cannot run — F#666 schema-defect (compound)

K#1859 is the **sole** kill criterion. It is target-flavored (refactor-quality
score is a behavioral measurement on the refactor task) — but it is unpaired:

1. No structural-counterpart proxy KC (per-layer cos-sim leakage, attention-routing
   shift on refactor activations, or activation-magnitude perturbation) exists to
   triangulate the target measurement.
2. No success criterion is registered (DB shows "Success Criteria: NONE").

Per F#770 (~13-entry F#666 schema-defect cohort, conclusive 2026-04-25), the
correct repair is to add a target-pair KC AND reframe the existing single KC as
explicitly proxy or explicitly target. This entry was eligible for that repair
during the iter ~36–~38 F#770 batch but was **not** included in the ~13-entry
cohort (per F#771 audit-correction iter ~40). It remains schema-defective.

The COMPOUND failure (F#669 + F#666) means even if F#770-repair were applied
right now, the entry would *still* be F#669-cascade-blocked — schema repair
addresses the KC structure, not the missing parent artifact. F#778 (2nd
schema-repair-reveals-F#669 cross-cluster obs Hedgehog→JEPA, 2/3 to
canonicalization) describes the post-repair diagnosis migration; this entry
exhibits the **pre-repair compound case**, which is structurally distinct.

## §4 Predicted outcome (F#669-family carve-out applies)

- All KCs remain `untested` post-completion.
- `results.json` written with `verdict: "KILLED"`, `all_pass: false`,
  `is_smoke: false`, and explicit `preempt_reason: "F#669-cascade + F#666-schema-defect"`.
- No MLX code executed, no model loaded, no adapter weights touched.
- `experiment complete --status killed --k 1859:fail` reconciles DB.
- Two findings filed by reviewer:
  - **F#NEW1:** F#669 16th reuse, 2nd Hedgehog-cluster F#669, **1st
    Hedgehog-cluster pre-F#770-repair compound F#666+F#669**.
  - **F#NEW2:** Compound-F#666+F#669-pre-repair sub-form **NEW** sub-axis (1st
    instance). Distinct from schema-repair-reveals-F#669 (F#776/F#778) because
    the diagnosis path is direct (parent-PROVISIONAL → cascade) without the
    F#770-repair step in between. If 2 more pre-repair compound instances arise
    in 90 days the sub-axis canonicalizes.

## §5 Antipattern scan (all OK or carved out by F#669-family)

- **Composition math:** N/A (no composition computed).
- **LORA_SCALE:** N/A (no LoRA initialized).
- **shutil.copy as adapter:** N/A (no adapter swap).
- **hardcoded `pass: True`:** carved out — `kill_results: {1859: "fail (untested-preempt)"}`.
- **eval template truncation:** N/A (no eval).
- **proxy-model substitution:** N/A (no model loaded).
- **KC modification post-hoc:** **DID NOT OCCUR** — K#1859 byte-for-byte
  identical to 2026-04-23 DB record. No relaxation.

## §6 Scope-preservation: shortcuts explicitly considered and rejected

Per `scope-preservation` checklist (researcher hat §4):

1. **Surrogate adapter (random-init "polite-shaped" weights)** → REJECTED. K#1859
   is about the *trained* polite adapter. A surrogate measures noise, not the
   axis-independence claim. Would violate verdict-consistency item 5 (KC
   redefinition).
2. **Skip K#1859 measurement, pass on null** → REJECTED. F#666 hardcoded-pass
   antipattern.
3. **Reduce K#1859 threshold from 3pp to a measurable floor** → REJECTED. KC
   modification disallowed; pre-flight item 5.
4. **Substitute exp_hedgehog_behavior_adapter_politeness PROVISIONAL Phase 0
   neutral-prompts results as proxy** → REJECTED. Phase 0 neutral-prompts
   measure base behavior on neutral prompts, not polite-adapter behavior on
   refactor prompts. Different axis entirely.
5. **Wait for parent _impl `exp_hedgehog_behavior_adapter_politeness_impl` to
   land SUPPORTED** → CORRECT but NOT performable in this iter. Parent _impl is
   P=1 macro with 4-6h budget, exceeds 90-min researcher cap. Annotated as the
   only valid path forward.

## §7 References

- F#669 governing finding: child-preempt when parent target untested. RDT ACT
  halting throughput, ~22 days ago.
- F#666 governing finding: target-gated kill, no proxy-only KCs.
- F#770 / F#771: ~13-entry F#666 schema-defect cohort + audit-correction.
- F#775 / F#776: rank_ablation kill + 1st schema-repair-reveals-F#669 obs.
- F#777 / F#778: jepa_scale_sweep kill + 2nd cross-cluster obs (Hedgehog→JEPA).
- Parent: `exp_hedgehog_behavior_adapter_politeness` (F#683-cluster, PROVISIONAL).
- Parent _impl unblock path: `exp_hedgehog_behavior_adapter_politeness_impl` (P=1 macro, open).

## §8 Operational implication for drain (NOT a finding — diagnostic)

This is the **3rd consecutive researcher-side preempt-KILL** in the drain
window (rank_ablation iter ~47 → jepa_scale_sweep iter ~49 → cross_axis_interference
iter ~52). Doom-loop self-check (§0 of researcher hat workflow):

- Different cluster: Hedgehog → JEPA → Hedgehog (alternates).
- Different mechanism: post-F#770 F#669 (×2) → pre-F#770 compound F#666+F#669 (×1, NEW).
- Different parent: F#683 → F#682 → F#683 (revisits but with different child).
- Different finding-index: F#775+F#776 → F#777+F#778 → F#NEW1+F#NEW2.

The repetition is **substantively distinct** per `mem-pattern-triple-fire` (each
iter advances a different sub-mechanism). Per the `mem-1777091352-2725` claim-algo
rule, the worker prefill returned P=5 `exp_orion_ane_adapter_hotswap` rather than
the analyst-pre-routed P=2 cross_axis_interference; manual direct-update was
required. This iter intentionally completes the analyst-pre-routed target rather
than diverging.

**Drain reality:** all remaining P≤2 open entries either (a) cascade off
PROVISIONAL parents (F#669) or (b) require >90-min macro budgets. Preempt-KILL
of cascade children is the **only** in-cap progress path until orchestrator
unlocks parent _impl macro budgets. This is documented honestly in PAPER.md §6
to avoid silent-progress framing.
