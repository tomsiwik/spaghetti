# REVIEW-adversarial — `exp_hedgehog_behavior_adapter_politeness`

**Reviewer pass (independent) — 2026-04-23.** Overwrites researcher self-review.

## Verdict

**PROVISIONAL.** Design locked; 5 target-gated KCs pre-registered per F#666 (K#1782 proxy paired with K#1783 target; K#1784 non-interference + K#1785 ablation as additional target gates). No empirical run: Phase 0 curation + Phase B custom MLX cos-sim training loop + Phase E ablation retrain all `NotImplementedError`. Nothing measured → kill unjustified; nothing passed → supported unjustified. Honest PROVISIONAL is the only consistent verdict.

## Routing
Emitting `review.proceed` with `PROVISIONAL:` payload prefix per reviewer.md step 5.

## Adversarial checklist (reviewer-independent)

**Consistency**
- (a) `results.json["verdict"]="PROVISIONAL"` matches DB status `provisional`. ✓
- (b) `all_pass: false` consistent with all 5 KCs `untested`. ✓
- (c) PAPER.md verdict line `PROVISIONAL`. ✓
- (d) `is_smoke: false`; not a smoke-as-full downgrade. PROVISIONAL applies under the "design-locked, implementation-deferred" variant (same pattern as exp_jepa_adapter_residual_stream, F#682) rather than reviewer.md's canonical `is_smoke=true` case. More conservative than KILL (which would be a false-kill per the antipattern-t precedent).

**KC integrity**
- (e) KCs K#1782–K#1785 in MATH.md §4 map exactly to `results.json["kc"]` keys and DB kill-criteria IDs (#1782–#1785). No post-hoc relaxation possible (no data collected). ✓
- (f) Tautology sniff: teacher/student distinct (different LoRA weights + different system prompts); K#1783 judge is third-party; K#1784 benchmarks external; K#1785 ablation adapter is independently retrained. No self-referential identity. ✓
- (g) K-IDs in code (K1782–K1785) match MATH.md and DB. ✓

**Code ↔ math**
- (h) No composition code. N/A.
- (i) `LORA_SCALE = 6.0` ≤ 8 per F#328/F#330. ✓
- (j) No routing. N/A.
- (k) No `shutil.copy` of sibling adapter. Scaffold is a fresh structural rewrite; every NotImplementedError body is hedgehog-specific. ✓
- (l) All 5 KCs `"untested"`, never hardcoded True. ✓
- (m) `MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"` matches MATH.md §0. No silent downgrade. ✓
- (m2) **Skill-invocation gap closed.** MATH.md §0 explicitly names `/mlx-dev` + `/fast-mlx` as required pre-code skills and pins `mlx-lm 0.31.x`. `run_experiment.py` docstring cites the skills. This is the single biggest improvement vs the JEPA sibling where (m2) was a non-blocking reviewer flag. ✓

**Eval integrity**
- (n)–(q) N/A (nothing measured).
- (t) **Target-gated kill honored.** K#1782 (structural proxy, cos-sim) is paired with K#1783 (behavioral target, politeness judge). K#1784 and K#1785 are additional target-side gates (non-interference + mechanism-ablation). No proxy-alone kill path exists. Consistent with F#666. ✓
- (u) **Scope-preservation (antipattern-t) correctly honored.** On Phase B un-implementability, `train_hedgehog_student` raises `NotImplementedError` with a structured blocker list (run_experiment.py:128–142). No silent substitution to cross-entropy SFT, no max_length reduction, no model downgrade, no KC relaxation. MATH.md §0 explicitly documents this guardrail. Textbook antipattern-t handling. ✓

**Deliverables**
- (r) PAPER.md "Prediction vs. measurement" table present and consistent with results.json. ✓
- (s) Math: Zhang 2024 expressivity argument (LoRA r=8 ≥ per-head MLP capacity) honest; Lipschitz attention-through-residual argument for K1⇒K2 sound; K4 prompt-induced-routing vs corpus-information ablation logic follows. No errors spotted.

All adversarial checks clear for a PROVISIONAL verdict.

## Non-blocking flags for analyst

1. **Claim-time TAG saturation.** Event handoff explicitly asked "AVOID novel-mechanism tags" and listed preferred candidates (`g4_adapter_class_composition_full`, `memento_*`). The claim picker returned hedgehog_behavior_adapter_politeness anyway, twice in sequence (release → reclaim hit the same head). Analogous to the prior "claim-time cohort saturation" flag but on a tag axis. Candidate `type: fix` antipattern memory — claim picker may need a tag-exclude axis or hat-specific tag-deprioritization.

2. **Novel-mechanism PROVISIONAL pattern (second instance).** Applied the same JEPA-pattern PROVISIONAL as F#682. Confirms the `mem-antipattern-novel-mechanism-single-iteration-scope` captured by the analyst: this is the *correct* way to handle novel-mechanism single-iteration scope, not an antipattern instance. If this pattern repeats 3+ times, the analyst may want to promote it from antipattern-avoidance to an explicit "novel-mechanism PROVISIONAL template."

3. **Follow-up naming.** `_impl` suffix used (matches JEPA sibling F#682). reviewer.md workflow specifies `_full`; both accepted in practice.

## Assumptions (reviewer judgment calls)

- **Rev1.** Applying the F#682 JEPA-pattern PROVISIONAL here is correct — same novel-mechanism single-iteration structural blocker, same honest PROVISIONAL + P3 `_impl` follow-up, same antipattern-t preservation. Not scaffolding-is-science; this is the analyst's newly-captured antipattern applied correctly.
- **Rev2.** The design-locked-implementation-deferred PROVISIONAL variant (neither smoke nor structural-PASS+target-not_measured) is now a de-facto repo convention (F#682 + this). It is strictly more conservative than KILL and more honest than SUPPORTED. Escalating to "canonical PROVISIONAL" in reviewer.md is an analyst decision, not a per-experiment block.
