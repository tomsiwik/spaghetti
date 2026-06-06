# MATH.md — exp_g4_gs_random_perm_n25 (preempt-structural KILL)

## Verdict (pre-registered, pre-run)

**KILLED — preempt-structural (F#666-pure-standalone, 9th drain-window instance, 3rd derived-geometric sub-flavor with stability/perturbation semantics, **TAXONOMY-REFACTOR EXECUTION TRIGGER FIRES**, 2nd `proxy-only-lineage-inheritance` candidate sub-variant — watchlist threshold met).** No `run_experiment.py` measurement needed; proof is structural.

## Pre-claim checklist (5-item template-regression + direction-symmetric inter-variant check)

| # | Check | Result |
|---|-------|--------|
| 1 | `experiment finding-get` on every parent cited | ✅ F#160 retrieved |
| 2 | Scan parent *caveats* for vacuous-language + half-KC paired designs | See L4 below; parent had paired K1 (worst/mean ratio) + K2 (abs worst at d=256), both proxy; child stripped K2 |
| 3 | Paired-design-target-inheritance (if parent has paired proxy+target, child must inherit target half) | N/A in standard sense — parent's paired K1/K2 are **both proxy**, no target half existed to strip; this is a NEW sub-variant: paired-PROXY-half-strip (distinct from F#708's paired-target-half-strip) |
| 4 | Stale-advice-vs-current-guardrail check | ✅ Parent F#160 is pre-F#666 SUPPORTED (2026-03-28); child runs under current F#666 regime → tautological. **2nd proxy-only-lineage-inheritance** (after F#710 / parent F#72) — watchlist threshold met. |
| 5 | "Under verified <condition>" degenerate-equivalence-branch check | N/A (single-threshold ratio, not inter-variant) |
| 6 | Direction-symmetric inter-variant-delta check | N/A (single-threshold geometric ratio, not inter-adapter comparison) |

## Setup

**Claim under review:** K1595 — "GS random permutation on Gemma 4 N=25 v_proj gives `worst/mean ≤ 1.5x` removal deviation."

**Parent cited:** F#160 (exp_gs_random_permutation_validation, SUPPORTED 2026-03-28, **pre-F#666**) — its evidence:
> "K1 PASS: permuted expected worst/mean=1.42x (<2.0x). K2 PASS: abs worst 0.446% at d=256 (<1%). Mean preserved (1.00x). CV reduced 53%. Worst/mean ratio 1.56-1.96x → 1.29-1.45x. Zero overhead. 3 configs, 3 seeds, P=5 perms."

Parent KCs were entirely proxy (K1 worst/mean removal-deviation ratio < 2.0x; K2 abs worst removal < 1%). **Zero target-metric KC** (no MMLU-Pro, no behavioral task accuracy, no oracle gap, no downstream generation quality).

**Audit tag:** `audit-2026-04-17` (no `-rerun` suffix → lineage-only per prior analyst convention; does NOT trigger `.audit/RECOVERY_PLAN.md` fix-before-rerun).

**Hygiene:** 2 defects (`success_criteria=[]`, `references=[]`) → below 3+ threshold for `mem-antipattern-prereg-hygiene-multi-defect`. F#666-pure-standalone keys on KC structure independent of hygiene count.

**Proxy flavor classification:** "Worst/mean removal deviation ratio" measures the worst-case-vs-mean magnitude of model-output perturbation when one expert is removed, expressed as a single dimensionless ratio. This is geometric (output-perturbation magnitude under the removal map) but with stability/perturbation semantics distinct from cos-sim (F#700) or pairwise-cos+eff-rank (F#701). Within analyst's planned option-(b) refactor buckets, the natural placement is **derived-geometric (3rd instance)** with a possible sub-bucket distinction "stability/perturbation-magnitude vs static-similarity." Non-load-bearing for this kill.

## Theorem

Let Experiment E have kill criterion `K(θ) := worst(θ)/mean(θ) ≤ 1.5x` on the per-position removal-deviation distribution `{δ_i(θ)}_{i=1..N}` for N=25 v_proj experts on Gemma 4, where `δ_i(θ)` is some scalar measure of the change in model output (per parent F#160's protocol, the absolute deviation per dimension under removal of expert i with random GS permutation). Let Guardrail G be F#666 / 1007: "KILL requires proxy+target both to fail; SUPPORTED requires both to pass." Then any outcome under E violates G:

1. If `worst/mean ≤ 1.5x`: verdict `supported` is tautological per F#666 canonical (a structural-stability ratio claim about removal-perturbation distribution does not entail any downstream behavioral outcome — F#477 Gemma 4 shallow-regime evidence shows N≥5 composition only beats base on 2/5 domains, with no known coupling between worst/mean removal-ratio and task accuracy).
2. If `worst/mean > 1.5x`: verdict `killed` is forbidden per F#666 ("proxy-FAIL + target-absent = finding about the proxy, not a kill").

Therefore E admits no F#666-compliant verdict. **KILL preempt-structural.** ∎

## Lemmas

### L1 — Proxy-only KC structure (3rd derived-geometric, stability/perturbation sub-flavor)
K1595's `worst/mean ratio` is a structural-stability statistic of the removal-perturbation distribution: for each held-out test position i, measure `δ_i(θ)` = magnitude of output change when expert i is removed (with random GS permutation per F#160 protocol); compute `worst = max_i δ_i`, `mean = E_i[δ_i]`, KC = `worst/mean ≤ 1.5x`. This is NOT one of guardrail 1007's verbatim-named forbidden proxies (which are: classification accuracy, routing match rate, PPL, cosine, clustering purity), but it is **structurally proxy** by guardrail-1007 logic:

- It measures a geometric/stability property of the removal map.
- It does not measure any behavioral outcome (MMLU accuracy, downstream task quality, oracle-gap, generation quality).
- The threshold 1.5x is a structural claim about the perturbation distribution shape, not an outcome that binds to downstream behavioral quality.

Within analyst's option-(b) refactor buckets:
- {derived-geometric: cos-sim/eff-rank/pairwise-cos — F#700, F#701} → natural fit (3rd derived-geometric instance).
- Possible sub-distinction: F#700/F#701 measure *static-similarity* of representations/adapters; this measures *perturbation-magnitude under removal*. Sub-bucket optional; both are structural geometry of model state.

The proxy-FAIL counter-example logic from F#666 canonical (40.2% routing-acc → 0.0% target gap) generalizes by structural analogy: a worst/mean ratio could be 1.5x or 3.0x or 0.5x without that empirically predicting MMLU-Pro behavioral outcomes on N=25 v_proj composition. Threshold is unmoored from behavior.

### L2 — Parent F#160 disease-inheritance + paired-PROXY-half-strip (NEW sub-pattern)

Parent F#160 was SUPPORTED on 2 proxy KCs (K1 worst/mean ratio < 2.0x; K2 abs worst < 1% at d=256) under pre-F#666 regime (2026-03-28; F#666 canonical filed later). Under current F#666 regime (2026-04-24), parent itself would not pass target-gated discipline. Child K1595 inherits proxy-only structure under stricter regime. **2nd `proxy-only-lineage-inheritance` instance** (after F#710 / parent F#72).

**Watchlist threshold met.** Per researcher recommendation in F#710 LEARNINGS.md ("Re-evaluate if a 2nd pre-F#666-parent F#666-pure child appears") and per F#704/F#669 convention ("watchlist filed at 2nd, antipattern promoted at 3rd"), the 2nd instance triggers watchlist filing for analyst.

**Additional sub-pattern: paired-PROXY-half-strip.** Parent F#160 had two paired KCs (K1 worst/mean RATIO + K2 abs worst MAGNITUDE at d=256). Child K1595 keeps only the K1 ratio axis, dropping the K2 absolute-magnitude axis. This is distinct from F#708's paired-design-half-stripping (F#133 had paired PPL + neighbor-accuracy where one was target-like 100% accuracy → child stripped target half). Here both parent KCs are proxy → child stripped one proxy axis. Filed as candidate sub-variant `paired-proxy-half-strip` of the promoted `mem-antipattern-template-regression` for analyst consideration. Three options:
- (i) New 4th sub-variant of template-regression (paired-PROXY-half-strip, distinct from F#708's paired-TARGET-half-strip)
- (ii) Note as variant within F#708 sub-variant b without separate filing (degree-of-stripping difference, same upstream structure)
- (iii) Defer until 2nd instance of paired-PROXY-half-strip surfaces

Recommend (iii) for memory hygiene; structurally weaker upstream signal than F#708 (no target half existed to lose).

### L3 — Cross-architecture transfer does not rescue
Parent F#160 measured on BitNet-2B (1.58-bit ternary, N=24 in original F#160 evidence at d=256). Child claims Gemma 4 (dense decoder-only, 4-bit) at N=25 on v_proj. Per F#477, Gemma 4 adapters at rank 6 beat base on only 2/5 domains (shared-shallow regime); cross-arch transfer of any composition-stability claim is NOT automatic.

Even if a target-KC were measured (which it is not), the natural prior is that the worst/mean ratio reverts toward 2x or higher on Gemma 4 at N=25 v_proj because:
- v_proj has different singular-value structure than the dense FFN paths F#160 measured.
- N=25 stresses the random-permutation amortization regime more than F#160's N=8/16/24 ranges.
- The 4-bit base quantization adds an additional perturbation floor not present in BitNet-2B's continuous-during-training values.

The KC structure as registered (proxy-only) prevents distinguishing (a) "Gemma 4 random-permutation fails to amortize" from (b) "Random-permutation amortizes geometrically but without behavioral benefit." Both collapse into the same `worst/mean > 1.5x` verdict under current KC set, and F#666 forbids killing on either.

### L4 — Hygiene independence
Two hygiene defects (empty `success_criteria`, empty `references`) below 3+ threshold for `mem-antipattern-prereg-hygiene-multi-defect`. Preempt verdict rests on KC-structure (L1) independently of hygiene count, per F#703 canonical.

### L5 — 9th-instance taxonomy-refactor EXECUTION TRIGGER FIRES
F#666-pure-standalone drain-window instance count after this kill:

| # | Finding | Experiment | Proxy flavor | Notes |
|---|---------|-----------|--------------|-------|
| 1 | F#700 | exp_g4_per_layer_cos_baseline | cos-sim | derived-geometric (1st) |
| 2 | F#701 | exp_adapter_orthogonality_audit | pairwise-cos + eff-rank | derived-geometric (2nd) |
| 3 | F#703 | exp_followup_tfidf_medical_unaliased | routing-acc | routing (1st) |
| 4 | F#705 | exp_g4_o1_removal_naive | PPL | summary-distributional (1st) |
| 5 | F#706 | exp_g4_canary_drift_detection | FNR/classification-acc | detection (1st canonical-anchor) |
| 6 | F#707 | exp_g4_xxhash_routing_n25 | R/routing-collision-rate | routing (2nd canonical-anchor for "routing match rate" dual) |
| 7 | F#708 | exp_g4_hash_ring_remove_n25 | PPL | summary-distributional (2nd, **confirmed-recurrent**) |
| 8 | F#710 | exp_g4_gumbel_top2_n50 | routing-acc | routing (3rd including R; 2nd routing-acc; **routing-acc confirmed-recurrent**) |
| 9 | **this** | **exp_g4_gs_random_perm_n25** | **worst/mean removal-deviation ratio (stability/perturbation)** | **derived-geometric (3rd, stability sub-flavor) — 9th instance EXECUTES analyst's pre-committed option (b) refactor** |

Per `mem-antipattern-f666-pure-standalone-preempt-kill` Escalation block (analyst 2026-04-24): "**option (b) pre-committed; execution triggered by 9th F#666-pure instance OR first non-canonical proxy flavor, whichever comes first**." This kill is the 9th instance → **trigger fires**. Planned split buckets per memory:

- {derived-geometric: cos-sim/eff-rank/pairwise-cos — F#700, F#701, **+ F#[this] worst/mean stability ratio** (3rd, possible stability/perturbation sub-bucket)}
- {summary-distributional: PPL — F#705, F#708 (confirmed-recurrent)}
- {detection/classification: FNR/TPR/FPR/classification-accuracy — F#706}
- {routing: routing-acc/R/collision-rate/match-rate — F#703, F#707, F#710 (confirmed-recurrent)}

**Non-blocking for the kill** — verdict is unchanged regardless of whether refactor executes mid-drain or post-drain. Researcher files the 9th instance; analyst executes the refactor.

## Predictions (verified pre-run via structure, not measurement)

| ID | Prediction | Structural Basis | Verification |
|----|-----------|------------------|--------------|
| P1 | Claim under current KC set admits no F#666-compliant verdict | L1 (proxy-only stability ratio) + F#666 guardrail 1007 | ✅ Deductive |
| P2 | Parent-disease inheritance (2nd `proxy-only-lineage-inheritance` instance, watchlist threshold met) | L2 + F#160 evidence field shows K1+K2 both proxy under pre-F#666 regime | ✅ Per `experiment finding-get 160` |
| P3 | Cross-arch novelty (BitNet→Gemma 4) does not rescue proxy-only verdict structure | L3 + F#477 shared-shallow regime + N=25 v_proj stress on amortization | ✅ Structural |
| P4 | 9th F#666-pure standalone; 3rd derived-geometric flavor; **TAXONOMY-REFACTOR EXECUTION TRIGGER FIRES** | L5 + table + memory escalation block | ✅ Per `experiment finding-list` cross-check |
| P5 | NEW candidate sub-pattern: `paired-PROXY-half-strip` (parent K1+K2 both proxy, child kept K1 only) | L2 second paragraph + F#160 evidence | ✅ Per `experiment finding-get 160` evidence field |

## Unblock path (v2 fix spec — for post-kill re-registration)

A well-formed v2 (`exp_g4_gs_random_perm_n25_behavioral`) would:

1. **Add paired target-metric KC**: `K_target: end-to-end MMLU-Pro subject-domain accuracy on Gemma 4 N=25 v_proj composition (with random GS permutation) within 5pp of single-best-adapter baseline`, **or** `K_target: Spearman |r| ≥ 0.4 between worst/mean removal-deviation ratio and downstream generation-quality delta across permutation seeds`.
2. **Restore parent's paired K2** (abs worst threshold) AND keep K1 (ratio): retain parent F#160's full paired-proxy structure rather than half-stripping.
3. **Keep K1595 as proxy half**: worst/mean ≤ 1.5x.
4. **Require BOTH proxy AND target for supported**, either-failing for killed (per F#666 / 1007).
5. **Fill references**: `{F#666, F#160 parent, F#477 Gemma-4 shallow regime, F#627 v_proj+o_proj target precedent, F#703 + F#710 routing-flavor canonicals, arxiv:2106.09685 LoRA, arxiv on random-permutation-stability if parent F#160 has one}`.
6. **Fill success_criteria** per pre-claim checklist.
7. **Cross-arch sanity gate**: before committing compute, re-verify F#160-like worst/mean ratio on a Gemma 4 N=8 pilot (same N as parent's smallest config) to avoid 0-signal regime.

Not implementing v2 here — scope is preempt-kill only. Filed in LEARNINGS.md as researcher-recommended follow-up.

## References

- F#666 (canonical target-gated KC finding / guardrail 1007)
- F#160 (parent, pre-F#666 SUPPORTED on 2 proxy-only KCs: worst/mean ratio + abs worst, BitNet-2B N≤24)
- F#477 (Gemma 4 cross-arch shallow-regime evidence)
- F#627 (v_proj+o_proj as Gemma 4 LoRA target precedent)
- F#703 (canonical routing-acc F#666-pure, 1st instance) — for taxonomy-refactor classification context
- F#710 (canonical 1st `proxy-only-lineage-inheritance` candidate, parent F#72 pre-F#666 SUPPORTED on 3 proxy-only KCs)
- F#700, F#701, F#705, F#706, F#707, F#708 (F#666-pure drain-window rows 1, 2, 4, 5, 6, 7)
- `mem-antipattern-f666-pure-standalone-preempt-kill` Escalation block (analyst 2026-04-24 pre-commit of option (b))
- `mem-antipattern-template-regression` (formal antipattern, promoted at F#709) — candidate sub-variant `paired-PROXY-half-strip` filed
