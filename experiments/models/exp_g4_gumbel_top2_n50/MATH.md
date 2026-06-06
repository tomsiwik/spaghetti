# MATH.md — exp_g4_gumbel_top2_n50 (preempt-structural KILL)

## Verdict (pre-registered, pre-run)

**KILLED — preempt-structural (F#666-pure-standalone, 8th drain-window instance, 2nd routing-accuracy flavor).** No `run_experiment.py` measurement needed; proof is structural.

## Pre-claim checklist (5-item template-regression + direction-symmetric inter-variant check)

| # | Check | Result |
|---|-------|--------|
| 1 | `experiment finding-get` on every parent cited | ✅ F#72 retrieved |
| 2 | Scan parent *caveats* for vacuous-language + half-KC paired designs | See L4 below; parent itself is proxy-only |
| 3 | Paired-design-target-inheritance (if parent has paired proxy+target, child must inherit target half) | ✅ Not applicable — parent F#72 had NO target KC to inherit; child inherits parent's disease |
| 4 | Stale-advice-vs-current-guardrail check | ✅ Parent F#72 is pre-F#666 SUPPORTED; child runs under current F#666 regime → tautological |
| 5 | "Under verified <condition>" degenerate-equivalence-branch check | N/A (not an inter-variant delta) |
| 6 | Direction-symmetric inter-variant-delta check | N/A (single-threshold, not inter-adapter comparison) |

## Setup

**Claim under review:** K1591 — "Gumbel top-2 routing on Gemma 4 at N=50 hits `acc >= 85%` on held-out queries."

**Parent cited:** F#72 (BitNet-2B N=49, SUPPORTED) — its K1 was "Gumbel top-2 routing accuracy 86.33% (83% real-data-only)". Parent KCs: K1 routing-acc (proxy), K2 gamma_uniform (proxy composition quality), K3 max-degradation (proxy per-domain). **Zero target KCs in parent.**

**Audit tag:** `audit-2026-04-17` (no `-rerun` suffix → lineage-only per analyst convention; does NOT trigger RECOVERY_PLAN.md fix-before-rerun).

**Hygiene:** 2 defects (success_criteria=[], references=[]) → below 3+ threshold. F#666-pure keys on KC structure independent of hygiene count.

## Theorem

Let Experiment E have kill criterion `K(θ) := acc(θ) ≥ 85%` on a held-out routing classification task, where `acc` is routing-classification accuracy (match rate between predicted expert id and oracle expert id). Let Guardrail G be F#666 / 1007: "KILL requires proxy+target both to fail; SUPPORTED requires both to pass." Then any outcome under E violates G:

1. If `acc(θ) ≥ 85%`: verdict `supported` is tautological per F#666 canonical (routing-acc 40.2% with 0.0% target gap already disproved the PPL-style "proxy-predicts-target" assumption).
2. If `acc(θ) < 85%`: verdict `killed` is forbidden per F#666 ("proxy-FAIL + target-absent = finding about the proxy, not a kill").

Therefore E admits no F#666-compliant verdict. **KILL preempt-structural.** ∎

## Lemmas

### L1 — Proxy-only KC structure
K1591's `acc` is routing classification match rate: for each held-out query q, the router emits argmax expert id π̂(q); the oracle (trained/ideal router) emits π*(q); `acc := E_q[𝟙{π̂(q) = π*(q)}]`. This is a direct instance of guardrail 1007's named-forbidden proxy "routing match rate" / "classification accuracy." No downstream generation-quality metric, no oracle-gap, no behavioral delta is in the KC set.

Under F#666 both the target finding (#257 oracle-gap) and the F#666 canonical counter-example (expert-router 40.2% routing-acc → 0.0% target gap on MMLU-Pro) demonstrate that `acc(proxy) ⊥ acc(target)` empirically on Gemma-family composition. A proxy threshold at 85% has no known-nonzero coupling to behavioral outcomes; the threshold is unmoored.

### L2 — Parent F#72 disease-inheritance
Parent F#72 was SUPPORTED on three proxy KCs (K1 routing-acc, K2 gamma_uniform, K3 max-degradation) with **zero target-metric KC**. The `Caveats:` field of F#72 explicitly documents:
- "9/49 domains use synthetic data (PPL=1.0 memorization, 100% trivial routing accuracy)"
- "adjusted real-data-only routing accuracy ~83%"
- "lora_a is trained (not frozen as MATH.md claims)"
- "VISION.md still references Grassmannian AP-packed A matrices -- needs project-wide reconciliation"

Parent was a pre-F#666 SUPPORTED (2026-03-26; F#666 canonical filed later). Under current (2026-04-24) F#666 regime, parent itself would not pass the target-gated discipline. Child K1591 inherits the proxy-only structure without adding a target half — strict disease-inheritance.

This is NOT template-regression-paired-design-half-stripping (F#708 sub-variant b) because parent had no paired target half to strip. It is a **proxy-only-lineage-inheritance** sub-pattern — child continues parent's disease under a stricter regime. Filed for analyst consideration as a potential new template-regression sub-variant (4th) or as a non-novel continuation of F#666-pure since outcome is the same.

### L3 — Cross-architecture transfer does not rescue
The child claims Gemma 4 (dense transformer, decoder-only) while parent measured BitNet-2B (1.58-bit ternary weights, different router topology). Per F#477, Gemma 4 adapters at rank 6 beat base on only 2/5 domains (shared-shallow regime) — cross-arch transfer of a mechanism proven on BitNet is NOT automatic on Gemma 4.

Even if a target-KC were measured (which it is not), the natural prior is that routing-accuracy degrades from 83% (BitNet real-data) toward chance on Gemma 4 unless re-tuned per-layer/per-module — a well-behaved research question. The KC structure as registered (proxy-only) prevents distinguishing (a) "Gemma 4 routing fails to generalize from BitNet" from (b) "Gemma 4 routing generalizes but without behavioral benefit." Both collapse into the same `acc < 85%` verdict under the current KC set, and F#666 forbids killing on either.

### L4 — Hygiene independence
Two hygiene defects (empty success_criteria, empty references) are below the 3+ threshold for hygiene-multi-defect. The preempt verdict rests on KC-structure (L1) independently of hygiene count, per F#703 canonical ("F#666-pure-standalone keys on KC structure, not hygiene count").

### L5 — 8th-instance taxonomy-refactor trigger (non-blocking for this claim)
F#666-pure-standalone drain-window instance count:

| # | Finding | Experiment | Proxy flavor |
|---|---------|-----------|--------------|
| 1 | F#700 | exp_g4_per_layer_cos_baseline | cos-sim |
| 2 | F#701 | exp_adapter_orthogonality_audit | pairwise-cos + eff-rank |
| 3 | F#703 | exp_followup_tfidf_medical_unaliased | routing-acc (1st) |
| 4 | F#705 | exp_g4_o1_removal_naive | PPL (1st) |
| 5 | F#706 | exp_g4_canary_drift_detection | FNR/classification-acc |
| 6 | F#707 | exp_g4_xxhash_routing_n25 | R/routing-collision-rate |
| 7 | F#708 | exp_g4_hash_ring_remove_n25 | PPL (2nd, confirmed-recurrent) |
| 8 | **this** | **exp_g4_gumbel_top2_n50** | **routing-acc (2nd, confirmed-recurrent)** |

Taxonomy-refactor trigger has been live since row 5. At row 8, a second proxy flavor (routing-acc) reaches confirmed-recurrent status. Three refactor options on file with the analyst:

- (a) Consolidate into single super-category (drop flavor tracking)
- (b) Split antipattern memory by proxy flavor (each with its own anchors list)
- (c) Add guardrail-1007-enumeration sub-section referencing named forbidden proxies with instance counts

**Non-blocking for the kill** — verdict is unchanged regardless of refactor choice.

## Predictions (verified pre-run via structure, not measurement)

| ID | Prediction | Structural Basis | Verification |
|----|-----------|------------------|--------------|
| P1 | Claim under current KC set admits no F#666-compliant verdict | L1 (proxy-only) + F#666 guardrail 1007 | ✅ Deductive |
| P2 | Parent-disease inheritance: F#72 is pre-F#666 SUPPORTED on 3 proxy-only KCs; child continues disease under current regime | L2 + F#72 `Caveats:` field | ✅ Per finding-get |
| P3 | Cross-arch novelty (BitNet→Gemma 4) does not rescue proxy-only verdict structure | L3 + F#477 shared-shallow regime | ✅ Structural |
| P4 | 8th-instance F#666-pure standalone; 2nd routing-acc sub-flavor (confirmed-recurrent) | L5 + table | ✅ Per finding-list |

## Unblock path (v2 fix spec — for post-kill re-registration)

A well-formed v2 (`exp_g4_gumbel_top2_n50_behavioral`) would:

1. **Add paired target-metric KC**: `K_target: end-to-end MMLU-Pro subject-domain accuracy within 5pp of oracle-adapter baseline at N=50`, or `K_target: Spearman |r| ≥ 0.4 between routing confidence and downstream generation-quality delta`.
2. **Keep K1591 as proxy half**: routing-acc ≥ 85% on held-out.
3. **Require BOTH for supported**, either-failing for killed (per F#666 / 1007).
4. **Fill references**: `{F#666, F#72 parent, F#257 oracle-gap, F#477 Gemma-4 shallow regime, F#703 routing-acc canonical, arxiv:1611.01144 Gumbel-softmax}`.
5. **Fill success_criteria** per pre-claim checklist.
6. **Cross-arch sanity gate**: before committing compute, re-verify F#72-like routing on a Gemma 4 N=5 pilot to avoid 0-signal regime.

Not implementing v2 here — scope is preempt-kill only. Filed in LEARNINGS.md as researcher-recommended follow-up.

## References

- F#666 (canonical target-gated KC finding / guardrail 1007)
- F#72 (parent, pre-F#666 SUPPORTED on 3 proxy-only KCs, BitNet-2B N=49)
- F#703 (canonical routing-acc F#666-pure, 1st instance)
- F#257 (oracle-gap target metric)
- F#477 (Gemma 4 cross-arch shallow-regime evidence)
- F#700, F#701, F#705, F#706, F#707, F#708 (F#666-pure drain-window rows 1-2, 4-7)
