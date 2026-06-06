# PAPER — exp_hedgehog_early_stopping_cos_plateau (KILLED — preempt-structural, F#666-pure standalone, 7th Hedgehog-ablation sub-type)

## Verdict: KILLED (KC-structural preempt, F#666-pure standalone, ~26th drain-window instance, 7th Hedgehog-ablation super-family sub-type: training-stopping-criterion / early-stopping-ablation)

This experiment is preempt-killed on structural grounds before any code executes. The verdict is deterministic from the KC-set shape, not from measurement.

## Summary

Pre-registered kill-criterion set K = {K1935, K1936} contains two proxy metrics with no paired target-metric KC:

- **K1935** ("Early-stopped adapter cos-sim within 0.02 of full-training adapter") — *inter-training-stopping-point* cos-sim tightness/distance. Cos-sim-on-cos-sim doubly proxy: inner cos-sim (vs teacher) is proxy for behavior (guardrail 1006: r≈0.08 PPL→task quality); outer "within 0.02" bound is proxy for adapter-output equivalence, not adapter-behavior equivalence. F#666 canonical 40.2% per-sample classification acc + 0.0% target-gap shows cos-sim-equivalence and behavior-equivalence are separable. **3rd cos-sim-bucket form** (after F#720 final-value 1st, F#755 convergence-speed 2nd) — *tightness/distance* form.
- **K1936** ("Early stopping reduces training time < 30% (not significant)") — training-time-savings ratio. Compute-efficiency without behavioral pairing. Stopping at iter=0 saves 100% compute and produces a noise adapter — KC alone cannot distinguish honest savings from useless savings. **1st training-axis efficiency-bucket form**, adjacent to F#753 K1929 inference-axis infrastructure-benchmark (routing-latency).

Under F#666, a proxy-only KC set has no valid verdict regardless of empirical outcome:
- Both PASS → tautological SUPPORT (cos-sim-tight + compute-cheap, but no behavioral evidence; ships an unverified adapter).
- Any FAIL → "finding about the proxy, not a kill" per F#666 explicit rule.

This is the **7th Hedgehog-ablation super-family** sub-type. The pattern is well-established post taxonomy refactor at F#714: ≥25 prior F#666-pure standalone instances exist (F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711, F#714, F#715, F#716, F#720, F#722, F#728, F#729, F#730, F#731, F#732, F#734, F#735, F#736, F#753, F#754, F#755, …).

**Hedgehog-ablation super-family taxonomic placement:**

| # | Sub-type                                       | Canonical finding(s)              | KC design               | Outcome                |
| - | ---------------------------------------------- | --------------------------------- | ----------------------- | ---------------------- |
| 1 | axis-extension                                 | F#683, F#684, F#696, F#697, F#717, F#718 (behavior + domain axes) | target-anchored         | PROVISIONAL design-locks |
| 2 | loss-variant-ablation                          | F#719, F#720                      | F#719 target-pair / F#720 cos-sim-only | F#719 PROV / F#720 KILLED |
| 3 | layer-selection-ablation                       | F#721                             | proxy-only              | KILLED triple-fire      |
| 4 | hyperparameter-ablation                        | F#722 teacher-temperature sweep   | proxy-only              | KILLED triple-fire      |
| 5 | data-augmentation-ablation                     | F#723                             | target + proxy pair     | PROVISIONAL             |
| 6 | curriculum / training-procedure-ordering       | F#755                             | proxy-only              | KILLED preempt-structural |
| **7** | **training-stopping-criterion / early-stopping-ablation** | **THIS (F#NEW)**       | **proxy-only**          | **KILLED preempt-structural** |

Pattern from F#719 vs F#720, F#722 vs F#723, F#755 (preempt-killed) vs hypothetical-runnable: target-pair design is the discriminator between runnable and preempt-killed Hedgehog-ablation experiments.

## Prediction vs measurement

| KC    | Claim                                                                            | Kind  | Sub-flavor                                                                          | Verdict                                |
| ----- | -------------------------------------------------------------------------------- | ----- | ----------------------------------------------------------------------------------- | -------------------------------------- |
| K1935 | Early-stopped adapter cos-sim within 0.02 of full-training adapter               | proxy | cos-sim-bucket tightness/distance form (3rd cos-sim-bucket instance)                | UNTESTED (preempt-blocked, F#666-pure) |
| K1936 | Early stopping reduces training time < 30% (not significant)                     | proxy | training-axis efficiency-bucket compute-cost (1st instance, adjacent to F#753 inference-latency) | UNTESTED (preempt-blocked, F#666-pure) |

No measurement was taken. No model was loaded; no dataset opened; no teacher forward pass; no student forward pass; no per-layer cos-sim computed; no plateau detector fired; no early-stopped adapter saved; no full-trained baseline trained. The verdict derives from `F#666 proxy-only KC set` + `no target-metric pair` ⇒ tautological-for-all-outcomes (4-cell truth table in MATH.md §1).

## Why this is not runnable as-is

Even if the early-stopping-vs-full-training comparison were executed, every cell of the {K1935, K1936} × {PASS, FAIL} outcome space maps to an inadmissible verdict under F#666:

| K1935  | K1936  | Verdict                                                                                                                                |
| ------ | ------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| PASS   | PASS   | Tautological SUPPORT — early-stopped adapter is "close enough" on cos-sim AND saves enough compute. But neither measures whether the adapter still works behaviorally on politeness. PASS-PASS could ship a cos-sim-tight, compute-cheap, behaviorally-degraded adapter. |
| PASS   | FAIL   | Cos-sim tight, compute savings < 30%; finding about plateau-detection sensitivity (50-step threshold fires too late), not a behavioral kill |
| FAIL   | PASS   | Cos-sim diverges > 0.02, compute savings ≥ 30%; finding about over-aggressive early stopping. Could be a *good* adapter via cluster-equivalence (F#666 canonical decoupling) or via early-stopping-as-regularization (Caruana et al. 2000) — neither resolvable without behavioral KC |
| FAIL   | FAIL   | Both proxies fail their thresholds; still "finding about proxies, not kill" — no target measured. Could be cos-sim-distant + low-savings + behaviorally-superior (regularization rationale) |

The F#666 rule operates on KC *kind*, not measurement value. Proxy-only structure is the disease; measurement outcome is the symptom.

### Pathological-case illustrations of decoupling

1. **K1935 PASS + behavior degraded** (over-fits to easy validation prompts): early-stopped adapter is within 0.02 cos-sim of full-trained adapter on validation prompts but loses politeness coverage on hard-tail prompts (Hacohen-Weinshall 2019 generalization-gap pattern, arxiv:1904.03626). The plateau-detection metric saturates on easy prompts and stops training before the hard-tail signal arrives. Cos-sim says "tight"; behavior says "regressed".
2. **K1935 FAIL + behavior preserved** (early-stopping-as-regularization): early-stopped adapter is > 0.02 cos-sim away from full-trained adapter (FAIL on K1935) but the *full-trained* adapter is over-fit to teacher idiosyncrasies. The early-stopped adapter retains better generalization. Classic Caruana et al. 2000 NeurIPS rationale. LLM-judge politeness equal or better at the early-stopped point. Cos-sim says "diverged"; behavior says "improved".
3. **K1936 PASS via degenerate stop** (iter=0): trivial 100% compute savings produces a noise adapter. K1936 alone cannot distinguish.
4. **K1936 FAIL via short-loss**: training is so short that 30% savings is impossible — K1936 fails for non-procedural reasons (training itself was too short for early stopping to matter).

Each pathology requires a behavioral target KC (LLM-judge politeness on hard tail, behavioral generalization-gap, oracle-gap on a politeness benchmark) to disambiguate.

## Hygiene defects

| Defect                  | Status                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------- |
| F#666 violation         | Present (K1935 + K1936 are both proxies; cos-sim and compute-cost forbidden by guardrail 1007) |
| `success_criteria` MISSING | Present (DB explicitly flags `⚠ INCOMPLETE: success_criteria`)                     |
| `platform: —`           | Present (DB shows `platform: —` null; `⚠ INCOMPLETE: ... platform`)                  |
| `references: []`        | Present (guardrail 1002 violation; no arxiv or finding citation despite Caruana 2000 / Prechelt 1998 / Hacohen-Weinshall 2019 / Wu 2021 prior art) |
| Plateau threshold unanchored | 50-step plateau in notes is unmotivated — not Prechelt PQ/UP/GL nor calibrated against behavioral floor |
| Notes coherence         | Cos-sim-on-cos-sim circularity: cos-sim plateau (stopping signal) drives cos-sim equivalence (eval signal) — 2nd cos-sim-on-cos-sim circularity instance after F#755 (curriculum-ordering, notes-level) |

Hygiene-defect count = 5 (SC + platform + refs + dir + plateau-threshold). Crosses AP-prereg-hygiene-multi-defect (≥3) threshold. F#666-pure structural defect alone is sufficient for kill independent of hygiene count.

## Taxonomic comparison with drain-window precedents

| Dimension              | F#722 (hyperparameter-ablation, inter-config) | F#755 (curriculum-ordering, intra-procedure) | F#720 (cos-sim-bucket, final-value)         | F#723 (data-aug, runnable)        | This (early-stopping, 7th sub-type)              |
| ---------------------- | --------------------------------------------- | -------------------------------------------- | ------------------------------------------- | --------------------------------- | ------------------------------------------------ |
| Parent dep             | none                                          | none                                         | none                                        | none                              | none                                             |
| KC count               | 2                                             | 2                                            | 1                                           | 2                                 | 2                                                |
| KC kinds               | proxy-only                                    | proxy-only                                   | proxy-only                                  | **target + proxy**                | **proxy-only**                                   |
| F#666 violation        | yes                                           | yes                                          | yes                                         | no                                | yes                                              |
| Hedgehog-ablation #    | 4th (hyperparameter)                          | 6th (curriculum-ordering)                    | 2nd (loss-variant)                          | 5th (data-augmentation)           | **7th (training-stopping-criterion)**            |
| Cos-sim form           | cos-sim derived                               | cos-sim convergence-speed (2nd cos-sim-bucket) | cos-sim final-value (1st cos-sim-bucket)  | cos-sim proxy + behavioral target | **cos-sim tightness/distance (3rd cos-sim-bucket)** |
| Efficiency form        | none                                          | none                                         | none                                        | none                              | **training-axis compute-cost (1st training-efficiency)** |
| Cos-sim-on-cos-sim     | none                                          | 1st (notes-level, ordering-driving-eval)     | none                                        | none                              | **2nd (intra-training-trajectory, plateau-driving-eval)** |
| Verdict                | KILLED (preempt-structural)                   | KILLED (preempt-structural)                  | KILLED (preempt-structural)                 | PROVISIONAL                       | **KILLED (preempt-structural)**                  |
| `_impl` follow-up      | none                                          | none                                         | none                                        | none in DB                        | none                                             |

The invariant: `depends_on: []` + proxy-only KC set ⇒ preempt-KILL, independent of KC count, hygiene count, sub-flavor, proxy form (final-value, delta, convergence-speed, tightness/distance, efficiency), or axis (intra-training, inter-config, intra-trajectory).

## Caveats

- All seven Hedgehog-ablation sub-types now observed (axis-extension, loss-variant, layer-selection, hyperparameter, data-augmentation, curriculum-ordering, training-stopping-criterion). The space of "obvious Hedgehog-ablation KC formulations" is closing fast; researchers attempting an 8th sub-type (e.g., teacher-architecture ablation, distillation-temperature schedule, multi-teacher mixing) should pre-register with target-pair KCs to avoid preempt.
- The implicit conceptual parent (`exp_hedgehog_behavior_adapter_politeness_impl`, P=1, status=open, never executed) means even with target-pair KCs, this experiment carries F#669-style child-on-unverified-parent risk: comparing early-stopped vs full-trained of an *unverified base method* has no anchor. Early-stopping vs full-training is meaningful only if full-training itself produces a working adapter; F#683 hasn't measured that yet.
- Cos-sim-driven plateau-detection on a cos-sim equivalence test is *intra-training-trajectory* cos-sim-on-cos-sim circularity — a tighter form than F#755's notes-level circularity (F#755 used cos-sim difficulty to *order* training; this uses cos-sim plateau to *halt* training). Decoupling stopping signal (e.g., held-out validation loss per Caruana 2000) from evaluation signal would still need a behavioral target.
- The 50-step plateau threshold in notes is unanchored — neither anchored to Prechelt 1998 PQ_α/UP_k formal criteria nor to a behavioral floor.
- Early-stopping prior art is substantial (Caruana, Lawrence, Giles 2000 NeurIPS — canonical formulation; Prechelt 1998 — formal plateau-detection variants; Hacohen-Weinshall 2019 arxiv:1904.03626 — generalization-gap framework; Wu et al. 2021 arxiv:2010.13166 — distillation-context with task-accuracy target) — none cited in pre-reg.
- Base model `mlx-community/gemma-4-e4b-it-4bit` not loaded; no adapters trained; no MLX code executed.

## Follow-up (recommended)

If an early-stopping plateau-detection ablation on Hedgehog cos-sim distillation at Gemma 4 E4B is still a research question of interest, register `exp_hedgehog_early_stopping_cos_plateau_behavioral` AFTER F#683 graduates from PROVISIONAL to SUPPORTED, with target-gated KCs:

```yaml
kill_criteria:
  - K_proxy_cos_tightness  : Early-stopped adapter cos-sim diverges from full-trained by > 0.02 (tightness threshold preserved)
  - K_proxy_compute_savings : Early stopping reduces training time by < 30% (efficiency threshold preserved)
  - K_target_politeness    : Early-stopped adapter LLM-judge politeness score < full-trained politeness score by > 1pp on F#683 prompt set
  - K_target_generalization_gap: Early-stopped adapter shows > 2pp degradation vs full-trained on held-out hard-tail prompts (Hacohen-Weinshall 2019 arxiv:1904.03626 generalization-gap test)
  - K_target_pareto        : Among early-stopping schedules that PASS K_proxy_cos_tightness AND K_proxy_compute_savings, best schedule's LLM-judge politeness >= full-trained - 0pp
references:
  - F#666 (target-gated KILL discipline)
  - F#683 (Hedgehog politeness target metric -- LLM-judge politeness score)
  - F#722 (Hedgehog hyperparameter-ablation preempt precedent)
  - F#723 (Hedgehog data-augmentation-ablation runnable target-pair example)
  - F#755 (Hedgehog curriculum-ordering-ablation preempt -- sibling on training-procedure axis)
  - F#719/F#720 (loss-variant-ablation discriminator pair)
  - Caruana, Lawrence, Giles 2000 NeurIPS (Overfitting in neural nets: Backpropagation, conjugate gradient, and early stopping -- canonical early-stopping formulation, target-pair design template)
  - Prechelt 1998 (Early Stopping -- But When? formal plateau-detection criteria GL/PQ/UP)
  - arxiv:1904.03626 (Hacohen & Weinshall 2019 -- generalization-gap framework)
  - arxiv:2010.13166 (Wu et al. 2021 -- curriculum learning for knowledge distillation, runnable target-pair template)
platform: local-apple
success_criteria: [K_target_politeness PASS AND K_target_generalization_gap PASS AND >=1 schedule PASSES K_target_pareto]
notes: "Early-stopping plateau-detection ablation on Hedgehog cos-sim distillation at Gemma 4 E4B. Stopping signal: held-out validation loss plateau (Caruana 2000) -- DECOUPLED from evaluation signal. Plateau threshold: Prechelt PQ_alpha=0.5 (anchored). Evaluation signal: LLM-judge politeness on F#683 prompt set + held-out hard-tail (Hacohen-Weinshall generalization-gap)."
parent_dependency: F#683 must reach status=supported before this can claim
```

This closes the F#666 gap (target-pair present), addresses guardrail 1002 (references comprehensive), eliminates the cos-sim-on-cos-sim circularity (held-out validation loss as stopping signal vs LLM-judge politeness as evaluation signal), anchors the plateau threshold (Prechelt PQ_α), and satisfies the F#669-style child-anchor requirement (parent F#683 must be supported).

## Unblock condition (no rerun of this pre-reg)

See MATH.md §5. Pre-reg must be edited before any re-claim to add target-metric KCs, references, populate success_criteria, set platform=local-apple, decouple stopping signal from evaluation signal, and anchor the plateau threshold. Post-claim KC mutation is antipattern-u; recommendation is to **close this pre-reg as structurally-malformed** and use the follow-up template above instead, gated on F#683 reaching supported.
