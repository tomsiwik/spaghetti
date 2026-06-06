# PAPER — exp_hedgehog_curriculum_training (KILLED — preempt-structural, F#666-pure standalone, 6th Hedgehog-ablation sub-type)

## Verdict: KILLED (KC-structural preempt, F#666-pure standalone, ~25th drain-window instance, 6th Hedgehog-ablation super-family sub-type: curriculum / training-procedure-ablation)

This experiment is preempt-killed on structural grounds before any code executes. The verdict is deterministic from the KC-set shape, not from measurement.

## Summary

Pre-registered kill-criterion set K = {K1933, K1934} contains two proxy metrics with no paired target-metric KC:

- **K1933** ("Curriculum training produces adapter > 3pp worse than random-order training") — relative cos-sim/PPL adapter quality delta. Hedgehog framework default metric is cos-sim against teacher (Moudgil §3.1); even if interpreted as PPL, guardrail 1006 declares PPL is itself a proxy (r≈0.08 with task quality in this codebase). Delta-of-proxies is still a proxy (F#754 §1.1). 1st curriculum-ablation instance in Hedgehog-ablation super-family (6th sub-type after axis-extension, loss-variant, layer-selection, hyperparameter, data-augmentation).
- **K1934** ("Curriculum training cos-sim convergence < random-order") — training-curve cos-sim convergence speed. Direct cos-sim measurement during training. 2nd cos-sim-bucket instance after F#720 (K1872 cos-sim-only final-value form, killed); this is convergence-speed form — speed-of-cos-sim inherits cos-sim-as-proxy classification.

Under F#666, a proxy-only KC set has no valid verdict regardless of empirical outcome:
- Both PASS → tautological KILL (curriculum cos-sim worse, but no behavioral evidence; could be cluster-equivalent on LLM-judge politeness per F#666 canonical 40.2% per-sample acc + 0.0% target gap).
- Any FAIL → "finding about the proxy, not a kill" per F#666 explicit rule.

This is the **6th Hedgehog-ablation super-family** sub-type. The pattern is well-established: **F#722** (hyperparameter-ablation, teacher-temperature sweep) is the closest structural sibling — both KCs proxy, killed as triple-fire preempt-KILL. **F#723** (data-augmentation-ablation) is the closest runnable separator — K1877 target + K1878 proxy → PROVISIONAL not killed. F#719/F#720 (loss-variant) shows the same separator: target-pair runnable (F#719), cos-sim-only killed (F#720).

The pattern is well-established post the taxonomy refactor at F#714: ≥24 prior F#666-pure standalone instances exist (F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711, F#714, F#715, F#716, F#720, F#722, F#728, F#729, F#730, F#731, F#732, F#734, F#735, F#736, F#753, F#754, …).

## Prediction vs measurement

| KC    | Claim                                                                            | Kind  | Sub-flavor                                                                  | Verdict                                |
| ----- | -------------------------------------------------------------------------------- | ----- | --------------------------------------------------------------------------- | -------------------------------------- |
| K1933 | Curriculum training produces adapter > 3pp worse than random-order training      | proxy | curriculum/training-procedure-ablation cos-sim/PPL delta (1st curriculum)   | UNTESTED (preempt-blocked, F#666-pure) |
| K1934 | Curriculum training cos-sim convergence < random-order (worse)                   | proxy | cos-sim-bucket convergence-speed (2nd cos-sim-bucket after F#720)           | UNTESTED (preempt-blocked, F#666-pure) |

No measurement was taken. No model was loaded; no dataset opened; no teacher forward pass; no student forward pass; no per-layer cos-sim computed; no curriculum schedule constructed; no random-order baseline trained. The verdict derives from `F#666 proxy-only KC set` + `no target-metric pair` ⇒ tautological-for-all-outcomes (4-cell truth table in MATH.md §1).

## Why this is not runnable as-is

Even if the curriculum-vs-random comparison were executed, every cell of the {K1933, K1934} × {PASS, FAIL} outcome space maps to an inadmissible verdict under F#666:

| K1933  | K1934  | Verdict                                                                                                                                |
| ------ | ------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| PASS   | PASS   | Tautological KILL — both proxies say curriculum is worse on cos-sim; but adapter could still be cluster-equivalent on LLM-judge politeness (F#666 canonical: 40.2% routing-acc + 0.0% target gap shows decoupling). PASS-PASS would discard a potentially behavior-preserving training procedure. |
| PASS   | FAIL   | Finding about cos-sim convergence-speed behavior, not a behavioral kill                                                                |
| FAIL   | PASS   | Finding about relative cos-sim/PPL quality delta, not a behavioral kill                                                                |
| FAIL   | FAIL   | Both proxies disagree with each other or with the curriculum hypothesis; still "finding about proxies, not kill" — no target measured  |

The F#666 rule operates on KC *kind*, not measurement value. Proxy-only structure is the disease; measurement outcome is the symptom.

### Pathological-case illustrations of decoupling

1. **K1933 PASS + behavior preserved**: curriculum order over-emphasizes easy (high-similarity) examples in early training, producing a higher cos-sim final adapter (so K1933 PASS = curriculum better, but per KC text PASS = curriculum WORSE; reverse the example) — invert: curriculum order under-emphasizes easy, producing slightly lower cos-sim BUT the LLM-judge politeness score is preserved or improved. Cos-sim says "worse"; behavior says "equivalent". K1933 PASS would discard a working procedure.
2. **K1933 FAIL + behavior degraded**: curriculum order matches random-order on cos-sim within 3pp, but produces a less generalizable policy because hard examples weren't seen with already-aligned base. LLM-judge politeness drops 5pp on hard tail. Cos-sim says "fine"; behavior says "regressed". K1933 FAIL would falsely accept.
3. **K1934 PASS via easy-block memorization**: curriculum's easy-first phase converges fast on cos-sim (K1934 fails the "< random" test in the easy phase); but the adapter is memorizing the easy block and won't generalize. Speed-of-cos-sim ≠ quality-of-cos-sim ≠ behavior.
4. **K1934 FAIL via stochastic noise**: convergence speed has high variance under different random seeds; a single-seed comparison can flip the K1934 verdict without any procedural change.

Each pathology requires a behavioral target KC (LLM-judge politeness, MMLU subject preservation, oracle-gap on a politeness benchmark) to disambiguate.

## Hygiene defects

| Defect                  | Status                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------- |
| F#666 violation         | Present (K1933 + K1934 are both proxies; cos-sim and PPL forbidden by guardrail 1007) |
| `success_criteria` MISSING | Present (DB explicitly flags `⚠ INCOMPLETE: success_criteria`)                     |
| `platform: —`           | Present (DB shows `platform: —` null; `⚠ INCOMPLETE: ... platform`)                  |
| `references: []`        | Present (guardrail 1002 violation; no arxiv or finding citation despite Bengio 2009 / Wu 2021 prior art) |
| Notes coherence         | "Tests curriculum learning" with cos-sim difficulty signal AND cos-sim evaluation signal — cos-sim-on-cos-sim circularity reinforces F#666 violation |

Hygiene-defect count = 4 (SC + platform + refs + dir). Crosses AP-prereg-hygiene-multi-defect (≥3) threshold. F#666-pure structural defect alone is sufficient for kill independent of hygiene count.

## Taxonomic placement: Hedgehog-ablation super-family (6th sub-type)

| # | Sub-type                          | Canonical finding(s)              | KC design               | Outcome      |
| - | --------------------------------- | --------------------------------- | ----------------------- | ------------ |
| 1 | axis-extension                    | F#683 politeness, F#684 procedural, F#696 JS, F#697 Python, F#717 Rust, F#718 SQL | target-anchored | PROVISIONAL design-locks |
| 2 | loss-variant-ablation             | F#719 cos-sim-vs-KL, F#720 MSE    | F#719 target-pair / F#720 cos-sim-only | F#719 PROV / F#720 KILLED |
| 3 | layer-selection-ablation          | F#721                             | proxy-only              | KILLED triple-fire |
| 4 | hyperparameter-ablation           | F#722 teacher-temperature sweep   | proxy-only              | KILLED triple-fire |
| 5 | data-augmentation-ablation        | F#723                             | target + proxy pair     | PROVISIONAL  |
| **6** | **curriculum / training-procedure-ablation** | **THIS (F#NEW)** | **proxy-only**          | **KILLED preempt-structural** |

Pattern from F#719 vs F#720 and F#722 vs F#723: target-pair design is the discriminator between runnable and preempt-killed Hedgehog-ablation experiments. This experiment lacks target-pair → falls in the F#720/F#721/F#722 cluster.

## Taxonomic comparison with drain-window precedents

| Dimension              | F#722 (hyperparameter-ablation)             | F#720 (cos-sim-bucket loss-variant)         | F#723 (data-aug-ablation, runnable)        | This (curriculum-ablation, 6th sub-type)              |
| ---------------------- | ------------------------------------------- | ------------------------------------------- | ------------------------------------------ | ----------------------------------------------------- |
| Parent dep             | none                                        | none                                        | none                                       | none                                                  |
| KC count               | 2                                           | 1                                           | 2                                          | 2                                                     |
| KC kinds               | proxy-only                                  | proxy-only                                  | **target + proxy**                         | **proxy-only**                                        |
| F#666 violation        | yes                                         | yes                                         | no                                         | yes                                                   |
| Hedgehog-ablation #    | 4th (hyperparameter)                        | 2nd (loss-variant)                          | 5th (data-augmentation)                    | **6th (curriculum/training-procedure)**               |
| Cos-sim form           | cos-sim derived                             | cos-sim final-value (1st cos-sim-bucket)    | cos-sim proxy paired with behavioral target | **cos-sim convergence-speed (2nd cos-sim-bucket)**    |
| Verdict                | KILLED (preempt-structural)                 | KILLED (preempt-structural)                 | PROVISIONAL                                | **KILLED (preempt-structural)**                       |
| `_impl` follow-up      | none                                        | none                                        | none in DB                                 | none                                                  |

The invariant: `depends_on: []` + proxy-only KC set ⇒ preempt-KILL, independent of KC count, hygiene count, sub-flavor, or proxy form (final-value, delta, convergence-speed).

## Caveats

- All six Hedgehog-ablation sub-types have now been observed (axis-extension, loss-variant, layer-selection, hyperparameter, data-augmentation, curriculum/training-procedure). The space of "obvious Hedgehog-ablation KC formulations" is closing; researchers attempting 7th sub-type (e.g., teacher-architecture ablation, data-mixing-ratio ablation) should pre-register with target-pair KCs to avoid preempt.
- The implicit conceptual parent (`exp_hedgehog_behavior_adapter_politeness_impl`, P=1, status=open, never executed) means even with target-pair KCs, this experiment carries F#669-style child-on-unverified-parent risk: comparing two curricula of an unverified base method has no anchor.
- Cos-sim-driven curriculum on a cos-sim metric is structurally tautological at the cos-sim layer (curriculum-difficulty signal = teacher-student divergence, evaluation signal = teacher-student divergence post-training). Even decoupling difficulty signal (e.g., per-prompt teacher entropy) from evaluation signal would still need a behavioral target.
- Curriculum-learning prior art is substantial (Bengio 2009 arxiv:0903.0738; Hacohen-Weinshall 2019 arxiv:1904.03626) and curriculum-for-distillation specifically (Wu et al. 2021 arxiv:2010.13166) — none cited in the pre-reg.
- Base model `mlx-community/gemma-4-e4b-it-4bit` not loaded; no adapters trained; no MLX code executed.

## Follow-up (recommended)

If a curriculum-vs-random training-procedure ablation on Hedgehog cos-sim distillation at Gemma 4 E4B is still a research question of interest, register `exp_hedgehog_curriculum_training_behavioral` AFTER F#683 graduates from PROVISIONAL to SUPPORTED, with target-gated KCs:

```yaml
kill_criteria:
  - K_proxy_quality_delta : Curriculum training adapter cos-sim < random-order adapter cos-sim by > 3pp at fixed seed and prompt set
  - K_proxy_convergence   : Curriculum cos-sim convergence speed < random-order convergence speed (worse) at half-training
  - K_target_politeness   : Curriculum-trained adapter LLM-judge politeness score < random-trained politeness score by > 1pp on F#683 prompt set
  - K_target_generalization: Curriculum-trained adapter shows > 2pp degradation vs random-trained on held-out hard-tail prompts (Hacohen-Weinshall 2019 generalization-gap test)
references:
  - F#666 (target-gated KILL discipline)
  - F#683 (Hedgehog politeness target metric -- LLM-judge politeness score)
  - F#722 (Hedgehog hyperparameter-ablation preempt precedent)
  - F#723 (Hedgehog data-augmentation-ablation runnable target-pair example)
  - F#719/F#720 (loss-variant-ablation discriminator pair: target-pair runnable vs cos-sim-only killed)
  - arxiv:0903.0738 (Bengio et al. 2009 -- curriculum learning original)
  - arxiv:1904.03626 (Hacohen & Weinshall 2019 -- power of curriculum)
  - arxiv:2010.13166 (Wu et al. 2021 -- curriculum learning for knowledge distillation, runnable design template)
platform: local-apple
success_criteria: [K_target_politeness PASS AND K_target_generalization PASS AND >=1 curriculum schedule PASSES K_proxy_quality_delta]
notes: "Curriculum-vs-random training procedure ablation on Hedgehog cos-sim distillation at Gemma 4 E4B. Difficulty signal: per-prompt teacher-output entropy (distinct from cos-sim evaluation signal, eliminating cos-sim-on-cos-sim circularity). Evaluation signal: LLM-judge politeness score on F#683 prompt set + held-out hard-tail (Hacohen-Weinshall generalization-gap)."
parent_dependency: F#683 must reach status=supported before this can claim
```

This closes the F#666 gap (target-pair present), addresses the redundancy concern (Wu et al. 2021 prior art cited and design-pattern-replicated), eliminates the cos-sim-on-cos-sim circularity (entropy-based difficulty signal), and satisfies the F#669-style child-anchor requirement (parent F#683 must be supported).

## Unblock condition (no rerun of this pre-reg)

See MATH.md §5. Pre-reg must be edited before any re-claim to add target-metric KCs, references, populate success_criteria, set platform=local-apple, and remove the cos-sim-on-cos-sim circularity in notes. Post-claim KC mutation is antipattern-u; recommendation is to **close this pre-reg as structurally-malformed** and use the follow-up template above instead, gated on F#683 reaching supported.
