# REVIEW (adversarial) — exp_spark_temporal_interference

**Verdict: PROCEED-as-KILLED.** Honest, real-model refutation. No blocking issues.

## No-fabrication checklist
- is_smoke=false; no hardcoded pass=True; accuracies computed from real Gemma-4 generations
  (wall clock 1760s, 80 items, 512-token decodes). verify-experiment.sh exits 0.
- Both adapters load and GENUINELY DIFFER: same 84 keys but NOT all-identical, mean|delta|
  layer0 lora_a = 0.0152, norms 1.68 vs 1.58. Not a file copy.
- Composition is sum of scale*(B_i@A_i) in activation space (L60-67), NOT (sumB)(sumA).
  scale=6.0 <= 8. Base entropy from FROZEN base_model, own cache, no adapters (L158-182).

## The control is REAL (key concern: D==E==47.5%)
D and E tie in AGGREGATE accuracy but are NOT the same outputs and the gate is NOT a no-op:
- Gate changes generations: D differs from C on 11/80 items, E from C on 7/80.
- D differs from E on 12/80 items (entropy-selected steps != random-selected steps -> different text).
- D flips both directions (right->wrong AND wrong->right), netting +1 vs C; E nets +1 too.
The 47.5==47.5 is a genuine wash, not shared-selection and not a zero-effect gate.
Only step-SELECTION differs between D and E (same code adapter zeroed, same count k per
sequence) -- exactly the pre-registered isolation of choice-point-specific vs dropout.

## Kill-criteria integrity
- DB KC (2288/2289/2290) text matches MATH.md section 5 and code thresholds (0.50 / 2pp / 0.05).
- Two of three KCs are target/behavioral (GSM8K accuracy) -- Target-Gated Kill Rule satisfied.
- K1 FAIL (recov 0.063 < 0.50), K2 FAIL (delta 0.0 < 2pp), K3 pass-but-moot -> all_pass=False -> KILLED.
- results.json verdict KILLED == PAPER.md verdict KILLED. No PROVISIONAL/PARTIAL contradiction.

## Premise attribution
B-C = 20.0pp (66.2->46.2): composition genuinely interferes. Kill = temporal-localization
reframe REFUTED, not no-interference-to-explain. PAPER.md attributes this correctly.

## Non-blocking note
- Experiment dir untracked in git, so KC-immutability not provable from history; mitigated by
  DB pre-registration matching MATH.md and code exactly. n=80 >= 15. q_proj-only caveat disclosed.
