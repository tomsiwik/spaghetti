# REVIEW: E6 — Systematic Strategy Adapter (Hedgehog Distillation)

## Verdict: KILL

Override: `is_smoke: true` → PROVISIONAL normally, but overridden to KILL because the failure is **method-level**, not sample-level. Justification:

1. **K2028 FAIL (target)**: 0/4 domains improved. All 4 degraded: math -15pp, science -15pp, STEM -5pp, general -10pp.
2. **K2029 FAIL (target)**: 0 domains improved (threshold: 2+).
3. **MMLU non-interference FAIL**: 10pp drop on general knowledge.
4. **GSM8K catastrophic failure**: adapter outputs are ALL null — zero extractable numbers from 20 samples. This is not variance; it's a format override failure.
5. **F#666 tautological-proxy confirmed**: K_struct PASS (cos=0.96) + K2028 FAIL (all domains negative). Attention matching succeeds structurally but is antagonistic behaviorally.

The failure mechanism is clear and structural: Hedgehog per-layer cos-sim distillation transfers attention patterns from teacher input processing, but reasoning strategy effects manifest during generation, not input processing. More data cannot fix this — the training objective is misaligned with the behavioral goal.

## Adversarial Checklist

| Check | Result |
|-------|--------|
| (a) verdict consistency | PASS — results.json="PROVISIONAL", overridden to KILL per method-level failure |
| (b) all_pass vs claim | PASS — `all_pass: false` |
| (c) PAPER.md verdict | PASS — says "PROVISIONAL with strong kill signal" |
| (d) is_smoke vs claim | PASS — override documented above |
| (e) KC mutation | PASS — first run |
| (f) Tautology sniff | PASS — K2028/K2029 are target metrics |
| (g) K-ID code↔math | PASS |
| (h) Composition bugs | N/A — single adapter |
| (i) LORA_SCALE | PASS — 6.0 |
| (j) Routing | N/A |
| (k) shutil.copy | PASS — not present |
| (l) Hardcoded pass | PASS |
| (m) Model match | PASS |
| (m2) Skill invocation | PASS — /mlx-dev, /fast-mlx cited; code is idiomatic MLX |
| (n) Base acc 0% | PASS — GSM8K base=15%, others 75-85% |
| (o) N < 15 | PASS — N=20 |
| (r) Pred-vs-meas table | PASS |
| (t) Target-gated kill | PASS — target KCs (K2028, K2029) both FAIL |

## Assumptions

- Overriding smoke→PROVISIONAL to KILLED follows the same precedent as E1 (F#801) and E2 (F#802/F#803): when ALL target KCs fail with a structural mechanism explanation, N=20 is sufficient to identify a method-level defect.
