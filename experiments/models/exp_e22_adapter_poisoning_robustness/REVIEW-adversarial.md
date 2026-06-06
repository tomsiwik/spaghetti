# E22 Review: Adapter Poisoning Robustness

## Verdict: PROVISIONAL

## Adversarial Checklist

| Item | Status | Notes |
|------|--------|-------|
| (a) verdict consistency | PASS | results.json="PROVISIONAL", is_smoke=true, both KCs pass |
| (b) all_pass consistency | PASS | all_pass=true matches PROVISIONAL claim |
| (c) PAPER.md verdict | PASS | "PROVISIONAL (smoke, N=20 QA, 3 layers)" |
| (d) is_smoke flag | PASS | is_smoke=true → PROVISIONAL, not SUPPORTED |
| (e) KC mutation | PASS | New experiment, no prior commits |
| (f) tautology | PASS | Both KCs are behavioral (QA accuracy), non-trivial thresholds |
| (g) KC code↔math match | PASS | worst_grass_drop < 30.0 and best_margin > 2.0 match MATH.md |
| (h) composition bug | PASS | Manual B @ A summation, no peft shortcuts |
| (i) LORA_SCALE | PASS | LORA_SCALE=6, within safe range |
| (j) routing | N/A | No routing |
| (k) shutil.copy | N/A | |
| (l) hardcoded pass | PASS | No hardcoded results |
| (m) model match | PASS | MATH.md=Gemma 4 E4B, code=mlx-community/gemma-4-e4b-it-4bit |
| (m2) skill invocation | PASS | /mlx-dev, /fast-mlx invoked; mx.eval/mx.clear_cache present |
| (n) base accuracy | PASS | 85.0% — healthy baseline |
| (o) sample size | PASS | N=20 ≥ 15 |
| (p) synthetic padding | N/A | |
| (q) cited baseline | N/A | |
| (r) pred-vs-meas table | PASS | Present in PAPER.md |
| (s) math correctness | PASS | Theorem correctly derives partial input-space containment |
| (t) target-gated kill | PASS | Both KCs are behavioral (QA accuracy) |
| (u) scope-changing fixes | PASS | Chat template + magnitude sweep are legitimate bug fixes |

## Assessment

Clean smoke pass. Both KCs behavioral:
- K2055 PASS: 25pp worst drop < 30pp threshold
- K2056 PASS: 55pp best protection margin > 2pp threshold

Key result: Grassmannian provides 55pp protection margin at 10× poison — far exceeding the 5-15pp prediction from MATH.md. F#815's B-matrix coupling prediction was correct for activation cosine but wrong for behavioral outcomes. The mechanism is input-space feature isolation, not output-space decorrelation.

is_smoke=true → PROVISIONAL. Full run needed with 35 layers, 5 clean adapters, 100 QA questions.
