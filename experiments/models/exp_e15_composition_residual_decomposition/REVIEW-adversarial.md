# REVIEW-adversarial.md — E15: Composition Residual Decomposition

## Verdict: KILL

## Adversarial checklist

| Check | Result | Note |
|-------|--------|------|
| (a) verdict consistency | PASS | results.json KILLED, DB killed |
| (b) all_pass vs claim | PASS | all_pass=false, killed |
| (c) PAPER.md verdict | PASS | "KILLED (smoke, method-level)" |
| (d) is_smoke vs claim | PASS | smoke, killed (no full-run claim) |
| (e) KC mutation | PASS | first run |
| (f) tautology | PASS | K2046_quality vacuously passes (norm_retention=1.0 at full rank) but does not drive verdict |
| (g) K-ID match | PASS | all 4 KCs match MATH.md ↔ code |
| (h) composition bugs | PASS | no sum lora_A/B |
| (i) LORA_SCALE | PASS | =6 |
| (j) routing | PASS | N/A |
| (k) shutil.copy | PASS | N/A |
| (l) hardcoded pass | PASS | data-driven |
| (m) model match | PASS | gemma-4-e4b-it-4bit both |
| (m2) skill invocation | PASS | /mlx-dev, /fast-mlx cited; mx.eval, mx.clear_cache present |
| (n) base accuracy | PASS | 10% — low but not driving verdict |
| (o) headline n | PASS | smoke N=10 appropriate |
| (p) synthetic padding | PASS | N/A |
| (q) baseline drift | PASS | N/A |
| (r) prediction table | PASS | present in PAPER.md |
| (s) math errors | PASS | B_i = W @ A_i^T sharing W structure is correct |
| (t) target-gated | PASS | K2045+K2045_target, K2046+K2046_quality — both proxy AND target FAIL |
| (u) scope-changing fixes | PASS | none |

## Override: is_smoke → KILL

Failure is method-level, not sample-level. Near-uniform SVD spectra (σ₁/σ₆ ≈ 1.2, top-3 energy 55% vs 50% uniform) are a structural consequence of B_i = W @ A_i^T projecting through a shared base weight W. Rank filtering concentrates representation in W's dominant directions, increasing not decreasing cross-adapter coupling. More layers or prompts cannot change this. Same precedent as E1/E2/E3/E4/E6.

## Key finding

F#817: SVD filtering of B matrices is structurally incapable of reducing composition interference. The path to reducing tau requires B-matrix orthogonality (training-time regularizer, independent training data, or post-hoc rotation), not post-hoc spectral filtering.
