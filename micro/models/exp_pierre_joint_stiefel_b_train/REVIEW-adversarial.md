# REVIEW — exp_pierre_joint_stiefel_b_train

**Verdict: KILL**

## Adversarial Checklist

| # | Check | Result |
|---|-------|--------|
| (a) | results.json verdict matches DB status | PASS — both KILLED |
| (b) | all_pass matches claim | PASS — false, K3+K4 fail |
| (c) | PAPER.md verdict matches | PASS — KILLED |
| (d) | Smoke → provisional | N/A — is_smoke=false |
| (e) | KC not modified post-run | PASS — untracked dir, KCs consistent across MATH.md/code/results |
| (f) | No tautological KC | PASS — K1 joint-vs-independent, K3 NLL perturbation, K4 composed accuracy |
| (g) | Code measures what MATH.md describes | PASS — all 4 KCs implemented as specified |
| (h) | Composition math correct | PASS — `Σ (A_j @ B_j) / K` at L456-459, not `(ΣA) @ (ΣB)` |
| (i) | LORA_SCALE safe | PASS — scale=6.0 |
| (j) | Routing non-tautological | PASS — round-robin by domain, each adapter trained on own data |
| (k) | No adapter copying | PASS |
| (l) | No hardcoded pass | PASS |
| (m) | Model matches | PASS — gemma-4-e4b-it-4bit in both MATH.md context and code |
| (n) | Base accuracy not 0% | PASS — individual scores 38-100% |
| (o) | n ≥ 15 | PASS — N_EVAL=50 |
| (p) | Target-metric KC present | PASS — K1 (accuracy), K4 (composed accuracy) |

## Notes

- Biology 501k% perturbation is an artifact of near-zero single-adapter NLL (1.56e-5), not a bug. Even well-conditioned domains (code: 3.4%, math: 18.8%) exceed the 1% threshold.
- PAPER.md correctly identifies the structural gap: B-orthogonality ≠ functional non-interference because A matrices are unconstrained. The analysis is sound.
- K1+K2 passing validates joint Stiefel training as a mechanism — the constraint doesn't hurt individual adapters and achieves near-perfect orthogonality (1.89e-07). This is a reusable finding.
