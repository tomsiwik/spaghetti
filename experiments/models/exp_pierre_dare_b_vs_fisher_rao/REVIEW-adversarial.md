# REVIEW-adversarial.md — exp_pierre_dare_b_vs_fisher_rao

**Verdict: KILL (confirmed)**

## Adversarial Checklist

| Check | Result | Note |
|-------|--------|------|
| (a) results.json verdict matches DB | FLAG | results.json="INCONCLUSIVE", DB/PAPER="KILLED". Override justified: K4 was miscalibrated (see below), K1 is unambiguous at -9.3pp. |
| (b) all_pass matches claim | PASS | All 4 KCs fail in results.json, consistent with KILLED. |
| (c) PAPER.md verdict matches DB | PASS | Both say KILLED. |
| (d) is_smoke → provisional | N/A | |
| (e) KC modified after first run | PASS | KCs in results.json match MATH.md pre-registration. |
| (f) Tautological KC | PASS | K1-K3 are meaningful comparisons. K4 miscalibrated but not tautological. |
| (g) Code measures what MATH.md describes | PASS | Composition methods implement exactly as specified. |
| (h) Independent A/B summation bug | PASS | B-only methods sum B correctly. Full-delta does `Σ(scale * A_k @ B_k)` — correct. |
| (i) LORA_SCALE unsafe | PASS | scale=6.0 from config. |
| (j) Single-sample routing | N/A | No routing in this experiment. |
| (k) shutil.copy fake adapter | PASS | No. |
| (l) Hardcoded pass | PASS | No. |
| (m) Model mismatch | PASS | MATH.md, code, and results.json all use `gemma-4-e4b-it-4bit`. |
| (n) Truncated eval | PASS | Non-zero scores across all benchmarks. |
| (o) n < 15 | PASS | N=50 per benchmark. |
| (p) Target-metric KC | PASS | K1-K3 are task accuracy, not proxy. |

## K4 Miscalibration Note

K4 compared Fisher-Rao avg (64.7%) against a 73.3% reference from full-delta DARE in a different experiment with different adapter set, eval N, and composition method. This is apples-to-oranges. The code correctly flags INCONCLUSIVE when K4 fails, but the researcher's override to KILLED is sound: the K1 decision criterion (-9.3pp) is the primary question and its answer is unambiguous. Future experiments should not reuse cross-method references as reproducibility gates.

## Code Quality

Composition math is correct. No A/B factoring bugs. DARE drop+rescale logic matches the paper (arxiv 2311.03099). The `_FusedDeltaLinear` pattern for M3 is clean. RNG seeding ensures reproducibility.

## Finding

B-space DARE (55.3%) loses to Fisher-Rao (64.7%) by 9.3pp. DARE's expectation preservation at B-level does not propagate through `A @ B` multiplicative interaction. Keep Fisher-Rao as Pierre default.
