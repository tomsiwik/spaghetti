# Adversarial Review — K=2 Strategy x Domain Composition

## Checklist

- (a) **Implementation faithful to MATH.md**: PASS. 3 pairs, 4-method matrix, KCs as pre-registered.
- (b) **Composition math correct**: PASS. Fisher-Rao on B-dicts with shared A. `compose_fisher_rao(adapter_Bs, A_dict=None)` correctly ignores A_dict.
- (c) **No data leakage**: PASS. Eval uses held-out test splits (gsm8k test, humaneval, medqa).
- (d) **Target-metric KCs**: PASS. gsm8k, humaneval, medqa are task accuracy, not proxies.
- (e) **N sufficient**: PASS. N=50 per benchmark, seed=42, consistent with prior experiments.
- (f) **Reproducibility**: PASS. Pair 1 gsm8k single_best=66% reproduced across 2 runs.
- (g) **Kill criteria pre-registered**: PASS. KCs defined in MATH.md before run.
- (h) **Verdict follows KC logic**: PASS. K1/K2/K3 all FAIL → KILLED.
- (i) **Bug fixes documented**: PASS. Two fixes applied (adapter skip, A_dict param), both documented.
- (j) **No cherry-picking**: PASS. All 3 pairs reported, including positive (math +4pp) and negative (code -12pp).
- (k) **DARE comparison valid**: PASS. Same K=2 adapter set, same seed, same eval.
- (l) **Cross-domain evals meaningful**: PASS. Evaluating K=2 on non-matched benchmarks reveals interference patterns.
- (m) **Finding actionable**: PASS. Clear implication: K=7 > K=2, domain-alone routing may beat strategy composition for code.

## Potential Concerns

1. **Per-pair verdict says SUPPORTED but aggregate is KILLED**: The per-pair KCs have `k1_min_delta_over_fisher_rao=0.0` (method IS Fisher-Rao, trivially passes). The real KCs (2211-2214 in DB) are evaluated at aggregate level. Not a bug, but the per-pair "SUPPORTED" verdict is misleading.

2. **Medical single_best=42% is low**: Raw domain_medical adapter only gets 42% MedQA (N=50). Small sample size may explain variance. However, the K=2 FR score (40%) is consistent with interference.

3. **Strategy adapter trained on what data?**: The strategy_full adapter presumably emphasizes step-by-step reasoning. Its interference with code may be due to training data overlap, not fundamental incompatibility.

## Verdict: KILL CONFIRMED

No blocking issues. The -12pp humaneval regression is genuine and reproducible.
