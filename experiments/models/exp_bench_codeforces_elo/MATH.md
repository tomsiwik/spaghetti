# MATH: Codeforces ELO Baseline

## Theorem 1: Proxy Validity
CodeElo (arXiv:2602.05891) measures Codeforces ELO by actual AC/WA submissions.
LiveCodeBench (arXiv:2403.07974) includes Codeforces-sourced competitive programming problems.
**Prediction**: LiveCodeBench pass@1 on competitive programming subset ≈ f(ELO),
where f is monotone increasing. This is a frontier-extension (unknown constant).

## Kill Criteria
- K1423: Base E4B ELO >= 800 (within 140 of Google's 940)
- K1424: Pierre code adapter ELO >= base + 50
- K1425: Stable ELO (std < 100 over 3 runs)

## Impossibility Structure
CodeElo requires Codeforces account + submission token via email to binyuan.hby@alibaba-inc.com.
Without token, direct ELO measurement is **impossible** on this system.
Proxy: LiveCodeBench competitive programming scores (already planned as exp_bench_livecodebench_v6).
ELO variance is ±394 per arXiv:2602.05891 — making K1425 (std < 100) likely to FAIL even with token.

## Status: KILLED — external service dependency
