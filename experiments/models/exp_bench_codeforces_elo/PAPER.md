# PAPER: Codeforces ELO Baseline

## Prediction vs Measurement

| Kill Criteria | Prediction | Measurement | Status |
|---|---|---|---|
| K1423: Base ELO >= 800 | ~700-900 range | NOT MEASURED | KILLED — needs CodeElo token |
| K1424: Code adapter ELO >= base + 50 | Likely pass | NOT MEASURED | KILLED |
| K1425: Stable ELO (std < 100) | FAIL (variance ±394) | NOT MEASURED | KILLED |

## Results

**Status**: Deferred — Codeforces ELO requires email registration for CodeElo submission token
(binyuan.hby@alibaba-inc.com, arXiv:2602.05891).

**Alternative**: exp_bench_livecodebench_v6 uses LiveCodeBench competitive programming problems
as a proxy for Codeforces performance. Already queued as pueue task (exp_bench_livecodebench_v6).

## Impossibility Note

Even with a token, K1425 (std < 100) is likely to FAIL: per arXiv:2602.05891, Codeforces ELO
has ±394 variance due to submission order sensitivity. The benchmark design has a structural
flaw that cannot be fixed without running hundreds of evaluations.

## Conclusion

Kill: all three criteria untestable without external service registration.
Superseded by: exp_bench_livecodebench_v6 (already designed and queued).
