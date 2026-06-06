# Adversarial Review: exp_bench_codeforces_elo

**Verdict: KILL CONFIRMED**

## Summary

Kill is mathematically sound and well-documented. No REVISE needed.

## Issues Found

### Blocking
None.

### Non-Blocking

1. **K1423/K1424 untestable** — CodeElo requires manual email registration
   (binyuan.hby@alibaba-inc.com) for a Codeforces submission token. This is a
   permanent external dependency with no workaround on this system. Both criteria
   require actual AC/WA submissions to Codeforces. Kill is correct.

2. **K1425 structurally impossible** — Per arXiv:2602.05891, Codeforces ELO has
   ±394 variance due to submission-order sensitivity. Requiring std < 100 is
   ~4× tighter than the mechanism allows. Even with infinite compute, this criterion
   cannot be met with a reasonable number of runs. The impossibility structure is
   correctly derived in MATH.md.

3. **Proxy already designed** — exp_bench_livecodebench_v6 uses competitive programming
   problems from Codeforces as a proxy. This is the correct superseding experiment.
   No information is lost; the LiveCodeBench proxy provides a comparable signal
   without external service dependencies.

## PAPER.md Assessment

Prediction-vs-measurement table is present and accurate. All three criteria correctly
labeled "KILLED" with explanations. No fabricated results.

## Mathematical Soundness

MATH.md correctly identifies: (1) external service dependency, (2) intrinsic ELO
variance of ±394 makes K1425 impossible, (3) LiveCodeBench as valid proxy.
Theorem 1 (proxy validity) is exploratory but reasonable — pass@1 on competitive
programming problems is monotone in skill. No errors.

## Conclusion

Kill is valid and clean. LEARNINGS.md should capture the impossibility structure
(±394 ELO variance, submission-order sensitivity) so future benchmark design avoids
ELO-stability criteria on auto-evaluation systems.
