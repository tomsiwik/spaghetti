# LEARNINGS: exp_bench_codeforces_elo

## Status: KILLED

## Core Finding
Codeforces ELO benchmarking is impossible on this system: CodeElo requires manual
email registration for a submission token, and even with access, ELO stability
criteria (std < 100) are structurally impossible — arXiv:2602.05891 shows ±394
variance due to submission-order sensitivity.

## Why
ELO on competitive programming has two failure modes: (1) external service
dependency requiring human registration, and (2) intrinsic ELO variance of ±394
from submission-order sensitivity makes any std < 100 target ~4× tighter than
the mechanism allows. This is not a resource problem — it's an impossibility
structure: you cannot achieve ELO stability without running hundreds of evaluations,
which requires the token in the first place.

## Impossibility Structure
Per arXiv:2602.05891: Codeforces ELO variance = ±394 (submission-order sensitive).
To achieve std < 100 requires N > (394/100)² ≈ 15.5 independent ELO estimates,
each requiring fresh submissions. No workaround exists without the API token.

## Implications for Next Experiment
Use LiveCodeBench (exp_bench_livecodebench_v6) as the proxy: competitive programming
problems from Codeforces, measured via pass@1, no external service dependency.
pass@1 on competitive programming is monotone in coding skill, making it a valid
proxy. Do NOT design experiments with ELO-stability criteria on auto-evaluation
systems — the variance floor is too high.
