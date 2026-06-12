# REVIEW (adversarial) — exp_wildcat_dilution_norm_vs_structure

## Reality checks — PASS
- `verify-experiment.sh` exit 0. `is_smoke:false`, 18,420s wall clock, 8 arms x 100 real
  greedy generations with per-sample gold/pred/ntok. Real model loads (gemma-4-e4b-it-4bit),
  real adapter files, distinct per-arm Frobenius norms (A 3.161, B 2.081, C 2.081, C2 1.370,
  D 3.080, E 0.431) — not a mock, not a copy.
- Consistency: results.json `verdict: provisional`, `all_pass: false`, PAPER verdict line
  PROVISIONAL — all agree. No kill fired (C=0.79 < B_mean-3pp=0.81).
- Integrity: thresholds in MATH.md == constants in run_experiment.py == PAPER. Dir is untracked
  (no git history to audit), but the pre-registered prediction was the OPPOSITE outcome
  (predicted pure-norm/killed, EM_B~0.66; measured EM_B=0.84, gap on structural side), which
  argues against post-hoc threshold tuning.

## Findings against the result
1. **Boundary is a float artifact, both ways.** Exact arithmetic: mean(B)=252/300=0.84,
   gap = 0.84-0.79 = exactly 1/20 = 5.0pp, which *meets* the pre-registered `>= 5pp` supported
   criterion; the code recorded 0.04999... and classified provisional. I do NOT upgrade:
   a knife-edge tie at n=100 (per-arm SE ~4pp; per-seed McNemar n.s.; pooled paired
   discordants 26 vs 11, p~0.02) is not "cleanly crossed". Provisional is the honest read.
2. **PAPER error:** "B/C/D/E all ≈ 2.081 — norm matching held" is false. D=3.080, E=0.431.
   Only B vs C is norm-matched.
3. **Confounded secondary claim.** The "outlier-driven" ordering E(0.92) > B(0.84) > D(0.68)
   is exactly what pure norm reduction also predicts, since D carries ~97% of the dense norm
   and E ~14%. Across all arms EM is monotone in ||delta||_F (3.16:0.60, 3.08:0.68, 2.08:0.79-0.87,
   1.37:0.84, 0.43:0.92). D/E probes as run cannot separate structure from norm; the ONLY
   norm-controlled evidence is the 5.0pp B-vs-C gap. PAPER's structural narrative overreaches.

## Route
**PROVISIONAL** — real run, no kill fired, primary result sits exactly on the pre-registered
boundary, secondary evidence confounded. Required follow-up before any supported claim:
(a) n>=300 or paired-significance pre-registration on B vs C; (b) norm-matched D'/E'
(rescale D and E to ||C||_F=2.081) so the structure probe is not a norm proxy.

## Re-review (2026-06-11) — after PAPER revision
- Finding 2 fixed: PAPER now states only B vs C is norm-matched (D=3.080, E=0.431, C2=1.370).
- Finding 3 fixed: D/E/C2 explicitly labeled norm-confounded, monotone-in-norm pattern stated,
  no structural claim drawn from secondary arms; follow-up (n>=300 B-vs-C, norm-matched D'/E') listed.
- Finding 1 unchanged by design: knife-edge 5.0pp gap at n=100 stays PROVISIONAL. Correct.
- verify-experiment.sh exit 0; results.json verdict/all_pass/is_smoke consistent with PAPER.
**ROUTE: PROVISIONAL — confirmed.**
