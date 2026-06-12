# PAPER — exp_spark_base_wins_bridges (REVISED per REVIEW-adversarial; re-run task-22 complete)

## Status
SUPPORTED on the revised harness (real run, `is_smoke:false`, total_time 4482 s, n=60 two-domain +
n=40 each pre-gate). The original SUPPORTED verdict used `best_single = max(math, code)` and OMITTED a
base-alone arm on the solve() task. This re-run adds `single-BASE` on the identical solve()
prompt/executed scorer and redefines `best_single = max(base_solve, math_solve, code_solve)`. The
kill-2311 thresholds are UNCHANGED. Crucially, base-alone is now the strongest single arm — and the
router still clears clause 1.

## Reframed claim (matches the data, not the original hunch)
The result is **"entropy-argmin 3-way routing where the frozen base usually wins,"** NOT "the base wins
the inter-domain bridge tokens." The base won **76.63%** of ALL emitted tokens, and the predicted bridge
pivots (`def`, `return`, `=`, `:`, newlines, punctuation) were only **17.42%** of those base wins.
Bridges are NOT the locus; the base wins the bulk of ordinary scaffold/prose tokens while the adapters
fire on their confident domain spans.

## Pre-gate (clause 3, evaluated FIRST) — own-domain EM vs base, n=40 each
| Domain | Base EM | Adapter EM | Delta (pp) |
|--------|---------|------------|------------|
| Math (math-adapter on math) | 0.125 | 0.600 | +47.5 |
| Code (code-adapter on code) | 0.000 | 0.125 | +12.5 |

Clause 3: `math_adapter_EM_math (0.600) > base_EM_math (0.125)` AND
`code_adapter_EM_code (0.125) > base_EM_code (0.000)` — both hold. `c3_pass = true`.

## Two-domain task — ALL FOUR arms (n = 60 GSM8K-as-`solve()`, executed scorer)
| Arm | EM | EM (count/60) |
|-----|-----|---------------|
| Base-alone (`single-BASE`) | 0.4333 | 26 |
| Math-only single adapter | 0.3500 | 21 |
| Code-only single adapter | 0.1167 | 7 |
| **best_single = max(base, math, code)** | **0.4333 (= base)** | 26 |
| 3-way entropy-argmin router | **0.6167** | 37 |

best_single_arm = **base** (0.4333). The new base-alone arm is now the strongest single arm — exactly
the live risk the review flagged — yet the router still beats it.

## Prediction vs measurement
- **Clause 1 (router lift vs best_single):** `router_EM (0.6167) − best_single_EM (0.4333) =
  +18.33pp` ≥ +3.0pp threshold. PASS. Because best_single = base here, this is also the implicit
  router-vs-base-alone test: `router_lift_vs_base_pp = +18.33pp` ≥ +3.0pp. PASS.
- **Clause 2 (bridge regime):** `base_win_fraction = 0.7663` (33240 / 43378 emitted tokens) ≥ 0.15.
  PASS. (Bridge pivots are 17.42% of base wins — descriptive only, not the mechanism.)
- **Clause 3 (pre-gate):** both adapters net-positive on own domain (above). PASS.
- **Non-collapse:** `router_items_differ_from_base = 60` of 60 (`router_frac_differ_from_base = 1.0`,
  `router_collapsed_to_base = false`). The router output differs from base-alone on every item, so the
  +18.33pp lift is NOT an artifact of the router silently echoing base tokens. PASS.

## kill-2311 clauses tested — which fired
All three clauses of kill-2311 were evaluated, plus the non-collapse guard:
| Clause | Test | Value | Threshold | Result |
|--------|------|-------|-----------|--------|
| C1 | router_EM − best_single_EM | +18.33pp | < +3.0pp ⇒ kill | did NOT fire |
| C2 | base_win_fraction | 0.7663 | < 0.15 ⇒ kill | did NOT fire |
| C3 | adapter > base on own domain (both) | +47.5pp / +12.5pp | ≤ 0 ⇒ kill | did NOT fire |
| non-collapse | router differs from base ≥1 item | 60/60 | 0 ⇒ artifact | did NOT fire |

No kill-2311 clause fired. `all_pass = true`, `verdict = "supported"`.

## Verdict line
**SUPPORTED.** router_EM 0.6167 − best_single_EM 0.4333 = +18.33pp ≥ +3.0pp (clause 1); base_win_fraction
0.7663 ≥ 0.15 (clause 2); pre-gate +47.5/+12.5pp (clause 3); router differs from base on 60/60 items
(non-collapse). best_single = base-alone (0.4333), so the lift is measured against the strongest single
arm, not an inflated max-of-adapters. The reviewer is the sole gate that seals the verdict.

## Caveats (REVIEW fix 3 — mandatory, do not omit in any downstream claim)
1. **3x compute.** The router runs THREE forward passes per emitted token (base + math + code) vs ONE
   for each single arm. The +18.33pp win is at 3x decode cost; NOT a compute-matched lift.
2. **Single seed.** Seed = 42 only. No seed variance characterized.
3. **Single task, small n.** One task family (GSM8K-as-solve()), n = 60; the executed-Python scorer
   couples math correctness and code runnability into one number.
4. **Replication required before any arc-level claim.** Replicate across ≥ 2 more seeds and ≥ 1
   additional two-domain task before any "first composition win" claim. As-is, a single-point existence
   result, not a general finding.
