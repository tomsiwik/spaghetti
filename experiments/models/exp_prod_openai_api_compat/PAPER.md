# PAPER.md — exp_prod_openai_api_compat

## Verdict
**KILLED_PREEMPTIVE** (infrastructure_blocked). 4-of-5 independent preempt
blocks fire (T1 ∧ T2 ∧ T3 ∧ T5-K). Verdict over-determined. Runner 1.52 s
pure stdlib; zero MLX, zero HTTP bind, zero model load.

## Prediction vs. Measurement

| ID | Prediction (MATH.md) | Measurement (results.json) | Status |
|----|----------------------|-----------------------------|--------|
| P1 | T1 shortfall ≥ 3 of 4 required artifacts | shortfall = **4/4** | **PASS** |
| P2 | T2 timing ≥ 120 min | **138.0 min** (16 × 3 × 3 × 45 s + 1800 s cold-start) vs 120 min ceiling | **PASS** |
| P3 | T3 DB has `success_criteria: []` + `⚠ INCOMPLETE` | both present, evidence line literal `⚠ INCOMPLETE: success_criteria, references, kill_results (all untested)` | **PASS** |
| P4 | T4 pin_ratio ≥ 0.60 (reinforce only) | pin_ratio = **0.00** (`.audit` directory absent) — does not reinforce, but does not block either | N/A (reinforcer did not fire; T4 is reinforce-only by design, not a gate) |
| P5 | Parent `exp_prod_mlxlm_integration.verdict = KILLED` with ≥ 3 preconditions | parent verdict = **KILLED**, **5 failed** preflight keys (T1B, T1C, T2, T3, DEP), dependency declared in target YAML | **PASS** |

## Kill Criteria (pre-registered, locked at claim)
- K1682 — stock OpenAI client + streaming round-trip: **fail (pre-empt, no run)**
- K1683 — adapter selection via header or `extra_body`: **fail (pre-empt, no run)**
- K1684 — tools / response_format / logprobs / stream_options parity: **fail (pre-empt, no run)**

All three unmeasured — but each is blocked by one or more of the 4 preempt
theorems. No empirical run needed; verdict is geometrically over-determined.

## Novel sub-axis this iteration: T5-K (parent-KILLED inheritance)

Distinct from the standard T5 source-scope breach theorem, which presumes
a **SUPPORTED** source with a narrower scope the child aims to extend.
When source is KILLED (as here), source-scope has no positive extent; the
child automatically inherits all parent preconditions that are not
independently resolved between parent kill and child claim.

Parent `exp_prod_mlxlm_integration` was killed on 5 preconditions:
  a. mlx-lm 0.31.2 has no plugin/loader API (T1B)
  b. no `pierre-g4e4b` model registered locally or in HF cache (T1C)
  c. server body schema validates `adapters` as `str`, not multi-adapter (T2)
  d. trained adapter safetensors missing (math/code/medical, 0 bytes) (T3)
  e. transitive grandparent `exp_prod_pip_package_pierre` is KILLED (DEP)

All 5 remain unresolved at this experiment's claim moment. The current
experiment's 3 KCs depend directly on (a), (b), (c), (d). No independent
resolution exists. **T5-K blocks.**

Proposed as a new F-finding sub-axis under F#652 (software-infrastructure-
unbuilt) per analyst arbitration when cap lifts.

## Assumptions / transparency (from MATH.md §4)
- **A1** grep scope limited to `pierre/**/*.py`, `macro/**/*.py`,
  `composer/**/*.py` (source of truth for the claim).
- **A2-A4** literal-decorator matching for FastAPI/Starlette; excludes
  markdown plans.
- **A5** T2 timing uses 45 s avg per OpenAI-parity call; compose reload is
  conservative (pro-preempt). Formula: `16*3*3*45s + 1800s cold-start`.
- **A6** T5-K uses parent `results.json.reason` and preflight block as
  canonical source, not PAPER prose.
- **A7** Runner is pure stdlib + shelled `experiment get`. Zero MLX, zero
  empirical run. Runtime 1.52 s wall.

## Anti-pattern antipattern checklist (pre-`experiment complete` per
Guardrail 1009)

| Check | Status |
|-------|--------|
| results.json.verdict not silent-upgraded | ✓ KILLED_PREEMPTIVE explicit |
| all_pass == false | ✓ |
| is_smoke == false | ✓ |
| No KC edit since MATH.md (git diff clean) | ✓ locked at claim |
| No composition math bug, no unsafe adapter scale, no tautological routing | ✓ N/A (no composition code) |
| No `shutil.copy` as new adapter | ✓ N/A |
| No hardcoded `"pass": True` | ✓ all KCs explicit `false` |
| No eval-template truncation | ✓ N/A (no eval) |
| No proxy-model substituted for target | ✓ N/A |
| KC measures wrong object | ✓ N/A (KC not measured; blocked pre-run) |
| N=smoke reported as full | ✓ `is_smoke=false`, not smoke-mode |

All gates pass. `--status killed` is appropriate.

## Remediation path (operator action required to lift preempt)
1. Build `pierre/server.py` with FastAPI `/v1/chat/completions` + SSE.
2. Build `pierre/cli.py` with `pierre serve` entry, register `console_scripts`.
3. Add `X-Pierre-Adapters` header handler + `extra_body.adapters` adapter
   selection, wire into Pierre adapter composition.
4. Train adapter safetensors for math/code/medical (addresses parent T3 too).
5. Resurrect parent chain (`exp_prod_pip_package_pierre`, then
   `exp_prod_mlxlm_integration`) before re-claim.

Until then: KILLED_PREEMPTIVE stays. No retry is productive.
