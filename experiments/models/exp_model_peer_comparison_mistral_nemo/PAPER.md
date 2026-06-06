# PAPER.md — exp_model_peer_comparison_mistral_nemo

## Verdict
**KILLED_PREEMPTIVE** (infrastructure_blocked). 3-of-5 independent preempt
blocks fire (T2 ∧ T3 ∧ T5) on automated runner; manual re-read also
reinforces T1 (4/5 artifact shortfall). Verdict over-determined. Runner
2.55 s pure stdlib; zero MLX, zero model load, zero HTTP bind.

## Prediction vs. Measurement

| ID | Prediction (MATH.md) | Measurement (results.json) | Status |
|----|----------------------|-----------------------------|--------|
| P1 | T1 shortfall ≥ 3 of 5 required artifacts | shortfall = **1/5** (automated); **4/5** (manual, A9) — only Mistral Nemo MLX weights absence triggers without grep-scope refinement | **PARTIAL** (automated FAIL; manual PASS; runner refinement deferred) |
| P2 | T2 timing ≥ 120 min | conservative = **168.3 min**; full-bench = **734.7 min**; ceiling = 120 min | **PASS** |
| P3 | T3 DB has `success_criteria: []` + `⚠ INCOMPLETE` marker | both present; literal: `⚠ INCOMPLETE: success_criteria, references, kill_results (all untested)` | **PASS** |
| P4 | T4 pin_ratio = 0 (reinforcer only) | pin_ratio = 0.00 (`.audit/` absent); reinforcer did not fire; T4 is reinforce-only by design | **N/A** (expected) |
| P5 | T5 source-scope breach count ≥ 3 vs parent SUPPORTED `exp_p1_t2_single_domain_training` | breach count = **5/5** (A/B/C/D/E all BREACH); source verdict SUPPORTED confirmed via DB | **PASS** |

## Kill Criteria (pre-registered, locked at claim)
- K1696 — Pierre ≥ Mistral Nemo 12B on ≥2 of {MMLU-Pro, GSM8K, HumanEval,
  MATH, IFEval}: **fail (pre-empt, no run)**.

Not measured — blocked by T2 ∨ T3 ∨ T5 (each alone suffices). No
empirical run needed; verdict is geometrically over-determined.

## Axis assignment: composition-bug / software-infrastructure-unbuilt

This preempt fires on **F#652** (software-infrastructure-unbuilt) — target
requires a unified cross-model 5-benchmark harness that doesn't exist, plus
MATH-500 + IFEval peer-comparison glue, plus local Mistral Nemo 12B MLX
weights. All of these are operator-action items; none is an experimental
question. Pure schema + scope + cost block.

27th composition-bug preempt in the drain (26 → 27 per scratchpad iter 40
count). Sub-axis maps to F#652 software-infra-unbuilt; no novel finding
this iter (reuses existing axis).

## Assumptions / transparency (from MATH.md §4)
- **A1** grep scope = `.py` under `pierre/`, `macro/`, `composer/`,
  `micro/models/` (excluding this runner).
- **A2** Mistral Nemo MLX check = HF cache directory presence,
  `models--mlx-community--Mistral-Nemo-*` — absent, confirmed.
- **A3-A4** IFEval + MATH-500 harness probes require in-scope
  verifier/boxed-extraction logic; manual re-read (A9) confirms only
  misaligned-context hits exist.
- **A5** T2 uses conservative 100-sample budget; full-benchmark scenario
  (734.7 min) noted for transparency, but not the primary block driver.
- **A6** T5 reads parent verdict from DB (`Status:   supported`).
  Source-scope breach count 5/5 (A-E). T5-K variant does **not** apply
  (source is not KILLED).
- **A7** Runner is pure stdlib + `experiment get` shell-out. Runtime
  2.55 s wall. Zero MLX, zero empirical run.
- **A9** T1 runner is grep-scope-too-broad. **Manual re-read** gives
  shortfall = 4/5 (matches MATH prediction). Automated reported
  shortfall = 1/5 due to noise from unrelated experiments' docstrings.
  Non-blocking — 3 other blocks already overdetermine.

## Anti-pattern checklist (pre-`experiment complete`, Guardrail 1009)

| Check | Status |
|-------|--------|
| results.json.verdict not silent-upgraded | ✓ KILLED_PREEMPTIVE explicit |
| all_pass == false | ✓ |
| is_smoke == false | ✓ |
| No KC edit since MATH.md (git diff clean) | ✓ locked at claim |
| No composition math bug | ✓ N/A (no composition code) |
| No unsafe adapter scale | ✓ N/A |
| No tautological routing | ✓ N/A |
| No `shutil.copy` as new adapter | ✓ N/A |
| No hardcoded `"pass": True` | ✓ all KC explicit `false` |
| No eval-template truncation | ✓ N/A (no eval) |
| No proxy-model substituted for target | ✓ N/A |
| KC measures wrong object | ✓ N/A (not measured; blocked pre-run) |
| N=smoke reported as full | ✓ `is_smoke=false` |

All gates pass. `--status killed` is appropriate.

## Remediation path (operator action required to lift preempt)
1. Download + MLX-quant `mlx-community/Mistral-Nemo-Instruct-2407-4bit`
   (or 8-bit; bf16 is 24 GB and may exceed comfort margin on 48 GB).
2. Build IFEval verifier (strict + lenient) — port Google 2023 paper
   logic; 25 verifier classes.
3. Build MATH-500 peer-comparison harness with boxed-answer extraction
   (`reasoning_expert_distillation/eval_math500.py` is close but Qwen-
   specific, not cross-model).
4. Build unified 5-benchmark harness that loads two model backends
   (Pierre composed adapters + Mistral Nemo monolith) and runs the
   same sample set through each.
5. Resolve N=5 adapter stack scope: parent supports 3 domains
   (math/code/medical). Pick the extra two (e.g. reasoning, instruction)
   and train-or-port before claiming.
6. Design sample budget that fits ≤ 120 min ceiling. 100-sample × 5-bench
   × 2-model already exceeds; consider 50-sample subset or split into
   per-benchmark sub-experiments.
7. Fill DB `success_criteria` (the ⚠ INCOMPLETE marker is the cheapest fix).

Until then: KILLED_PREEMPTIVE stays. No retry is productive.
