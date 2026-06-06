# exp_g4_canary_drift_detection — PAPER.md

**Verdict: KILLED (preempt, F#666-pure standalone)**

## Abstract

The pre-reg K1581 — "FNR ≤ 5% on synthetic-corrupted adapter" — is a single proxy-metric KC (False Negative Rate = classification accuracy per guardrail 1007's explicit enumeration) with no paired target-metric KC and `depends_on: []`. Parent F#156 (`exp_canary_quality_detection`) was target-anchored via mechanistic linkage `Degradation ~ f(rho)*g(cos)`, but that anchor is NOT inherited operationally by this pre-reg. Per F#666, running the experiment is guaranteed to produce an unidentifiable verdict. This filing is a preempt-KILL scaffold (no compute) per `reviewer.md §5 KILL (preempt-structural — F#666-pure standalone)` clause and precedents F#700/F#701/F#703/F#705.

## Prediction vs Measurement

| KC | Prediction | Measurement | Result |
|----|------------|-------------|--------|
| K1581 (FNR ≤ 5% on synthetic-corrupted adapter) | not measured — proxy-only KC unidentifiable under F#666 | not measured — no compute executed | **untested** |

No measurements were taken. MLX was not loaded. Gemma 4 was not loaded. No adapters were constructed. No canary queries were synthesized.

## Why this is KILLED (structural, not mechanism)

Exhaustive 2¹ truth table over K1581 ∈ {PASS, FAIL}:

| K1581 outcome | F#666 interpretation | Identifiability |
|---------------|---------------------|-----------------|
| PASS (FNR ≤ 5% on synthetic) | Tautological SUPPORT. Detector works on a SYNTHETIC distribution; says nothing about whether it detects compositions that actually degrade user-visible task accuracy. Parent F#156 established deployment regime `rho=0.89` — untested whether synthetic matches that regime. Reviewer applies antipattern-t. | Unidentifiable |
| FAIL (FNR > 5% on synthetic) | Per F#666: "proxy-FAIL + target-absent = a finding about the proxy, not a kill". Synthetic corruption may be unrealistically subtle (harder than production) or aggressive (easier than production). Behavioral quality drop is not established either way. | Unidentifiable |

Both outcomes are unidentifiable. The KC structure itself — not the mechanism — guarantees an ambiguous verdict. This is the F#666-pure standalone signature.

## Taxonomic row (drain-window position 5)

| # | Experiment | Pattern (proxy flavor) | Date | §5 clause status |
|---|------------|-------------------------|------|------------------|
| 1 | F#700 `exp_g4_per_layer_cos_baseline` | F#666-pure (cos-sim) | 2026-04-24 | promoted |
| 2 | F#701 `exp_adapter_orthogonality_audit` | F#666-pure (pairwise-cos + eff-rank) | 2026-04-24 | promoted |
| 3 | F#703 `exp_followup_tfidf_medical_unaliased` | F#666-pure (routing weighted-acc) | 2026-04-24 | promoted |
| 4 | F#705 `exp_g4_o1_removal_naive` | F#666-pure (PPL) | 2026-04-24 | already promoted, lexical-expansion |
| **5** | **`exp_g4_canary_drift_detection` (this filing)** | **F#666-pure (FNR = classification-accuracy on synthetic)** | **2026-04-24** | **already promoted, near-canonical** |

Delta at row 5: first drain-window instance where the proxy is **FNR on a detection classifier** — near-canonical to guardrail 1007's "classification accuracy" enumeration. Prior 4 rows exercised derived proxies. Row 5 confirms the clause applies to the canonical named case. Potential **taxonomy-refactor trigger** for analyst (5+ instances), though clause remains operationally correct without refactor.

## Unblock path

Re-register as `exp_g4_canary_drift_target_paired` with:
- **K1 (target, load-bearing):** On held-out real N=25 compositions that DEGRADE HumanEval PASS@1 by ≥ 3pp, canary TPR ≥ 95%.
- **K2 (target, false-positive bound):** On held-out N=25 compositions that DO NOT degrade (drop ≤ 1pp), canary FPR ≤ 10%.
- **K3 (proxy, sanity):** FNR ≤ 5% on synthetic corruption (the original K1581) — not load-bearing.
- **K4 (mechanistic anchor):** Pearson correlation between canary score and (rho·cos) ≥ 0.5 — inherits F#156's mechanism.

KILL requires K1 FAIL OR K2 FAIL. SUPPORTED requires K1 PASS + K2 PASS. See MATH.md §8 for the full yaml template.

**Do NOT patch K1581 via `experiment update`** — KC mutation post-claim is antipattern-u.

## Parent motivations untouched

- **F#156** (`exp_canary_quality_detection`, supported, 2026-03-28) — Canary FNR=2.0% CI[1.9%, 2.1%] with mechanistic linkage "Degradation ~ f(rho)*g(cos)". Status unchanged. This filing does not re-open F#156 — it rejects re-running the proxy-only test on Gemma 4 without inheriting the mechanism-anchor.
- **F#594** (`exp_composition_health_kl_divergence`, killed) — cautionary note: canary-passing adapters can still leak information (+15pp activation-probe advantage). Reinforces that canary-proxy-PASS alone is not a deployment guarantee.

## No `_impl` companion

Preempt-structural KILL excludes `_impl` per F#687/F#698/F#699/F#700/F#701/F#703/F#705 + `reviewer.md §5` F#666-pure clause. Unblock is pre-reg-external.

## Skills invocation disclosure

`/mlx-dev` and `/fast-mlx`: **Not invoked. No MLX code written.** `run_experiment.py` imports `json + pathlib` only. Canonical preempt form per F#700/F#701/F#703/F#705.

— End PAPER.md —
