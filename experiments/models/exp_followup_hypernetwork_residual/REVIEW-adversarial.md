# REVIEW-adversarial — exp_followup_hypernetwork_residual

**Verdict:** `PROCEED`
**Reviewer pass:** 2026-04-19

## Scope

Review of `MATH.md` (commit `99d8593`), `run_experiment.py`, `results.json`,
and `PAPER.md` against the reviewer-hat checklist (a)–(s).

## Adversarial checklist

### Consistency (blocking)
- **(a)** `results.json["verdict"] = "PROVISIONAL"` matches DB status
  `provisional`. No silent upgrade to `supported`. **PASS**
- **(b)** `results.json["all_pass"] = null`. Real-data K1/K2 were vacated,
  not passed — explicit. **PASS**
- **(c)** PAPER.md header contains "PROVISIONAL" verbatim. **PASS**
- **(d)** `is_smoke = false`. Run is not a degraded-N smoke that was
  mislabeled. **PASS**

### KC integrity (blocking)
- **(e)** `git log MATH.md` = single commit `99d8593`;
  `git diff 99d8593..HEAD -- MATH.md` is empty; no unstaged edits.
  KCs K1/K2/K3/K_vacate/K1s/K2s are frozen since pre-registration. **PASS**
- **(f)** Tautology sniff: K1s compares ridge-predicted residual vs
  mean baseline at held-out topic; residual target
  `(B_t − μ_{−t})` is *not* in the training set (LOO). No
  `e=0→0` / `x==x` / single-adapter-composition shape. **PASS**
- **(g)** K1s code (`run_experiment.py:216`) computes
  `mean_mse_ratio <= 0.95` — matches MATH.md §6. K2s code
  (`:217`) computes `median_rho > 0.1` — matches MATH.md §6. **PASS**

### Code ↔ math (blocking)
- **(h)** No `sum(lora_A)` / `add_weighted_adapter` / independent-key
  summation. Composition antipattern N/A (no composition here). **PASS**
- **(i)** `LORA_SCALE = 5.0` (`run_experiment.py:37`). Under the safe
  cap from Findings #328/#330. **PASS**
- **(j)** No routing: pure LOO on synthetic data. **PASS**
- **(k)** No `shutil.copy`. **PASS**
- **(l)** No hardcoded `"pass": True`; both K1s and K2s derive from
  measured aggregates (`run_experiment.py:216-218`). **PASS**
- **(m)** `model_target = "microsoft/BitNet-b1.58-2B-4T"` in results,
  but real-data path is gated behind `K_vacate` and was not entered.
  Synthetic-proxy is pure numpy. No proxy substitution on the real
  path. **PASS** (honestly gated)
- **(m2)** Skill invocation: this iteration only runs numpy
  synthetic-proxy code, no MLX calls are executed; `/mlx-dev` is not
  required for a vacated-real-data run. If/when the real-data K1/K2
  path runs, skill invocation will be re-checked by the next
  reviewer. **PASS** (out of scope this iteration)

### Eval integrity (non-blocking)
- **(n)** No thinking-mode channel in synthetic proxy. N/A.
- **(o)** N = 24 LOO folds on synthetic proxy. Above headline n=15. **PASS**
- **(p)** No padding: all 24 synthetic domains are genuine draws from
  the pre-registered generative model (`_generate_synthetic`). None
  are B=0 / random-Gaussian fillers. **PASS**
- **(q)** No cited external baseline; eval is internal mean-baseline.
  No drift. **PASS**

### Deliverables (blocking)
- **(r)** Prediction-vs-measurement table present in PAPER.md §3
  with all of P1–P5 rows. **PASS**
- **(s)** Math check of Thm 1: `||B_pred − B_t||² = ||δ̂_t − δ_t||² =
  σ̂² + σ² − 2σ σ̂ ρ`. Baseline (eq. 4) equals `σ²`. The inequality
  `σ̂² − 2σ σ̂ ρ < 0 ⇔ ρ > σ̂/(2σ)` is correct. At `σ̂ = σ` the
  threshold reduces to `ρ > 0.5`, consistent with the synthetic
  median ρ = 0.58 giving a 13.4 % MSE reduction. **PASS**

## Honest judgment calls / assumptions

1. **Linear-synthetic proxy is mechanism-plausibility, not transfer.**
   The synthetic uses a *linear* generative model and a *linear*
   ridge hypernetwork, so the test answers: "does ridge recover a
   known-linear low-rank mapping from 23 samples?" — yes, which
   falsifies the null "residual form is vacuous". It does not test
   transfer to BitNet's presumably non-linear adapter manifold.
   PAPER.md §6 acknowledges this. Acceptable for `provisional`.

2. **MATH.md-draft MLP → closed-form ridge deviation.** MATH.md §6
   mentions a 3-layer MLP with random projection. Implementation
   used closed-form ridge on the full `40 960`-dim target instead.
   This is a strict *improvement* (the random projection would have
   been lossy by construction — back-projection orthogonal to
   `span(U_b)`), and the KC thresholds are unchanged. PAPER.md §3
   logs the deviation. Not a KC-moving-goalpost.

3. **Per-fold MSE-ratio failures (5/24 > 0.95).** Folds `math`,
   `cooking`, `education`, `sports` had per-fold mse_ratio > 1.
   K1s is pre-registered on the *mean*, not the per-fold, so this
   does not flip the verdict — but the reviewer flags it as
   interesting: the mean is dragged above threshold by 19 folds
   with strong signal. If the real-data experiment runs, some
   domains may fail individually.

## Final verdict

**PROCEED** with `status=provisional`. Route to analyst for
LEARNINGS.md.

Honest-failure note to carry forward: the experiment closes the
tautology hole in the parent and establishes mathematical
plausibility of the residual mechanism, but *does not* yet claim
transfer to BitNet. The blocker flagged in results.json
(`regenerate exp_real_data_25_domain_adapters at LORA_SCALE=5`)
is the correct next action before any `supported` verdict is
possible.
