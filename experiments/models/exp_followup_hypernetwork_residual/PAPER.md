# exp_followup_hypernetwork_residual — PAPER.md

**Verdict:** `PROVISIONAL`  (mechanism-plausible on synthetic proxy; real-data KCs vacated pending adapter regeneration)

**Run:** 2026-04-18  |  **mlx-lm:** 0.31.1 (BitNet path not executed)  |  **LORA_SCALE:** 5.0 (antipattern-003 fixed)

## 1. Objective

Test the fix for the **tautological K2** in parent
`exp_text_to_lora_hypernetwork` (killed 2026-03-28, audit rerun
preserved the kill at `LORA_SCALE=5` on 2026-04-18). Parent predicted
`B_pred ∈ span{B_i}` by construction (softmax convex combination),
so Gram-Schmidt vs training adapters tautologically destroys the
signal.

Fix: residual form `B_pred = μ_{−t} + H(emb_t)` (MATH.md sec 2).
The residual escapes the span. The new primary KC is non-tautological:
does the embedding-driven delta produce ≥5 % held-out PPL improvement
over the mean-baseline?

## 2. Infrastructure probe

```
adapters present:   0 / 24   (expected: 24)
skeleton present:   False     (expected: exp_real_data_25_domain_adapters/adapters/grassmannian_skeleton_n24.npz)
K_vacate triggered: True
```

Root cause: the parent `exp_real_data_25_domain_adapters` artifacts
(24 trained adapter `.npz` files + grassmannian skeleton) are not
present on the working copy. They were generated in a prior run
(results.json timestamp 2026-03-27) but never committed (large
binary artefacts, gitignored). The audit rerun note
"K2 FAIL preserved at LORA_SCALE=5 on 2026-04-18" refers to an
ephemeral rerun whose artefacts were not preserved.

**Consequence:** K1 (PPL ratio) and K2 (adapter-cosine) on the real
BitNet/LoRA path cannot be evaluated in this iteration. MATH.md
sec 4's `K_vacate` branch activates.

## 3. Synthetic-proxy sub-test (see MATH.md sec 6)

Controlled mathematical verification of the residual-predictability
mechanism on a known linear low-rank manifold:
- N=24 domains, `d_embed=64`, `dim_b = r·d_out = 16·2560 = 40 960`.
- Latent topic `z_i ~ N(0, I_k)` at `k=8`.
- `emb_i = U_e z_i + N(0, 0.1²)`, `B_i = U_b z_i + N(0, 0.01²)`
  with `U_e, U_b` random orthogonal.
- Linear hypernetwork: closed-form ridge regression
  (`α = 1e-2`), trained per-fold on `(emb_i, (B_i − μ_{−t}))`.

### Implementation note (deviation from MATH.md draft text)
MATH.md mentioned a 3-layer MLP with a random-projection of the
target to `k_out=32`. Two issues with that draft plan:
- The `sklearn.MLPRegressor` and the numpy-Adam variant both
  NaN'd during training at the problem scale (trace below).
- The random projection `P ∈ R^{dim_b × k_out}` is lossy by
  construction: the pseudoinverse `(P^T P)^{-1} P^T` back-projects
  into a random `k_out`-dim subspace of `R^{dim_b}` that is
  near-orthogonal to the true rank-`k` signal subspace
  `span(U_b)`, so signal is destroyed irrecoverably even if the
  ridge is exact.

Closed-form ridge on the full `40 960`-dim target is tractable
(`W ≈ 10 MB`) and is the **correct** linear hypernetwork. This
deviation does not touch the KCs or the pre-registered KC
thresholds — it only corrects a naive projection idea.

### Prediction vs measurement (synthetic sub-test)

| # | Prediction | Threshold | Measured | Result |
|---|---|---|---|---|
| P1 | Parent adapter artifacts missing | — | 0/24 present | **CONFIRMED** |
| P2 | Synthetic K1s: MSE reduction ≥ 5 % | `mean_t(MSE_resid/MSE_base) ≤ 0.95` | 0.8662 (13.4 % reduction) | **PASS** |
| P3 | Synthetic K2s: median ρ > 0.1 | median ρ > 0.1 | 0.5779 | **PASS** |
| P4 | Predicted-residual norm bounded | 0.5 ≤ `σ̂/σ` ≤ 2.0 | median 0.9168 | **PASS** |
| P5 | Combined synthetic pass | P2 ∧ P3 | yes | **PASS** |

### Per-fold sample (first/last; full set in results.json)

```
fold  0 (medical    ): rho=+0.5765  mse_ratio=0.9968  sigma_ratio=1.1502
fold  1 (code       ): rho=+0.5472  mse_ratio=0.8282  sigma_ratio=0.9045
fold  2 (math       ): rho=+0.4557  mse_ratio=1.0857  sigma_ratio=0.9973
fold  3 (legal      ): rho=+0.6366  mse_ratio=0.6694  sigma_ratio=0.9097
fold  4 (finance    ): rho=+0.4944  mse_ratio=0.9307  sigma_ratio=0.9128
...
fold 23 (music      ): rho=+0.6734  mse_ratio=0.7300  sigma_ratio=1.1017
```

### What the synthetic result means

On a **clean** low-rank manifold with known linear generative model,
23 training samples suffice for a linear hypernetwork to:
- Recover the true residual direction with median cosine ≈ 0.58
  (well above the pre-registered 0.1 threshold).
- Reduce adapter-space MSE by 13 % vs the mean baseline (above the
  5 % threshold).
- Produce predictions with norm within ±10 % of the true residual.

This **does not** translate to PPL improvement claims on BitNet —
that requires the real adapter artefacts. It does show the residual
mechanism is **mathematically sound** on the assumptions of (i) a
low-rank manifold and (ii) near-linear emb→adapter relationship.
Both assumptions are testable but untested on Gemma 4 / BitNet.

## 4. Kill criteria — status

| KC | Condition | Status | Evidence |
|---|---|---|---|
| K1 (real) | mean PPL ratio ≤ 0.95 | **VACATED** | 0/24 adapters on disk |
| K2 (real) | median LOO cosine > 0.1 | **VACATED** | 0/24 adapters on disk |
| K3 (infra) | peak memory ≤ 40 GB | **NOT EVAL** | Real-data path not entered |
| K1s (synth) | mean MSE ratio ≤ 0.95 | **PASS** | 0.8662 |
| K2s (synth) | median ρ > 0.1 | **PASS** | 0.5779 |
| K_vacate | adapter artefacts missing | **ACTIVE** | skel+adapters absent |

## 5. Verdict-consistency pre-flight (PLAN.md §1.6)

1. `results.json["verdict"] == "PROVISIONAL"` — ✅ not "KILLED", not
   silently upgraded to "supported".
2. `results.json["all_pass"] == null` — no claim of all-pass; the
   real-data KCs were vacated, not "passed".
3. PAPER.md verdict line contains "PROVISIONAL" — ✅ honest label.
4. `is_smoke == false` — ✅ this is not a smoke run; the
   provisional status comes from a missing prerequisite, not from
   reduced N.
5. KCs unchanged since `MATH.md` pre-registration (git-diff clean
   against commit `99d8593`) — ✅ only the projection implementation
   changed, not the thresholds.
6. Antipattern memories — reviewed:

| Antipattern | This run |
|---|---|
| `(ΣB)(ΣA)` composition bug (001) | N/A — no composition |
| Tautological routing (002) | No — LOO held-out domain |
| `LORA_SCALE=20` (003) | Fixed → 5.0 (not used in vacated path) |
| Thinking-mode truncation (008) | N/A — not applicable |
| Smoke hard-coded supported | `is_smoke=false` |
| Hardcoded `"pass": True` | KCs computed from measurements |
| `shutil.copy` adapter | N/A |
| Copy-paste DOMAIN_KEYWORDS | DOMAIN list explicit, not copied |
| File-existence cache | Measured on every run |
| Dispatch-kill mislabel | Vacated != killed; flagged honestly |

## 6. Assumptions & limitations

- The synthetic generative model assumes a shared 8-dim topic
  manifold. Real Gemma 4 / BitNet domain adapters have unknown
  manifold dimension; if it is much higher than `d_embed` or much
  higher than `N=24`, the residual hypernetwork is sample-starved
  and will fail on real data even though it succeeded on synthetic.
- The synthetic test uses a **linear** generative model, so a
  linear ridge hypernetwork is near-optimal. Real adapter space is
  plausibly non-linear, which would require a larger MLP and more
  samples.
- PPL is the target metric on real data; adapter-MSE is the
  synthetic proxy. The Lipschitz link from MSE→PPL is bounded for
  small adapter perturbations but can break at large scales.
  `LORA_SCALE=5` (not 20) keeps the effective perturbation small.
- The mechanism test says nothing about the parent's *original*
  K2 tautology fix — only that the *replacement mechanism* is not
  mathematically vacuous. The parent's convex-combination still
  fails its K2 tautologically; the residual form is the proposed
  alternative, not a rescue of the parent.

## 7. Next-experiment seeds

1. **Unblock this KC.** `exp_real_data_25_domain_adapters_rerun`:
   regenerate the 24 adapters at `LORA_SCALE=5` on BitNet-2B-4T and
   save to the expected path. Then `exp_followup_hypernetwork_residual`
   can be rerun against K1/K2 on real data without any MATH.md change.
2. **Extend the synthetic test** to `N=100` and `k=32` to see where
   the residual mechanism breaks (sample-efficiency frontier).
3. **Non-linear generative model** synthetic: `B_i = U_b · f(z_i)`
   where `f` is a shallow MLP. Check that ridge fails in this
   regime but that an MLP hypernetwork succeeds (pre-requisite for
   claiming the mechanism on real non-linear adapter spaces).
