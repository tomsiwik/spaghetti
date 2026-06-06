# MATH.md — Followup: Crystallize + base-promotion flywheel with heterogeneous real-user adapters

## Motivation

Parent experiment `exp_p1_t6_crystallize_domain` (Finding #451) supported the claim that
averaging N same-domain B-matrices yields an adapter with `cos(crystal, B*) ≈ 0.977`,
beating the mean individual-user cosine of `≈ 0.894`. The supporting construction used
5 users per domain built as `B_u = B_canonical + ε_u` with `ε_u ~ N(0, σ² I)` at
`σ_frac = 0.5 × std(B)` — iid Gaussian noise around a single canonical centroid.

The 2026-04-17 audit flagged this construction as **synthetic-by-construction**: users
are literally `B* + iid_noise`, so `(1/N)Σ ε_u` shrinks by `√N` trivially (LLN applied
to its own construction). The test does not interrogate whether *real* users —
trained with heterogeneous LR, step counts, seeds, and data mixtures — actually
cluster tightly enough around a single domain centroid `B_D*` to admit crystallization.

**Core question**: Under realistically *heterogeneous* perturbations (varying per-user
noise variance, non-zero per-user drift, rank-structured rather than iid deviations),
does crystallization still satisfy `cos(B_crystal, B*) ≥ 0.95`?

---

## Background

### Model Soup heterogeneity condition (Wortsman 2022, arxiv:2203.05482)
Soup averaging improves generalization *only when* all constituents share the same
pre-trained basin. If fine-tuning explores different loss-landscape basins (e.g. due
to high LR, long training, divergent data), soup degrades — see their Fig 3 ablation
with varied LR. Our analogue: LoRA B-matrices initialised at zero share the same
basin, but heterogeneous (LR, steps, data) trajectories *may* depart from a single
centroid.

### Task Arithmetic failure under bias (Ilharco 2023, arxiv:2212.04089)
Task-vector averaging is unbiased only if individual task vectors have zero-mean
deviations from the target. When per-user training drifts systematically (e.g. tokenizer
bias, LR-scale bias), the mean of the deviations is non-zero and crystallization
converges to `B* + μ`, not `B*`.

### Heterogeneous LLN (Lindeberg–Feller)
Classical LLN needs iid; Lindeberg–Feller extends to independent but not identically
distributed variables with a uniform-boundedness condition on variances. The sample
mean converges to the *mean of means*, not to a single target. Applied to crystallization:
`B_crystal → (1/N) Σ μ_u`, which equals `B*` only if `E[ε_u] = 0` for all u.

---

## Theorem 1: Crystallization Quality Under Heterogeneous Noise

**Setting**: N users in domain D, each with perturbation decomposition
```
B_u = B_D* + μ_u + η_u
```
where:
- `μ_u ∈ ℝ^d` is a *systematic* per-user drift (from different LR scales, step counts,
  data distributions — a user with aggressive LR overshoots, a user with few steps
  under-converges).
- `η_u ~ N(0, σ_u² / d · I_d)` is iid-style residual noise with per-user variance
  `σ_u²` (different seeds/data).

**Heterogeneity parameters**:
- `σ̄² = (1/N) Σ σ_u²` — mean noise variance.
- `Var(σ_u²) = bias-structured dispersion` — how much users disagree in scale.
- `μ̄ = (1/N) Σ μ_u` — collective drift (ideally zero; non-zero under systematic bias).

**Theorem 1 (Heterogeneous Crystallization Error)**:
```
E[‖B_crystal − B_D*‖²_F] = ‖μ̄‖² + σ̄² / N
```

**Proof**:

Step 1. Linearity: `B_crystal − B_D* = (1/N) Σ (μ_u + η_u) = μ̄ + η̄` with
`η̄ = (1/N) Σ η_u`.

Step 2. `E[‖μ̄ + η̄‖²] = ‖μ̄‖² + 2 E[⟨μ̄, η̄⟩] + E[‖η̄‖²]`. Cross-term
`E[⟨μ̄, η̄⟩] = ⟨μ̄, E[η̄]⟩ = 0`.

Step 3. `E[‖η̄‖²] = (1/N²) Σ E[‖η_u‖²] = (1/N²) Σ σ_u² = σ̄²/N` (each
`E[‖η_u‖²] = d · σ_u²/d = σ_u²`).

Combining: `E[‖B_crystal − B_D*‖²] = ‖μ̄‖² + σ̄²/N`. **QED.**

**Corollary 1.1 (Cosine Floor)**: Using the approximation
`cos(B_crystal, B*) ≈ ‖B*‖ / √(‖B*‖² + ‖μ̄‖² + σ̄²/N)`, the cosine is upper-bounded
by a function of `‖μ̄‖` and `σ̄²/N`. Averaging only shrinks the residual `σ̄²/N` — it
does **not** reduce `‖μ̄‖`. Therefore:
```
lim_{N→∞} E[cos(B_crystal, B*)] = ‖B*‖ / √(‖B*‖² + ‖μ̄‖²)
```
The cosine is floor-bounded by `‖μ̄‖ / ‖B*‖`, independent of N.

**Implication**: Crystallization works **iff `‖μ̄‖` is negligible relative to `‖B*‖`.**
Under the original parent experiment (`μ_u = 0` by construction, σ constant), the
floor is 1 and any N gives `cos → 1`. Under heterogeneous real users with non-zero
`μ̄`, the floor is less than 1 and may violate K1564.

---

## Theorem 2: KC Threshold Derivation

**Claim**: To satisfy K1564 `cos(crystal, B*) ≥ 0.95`, heterogeneity must satisfy
```
‖μ̄‖² + σ̄²/N  ≤  (1/0.95² − 1) · ‖B*‖²  ≈  0.108 · ‖B*‖²
```

**Proof**: Solve `‖B*‖ / √(‖B*‖² + E) ≥ 0.95` for `E`:
`‖B*‖² / (‖B*‖² + E) ≥ 0.9025`, so `E ≤ ‖B*‖² · (1/0.9025 − 1) ≈ 0.108 ‖B*‖²`. **QED.**

For Gemma 4 E4B, parent experiment reported `‖B*_math‖ ≈ 5.7618`, so the error
budget is `‖B*‖² · 0.108 ≈ 3.58`. With N=5 canonical users at `σ = 0.5 · std(B) ≈ 0.003713`
and `d = 602,112`, residual `σ̄²/N ≈ 8.30/5 = 1.66`. This leaves **only 1.92 squared
Frobenius units for `‖μ̄‖²`** — i.e. `‖μ̄‖ ≤ 1.39`. This is a tight budget: any
systematic drift of magnitude `> 24%` of `‖B*‖` violates K1564.

---

## What real-user heterogeneity looks like

When real users train with (LR, steps, seed, data) = variable, four structural effects
arise that pure iid `N(0, σ²I)` cannot capture:

1. **Scale heterogeneity**: High-LR users take larger steps → `σ_u` larger; low-LR users
   → `σ_u` smaller. Effect on theorem: `σ̄²` becomes the *arithmetic mean* of widely
   varying `σ_u²`, not a constant. K1564 threshold still applies but with larger `σ̄²`.
2. **Convergence heterogeneity**: Few-step users haven't fully descended toward `B_D*`
   → `μ_u` points *toward* origin (undershoot); many-step users → `μ_u ≈ 0`. `μ̄` is
   non-zero, biased toward origin.
3. **Data-dependent drift**: Different data subsets pull toward different subdomains
   of D. `μ̄` may be non-zero but smaller than individual `μ_u` (partial cancellation).
4. **Low-rank structure**: LoRA updates live in a rank-r subspace, so `μ_u`, `η_u`
   are rank-r, not full rank d. This changes the effective dimension in the variance
   calculation.

Our heterogeneous simulation below models (1), (2), (3), (4) directly.

---

## Experimental Setup

### Heterogeneous user model

For each domain D with canonical `B_D*` (loaded from saved safetensors **when
available** — otherwise randomly drawn on `unit_sphere × norm_scale` per-domain to
preserve scale). For user u ∈ {1..N}:

1. **Hyperparameter draw**: sample `(lr_u, steps_u, seed_u)` from a heterogeneous
   distribution. Specifically:
   - `lr_u ~ LogUniform(1e-5, 1e-3)` (4 orders of magnitude — realistic real-user
     variation).
   - `steps_u ~ Uniform{100, 200, 400, 800, 1600}` (16× range).
   - `seed_u ~ UniformInt(0, 2^31)`.
2. **Convergence model**: `convergence_u = 1 − exp(−lr_u · steps_u / τ)` with
   `τ = 1.0` calibrated so that `(lr=1e-4, steps=400)` gives `convergence ≈ 0.96`.
3. **Scale model**: `σ_u = 0.5 · std(B_D*) · (lr_u / 1e-4)^0.3` — higher LR leaves
   larger residual noise.
4. **Bias direction**: drawn per-user as rank-r random subspace aligned with `B_D*`'s
   low-rank structure.
5. **Construction**: `B_u = convergence_u · B_D* + (1 − convergence_u) · bias_drift_u
   + η_u`, where `η_u ~ N(0, σ_u² I_r)` in the rank-r subspace.

### Kill criterion K1564 (pre-registered, no relaxation allowed)

```
K1564: mean over 5 domains of cos(B_crystal^D, B_D*) ≥ 0.95
```

with 5 heterogeneous users per domain.

### Additional non-pass-gated observational metrics

- `mean_individual_cos`: `mean_u cos(B_u, B_D*)`, baseline.
- `crystal_gain`: `cos(B_crystal, B_D*) − mean_individual_cos`.
- `mu_bar_norm_frac`: `‖μ̄‖ / ‖B*‖` — how much collective drift remains.
- `sigma_spread`: `max(σ_u) / min(σ_u)` — realized heterogeneity factor.

### Data provenance and infrastructure note

The original experiment referenced canonical `B*` from real trained adapters at:
- `micro/models/exp_p1_t2_single_domain_training/adapters/{math,code,medical}/adapters.safetensors`
- `micro/models/exp_p1_t2_multi_domain_5/adapters/{legal,finance}/adapters.safetensors`

These are **gitignored** and not on disk (verified: only `adapter_config.json`
present, no `.safetensors`). The same infra blocker kills the real-user training
path completely. Two paths:

1. **`REAL_ADAPTERS_AVAILABLE=True`**: Load real `B*`, generate 5 heterogeneous users
   per domain via the model above (synthetic but with realistic LR/step/seed structure).
   Limitation: users are still partially simulated; only `B*` is real.
2. **`REAL_ADAPTERS_AVAILABLE=False` (actual state, 2026-04-19)**: Draw a synthetic
   `B*` with the same norm/std statistics as reported in the parent experiment
   (`‖B*_math‖ ≈ 5.76`, `std ≈ 0.0074`, `d = 602,112`). All users are synthetic.

We run case (2) as the primary evaluation. The verdict must explicitly flag that
K1564 is tested **against a heterogeneous-synthetic** construction, not real trained
adapters. If the heterogeneous synthetic construction *still* supports the 0.95
floor, that is evidence the claim is robust to heterogeneity (though not to full
real-user drift). If it *fails*, K1564 is killed regardless of simulation fidelity
— you cannot hide a failure behind simulation caveats.

### Kill criteria predictions

| Criterion | Metric | Predicted (synthetic parent-style iid) | Predicted (heterogeneous) | Pass @ 0.95? |
|-----------|--------|------|------|---|
| K1564 (mean) | mean cos(crystal, B*) | 0.977 (parent) | 0.82–0.92 (theorem-predicted under heterogeneity) | **fail** expected |

Rationale for prediction: under heterogeneous (lr, steps, seed), `μ̄` is non-zero
because low-step users systematically under-converge (bias toward origin). The
theorem predicts `cos ≈ ‖B*‖ / √(‖B*‖² + ‖μ̄‖² + σ̄²/N)` with `‖μ̄‖ ≈ 0.4·‖B*‖`
from `(1 − mean_convergence) · ‖B*‖ ≈ 0.5·‖B*‖` under our LR distribution.
This gives `cos ≈ 1/√(1 + 0.25) ≈ 0.89`, below the 0.95 threshold.

**Interpretation**: if measurement matches prediction → K1564 **killed** with
structural explanation (heterogeneous real users violate the zero-drift LLN
condition). If measurement exceeds 0.95 → K1564 **supported** under heterogeneity,
strengthening the parent claim.

---

## References

- Model Soup: Wortsman et al. 2022, arxiv:2203.05482
- Task Arithmetic: Ilharco et al. 2023, arxiv:2212.04089
- Lindeberg–Feller CLT/LLN: Billingsley, *Probability and Measure*, Ch 27.
- FedAvg heterogeneity: Li et al. 2020, arxiv:1907.02189 (Section 3: non-iid bias)
- Parent experiment: `exp_p1_t6_crystallize_domain` (Finding #451)
- Audit flag: `audit-2026-04-17`, motivation `supported_15.md exp_p1_t6 synthetic-by-construction`.

## Assumptions / Limitations

1. `‖B*‖` and `std(B*)` statistics are borrowed from parent experiment logs when
   real adapter weights are absent. If the disk state differs, swap in real values.
2. The convergence model `convergence = 1 − exp(−lr · steps / τ)` is a first-order
   approximation of gradient-descent convergence rates; Newton-like accelerators
   would lower `τ`. Robust to factor-of-2 calibration error.
3. We simulate rank-r perturbations where r=8 is Gemma-4 LoRA rank from parent;
   if target rank differs, the dimension in variance calc scales accordingly.
4. K_vacate clause: if the simulation crashes because real `B*` adapters are
   required *and* produce artefacts not captured by norm/std statistics alone,
   we K_vacate and flag the second infra-blocker instance (tracked on the
   infra_adapter_rerun queue).
