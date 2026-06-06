# Residual Hypernetwork: MATH.md

> Pre-registered before code (commit pin required). `experiment_id`:
> `exp_followup_hypernetwork_residual`. Motivated by audit
> 2026-04-17 finding that parent `exp_text_to_lora_hypernetwork`
> had a **tautological K2** (Gram-Schmidt vs training adapters of a
> convex combination of those same adapters → retention = 0 by
> construction).

## References

- Parent (killed): `exp_text_to_lora_hypernetwork` (K1 PASS
  max_ppl_ratio=1.62; K2 FAIL retention=0.0045; K3 PASS).
  `LORA_SCALE=20` used in parent — antipattern `mem-antipattern-003`.
- T2L Sakana (`arxiv:2506.06105`, ICML 2025) — hypernetwork generates
  LoRA B from text description; our parent was an ablation that
  replaced its large-scale training (thousands of pairs) with a
  24-domain LOO regime.
- FlyLoRA (`arxiv:2510.08396`) — JL-lemma orthogonality with
  frozen random A (used here as the Grassmannian skeleton).
- Base: BitNet-2B-4T (`microsoft/BitNet-b1.58-2B-4T`), mlx-lm 0.31.1.

## 1. Failure mode (what this experiment prevents)

**Parent failure mode.** Let `{B_1,…,B_N}` be trained adapter
B-matrices for N=24 domains and let the parent hypernetwork output
`B_pred = Σ_i α_i(emb) · B_i` with `α = softmax(MLP(emb))`. Then
`B_pred ∈ span{B_i}` by construction, so the Gram-Schmidt projection
against `{B_1,…,B_N}\{B_t}` leaves a residual of norm
`O(1/√(N−1))` purely from numerical rounding — retention is
tautologically driven to zero.

**New failure mode (to prevent).** Even after removing the span
constraint, with only N=24 training pairs `(emb_i, B_i)` and
`dim(B_i) ≈ 10.9M`, a hypernetwork has **no guarantee** of
generalising. The question this experiment is designed to answer is
whether the 24 embeddings carry *any* domain-specific predictive
signal *beyond the mean adapter*.

## 2. Decomposition

Any B_t admits the trivial additive decomposition against the
held-out-mean:

```
B_t = mean_not_t + (B_t − mean_not_t)
    = μ_{−t}      +   δ_t                    … (1)
```

where `μ_{−t} := (1/(N−1)) · Σ_{i≠t} B_i` is the held-out mean and
`δ_t` is the residual (domain-specific variation). The residual
satisfies `E_t[δ_t] = 0` by construction (sum-to-zero across the
N LOO folds, up to the bias from excluding `t` from `μ_{−t}`).

**Residual hypernetwork.** Train `H_θ : R^d → R^{dim(B)}` on pairs
`{(emb_i, B_i − μ_{−i})}` for `i ≠ t`, then predict

```
B_pred_t = μ_{−t} + H_θ(emb_t)                … (2)
```

This is the *residual form* — the prediction is **not** restricted to
`span{B_i}` (because `μ_{−t}` is generically not in that span when
each B_i has its own domain-specific directions).

## 3. Theorem (necessary condition, not sufficient)

**Thm 1 (residual-predictability bound).** Let `ρ_t := cos(δ_t,
δ̂_t)` where `δ̂_t := H_θ(emb_t)`, and let `σ_t := ||δ_t||_F`,
`σ̂_t := ||δ̂_t||_F`. Then the expected adapter-level MSE satisfies

```
||B_pred_t − B_t||_F² = σ̂_t² + σ_t² − 2 σ_t σ̂_t ρ_t    … (3)
```

and the baseline (no hypernetwork) satisfies

```
||μ_{−t} − B_t||_F² = σ_t².                              … (4)
```

**Corollary.** The residual form beats the baseline in adapter-space
MSE iff

```
σ̂_t² − 2 σ_t σ̂_t ρ_t < 0
⇔   ρ_t > σ̂_t / (2 σ_t).                               … (5)
```

With norm-matched prediction (`σ̂_t = σ_t`, forced via training or
L2 normalisation of the head), the bound reduces to `ρ_t > 0.5`.
For MSE to translate into PPL, a further **smoothness** assumption
is needed (Lipschitz continuity of NTP loss in adapter weights),
which is true for bounded-norm adapters on frozen unpacked BitNet.

## 4. Kill criteria (pre-registered; locked by git hash at end of
this file)

**K1 (primary).** Residual hypernetwork beats the `μ_{−t}` baseline
on held-out NTP PPL by ≥5 % on the **mean** of the eval-domain set.

```
K1 PASS iff  mean_t( PPL_residual(t) / PPL_mean_baseline(t) ) ≤ 0.95
```

Eval-domain set: `{medical, code, math, legal, cooking, sports}`
(same 6 as parent for apples-to-apples comparison).

**K2 (secondary, necessary condition from Corollary 5).**
Median LOO predictive cosine `median_t ρ_t > 0.1` across
all 24 LOO folds (i.e. embedding carries *some* residual signal).

**K3 (infrastructure).** Peak memory ≤ 40 GB. (Parent: 24 GB — this
trivially holds unless the residual-form rewrite regresses memory.)

**K_vacate (blocker).** If the parent adapter artifacts are missing
from disk (pre-requisite from `exp_real_data_25_domain_adapters`),
K1 and K2 cannot be evaluated on BitNet and the run is marked
`provisional` with a synthetic-proxy sub-test carrying its own
sub-KCs (K1s, K2s — see §6).

## 5. Antipattern checklist (must all hold, per PLAN.md §1.6)

| Antipattern | This experiment |
|---|---|
| (Σ B_i)(Σ A_i) composition bug (001) | N/A — no composition; single-adapter eval |
| Tautological routing (002) | No — LOO ensures adapter≠target-label |
| `LORA_SCALE=20` inflated claim (003) | **Fixed to 5** (parent rerun audit also at 5) |
| Thinking-mode truncation (008) | N/A — NTP prose, no thinking channel |
| Smoke-as-full (smoke hard-coded supported) | `is_smoke: false` |
| Hardcoded `"pass": True` | KCs computed from measurements |
| `shutil.copy` as new adapter | N/A |
| Copy-paste DOMAIN_KEYWORDS bug | No DOMAIN_KEYWORDS in this run |
| Grassmannian-A on wrong model (Qwen proxy) | N/A |
| Eval-template truncation → base≈0 % | PPL eval uses raw valid.jsonl |

**Verdict-consistency note.** `results.json["verdict"]` will be
`KILLED` if K1 or K2 fail (when not vacated). `PROVISIONAL` only in
the `K_vacate` branch. No auto-upgrade path to `supported`.

## 6. Synthetic-proxy sub-test (activates on `K_vacate`)

When the BitNet adapter files are missing, run a controlled
mathematical verification on structured synthetic data to test the
*residual-predictability* hypothesis independently of the specific
BitNet/NTP setup. This is an honest partial test: it can falsify the
mechanism (if even a clean low-rank synthetic setup fails, the real
setup cannot succeed) but cannot confirm transfer to BitNet.

### Design
- `d_embed = 64`, `d_out = 2560`, `r = 16` (LoRA shape),
  `N = 24` domains.
- Generate 24 latent "topic vectors" `z_i ~ N(0, I_k)` at `k = 8`
  (low-dim manifold).
- Synthetic embedding: `emb_i = U_e z_i + noise` with
  `U_e ∈ R^{d_embed × k}` random orthogonal, noise `N(0, 0.1² I)`.
- Synthetic B-matrix: `B_i = U_b z_i + ε_i` with
  `U_b ∈ R^{(r·d_out) × k}` random orthogonal, `ε_i ~ N(0, 0.01² I)`
  (domain-specific residual is almost entirely topic-driven).
- Train a 3-layer MLP hypernetwork on `{(emb_i, B_i − μ_{−i})}`
  LOO for each held-out `t`.
- Measure ρ_t (cosine between predicted residual and true residual)
  and synthetic-MSE reduction vs baseline (eq. 5).

### Sub-kill-criteria
- **K1s**: Across 24 LOO folds, `mean_t(MSE_residual / MSE_baseline) ≤ 0.95`
  (5 % adapter-space MSE reduction). Inherits the norm of the PPL KC.
- **K2s**: Median `ρ_t > 0.1` (same as K2 on real data).

### Purpose
This sub-test answers: *"In a clean setup where embedding truly
encodes a low-rank topic manifold, does the 24-sample hypernetwork
generalise?"* If yes → the mechanism is sound and the BitNet
failure (if any) lies in embedding-adapter disalignment, not in
sample efficiency. If no → the residual hypernetwork mechanism is
fundamentally sample-starved at N=24 and the fix must be more data,
not a form change.

## 7. Predictions (prediction-vs-measurement in PAPER.md)

| # | Prediction | Threshold | Source |
|---|---|---|---|
| P1 | Parent adapter artifacts missing from disk | — | Pre-run infra probe |
| P2 | Synthetic-proxy K1s: MSE reduction ≥ 5 % | MSE ratio ≤ 0.95 | Corollary (5) at σ̂≈σ |
| P3 | Synthetic-proxy K2s: median ρ > 0.1 | median ρ > 0.1 | Low-rank manifold signal |
| P4 | Predicted-residual norm `σ̂` within 0.5×–2× `σ` | 0.5 ≤ σ̂/σ ≤ 2.0 | MLP without norm-matching |
| P5 | Synthetic K1s-PASS + K2s-PASS with `k=8, N=24` | both pass | Low-dim manifold hypothesis |

If **all of P2–P4 hold** on synthetic data: emit follow-up task
`regenerate-24-adapters` (prereq for the real-data KC) and mark
`provisional`.

If any of **P2–P4 fail**: conclude the residual-form mechanism is
sample-starved at N=24 even on clean data; the fix for the parent
kill is not "residual form", it is "more training pairs". Mark
`killed`.

## 8. Git pin

This MATH.md is pre-registered at commit (to be recorded on first
commit of this file). KCs K1/K2/K3/K_vacate and sub-KCs K1s/K2s are
frozen — any change requires a v2 experiment.
