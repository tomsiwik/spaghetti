# PAPER — Conflict-subspace deflation does not beat uniform-1/N merge

**Experiment:** `exp_spark_conflict_subspace_deflation`
**Verdict: KILLED** (kill 2307). `all_pass=false`, `is_smoke=false`, wall clock 2677 s (~45 min).

## Claim under test

Merging two real LoRA adapters (math/GSM8K + medical/MedQA) by summing their delta-weights
`D^ℓ = Δ_math^ℓ + Δ_med^ℓ` concentrates composition damage in a tiny shared subspace. Deflating the
top-k≤4 right-singular directions of `D` per layer (keeping matched coefficient c=1/2 everywhere else)
should recover behavioral accuracy that uniform-1/N scaling leaves on the table.

**Prediction:** `acc_aggregate(best-k deflated) − acc_aggregate(uniform-1/N) ≥ +3.0 pp`.
**Refutation (kill 2307):** `Δacc < +3.0 pp ⇒ KILLED`.

Real setup: frozen `mlx-community/gemma-4-e4b-it-4bit`, rank-6 LoRA on `q_proj` across 42 layers,
s=6.0, c=1/2. Eval = 80 GSM8K (#### integer exact-match) + 80 MedQA-USMLE-4-options (answer-letter
exact-match), greedy decode. Composition is `Σ_i A_iB_i`, SVD on CPU stream, no proxy.

## Measured results — three arms, per-domain + aggregate

| Arm | math (80) | med (80) | aggregate (160) |
|---|---|---|---|
| base (no adapter) | 0.6750 (54) | 0.5125 (41) | **0.5938** |
| uniform-1/N (standing baseline) | 0.6500 (52) | 0.5375 (43) | **0.5938** |
| deflate k=1 | 0.6750 (54) | 0.5000 (40) | 0.5875 |
| deflate k=2 (**best-k**) | 0.7125 (57) | 0.5000 (40) | **0.6062** |
| deflate k=3 | 0.6750 (54) | 0.4875 (39) | 0.5813 |
| deflate k=4 | 0.7250 (58) | 0.4875 (39) | **0.6062** |

Best-k aggregate (k=2, tied with k=4) = 0.6062.

## Prediction vs measurement (the kill)

- Predicted: Δacc ≥ +3.0 pp.
- **Measured: Δacc = best-k(0.6062) − uniform-1/N(0.5938) = +1.25 pp.**
- Threshold +3.0 pp **not** met → **kill 2307 fires → KILLED**.

Note the +1.25 pp aggregate gain is not a clean recovery: deflation *helps math*
(0.65 → 0.71/0.73) but *hurts medical* (0.5375 → 0.50/0.4875) at every k. The two domains move in
opposite directions, so the shared-subspace null is not selectively removing a clash mode — it is
trading one domain's accuracy for the other's. Uniform-1/N also fails to beat the no-adapter base
(both 0.5938), so there is no composition damage for deflation to recover in the first place.

## D singular-value spectrum / top-k energy

Mean normalized singular values of `D^ℓ` (averaged over 42 layers), σ_i/σ_1:

```
σ1..σ12 = [1.000, 0.718, 0.534, 0.400, 0.304, 0.236, 0.183, 0.139, 0.095, 0.063, 0.043, 0.029]
σ1/σ12 ≈ 35.0
```

Mean top-k energy fraction (cumulative ‖·‖²_F captured):

```
k=1: 0.464   k=2: 0.697   k=3: 0.826   k=4: 0.899
```

The auxiliary "tiny global shared subspace" premise is only weakly supported: a single mode holds
46% of summed energy and the top-4 hold 90%, but the spectrum decays smoothly (no σ₁≫σ₂ cliff;
σ₂/σ₁=0.72, σ₃/σ₁=0.53). Energy concentration ≠ damage concentration: deflating the high-energy
modes removed useful medical capacity, confirming the premise that "damage lives in the top SVD
directions of the summed delta" is false for these adapters.

## Verdict

**KILLED.** Δacc = +1.25 pp < +3.0 pp threshold (kill 2307). Conflict-subspace deflation of the
summed LoRA delta does not recover meaningful behavioral accuracy over the uniform-1/N merge; the
top-singular-mode null trades math for medical rather than removing a shared clash mode, and
uniform-1/N itself matches the no-adapter base.
