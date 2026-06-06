# E14-full: Grassmannian Activation Orthogonality — Full Run

## Result: KILLED

K2057 (bound holds): **PASS** — 0% violation rate across 22 valid layers.
K2058 (decorrelation measurable): **FAIL** — mean benefit 0.0018 < 0.01 threshold.

Per F#666: Proxy-PASS + Target-FAIL = tautological proxy, **kill on target**.

## Summary

The smoke run (3 layers, 3 adapters, 10 prompts) was PROVISIONAL with decorrelation benefit 0.0175. The full run (35 target layers, 5 adapters, 50 prompts) reveals the smoke result was not representative.

## Key Measurements

### Per-Layer Decorrelation (22 valid layers, layers 0-22)

| Layer | Grassmannian |cos| | Random |cos| | Benefit |
|-------|---------------------|----------------|---------|
| 0 | 0.0434 | 0.0426 | -0.0008 |
| 1 | 0.0358 | 0.0452 | +0.0094 |
| 2 | 0.0325 | 0.0442 | +0.0117 |
| 3 | 0.0332 | 0.0424 | +0.0092 |
| 10 | 0.0415 | 0.0306 | **-0.0109** |
| 18 | 0.0541 | 0.0394 | **-0.0147** |
| 22 | 0.0523 | 0.0378 | **-0.0145** |
| **Mean** | **0.038** | **0.040** | **+0.0018** |

12 layers positive benefit, 10 layers negative benefit. Distribution centered at zero.

### Bound Vacuousness
- sigma_max(B^T B) ≈ 38-50 across all layers
- Predicted bounds >> 1.0 while measured cos ≈ 0.03-0.05
- 0% violation rate is uninformative (bound holds by 20-30x margin)

### Capture Issue
Layers 24-40 (13 layers) produced 0 hidden states. The CaptureWrapper approach fails for later layers, likely because the model's forward pass handles these layers differently (possibly batched or through a different code path). This is a measurement limitation, not a finding about Grassmannian.

However, the 22 valid layers span the full range of early and middle layers and show consistent zero-mean benefit. There is no reason to expect late layers would differ — the B = W @ A^T construction is identical at every layer.

## Smoke vs Full Comparison

| Metric | Smoke (3 layers) | Full (22 layers) |
|--------|-------------------|-------------------|
| Mean benefit | 0.0175 | 0.0018 |
| K2044 | PASS | FAIL |
| Positive layers | 2/3 (67%) | 12/22 (55%) |

The smoke layers (0, 6, 20) happened to be on the positive side. At scale, roughly equal numbers of layers show positive and negative benefit, washing out to noise.

## Mechanism

In high dimensions (d_in = 2560), random projections are already approximately orthogonal by Johnson-Lindenstrauss concentration. Grassmannian enforces exact orthogonality (A_i^T A_j = 0), but the marginal benefit over random (max overlap ~0.32) is swamped by the B_1^T B_2 coupling term which dominates activation cosine.

The smoke run's 3-layer sample had upward bias. At N=22 layers, the law of large numbers reveals the true distribution: centered at zero.

## Implications

1. **Grassmannian provides no reliable activation decorrelation** at full scale. The ~33% reduction from smoke was sampling noise.
2. **F#815 UPGRADED from provisional to confirmed**: B_1^T B_2 is the dominant interference source, Grassmannian A provides negligible activation-level benefit.
3. **E22's poisoning protection (F#821) operates through a different mechanism** — input-space feature isolation under adversarial perturbation, NOT activation-level decorrelation. The two findings are consistent: Grassmannian doesn't help activation cosine but DOES help behavioral robustness under poisoning.
4. **Bound is structurally vacuous** — sigma_max ~ 40-50 ensures the spectral norm bound holds trivially. No tightening possible without constraining B.

## Verdict

**KILLED** on K2058 target failure. Grassmannian A does not provide measurable activation decorrelation at full scale. The theoretical zero-mean guarantee (Lemma 1) holds but is behaviorally irrelevant — variance dominates, and Grassmannian does not reduce variance (Lemma 3).

is_smoke: false
