# E14-full: Grassmannian Activation Orthogonality (Full Run)

## Type
Verification — full-scale confirmation of smoke-validated bound and decorrelation.

## Prior Art
- **E14 smoke (PROVISIONAL)**: K2043 PASS (0% violation), K2044 PASS (decorrelation 0.0175). But only 3 layers, 3 adapters, 10 prompts.
- **Finding #815**: Grassmannian decorrelates ~33% but bound is vacuous (sigma_max ~40-50).
- **Finding #817 (E15)**: B matrices share W's output space. SVD filtering counterproductive.
- **Finding #821 (E22)**: Grassmannian provides 55pp poisoning protection via input-space isolation, NOT B-matrix decorrelation. F#815 correct for activation cosine, wrong for behavioral.
- **2510.03262**: Weight-space orthogonality insufficient for semantic compositionality.

## Smoke Results (3 layers, 3 adapters, 10 prompts)
- K2043: 0% bound violation (bound vacuous by ~30x — sigma_max ~40-50)
- K2044: decorrelation benefit 0.0175 > 0.01 threshold
- Grassmannian mean|cos| = 0.034, random = 0.051 (~33% reduction)
- Layer 6 anomaly: near-zero benefit (6-layer periodicity hypothesis)

## Full Run Design
- **35 layers** (all non-global-attention: excluding {5,11,17,23,29,35,41})
- **5 adapters** (Grassmannian + random sets)
- **50 prompts** (diverse QA topics)
- **50 training steps** (vs 20 in smoke)

## Theorem (unchanged from E14 smoke)

Grassmannian A guarantees E_x[cos(delta_i, delta_j)] = 0 (Lemma 1: zero mean over isotropic inputs) but per-sample bound depends on sigma_max(B_1^T B_2), which Grassmannian does NOT constrain.

See E14 smoke MATH.md for full proof (Lemmas 1-3, Theorem).

## Predictions for Full Run
1. Bound violation rate remains ~0% (bound is vacuous by design — sigma_max >> 1)
2. Decorrelation benefit persists across all 35 layers (mean > 0.01)
3. Layer 6 anomaly may reveal 6-layer periodicity pattern across full stack
4. sigma_max(B^T B) distribution characterizes the vacuousness of the bound

## Kill Criteria (pre-registered)

**K2057** (inherits K2043 structure): Bound violation rate > 10% across 35 layers.

**K2058** (inherits K2044 structure): Mean decorrelation benefit < 0.01 at full scale.

Target-gating per F#666: K2057 is structural, K2058 is the target. Both must fail to kill.
