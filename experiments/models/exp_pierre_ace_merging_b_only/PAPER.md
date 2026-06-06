# PAPER.md — ACE-Merging adapted to Pierre's shared-A architecture

## Result

**KILLED.** ACE-Merging catastrophically degrades task quality when applied to Pierre's shared-A B-only architecture, scoring 26.7% avg vs Fisher-Rao's 64.7% — a -38pp collapse.

## Prediction vs Measurement

| Metric | Predicted | Measured | Match? |
|---|---|---|---|
| ACE avg ≥ Fisher-Rao + 3pp | 67.7%+ | 26.7% | NO (-38pp) |
| DARE avg − ACE avg ≤ 4pp | ≤75.3% gap | 44.7pp gap | NO |
| Preprocessing ≤ 30s | ~10s | 6.0s | YES |

## Method comparison (N=50/bench, seed=42)

| Method | GSM8K | HumanEval | MedQA | Avg |
|---|---|---|---|---|
| Single-best (per bench) | 66.0 | 78.0 | 42.0 | 62.0 |
| Fisher-Rao (Pierre default) | 68.0 | 68.0 | 58.0 | 64.7 |
| **ACE-Merging (under test)** | **8.0** | **24.0** | **48.0** | **26.7** |
| DARE full-delta (upper bound) | 72.0 | 80.0 | 62.0 | 71.3 |

## Kill Criteria

| KC | Threshold | Measured | Verdict |
|---|---|---|---|
| K1 beats Fisher-Rao | +3pp | -38.0pp | **FAIL** |
| K2 close to DARE | ≤4pp gap | 44.7pp gap | **FAIL** |
| K3 preprocessing budget | ≤30s | 6.0s | **PASS** |

## Analysis

The failure is catastrophic and structural, not marginal:

1. **GSM8K collapsed to 8%** (from 68% Fisher-Rao). This is below random-guess levels for a math benchmark, indicating the merged delta actively corrupts the model's reasoning capability.

2. **HumanEval dropped to 24%** (from 68%). Code generation similarly destroyed.

3. **MedQA held at 48%** (vs 58% Fisher-Rao) — the least affected, but MedQA is MCQ with 4-5 choices so 48% is near the ceiling of informed guessing.

### Why ACE fails on shared-A

ACE-Merging's Theorem 1 derives per-task input covariance from the delta matrix alone: $\hat\Sigma_t = \tilde{W}_t^\top \tilde{W}_t$. This assumes each $\Delta W_t$ is independently parameterized. In Pierre's shared-A regime, $\Delta W_t = s \cdot A \cdot B_t$ — all deltas share the same left factor $A$, so the covariance matrices are dominated by $A^\top A$ structure rather than capturing meaningful per-task variation. The "covariance weighting" degenerates into weighting by shared structure, not task-specific information.

The closed-form inverse $(D^{-1})$ then amplifies this degeneracy: the denominator matrix is nearly rank-deficient (all $\hat\Sigma_t$ share $A^\top A$ eigenstructure), making the inverse numerically unstable and producing a merged delta that is far from any individual task vector.

### Contrast with DARE

DARE full-delta also materializes per-adapter $\Delta W_t$, but applies stochastic pruning + rescaling without computing covariance. It sidesteps the shared-A degeneracy because it never tries to infer per-task structure from the deltas — it just prunes and averages. This is why DARE achieves 71.3% while ACE collapses to 26.7%.

## Implications

1. **ACE-Merging is incompatible with shared-A architectures.** The covariance derivation (Theorem 1) requires independently-parameterized deltas. Shared-A violates this assumption.

2. **Methods that infer per-task structure from $\Delta W$ are suspect under shared-A.** Any technique relying on $\Delta W_t^\top \Delta W_t$ will see $A^\top A$ dominance. This includes potential future ports of RegMean, Fisher-weighted merging with delta-derived Fisher approximations, etc.

3. **Simple methods (Fisher-Rao, DARE) remain the correct composition strategy** for Pierre's architecture, precisely because they don't attempt to extract structure that shared-A cannot provide.

## Config

- Model: `mlx-community/gemma-4-e4b-it-4bit`
- Adapters: 7 PoLAR (4 strategy + 3 domain), q_proj, rank=6, scale=6.0
- Shared-A donor: strategy_full
- ACE params: eps=0.01, tau=0.3, k_frac=0.3, spectral_refinement=enabled
- N=50/bench, seed=42

## Verdict

**KILLED** — ACE-Merging does not transfer to shared-A architecture. The covariance derivation requires independent parameterization of task deltas; shared-A violates this, causing degenerate merges.
