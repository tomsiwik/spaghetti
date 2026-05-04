# Composition Method Ablation

## Type
Guided exploration — proven composition mechanism (Finding #828 FusedDeltaLinear, Finding #829 uniform 1/N works), unknown relative ordering of three routing strategies.

## Reference
- Finding #828: `__call__` override destroys behavioral accuracy; `_FusedDeltaLinear` module replacement is the correct pattern.
- Finding #829: Uniform 1/N composition works when applied correctly (avg 71.1). DARE adds +2.2pp marginal benefit. Three prior "composition KILLs" were FALSE KILLS from the `__call__` bug.
- M2P gate: 99.6% holdout classification accuracy, entropy 0.039 nats, top-1 weight 0.993 (from exp_pierre_m2p_gated_composition Phase 3).

## Hypothesis
M2P-gated continuous mixing outperforms both uniform 1/N averaging and hard top-1 routing when composition is applied correctly via `_FusedDeltaLinear`.

## Mechanism
Three composition strategies over the same 7 PoLAR adapters:
- **M1 (Uniform 1/N):** ΔW = (1/N) Σ (A_i @ B_i). Each adapter contributes equally. Known working baseline from Finding #829.
- **M2 (Hard top-1):** Oracle routing — select the single best adapter per benchmark domain. Upper bound on routing-based composition.
- **M3 (M2P-gated continuous):** Trained 2-layer MLP predicts softmax weights w_i per prompt → ΔW = Σ w_i (A_i @ B_i). Gate from exp_m2p (99.6% accuracy, entropy 0.039). Applied via `_FusedDeltaLinear`, NOT `__call__` override.

## Predictions

| Method | GSM8K | HumanEval | MedQA | Avg |
|--------|-------|-----------|-------|-----|
| M1 (uniform) | ~63 | ~90 | ~60 | ~71 |
| M2 (top-1 oracle) | ~70 | ~87 | ~50 | ~69 |
| M3 (M2P-gated) | ~68 | ~88 | ~55 | ~70 |

Rationale: M2P gate is so peaked (top-1 weight 0.993) that M3 ≈ M2 in practice. The uniform method wins on MedQA/HumanEval because averaging creates beneficial interference (Finding #829 showed uniform beating best-single on those). M2 wins only on GSM8K (domain_math is best single there). M1 should win overall because uniform averaging is optimal when adapters are non-interfering.

## Kill Criteria (pre-registered)

- **K2121** (target): M2P-gated avg accuracy > both uniform AND top-1 avg accuracy. Threshold: strict >. Prediction: FAIL (uniform wins because peaked gate degenerates to top-1, losing the beneficial averaging on cross-domain benchmarks).
- **K2122** (target): All three methods within 1.5× latency of the fastest. Threshold: max/min ≤ 1.5. Prediction: PASS (M1 and M3 are same-cost; M2 skips composition but loads the same model).
- **K2123** (proxy): M2P-gated Spearman ρ(gate-confidence, per-prompt-correctness) ≥ 0.3. Prediction: MARGINAL — gate is so peaked that confidence variance is too low for meaningful correlation.
- **K2124** (diagnostic): Characterize prompts where ALL methods fail. No pass/fail threshold.

## Platform
- Base model: `mlx-community/gemma-4-e4b-it-4bit`
- mlx-lm version: current (0.31+)
- Hardware: M5 Pro 48GB
- Composition: `_FusedDeltaLinear(nn.Module)` proper module replacement (NOT `__call__` override)
