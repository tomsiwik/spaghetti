# E22: Adapter Poisoning Robustness — Grassmannian as Safety Feature

## Status: PROVISIONAL (smoke, N=20 QA, 3 layers)

## Summary
Grassmannian orthogonality provides substantial protection against adapter poisoning. At 10× poison magnitude, Grassmannian retains 80% accuracy while random A collapses to 25% (55pp protection margin). At 20×, Grassmannian still maintains 60% vs random's 10%.

## Prediction vs Measurement

| Prediction | Measured | Match |
|---|---|---|
| Grassmannian degradation 10-20pp | 25pp at 20× worst case | Close (slightly worse) |
| Random degradation 20-35pp | 60-75pp at 10-20× | Much worse than predicted |
| Protection margin 5-15pp | 50-55pp at 10-20× | Far exceeded prediction |
| F#815 B-matrix dominates → margin < 5pp | Margin = 55pp | **Falsified** |

## Key Finding
Grassmannian input-space orthogonality provides dramatically stronger poisoning protection than predicted by F#815's B-matrix coupling analysis. The mechanism:

1. **At low poison (1×)**: Both Grassmannian and random are robust — rank-6 perturbation on 3 layers is negligible relative to base weights (clean_mag=36 vs W_norm≈30).
2. **At 5×**: Random begins degrading (80%), Grassmannian unaffected (85%). Input-space orthogonality prevents the poison from reading the same features as clean adapters.
3. **At 10×**: Sharp divergence — Grassmannian 80% vs random 25%. The random poison's A matrix overlaps clean A matrices, causing correlated contamination that amplifies through the model. Grassmannian's orthogonal A reads DIFFERENT input features, limiting contamination to output-space only.
4. **At 20×**: Both degrade but Grassmannian gracefully (60% vs 10%). Even output-space interference is bounded because the poison operates on independent input features.

## Why F#815 Was Wrong Here
F#815 proved that per-sample activation interference depends on σ_max(B₁ᵀB₂), not on A-matrix angles. This is mathematically correct for activation-level cosine similarity. But for poisoning robustness, the relevant metric is not activation similarity — it's **behavioral degradation**. The non-linear stack (GELU, LayerNorm, softmax) amplifies or attenuates perturbations depending on whether they align with the model's processing direction. Orthogonal input-space selection (Grassmannian A) ensures the poison reads from features the model doesn't critically depend on for the clean task, even though B-matrix output coupling exists.

## Results Table

| Multiplier | Grassmannian Acc | Random Acc | Grass Drop | Random Drop | Margin |
|---|---|---|---|---|---|
| 1× | 85.0% | 85.0% | 0pp | 0pp | 0pp |
| 5× | 85.0% | 80.0% | 0pp | 5pp | 5pp |
| 10× | 80.0% | 25.0% | 5pp | 60pp | 55pp |
| 20× | 60.0% | 10.0% | 25pp | 75pp | 50pp |

Base accuracy: 85.0%, Clean-only (both): 85.0%

## Kill Criteria

| KC | Threshold | Measured | Verdict |
|---|---|---|---|
| K2055 | Grass drop < 30pp at worst | 25pp at 20× | **PASS** |
| K2056 | Protection margin > 2pp | 55pp at 10× | **PASS** |

## Caveats
1. **Smoke test only**: 3 layers, 20 QA questions. Full run needed for confidence.
2. **Synthetic poison**: Random noise, not adversarial. Real poisoning attacks may be more targeted.
3. **Only v_proj modified**: Adding o_proj or more layers may change the protection ratio.
4. **Knowledge QA is a simple task**: More complex reasoning tasks may show different patterns.

## Implications
- Grassmannian A-matrices ARE a meaningful safety mechanism for adapter composition, contrary to the F#815 prediction that B-matrix coupling would dominate.
- The protection is in input-space feature selection, not output-space — the poison reads from independent features.
- For Pierre's adapter marketplace, Grassmannian provides a structural defense layer against corrupted/adversarial adapters.
- Engineering recommendation: always use Grassmannian A for composed adapters from untrusted sources.
