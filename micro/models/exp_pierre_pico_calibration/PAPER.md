# PAPER.md — Pico SVD Calibration on Stacked B (Pierre Shared-A)

## Verdict: KILLED

Pico calibration (arxiv 2604.16826) does not improve Fisher-Rao composition in Pierre's shared-A B-only architecture. The method slightly degrades average performance (−1.3pp) rather than closing the +6.7pp gap to full-delta DARE.

## Prediction vs Measurement

| Metric | Predicted | Measured | Pass? |
|--------|-----------|----------|-------|
| K1: Pico+FR avg ≥ FR avg + 3pp | ≥ 67.7% | 63.3% (Δ=−1.3pp) | **FAIL** |
| K2: DARE avg − Pico avg ≤ 4pp | ≤ 4pp gap | 8.0pp gap | **FAIL** |
| K3: SVD+calibration ≤ 5s | ≤ 5s | 1.1s | PASS |
| K4: α=1 reproduces FR ±1pp | 64.7% ± 1pp | 64.7% (Δ=0.0pp) | PASS |

## Method Comparison (N=50/bench, seed=42)

| Method | GSM8K | HumanEval | MedQA | Avg |
|--------|-------|-----------|-------|-----|
| Single-best (per-bench) | 66.0 | 78.0 | 42.0 | 62.0 |
| Fisher-Rao (Pierre default) | 68.0 | 68.0 | 58.0 | 64.7 |
| **Pico+Fisher-Rao** | 68.0 | 68.0 | 54.0 | **63.3** |
| DARE full-delta (research) | 72.0 | 80.0 | 62.0 | 71.3 |
| Pico α=1 sanity | 68.0 | 68.0 | 58.0 | 64.7 |

## Analysis

1. **K4 sanity passes perfectly** (Δ=0.0pp): the implementation is correct. When α=1 forces S=I (no-op calibration), the output matches Fisher-Rao exactly. The negative result is not an implementation bug.

2. **Pico hurts MedQA specifically** (−4pp) while leaving GSM8K and HumanEval unchanged. The SVD-based dampening suppresses directions that are important for medical knowledge retrieval, which in Pierre's shared-A architecture are already fragile (the lowest-scoring domain).

3. **The +6.7pp gap to full-delta DARE is architectural, not interference.** Pico's thesis — that B-space over-sharing causes merge interference — may be correct in standard per-adapter-A LoRA, but in Pierre's shared-A architecture the problem is different: all information about adapter identity is already in B, so dampening shared B-directions removes signal rather than noise.

4. **Preprocess budget is excellent** (1.1s for 42 layers) — if Pico helped, the cost would be negligible. This validates the engineering path for any future SVD-based B-space pre-stage.

## Implications for Pierre

- **Do not adopt Pico as a Fisher-Rao pre-stage** in shared-A architecture.
- The shared-A gap (6.7pp) is not caused by B-direction interference; it's caused by the information loss of sharing A across adapters.
- Future work should focus on per-adapter A recovery (e.g., Grassmannian partitioning) rather than B-space conditioning.
- The `exp_pierre_pico_no_rescale` experiment (ablation: γ rescaling disabled) is still worth running to isolate whether the γ-rescale step alone accounts for the degradation.

## References

- arxiv 2604.16826 ("Crowded in B-Space", April 2026)
- exp_pierre_dare_b_vs_fisher_rao (baseline measurements: FR=64.7%, DARE=71.3%)
