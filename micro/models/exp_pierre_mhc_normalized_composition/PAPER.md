# mHC-style Sinkhorn-Knopp Normalization of Composed Delta

## Abstract
We test whether applying DeepSeek V4's mHC (Sinkhorn-Knopp) normalization to DARE-composed LoRA deltas can bound spectral norm and stabilize math reasoning. The approach catastrophically fails: spectral norm increases 9000× and the model outputs garbage (0% on all benchmarks).

## Method
1. Compute composed ΔW per layer via DARE (7 PoLAR adapters, drop_rate=0.90)
2. Map to positive orthant: M = exp(clip(ΔW, ±30))
3. Apply 20 iterations of Sinkhorn-Knopp alternating row/column normalization
4. Map back: ΔW_norm = log(M)
5. Apply via _FusedDeltaLinear (Finding #831)

## Results

| Method | GSM8K | HumanEval | MedQA | Avg |
|--------|-------|-----------|-------|-----|
| DARE (baseline) | 63.3% | 90.0% | 66.7% | 73.3% |
| mHC-DARE | 0.0% | 0.0% | 0.0% | 0.0% |

### Spectral Norms
| | Mean | Max |
|---|---|---|
| Before SK | 2.631 | 2.982 |
| After SK | 19,038 | 26,934 |

SK preprocessing: 70,115ms (350× over 200ms budget)

## Kill Criteria
- K2150 FAIL: 0% accuracy (model destroyed)
- K2151 FAIL: spectral norm 26,934 (needed ≤1.05)
- K2152 FAIL: 70,115ms preprocessing (needed ≤200ms)
- K2153 FAIL: -73.3pp vs DARE

## Analysis
The fundamental error: Sinkhorn-Knopp makes M doubly-stochastic (‖M‖₂ = 1), but log(M) does NOT inherit this bound. For a doubly-stochastic matrix with entries in (0,1), log entries are all ≤0, and the spectral norm of the log can be arbitrarily large depending on how close entries are to 0.

DeepSeek V4's mHC works because it constrains the weight matrix W directly (not a delta), operates on square attention matrices with specific initialization, and doesn't use an exp/log round-trip. Our adaptation — applying SK to exp(ΔW) then taking log — is mathematically unsound.

Additionally, the 42 layers of (2048×2048) float64 Sinkhorn iterations take ~70s in numpy — 350× over budget even without the correctness issue.

## Verdict: KILLED
