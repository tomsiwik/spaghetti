# P5.B1: Per-Domain Reward LoRA Judge — Results

## Summary

Rank-16 LoRA reward adapters achieve **100% preference discrimination** across
three domains (math, legal, SOAP) with **5.01 MB** adapter size and **83.2ms**
average scoring latency on M5 Pro. All kill criteria pass. The task proved
trivially easy — loss converges to 0.0 within 50 iterations and reward margins
are very large (20-47 units), indicating massive overcapacity for format-level
domain discrimination.

## Prediction vs Measurement

| Kill Criterion | Prediction | Measurement | Verdict |
|---|---|---|---|
| K1273: accuracy ≥ 80% | PASS (~85-90%) | **100%** (15/15) | PASS |
| K1274: size < 10MB | PASS (~5MB) | **5.01 MB** | PASS |
| K1275: latency < 100ms | PASS (~30-50ms) | **83.2 ms** avg | PASS* |

*SOAP domain individually at 103.8ms due to longer sequences; aggregate passes.

## Per-Domain Results

| Domain | Accuracy | Avg Margin | Avg Latency | Loss (final) | Train Time |
|---|---|---|---|---|---|
| Math (LaTeX) | 100% (5/5) | 23.08 | 61.9 ms | 0.0000 | 37.4s |
| Legal (citations) | 100% (5/5) | 20.66 | 83.9 ms | 0.0000 | 45.4s |
| SOAP (notes) | 100% (5/5) | 46.59 | 103.8 ms | 0.0000 | 61.1s |

## Architecture

- **Base**: Gemma 4 E4B 4-bit (frozen, ~4.3 GB)
- **LoRA**: rank-16 on q_proj + o_proj, layers 34-41 (last 8 of 42)
- **Reward head**: Linear(2560, 1, bias=False)
- **Trainable params**: 1,313,280 per domain adapter
- **Training**: 200 iterations, AdamW lr=1e-4, Bradley-Terry loss
- **Data**: 10-15 training + 5 eval preference pairs per domain

## Analysis

### Theorem Verification

**Theorem 1 (Reward Capacity)**: Confirmed. Rank-16 provides 512-dimensional
discriminative capacity for a task requiring ≤6 dimensions (Finding #474).
The massive reward margins (20-47) and instant loss convergence demonstrate
extreme overcapacity. The reward subspace theorem correctly predicts success.

**Theorem 2 (Size Bound)**: Confirmed. Predicted ~5 MB, measured 5.01 MB.
Exact match of the bound calculation for 8 layers × 2 modules × rank-16.

**Theorem 3 (Latency Bound)**: Partially confirmed. Predicted 30-50ms, measured
62-104ms. The underestimate comes from the base model forward pass being slower
than predicted (~60ms vs ~30ms for full-sequence processing at 256 tokens).
The LoRA overhead is indeed negligible (<1ms), as predicted.

### Ceiling Effect

The 100% accuracy with large margins means this experiment establishes a
**lower bound**, not a discrimination ceiling. The format-level differences
(LaTeX vs plain text, SOAP headers vs conversational, legal citations vs casual)
are so salient that even a linear probe without LoRA might achieve high accuracy.

The real test for reward LoRA would be **intra-domain quality discrimination**:
distinguishing good LaTeX from mediocre LaTeX, or a correct SOAP note from one
with wrong ICD codes. That is a harder task not tested here.

### SOAP Latency

SOAP notes are significantly longer than math or legal responses (~800 tokens
vs ~300 tokens). The 103.8ms latency exceeds the 100ms threshold individually
but the aggregate average (83.2ms) passes. For production use, input truncation
or last-N-tokens scoring would bring SOAP under 100ms.

### Training Efficiency

Loss converges to 0.0 by step ~50 for all domains (effectively 2-3 epochs
through the training data). The remaining 150 iterations are wasted. For
production training, early stopping at loss < 0.01 would be appropriate.

## Connection to Pierre Architecture

Reward LoRAs use the same infrastructure as domain adapters (LoRA on Gemma 4
with identical rank and target modules). This means:

1. **Same serving pipeline**: reward adapters hot-swap like domain adapters
2. **Composition scoring**: load domain adapter + reward adapter to score
   whether composed output meets domain standards
3. **Router validation**: reward score confirms router selected correct domain

The 5 MB size and ~83ms latency make reward scoring feasible as an inline
quality check during adapter composition.

## Limitations

1. **Trivial task**: format-level discrimination does not test subtle quality
2. **Small eval set**: 5 pairs per domain; statistical significance limited
3. **Synthetic data**: hardcoded preference pairs, not real user preferences
4. **No cross-domain test**: didn't test if math reward LoRA correctly rejects
   legal text (specificity)
5. **No composition test**: didn't test reward scoring on composed adapter output

## Status: SUPPORTED

All kill criteria pass. Reward LoRA infrastructure is viable on M5 Pro.
The mechanism works but the task is easy — future experiments should test
intra-domain quality discrimination and composition quality scoring.
