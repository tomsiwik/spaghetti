# PAPER.md: M2P v5 SHINE-Aligned Base-as-Encoder

## Measurement vs Prediction

| Metric | Predicted | Measured | Match |
|--------|-----------|----------|-------|
| quality_ratio | 0.60-0.85 | 0.8333 | YES (exceeds v3's 0.75) |
| M2P transformer params | ~48M | ~50.4M | YES |
| Total M2P params | ~401M | ~402.7M | YES |
| K1: grad_norm > 0 | > 0 | 8.18 | YES |

## Analysis

The architecture change to route memory tokens through the frozen base model and use alternating row/column attention (SHINE) was highly successful. It resolved the mean-pooling bottleneck issue by allowing layer-specific representation extraction.

Quality ratio increased from 0.754 (v3/v4) to 0.8333 (v5), showing that preserving per-layer feature differentiation via cross-layer attention directly improves adapter synthesis. 

## Next Steps

This validates the v5 SHINE-aligned M2P on the 0.6B scale. The critical path now requires validating this exact architecture on the 4B scale (`exp_m2p_qwen4b_gsm8k_v5`) to prove that we have resolved the mode collapse observed in Finding #400.
