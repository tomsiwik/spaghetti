# LEARNINGS: exp_p9_ttlora_quality

## Core Finding
TT-LoRA rank-6 retains 84.4% of standard LoRA quality on GSM8K while using 12.4x fewer parameters (64K vs 798K) and producing a 154 KB adapter — enabling 25 domain adapters in 3.75 MB total vs 75 MB for LoRA.

## Why
Oseledets (2011) TT decomposition constrains corrections to a Kronecker-structured submanifold. Both methods produce identical rank-6 corrections by construction (Theorem 2), but TT-LoRA's reachable set is a proper subset of LoRA's. For GSM8K math reasoning, ~84% of the optimal v_proj direction lies within this submanifold — enough for useful domain adaptation.

## Key Numbers
- Quality ratio: 84.4% (65/100 vs 77/100 GSM8K accuracy)
- Param compression: 12.4x (precision-independent)
- Adapter size: 154 KB (float16) — 20x vs float32 LoRA, 10x at equal precision
- Training cost: 4x slower (87 min vs 22 min for 1000 steps on M5 Pro)
- Final loss: TT-LoRA 0.369 vs LoRA 0.403 (lower final loss despite lower accuracy — possible over-training)

## Implications for Next Experiment
1. **Fewer training steps**: TT-LoRA's lower final loss (0.37 vs 0.40) suggests possible over-training; 300-500 steps may close the speed gap while maintaining quality. Test whether 500-step TT-LoRA matches 1000-step LoRA.
2. **Multi-domain composition**: At 154 KB/adapter, the composition path is unblocked. Next: verify TT-LoRA adapters compose with the same null-space interference guarantees as standard LoRA (Finding #225).
3. **Primary metric**: Use 12.4x param compression (precision-independent) as the headline number, not the 20x size comparison.

## References
- arXiv:2504.21190 (Batselier et al. 2025, TT-LoRA MoE) — source paper
- Finding #515: MLX port confirmed (8.3x compression, 1.36x latency overhead)
- Finding #516: This experiment (84.4% quality, 12.4x compression, supported)
