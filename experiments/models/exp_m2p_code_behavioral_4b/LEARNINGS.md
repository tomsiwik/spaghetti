# LEARNINGS: exp_m2p_code_behavioral_4b

**Status**: supported (Finding #407)

## Core Finding
Code M2P without SFT residual degrades Qwen3-4B code quality from 42.2% → 6.7% (code_qr=0.158).
This is anti-format interference: the M2P's B-matrices corrupt strong base model priors when no SFT quality floor is enforced.

## Why
The base 4B model already has strong code capability (42.2% pass@1). Without B_applied = B_sft + residual,
the M2P optimization converges to B-matrices that minimize training cross-entropy on the training distribution
but have σ_max >> 0 at eval time, catastrophically interfering with existing base model code knowledge.
Theorem 2's lower bound requires zero-init heads (B ≈ B_sft at step 0); without SFT floor, no lower bound
on qr is derivable (Finding #403 vs this experiment is the proof-by-contrast).

## Implications for Next Experiment
SFT-residual is mandatory for ALL domains where the base model has meaningful capability (code, math, reasoning).
Next: train code SFT adapter (rank=4) for Qwen3-4B-4bit, use sft_b_matrices as residual base in M2PNetworkV6.
Prediction: init_code_qr = 1.0 exactly (zero-init theorem), final code_qr >= 1.0 (same structure as math Finding #403).

## References
- Finding #403: SFT-residual math at 4B (qr=1.175, zero-init theorem verified)
- Finding #395: Format overfitting at 0.6B (different disease — weak base vs. strong base)
- He et al. (2016): Residual connections as structural guarantee (not a heuristic)
