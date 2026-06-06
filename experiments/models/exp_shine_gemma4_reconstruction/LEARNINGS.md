# LEARNINGS: SHINE S2 — Context Reconstruction via M2P-Generated LoRA

## Core Finding
M2P hypernetwork (3.6M params) successfully trains via gradient flow through 4-bit quantized Gemma 4 layers, generating LoRA that reduces test CE by 86.6% — but falls into the centroid trap (cos=0.998), producing one universal adapter rather than context-specific ones.

## Why
MLX's `mx.quantized_matmul` propagates gradients w.r.t. input correctly, enabling hypernetwork training via LoRA injection on quantized models (novel, non-trivial). Centroid trap arises because 40 chunks from 10 Wikipedia passages are too homogeneous: a single adapter minimizes average loss, so M2P has no gradient signal to learn context-specificity. Cited: arXiv:2602.06358 (SHINE).

## Implications for S3
1. **Contrastive loss is necessary** — more data alone won't break the centroid trap without a mechanism that penalizes similar LoRA for dissimilar contexts (reviewer insight)
2. **Mixed q_proj dims (2048/4096)** handled correctly in M2P output projection — carry forward
3. **Resource headroom**: 434ms/step, 4.68GB active — room for larger M2P and longer training
4. **S3 must include behavioral samples** — CE reduction alone insufficient per proof-first guardrails
5. **Overfitting (6× train/test gap)** suggests training schedule should be adjusted in S3
