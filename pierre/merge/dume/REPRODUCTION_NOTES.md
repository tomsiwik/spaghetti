# DUME Reproduction Notes

Paper: [Training-Free Dynamic Upcycling of Expert Language Models](https://arxiv.org/abs/2603.29765)
Authors: Eros Fani, Oguzhan Ersoy
Official code: https://github.com/gensyn-ai/dume

## What This Implementation Covers

- **MoErging** (Section 2.2): averaging non-MLP parameters across dense experts, keeping MLP blocks as separate MoE experts
- **Ridge regression router** (Section 2.1, 2.3): closed-form W* = (X^TX + lambda I)^{-1} X^TY
- **Incremental statistics accumulation** (Eq. 3, 7): sequential update of A_l and b_l matrices
- **Column normalization** (Section 2.3): per-column L2 normalization for domain imbalance
- **Top-k routing** (Eq. 4): Softmax-based gating with k=1 default
- **Deterministic extraction forward** (Algorithm 1): during statistics collection, tokens are routed to their domain's expert (not through the router)

## What Is NOT Implemented

- Base model loading (use mlx-lm for LLaMA/Qwen model loading)
- DUME+ (additional router finetuning after initialization)
- DUME OOD (out-of-distribution router extraction)
- Specific dataset processing pipelines from Section 3.1

## Key Implementation Decisions

1. **mx.linalg.solve instead of explicit inverse**: More numerically stable than computing (A + lambda I)^{-1} directly. Equivalent result.

2. **RouterStatistics.B uses .at[].add()**: The one-hot Y matrix means only one column of b_l is updated per domain batch. We accumulate directly into that column.

3. **MoE block uses sequential expert dispatch**: For each top-k slot, we gather tokens per expert, forward, and scatter back. This avoids materializing D copies of the full tensor.

4. **All accumulation in float32**: The A_l matrices are H x H (e.g., 2048 x 2048 = 16MB in float32). Accumulating in lower precision would lose information over many batches.

## Hyperparameters (from paper)

- lambda = 0.1 (Appendix A discusses sensitivity)
- top_k = 1 (matches dense expert forward cost)
- Token window: full sequence works; shorter windows (e.g., 512) retain most performance (Appendix A)

## Computational Costs (Section 2.3)

- Forward cost: O(n * T * H^2) for n samples, T tokens, H hidden dim
- Ridge solve: O(H^3) per layer (matrix solve on H x H system)
- Total: O(n*T*H^2 + L*H^3) — no backpropagation needed

## Verification Checklist

- [ ] MoErging produces correct averaged attention weights
- [ ] Ridge regression router matches sklearn RidgeClassifier on toy data
- [ ] Incremental accumulation matches single-batch solution
- [ ] Column normalization produces unit-norm columns
- [ ] Top-1 routing selects correct expert for in-domain data
