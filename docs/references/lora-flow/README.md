# LoRA-Flow: Dynamic LoRA Fusion for Large Language Models in Generative Tasks

**Source**: https://arxiv.org/abs/2402.11455

**Authors**: Hanqing Wang, Bowen Ping, Shuo Wang, Xu Han, Yun Chen, Zhiyuan Liu, Maosong Sun

**Code**: https://github.com/PKU-TANGENT/LoRA-Flow

## Key Insight

Dynamic per-layer per-token fusion gate for combining independently-trained LoRA adapters.
Uses a lightweight gate: w^l = softmax(W_gate^l @ x_t^l) + b^l per layer, where the gate
learns to route different tokens to different expert mixtures.

Gate parameters are extremely small (0.26M for Llama-2-7b with k=2, ~0.2% of a single LoRA).
Only gate params trained (frozen backbone + frozen LoRAs). 200 training examples suffice.

## Relevance to Our Work

LoRA-Flow is the most expressive composition method in the hierarchy:
SOLE (c=1) subset CAT (c=w_i^l) subset LoRA-Flow (c=f(x,l)).

Our micro experiment (exp_lora_flow_comparison) shows that under structural
orthogonality, the additional expressivity provides zero quality gain --
the optimal weights are trivially 1.0 when experts do not interfere.

Key scaling limitation: gate params = L*k*(d+1). At production scale
(d=4096, L=32), k=500 requires 65.6M gate params, exceeding a single
LoRA adapter (40.4M). SOLE's zero-parameter composition is fundamentally
better suited for large expert libraries.

## Key Results from Paper

- MGSM (math): 37.6 vs LoRA-Hub 28.7 vs Average 13.9 (Llama-2-7b)
- HumanEval (code): 22.6 vs LoRA-Hub 20.3 vs Average 17.7
- Dynamic weights consistently outperform static merging on generative tasks
- The improvement is largest when tasks require switching between skills
  within a single generation (e.g., language + math)

## Positioning vs SOLE

| Property | SOLE | LoRA-Flow |
|----------|------|-----------|
| Use case | N>>10 orthogonal experts | k<=10 overlapping skills |
| Params | 0 | L*k*(d+1) |
| Routing | Static (all-included) | Dynamic (per-token) |
| Evolution | Clone-compete | Retrain gate |
| Scale limit | N_max ~ d^2/r^2 | Gate size O(k*d*L) |

Complementary, not competing. LoRA-Flow for specialized binary composition,
SOLE for large-scale expert library management.
