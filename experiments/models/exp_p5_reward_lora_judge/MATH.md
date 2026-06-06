# P5.B1: Per-Domain Reward LoRA Judge — Mathematical Framework

## Problem Statement

Given composable domain adapters, we need a cheap quality signal to score whether
a response matches domain-specific criteria. Full RLHF is too expensive; we need
a per-domain reward function that runs in <100ms and fits in <10MB.

**Hypothesis**: A rank-16 LoRA adapter + linear reward head trained with
Bradley-Terry loss on preference pairs can discriminate domain-appropriate from
domain-inappropriate text with ≥80% accuracy.

**Grounding**: arXiv:2506.05748 shows reward LoRA (0.8% params) achieves 96.2%
on RewardBench. We test whether this transfers to domain-specific quality scoring
on Gemma 4 E4B.

## Definitions

Let M be a frozen LLM with hidden dimension d. For input token sequence x ∈ V^T:
- h(x) = backbone(x) ∈ R^{T×d} — hidden states after final norm
- h_last(x) = h(x)[T-1] ∈ R^d — last-token representation

A **reward LoRA** is (ΔW, w) where:
- ΔW = {B_l A_l : l ∈ L_target} is a rank-r LoRA on target layers
- w ∈ R^d is a linear reward head

The reward function: r(x) = w^T · h_last(x; M + ΔW)

## Theorem 1 (Reward Subspace Capacity)

**Theorem**: Let f: R^d → {0,1} be a domain quality indicator whose decision
boundary has intrinsic dimensionality k. A rank-r LoRA on m modules across l
layers, combined with a linear reward head, can represent any linear separator
in a subspace of dimension min(d, 2·m·l·r + d).

**Proof**: Each LoRA module adds ΔW_i = B_i A_i where B_i ∈ R^{d_out×r},
A_i ∈ R^{r×d_in}. The modified hidden state h' = f(Wh + B·A·h) introduces
r new directional components per module per layer. Across l layers with m
modules each, the representation gains up to m·l·r additional degrees of
freedom (not all independent due to nonlinearities, but each adds capacity).

The reward head w ∈ R^d projects the final representation to a scalar. The
combined discriminator w^T h'(x) operates in the span of original features
plus LoRA-injected directions.

For our setting: r=16, m=2 (q_proj + o_proj), l=8 (last 8 layers).
Capacity = 2·2·8·16 = 512 dimensions. Our domains are separable in ≤6
dimensions (Finding #474: TF-IDF routing achieves 97.3% with sparse features),
so 512 >> 6 provides massive overcapacity. ∎

**Corollary**: For domain discrimination with intrinsic dimensionality k ≤ 6,
rank-16 LoRA is vastly overparameterized. The experiment should converge
quickly (predicted: <200 iterations).

## Theorem 2 (Size Bound)

**Theorem**: For Gemma 4 E4B (d=2560, n_layers=42) with rank-r LoRA on m
target modules over l layers, the adapter size in bytes is:

S = Σ_{i∈layers} Σ_{j∈modules} 2 · (d_in^{(i,j)} · r + r · d_out^{(i,j)}) · sizeof(dtype) + d · sizeof(dtype)

**Proof**: Each LoRA module stores A ∈ R^{r×d_in} and B ∈ R^{d_out×r}.
The reward head stores w ∈ R^d. Factor 2 accounts for A and B matrices.

For E4B with float16 (2 bytes), r=16, targeting q_proj + o_proj on layers 34-41:
- Sliding attention layers (d_q=2048): 2·(2560·16 + 16·2048)·2 + 2·(2048·16 + 16·2560)·2 = 589,824 bytes
- Full attention layers (d_q=4096): 2·(2560·16 + 16·4096)·2 + 2·(4096·16 + 16·2560)·2 = 851,968 bytes
- Reward head: 2560·2 = 5,120 bytes

With ~6 sliding + ~2 full in layers 34-41:
Total ≈ 6·589,824 + 2·851,968 + 5,120 = 5,247,000 bytes ≈ 5.0 MB < 10 MB ∎

**Prediction (K1274)**: PASS. Adapter size ≈ 5 MB, well under 10 MB threshold.

## Theorem 3 (Latency Bound)

**Theorem**: Reward scoring latency on M5 Pro is bounded by:
t_reward ≤ t_base_fwd + t_lora + t_head

where:
- t_base_fwd: base model forward pass on quantized weights ≈ 20-40ms for T≤256
- t_lora: additional LoRA computation = 4·m·l·d·r FLOPs. For m=2, l=8, d=2560, r=16:
  t_lora ≈ 2.6M FLOPs. At ~4 TFLOPS → <1ms
- t_head: linear projection d→1: negligible

**Prediction (K1275)**: PASS. Total ≈ 30-50ms, well under 100ms threshold.

## Predictions Summary

| Kill Criterion | Prediction | Rationale |
|---|---|---|
| K1273: ≥80% agreement | PASS (~85-90%) | Domain quality differences are large (format, terminology); rank-16 overcapacity (512 dim vs 6 intrinsic) |
| K1274: <10MB adapter | PASS (~5MB) | 8 layers × 2 modules × rank-16 × float16 + reward head |
| K1275: <100ms latency | PASS (~30-50ms) | Single forward pass on 4-bit model; LoRA overhead <1ms |

## Training Design

**Loss**: Bradley-Terry preference model:
L = -log σ(r(x, y_w) - r(x, y_l))

where y_w is preferred (domain-appropriate) and y_l is rejected (domain-inappropriate).

**Data**: Synthetic preference pairs per domain:
- Preferred: correct domain formatting (LaTeX, SOAP, legal citations)
- Rejected: casual/generic responses without domain formatting
- 15 training + 5 evaluation pairs per domain

**Architecture**:
- Base: Gemma 4 E4B 4-bit (frozen)
- LoRA: rank-16 on q_proj + o_proj of last 8 layers
  - Note: v_proj on layers 22+ is unused due to KV sharing (num_kv_shared_layers=20)
  - q_proj modifies attention patterns; o_proj modifies output projection
- Reward head: Linear(2560, 1, bias=False)

## Connection to Pierre Architecture

Domain-specific reward LoRAs enable:
1. **Online quality monitoring**: score adapter outputs without human evaluation
2. **Routing validation**: verify router selects correct adapter by checking reward
3. **Composition quality**: score composed outputs to detect interference

Each reward adapter is itself composable — it uses the same LoRA infrastructure
as domain adapters but targets a different objective (scoring vs generation).
