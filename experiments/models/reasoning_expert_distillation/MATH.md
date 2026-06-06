# Reasoning Expert Distillation: Mathematical Foundations

## 1. Problem Formulation

### 1.1 Core Claim

Reasoning capability is **orthogonal** to domain knowledge in LoRA weight space.
A reasoning adapter R and a domain adapter D can be composed via pre-merge addition:

```
W_composed = W_base + Delta_R + Delta_D
```

where Delta_R captures chain-of-thought reasoning patterns and Delta_D captures
domain-specific knowledge, with minimal interference:

```
|cos(vec(Delta_R), vec(Delta_D))| << 1
```

### 1.2 Notation

| Symbol | Dimensions | Description |
|--------|-----------|-------------|
| W_base | (d_out, d_in) | Frozen base model weight matrix |
| A_R | (r, d_in) | Reasoning LoRA A-matrix |
| B_R | (d_out, r) | Reasoning LoRA B-matrix |
| A_D | (r, d_in) | Domain LoRA A-matrix |
| B_D | (d_out, r) | Domain LoRA B-matrix |
| Delta_R | (d_out, d_in) | Reasoning delta: (alpha/r) * B_R @ A_R |
| Delta_D | (d_out, d_in) | Domain delta: (alpha/r) * B_D @ A_D |
| r | scalar | LoRA rank (16) |
| alpha | scalar | LoRA scaling factor (16) |
| d | scalar | Model hidden dimension (3584 for Qwen2.5-7B) |
| d_ff | scalar | FFN intermediate dimension (18944) |
| L | scalar | Number of transformer layers (28) |
| N_mod | scalar | Number of target modules per layer (7) |

## 2. Orthogonality of Reasoning vs Domain

### 2.1 Expected Cosine Similarity

For independently trained LoRA adapters, the expected cosine similarity
in weight space is bounded by the Johnson-Lindenstrauss concentration:

```
E[|cos(vec(Delta_R), vec(Delta_D))|] <= sqrt(2 / (pi * D))
```

where D is the total parameter count of the flattened delta vector.

For Qwen2.5-7B with rank-16, all-modules LoRA:

```
D_per_layer = 2 * r * (d + d + d_kv + d_kv + d + d_ff + d_ff + d_ff)
            = 2 * 16 * (3584 + 3584 + 512 + 512 + 3584 + 18944 + 18944 + 3584)
            = 2 * 16 * 53248
            = 1,703,936

D_total = L * D_per_layer = 28 * 1,703,936 = 47,710,208
```

Expected random cosine:

```
E[|cos|] = sqrt(2 / (pi * 47,710,208)) = sqrt(1.335e-8) = 1.155e-4
```

Empirically observed at d=896 (Qwen2.5-0.5B): |cos| = 0.0002, which is
2x the random baseline. At d=3584 (7B), we expect even lower values due
to higher dimensionality.

### 2.2 Why Reasoning is Orthogonal to Domain Knowledge

**Intuition**: Reasoning (chain-of-thought, step decomposition, verification)
operates on the model's **computation patterns** -- how it chains intermediate
steps. Domain knowledge operates on **content patterns** -- what facts and
relationships it knows. These occupy different subspaces of the weight matrix.

**Formal argument**: Let Span(A_R) denote the column space of A_R (the
"input subspace" that reasoning attends to). For orthogonal composition:

```
Span(A_R) ∩ Span(A_D) ≈ {0}
```

This holds when:
1. rank(A_R) + rank(A_D) <= d_in (capacity constraint: 16 + 16 = 32 << 3584)
2. Training data induces different gradient directions (math reasoning traces
   vs domain Q&A pairs have different loss landscapes)

The capacity condition is trivially satisfied. The gradient direction condition
is the empirical hypothesis being tested.

### 2.3 Interference Bound

When composing via pre-merge addition:

```
h_composed = (W_base + Delta_R + Delta_D) @ x
           = W_base @ x + Delta_R @ x + Delta_D @ x
           = h_base + h_R + h_D
```

The interference term is:

```
||h_R||^2 + ||h_D||^2 + 2 * <h_R, h_D>
```

For orthogonal deltas, the cross-term <h_R, h_D> is small:

```
|<h_R, h_D>| = |x^T @ Delta_R^T @ Delta_D @ x|
             <= ||Delta_R||_F * ||Delta_D||_F * ||x||^2 * |cos(Delta_R, Delta_D)|
```

At |cos| ~ 0.0002, the cross-term is negligible compared to the individual
contributions. This means:

- Domain quality degradation from adding reasoning adapter is bounded by
  the cross-term, which is proportional to |cos|.
- We predict degradation < 1% for truly orthogonal adapters at d=3584.

## 3. Distillation Process

### 3.1 Hard Distillation from Reasoning Traces

Given a teacher model T (DeepSeek-R1 671B) and student model S (Qwen2.5-7B):

**Step 1**: Generate reasoning traces from T on math problems.

```
For each problem p_i:
    t_i = T.generate(p_i)  # Includes <think>...</think> trace
    a_i = T.extract_answer(t_i)  # Final boxed answer
```

**Step 2**: Format as supervised fine-tuning data.

```
Input:  system_prompt + user: p_i
Target: <think>\n{reasoning_trace}\n</think>\n\n{answer_with_boxed}
```

**Step 3**: Train LoRA adapter on S to minimize cross-entropy loss on
the full sequence including the reasoning trace.

```
L = -1/T * sum_{j=1}^{T} log P_S(token_j | token_{<j}; W_base + Delta_R)
```

Key design choice: The loss is computed over the **entire** assistant response
including the <think>...</think> tokens. This forces the student to learn
the reasoning process, not just the final answer.

### 3.2 Why Hard Distillation, Not RL

DeepSeek-R1 used RL (GRPO) to develop reasoning from scratch. However, for
distillation into a LoRA adapter:

1. **RL requires reward signal**: Verifying math answers is possible but
   adds complexity. SFT on traces is simpler and sufficient.
2. **LoRA capacity constraint**: Rank-16 LoRA has ~47M params. RL exploration
   over this space is inefficient. SFT directly imprints the target behavior.
3. **Budget constraint**: RL training requires 10-100x more compute than SFT.
   At ~$0.25/expert budget, SFT is the only viable approach.
4. **Prior art validates this**: DeepSeek-R1 paper shows SFT distillation
   of R1 traces into Qwen-7B achieves strong results (reportedly >55% on MATH-500
   for the distilled 7B model).

### 3.3 Training Budget

Dataset: rasbt/math_distill (~12K examples with thinking traces from DeepSeek-R1)

```
Tokens per example (avg): ~1000 (problem: ~50, thinking: ~800, answer: ~150)
Total training tokens: 12K * 1000 = 12M tokens
Tokens per step (batch=4, seq_len=2048, packing): ~8K tokens/step
Steps for 1 epoch: 12M / 8K = 1500 steps
Our budget: 500 steps ≈ 0.33 epochs
```

This is under-trained by choice. The hypothesis is that reasoning capability
can be captured with modest training (analogous to how domain experts train
in 300 steps). If 500 steps is insufficient, the kill criterion will catch it.

## 4. Composition Mechanics

### 4.1 Pre-Merge Composition

Given k adapters with deltas {Delta_1, ..., Delta_k}, pre-merge computes:

```
W_composed = W_base + sum_{i=1}^{k} Delta_i
```

For our experiment with k=2 (reasoning + domain):

```
W_composed = W_base + (alpha/r) * (B_R @ A_R + B_D @ A_D)
```

### 4.2 Cost Analysis

**Training cost** (reasoning adapter):

```
Time: ~500 steps * 1.5s/step = 750s ≈ 12.5 min
GPU cost: 12.5/60 * $0.34 = $0.07
Dataset cost: $0 (using existing rasbt/math_distill)
Total: ~$0.07
```

Note: This is lower than pilot50 ($0.44/expert) because we use an existing
public dataset instead of generating data via API.

**Composition cost** (pre-merge at inference):

```
Delta merge: one-time O(L * N_mod * d^2) addition = ~47M FLOPs
Per-token overhead: 0% (merged into base weights)
```

### 4.3 Scaling: N Capability Adapters

If reasoning is one of N composable capability types, the total composed model:

```
W = W_base + sum_{i=1}^{N_cap} Delta_cap_i + sum_{j=1}^{N_domain} Delta_domain_j
```

Orthogonality constraint requires:

```
N_total = N_cap + N_domain <= d^2/r^2 ≈ 50,176 (at d=3584, r=16)
```

Even with 100 capability adapters and 5,000 domain experts, we use only
~10% of the orthogonal capacity.

## 5. Kill Criteria Formalization

### K1: Reasoning Distillation Success

```
Acc(base + Delta_R, MATH-500) - Acc(base, MATH-500) > 10pp
```

Rationale: DeepSeek-R1-Distill-Qwen-7B reportedly achieves ~55% on MATH-500.
Base Qwen2.5-7B achieves ~30-40% (estimate). Our LoRA distillation should
capture at least 10pp of this gap to be considered successful.

### K2: Composition Interference

```
For all domain experts D_i:
    PPL(base + Delta_D_i + Delta_R, eval_D_i) / PPL(base + Delta_D_i, eval_D_i) - 1 <= 5%
```

Rationale: If reasoning adapter degrades domain quality by more than 5%, the
composition tax is too high. SOLE orthogonality predicts <1% degradation.

### K3: Composition Superiority

```
Acc(base + Delta_R + Delta_D_math, MATH-500) > max(
    Acc(base + Delta_R, MATH-500),
    Acc(base + Delta_D_math, MATH-500)
)
```

Rationale: If the composed model does not outperform the best single adapter,
composition adds no value for reasoning tasks. This would imply reasoning and
domain knowledge are redundant, not complementary.

## 6. Worked Example

At micro scale (d=64, r=8, 4-layer model):

```
D_total = 4 layers * 7 modules * 2 * 8 * 64 = 28,672 params
E[|cos|] = sqrt(2 / (pi * 28672)) = 0.0047

With d=3584, r=16:
D_total = 28 * 7 * 2 * 16 * ~3584 = ~47M params
E[|cos|] = sqrt(2 / (pi * 47e6)) = 0.000116
```

At d=3584, two independently trained adapters should have cosine similarity
~0.0001 -- essentially orthogonal. This predicts:

- Interference < 0.01% (negligible)
- Domain degradation < 0.1% (far below 5% threshold)
- Composition behaves as pure addition (no destructive interference)

## 7. Assumptions

1. **DeepSeek-R1 traces are sufficient training signal**: The reasoning
   patterns in R1 traces generalize beyond the specific math problems.
   Could fail if reasoning is problem-specific rather than general.

2. **Rank-16 has sufficient capacity for reasoning**: Chain-of-thought
   requires pattern reuse (step decomposition, verification). If reasoning
   requires novel computation patterns not present in the base model,
   rank-16 may be insufficient.

3. **SFT captures reasoning, not just memorization**: The student could
   memorize answer patterns without learning the reasoning process.
   Detectable by comparing performance on in-distribution vs OOD problems.

4. **Pre-merge composition preserves reasoning**: Adding a domain adapter
   could disrupt the reasoning trace generation. The <think> tokens
   might be sensitive to weight perturbation.

5. **Qwen2.5-7B has latent reasoning capability**: The base model must
   have sufficient architecture to support chain-of-thought. LoRA cannot
   add fundamentally new computation, only redirect existing capacity.
