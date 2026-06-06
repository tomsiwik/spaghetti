# Math: Top-2 Output-Space Composition

## 0. Failure Mode & Impossibility Structure

### Failure Mode: Cross-Term Interference in Parameter Merge

When composing two LoRA adapters via parameter merge (weight-space averaging),
the composed output contains cross-terms that corrupt the signal:

```
y_merge = (W + alpha_1 * B_1 A_1 + alpha_2 * B_2 A_2) x
        = Wx + alpha_1 * B_1(A_1 x) + alpha_2 * B_2(A_2 x)
```

This looks clean, but the problem emerges at the *effective weight* level.
When alpha_1 + alpha_2 = 1 (normalized), each adapter contributes only a fraction
of its trained delta. The model trained adapter 1 expecting the FULL delta B_1 A_1,
not 0.5 * B_1 A_1. This scaling dilutes each adapter's effect.

More critically, with our Grassmannian skeleton (A_i orthogonal), the adapters
are "safe" but also cannot constructively interfere. Parameter merge with
orthogonal adapters is a strict convex combination -- no superlinear gains
are possible (proven in exp_lora_soups_cat: "no superlinear composition" finding).

### Impossibility Structure: Output-Space Composition Eliminates Dilution

Output-space composition runs each adapter independently at full strength:

```
y_output = (1/k) * sum_{i in top-k} (W + B_i A_i) x
         = Wx + (1/k) * sum_{i in top-k} B_i(A_i x)
```

Key mathematical property: **each adapter contributes its FULL delta B_i(A_i x)**
scaled by 1/k, rather than a fractional delta merged into shared weights.

For k=2 with complementary domains:
- Adapter 1 specializes on tokens where A_1 x has large projection
- Adapter 2 specializes on tokens where A_2 x has large projection
- If routing is correct, each adapter contributes positively on its tokens

**Why superlinear is possible:** When adapter i is selected for a token, it
contributes its full B_i(A_i x) to 1/2 of the output. The base model Wx
contributes to both terms. So the effective adaptation strength per selected
adapter is (1/2) * B_i(A_i x) -- which is MORE than the parameter-merge
case of (1/5) * B_i(A_i x) when all 5 adapters are merged.

Concretely: output-space top-2 gives each adapter 2.5x the effective
weight of uniform-5 merge (1/2 vs 1/5).

**Impossibility of interference:** With orthogonal A matrices and output-space
composition, adapter i's contribution B_i(A_i x) is computed in isolation.
There are no cross-terms B_i(A_j x) because each forward pass uses only
one adapter. Cross-adapter interference is structurally impossible.

Reference: LoRI (arXiv:2504.07448) proves that output-space composition
eliminates cross-terms. MoE practice (Mixtral top-2/8, DeepSeek-V3 top-2/256)
validates this at scale -- every production MoE uses output-space, not
parameter-merge composition.

## 1. Mechanism Definition

### Output-Space Top-k Composition

Given:
- Base model f_W: R^V -> R^V (maps token sequence to logit distribution)
- N LoRA adapters {(A_i, B_i)}_{i=1}^N, where A_i in R^{d_in x r}, B_i in R^{r x d_out}
- Router g: R^{d_in} -> R^N that scores each adapter's relevance
- k = 2 (top-k parameter)

For input x:

1. **Route:** Compute scores s = g(x), select top-k indices I = argtop2(s)
2. **Independent forward passes:** For each i in I:
   y_i = f_{W + B_i A_i}(x)  -- full forward pass with adapter i applied
3. **Average logits:**
   y = (1/k) * sum_{i in I} y_i

### Routing Mechanism

Simple embedding-similarity routing (no learned router needed for 5 domains):

For each adapter i, compute a domain embedding e_i from the adapter's A matrices:
```
e_i = mean over layers of ||A_i||_F-normalized first singular vector of A_i
```

At inference, route based on cosine similarity between input embedding and
domain embeddings. This avoids the binary-head-collapse problem identified
in exp_binary_routing_head_collapse.

### Complexity

Per-token cost:
- Parameter merge: 1 forward pass at (W + sum deltas), O(d^2) per layer
- Output-space top-2: 2 forward passes at (W + delta_i) each, O(2 * d^2) per layer
- Overhead: exactly 2x FLOPs, 2x latency (sequential) or 1x latency (parallel)

Memory:
- Parameter merge: 1 model copy + merged weights
- Output-space: 1 model copy + 2 adapter weight sets (negligible: 2 * 21MB = 42MB)
  Actually only 1 model copy needed -- apply adapter, run forward, swap adapter, run again

## 2. Why It Works

The mathematical guarantee comes from two properties:

**Property 1: No cross-terms (LoRI theorem)**
In parameter merge: y = (W + sum alpha_i B_i A_i) x
The nonlinear layers process sum(alpha_i B_i A_i x) as a single combined signal.
After nonlinearities, cross-adapter interactions emerge:
  sigma(B_1 A_1 x + B_2 A_2 x) != sigma(B_1 A_1 x) + sigma(B_2 A_2 x)

In output-space: each sigma(B_i A_i x) is computed independently. The only
interaction is the final logit average, which is linear.

**Property 2: Higher effective adapter weight**
With N=5 uniform merge: each adapter contributes (1/5) of its delta.
With top-2 output-space: each selected adapter contributes (1/2) of its delta.
Effective amplification: 2.5x per selected adapter.

For a correctly-routed domain query, this means the relevant adapter's signal
is 2.5x stronger than under uniform merge, while irrelevant adapters contribute
nothing (they are not selected).

## 3. What Breaks It

**Failure condition 1: Router selects wrong adapters**
If the router consistently picks adapters with NO relevant knowledge, output-space
composition degenerates to averaging two copies of the base model (since irrelevant
adapter deltas produce noise that averages to ~0). This would match but not beat
the base model.

Kill criterion K1: If output-space top-2 does not beat single best adapter,
the routing is failing to provide complementary expertise.

**Failure condition 2: Speed too slow**
Two sequential forward passes double latency. If base model is slower than
expected (e.g., bf16 unpack overhead), we may hit K2 (<30 tok/s).

Known: Falcon-E-3B with bf16 unpack runs at ~variable speed. The prior
experiment took ~7s per GSM8K question with 256 max tokens.

**Failure condition 3: Adapter deltas too small to matter**
If the ternary adapters have small effective deltas (||B_i A_i|| << ||W||),
then the 2.5x amplification of a small signal is still small. The base model
dominates and composition provides negligible benefit.

## 4. Assumptions

1. **Adapters are trained and specialized.** Justified: exp_falcon_e3b_composition
   trained 5 domain adapters with measurable PPL improvement on domain data.

2. **Routing can identify relevant adapters.** Justified: exp_softmax_router_scaling
   showed softmax router matches oracle at N=24. For N=5, simple similarity
   should suffice.

3. **Logit averaging is a reasonable aggregation.** Justified: this is exactly
   what MoE models do (Mixtral, DeepSeek-V3). Logit-space is approximately
   linear for small perturbations.

4. **Two forward passes fit in latency budget.** Assumption: each pass ~50 tok/s,
   so top-2 ~25 tok/s minimum. Needs testing -- K2 threshold is 30 tok/s.

## 5. Complexity Analysis

| Operation | FLOPs | Memory |
|-----------|-------|--------|
| Base forward | O(L * d^2 * n) | ~6.7 GB (bf16 unpack) |
| Apply adapter i | O(L * d * r * n) | +21 MB per adapter |
| Top-2 output-space | 2 * O(L * d^2 * n) | ~6.7 GB + 42 MB |
| Parameter merge (N=5) | O(L * d^2 * n) | ~6.7 GB (merged) |
| Top-2 parameter merge | O(L * d^2 * n) | ~6.7 GB (merged) |

Where L=32 layers, d=2048, r=16, n=sequence length.

Output-space is 2x FLOPs but same memory (sequential adapter application).

## 6. Worked Example

d=64, r=4, N=5 adapters, k=2.

Input token x in R^64.

**Parameter merge (uniform 1/5):**
```
delta_W = (1/5) * sum_{i=1}^5 B_i A_i    (shape 64x64)
y = (W + delta_W) x
Each adapter contributes 20% of its delta.
```

**Output-space top-2:**
```
Router selects adapters {2, 4} (medical, legal).
y_2 = (W + B_2 A_2) x    -- 100% of medical delta
y_4 = (W + B_4 A_4) x    -- 100% of legal delta
y = 0.5 * y_2 + 0.5 * y_4
Each selected adapter contributes 50% of its delta.
```

Medical adapter's effective contribution:
- Merge: 0.2 * B_2(A_2 x) = 0.2 * delta
- Top-2: 0.5 * B_2(A_2 x) = 0.5 * delta
- Amplification: 2.5x

## 7. Connection to Architecture

Output-space composition is exactly how production MoE models work:
- Mixtral: top-2 of 8 FFN experts, output averaged with gate weights
- DeepSeek-V3: top-2 of 256 experts + 1 shared expert
- Our architecture: top-2 of N LoRA adapters, logit averaged

The difference from full MoE: our "experts" are LoRA deltas applied to the
full model, not separate FFN blocks. This means:
- Each expert shares the base model (memory efficient)
- Each expert modifies all layers (more expressive than per-layer MoE)
- Routing is per-sequence, not per-token (matching our finding that
  per-sequence routing is correct granularity)

This is the first test of MoE-style output-space composition in our stack.
If it works, it validates the core thesis: compose only the relevant experts
at runtime, pay only for what you use.
