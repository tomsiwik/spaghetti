# Text-to-LoRA Hypernetwork: Mathematical Foundations

## References

- Text-to-LoRA (arxiv 2506.06105, ICML 2025): Hypernetwork generates task-specific LoRA
  from text description in single forward pass. Architecture: text encoder -> MLP -> B matrix.
- FlyLoRA (arxiv 2510.08396): Frozen random A as implicit router, JL-lemma orthogonality.
- exp_real_data_25_domain_adapters: 24 trained adapters, Grassmannian A, ternary B.
- exp_adapter_distillation_from_large: Distribution mismatch kills cross-model transfer.

## 1. Mechanism Definition

### LoRA Structure Recap

For each projection p in {q,k,v,o,gate,up,down} at each layer l:

    delta_W_{l,p} = A_{l,p,i} @ B_{l,p,i}

where:
- A_{l,p,i} in R^{d_in x r}: frozen Grassmannian skeleton (domain index i selects pre-computed A)
- B_{l,p,i} in R^{r x d_out}: trained adapter weight (this is what we want to GENERATE)
- r = 16 (LoRA rank)

Total B parameters per adapter: sum over 30 layers x 7 projections:
- q_proj: 16 x 2560 = 40,960
- k_proj: 16 x 640 = 10,240
- v_proj: 16 x 640 = 10,240
- o_proj: 16 x 2560 = 40,960
- gate_proj: 16 x 6912 = 110,592
- up_proj: 16 x 6912 = 110,592
- down_proj: 16 x 2560 = 40,960

Per layer: 364,544 params. Total: 30 x 364,544 = 10,936,320 params per adapter.

### Hypernetwork Architecture (T2L-style)

The hypernetwork H maps a text description to B matrices:

    H: R^{d_embed} -> R^{10,936,320}

This is intractable to output directly. T2L uses a shared-trunk + per-projection-head design:

    z = TextEncoder(description)           # z in R^{d_embed}
    h = MLP_trunk(z)                        # h in R^{d_hidden}
    B_{l,p} = Head_{l,p}(h)                # reshape to (r, d_out_p)

With 210 projection targets and a shared trunk, this factorizes the problem.

**Simplified feasible design for micro-experiment:**

Since 210 heads x (r x d_out) is still ~11M output params, we use a further factorization:

    B_{l,p} = U_p @ diag(s_{l,p}(h)) @ V_p^T

where:
- U_p in R^{r x k}, V_p in R^{d_out x k}: shared low-rank bases per projection type (7 types)
- s_{l,p}(h) in R^k: per-layer-projection scaling predicted by hypernetwork
- k << min(r, d_out): factorization rank (e.g., k=8)

This reduces output dimension from 10.9M to 30 * 210 * k = 50,400 (at k=8).

**Even simpler (Option C baseline):**

Nearest-neighbor in embedding space: embed each domain description, find closest
trained adapter, use that. This tests text->adapter mapping without hypernetwork training.

## 2. Why It Works (or Should Work)

T2L shows that text descriptions contain sufficient information to predict task-specific
adapter parameters because:

1. **Semantic clustering:** Domain descriptions cluster in embedding space the same way
   task-relevant adapters do. If medical and health_fitness are close in text space,
   their trained B-matrices should be related (similar data distributions).

2. **Low-rank structure of adapter space:** The 24 trained adapters span a low-dimensional
   manifold in B-parameter space. The hypernetwork learns a mapping from text space to
   this manifold.

3. **Grassmannian A provides structure:** Since A is frozen and domain-indexed, the
   hypernetwork only needs to predict B. The A matrix already encodes "which subspace
   this adapter operates in" -- B encodes "how much and in what direction within that
   subspace."

### Mathematical justification for embedding-based retrieval (Option C)

Let phi(text) in R^d be the mean-pooled hidden state from the base model. Define:

    adapter_select(text) = argmin_i ||phi(text) - phi(description_i)||_2

This works if the embedding space separates domains -- which we already know it does
(softmax router achieves 40% per-sample accuracy using hidden states, and routing
errors are benign within semantic clusters).

## 3. What Breaks It

### K1: T2L-generated adapter PPL > 3x trained adapter

This fails when:
- The hypernetwork has insufficient capacity to model the B-matrix manifold
- The text description doesn't capture enough about the target distribution
- 24 training examples (adapters) is far too few for a supervised hypernetwork

**Expected failure:** With only 24 training pairs (description, B-matrix), a hypernetwork
will massively overfit. Leave-one-out validation will show poor generalization.
This is the fundamental challenge: T2L trains on thousands of task-adapter pairs.

### K2: Post-processed adapters lose > 50% quality after orthogonal projection

Given existing adapters {B_1, ..., B_N} and a generated adapter B_new, project:

    B_new_proj = B_new - sum_i <B_new, B_i> / ||B_i||^2 * B_i

(Gram-Schmidt against existing adapter B subspaces, applied per-projection)

This fails when B_new is highly correlated with existing adapters -- which is likely
since the hypernetwork is trained ON those adapters.

### K3: Memory constraint

Hypernetwork + base model must fit in 48GB.
- Base model: ~1.18 GB (ternary, quantized)
- Hypernetwork: depends on size. A simple MLP with d_hidden=512 and 210 heads:
  - Trunk: 2560*512 + 512*512 = ~1.6M params (~6.4MB)
  - Heads: 210 * (512 * average_output_dim) -- average B per projection ~52K params
  - 210 * 512 * 16 = 1.7M per head type... too big
  - Factored: 210 * k * 512 = 860K params at k=8 (~3.4MB)
  - Total: ~10MB. K3 trivially passes.

## 4. Projection Operator for Composition Safety

After generating B_new for a new domain, enforce Grassmannian compatibility:

For each projection p at each layer l, let B_existing = {B_{l,p,1}, ..., B_{l,p,N}}.

    B_{l,p,new}^proj = B_{l,p,new} - sum_i (B_{l,p,new} @ B_{l,p,i}^T) / ||B_{l,p,i}||_F^2 * B_{l,p,i}

where the inner product is Frobenius. This is the standard orthogonal complement projection.

**Quality loss from projection:** If B_new has large components along existing B_i directions,
projection removes signal. The fraction of variance retained is:

    rho = ||B_proj||_F^2 / ||B_new||_F^2

If rho < 0.5, we've lost >50% of the adapter's information -> K2 FAIL.

## 5. Complexity Analysis

- **Embedding extraction:** One forward pass through base model per description: O(L * d^2 * seq_len)
- **Hypernetwork forward:** O(d_embed * d_hidden + d_hidden * output_dim) per adapter
- **Projection:** O(N * r * d_out) per projection, 210 projections = O(210 * N * r * d_out)
- **Evaluation:** Standard PPL computation on validation data

Total experiment time: ~10-15 min (embedding + hypernetwork train + eval)

## 6. Worked Example (micro scale)

d_embed = 2560, k = 8, 24 domains, 7 projection types.

Hypernetwork trunk: 2560 -> 256 -> 256 (131K params)
Per-layer-projection heads: 256 -> 8 (for scaling factors)
Total heads: 30 layers * 7 projections = 210 heads of size 256->8 = 430K params
Shared bases U_p, V_p: 7 types * (16*8 + max_d_out*8) ~= 7 * (128 + 55K) = ~387K params

Total hypernetwork: ~948K params (~3.8 MB in float32).

## 7. Experimental Design

### Phase 1: Baseline -- Nearest-neighbor retrieval (Option C)
- Embed 24 domain descriptions through BitNet-2B-4T
- For each domain, find nearest neighbor (leave-one-out)
- Use nearest neighbor's adapter, evaluate PPL on target domain
- This gives the "free lunch" baseline without any training

### Phase 2: Hypernetwork training (Option A simplified)
- Train small MLP: embedding -> B-matrix
- Leave-one-out cross-validation (train on 23, predict 1)
- Evaluate generated adapter PPL vs trained adapter PPL

### Phase 3: Orthogonal projection test
- Take generated adapters, project against existing trained adapters
- Measure quality retention (rho = ||B_proj||^2 / ||B||^2)
- Evaluate projected adapter PPL

### Kill criteria evaluation:
- K1: max(generated_PPL / trained_PPL) across domains. KILL if > 3.0
- K2: mean(projected_PPL - generated_PPL) / mean(generated_PPL). KILL if > 0.5
- K3: peak memory during any phase. KILL if > 48 GB
