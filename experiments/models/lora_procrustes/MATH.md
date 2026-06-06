# LoRA Procrustes Linear Decomposition: Mathematical Foundations

## 1. Setup

We have a pretrained base model with parameters W_base. For each domain k in
{1, ..., N}, we fine-tune LoRA adapters on the MLP layers (fc1 and fc2) while
freezing all base weights.

Each LoRA adapter consists of low-rank matrices:
- A_k in R^{d_in x r} (down-projection)
- B_k in R^{r x d_out} (up-projection)

The LoRA delta is:
  dW_k = (alpha / r) * A_k @ B_k  in R^{d_in x d_out}

where r is the rank and alpha is the scaling factor.

The effective weight for domain k is:
  W_k = W_base + dW_k

## 2. The Key Linearity Property

The LoRA delta dW_k is a **pure linear correction** to the frozen base weight.
For any input x:

  W_k @ x = W_base @ x + dW_k @ x

This is exact -- no activation function intervenes between W_base and dW_k.

**Contrast with CapsuleGroups (exp3_procrustes_decomp):**
In capsule groups, each group applies: output = B @ ReLU(A @ x).
Decomposing into shared + unique fails because:
  B_shared @ ReLU(A_shared @ x) + B_unique @ ReLU(A_unique @ x)
  != B_combined @ ReLU(A_combined @ x)

ReLU breaks the decomposition. LoRA deltas have no such problem.

## 3. Shared/Unique Decomposition

Given N domain-specific LoRA deltas {dW_1, ..., dW_N}, decompose each into
shared (common across all domains) and unique (domain-specific) components.

**Shared delta** (mean of all domain deltas):
  dW_shared = (1/N) * sum_k dW_k

**Unique delta** for domain k:
  dW_unique_k = dW_k - dW_shared

**Exact reconstruction:**
  dW_shared + dW_unique_k = dW_k  (exactly, for all k)

**Linearity guarantee:**
  For any input x and any domain k:
  (W_base + dW_shared + dW_unique_k) @ x = (W_base + dW_k) @ x

This holds exactly because matrix addition and multiplication are linear.

## 4. Orthogonality of Unique Components

When N = 2 and the decomposition is the mean:
  dW_unique_A = (dW_A - dW_B) / 2
  dW_unique_B = (dW_B - dW_A) / 2 = -dW_unique_A

The unique components are exact negatives of each other.

For general N, the unique components sum to zero:
  sum_k dW_unique_k = sum_k (dW_k - dW_shared) = N * dW_shared - N * dW_shared = 0

## 5. Norm Decomposition

The Frobenius norm of the delta decomposes as:
  ||dW_k||^2 = ||dW_shared||^2 + ||dW_unique_k||^2 + 2 * <dW_shared, dW_unique_k>

The cross term <dW_shared, dW_unique_k> is generally nonzero.

Across all N domains:
  sum_k ||dW_k||^2 = N * ||dW_shared||^2 + sum_k ||dW_unique_k||^2

because sum_k <dW_shared, dW_unique_k> = <dW_shared, sum_k dW_unique_k> = 0.

The **shared fraction** is defined as:
  f_shared = ||dW_shared||_total / (||dW_shared||_total + ||dW_unique||_total)

where the total norms aggregate across all layers and weight matrices.

## 6. Composition Architecture

**Concatenated LoRA (baseline):**
At inference, a router selects top-k of N expert deltas per token:
  MLP_output = ReLU((W_fc1_base + sum_k w_k * dW_fc1_k) @ x) @ (W_fc2_base + sum_k w_k * dW_fc2_k)

where w_k are routing weights (sum to 1 for selected experts, 0 for others).

**Decomposed LoRA:**
Shared deltas are baked into base weights (always active).
Unique deltas are routed per token:
  W_fc1_eff = (W_fc1_base + dW_fc1_shared) + sum_k w_k * dW_fc1_unique_k
  W_fc2_eff = (W_fc2_base + dW_fc2_shared) + sum_k w_k * dW_fc2_unique_k

**Important subtlety:** The linearity guarantee holds for the weight matrices,
but the full MLP has a ReLU between fc1 and fc2:
  MLP(x) = fc2(ReLU(fc1(x)))

This means: routing between full expert MLPs (base+delta_k) is NOT the same as
applying routed deltas separately to fc1 and fc2. Both the concatenated and
decomposed approaches correctly route by running each expert's full MLP
(with its modified weights) and combining outputs.

## 7. Why Decomposition is Equivalent at N=2

With N=2 domains, the decomposed model with routing is functionally equivalent
to the concatenated model with routing. This is because:

1. The shared component = task arithmetic (mean of deltas)
2. The unique components = +/- half the difference
3. For any routing weights (w_A, w_B) summing to 1:
   shared + w_A * unique_A + w_B * unique_B
   = (dW_A + dW_B)/2 + w_A*(dW_A - dW_B)/2 + w_B*(dW_B - dW_A)/2
   = (dW_A + dW_B)/2 + (w_A - w_B)*(dW_A - dW_B)/2
   = w_A * dW_A + w_B * dW_B

The decomposition adds no information at N=2 -- it's an algebraic identity.

**At N>2, decomposition becomes non-trivial:** the shared component captures
genuine common structure, and the unique components are not simply negatives
of each other. The decomposition then provides:
- A smaller routing space (N unique deltas vs N full deltas)
- Shared knowledge is always active regardless of routing

## 8. Worked Example at d=64, r=8

Given: n_embd=64, lora_rank=8, n_layer=4
- A_k in R^{64 x 8}, B_k in R^{8 x 256} (for fc1: d -> 4d)
- A_k in R^{256 x 8}, B_k in R^{8 x 64} (for fc2: 4d -> d)

LoRA params per domain per layer:
  fc1: 64*8 + 8*256 = 2560
  fc2: 256*8 + 8*64 = 2560
  Total per layer: 5120
  Total: 4 * 5120 = 20,480

Full delta matrices per layer:
  fc1: dW in R^{64 x 256} = 16,384 elements
  fc2: dW in R^{256 x 64} = 16,384 elements
  Total per layer: 32,768
  Total across layers: 4 * 32,768 = 131,072

But each delta is rank-8, so only 20,480 degrees of freedom.

Router params per layer: 64 * N (N = number of experts)

## 9. Assumptions

1. LoRA deltas are small relative to base weights (||dW|| << ||W_base||).
2. Domains are trained from the same frozen base (alignment is free).
3. The shared component is substantial (>10% of total delta norm).
4. The router can learn to select appropriate unique experts.
5. The full MLP forward (with ReLU) is computed per-expert and combined.

## 10. Falsification Criteria

| Criterion | Threshold | Kill if |
|-----------|-----------|---------|
| Decomposed composition vs concatenated LoRA | <3% gap | >3% |
| Shared fraction of delta norm | >10% | <10% |

## 11. Computational Cost

Training: Identical to standard LoRA fine-tuning (per domain).
Decomposition: One mean computation per matrix -- O(N * d_in * d_out) additions.
Composition model: N expert MLPs per layer (same as concatenated).
Router calibration: 100 steps, only router weights (64*N per layer).

Total experiment time: ~3 minutes per seed.
