# LoRA Soups CAT: Mathematical Foundations

## 0. Failure Mode & Impossibility Structure

**Failure mode:** Uniform 1/N composition dilutes each adapter's contribution. At N=5 with 1/N scaling, each adapter contributes only 20% of its delta. Domains where the adapter provides large improvement are diluted by irrelevant adapters.

**What makes optimal weighting impossible to find without data?** The optimal per-layer weighting depends on how each adapter's contribution interacts with others at each layer. Without calibration data, we cannot distinguish layers where adapter i contributes meaningfully from layers where it introduces noise. The composition landscape is smooth and convex (proven by exp_composition_interpolation_landscape), but the optimal point varies per evaluation distribution.

**What CAT makes impossible:** CAT makes sub-optimal layer-wise weighting structurally impossible by gradient descent on calibration data. Given:
- M modules (420 per-tensor entries), N adapters, M*N learnable scalars alpha_i^m
- Calibration loss L_cal(alpha) = E_{x~D_cal}[-log p(x | W + sum_i alpha_i^l * Delta_W_i^l)]
- L_cal is differentiable w.r.t. alpha (linear combination, chain rule through model)
- The landscape is convex in alpha at each layer (proven: composition_interpolation_landscape showed smooth monotonic curves for 2-adapter pairs, single basin for 3-adapter simplex)

**Impossibility proof sketch:** If alpha* = argmin L_cal(alpha), then for any uniform weighting alpha_u = 1/N for all i,l: L_cal(alpha*) <= L_cal(alpha_u). Equality holds iff the uniform weighting is already optimal. The composition_interpolation_landscape experiment showed uniform is 0.7% from optimal for 2 adapters; for 5 adapters across diverse layers, the gap could be larger.

**What KILLS this:** If the landscape is so flat that gradient descent on alpha finds no signal (all alpha converge to ~1/N), then CAT = uniform = no improvement. This is K1's mechanism. From flat_lora_training: sharpness <0.3%, suggesting the landscape may be too flat for per-layer differentiation.

## 1. Mechanism Definition

### LoRA Soups CAT (Composition via Adaptive Training)

Given:
- Base model W = {W^l} for l = 1,...,L layers (L=24 for BitNet-2B-4T)
- N = 5 domain adapters, each consisting of Delta_W_i = {B_i^l @ A_i^l} for each adapted layer
- Each layer has 7 LoRA targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Each LoRA target has 2 weight matrices: lora_a (R^{d_in x r}) and lora_b (R^{r x d_out})
- Total adapted modules (individual weight tensors): M = L * 7 * 2 = 336 (but see implementation note)

**Implementation note (Fix 2):** In the actual code, adapter parameters are stored per-tensor
(each lora_a and lora_b is a separate entry in the adapter dict). The code discovers M from
`sorted(adapter_list[0].keys())`, which yields M=420 entries (some modules have additional
parameters beyond the 7*2=14 per layer, including bias terms and scale factors). The true
parameter count is **N * M = 5 * 420 = 2100 scalars**, not 840 as originally stated.

**Per-module learnable coefficients:**
```
alpha = {alpha_i^m} for i=1,...,N, m=1,...,M
alpha_i^m in R (scalar, initialized to 1/N)
```

**Composed weight at module m:**
```
W_composed^m = W_base^m + sum_{i=1}^{N} alpha_i^m * (B_i^m @ A_i^m)
```

Note: when M counts individual weight tensors (lora_a, lora_b separately),
the composition scales each tensor independently rather than per-LoRA-module.
This is slightly finer-grained than the paper's per-layer formulation.

**Shapes:**
- A_i^m in R^{d_in x r}, B_i^m in R^{r x d_out} (rank r=16)
- alpha_i^m in R (scalar)
- Total learnable parameters: N * M = 5 * 420 = 2100 scalars

**Training objective:**
```
min_{alpha} E_{(x,y) ~ D_cal} [ CrossEntropy(f(x; W + sum_i alpha_i * Delta_W_i), y) ]
```

Where D_cal is 5% of training data from each domain (calibration set).

### Merge methods for comparison

**Uniform merge (baseline):**
alpha_i^m = 1/N = 0.2 for all i, m

**Task Arithmetic:**
alpha_i^m = lambda for all i, m (single global scalar, tuned)

**TIES-Merging (Yadav et al., 2023):**
1. Trim: zero out small-magnitude parameters (top-k% kept)
2. Elect sign: majority vote per parameter position
3. Disjoint merge: sum only parameters with elected sign

**DARE (Yu et al., 2023):**
1. Random drop: independently mask each parameter with probability p
2. Rescale: multiply surviving parameters by 1/(1-p)
3. Merge: sum rescaled deltas

**DO-Merging (direct orthogonal merge):**
Scale adapters to minimize pairwise interference using cosine similarity.
For our near-orthogonal adapters (|cos|=0.001), this degenerates to uniform.

## 2. Why It Works

CAT exploits the observation that different layers contribute unevenly to each domain's task. From the MoLoRA literature (arxiv 2603.15965), per-token routing selects different experts at different positions. CAT is the static analog: learn which layers benefit most from which adapter.

**Key mathematical property:** The loss L_cal(alpha) is differentiable w.r.t. alpha because:
1. alpha enters linearly into weight computation
2. Weight -> logits is differentiable (standard backprop)
3. CrossEntropy loss is differentiable

The gradient:
```
dL/d(alpha_i^m) = dL/dW^m * (B_i^m @ A_i^m)
```

This is the inner product of the model gradient at module m with adapter i's delta. If adapter i's delta aligns with the negative gradient direction at module m, alpha_i^m should increase; if orthogonal, it should stay near 0.

**LoRA Soups paper result (arXiv 2410.13025):** On Llama-2-7B with 2 task LoRAs, CAT achieves 43% improvement over uniform merge and 257% superlinear composition. The mechanism: different layers have different optimal adapter weights; learning these per-layer scalars captures task-specific layer importance.

## 3. What Breaks It

**Scenario 1: Landscape too flat (K1 kill mechanism)**
If |cos(adapter_i, adapter_j)| << 1 and the loss landscape is convex around 1/N, then gradient descent on alpha converges to ~1/N with negligible deviation. From flat_lora_training: sharpness <0.3%, cos=0.001. The gradient signal dL/d(alpha_i^m) may be too weak to distinguish from noise.

**Scenario 2: Calibration data insufficient**
With 5% calibration data (~50 samples per domain), the gradient estimate has high variance. At 2100 parameters and ~250 calibration sequences, we have ~250/2100 ~ 0.12 sequences per parameter -- severely underdetermined. However, alpha is very low-dimensional and starts at a good point (1/N).

**Scenario 3: Overfitting calibration set**
2100 parameters on ~250 samples. This could overfit. Regularization toward 1/N helps: L_reg = lambda * ||alpha - 1/N||^2.

## 4. Assumptions

1. **Adapter quality varies by layer and domain** -- justified by exp_depth_routed_adapters showing depth weights stay near-uniform (contradicts this assumption at L=4). However, at L=24 with 7 modules per layer = 168 modules, more variation expected.

2. **5% calibration data is sufficient** -- from LoRA Soups paper, they use 5% with success. However, their model (Llama-2-7B) has stronger signal than BitNet-2B.

3. **Static weighting captures the important variation** -- justified by composition_interpolation_landscape showing smooth landscapes. Per-input variation (routing) adds only marginal benefit per MoLoRA experiment.

4. **Adapters are already trained and frozen** -- from flat_lora_training, we have 5 standard adapters with known quality.

## 5. Complexity Analysis

**Training CAT coefficients:**
- Parameters: 2100 scalars (5 adapters x 420 per-tensor modules)
- Forward pass: standard model forward (dominated by base model, not alpha)
- Backward pass: only through alpha (adapter weights frozen)
- Training: LR sweep over {1e-4, 1e-3, 1e-2, 1e-1} x 200 steps each ~ 4 runs total
- Optimization: In practice, we only need gradients w.r.t. alpha, not adapter weights

**Inference (post-merge):**
- Pre-compute: W_composed^m = W_base^m + sum_i alpha_i^m * B_i^m @ A_i^m
- Cost: identical to base model (merged weights, zero overhead)
- This is the key advantage: CAT is a training-time technique with zero inference cost

## 6. Worked Example (d=64, r=4, N=2, L=2)

Base weight W^1 in R^{64x64}
Adapter 1: A_1^1 in R^{64x4}, B_1^1 in R^{4x64}, Delta_1^1 = B_1^1 @ A_1^1
Adapter 2: A_2^1 in R^{64x4}, B_2^1 in R^{4x64}, Delta_2^1 = B_2^1 @ A_2^1

Initialize: alpha_1^1 = alpha_2^1 = 0.5

Composed: W^1 = W_base^1 + 0.5 * Delta_1^1 + 0.5 * Delta_2^1

After gradient descent on calibration data mixing both domains:
alpha_1^1 = 0.7, alpha_2^1 = 0.3 (layer 1 favors adapter 1)
alpha_1^2 = 0.4, alpha_2^2 = 0.6 (layer 2 favors adapter 2)

Merged: W_final^1 = W_base^1 + 0.7 * Delta_1^1 + 0.3 * Delta_2^1
        W_final^2 = W_base^2 + 0.4 * Delta_1^2 + 0.6 * Delta_2^2

Superlinear test: If PPL(W_final) < min(PPL(adapter_1_only), PPL(adapter_2_only)) on some domain, that's superlinear.

## 7. Connection to Architecture

CAT fits naturally into our pre-merge serving pipeline:
1. Train adapters independently (existing pipeline)
2. Learn alpha on calibration data (new: one-time optimization)
3. Pre-merge with learned weights (existing: pre-merge serving at 0% overhead)
4. Serve merged model (existing: base model speed)

**Note on near-orthogonality (Fix 5 from review):** The adapters used in this experiment
are standard LoRA from flat_lora_training, NOT Grassmannian-initialized. The observed
near-orthogonality (|cos|=0.001) is a consequence of high-dimensional concentration of
measure: for random vectors in R^d with d ~ 17.2M parameters, E[|cos|] ~ 1/sqrt(d) ~ 0.0002.
This is the Johnson-Lindenstrauss effect, not a property of Grassmannian construction.

**Implication:** The conclusions of this experiment (smooth landscape, modest CAT gains,
TIES superiority) apply to ANY set of independently-trained high-dimensional adapters,
not just Grassmannian architectures. Near-orthogonality is the generic case for
adapters with millions of parameters.

If Grassmannian-initialized adapters were used, we would expect:
- Even smoother alpha landscape (guaranteed orthogonality rather than statistical)
- Potentially less benefit from CAT (if orthogonality is exact, weighting matters even less)

This is the fundamental tension: orthogonality enables composition but may eliminate the need for learned weighting.
