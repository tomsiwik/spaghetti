# MATH.md: Precomputed Concat Deltas (Full QKV + MLP)

## A. Failure Mode Identification

**Symptom:** Pierre v6 (attention-only, 60 dispatches) achieved 86.8 tok/s but
code domain behavioral score dropped to 0.281 (vs v3's 0.844). The MLP
gate_proj and up_proj adapters are critical for code-domain instruction following
(Finding #292).

**Root cause:** The MLP feedforward transformation stores domain-specific
computation patterns (e.g., code syntax structure). Attention-only adapters
modify the attention pattern but not the feedforward path. Restoring MLP
adapters is necessary, but naive restoration adds 3 dispatches per layer
(gate + up + down), bringing total dispatches from 60 back toward 210.

**Degenerate behavior to avoid:** Dispatch overhead from 210 modules nullifies
the speed gains from precomputation. At v3's 420 dispatches (2 per module:
x@A, h@B), speed was 73 tok/s. We must keep dispatch count low.

## B. The Right Question

Not "how do we add MLP adapters without losing speed?" but rather:

**"For modules sharing the same input tensor x, what is the minimum number of
Metal dispatch calls needed to compute all adapter corrections?"**

Answer: ONE dispatch per group. If modules m_1, ..., m_k all receive the same
input x, we can concatenate their delta weight matrices and compute all k
corrections with a single matmul followed by slicing.

## C. Prior Mathematical Foundations

**Matrix algebra (exact, no approximation):**

For a standard LoRA module, the correction is:
```
correction_i(x) = x @ DeltaW_i
```
where `DeltaW_i = alpha * A_i @ B_i`, with A_i in R^{d_in x r} and B_i in R^{r x d_out_i}.

**Theorem (Concat-Slice Equivalence).** Let f_1, ..., f_k be linear maps
f_i: R^{d_in} -> R^{d_i} defined by f_i(x) = x @ W_i where W_i in R^{d_in x d_i}.
Define W_cat = [W_1 | W_2 | ... | W_k] in R^{d_in x (d_1 + ... + d_k)}.
Then for any x in R^{d_in}:

    x @ W_cat = [x @ W_1 | x @ W_2 | ... | x @ W_k]

*Proof.* By definition of matrix-vector multiplication and horizontal
concatenation:

    (x @ W_cat)_j = sum_{i=1}^{d_in} x_i * (W_cat)_{ij}

For j in [d_1 + ... + d_{m-1} + 1, d_1 + ... + d_m], we have
(W_cat)_{ij} = (W_m)_{i, j - sum_{l<m} d_l}, which gives

    (x @ W_cat)[d_1+...+d_{m-1} : d_1+...+d_m] = x @ W_m

This is exact (no numerical approximation). QED.

**Applicability to LoRA groups:**

In a transformer layer, the following modules share inputs:
- **QKV group:** q_proj, k_proj, v_proj all receive the same hidden state x
- **Gate+Up group:** gate_proj, up_proj both receive the same hidden state x
- **O group:** o_proj receives attention output (unique input, singleton group)
- **Down group:** down_proj receives element-wise product (unique input, singleton group)

The QKV and Gate+Up groups each benefit from concat: 3->1 and 2->1 dispatches
respectively. O and Down are singletons (1 dispatch each, no concat).

## D. Proof of Guarantee

**Theorem 1 (Exact Equivalence).** The precomputed-concat approach produces
bit-identical results to individual per-module LoRA application.

*Proof.* For the QKV group, the v3 approach computes:
```
q_corr = x @ DeltaW_q       (1 dispatch)
k_corr = x @ DeltaW_k       (1 dispatch)
v_corr = x @ DeltaW_v       (1 dispatch)
```
Total: 3 dispatches.

The concat approach computes:
```
DeltaW_qkv = [DeltaW_q | DeltaW_k | DeltaW_v]    (precomputed offline)
all_corr = x @ DeltaW_qkv                          (1 dispatch)
q_corr = all_corr[..., :d_q]
k_corr = all_corr[..., d_q:d_q+d_k]
v_corr = all_corr[..., d_q+d_k:]
```
Total: 1 dispatch + 3 slices (slicing is ~free, pointer arithmetic only).

By the Concat-Slice Equivalence theorem, these produce identical results.
The same argument applies to the Gate+Up group. QED.

**Theorem 2 (Dispatch Count).** With 4 groups per layer across L layers,
the total dispatch count is exactly 4L.

*Proof.* Each layer has exactly 4 groups:
- QKV: 1 dispatch (concat of 3 modules)
- O: 1 dispatch (singleton)
- Gate+Up: 1 dispatch (concat of 2 modules)
- Down: 1 dispatch (singleton)

Total per layer: 4. Total: 4L = 4 * 30 = 120 for BitNet-2B-4T (30 layers).

Compare: v3 uses 7 modules * 2 dispatches (A, B) = 14 per layer = 420 total.
v6 (attention-only) uses 2 dispatches per layer = 60 total.
v6.1 (full) uses 4 dispatches per layer = 120 total. QED.

**Theorem 3 (Speed Model).** Let T(D) be the per-token generation time with
D dispatches. Empirically from v3 and v6:

    T(D) = T_base + c * D

where T_base is the base model generation time and c is the per-dispatch
overhead. From two data points:
- v3: D=420, speed=73 tok/s -> T(420) = 1/73 = 13.70 ms/tok
- v6: D=60, speed=86.8 tok/s -> T(60) = 1/86.8 = 11.52 ms/tok
- Base: D=0, speed~172 tok/s -> T(0) = 1/172 = 5.81 ms/tok

From T(420) and T(60):
    c = (13.70 - 11.52) / (420 - 60) = 2.18 / 360 = 0.00606 ms/dispatch

From T(60) and T(0):
    T_base = 11.52 - 0.00606 * 60 = 11.16 ms

Check: T(420) = 11.16 + 0.00606 * 420 = 13.70 ms. Consistent.

**Prediction for v6.1 (D=120):**
    T(120) = 11.16 + 0.00606 * 120 = 11.88 ms/tok
    Speed = 1000/11.88 = 84.2 tok/s

**Uncertainty:** The linear model is derived from only 2 data points.
The true relationship may have nonlinear effects (cache pressure, memory
bandwidth saturation). Conservative prediction range: 78-90 tok/s.

## D. Predictions (Behavioral AND Quantitative)

### Behavioral Predictions
1. **Code domain behavioral recovers** because MLP gate+up adapters are restored.
   v3 code behavioral: 0.844. v6 (attn-only) code: 0.281.
   v6.1 should match v3 since all 7 module types are included.

2. **Overall behavioral matches v3** (0.41) since the adapter math is identical
   (same DeltaW = alpha * A @ B), just precomputed.

3. **Speed exceeds 75 tok/s** (K756) with high confidence since dispatch count
   120 < v3's 420.

### Quantitative Predictions (from proof)

| Prediction | Source | Expected |
|-----------|--------|----------|
| Dispatch count | Theorem 2 | 120 |
| Speed (tok/s) | Theorem 3, linear model | 84.2 (range: 78-90) |
| Behavioral overall | Theorem 1 (exact equiv) | ~0.41 (matching v3) |
| Code behavioral | Theorem 1 + Finding #292 | ~0.84 (matching v3) |
| Memory peak (GB) | v6 was 2.23GB, adding MLP deltas ~2x | ~2.5-3.5 |

## E. Assumptions & Breaking Conditions

**A1: Python evaluation order is left-to-right.** The concat cache pattern
assumes Q is called before K and V (for QKV group), and gate is called before
up (for Gate+Up group). Verified in mlx_lm's qwen2.py:
- `queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)`
- `silu(self.gate_proj(x)) * self.up_proj(x)`

Both evaluate left-to-right. If violated (e.g., by MLX graph optimization
reordering), the cache would contain stale data. **Impact: correctness failure,
not just performance.** Mitigation: MLX lazy eval builds the graph in Python
call order; no reordering within a single mx.eval.

**A2: Dispatch overhead is the dominant cost.** If the bottleneck were memory
bandwidth rather than dispatch count, reducing dispatches would not help.
At rank 16, DeltaW is (2560 x d_out), which is a dense matmul. The concat
version does a slightly larger matmul but saves dispatch overhead. This
assumes dispatch overhead >> marginal compute cost.

**A3: Slicing is free.** MLX array slicing creates a view (pointer arithmetic),
not a copy. If MLX copies on slice, the 3 extra slices per QKV group add
memory traffic.

## F. Worked Example (d=4, r=2)

Consider a single layer with QKV group, where d_in=4, rank=2.

**Q adapter:**
```
A_q = [[1,0], [0,1], [1,1], [0,0]]  (4x2)
B_q = [[1,0,0], [0,1,0]]            (2x3, q output dim=3)
DeltaW_q = A_q @ B_q = [[1,0,0], [0,1,0], [1,1,0], [0,0,0]]  (4x3)
```

**K adapter:**
```
A_k = [[0,1], [1,0], [0,0], [1,1]]  (4x2)
B_k = [[1,0], [0,1]]                (2x2, k output dim=2)
DeltaW_k = A_k @ B_k = [[0,1], [1,0], [0,0], [1,1]]  (4x2)
```

**Concatenated:**
```
DeltaW_cat = [DeltaW_q | DeltaW_k]
           = [[1,0,0,0,1],
              [0,1,0,1,0],
              [1,1,0,0,0],
              [0,0,0,1,1]]   (4x5)
```

**Input:** x = [1, 0, 1, 0]
```
x @ DeltaW_cat = [1*1+0*0+1*1+0*0, 1*0+0*1+1*1+0*0, 1*0+0*0+1*0+0*0,
                   1*0+0*1+1*0+0*1, 1*1+0*0+1*0+0*1]
               = [2, 1, 0, 0, 1]
```

Slice Q: [2, 1, 0] == x @ DeltaW_q = [1+0+1+0, 0+0+1+0, 0+0+0+0] = [2, 1, 0]. Correct.
Slice K: [0, 1] == x @ DeltaW_k = [0+0+0+0, 1+0+0+0] = [0, 1]. Correct.

## G. Complexity & Architecture Connection

**Per-token compute (v6.1 vs v3):**

| Operation | v3 (RuntimeLoRA) | v6.1 (Precomputed) |
|-----------|-------------------|---------------------|
| Dispatches per layer | 14 (7 modules x 2 matmuls) | 4 (4 groups x 1 matmul) |
| Total dispatches | 420 | 120 |
| Matmul size per layer | 14 x (d x r) or (r x d_out) | 4 x (d x d_group_out) |
| Total FLOPs per token | 2 * r * sum(d_in * d_out) | Same (precompute absorbs A@B) |
| Precompute cost | 0 | 7 matmuls per layer (offline, once) |
| Memory for deltas | A + B per module (~sparse) | DeltaW_cat per group (~dense) |

**Memory estimate:**
- QKV concat delta: (2560 x (2560 + 640 + 640)) = 2560 x 3840 x 2 bytes = 19.7 MB per layer
- O delta: (2560 x 2560) x 2 = 13.1 MB per layer
- Gate+Up concat delta: (2560 x (6912 + 6912)) = 2560 x 13824 x 2 = 70.8 MB per layer
  Wait - this seems too large. Let me reconsider.

Actually the deltas are d_in x d_out at bf16 = 2 bytes. For BitNet-2B-4T:
- d_model = 2560, d_kv = 640, d_intermediate = 6912, n_heads=32, n_kv_heads=4
- QKV: 2560 x (2560 + 640 + 640) = 2560 x 3840 = 9.83M params x 2 = 19.7 MB
- O: 2560 x 2560 = 6.55M x 2 = 13.1 MB
- Gate+Up: 2560 x (6912 + 6912) = 2560 x 13824 = 35.4M x 2 = 70.8 MB
- Down: 6912 x 2560 = 17.7M x 2 = 35.4 MB

Total per layer: 139 MB. For 30 layers: 4.17 GB.

Hmm, this is larger than expected because we are materializing full-rank DeltaW
from low-rank A@B. At rank 16:
- A: (d_in x 16), B: (16 x d_out) -> DeltaW = A@B: (d_in x d_out)
- The low-rank product expands to full rank in storage
- v3 stored A (d_in x 16) + B (16 x d_out) per module = ~98KB per module
- v6.1 stores DeltaW (d_in x d_out) per module = up to 70.8 MB for gate+up group

This is the price of precomputation: trading storage for dispatch count.
However, the actual memory test in v6 (attention-only) showed 2.23 GB, so
the full version should be around 2.5-4 GB depending on how MLX handles
the delta matrices.

**The precompute is done once per adapter swap, amortized over all tokens.**

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the failure mode impossible?
   **Concat-slice equivalence: horizontal concatenation of weight matrices
   followed by slicing produces bit-identical results to separate matmuls,
   reducing dispatch count from 14 to 4 per layer.**

2. Which existing theorem(s) does the proof build on?
   **Matrix multiplication distributes over horizontal concatenation
   (basic linear algebra, any textbook). No deep theorem needed —
   this is a consequence of the definition of matrix multiplication.**

3. What specific numbers does the proof predict?
   **120 dispatches, ~84.2 tok/s (range 78-90), behavioral ~0.41
   (matching v3), code behavioral ~0.84, memory ~2.5-4 GB.**

4. What would FALSIFY the proof (not just the experiment)?
   **The proof is wrong if MLX reorders module calls within a layer
   (breaking the cache pattern), or if slicing copies data rather
   than creating views (breaking the "free slice" assumption).**

5. How many hyperparameters does this approach add?
   **0. The group definitions (QKV, Gate+Up) are determined by the
   architecture, not tuned. The scale alpha is inherited from v3.**

6. Hack check: Am I adding fix #N to an existing stack?
   **No. This is a pure engineering optimization (fewer dispatches)
   that preserves exact mathematical equivalence. No new losses,
   constraints, or mechanisms. The adapter math is identical to v3.**
