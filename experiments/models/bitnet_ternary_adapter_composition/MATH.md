# BitLoRA Ternary Adapter Composition: Mathematical Foundations

## Notation

| Symbol | Shape/Type | Description |
|--------|-----------|-------------|
| $W$ | $(d_{in}, d_{out})$ | Base model weight matrix (ternary: values in $\{-1, 0, 1\} \cdot \alpha$) |
| $\alpha_W$ | scalar | Per-matrix scale factor: $\alpha_W = \text{mean}(\|W\|)$ |
| $A_i$ | $(d_{in}, r)$ | LoRA down-projection for expert $i$ |
| $B_i$ | $(r, d_{out})$ | LoRA up-projection for expert $i$ |
| $\Delta_i$ | $(d_{in}, d_{out})$ | Expert delta: $\Delta_i = A_i B_i$ |
| $r$ | scalar | LoRA rank |
| $N$ | scalar | Number of composed experts |
| $d$ | scalar | Model hidden dimension |
| $\mathcal{Q}_T(\cdot)$ | function | Ternary quantizer: $\text{RoundClip}(W / \text{mean}(\|W\|), -1, 1) \cdot \text{mean}(\|W\|)$ |
| $\mathcal{Q}_4(\cdot)$ | function | INT4 quantizer: uniform 4-bit with per-tensor scale |

## 1. Base Model: Ternary Weights (BitNet b1.58)

The base model weight matrix $W$ is ternary:

$$W = \alpha_W \cdot \hat{W}, \quad \hat{W}_{ij} \in \{-1, 0, 1\}$$

where $\alpha_W = \frac{1}{n \cdot m} \sum_{i,j} |W_{ij}|$ (absmean scaling).

**Key property**: $\|W\|_F^2 = \alpha_W^2 \cdot \|\hat{W}\|_0$ where $\|\hat{W}\|_0$ counts nonzero entries. The weight norm is fully determined by sparsity and a single scalar.

## 2. Three Adapter Quantization Conditions

### Condition (a): FP16 LoRA (baseline)

Standard LoRA. Effective weight after composition:

$$W_{\text{eff}} = W + \frac{1}{N} \sum_{i=1}^N A_i B_i$$

Adapters $A_i, B_i$ are full-precision (FP32 in micro, FP16 at scale).

### Condition (b): Ternary LoRA via QAT

During **forward pass**, adapter matrices are quantized to ternary:

$$\tilde{A}_i = \mathcal{Q}_T(A_i) = \alpha_{A_i} \cdot \text{RoundClip}\left(\frac{A_i}{\alpha_{A_i}}, -1, 1\right)$$

$$\tilde{B}_i = \mathcal{Q}_T(B_i) = \alpha_{B_i} \cdot \text{RoundClip}\left(\frac{B_i}{\alpha_{B_i}}, -1, 1\right)$$

During **backward pass**, gradients pass through quantization via the Straight-Through Estimator (STE):

$$\frac{\partial \mathcal{L}}{\partial A_i} \approx \frac{\partial \mathcal{L}}{\partial \tilde{A}_i}$$

Latent weights $A_i, B_i$ are maintained in FP32 and updated by the optimizer. Only the forward computation uses quantized values.

**Effective delta** (ternary):

$$\tilde{\Delta}_i = \tilde{A}_i \tilde{B}_i = \alpha_{A_i} \alpha_{B_i} \cdot \hat{A}_i \hat{B}_i$$

where $\hat{A}_i, \hat{B}_i$ are ternary $\{-1, 0, 1\}$ matrices. The product $\hat{A}_i \hat{B}_i$ has integer entries in $[-r, r]$.

### Condition (c): INT4 LoRA

Same QAT framework but with 4-bit uniform quantization:

$$\tilde{A}_i^{(4)} = s \cdot \text{clamp}\left(\text{round}\left(\frac{A_i}{s}\right), -8, 7\right)$$

where $s = \max(|A_i|) / 7$ (symmetric quantization, 16 levels).

## 3. Why Ternary Adapters Might Compose Better

### 3.1 Magnitude Boundedness

For FP16 adapters, the delta norm $\|\Delta_i\|_F$ is continuous and unbounded across experts. Different domains may produce deltas with wildly different magnitudes.

For ternary adapters, the delta is:

$$\tilde{\Delta}_i = \alpha_{A_i} \alpha_{B_i} \cdot M_i, \quad M_i \in \mathbb{Z}^{d_{in} \times d_{out}}, \quad |M_{ij}| \leq r$$

The integer matrix $M_i$ is bounded by rank $r$. Cross-expert magnitude variation comes only from the scalar products $\alpha_{A_i} \alpha_{B_i}$, which are more uniform because:
- Ternary quantization clusters weight values around $\{-\alpha, 0, +\alpha\}$
- The absmean $\alpha$ is an average over all entries, stabilizing across domains

**Prediction**: Coefficient of variation of $\|\tilde{\Delta}_i\|_F$ across experts should be lower for ternary than FP16.

### 3.2 Reduced Interference via Discretization

The pairwise cosine similarity between two deltas:

$$\cos(\Delta_i, \Delta_j) = \frac{\langle\text{vec}(\Delta_i), \text{vec}(\Delta_j)\rangle}{\|\Delta_i\|_F \|\Delta_j\|_F}$$

For ternary deltas, the inner product $\langle\text{vec}(\tilde{\Delta}_i), \text{vec}(\tilde{\Delta}_j)\rangle$ is an integer (scaled by $\alpha$ products). Discrete inner products concentrate more tightly around zero for random ternary vectors than continuous ones.

**Johnson-Lindenstrauss analogy**: Random ternary projections preserve distances nearly as well as Gaussian projections, but with sparser representations. The discretization acts as implicit regularization.

### 3.3 Composition PPL Bound

For equal-weight composition on ternary base:

$$W_{\text{eff}} = \alpha_W \hat{W} + \frac{1}{N}\sum_i \alpha_{A_i}\alpha_{B_i} \hat{A}_i\hat{B}_i$$

The perturbation relative to base:

$$\frac{\|\Delta_{\text{composed}}\|_F}{\|W\|_F} = \frac{\frac{1}{N}\|\sum_i \alpha_{A_i}\alpha_{B_i} M_i\|_F}{\alpha_W\sqrt{\|\hat{W}\|_0}}$$

For ternary adapters, the numerator's growth with $N$ is bounded because integer matrices concentrate.

## 4. Computational Cost

| Condition | Training FLOPs per step | Adapter memory (inference) | Composition cost |
|-----------|------------------------|---------------------------|------------------|
| FP16 LoRA | $O(B \cdot T \cdot d \cdot r)$ | $2(d_{in} \cdot r + r \cdot d_{out}) \cdot 4$ bytes | Same |
| Ternary LoRA | $O(B \cdot T \cdot d \cdot r) + O(d \cdot r)$ quantize | $2(d_{in} \cdot r + r \cdot d_{out}) \cdot 0.2$ bytes* | Same |
| INT4 LoRA | $O(B \cdot T \cdot d \cdot r) + O(d \cdot r)$ quantize | $2(d_{in} \cdot r + r \cdot d_{out}) \cdot 0.5$ bytes | Same |

*Ternary: ~1.58 bits/param, practically ~2 bits with scale factors.

**Memory savings at scale** (per adapter, rank-16, d=4096):
- FP16: $2 \times 4096 \times 16 \times 2$ bytes $\times$ (28 layers $\times$ 7 modules) = ~57 MB
- Ternary: ~7 MB (8x reduction)
- INT4: ~14 MB (4x reduction)

## 5. Worked Example (micro scale: d=64, r=4, N=5)

Base weight matrix $W \in \mathbb{R}^{64 \times 64}$, ternary:
- $\alpha_W \approx 0.02$ (from training with scale 0.02)
- $\|\hat{W}\|_0 \approx 2731$ (67% nonzero, typical for absmean)
- $\|W\|_F = 0.02 \times \sqrt{2731} \approx 1.05$

FP16 adapter delta $\Delta_i = A_i B_i$ where $A_i \in \mathbb{R}^{64 \times 4}$, $B_i \in \mathbb{R}^{4 \times 64}$:
- Typical $\|\Delta_i\|_F \approx 0.5$ after training
- Composed: $\|\frac{1}{5}\sum_i \Delta_i\|_F \approx 0.1$ to $0.3$

Ternary adapter delta $\tilde{\Delta}_i = \alpha_A \alpha_B \hat{A}_i \hat{B}_i$:
- $\alpha_A \approx 0.3$, $\alpha_B \approx 0.2$ (typical after QAT convergence)
- Integer matrix $M_i = \hat{A}_i\hat{B}_i$ has entries in $[-4, 4]$
- $\|\tilde{\Delta}_i\|_F = 0.06 \times \|M_i\|_F \approx 0.06 \times 50 \approx 3.0$

Wait -- ternary deltas could be LARGER than FP16 if $\alpha$ products are not controlled. This means magnitude bounding is NOT automatic; it depends on the learned scale factors. This is an important finding to validate empirically.

## 6. Kill Criteria (Formalized)

**K1**: $\exists$ domain $d$: $\text{PPL}_{\text{ternary}}^{(d, \text{single})} > 1.05 \times \text{PPL}_{\text{fp16}}^{(d, \text{single})}$ averaged over seeds $\Rightarrow$ KILL

**K2**: $\text{mean}_d\left[\text{PPL}_{\text{ternary}}^{(d, \text{composed})}\right] > \text{mean}_d\left[\text{PPL}_{\text{fp16}}^{(d, \text{composed})}\right]$ $\Rightarrow$ KILL

**K3**: Any ternary adapter fails to converge (final loss > 2x FP16 final loss) $\Rightarrow$ KILL

## 7. Assumptions

1. Micro-scale (d=64, r=4) directionally predicts macro-scale (d=4096, r=16) behavior
2. Post-quantized ternary base (from FP16 training) approximates natively-trained BitNet
3. STE provides sufficient gradient signal for ternary QAT convergence at this scale
4. The 5 toy domains provide sufficient diversity to test composition effects
5. Equal-weight 1/N composition is the right test (not routing-weighted)
