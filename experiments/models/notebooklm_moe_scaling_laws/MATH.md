# MoE Scaling Laws for LoRA Composition: Mathematical Analysis

## 0. Failure Mode & Impossibility Structure

### Failure Mode: Misapplying MoE Scaling Laws to LoRA Composition

The specific degenerate behavior this research addresses: treating LoRA-based
expert composition as traditional MoE and drawing wrong conclusions about
minimum viable scale, optimal expert count, and routing strategy.

**The failure:** Our interpretation of Apple's MoE scaling laws (arxiv:2501.12370,
ICML 2025) suggests MoE becomes suboptimal below ~500M params (optimal sparsity
S* -> 0 at small N). This threshold applies to traditional MoE with full FFN
experts. Naively applying it to our architecture would lead to abandoning LoRA
composition at 2B scale, which is wrong.

**Why it's wrong (mathematical structure):**

Traditional MoE expert: E_i(x) = W_down_i * sigma(W_up_i * x), with
O(d_model * d_ffn) = O(d^2) parameters per expert.

LoRA expert: Delta_i(x) = B_i * A_i * x, with O(d * r) parameters per expert,
where r << d.

The Apple scaling law:
```
L(N, D, S) = a*N^alpha + b*D^beta + c*(1-S)^lambda + d*(1-S)^delta * N^gamma + e
```
where S = sparsity = 1 - N_active/N_total.

For traditional MoE with N_experts experts, top-k routing:
- N_total = N_shared + N_experts * d_ffn * d_model
- N_active = N_shared + k * d_ffn * d_model
- S = 1 - k/N_experts (approximately)

For LoRA-MoE with N adapters, top-k routing:
- N_total = N_base + N * 2 * r * d_model * L_adapted
- N_active = N_base + k * 2 * r * d_model * L_adapted
- Expert size ratio: (2*r*d)/(2*d_ffn*d) = r/d_ffn (or r/(1.5*d_ffn) for SwiGLU)

At our scale (d=2560, r=16, d_ffn=6912):
- LoRA expert size: 2 * 16 * 2560 = 81,920 params per layer
- FFN expert size (full FFN with up+down projection): 2 * 2560 * 6912 = 35,389,440 params/layer
  (or 3 * 2560 * 6912 = 53,084,160 with SwiGLU gating)
- **Ratio: 1:432 (up+down) to 1:648 (SwiGLU)**

This means LoRA experts are **432-648x smaller** than full FFN experts. The MoE
scaling law's "minimum viable expert size" does not apply -- LoRA experts are not
trying to learn a complete FFN transformation, they're learning a low-rank
perturbation to an existing one.

**Impossibility structure:** The failure mode (applying FFN-MoE scaling laws to
LoRA composition) is made impossible by recognizing that runtime LoRA is
mathematically output-space composition, not parameter-space MoE:

```
h = W*x + sum_{i in topK} g_i * B_i * A_i * x
```

This is identical to MoE output-space composition where each "expert" is the
low-rank function f_i(x) = B_i * A_i * x. No cross-terms exist because each
adapter acts independently on x (confirmed by LoRI, arxiv:2504.07448, and
our reference doc MULTI_ADAPTER_SELECTION_MOE_COMPOSITION.md Section 4.2).

The impossibility is structural: as long as we use runtime LoRA (addmm) rather
than pre-merge, interference between adapters is zero by construction within
each adapted layer. Cross-layer interference through nonlinearities (LayerNorm
rescaling, attention's quadratic interactions) is empirically small
(gamma=0.982 at N=25, all domains benefit) but not zero by construction.

## 1. Mechanism Definition

### 1.1 Traditional MoE Scaling (Apple, arxiv:2501.12370)

The Apple scaling law models loss as a function of total parameters N, data D,
and sparsity S:

```
L(N, D, S) = a*N^alpha + b*D^beta + c*(1-S)^lambda + d*(1-S)^delta * N^gamma + e
```

Key findings at small scale:
- Optimal sparsity S* increases with N (converges to 1 for large N)
- At small N, S* -> 0 (dense is optimal)
- Crossover: at 10^20 FLOPs, 32-expert MoE (669M active params) beats
  dense 1.7B trained on 9.7B tokens, but requires 24.9B tokens (2.5x more data)
- MoE is data-hungry: needs E*D tokens where E ~= 2-4x relative to dense
  (Joint MoE Scaling Laws, arxiv:2502.05171)

### 1.2 LoRA Composition (Runtime/Output-Space)

For our architecture, the forward pass at each adapted layer:
```
h = W*x + sum_{i in S(x)} g_i(x) * B_i * A_i * x
```
where:
- W in R^{d_out x d_in}: frozen base weight
- A_i in R^{r x d_in}: frozen Grassmannian projection (skeleton)
- B_i in R^{d_out x r}: trained ternary response matrix
- S(x) subset {1,...,N}: routing selection set (top-k by softmax router)
- g_i(x): routing gate weight for adapter i

Complexity per adapted layer:
- Base: O(d_out * d_in) -- already computed
- Per adapter: O(d_in * r + d_out * r) = O(d * r) for r << d
- k adapters: O(k * d * r)
- Overhead ratio: k*r/d_in (for k=2, r=16, d=2560: 1.25%)

### 1.3 Key Mathematical Distinction

**Traditional MoE:** Each expert learns a COMPLETE function f_i: R^d -> R^d.
The expert must have enough capacity to represent useful transformations.
At small d, the minimum expert size constrains the minimum model size.

**LoRA composition:** Each expert learns a PERTURBATION Delta_i: R^d -> R^d of
rank r. The base model provides the "dense" transformation; experts only need to
encode the *direction of change*. This is why it works at 117M params
(GPT-2 Small, Cao et al., arxiv:2508.11985) -- the perturbation can be tiny
because the base model already does most of the work.

## 2. Why LoRA Composition Works at Small Scale

### 2.1 The Superposition Principle (Cao et al., arxiv:2508.11985)

Independently trained LoRA adapters on disjoint domains are approximately
orthogonal in high-dimensional parameter space. For rank-r adapters in
d-dimensional space:

```
E[|cos(vec(Delta_i), vec(Delta_j))|] ~ sqrt(r/d)
```

At our scale (r=16, d=2560): expected cosine ~ 0.079.

With Grassmannian A-matrices (A_i^T * A_j = 0):
```
Delta_i^T * Delta_j = B_i^T * (A_i^T * A_j) * B_j = 0
```
regardless of B correlation. Empirically confirmed: cos = 0.00125 (50x below
theory, VISION.md).

**Quantitative evidence from Cao et al.:**
- GPT-2 Small (117M), rank-4 LoRA
- Math + Medicine: PPL improved -9.10% (constructive composition)
- Math + Finance: PPL degraded +4.54% (mild interference)
- Finance + Medicine: PPL degraded +27.56% (destructive interference)
- RMS cosine similarity correlates approximately linearly with PPL change

### 2.2 Why MoE Needs Larger Scale But LoRA Doesn't

The Apple scaling law shows optimal sparsity S* -> 0 at small N because:
1. Small N means small d (model dimension)
2. Small d means each FFN expert has few parameters (O(d^2))
3. Underparameterized experts can't learn useful transformations
4. Therefore: dense is better (use all params for one expert)

LoRA sidesteps this entirely:
1. The base model provides the dense transformation (pretrained)
2. Each expert only needs O(d*r) params for a rank-r perturbation
3. The perturbation doesn't need to be "useful on its own" -- it modifies
   an already-useful base function
4. Therefore: composition works even at 117M total params

**The key insight:** MoE scaling laws measure when EXPERTS become viable.
LoRA scaling laws (if they existed) would measure when PERTURBATIONS become
viable. Perturbations are always viable if the base model is good enough.

### 2.3 Empirical Confirmation from Our Project

Our BitNet-2B-4T experiments confirm LoRA composition works well below the
MoE threshold:
- 2B ternary base (effectively ~400M active-equivalent in FP16 compute)
- 5 domain adapters compose with PPL 7.96 vs 8.69 base (+8.4%)
- 24 real-data adapters: all-24 uniform gives -29.1% vs base
- N=25 scaling: gamma = 0.982 (near-linear, no degradation)
- Orthogonality: |cos| = 0.0238 at N=24 (6.7x below N_max=160)

## 3. What Breaks It

### 3.1 Cross-Term Interference (Parameter-Space Merging Only)

If adapters are MERGED into weights (pre-merge):
```
W_composed = W + sum_i alpha_i * B_i * A_i
y = W_composed * x = W*x + (sum_i alpha_i * B_i * A_i) * x
```

For N adapters, this creates N*(N-1)/2 cross-terms when expanded. With
orthogonal A (Grassmannian), cross-terms vanish:
```
(B_i * A_i)^T * (B_j * A_j) = A_i^T * B_i^T * B_j * A_j
```
If A_i^T * A_j = 0, this is NOT guaranteed zero (only A_i^T * ... * A_j
path vanishes). The cross-term has structure B_i^T * B_j which can be nonzero.

**However:** With runtime LoRA (addmm), cross-terms literally do not exist.
Each B_i * A_i * x is computed independently and summed in output space.
Pre-merge is mathematically inferior for composition (confirmed: "Pre-merge
WORSE for ternary (-36%, destroys BW advantage)", VISION.md).

### 3.2 Routing Collapse at High N

From our experiments:
- Binary routing heads collapse at N>10 (46% base-only fallback)
- Softmax router matches oracle at N=24 (0% fallback, 0% quality gap)
- Random routing beats trained binary routing at N>=10
- Per-layer depth routing degrades quality -18.3%

**The scaling limit is routing, not composition.** The Welch bound constrains
the minimum pairwise correlation of N unit vectors in R^d:
```
max_{i!=j} |<v_i, v_j>| >= sqrt((N - d) / (d * (N - 1)))
```
For near-orthogonal vectors (correlation -> 0), the maximum number of nearly
orthogonal directions is approximately d. At d_router = 2560, this provides
ample capacity for N=25-50 experts. The practical limit is data quality
(distinguishable domain embeddings), not geometric capacity.

### 3.3 Adapter Quality vs. Base Model Quality

From our Oracle PPL spread analysis:
- Quality spread across 24 adapters: 6.2x
- Quality driven by BASE MODEL per-domain capability, not adapter training
- LOO pruning removes 5/24 (20.8%) with +0.43% same-domain impact
- Worst adapters are on domains where the base model is weakest

**Implication:** Scaling expert count beyond base model capacity is wasteful.
The optimal N is bounded by the number of domains where the base model has
sufficient prior knowledge for a rank-16 perturbation to be useful.

## 4. Optimal Expert Count for Our Parameter Budget

### 4.1 Theoretical Capacity

Grassmannian capacity (purely geometric):
```
N_max = d^2 / r^2 = 2560^2 / 16^2 = 25,600 experts
```

Memory budget (M5 Pro 48GB, from our analysis):
```
N_max_memory = (48GB - base - KV - overhead) / per_adapter
             = (48 - 1.18 - 0.8 - 8) / 0.0452
             = 853 adapters
```

Practical quality limit (from scaling experiments):
```
N_quality ~ 25-50 (gamma > 0.95 range)
```

### 4.2 Recommendations for Expert Count

Based on the literature and our empirical results:

1. **k=2 activated per token** (DeepSeek-V3, Mixtral both use top-2).
   LoRA Soups: k=2 skill composition gives super-linear gains. k>=3
   loses to data mixing. Our architecture should select top-2 experts.

2. **N=8-25 total experts** is the sweet spot for 2B base:
   - N=8: matches Mixtral's expert count, proven at production scale
   - N=25: proven to scale with gamma=0.982 in our experiments
   - N>25: marginal returns (adapters cover diminishing knowledge gaps)
   - N>50: memory starts constraining (2.79 GB at N=50)

3. **Reasoning domains need fewer, larger experts.** MoE scaling research
   shows reasoning tasks saturate/regress with too many experts while
   knowledge tasks (TriviaQA) improve. Recommendation: rank-32 for
   reasoning (math, code), rank-16 for knowledge (medical, legal, finance).

## 5. Complexity Analysis

### 5.1 FLOPs Comparison

| Method | FLOPs per token per layer | At our scale |
|--------|--------------------------|--------------|
| Dense base | 2 * d_out * d_in | 13.1M |
| +k LoRA (runtime) | + 2k * d * r | +0.164M (k=2) |
| +k LoRA (pre-merge) | same as dense (after merge) | 13.1M |
| Traditional MoE top-k | 2k * d * d_ffn / N | 2.46M (k=2, N=8) |
| CoMoL core-space | + 2*d*r + N*r^2 | +0.086M (N=8) |

Runtime LoRA overhead: 1.25% for k=2, r=16 -- negligible.

### 5.2 Memory Comparison

| Component | Our Architecture | Traditional MoE |
|-----------|-----------------|-----------------|
| Base model | 1.18 GB (ternary) | ~4 GB (FP16 2B) |
| Per expert | 45.2 MB (LoRA r=16) | ~55 MB (FFN, FP16) |
| 25 experts | 2.26 GB total | 5.37 GB total |
| Router | 82 KB (softmax) | ~10 MB (per-layer) |

## 6. Worked Example at Micro Scale (d=64, r=4, N=4)

Base weight W in R^{64x64}, 4 LoRA adapters with A_i in R^{4x64}, B_i in R^{64x4}.

Input x in R^{64}:
1. Base output: h_base = W * x (64 multiplies per output dim)
2. Router scores: g = softmax(W_router * x), select top-2
3. Say g_1 = 0.6, g_3 = 0.4 selected
4. Adapter 1: delta_1 = B_1 * (A_1 * x) = B_1 * z_1 where z_1 in R^4
5. Adapter 3: delta_3 = B_3 * (A_3 * x) = B_3 * z_3
6. Output: h = h_base + 0.6 * delta_1 + 0.4 * delta_3

FLOPs:
- Base: 64*64 = 4096
- Per adapter: 64*4 + 64*4 = 512
- 2 adapters: 1024
- Router: 64*4 = 256
- Total overhead: (1024+256)/4096 = 31.2% (but at d=2560, r=16: 1.25%)

Orthogonality check (Grassmannian A):
- A_1^T * A_3 = 0_{4x4} (by construction)
- delta_1^T * delta_3 = z_1^T * B_1^T * B_3 * z_3
  (B correlation doesn't matter for output-space composition because
   we sum outputs, not parameters)

## 7. Connection to Architecture

### 7.1 Our Runtime LoRA = Output-Space MoE

Our serving architecture (addmm fusion, 97 tok/s with adapter) computes:
```
h = W*x + B*A*x  (via addmm kernel)
```

For multi-adapter routing:
```
h = W*x + sum_{i in top-k} g_i * B_i * (A_i * x)
```

This is mathematically identical to MoE output-space composition (confirmed by
MULTI_ADAPTER_SELECTION_MOE_COMPOSITION.md Section 4.2, LoRI arxiv:2504.07448
Section on concatenated composition). Each LoRA adapter is a "micro-expert"
that specializes the base model's output.

### 7.2 Implications for MoE Scaling Laws

Since our architecture IS output-space MoE with low-rank experts:
- MoE scaling laws DO apply conceptually (more experts + routing = better)
- But the coefficients are DIFFERENT (alpha, beta, gamma in Apple's formula)
- Expert size is 432-648x smaller than full FFN experts
- Therefore: the "minimum viable scale" is hundreds of times lower than traditional MoE
- At 2B base, this puts us well above any reasonable minimum
- The binding constraint is BASE MODEL quality, not expert count or size

### 7.3 Production Architecture Comparison

| Aspect | DeepSeek-V3 | Qwen3-MoE | SOLE (ours) |
|--------|------------|-----------|-------------|
| Expert type | Full FFN | Full FFN | LoRA (B*A) |
| Expert size | 17.7M | ~5M | 81.9K |
| Total experts | 256 | 128 | 25 (proven) |
| Active experts | 2 | 8 | 2 (recommended) |
| Gating | Sigmoid+bias | Softmax | Softmax |
| Balance | Aux-loss-free bias | Global-batch loss | N/A (external) |
| Shared expert | Yes (1) | No | Base model (implicit) |

Our base model IS the "shared expert" in DeepSeek's terminology. All LoRA
adapters are perturbations to this shared expert. This is actually more
memory-efficient than DeepSeek's approach because we don't duplicate the
shared computation.
