# Pointer Routing (No Merge): Mathematical Foundations

## 1. Mechanism Definition

### The Two Composition Spaces

**Parameter-space merge (current uniform 1/N):**
For N adapters with LoRA deltas Delta_W_i = B_i A_i (B_i in R^{d_out x r}, A_i in R^{r x d_in}), uniform composition computes:

    y = (W + (1/N) sum_{i=1}^{N} B_i A_i) x

The nonlinearity sigma (RMSNorm, SiLU in MLP, softmax in attention) sees AVERAGED weight parameters. Cross-terms B_i A_j (i != j) are zero by Grassmannian orthogonality (A_i^T A_j = 0), but the 1/N dilution weakens each expert's contribution: each adapter contributes only scale/N of its intended perturbation.

**Output-space selection (pointer routing):**
Layer l selects expert pi(l, x) based on input x:

    y_l = W_l x + B_{pi(l,x)} A_{pi(l,x)} x * scale

The nonlinearity sees ONE expert's full-strength weight perturbation. No 1/N dilution. No cross-terms because only one adapter is active per layer.

### Pointer Routing Variants

Given L layers, N adapters, input hidden state h_l in R^{d} at layer l:

**(a) Learned Gate (per-layer linear classifier):**

    pi_a(l, x) = argmax_i (g_l^T h_l + b_l)_i

where g_l in R^{d x N}, b_l in R^N are learned per-layer routing parameters.
Total routing params: L * (d * N + N) = 30 * (2560 * 5 + 5) = 384,150.

**(b) Hash Lookup (SOLE hash-ring):**

    pi_b(l, x) = hash(l, domain(x)) mod N

Deterministic per-layer assignment based on input domain and layer index.
No learned parameters. O(1) lookup.

**(c) Input-Dependent MLP Router:**

    pi_c(l, x) = argmax_i MLP_l(mean(h_l))

where MLP_l: R^d -> R^N is a 2-layer network with hidden dim d_h:
    MLP_l(z) = W2_l * ReLU(W1_l * z + b1_l) + b2_l

with W1_l in R^{d_h x d}, W2_l in R^{N x d_h}.
Total routing params: L * (d * d_h + d_h + d_h * N + N).
At d_h = 64: 30 * (2560*64 + 64 + 64*5 + 5) = 4,929,750.
At d_h = 16: 30 * (2560*16 + 16 + 16*5 + 5) = 1,231,830.

### Tensor Shapes (BitNet-2B-4T)

| Object | Shape | dtype |
|--------|-------|-------|
| h_l (hidden state) | (seq, d) = (seq, 2560) | bfloat16 |
| A_i (frozen Grassmannian) | (d_in, r) = (2560, 16) or (6912, 16) | bfloat16 |
| B_i (ternary STE) | (r, d_out) = (16, 2560) or (16, 6912) | bfloat16 |
| g_l (gate weights) | (d, N) = (2560, 5) | bfloat16 |
| W1_l (MLP router) | (d_h, d) = (64, 2560) | bfloat16 |

### Per-Layer vs Per-Sequence vs Per-Token

| Granularity | Description | Cost |
|-------------|-------------|------|
| Per-token (MoE standard) | Each token picks different expert per layer | N forward passes, O(seq * L * N) |
| Per-sequence (our prior work) | All tokens in sequence use same expert per layer | 1 routing decision per layer, O(L * N) |
| Per-layer (this experiment) | Input selects expert per layer, same for all tokens in sequence | O(L) routing decisions |

This experiment uses per-LAYER, per-SEQUENCE routing: for a given input sequence, each layer independently picks ONE adapter. Different layers can pick different adapters, enabling depth-wise specialization (e.g., layer 0 picks medical vocab, layer 15 picks code structure, layer 29 picks legal reasoning).

## 2. Why It Works

### Switch Transformer Argument (Fedus et al., 2101.03961)

Switch Transformer proved that top-1 expert selection per layer is sufficient for MoE quality, despite using only 1/N of expert capacity per token. The key insight: different layers learn to specialize on different aspects of the input. Early layers capture lexical/syntactic patterns; later layers capture semantic/task-specific patterns.

Our setting is analogous: each layer applies ONE adapter at full strength. The adapter selection varies by layer, allowing depth-wise specialization.

### Output-Space vs Parameter-Space Composition

Let f_i(x) = (W + B_i A_i) x be the output of layer with adapter i at full strength.

**Parameter merge:** f_merge(x) = (W + (1/N) sum_i B_i A_i) x
                    = W x + (1/N) sum_i B_i A_i x
                    = W x + (1/N) sum_i (f_i(x) - W x)
                    = (1 - 1) W x + (1/N) sum_i f_i(x)  [only for linear case]

For LINEAR layers, 1/N parameter merge = 1/N output average. But the next operation is NONLINEAR (RMSNorm + activation):

    sigma(f_merge(x)) != (1/N) sum_i sigma(f_i(x))

Jensen's inequality: for convex sigma, sigma(E[f]) <= E[sigma(f)].
For concave sigma (like the squashing in RMSNorm), the opposite holds.
The point is: they are NOT equal. Output-space selection avoids this mismatch entirely because sigma sees one expert's output, not an average.

### Hash Layers Argument (Roller et al., 2106.04426)

Hash Layers showed that O(1) hash-based routing achieves comparable quality to learned routing in Transformer feed-forward layers. Random hash functions are sufficient because:
1. Expert parameters adapt during training to the data assigned by the hash.
2. The hash implicitly regularizes by preventing co-adaptation.

In our setting, adapters are pre-trained (not adapted to hash assignment), so hash routing is a WEAKER baseline. Learned gates should outperform.

## 3. What Breaks It

### Failure Mode 1: Adapter Fragmentation

If per-layer selection scatters adapters across layers without coherent specialization, the model receives conflicting signals at different depths. With 30 layers and 5 adapters, there are 5^30 possible layer-adapter assignments. Most are incoherent.

**Kill condition (K2):** If learned routing converges to uniform random or constant assignment (all layers pick same adapter), depth specialization fails. Measured by: entropy of per-layer adapter distribution and chi-square test of assignment independence across layers.

### Failure Mode 2: Full-Strength Interference

At 1/N uniform, each adapter contributes scale/N = 20/5 = 4.0 of its perturbation. At full strength (pointer routing), one adapter contributes scale = 20.0 — a 5x increase per layer. If any adapter's perturbation is too large, it could destabilize the residual stream.

**Mitigation:** Scale adjustment. Test both scale=20.0 (full) and scale=20.0/sqrt(N) (geometric mean).

### Failure Mode 3: Routing Collapse

Learned gates may collapse to always selecting the same expert (mode collapse), especially with small routing networks trained on limited data. This is equivalent to failure mode 1.

**Detection:** Per-layer expert usage counts. If any expert is selected <5% of the time across all layers, routing has partially collapsed.

## 4. Assumptions

1. **Pre-trained adapters are interchangeable across layers.** Each adapter was trained on ALL layers simultaneously. Applying adapter i only at layer l uses parameters trained jointly but evaluated in isolation. Assumption justified by: Switch Transformer uses per-layer expert selection despite experts trained in context of the full model. Risk if wrong: adapter parameters may be calibrated for the residual stream contribution of all OTHER layers' adapters being active. Mitigation: compare against single-adapter oracle (where all layers use the same adapter).

2. **Hidden states carry domain signal.** The router needs h_l to contain enough domain information for classification. Justified by: prior routing heads achieve 99.9% accuracy on these 5 domains from hidden states (real_data_domain_experts experiment). Risk: later layers may have less domain-discriminative signal due to convergent representations.

3. **Depth-wise specialization is useful.** Different layers benefit from different domain expertise. Justified by: Cornerstone Layers (2409.14381) show layer criticality varies by task; early layers are more transferable, later layers more task-specific. Risk: at 2B scale with only 5 domains, depth variation may not emerge.

## 5. Complexity Analysis

| Method | Routing FLOPs | Adapter FLOPs | Total FLOPs per token |
|--------|---------------|---------------|----------------------|
| Uniform 1/N | 0 | N * (d*r + r*d) * L * 7 = 5 * 2 * 2560 * 16 * 30 * 7 = 172M | 172M |
| Pointer (gate) | L * d * N = 384K | 1 * (d*r + r*d) * L * 7 = 34.4M | 34.8M |
| Pointer (hash) | L * O(1) = ~30 | 34.4M | 34.4M |
| Pointer (MLP d_h=64) | L * (d*d_h + d_h*N) = 4.9M | 34.4M | 39.3M |

Pointer routing uses **5x fewer adapter FLOPs** than uniform 1/N because only 1 adapter is applied per layer instead of N=5. The routing overhead is negligible (0.3-14% of adapter FLOPs).

**Memory:** All methods load all N adapter B-matrices (A-matrices from skeleton). No memory savings from pointer routing vs uniform. The savings are in compute only.

## 6. Worked Example (L=4 layers, N=3 adapters, d=8, r=2)

Input sequence: medical question about medication dosage.

**Uniform 1/N:** Every layer applies all 3 adapters at scale/3 = 6.67:
    Layer 0: y = Wx + 6.67 * (B0 A0 + B1 A1 + B2 A2) x
    Layer 1: y = Wx + 6.67 * (B0 A0 + B1 A1 + B2 A2) x
    (same for layers 2, 3)

**Pointer routing (learned gate):**
    Router at layer 0: g0^T h0 = [2.1, 0.3, -0.5] -> pick adapter 0 (medical)
    Router at layer 1: g1^T h1 = [0.8, 2.4, 0.1] -> pick adapter 1 (code - for numeric formatting)
    Router at layer 2: g2^T h2 = [2.5, -0.1, 0.7] -> pick adapter 0 (medical)
    Router at layer 3: g3^T h3 = [1.9, 0.2, 1.8] -> pick adapter 0 (medical)

    Layer 0: y = Wx + 20.0 * B0 A0 x  (full medical adapter)
    Layer 1: y = Wx + 20.0 * B1 A1 x  (full code adapter for numeric structure)
    Layer 2: y = Wx + 20.0 * B0 A0 x  (medical again)
    Layer 3: y = Wx + 20.0 * B0 A0 x  (medical for output)

Depth-wise specialization: layer 1 borrows code formatting while other layers handle medical content.

## 7. Connection to Architecture

**Relation to SOLE hash-ring:** SOLE uses a consistent hash ring for per-sequence routing: hash(input) maps to ONE adapter applied at ALL layers. Pointer routing extends this to per-LAYER granularity. The hash variant (b) is a direct extension of SOLE: hash(layer_id || input_domain) mod N.

**Relation to per-token routing (MoLoRA):** MoLoRA routes per-token with Gumbel-sigmoid gates. Pointer routing is coarser: per-sequence, per-layer. This is computationally cheaper (L routing decisions vs seq*L routing decisions) and avoids the need for differentiable routing during inference.

**Relation to runtime LoRA:** The runtime LoRA serving pathway (proven at 12.3 tok/s) applies adapters during forward pass without pre-merging. Pointer routing is a GENERALIZATION: instead of applying the SAME adapter at every layer, apply different adapters at different layers. The serving infrastructure is identical — just index a different adapter per layer.

**Production reference:** DeepSeek-V3 uses 256 experts with top-8 routing per token per layer. Each layer independently selects experts. Our experiment tests the same principle at adapter scale: independent per-layer selection from a pool of pre-trained experts.
