# Softmax Router Scaling: Mathematical Foundations

## 1. Mechanism Definition

### Binary Sigmoid Routing (Previous, Failed)

For N domains, the binary approach trains N independent classifiers:

$$h_i: \mathbb{R}^d \to [0,1], \quad h_i(x) = \sigma(W_2^{(i)} \cdot \text{ReLU}(W_1^{(i)} x))$$

where $W_1^{(i)} \in \mathbb{R}^{32 \times 2560}$, $W_2^{(i)} \in \mathbb{R}^{1 \times 32}$.

Decision: adapter $i$ fires iff $h_i(x) > 0.5$.

**Parameter count:** $N \times (2560 \times 32 + 32 + 32 \times 1 + 1) = N \times 82,017$

### Multi-class Softmax Router (This Experiment)

Single router mapping hidden states to a distribution over N classes:

$$r: \mathbb{R}^d \to \Delta^{N-1}, \quad r(x) = \text{softmax}(W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2)$$

where $W_1 \in \mathbb{R}^{128 \times 2560}$, $b_1 \in \mathbb{R}^{128}$, $W_2 \in \mathbb{R}^{N \times 128}$, $b_2 \in \mathbb{R}^N$.

**Parameter count:** $2560 \times 128 + 128 + 128 \times N + N = 327,808 + 129N$

At N=24: 330,904 params (single model) vs. 1,968,408 params (24 binary heads).

Selection: top-k from $r(x)$:
- Top-1: $j^* = \arg\max_j r_j(x)$, activate adapter $j^*$ only
- Top-2: $\{j_1^*, j_2^*\} = \text{argtop2}(r(x))$, activate both with scale/2

## 2. Why It Works

### The Class Imbalance Problem

With N binary heads, each head sees a 1:$(N-1)$ class imbalance. At N=24, this is 1:23.

The optimal strategy for minimizing BCE with 1:23 imbalance:
- Always predict 0 (negative): accuracy = 23/24 = 95.8%
- The gradient from the single positive example per batch is overwhelmed

Empirically observed: positive recall < 15% for most domains at N>=10.

### Softmax Eliminates Imbalance

Cross-entropy loss on a balanced N-class problem (equal samples per domain):

$$\mathcal{L} = -\frac{1}{|D|} \sum_{(x,y) \in D} \log r_y(x)$$

Each training sample contributes gradient to ALL N logits simultaneously:
- Correct class gets $\nabla_{z_y} \mathcal{L} = r_y(x) - 1$ (push up)
- Wrong classes get $\nabla_{z_j} \mathcal{L} = r_j(x)$ (push down)

No sample is wasted. Every forward pass provides signal for every class.

### Guaranteed Selection (Zero Fallback)

$\text{softmax}(z)_j > 0$ for all $j$. Therefore $\arg\max_j r_j(x)$ always
returns a valid index. There is no "no adapter selected" state.

This eliminates the 46% base-only fallback observed with binary heads at N=24.

## 3. What Breaks It

**Calibration collapse at very high N:** As $N \to \infty$, the softmax
temperature must scale. With fixed temperature $\tau=1$, the maximum
probability $\max_j r_j(x)$ decreases as $O(1/N)$ for uniform logits.
This means the router becomes less confident. At our scale (N=24),
this is not a concern (1/24 = 4.2% random baseline).

**Hidden state separability:** If hidden states from different domains overlap
significantly in the model's representation space, no router can separate them.
The previous experiment showed 100% accuracy at N=5, suggesting good separability
in the first 5 domains. The question is whether this holds to N=24.

**Kill criterion K1:** If top-1 accuracy < 50% at N=24, the hidden states are
not separable enough for any linear-probe-style router. This would indicate a
fundamental representation limitation, not a routing architecture issue.

## 4. Assumptions

1. **Hidden states are domain-discriminative.** Justified: prior experiment showed
   100% routing accuracy at N=5 with tiny binary heads. The question is scaling.

2. **Mean pooling preserves domain signal.** Justified: standard approach for
   sequence classification (Sentence-BERT, etc.). Alternative: CLS token or
   last-token pooling.

3. **Two-layer MLP with 128 hidden units has sufficient capacity.** Justified:
   128-dim hidden should separate 24 classes from 2560-dim input easily
   (information-theoretic: needs log2(24) = 4.6 bits from 2560 dims).

4. **Cross-entropy is appropriate loss.** Justified: standard for multi-class
   classification. MoLoRA (arXiv 2603.15965) uses the same approach.

## 5. Complexity Analysis

| Component | Binary (N heads) | Softmax (single) |
|-----------|------------------|-------------------|
| Parameters | 82,017N | 327,808 + 129N |
| Training FLOPs/step | O(N * d * h) | O(d * H + H * N) |
| Inference FLOPs | O(N * d * h) | O(d * H + H * N) |
| At N=5 | 410K params | 328K params |
| At N=24 | 1.97M params | 331K params |
| At N=100 | 8.2M params | 341K params |

Where d=2560, h=32 (binary hidden), H=128 (softmax hidden).

The softmax router scales as O(N) in only the output layer, while binary
routing scales O(N) in the entire network.

## 6. Worked Example (N=4)

Input: hidden state $x \in \mathbb{R}^{2560}$ (mean-pooled).

Forward pass through router:
1. $h = \text{ReLU}(W_1 x + b_1)$, $h \in \mathbb{R}^{128}$
2. $z = W_2 h + b_2$, $z \in \mathbb{R}^4$, e.g., $z = [2.1, -0.3, 0.5, -1.2]$
3. $r = \text{softmax}(z) = [0.65, 0.06, 0.13, 0.02]$ (sums to ~0.86 + normalization)

Top-1: select adapter 0 (p=0.65). Top-2: select adapters 0 and 2.

Composition for top-1 on layer $l$, module $m$:
$$y = W_m x + \frac{\alpha}{1} (x A_0^{(l,m)}) B_0^{(l,m)}$$

Composition for top-2:
$$y = W_m x + \frac{\alpha}{2} [(x A_0^{(l,m)}) B_0^{(l,m)} + (x A_2^{(l,m)}) B_2^{(l,m)}]$$

## 7. Connection to Architecture

The softmax router is a direct replacement for the binary routing heads in the
SOLE architecture. It takes the same input (mean-pooled hidden states from the
base model) and produces the same output (which adapter(s) to activate).

**vs. MoLoRA (arXiv 2603.15965):** MoLoRA uses per-token softmax routing
integrated into each transformer layer. Our approach is per-sequence routing
from the final hidden state. This is correct for our architecture because
prior experiment (exp_pointer_routing_no_merge) showed per-layer mixing is
harmful: adapters are calibrated for same-adapter residual stream at all layers.

**vs. DeepSeek-V3:** Uses auxiliary-loss-free load balancing with bias terms
in the router. We don't need load balancing because our "experts" serve
different domains, not different compute patterns.
