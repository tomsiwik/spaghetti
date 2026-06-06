# SHINE S2: Context Reconstruction via M2P-Generated LoRA

## Type: Frontier Extension
**Prior result**: Finding #482 (S1 memory extraction produces non-degenerate states),
Finding #339 (M2P generates 66.6% of SFT quality on toy model).
**Gap**: Does M2P work on real Gemma 4 scale? Can generated LoRA reconstruct context?

## Grounding
- arXiv:2602.06358 (SHINE) S3.3: M2P trained with NTP loss generates LoRA in one pass
- Finding #482: Memory states (42, 32, 2560) with mean cross-layer cos=0.182 (non-degenerate)
- Finding #339: M2P achieves 66.6% of SFT quality on toy model (d=128, 4 layers)
- Finding #362: M2P scales to d_model=1024 with 99.6% of SFT

## Theorem (Gradient Flow Through Generated LoRA)

**Setup.** Let $f_\theta$ be a frozen LM with L layers, hidden dim d, and
$q_l: \mathbb{R}^d \to \mathbb{R}^d$ denoting layer l's q_proj linear.
Let $M_\phi: \mathbb{R}^{L \times M \times d} \to \{(A_l, B_l)\}_{l=1}^L$
be an M2P transformer with parameters $\phi$, producing LoRA pairs
$A_l \in \mathbb{R}^{d \times r}$, $B_l \in \mathbb{R}^{r \times d}$.

**LoRA injection.** At each layer, replace $q_l(x)$ with
$\tilde{q}_l(x) = q_l(x) + x A_l B_l$. The modified forward pass is:
$$h_{l+1} = \text{Layer}_l(h_l; \tilde{q}_l)$$

**NTP loss.** $\mathcal{L}(\phi) = \text{CE}(\text{logits}(h_L), \text{targets})$

**Claim.** $\nabla_\phi \mathcal{L} \neq 0$ whenever:
1. $\nabla_{A_l, B_l} \tilde{q}_l(x) \neq 0$ (holds: $\partial(xAB)/\partial A = x^T \otimes B$, $\partial(xAB)/\partial B = (xA)^T$)
2. $\nabla_\phi (A_l, B_l) \neq 0$ (holds: M2P is a transformer with differentiable operations)
3. $\nabla_{\text{logits}} \mathcal{L} \neq 0$ (holds: CE gradient is nonzero for non-degenerate predictions)

**Proof.** By chain rule:
$$\nabla_\phi \mathcal{L} = \sum_{l=1}^L \frac{\partial \mathcal{L}}{\partial \tilde{q}_l} \cdot \frac{\partial \tilde{q}_l}{\partial (A_l, B_l)} \cdot \frac{\partial (A_l, B_l)}{\partial \phi}$$

Each factor is nonzero under conditions (1)-(3). The composition of nonzero
linear maps is nonzero generically (fails only on a measure-zero set of $\phi$).

For quantized base weights: $q_l(x)$ uses quantized matmul but produces float output.
The LoRA term $xA_lB_l$ is pure float. The gradient w.r.t. $x$ through a quantized
layer equals $\text{grad} \cdot \text{dequant}(W)$, which is nonzero. Therefore gradient
flows through the full residual stream. QED.

## Context-Specificity Argument

**Why M2P avoids the centroid trap:** Each training step uses a DIFFERENT context
paragraph $c_i$. The memory states $S(c_i) \in \mathbb{R}^{L \times M \times d}$
vary per context (S1 verified: mean cross-layer cos=0.182, implying distinct per-layer
representations). M2P must map each distinct $S(c_i)$ to a LoRA that minimizes loss
on $c_i$ specifically. A single "average" LoRA cannot achieve this because different
contexts have different optimal next-token distributions.

## Predictions

| ID | Prediction | Threshold | Reasoning |
|----|-----------|-----------|-----------|
| D1 | M2P training loss decreases | >20% reduction | Gradient signal exists (theorem) + S1 non-degeneracy |
| D2 | CE with LoRA < 2x base CE | ratio < 2.0 | Generated LoRA captures context info |
| D3 | Different contexts → different LoRA | cos(LoRA_i, LoRA_j) < 0.9 | Context-specificity, not centroid |

## Kill Criteria (from experiment DB)

- **K1255**: M2P training loss decreases > 20% over 1000 steps
- **K1256**: Generated LoRA enables context reconstruction (CE < 2x base CE)
- **K1257**: Completion accuracy > random baseline with generated LoRA

## Failure Modes

1. **Vanishing gradient through quantized layers**: If gradient through
   `mx.quantized_matmul` w.r.t. input is too small, M2P won't learn.
   Detection: loss stays flat. Fix: dequantize q_proj layers.

2. **LoRA scale collapse**: M2P outputs near-zero LoRA weights (trivial solution
   where adapted model ≈ base model). Detection: ||A_l B_l||_F ≈ 0.
   Fix: scale initialization, add regularizer encouraging nonzero LoRA.

3. **Memory state degeneracy**: Despite S1 showing cos=0.182, after projection
   to m2p_dim=128, states might become degenerate. Detection: M2P output
   identical for different contexts. Fix: increase m2p_dim.
