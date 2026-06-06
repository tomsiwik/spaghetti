# FFN Neurons as Key-Value Memories in Gemma 4 E4B

## Prior Work
Geva et al. (arXiv:2012.14913) proved that Transformer FFN layers function as
key-value memories: each neuron stores a key (input pattern detector) and value
(output distribution over vocabulary). Upper-layer value vectors predict next
tokens with ~3.5% agreement (vs 0.0004% random with 260K vocab).

## Theorem: GeGLU FFN as Gated Key-Value Memory

**Setup.** Gemma 4 E4B uses GeGLU activation:
$$\text{FFN}(x) = W_{\text{down}} \cdot (\sigma(W_{\text{gate}} x) \odot W_{\text{up}} x)$$
where $\sigma = \text{GELU}_{\text{approx}}$, $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d_{ff} \times d}$,
$W_{\text{down}} \in \mathbb{R}^{d \times d_{ff}}$, $d = 2560$.

**Definition.** For neuron $i$ in layer $\ell$:
- **Key**: The pair $(w_{\text{gate},i}, w_{\text{up},i})$ — row $i$ of gate and up projections
- **Activation**: $a_i(x) = \sigma(w_{\text{gate},i}^\top x) \cdot (w_{\text{up},i}^\top x)$
- **Value**: $v_i = W_{\text{down}}[:, i]$ — column $i$ of the down projection

**Theorem 1** (Decomposition into neuron contributions).
$$\text{FFN}(x) = \sum_{i=1}^{d_{ff}} a_i(x) \cdot v_i$$

*Proof.* Direct from matrix-vector product definition. The output of the GeGLU
is $h = [a_1(x), \ldots, a_{d_{ff}}(x)]^\top$. Then $W_{\text{down}} h = \sum_i h_i \cdot W_{\text{down}}[:, i] = \sum_i a_i(x) \cdot v_i$. QED.

**Theorem 2** (Value vector vocabulary projection).
For tied embeddings $E \in \mathbb{R}^{V \times d}$ and logit softcapping $\tau$:
$$p_i = \text{softmax}(\tanh(E \cdot v_i / \tau) \cdot \tau)$$
gives the vocabulary distribution induced by neuron $i$'s value vector.

**Theorem 3** (Gating sparsifies activation).
The GELU gate $\sigma(w_{\text{gate},i}^\top x)$ is approximately 0 for
$w_{\text{gate},i}^\top x < -3$ and approximately linear for $w_{\text{gate},i}^\top x > 3$.
This means each neuron only contributes to the output for inputs that match its
gate pattern, creating a sparse, interpretable activation structure.

## Architecture-Specific Details

Gemma 4 E4B has 42 dense FFN layers (no MoE):
- Layers 0-21: $d_{ff} = 10240$ (standard)
- Layers 22-41: $d_{ff} = 20480$ (double-wide, KV-shared layers)
- Total neurons: $22 \times 10240 + 20 \times 20480 = 634,880$

The double-wide upper layers have 2x capacity for knowledge storage,
consistent with the finding that upper layers store more specific factual
associations (Geva et al., Section 5).

## Predictions

### K1: Pattern Specificity (>=50%)
Geva et al. found 65-80% of top neurons match identifiable patterns in a
16-layer model. With 42 layers and larger intermediate dimensions, patterns
may be more distributed. We predict >=50% of frequently-activated neurons
show domain-specific activation (>70% of triggers from a single domain).

**Quantitative prediction**: 55-70% pattern rate.

### K2: Domain Clustering
If neurons specialize, the overlap between domain-specific neuron sets
(math-math) should exceed cross-domain overlap (math-medical) by >=2x.

**Quantitative prediction**: Intra-domain Jaccard similarity >= 2x inter-domain.

### K3: Next-Token Agreement (>=1%)
Geva et al. found 3.5% with a 16-layer model and 268K vocab. Gemma 4 has
262K vocab (similar) but 42 layers (deeper = more refined representations).
However, 4-bit quantization degrades value vector precision.

**Quantitative prediction**: 1.5-4% agreement in layers 35-41.
Random baseline: 1/262144 = 0.0004%.

## Kill Criteria (from experiment DB)
- K1372: >=50% of top-activating neurons match identifiable patterns
- K1373: Domain-specific neurons cluster (intra >= 2x inter)
- K1374: Upper-layer value vectors predict next token with >1% agreement

## What Could Kill This
1. **4-bit quantization destroys value vectors**: Dequantized weights may be
   too noisy for meaningful vocabulary projections. Kill K3.
2. **GeGLU spreads activation**: Unlike ReLU (which creates hard zeros), GELU
   produces small but non-zero activations, making pattern identification
   harder. Kill K1.
3. **Per-layer input gating (PLE)**: Gemma 4's per-layer embeddings add a
   secondary information channel that may reduce FFN's role as primary
   knowledge storage. Kill K1/K2.
