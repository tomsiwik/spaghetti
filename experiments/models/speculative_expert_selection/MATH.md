# Mathematical Foundations: Speculative Expert Selection

## 1. Mechanism Definition

### Setup
We have N domain experts with a softmax router R: R^d -> [N] that maps a hidden state
h_t at token position t to a selected expert index:

  e_t = argmax_k softmax(W2 * relu(W1 * h_t))_k

where W1 in R^{128 x 2560}, W2 in R^{N x 128}, and h_t in R^{2560} is the mean-pooled
hidden state at the router's observation layer.

### Speculative Selection
Define the speculation predictor as the identity map on expert indices:

  e_hat_{t+1} = e_t

That is, we predict the next token's expert is the same as the current token's expert.
If e_hat_{t+1} = e_{t+1} (hit), we skip the router forward pass for token t+1.
If e_hat_{t+1} != e_{t+1} (miss), we run the router and use its output.

### Cost Model
Let C_R be the cost of one router forward pass (measured: 0.21ms, 0.58% of base forward).
Let C_gen be the cost of one token generation step (measured: ~36ms for base + adapter).

Without speculation: total cost per token = C_gen + C_R
With speculation at hit rate p: total cost per token = C_gen + (1-p)*C_R + epsilon

where epsilon is the comparison cost (negligible: one integer equality check).

Speedup = C_R * p / (C_gen + C_R)

At p=0.80 (target S1): speedup = 0.21 * 0.80 / (36 + 0.21) = 0.168 / 36.21 = 0.46%
At p=1.00 (perfect):   speedup = 0.21 / 36.21 = 0.58%

**Critical observation:** Even at 100% hit rate, speculative expert selection can only
save 0.58% of total inference time because the router is already only 0.58% of cost.
The maximum possible speedup is bounded by the router's fraction of total compute.

## 2. Why It Works (or Doesn't)

### Autocorrelation Model
Model expert selection as a first-order Markov chain with transition matrix
P in R^{N x N}, where P_{ij} = Pr(e_{t+1}=j | e_t=i).

For domain-coherent text (e.g., a medical article), the hidden states evolve smoothly
along the residual stream. Consecutive tokens within the same semantic context produce
similar hidden states, so:

  ||h_{t+1} - h_t|| << ||h_t||  (within-domain)

If the router decision boundary is far from h_t (i.e., the token is deep within a
domain cluster), then h_{t+1} will also be classified to the same expert.

The self-transition probability for domain i is:
  P_{ii} = Pr(argmax softmax(W2*relu(W1*h_{t+1})) = i | argmax softmax(W2*relu(W1*h_t)) = i)

For well-separated clusters, this is approximately:
  P_{ii} approx 1 - Pr(h_{t+1} crosses any decision boundary | h_t in region_i)

### Expected Hit Rate Under Markov Model
If the text is drawn from a single domain with M tokens, and the router has
within-domain stability p_stable, then:

  E[hits] = (M-1) * p_stable

The overall hit rate across D domains with M_d tokens each:
  p_overall = sum_d (M_d - 1) * p_stable_d / sum_d (M_d - 1)

### Connection to Prior Results
**exp_pointer_routing_no_merge** found that per-sequence routing equals per-token routing
on clean domain text (delta = -0.46%). This means the router produces the SAME expert
for nearly all tokens within a domain-coherent sequence. This directly implies high
autocorrelation: if all tokens in a sequence get the same expert, then P_{ii} approx 1.

**exp_softmax_router_scaling** showed 40% classification accuracy but oracle-matching
quality, because within-cluster misrouting is benign. For autocorrelation, what matters
is not whether the router picks the RIGHT expert, but whether it picks the SAME expert
consecutively. If the router consistently maps a domain to expert k (even if k is "wrong"),
autocorrelation will be high.

## 3. What Breaks It

### Domain Boundaries
At a transition from domain A to domain B text, the expert must change.
If text alternates domains every S tokens:
  p_overall = (S-1)/S

At S=2 (every other token changes domain): p=0.50 (K1 boundary)
At S=10: p=0.90
At S=100: p=0.99

### Within-Domain Topic Shifts
Even within a single domain, subtle topic shifts may cause the router to oscillate
between semantically similar experts (e.g., "philosophy" vs "history" in the
philosophy-history-agriculture cluster identified by exp_softmax_router_scaling).

### The Real Problem: Router Cost Is Already Negligible
From exp_molora_per_token_routing:
- Router forward pass: 0.21ms (0.58% of total)
- Token generation: ~36ms

Even eliminating the router entirely saves <0.6%. This means:
- K1 (prediction accuracy >= 60%) is scientifically interesting but practically irrelevant
- K2 (net speedup > 10%) is mathematically IMPOSSIBLE given the cost structure

The experiment will measure autocorrelation as a scientific question about expert
selection dynamics, while honestly acknowledging the speedup ceiling.

## 4. Assumptions

1. **Hidden state smoothness:** Consecutive tokens have similar hidden states
   within domain-coherent text. Justified by: transformer residual stream is
   a continuous function of input, and consecutive tokens share context.

2. **Router stability:** Small perturbations in hidden state don't change the
   argmax of the softmax router. Justified by: exp_softmax_router_scaling shows
   semantic clustering with well-separated clusters.

3. **Domain coherence:** Evaluation text consists of single-domain passages.
   Justified by: our 24-domain eval data is domain-pure by construction.

4. **Router overhead is representative:** The 0.21ms measurement from
   exp_molora_per_token_routing applies to the softmax router. Softmax router
   is similar size (330K params vs per-adapter heads).

## 5. Complexity Analysis

Router forward pass: 2560*128 + 128*N = 327,680 + 128*24 = 330,752 FLOPs
Comparison (speculation): 1 integer compare = O(1)
Memory: O(1) extra (store one integer: previous expert index)

The speculation mechanism adds essentially zero overhead. The question is purely
whether hit rate is high enough to be interesting.

## 6. Worked Example (N=5 domains)

Suppose domains = {medical, code, math, legal, finance} with 20 tokens each.

Token stream from medical article: [med, med, med, ..., med] (20 tokens)
Expert selection: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

Speculation hits: 19/19 = 100% (first token has no prediction)

Now consider mixed text: [med, med, code, code, med, ...]
Expert selection: [3, 3, 1, 1, 3, ...]
Speculation hits: 3/4 = 75% (miss at positions 3 and 5)

## 7. Connection to Architecture

### Speculative Decoding Analogy (Leviathan et al., arXiv 2211.17192)
Speculative decoding uses a small draft model to predict multiple tokens, then verifies
with the large model. Our approach is simpler: the "draft" is just the previous expert
index. Unlike speculative decoding where rejection requires re-generation, our rejection
simply runs the router (which we would have run anyway). There is no wasted computation
on misses -- only the comparison overhead (negligible).

### Production MoE Routing (DeepSeek-V3)
DeepSeek-V3 uses per-token per-layer routing with 256 experts. They do NOT use
speculative selection because (a) their routing is per-layer (experts change each layer),
and (b) routing is baked into the forward pass, not a separate overhead. Our case is
different: SOLE uses per-sequence routing where the expert stays fixed across all layers,
making temporal autocorrelation a natural property.

### Practical Implication
Given the 0.58% ceiling, speculative expert selection is NOT a performance optimization.
It is an empirical measurement of expert selection dynamics -- answering whether expert
choices are temporally stable, which informs cache-friendly serving strategies.
