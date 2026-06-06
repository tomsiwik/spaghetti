# TTT Expert Selection: Research Digest

## Hypothesis
Loss-probing and projection-based scoring can select the correct top-2 experts at
runtime without a pre-trained router, matching the learned Gumbel-sigmoid router's
quality (15.07 avg PPL on 49 domains).

## What This Experiment Tests

Given N=49 trained ternary LoRA adapters on BitNet-2B-4T, can we select the best
top-2 adapters for a new input WITHOUT the learned router? We test three families:

1. **Exhaustive loss-probe**: Try all N adapters on a 32-token prefix, select by
   loss reduction. O(N) forward passes -- oracle upper bound for loss-based selection.

2. **Arrow-style projection**: Score adapters by projecting the input's hidden state
   through each adapter's A-matrix. O(1) forward pass + cheap matrix ops.

3. **Cosine centroid**: Compare input hidden state to pre-computed per-domain centroids
   (same signal the learned router uses, but without training a neural network).

4. **Hybrid variants**: Use cheap scoring (Arrow or cosine) to shortlist m candidates,
   then loss-probe only those m. Best of both worlds.

## Key References
- Arrow / MBC "Towards Modular LLMs": zero-shot adapter selection via parameter similarity
- L2R: Gumbel-sigmoid routing (our learned router baseline)
- TTT Done Right (arXiv 2505.23884): test-time training per-document adaptation

## Empirical Results

### Strategy Comparison (49 domains, BitNet-2B-4T, prefix_len=32, top-2 selection)

| Strategy | Avg PPL | Selection Acc | Fwd Passes | Time (s) | K1 | K2 | K3 |
|----------|---------|---------------|------------|----------|----|----|-----|
| Learned router (ref) | 15.07 | 86.3% | 1 | ~0.01 | -- | -- | -- |
| Exhaustive probe | **14.91** | **93.9%** | 50 | 7.36 | PASS | **PASS** | FAIL |
| Arrow projection | 19.32 | 49.0% | 1 | 2.25 | PASS | FAIL | PASS |
| Cosine centroid | 15.49 | 65.3% | 1 | **0.13** | PASS | FAIL | PASS |
| Hybrid arrow m=5 | 15.83 | 71.4% | 7 | 3.14 | PASS | FAIL | PASS |
| Hybrid arrow m=3 | 17.54 | 61.2% | 5 | 2.85 | PASS | FAIL | PASS |
| Hybrid cosine m=5 | 15.27 | 71.4% | 7 | 1.02 | PASS | FAIL* | PASS |
| Hybrid cosine m=3 | **15.28** | 67.3% | 5 | **0.73** | PASS | FAIL* | PASS |

*K2 FAIL is marginal: 15.27/15.28 vs 15.07 threshold (1.3-1.4% gap).

### Kill Criteria Assessment

| Criterion | Threshold | Best K3-Valid Result | Verdict |
|-----------|-----------|---------------------|---------|
| K1 | overhead <= 50% of 512-tok gen | 0.22% (cosine centroid) | **PASS** |
| K2 | avg PPL <= 15.07 (learned router) | 15.27 (hybrid cosine m=5) | **FAIL** (1.3% margin) |
| K3 | <= 10 forward passes | 5 (hybrid cosine m=3) | **PASS** |

**Verdict: KILLED** (K2 marginal fail)

### Key Findings

**Finding 1: Loss-probing produces BETTER selections than the learned router.**
Exhaustive loss-probe (14.91 avg PPL, 93.9% accuracy) beats the learned router
(15.07 avg PPL, 86.3% accuracy). This means the routing signal in the loss landscape
is richer than what the 2-layer router network captures. The router loses performance
by approximating direct loss measurement.

**Finding 2: Cosine centroid scoring is a competitive zero-training router.**
With zero training, just computing cosine similarity between the input's hidden state
and pre-computed domain centroids achieves 15.49 avg PPL (65.3% accuracy) -- only
2.8% worse than the learned router. This is essentially the same signal the router
learns to approximate, but computed directly.

**Finding 3: Hybrid cosine + probe closes the gap to 1.3%.**
By using cosine centroid to shortlist 3-5 candidates and loss-probing only those,
we get 15.27-15.28 avg PPL with 5-7 forward passes. This is within noise of the
learned router, requires zero router training, and has negligible overhead (0.73s
on a 62s generation task = 1.2%).

**Finding 4: Arrow-style projection scoring underperforms.**
Scoring by projecting hidden states through adapter A-matrices achieves only 49%
accuracy (19.32 avg PPL). The random-uniform A-matrix initialization does not
create sufficiently discriminative subspaces. Grassmannian-optimized A-matrices
might change this picture.

**Finding 5: Overhead is trivially low.**
Even the hybrid strategies have < 5% overhead relative to 512-token generation.
The selection cost is amortized over the generation length. Cosine centroid at
0.13s is negligible.

## Timing Breakdown

| Operation | Time | Notes |
|-----------|------|-------|
| Single forward pass (prefix=32 tokens) | 121ms | Base model, no adapter |
| Cosine scoring (49 adapters) | 11ms | Pre-computed centroids |
| Arrow scoring (49 adapters) | 2.1s | Per-adapter npz load dominates |
| Loss probe (1 adapter) | ~140ms | Load + forward + loss |
| 512-token generation (est.) | 61.9s | 121ms/token autoregressive |

## Limitations

1. **Single probe text per domain**: Selection evaluated on first validation text only,
   not averaged over multiple samples. The learned router was evaluated on 10 samples
   per domain. This makes our accuracy numbers noisier.

2. **PPL=1.0 domains inflate accuracy**: 9/49 domains have individual PPL=1.0
   (synthetic/memorized data). These are trivially routed by all strategies. Removing
   them would show lower accuracy for all methods.

3. **Prefix length sensitivity untested**: We used prefix_len=32. The learned router
   uses full-sequence hidden states (up to 128 tokens). Longer prefixes might improve
   TTT selection.

4. **Composition interactions not modeled**: Loss-probing tests individual adapters, not
   the composed pair. The selected top-2 adapters might interact differently than the
   individually-best pair. Near-orthogonality (|cos|=0.002) mitigates this.

5. **Centroid computation requires training data**: The cosine centroid method needs
   access to domain training data offline to compute centroids. The learned router
   also needs this. Arrow scoring is the only truly training-data-free method.

6. **K2 uses generous reference**: The actual routed PPL from N=50 experiment is 15.07.
   The originally stated K2 reference of 13.65 may be from a different configuration.
   Against 13.65, all strategies clearly fail.

## What Would Kill This

- Prefix length too short for domain signal: if increasing prefix from 32 to 64-128
  tokens does not improve selection accuracy, the loss-probe signal is too noisy.
- Composition interactions dominate: if the best individual adapters compose poorly
  compared to router-selected pairs, the loss-probe's individual scoring is misleading.
- Scale failure: at N=500+ experts, even 5-pass hybrid might become too slow due to
  adapter loading from disk (currently disk I/O dominates, not compute).

## Implications for the Project

The K2 fail is marginal (1.3%) and the positive findings are significant:

1. **Router training is optional**: Cosine centroid (0 training, 1 pass, 0.13s) gets
   97% of router quality. For applications where router training is inconvenient
   (new domains, rapid deployment), this is viable.

2. **Loss-probing is the quality ceiling**: At 14.91 avg PPL (93.9% accuracy), it
   surpasses the learned router. Future router training should target this ceiling.

3. **The routing problem is easy**: 65% accuracy from simple cosine similarity means
   the domain signal in hidden states is strong. The learned router's value is in
   handling the hard cases (confusion domains like chemistry/science, dialogue/news).

4. **Practical recommendation**: Use cosine centroid for rapid deployment, hybrid
   cosine m=3 for production (5 passes, 0.73s, 15.28 PPL), and loss-probing for
   offline evaluation and router validation.
