# Dynamic Adapter Addition: Mathematical Foundations

## 1. Mechanism Definition

### N+1 Adapter Composition

Given N trained LoRA adapters {(A_i, B_i)}_{i=1}^N with routing heads {h_i}_{i=1}^N,
"hot-adding" adapter N+1 means:

1. Train adapter (A_{N+1}, B_{N+1}) on new domain data
2. Train routing head h_{N+1} on hidden states from new domain vs all existing domains
3. At inference, route using all N+1 heads with NO retraining of h_1...h_N

The composed weight update for input x is:

  delta_W(x) = sum_{i in top_k(S(x))} w_i(x) * B_i @ A_i

where S_i(x) = sigmoid(h_i(pool(H(x)))) is the routing score from head i,
top_k selects the k highest-scoring adapters, and w_i(x) = S_i / sum_j S_j
normalizes selected scores.

### Routing Head Architecture

Each head h_i: R^d -> [0,1] is a 2-layer MLP:
  h_i(z) = sigma(W_2 * relu(W_1 * z + b_1) + b_2)

where z = mean_pool(H(x)) in R^d is the mean-pooled hidden state from the base model,
W_1 in R^{d x h}, W_2 in R^{h x 1}, h=32.

Parameters per head: d*h + h + h*1 + 1 = d*h + 2h + 1.
At d=2560, h=32: 81,953 params (0.38% of one adapter's 21.6M params).

## 2. Why Hot-Addition Should Work

**Orthogonality guarantee:** With Grassmannian/random A matrices at d=2560, r=16,
new adapter A_{N+1} is near-orthogonal to all existing A_i by the JL lemma:
  E[|cos(vec(B_{N+1}@A_{N+1}), vec(B_i@A_i))|] ~ O(1/sqrt(d)) = O(0.02)

Prior result: mean |cos| = 0.001 at N=5, 40x below 0.05 threshold.
At N=6, expected max |cos| increase is negligible (Welch bound still far from d^2/r^2 = 25,600 capacity).

**Routing head independence:** Each head h_i is trained independently on binary
classification (own domain vs rest). Adding h_{N+1} does NOT change the decision
boundaries of h_1...h_N because:
- Heads share no parameters
- Hidden states come from frozen base model (no adapter applied during routing)
- Each head's training data is unchanged

The only interaction is through the top-k selection: h_{N+1} can "steal" a top-k
slot from some h_i. But if h_{N+1} fires correctly (high on its domain, low on others),
this only happens on domain N+1 inputs where h_{N+1} SHOULD be selected.

**Composition quality:** Under 1/N uniform composition at N+1:
- Each existing adapter's effective scale drops from 1/N to 1/(N+1)
- This dilutes existing adapter contributions by factor N/(N+1) = 5/6 = 0.833
- BUT with top-k routing, only k=2 adapters are active, so scale is ~1/2 regardless of N
- Therefore routed composition is N-invariant in effective scale

## 3. What Breaks It

**Cross-domain confusion:** If h_{N+1} fires on domains 1...N (false positives),
it displaces correct adapters from top-k selection, degrading quality.
Kill condition: routing accuracy of h_{N+1} < 70%.
Prior K1 result: 95% accuracy, so 25% margin.

**Adapter interference at high N:** If B_{N+1} has high cosine with some B_i despite
A-orthogonality, the effective delta sum can degrade. But at mean |cos|=0.001, this
requires N >> d^2/r^2 = 25,600 to matter.

**Distribution shift in hidden states:** If the 6th domain is very different from
the original 5, the base model's hidden state distribution may shift enough that
existing routing heads see OOD inputs and make errors. Mitigated by using a
domain (science) that shares vocabulary with existing domains.

## 4. Assumptions

1. **Base model frozen:** Hidden states for routing come from the frozen base.
   If the base were fine-tuned, routing heads would need retraining.
   Justified by: all prior experiments freeze the base.

2. **Domain separability:** New domain (science) is distinguishable from existing
   5 domains in hidden state space. Justified by: prior routing heads achieve
   95-100% accuracy on 5 domains.

3. **Small N regime:** N=6 is well within capacity (25,600). At N>>100,
   orthogonality guarantees weaken. Justified by: N=25 scaling experiment
   showed gamma=0.982.

## 5. Complexity Analysis

Adding one adapter: O(TRAIN_ITERS * seq_len * d * r) for LoRA training
Adding one routing head: O(HEAD_STEPS * d * h) where h=32
Total overhead: ~2 minutes (adapter) + ~10 seconds (head)

Inference: one extra head forward per token: O(d * h) = O(81,920) FLOPs
At d=2560, this is 0.003% of one transformer layer forward pass.

## 6. Worked Example (N=5 -> N=6)

Prior N=5 results (bitnet_2b_real_composition):
- Base avg PPL: 8.69
- Individual avg PPL: 6.40 (26.3% improvement)
- Composed 1/N avg PPL: 7.96 (8.4% improvement)
- Mean |cos|: 0.001

Adding science adapter:
- Expected individual PPL: ~4-7 (similar to other domains)
- Expected |cos| with existing: ~0.001-0.003
- Expected routing accuracy: 85-100% (science is distinctive)
- Expected N=6 routed PPL: <= N=5 routed PPL on original 5 domains
  (science head should NOT fire on non-science inputs)

## 7. Connection to Architecture

This experiment validates the "plug-and-play" promise from VISION.md:
"add/remove expert = pointer change." Specifically:
- No retraining of existing components
- New adapter carries its own routing head
- Composition quality is maintained or improved

Production reference (DeepSeek-V3): Uses auxiliary-loss-free load balancing
across 256 experts. Our approach is simpler (per-adapter binary heads) but
tests the same principle: adding experts should not degrade existing ones.
