# Gap-as-Signal at N>2: Mathematical Foundations

## Notation

| Symbol | Shape/Type | Definition |
|--------|-----------|------------|
| W_base | (d, d') | Frozen base model weight matrix |
| Delta_i | (d, d') | LoRA delta for expert i: (alpha/r) * A_i @ B_i |
| N | scalar | Number of experts in the pool |
| k | scalar | top_k experts selected per token |
| C(N,k) | scalar | Number of possible expert subsets = N! / (k!(N-k)!) |
| cos(i,j) | scalar | Pairwise cosine between flattened Delta_i, Delta_j |
| G_CE | scalar | Cross-entropy gap between composed and joint models |
| G_KL | scalar | KL divergence gap |
| Q | scalar | Quality gap: (avg_val - joint_val) / joint_val * 100 |
| S_acc | scalar | Selection accuracy: fraction of tokens where domain expert is in top_k |

## The Selection Problem

### N=2, top_k=2 (proven experiment)

With N=2 experts and top_k=2, every token activates BOTH experts. The router
learns only the mixing weight w in [0,1]:

```
output = w * expert_0(x) + (1-w) * expert_1(x)
```

This is a 1-dimensional optimization problem. C(2,2) = 1 possible subset.

### N=4, top_k=2 (this experiment)

With N=4 experts and top_k=2, the router must:
1. SELECT which 2 of 4 experts to activate: C(4,2) = 6 possible pairs
2. LEARN mixing weights between the selected pair

This is a 6-way discrete selection + continuous mixing problem.

### General case: N experts, top_k=k

```
C(N,k) = N! / (k! * (N-k)!)
```

| N | k | Subsets | Problem type |
|---|---|---------|-------------|
| 2 | 2 | 1 | Mixing only |
| 4 | 2 | 6 | Selection + mixing |
| 8 | 2 | 28 | Selection + mixing |
| 16| 2 | 120 | Selection + mixing |

The selection problem grows combinatorially with N.

## Hypothesis Extension

### Claim (Gap-as-Signal for Selection)

If the gap-as-signal hypothesis holds at N=2 (mixing only), it should also
hold at N>2 (selection + mixing) because the same mechanism applies:

1. Orthogonal experts produce distinguishable outputs per token
2. The router gradient signal is proportional to the output difference
3. Selection requires even STRONGER gradient signals than mixing

### Prediction 1: Gap-quality correlation persists

```
r^2(G_CE, Q) >= 0.3 at N=4, top_k=2
```

This is the same kill criterion as N=2 but in the harder selection regime.

### Prediction 2: Selection accuracy improves with orthogonality

When experts are orthogonal, each expert's delta modifies a unique subspace.
Token x that requires subspace i should produce loss_i(x) << loss_j(x) for
j != i, giving the router a clear signal to select expert i.

When experts are correlated, loss_i(x) ~ loss_j(x) for all j, and the router
has no gradient signal to learn selection.

### Expected selection accuracy

Random selection at N=4, top_k=2: P(domain expert in top_k) = k/N = 2/4 = 0.50.

If the router learns domain-specific selection, we expect accuracy > 0.50
for orthogonal experts.

## Projection Method for N>2

For N=2, Gram-Schmidt projection creates target cosine between a single pair.
For N>2, we need controlled pairwise cosines across ALL pairs.

### Sequential projection

Keep expert 0 fixed. For each subsequent expert i:

**Case cos=0 (orthogonal):** Full Gram-Schmidt:
```
v_i = b_i - sum_{j<i} <b_i, e_j> * e_j
```
where e_j = result_j / ||result_j|| are the orthonormalized previous experts.
Then rescale: v_i = v_i / ||v_i|| * ||b_i||.

**Case cos=c > 0 (correlated):** Project toward mean direction:
```
mean_dir = (1/i) * sum_{j<i} result_j
a_hat = mean_dir / ||mean_dir||
b_perp = b_i - <b_i, a_hat> * a_hat
b_perp_hat = b_perp / ||b_perp||
v_i = c * ||b_i|| * a_hat + sqrt(1-c^2) * ||b_i|| * b_perp_hat
```

This achieves approximate target cosine between expert i and the
average of all previous experts. Actual pairwise cosines deviate
from the target but maintain the monotonic trend.

### Actual cosine structure

For target_cos=0: all pairwise cosines are exactly 0 (Gram-Schmidt guarantee).
For target_cos=c>0: pairwise cosines cluster around c but are not uniform.
The mean pairwise cosine tracks the target cosine monotonically.

## Computational Cost

Training: O(N * finetune_steps * n_layer * d^2 * B * T) per seed.
Projection: O(N^2 * D) where D = total delta dimension.
Calibration: O(cal_steps * N * n_layer * d^2 * B * T).
Selection measurement: O(n_batches * n_layers * N * d^2 * B * T).

At micro scale (d=64, N=4, T=32, B=32):
- Training: ~4x the N=2 experiment
- Total: ~2.5 min per seed, ~7.5 min for 3 seeds

## Worked Example

d=64, n_head=4, n_layer=4, r=8, N=4 experts, top_k=2:

1. Delta dimension per expert: D = 4 * 2 * 64 * 256 = 131,072
2. Natural pairwise cosines: ~0.01-0.05 (near-orthogonal, consistent with r/sqrt(D))
3. C(4,2) = 6 possible expert pair selections per token
4. Random selection accuracy: 2/4 = 0.50
5. At target cos=0.0:
   - CE gap ~ 0.017, KL gap ~ 0.043
   - Quality: +3.9% vs joint
6. At target cos=0.9:
   - CE gap ~ 0.090, KL gap ~ 0.141
   - Quality: +14.0% vs joint (3.6x worse)
7. Gap-quality correlation: r^2 = 0.82

## Assumptions

1. **Fixed N=4:** We test one pool size. The scaling to N=8, N=16 is
   predicted but untested.

2. **Quaternary domain split:** The 4 domains (a-f, g-m, n-s, t-z) are
   structurally similar (all character-level names). Domain boundaries
   are arbitrary. Real macro-scale domains would have stronger separation.

3. **Selection accuracy baseline:** At N=4, top_k=2, random accuracy is
   exactly 0.50. This leaves little room to measure improvement above
   chance at micro scale. N=8 would have baseline 0.25, giving more room.

4. **Router capacity:** A linear router (d -> N) may lack capacity to
   learn domain-specific selection at d=64, N=4. The router has only
   64*4=256 parameters per layer, which may be insufficient for a
   4-way selection problem.
