# Meta-Scaffold: Mathematical Foundations

## Notation

| Symbol | Meaning | Dimensions |
|--------|---------|------------|
| W | Scaffold weight matrix | (d_out, d_in) |
| A_i | LoRA low-rank factor A for adapter i | (d_in, r) |
| B_i | LoRA low-rank factor B for adapter i | (r, d_out) |
| Delta_i | LoRA update: x @ A_i @ B_i * alpha | (d_in, d_out) |
| D | Set of domains | |D| = 5 |
| K | Inner loop steps | 50 |
| M | Outer loop (meta) steps | 100 |
| eta_in | Inner loop learning rate | 1e-3 |
| eta_out | Outer loop learning rate | 1e-4 |
| lambda | Composition penalty weight | 0.5 |
| N | Number of adapters composed | 5 |

## Bilevel Optimization

### Objective

The meta-scaffold objective is a bilevel optimization:

```
min_W  L_meta(W) = (1/|D|) * sum_{d in D} L_d(W, theta_d*(W)) + lambda * C(theta_1*(W), ..., theta_D*(W))
```

where:
- `theta_d*(W) = argmin_{theta_d} L_d(W, theta_d)` is the optimal adapter for domain d given scaffold W
- `L_d(W, theta_d)` is the domain loss with scaffold W and adapter theta_d
- `C(...)` is the composition penalty (pairwise adapter cosine similarity)

### Inner Loop (Adapter Training)

For each domain d, train adapter parameters theta_d = {A_d, B_d} for K steps:

```
theta_d^{k+1} = theta_d^k - eta_in * grad_{theta_d} L_d(W, theta_d^k)
```

The inner loop produces adapted parameters theta_d^K(W) as a function of W.

### Outer Loop (FOMAML)

First-order MAML approximation: ignore second-order terms in d(theta_d^K)/dW.

At each meta-step m:
1. Sample D' subset D domains
2. For each d in D': run inner loop K steps, get theta_d^K
3. Compute meta-loss with mean adapter (simulating composition):
   ```
   theta_mean = (1/|D'|) * sum_{d in D'} theta_d^K
   ```
4. Compute gradient of outer loss w.r.t. W:
   ```
   W^{m+1} = W^m - eta_out * grad_W L_outer(W, theta_mean)
   ```

### Composition Penalty

Pairwise cosine similarity between flattened adapter vectors:

```
C = (1 / (|D'| choose 2)) * sum_{i<j} |cos(flatten(theta_i), flatten(theta_j))|
```

where `cos(u, v) = (u . v) / (||u|| * ||v||)`.

## Computational Cost

### Per Meta-Step
- Inner loop: |D'| * K forward-backward passes = 3 * 50 = 150
- Composition penalty: O(|D'|^2 * P) where P = total adapter params
- Outer gradient: 1 forward-backward pass per sampled domain = 3
- **Total per meta-step: ~153 forward-backward passes**

### Full Meta-Training
- M meta-steps * 153 passes = 15,300 forward-backward passes
- Plus 2000-step warm-start pretraining
- **Total: ~17,300 forward-backward passes**

### Comparison to Standard
- Standard: 2000 pretrain steps (batch=4) = 2000 passes
- Meta adds 8.65x more compute for the scaffold optimization

### Memory
- Must store |D'| adapter parameter sets simultaneously
- Per adapter: 2 * n_layers * n_targets * r * d = 2 * 6 * 7 * 16 * 256 = 344,064 params
- |D'| = 3 adapters: ~1M extra parameters in memory
- Scaffold params: 6.4M
- **Peak memory: ~8.5M parameters** (manageable at micro scale)

## Key Assumption: FOMAML Sufficiency

FOMAML drops the second-order term:

```
d theta_d^K / dW = product_{k=0}^{K-1} (I - eta_in * H_k)
```

where H_k is the Hessian of L_d at step k. This product accumulates over K=50 steps.
With K=50, the dropped term can be substantial, meaning FOMAML may not capture the
true meta-gradient direction.

**Risk**: If the inner loop landscape is highly non-linear in W, FOMAML will provide
noisy/incorrect outer gradients, preventing convergence. This is exactly what we observe.

## Worked Example (d=256, r=16, K=50, M=100)

**Inner loop (1 domain)**:
- Adapter params: 2 * 6 layers * 7 targets * 16 * 256 = 344,064
- 50 gradient steps on single-sequence batches
- Typical loss: 3.2 -> 2.8 (converges)

**Outer loop (1 meta-step)**:
- 3 domains sampled, each inner-looped for 50 steps
- Mean adapter computed (1/3 scaling)
- Outer gradient computed on 3 val sequences
- Scaffold update: W <- W - 1e-4 * grad

**Result**: Meta-loss oscillates around 2.83 +/- 0.10 without clear convergence trend.
The 1.2% reduction from first-10 to last-10 average is below the 5% convergence threshold.

## Why FOMAML Failed Here

1. **FOMAML gradient noise**: With K=50 inner steps, the FOMAML approximation loses
   significant information about how scaffold changes propagate through the inner loop.

2. **Scaffold degradation**: The outer loop updates destabilized the pretrained scaffold
   (base PPL 11-20 -> 59-170), destroying the language modeling quality built during
   pretraining. The meta-gradient does not preserve base quality.

3. **Ternary quantization amplifies damage**: Meta-scaffold PPL after ternary quantization
   is 12x worse than standard (86-381 vs 13-22). The FOMAML updates pushed weights
   into distributions that quantize poorly.

4. **Adapter resilience masks scaffold damage**: Despite the ruined scaffold, adapters
   trained on meta-scaffold still achieve reasonable PPL (15-16 vs 11-15 for standard),
   because LoRA adapters can partially compensate for base quality loss.
