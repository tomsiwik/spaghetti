# Parameter-Matched Monolithic Ablation: Mathematical Analysis

## Notation

| Symbol | Meaning | Value |
|--------|---------|-------|
| d | model hidden dimension | 2560 |
| L | number of transformer layers | 30 |
| M | target modules per layer | 7 (q,k,v,o,gate,up,down) |
| r_s | SOLE expert rank | 16 |
| N | number of SOLE experts | 5 |
| r_m | monolithic rank (matched) | 80 |

## Parameter Counting

### SOLE total parameters

Each LoRA adapter with rank r on a d x d linear layer has:
  P_lora(r) = d * r + r * d = 2 * d * r

For 7 target modules per layer, 30 layers:
  P_expert(r) = 7 * 30 * 2 * d * r = 420 * d * r

At r=16:
  P_expert(16) = 420 * 2560 * 16 = 17,203,200

Note: Some modules have different output dims (k_proj, v_proj use d/n_heads
for GQA). The actual count from the prior experiment is 21,626,880 per expert
(accounts for actual module dimensions in BitNet-2B-4T architecture).

SOLE total:
  P_SOLE = N * P_expert(r_s) = 5 * 21,626,880 = 108,134,400

### Rank-80 monolithic parameters

  P_mono(80) = P_expert(80) = 5 * P_expert(16) = 108,134,400

This is exact: rank scales linearly with parameter count, so r=80 = 5 * r=16.

The actual count will be 5 * 21,626,880 = 108,134,400, matching SOLE exactly.

## Why Specialization Should Beat Rank Scaling

### Information-theoretic argument

Consider the effective rank of the combined multi-domain gradient signal.
If domains are largely independent (supported by |cos| = 0.002 between
domain adapters), the gradient matrix G from union training has approximate
block-diagonal structure:

  G_union ~ diag(G_medical, G_code, G_math, G_legal, G_creative)

A rank-80 monolithic adapter must represent all 5 domain subspaces in a
single low-rank factorization W = BA where B in R^{d x 80}, A in R^{80 x d}.

The effective rank of the union gradient is at most 5 * r_eff(single_domain).
If each domain needs rank ~16 for adequate representation, the union needs
rank ~80. So rank-80 can in principle represent all domains.

However, the monolithic adapter must share its rank budget across domains
in proportion to training data frequency (uniform here: 20% each). The key
loss is that interference between domain gradients during training causes
the optimizer to find a COMPROMISE rather than the domain-optimal solution.

### SOLE avoids interference by construction

SOLE trains each domain independently. The resulting adapter Delta_i is
optimal for domain i (within the rank-r constraint). No gradient from domain
j ever perturbs the optimization trajectory for domain i.

At inference, routing selects Delta_i for queries from domain i, so the
model sees the domain-optimal adapter.

### Prediction

The monolithic r=80 should improve over monolithic r=16 (more capacity).
But it should still lose to SOLE routed on most domains because:

1. Training interference: gradient updates from domain j pull the shared
   subspace away from domain i's optimum.

2. Subspace competition: the 80-dim subspace must be carved into ~5 regions.
   Each domain gets effective rank ~16, same as SOLE -- but with interference.

3. The creative exception may persist or expand: domains that benefit from
   cross-domain transfer (creative writing benefits from code/math patterns)
   should still favor monolithic.

### Expected outcome

- SOLE wins 3-4 domains (specialization advantage)
- Monolithic r=80 wins 1-2 domains (cross-domain transfer advantage)
- Average gap narrows from -5.8% (vs r=16) to perhaps -2% to -4%
- Monolithic r=80 beats monolithic r=16 on all domains (more capacity)

### Kill scenario

If monolithic r=80 wins 3+ domains, it means the cross-domain transfer
benefit outweighs the specialization benefit. SOLE's value would be purely
operational (modularity, no forgetting, incremental updates), not quality.

## Computational Cost

Training: 2000 steps with rank-80 LoRA on 2560-dim model.
Per-step FLOPs dominated by LoRA forward/backward:
  Forward: O(B * S * d * r) per module = O(1 * 128 * 2560 * 80) = O(26M) per module
  7 modules * 30 layers = 210 modules -> O(5.5B) FLOPs per step
  vs rank-16: O(1.1B) FLOPs per step -> 5x slower per step

Expected runtime: ~1 hour (vs ~17 min for rank-16 monolithic).
