# DUME: Training-Free Dynamic Upcycling of Expert Language Models

MLX implementation of [DUME](https://arxiv.org/abs/2603.29765) (Fani & Ersoy, 2025).

## Overview

DUME converts multiple domain-specialized dense expert models into a single Mixture-of-Experts (MoE) model without any training. It:

1. **MoErges** dense experts: averages all non-MLP parameters, keeps MLP blocks as separate experts
2. **Initializes routers** via closed-form ridge regression on hidden states with domain labels
3. **Supports incremental expert addition** by updating X^TX and X^TY matrices

## Structure

```
dume/
  configs/base.yaml     -- hyperparameters (lambda, top_k, etc.)
  src/
    model.py            -- DUMEMoEBlock, DUMEModel (MoE with top-k routing)
    router.py           -- RidgeRouter, RouterStatistics, solve_ridge
    merge.py            -- moerge(): average non-MLP, keep MLP as experts
```

## Key Equations

- **Ridge regression**: `W* = (X^TX + lambda I)^{-1} X^TY`
- **Incremental update**: `A_l += F^T F`, `b_l += F^T Y` (schedule-invariant)
- **Column normalization**: `W*[:,c] /= ||W*[:,c]||` (domain imbalance)
- **Routing**: `g(x; W_l) = Top-k(Softmax(W_l^T x))`

## Usage

```python
import mlx.core as mx
from pierre.merge.dume.src.merge import moerge
from pierre.merge.dume.src.router import RouterStatistics, solve_ridge
from pierre.merge.dume.src.model import DUMEMoEBlock

# 1. MoErge: average non-MLP params, keep MLPs separate
shared, expert_mlps = moerge([expert_0_params, expert_1_params, ...])

# 2. Accumulate router statistics
stats = RouterStatistics(hidden_dim=2048, num_experts=4)
for domain_id, dataset in enumerate(domain_datasets):
    for batch_hidden_states in dataset:
        stats.update(batch_hidden_states, domain_id)

# 3. Solve for optimal router weights
W_star = solve_ridge(stats, lam=0.1, column_normalize=True)

# 4. Assign to MoE block
moe_block = DUMEMoEBlock(hidden_dim=2048, intermediate_dim=8192, num_experts=4)
moe_block.router.weight = W_star
```

## Platform

Apple Silicon M5 Pro 48GB. MLX only.
