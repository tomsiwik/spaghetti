# Exp 4: Scaling Composition to N=5 Experts

## Research Question

Does the shared-base composition protocol scale from N=2 to N=5 domains?
Specifically: do capsule group subspaces remain approximately orthogonal,
and does composition quality stay within 5% of joint training?

## Variables

| Symbol | Shape | Description |
|--------|-------|-------------|
| D | scalar | Number of domains (D=5: a-e, f-j, k-o, p-t, u-z) |
| G_d | scalar | Capsule groups per domain (G_d=4) |
| G | scalar | Total composed groups = D × G_d = 20 |
| k | scalar | Top-k active groups = D × k_d = 5×2 = 10 |
| n_caps | scalar | Capsules per group = 64 |
| d | scalar | Embedding dimension = 64 |
| W_base | varies | Shared pretrained base weights (attention, embeddings) |
| W_d | (n_layer, G_d, ...) | Capsule group weights fine-tuned on domain d |
| Δ_d | vector | Weight delta = flatten(W_d - W_base_capsules) per layer |
| W_r | (d, G) | Composed router weights per layer |

## Protocol

### Phase 1: Pretrain Shared Base
Train CapsuleMoEGPT(G=4, k=2) on all domains jointly for 300 steps.
This establishes the shared attention/embedding space.

### Phase 2: Domain Fine-tuning
For each domain d ∈ {1..5}:
- Copy base model
- Freeze all parameters except capsule groups
- Fine-tune on domain d for 300 steps
- Extract fine-tuned capsule groups: groups_d[layer][group]

### Phase 3: Composition
- Create CapsuleMoEGPT(G=20, k=10) using shared base attention/embeddings
- Slot in domain capsule groups: groups[0:4]=domain1, groups[4:8]=domain2, ..., groups[16:20]=domain5
- Router randomly initialized

### Phase 4: Calibration
- Freeze all except router
- Train router on data from all 5 domains (rotating batches), 200 steps
- This is 2× the N=2 calibration (100 steps) to test linear scaling

### Phase 5: Baselines
- Joint training: CapsuleMoEGPT(G=20, k=10) trained on all 5 domains, 5×300 steps
- Task arithmetic: average all 5 deltas (Δ_composed = mean(Δ_1..5))
- Concatenation + uniform: composed model, w_g = 1/G

## Orthogonality Analysis

For each layer l:
- Compute Δ_d^l = flatten(groups_d^l) - flatten(groups_base^l), shape (G_d × n_caps × 2 × d,)
- For all (D choose 2) = 10 pairs (i,j):
  - cosine_sim(Δ_i^l, Δ_j^l)
- Report mean and max pairwise cosine similarity

At N=2, prior experiments found cosine ≈ 0.000 (natural orthogonality).
Hypothesis: orthogonality holds at N=5 because d=64 provides enough
dimensions for 5 independent subspaces. (Each Δ_d lives in a subspace
of dim ≈ G_d × n_caps = 256 within the full parameter space.)

## Parameter Count

### Per-domain model (during fine-tuning):
Same as capsule_moe: ~203K total, only capsule groups trainable.
Capsule groups: 4 layers × 4 groups × 64 capsules × 2 × 64 = 131,072 trainable

### Composed model:
- Base (attention + embeddings): ~72K (frozen)
- Capsule groups: 4 layers × 20 groups × 64 capsules × 2 × 64 = 655,360
- Router: 4 layers × (64 × 20) = 5,120
- Total: ~732K
- Active per token: base + k×group = ~72K + 10 × 8,192 = ~154K

### Joint baseline:
Same architecture: CapsuleMoEGPT(G=20, k=10) = ~732K total

## Assumptions

1. **Character-level domains are separable enough at N=5**: The quintary split
   creates 5 overlapping character distributions. These may be even harder to
   distinguish than binary. This tests composition under adverse conditions.

2. **k/G ratio transfers**: We maintain k/G = 0.5 (10/20) as at N=2 (4/8).
   The sparse routing experiment validated k=2 for G=8; we assume the ratio
   transfers.

3. **Calibration scales linearly**: 200 steps for N=5 (vs 100 for N=2).
   If insufficient, we note the scaling factor needed.

4. **Shared base quality**: 300 steps of base pretraining is sufficient for
   5 domains. May need more steps as domain diversity increases.

## Falsification Criteria

| Criterion | Threshold | Kill? |
|-----------|-----------|-------|
| Composition+cal vs joint | >5% | KILL |
| Composition+cal vs N=2 result (-0.2%) | >3× worse (>0.6%) | CONCERN |
| Max pairwise cosine sim of deltas | >0.5 | CONCERN (note, not kill) |
| Calibration needs >400 steps | — | SCALING CONCERN |
| Any single domain >10% worse | — | PARTIAL KILL |

## Computational Cost

- 5 domain fine-tunings × 300 steps + 1 base pretrain × 300 steps + 1 calibration × 200 steps = 2,000 training steps
- Joint baseline: 5 × 300 = 1,500 steps
- 3 seeds × ~3,500 total steps = ~10,500 steps
- At ~100K tok/s: well under 5 minutes
