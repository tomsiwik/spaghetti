# Skip-List Routing under Composition: Mathematical Foundations

## Notation

All notation follows the parent skip_list_routing/MATH.md. Additional symbols
for the composition protocol:

| Symbol | Shape / Type | Definition |
|--------|-------------|------------|
| D | set | Set of domain indices {1, ..., M} (M=2 for binary split) |
| delta_m | weight dict | Expert weight delta after domain m finetuning |
| bar_delta | weight dict | Weight-averaged expert: (1/M) * sum_m delta_m |
| E_i^(m) | CapsuleGroup | Leaf expert i after domain m finetuning |
| E_i^avg | CapsuleGroup | Leaf expert i after weight averaging |
| E_{i,k}^avg | virtual | Coarse expert at level k (weight-avg of averaged children) |
| c_k^joint(x) | (B,T,1) | Confidence gate at level k under joint training |
| c_k^comp(x) | (B,T,1) | Confidence gate at level k after composition+calibration |

## Composition Protocol

The shared-base composition protocol applied to skip-list routing:

### Phase 1: Pretrain Base (300 steps)
Train full SkipListRoutingGPT on all domains jointly. This produces base weights
theta_base including:
- Embedding/attention weights (shared backbone)
- Leaf expert weights {E_i}_i=0..N-1
- Level routers {R_k}_k=0..L
- Confidence gates {g_k, b_k}_k=1..L

### Phase 2: Domain Finetuning (200 steps per domain)
For each domain m in D:
1. Load theta_base
2. Freeze: embeddings, attention, norms, LM head
3. Train: leaf experts, routers, and confidence gates
4. Save delta_m = {all skip_pool parameters after finetuning}

### Phase 3: Weight Averaging
For every expert parameter key:
```
bar_delta[key] = (1/M) * sum_{m=1}^{M} delta_m[key]
```

This averages:
- Leaf expert weights: E_i^avg = (1/M) * sum_m E_i^(m)
- Router weights: R_k^avg = (1/M) * sum_m R_k^(m)
- Confidence gates: g_k^avg = (1/M) * sum_m g_k^(m), b_k^avg = (1/M) * sum_m b_k^(m)

### Phase 4: Calibration (100 steps)
Freeze everything except routers and confidence gates. Train on mixed-domain data.
This re-learns:
- Which experts to route tokens to (per-level routers)
- How much weight to assign per level (confidence gates)

## Key Question: Coarse Expert Quality under Composition

The critical mechanism under composition is the coarse expert construction.
At level k > 0, coarse expert i is:

```
E_{i,k}^avg(x) = (1/2) * [E_{2i,k-1}^avg(x) + E_{2i+1,k-1}^avg(x)]
```

After weight averaging, each leaf expert E_i^avg is itself an average of
M domain-specific experts. The coarse expert at level k is then an average
of 2^k averaged experts -- a double averaging.

### Concern: Signal Dilution

If domain-specific experts specialize (E_i^(1) differs substantially from
E_i^(2)), then:
1. Weight averaging dilutes each leaf expert
2. Coarse averaging further dilutes across siblings
3. The coarsest level (averaging ALL experts) becomes maximally diluted

This could push confidence gates to assign more weight to Level 0 (where
individual, less-diluted experts can be selected), reducing the level-weight
concentration observed in single-domain training (60.6% above Level 0).

### Counter-argument: Calibration Restores Gates

The calibration phase re-trains confidence gates on mixed-domain data. If the
weight-averaged experts at coarse levels are still useful (providing reasonable
predictions for easy tokens), the gates can re-learn appropriate level weights.

The gate only needs to distinguish "easy tokens" (any expert works, coarse is
fine) from "hard tokens" (need specific expert selection at Level 0). Under
composition, the definition of "easy" changes but the mechanism is preserved.

## Composition Gap Analysis

The composition gap is defined as:
```
gap = (val_loss_composed - val_loss_joint) / val_loss_joint * 100%
```

For hierarchical routing (tree baseline), the composition gap was +0.17%
(vs flat's +0.26%). Both are well within the 5% kill threshold.

For skip-list routing, the composition gap has two components:
1. Expert quality gap (same as flat): weight-averaged experts vs jointly-trained
2. Level assignment gap (unique to skip-list): confidence gates must re-learn
   level weights appropriate for composed experts

The total gap should be bounded by:
```
gap_skip <= gap_flat + gap_level_assignment
```

where gap_level_assignment is the additional cost of re-calibrating the
confidence gates.

## Kill Criteria Formalization

### KC1: Composition Gap
```
gap_skip - gap_flat > 3%  =>  KILL
```

This tests whether the additional hierarchical structure (confidence gates,
multi-level routing) introduces composition-specific degradation beyond what
flat routing suffers.

### KC2: Level-Weight Collapse

Define weight_above_L0 = sum_{k=1}^{L} mean(w_k(x)) over validation tokens.

```
weight_above_L0 < 10%  =>  KILL (collapse to Level 0 dominance)
```

Also check for uniform collapse:
```
max(w_k) - min(w_k) < 5pp for all k  =>  KILL (uniform = no adaptive routing)
```

Reference: single-domain weight_above_L0 = 60.6%.

## Worked Example

Setup: N=8 experts, L=3 levels, M=2 domains (a-m vs n-z).

After composition:
- 8 leaf experts, each averaging 2 domain-specific experts
- 4 Level-1 coarse experts (each averaging 2 averaged leaves = 4 original)
- 2 Level-2 coarse experts (each averaging 4 averaged leaves = 8 original)
- 1 Level-3 coarse expert (averaging all 8 averaged leaves = 16 original)

Calibration trains: 4 routers (sizes 8, 4, 2, 1) + 3 confidence gates.
Router parameters: 64*(8+4+2+1) = 960.
Gate parameters: 3*(64+1) = 195.
Total calibration parameters per layer: 1155.
Across 4 layers: 4620 parameters (2.2% of 206,732 total).

This small calibration budget (2.2% of model) must re-learn both expert
selection AND level assignment -- a harder task than flat routing calibration
(which only re-learns expert selection, ~512 params/layer = 0.25%).

The higher calibration parameter count (4.5x vs flat) partially compensates
for the harder task.
