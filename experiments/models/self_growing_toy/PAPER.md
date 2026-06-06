# Self-Growing Model: Proof Verification Report

## Theorem

**Claim (ReLoRA-based growth):** Sequential adapter promotion (train LoRA adapter,
extract delta, promote into base, repeat) is equivalent to gradient accumulation
(ReLoRA, Lialin et al. 2307.05695). Under this equivalence, a random-init model
should grow monotonically through adapter promotion, with each promotion improving
the base on the promoted domain while preserving prior domains.

## Predictions vs Measurements

| Prediction (from MATH.md) | Measured | Match? |
|----------------------------|----------|--------|
| P1: Each promotion improves its domain | ALL 5 improved (+0.71 to +2.00) | YES |
| P2: Mean loss improves monotonically | FAIL at promotion 3 (-0.074) | NO |
| P3: 5th adapter trains no slower than 1st | 0.94x (faster) | YES |
| P4: Grown/baseline ratio < 3.0 | 3.23x | NO (marginal) |
| P5: Final grown >> random init (>50%) | 19.8% improvement | NO |

**Overall: 2/5 predictions match, 3/5 fail. Experiment KILLED on K841, K843.**

## Hypothesis

Sequential LoRA adapter promotion from random init produces a monotonically improving
base model through gradient accumulation equivalence (ReLoRA).

**Status: KILLED.** The gradient accumulation equivalence holds directionally (each
promotion captures domain knowledge) but catastrophic interference between sequential
promotions limits accumulation to 19.8% improvement vs 75.2% for joint training.

## What This Experiment Does

Tests the ENTIRE Pierre self-growing architecture on a toy GPT model:
1. Start from RANDOM INIT (d=64, 4 layers, 4 heads, 215K params)
2. For each of 5 domains: train LoRA adapter (rank 4, 300 steps) -> SVD solidify
   -> promote delta into base weights
3. Compare grown base to jointly-trained baseline (same total compute budget: 1500 steps)

Type: frontier extension. Extends ReLoRA (proven for continuous training with periodic
merge) to sequential multi-domain promotion from random init.

## Key References

- ReLoRA (Lialin et al., 2307.05695): Periodic LoRA merging equivalent to gradient accumulation
- Davis-Kahan sin-theta theorem: Subspace rotation bounded by perturbation norm
- Finding #329: SVD composition fails for multi-adapter averaging (different mechanism)
- Finding #326: SVD improvement is magnitude reduction, not directional selection

## Empirical Results

### Kill Criteria

| Criterion | Result | Detail |
|-----------|--------|--------|
| K841: No promotion degrades base | **FAIL** | Promotion 3 (repeat) causes -0.074 mean degradation |
| K842: 5th adapter not slower than 1st | **PASS** | 0.94x ratio, loss@50 improves 4.62->3.98 |
| K843: Grown < 3x worse than baseline | **FAIL** | 3.23x ratio (threshold 3.0x) |

### Loss Trajectory (base quality after each promotion)

```
Step    arithmetic   reverse    repeat      sort     parity      Mean
init        5.026     4.946     4.989     4.921     5.038     4.984
+arith      3.788     5.077     4.771     5.053     5.009     4.740
+rev        4.416     3.831     4.201     3.857     4.675     4.196
+repeat     4.386     4.434     3.023     4.403     5.103     4.270
+sort       4.552     3.701     3.784     3.688     4.708     4.087
+parity     4.740     4.107     4.440     3.984     2.707     3.996
```

**Pattern: sawtooth.** Each promotion improves its own domain by ~1.0 but damages
2-3 other domains by ~0.3-0.7. The last-promoted domain (parity) retains most
benefit (2.71 vs init 5.04) because nothing overwrites it.

### Domain-Level Results

| Domain | Random Init | Grown | Jointly Trained | Grown/Joint |
|--------|-------------|-------|-----------------|-------------|
| arithmetic | 5.026 | 4.740 | 1.747 | 2.71x |
| reverse | 4.946 | 4.107 | 1.628 | 2.52x |
| repeat | 4.989 | 4.440 | 0.661 | 6.72x |
| sort | 4.921 | 3.984 | 1.449 | 2.75x |
| parity | 5.038 | 2.707 | 0.705 | 3.84x |
| **Mean** | **4.984** | **3.996** | **1.238** | **3.23x** |

The grown model improves 19.8% over random init. The jointly-trained baseline
improves 75.2%.

### Training Speed (K842)

| Adapter | Time | Final Loss | Loss@50 |
|---------|------|-----------|---------|
| 1st (arithmetic) | 1.7s | 3.741 | 4.621 |
| 2nd (reverse) | 1.7s | 3.745 | 4.275 |
| 3rd (repeat) | 1.8s | 3.040 | 3.873 |
| 4th (sort) | 1.6s | 3.720 | 3.013 |
| 5th (parity) | 1.6s | 2.703 | 3.979 |

Training speed is constant (1.7s avg). Convergence at step 50 improves
(4.62 -> 3.98), confirming the promoted base provides better features.

### Ablation: 500 steps per adapter (2500 baseline)

Same pattern with worse results: 3.39x ratio (vs 3.23x at 300 steps).
More training per adapter = larger deltas = more cross-domain interference.
Confirms the problem is interference magnitude, not training budget.

## Root Cause Analysis

The failure has a clear mathematical explanation:

**1. Catastrophic interference from sequential rank-4 updates.**
Each adapter's delta modifies the same weight matrices. With rank 4 at d=64,
each delta touches a 4/64 = 6.25% subspace. But these subspaces overlap
because the LoRA A matrices are random (not orthogonal). After K=5 promotions,
the accumulated perturbation mixes all 5 domain signals, and the last promotion
has the strongest residual effect.

**2. No interference protection during promotion.**
Unlike COMPOSITION (where Grassmannian A-matrices provide 17x decorrelation),
promotion uses random A-matrices for each adapter. The deltas are not guaranteed
to be orthogonal, so each promotion partially overwrites previous ones.

**3. Random init is a terrible starting point.**
The jointly-trained baseline starts from random init too, but trains ALL weights
(full rank) simultaneously. The grown model only updates rank-4 projections
sequentially. A rank-4 adapter cannot learn as much as a full-rank update.

## Implications for Pierre

**Self-growing from random init is not viable with naive sequential promotion.**
However, the mechanism WORKS in the sense that:
- Each promotion captures domain knowledge (P1 PASS)
- Training speed doesn't degrade (K842 PASS)
- The base does improve overall (19.8% from random init)

**What would make it work:**
1. **Start from a pre-trained base** (not random init). ReLoRA was validated on
   models that already had useful representations. The adapters only need to add
   domain-specific refinements, not learn general features from scratch.
2. **Orthogonal promotion** -- use Grassmannian A-matrices for each promotion to
   minimize cross-domain interference (same mechanism that makes composition work).
3. **Elastic Weight Consolidation (EWC)** -- protect important weights from being
   overwritten by computing Fisher information after each promotion.
4. **More promotions with smaller deltas** -- reduce scale to limit interference
   per promotion, accumulate more gradually.

## Limitations

1. **Toy scale (d=64):** At d=2560, the rank-4 subspace is 0.16% of the weight
   space (vs 6.25% at d=64). Interference should be much lower at scale.
2. **Character-level tasks:** Real NLP tasks may have different domain separation
   characteristics than arithmetic/reverse/repeat/sort/parity.
3. **No pre-trained base:** The ReLoRA equivalence was proven for fine-tuning,
   not training from scratch. Starting from random init violates the implicit
   assumption that the base already has useful features.
4. **Only 5 domains, 300 steps each.** More domains with more training would
   test the scaling behavior.

## What Would Kill This at Scale

Even with fixes, self-growing fails if:
- Interference grows faster than knowledge accumulation at large d
- The ReLoRA equivalence breaks down for domain-specialized (vs continuous) training
- The grown base cannot match 50% of jointly-trained quality after 20+ promotions
