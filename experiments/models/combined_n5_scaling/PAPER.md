# Combined Parallel + Pure-Linear Composition at N=5 Domains: Research Digest

## Hypothesis

The combined parallel+pure-linear architecture (validated at N=2 with
+1.48% degradation) will NOT degrade composition quality by more than 8%
when scaled to N=5 domains.

**Falsifiable**: N=5 composition gap >8% for parallel+pure-linear architecture.

**Result: PASS.** Composition gap is +3.32% (threshold 8%, margin 2.4x).
Zero catastrophic failures across 6 runs. The architectural penalty from
parallel+pure-linear is actually *smaller* at N=5 than at N=2.

---

## What This Experiment Is

This is a **scaling validation** of the combined parallel+pure-linear
architecture from N=2 to N=5 domains. The N=2 experiment showed the
two modifications (parallel blocks + all-linear attention) compose with
approximately additive effects and +1.48% degradation vs the sequential
+hybrid baseline. The adversarial reviewer flagged that more domains
competing for capsule routing could amplify the interaction between
parallel execution and linear attention.

The experiment uses 5 domains from the quintary character split (a-e,
f-j, k-o, p-t, u-z) and compares two conditions:

- **seq_hybrid** (baseline): Sequential blocks + hybrid 3:1 attention
- **par_pure_linear** (test): Parallel blocks + all-GatedDeltaNet attention

Each condition runs the full composition protocol: pretrain shared base,
fine-tune per domain (attention frozen), compose by concatenation (20
groups, top-10), calibrate router (200 steps, round-robin), evaluate.

---

## Lineage in the Arena

```
gpt (dense baseline)
 |-- capsule_moe (routed capsule groups)
      |-- full_gdn_stack_capsule_moe (full GatedDeltaNet)
           |-- parallel_pure_linear_capsule_moe (N=2 validated, PROVEN)
                |-- combined_n5_scaling (N=5 test, THIS EXPERIMENT)
```

---

## Key References

**Exp parallel_pure_linear_combined (N=2)**: +1.48% degradation, effects
additive (interaction +0.31%), zero catastrophic failures across 20 runs.
5 seeds per condition.

**Exp prune_compose_n5 (N=5 pruning)**: Pre-composition pruning at N=5
matches compose-then-prune within +0.02%. Validates that composition
protocol works at N=5 for the standard relu_router architecture.

**Exp n5_identity_scaling**: Capsule identity Jaccard degrades gracefully
from 0.871 (N=2) to 0.792 (N=5). Pre-composition profiling remains viable.

---

## Experimental Design

Two conditions, 3 seeds each (seeds 0, 1, 2):

| Condition | Block Type | Attention | Layer Types |
|-----------|-----------|-----------|-------------|
| seq_hybrid | Sequential (norm1->attn->norm2->mlp) | 3:1 hybrid | LLL-F |
| par_pure_linear | Parallel (norm->attn+mlp) | All linear | LLLL |

Protocol per seed per condition:
1. Joint training baseline: 1500 steps on all 5 domains (round-robin)
2. Pretrain shared base: 300 steps on all data
3. Fine-tune per domain: 300 steps each, 5 domains (attention frozen)
4. Compose: concatenate all 5 domain groups (4->20 groups, k=2->10)
5. Calibrate router: 200 steps round-robin
6. Evaluate: per-domain validation loss

---

## Empirical Results

### Main Results (3 seeds: 0, 1, 2)

| Condition | Composed (mean) | Joint (mean) | Gap mean | Gap median | Gap std |
|-----------|----------------|-------------|----------|------------|---------|
| seq_hybrid | 0.4996 | 0.4871 | +2.57% | +2.73% | 0.33% |
| par_pure_linear | 0.5055 | 0.4893 | +3.32% | +3.06% | 1.41% |

### Kill Criterion

    par_pure_linear N=5 composition gap: +3.32%
    Threshold: >8%
    Result: PASS (margin 2.4x)

### Cross-Architecture Comparison

| Metric | Value |
|--------|-------|
| Composed loss degradation (par_pl vs seq_h) | +1.19% |
| Gap difference (par_pl - seq_h) | +0.75pp |

### N=2 vs N=5 Scaling

| Metric | N=2 | N=5 | Change |
|--------|-----|-----|--------|
| par_pure_linear gap | +0.96% | +3.32% | +2.36pp |
| seq_hybrid gap | -0.50% | +2.57% | +3.07pp |
| Cross-architecture degradation | +1.48% | +1.19% | **-0.29pp** |

The cross-architecture degradation (how much worse par_pure_linear is
compared to seq_hybrid) actually *decreased* from +1.48% at N=2 to
+1.19% at N=5. The architectural penalty does not amplify with domain
count. Both conditions degrade similarly going from N=2 to N=5 (+2.36pp
for par_pl vs +3.07pp for seq_h).

### Per-Seed Composition Gaps

| Seed | seq_hybrid | par_pure_linear |
|------|-----------|----------------|
| 0 | +2.79% | +2.06% |
| 1 | +2.73% | +3.06% |
| 2 | +2.19% | +4.83% |

Seed 2 of par_pure_linear shows the highest gap (+4.83%), still well
within the 8% threshold. The higher variance in par_pure_linear (std
1.41% vs 0.33%) is consistent with the N=2 finding that pure-linear
attention has slightly more seed-to-seed variability.

### Per-Domain Composition Gaps (par_pure_linear, seed-averaged)

| Domain | Joint | Composed | Gap |
|--------|-------|----------|-----|
| a-e | 0.4894 | 0.4982 | +1.79% |
| f-j | 0.4894 | 0.5077 | +3.74% |
| k-o | 0.4949 | 0.5069 | +2.42% |
| p-t | 0.5058 | 0.5240 | +3.59% |
| u-z | 0.4672 | 0.4910 | +5.09% |

The u-z domain shows the largest gap (+5.09%). This is the smallest
domain (7.4% of data in the quintary split) and consistently shows
higher composition gaps across experiments. The gap is not specific
to the par_pure_linear architecture.

Zero catastrophic failures (gap > 20%) across all 6 runs.

---

## Key Findings

### Finding 1: Composition Gap Within Threshold (+3.32%, margin 2.4x)

The par_pure_linear architecture composes at N=5 with +3.32% gap vs
joint training. This is well within the 8% threshold (2.4x margin).
The combined architecture remains composition-safe at practical domain
counts.

### Finding 2: Architectural Penalty Does Not Amplify with N

The cross-architecture degradation (par_pl vs seq_h) is +1.19% at N=5,
compared to +1.48% at N=2. The -0.29pp change suggests the parallel
+pure-linear penalty is N-independent or slightly N-beneficial. The
reviewer's concern that "more domains competing for capsule routing"
would amplify the interaction is empirically falsified.

### Finding 3: Both Conditions Degrade Similarly from N=2 to N=5

Both architectures show ~2.5-3pp increased gap going from N=2 to N=5:
- seq_hybrid: -0.50% -> +2.57% (+3.07pp)
- par_pure_linear: +0.96% -> +3.32% (+2.36pp)

This confirms that the N-scaling degradation is a property of the
composition protocol (more domains = more interference), not of the
block/attention architecture. The parallel+pure-linear modification
does not interact with this scaling.

### Finding 4: Variance Higher in par_pure_linear

par_pure_linear gap std = 1.41% vs seq_hybrid 0.33% (4x higher). This
is consistent with the N=2 finding. The pure-linear attention mechanism
has more seed sensitivity at N=5 (max gap 4.83% vs 2.79%). Still well
within threshold but worth monitoring at higher N.

### Finding 5: Per-Domain Gap Concentration in Small Domains

The u-z domain (smallest, 7.4% of data) shows the highest composition
gap (+5.09%). This is likely a calibration data quantity effect: with
200 steps round-robin, u-z gets only ~40 calibration batches from its
own data. Increasing calibration budget or using proportional sampling
could help.

---

## Micro-Scale Limitations

1. **3 seeds per condition.** The 8% kill threshold has 2.4x margin,
   so the verdict is robust. But the per-seed variance (std 1.41% for
   par_pl) means individual runs can reach ~5%. More seeds would better
   characterize the tail.

2. **Toy domains.** All 5 domains are character-level name generation
   split by first letter. Real-world domains (code, math, prose) have
   much more distinct representations, which could either help (cleaner
   routing separation) or hurt (more interference) composition.

3. **d=64, L=4, T=32.** The GatedDeltaNet state capacity (16x16) has
   not been stressed. At macro scale, 24+ all-linear layers with 256x256
   states might show cumulative saturation effects.

4. **Unequal domain sizes.** The quintary split ranges from 7.4% (u-z)
   to 32.7% (a-e). The per-domain gap analysis shows size-correlated
   degradation. Balanced domains might show lower maximum per-domain gap.

5. **200 calibration steps.** Scaled from 100 at N=2, but per-domain
   calibration budget (40 batches) is lower than at N=2 (50 batches).
   The standard capsule_moe N=5 experiment used 100 steps successfully,
   suggesting 200 is conservative.

---

## What Would Kill This

### At Micro Scale (tested)

- **N=5 gap >8%**: NOT KILLED. Gap is +3.32% (2.4x margin).

### At Macro Scale (untested)

- **GatedDeltaNet state saturation** at d_h=256+ over 24+ all-linear
  layers. A single full attention layer (the "scaffolding" the hybrid
  3:1 provides) might be essential at scale for periodic state refresh.

- **The 1.41% gap std growing with N.** If variance scales super-linearly,
  the tail risk at N=10+ could breach thresholds even if the mean is fine.

- **Per-domain gap exceeding tolerance.** The u-z domain at +5.09% is
  already showing stress. If a macro domain has similarly low data volume,
  its composition gap could be unacceptable.

- **True hardware parallelism changing the arithmetic.** At micro scale,
  "parallel" blocks execute sequentially on MLX. Real GPU stream
  parallelism could expose numerical differences from concurrent
  computation.

---

## Summary

The combined parallel+pure-linear architecture composes safely at N=5
domains. The composition gap (+3.32%) is well within the 8% kill
threshold. The architectural penalty from using parallel blocks with
all-linear attention does not amplify with domain count -- it is
actually slightly smaller at N=5 (+1.19%) than at N=2 (+1.48%). Both
architectures degrade similarly from N=2 to N=5 (~2.5-3pp), confirming
that the N-scaling effect is a composition protocol property, not an
architectural interaction. Zero catastrophic failures across 6 runs.

The simplified composition-safe block:
```
x_{l+1} = x_l + GDN(Norm(x_l)) + CapsulePool(Norm(x_l))
```
is now validated at both N=2 (5 seeds, +0.96%) and N=5 (3 seeds, +3.32%).
