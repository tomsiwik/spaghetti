# LSH Capsule Routing: Research Digest (Revised)

## Hypothesis

Replacing learned softmax group routing with fixed random-projection
locality-sensitive hashing (LSH) will match softmax routing quality within
3%, while eliminating all routing parameters and enabling O(1) expert
selection at scale.

**Falsifiable**: If LSH routing quality is >3% worse than learned softmax
routing (with or without balance loss), or if LSH requires >4 hash tables
to match softmax quality (diminishing returns), the approach is dead.

---

## What This Model Is

`LSHCapsuleRoutingGPT` replaces the learned softmax router in each
CapsulePool with **fixed random-projection hash tables**. Each hash table
is a random Gaussian matrix R_t in R^{G x d} that maps tokens to expert
groups via argmax(R_t @ x). Multiple tables (T) vote on which experts to
select; top-k by vote count are activated.

### How It Works

1. **Hash computation**: For each of T hash tables, compute R_t @ x
   (a fixed random projection) and take argmax to select one expert.

2. **Vote accumulation**: Count how many tables vote for each expert.
   Top-k experts by vote count are selected. Ties are broken by
   accumulated projection scores.

3. **Weight computation**: Softmax over accumulated scores for selected
   experts gives data-dependent routing weights -- without any learned
   parameters.

4. **Output**: Weighted sum of selected CapsuleGroup outputs, identical
   to the softmax CapsulePool but with hash-based selection.

### Why It Exists

The learned softmax router costs O(N*d) parameters and O(N*d) FLOPs,
scaling linearly with the number of experts N. At our target scale
(N=100-1000 experts from independent contributors), routing becomes
a bottleneck:

- At N=256, the softmax router has 256*896 = 229K params PER LAYER
- Router calibration requires mixed-domain data for all N domains
- Adding a new expert requires recalibrating the entire router

LSH routing eliminates all three problems:
- Zero routing parameters regardless of N
- Zero calibration needed -- new experts just need hash keys
- Routing cost with fixed T is independent of N (with binary LSH)

The trade-off is routing quality: random projections are data-independent
and may not route as precisely as a learned router. This experiment
measures exactly that trade-off.

---

## Lineage in the Arena

```
gpt -> moe -> capsule_moe -> lsh_capsule_routing
                             (replaces learned softmax with random-projection LSH)
```

---

## Key References

**SimHash / Random-Projection LSH** (Charikar 2002): The foundational
result that random hyperplane projections preserve angular similarity.
P[h(x) = h(y)] = 1 - theta(x,y)/pi. Our work uses the multi-table
extension with argmax over G projections.

**Hash Layers** (NeurIPS 2021, Chen et al.): Replaces softmax routing
in MoE with hash-based routing. Competitive with Switch Transformer at
scale. Our work validates the principle at micro scale with capsule
groups instead of full FFN experts.

**PEER** (DeepMind 2024, He et al.): Scales to 1M+ experts using
product-key retrieval -- a structured form of hash-based routing.
Product keys decompose the d-dimensional lookup into two sqrt(d)-
dimensional lookups, achieving O(sqrt(N)) routing.

**Switch Transformer** (Fedus et al., 2021): Demonstrates that k=1
top-expert routing works at scale with large individual experts. Our
LSH approach extends to k>1 via multi-table voting.

---

## Empirical Results

### Controls and Experimental Design

Per adversarial review, we include four control conditions to ensure
a fair comparison:

| Config | Description | Routing | Balance Loss |
|--------|-------------|---------|-------------|
| softmax_k2 | Original CapsuleMoE baseline | Learned softmax, top-2 | 0.01 * balance_loss |
| softmax_no_bal | CapsuleMoE without balance loss | Learned softmax, top-2 | 0.0 (aux_loss=0) |
| uniform | CapsuleMoEUniform | All groups 1/G, no routing | 0.0 |
| lsh_T{1,2,4,8} | LSH routing with T hash tables | Fixed random projection | 0.0 (not learnable) |

All LSH models and the softmax_no_bal model train with aux_loss=0.0 to
remove the confound where different models optimize slightly different
loss functions. The softmax_k2 config retains its balance loss to show
the effect of balance loss on routing behavior.

### Single-Domain Quality (500 steps, 3 seeds)

| Config | Params | Val Loss (mean) | vs SM-noBal | vs SM-bal | Throughput |
|--------|--------|-----------------|-------------|-----------|------------|
| softmax_k2 | 204,160 | 0.5187 | -0.50% | baseline | 2,139K tok/s |
| softmax_no_bal | 204,160 | 0.5213 | baseline | +0.51% | 2,112K tok/s |
| uniform | 204,160 | 0.5169 | -0.85% | -0.35% | 2,564K tok/s |
| **LSH T=1** | **202,112** | **0.5168** | **-0.87%** | -0.36% | 1,949K tok/s |
| **LSH T=2** | **202,112** | **0.5144** | **-1.34%** | -0.84% | 1,902K tok/s |
| LSH T=4 | 202,112 | 0.5181 | -0.62% | -0.11% | 1,817K tok/s |
| LSH T=8 | 202,112 | 0.5196 | -0.33% | +0.18% | 1,576K tok/s |

Kill criterion 1 (LSH >3% worse than softmax): **PASSES** -- all LSH
configs are within 1.4% of the fair baseline (softmax_no_bal), and
all are slightly better (negative delta), not worse.

Kill criterion 2 (requires >4 tables to match): **PASSES** -- T=1
already matches (and slightly beats) the softmax baseline.

### Per-Seed Breakdown

| Seed | SM-bal | SM-noBal | Uniform | LSH T=1 | LSH T=2 | LSH T=4 | LSH T=8 |
|------|--------|----------|---------|---------|---------|---------|---------|
| 42 | 0.5116 | 0.5175 | 0.5090 | 0.5128 | 0.5101 | 0.5137 | 0.5121 |
| 123 | 0.5176 | 0.5124 | 0.5141 | 0.5104 | 0.5123 | 0.5114 | 0.5194 |
| 777 | 0.5269 | 0.5341 | 0.5276 | 0.5272 | 0.5207 | 0.5293 | 0.5274 |

### Paired Statistical Tests (vs softmax_no_bal, 3 seeds)

| Config | t-statistic | p-value (two-tailed) | Significant? |
|--------|-------------|---------------------|-------------|
| softmax_k2 | -0.666 | 0.574 | No |
| uniform | -1.404 | 0.295 | No |
| lsh_T1 | -3.137 | 0.088 | No (directional) |
| lsh_T2 | -1.810 | 0.212 | No |
| lsh_T4 | -2.825 | 0.106 | No (directional) |
| lsh_T8 | -0.394 | 0.731 | No |

**No configuration achieves statistical significance (p<0.05) against
softmax_no_bal with only 3 seeds.** The best directional evidence is
LSH T=1 (p=0.088) and LSH T=4 (p=0.106). With 3 seeds, these results
provide directional evidence only, not statistical proof.

### Key Finding: Routing Quality Is Irrelevant at Micro Scale

The most important result is that **uniform routing (no routing at all)
performs within 0.85% of softmax_no_bal**, and this difference is not
statistically significant (p=0.295). All routing variants -- uniform,
LSH at any T, softmax with or without balance loss -- produce
statistically indistinguishable results at G=8 with homogeneous
character-level data.

This means the original finding that "LSH beats softmax" was an artifact
of the balance loss handicapping the softmax router, combined with noise.
When the comparison is fair, all routing methods match within noise.

### Routing Behavior Comparison

| Metric | SM (balance) | SM (no balance) | LSH T=4 |
|--------|-------------|-----------------|---------|
| Normalized routing entropy | 0.73-0.86 | 0.10-0.33 | 0.13-0.26 |
| Expert utilization (all used?) | Yes (1.000) | Yes (1.000) | No (0.007-0.907) |
| Routing parameters per layer | 512 | 512 | 0 |

**Removing the balance loss produces sharp softmax routing** (entropy
0.10-0.33), comparable to LSH sharpness (0.13-0.26). The balance loss
was the cause of the near-uniform softmax routing, not a limitation of
softmax routing itself. Without balance loss, the softmax router learns
sharp expert preferences.

### Throughput

| Config | tok/s | vs softmax_k2 |
|--------|-------|--------------|
| uniform | 2,564K | +20% |
| softmax_k2 | 2,139K | baseline |
| softmax_no_bal | 2,112K | -1% |
| LSH T=1 | 1,949K | -9% |
| LSH T=2 | 1,902K | -11% |
| LSH T=4 | 1,817K | -15% |
| LSH T=8 | 1,576K | -26% |

LSH routing is slower than softmax at micro scale because the current
implementation loops over T tables, computing T separate matrix-vector
products. The uniform baseline is fastest (no routing computation at all).

---

## Parameter Comparison

| Component | Softmax (G=8) | LSH (G=8, T=4) | Difference |
|-----------|--------------|-----------------|------------|
| Routing params/layer | 512 | 0 | -100% |
| Total routing params | 2,048 | 0 | -100% |
| Capsule params | 196,608 | 196,608 | 0% |
| **Total model params** | **204,160** | **202,112** | **-1.0%** |

At macro scale (d=896, G=256, 24 layers):
- Softmax routing: 256 * 896 * 24 = 5.5M params
- LSH routing: 0 params
- Savings: 5.5M params (~1.5% of a 370M model, ~0.8% of 671M)

---

## FLOP Scaling Comparison

The actual routing FLOP cost for the current implementation is:

| Method | Routing FLOPs per token per layer |
|--------|----------------------------------|
| Softmax | O(G * d) = O(N * d) |
| LSH (T tables) | O(T * G * d) = O(T * N * d) |

At micro scale (G=8), LSH costs T times more than softmax -- a
disadvantage. **The cost savings at large N claimed in the original
version of this paper were based on hash-table pre-indexing (O(T*d)
with O(1) bucket lookup), which is not what the current implementation
does.** The actual implementation computes R_t @ x for all T tables,
each projecting against all G groups, giving O(T * G * d) per token.

To achieve the O(T*d) scaling independent of N, one would need:
- Binary LSH hash functions (e.g., sign(R_t @ x) producing bit vectors)
- Pre-indexed hash tables mapping bit patterns to expert sets
- Bucket lookup instead of argmax over all G projection scores

This is feasible but not implemented. The current experiment tests
routing QUALITY (which validates), not routing SPEED at large N
(which would require binary LSH).

---

## Why Results Are Indistinguishable at Micro Scale

Three factors explain why all routing variants produce the same quality:

1. **G=8 is too small for routing to matter.** With only 8 expert groups
   and k=2 selected, even random routing activates 25% of groups per
   token. The per-group capacity (32 capsules) is sufficient for each
   group to handle any input. There is no cost to routing "wrongly."

2. **Homogeneous data.** Character-level names have minimal structure
   that would benefit from expert specialization. There are no natural
   domains to separate.

3. **Short training (500 steps).** Routing patterns may differentiate
   with longer training as groups specialize to different input regions.
   At 500 steps, groups may not have specialized enough for routing
   quality to create measurable differences.

**Prediction for macro scale**: At larger N with diverse data (code vs
prose vs math), routing quality will matter. The uniform baseline should
degrade, learned routing should improve, and LSH should fall between
them. The crossover point where routing quality becomes consequential
is an empirical question requiring N >> 8 and diverse data.

---

## Micro-Scale Limitations

1. **Small expert count.** With G=8, the experiment cannot distinguish
   routing quality because the problem is too easy for all methods.
   The value proposition of LSH (zero params, O(1) at large N) requires
   G=64-1000+.

2. **Homogeneous data.** Character-level names have minimal domain
   structure. With truly diverse data (code vs prose vs math), learned
   routing should provide more value than random partitions.

3. **No composition test.** This experiment tests single-domain quality
   only. The critical question for the contribution protocol -- can new
   experts be added to the LSH routing without recalibration? -- is
   untested but architecturally guaranteed (new expert just needs a
   hash key).

4. **Naive implementation.** The current implementation loops over T
   tables and computes all G experts. Production LSH would use binary
   hash-table lookup for O(1) per table. The throughput comparison is
   not meaningful for scaling arguments.

5. **Fixed projections not optimized.** The random projections are
   drawn once from N(0, 1/d). Data-dependent projections (learned
   once on a calibration set, then frozen) might improve quality
   further.

6. **Only 3 seeds.** All statistical tests are underpowered. Differences
   of 0.5-1.3% cannot be reliably detected with n=3.

---

## What Would Kill This

### At Micro Scale (tested, survived)

- **LSH quality >3% worse than softmax.** SURVIVED. LSH is -0.33% to
  -1.34% better (or rather, within noise) of the fair softmax baseline.
  The kill criterion is decisively passed.

- **Requires >4 hash tables.** SURVIVED. T=1 already matches the
  softmax baseline. No diminishing returns problem.

### What This Experiment CANNOT Distinguish

- **LSH vs softmax routing quality.** Both are indistinguishable from
  uniform routing at G=8. The experiment passes kill criteria but
  provides no evidence that LSH routing is GOOD -- only that it is
  not BAD enough to trigger the 3% kill threshold.

### At Macro Scale (untested)

- **Diverse data breaks random routing.** If code/prose/math tokens
  need different experts and random partitions don't separate them,
  learned routing could be 10-20% better. This is the most likely
  failure mode.

- **Load imbalance at scale.** LSH utilization std is 0.29-0.33 at
  G=8 (higher than originally reported because the utilization
  measurement now uses per-expert selection frequency, not weighted
  probability). At G=256, the imbalance could concentrate load on a
  few experts, causing both quality degradation and compute inefficiency.

- **Composition interference via hash collisions.** When composing
  experts from different contributors, hash collisions could route
  tokens to wrong experts.

- **Non-stationary representations.** As the model trains, hidden-state
  distributions shift. Fixed hash functions that were good at init may
  become poor later.

---

## Summary

LSH capsule routing **passes both kill criteria**: it is not worse than
3% vs learned softmax, and T=1 already matches. However, the revised
experiment with proper controls reveals that the original claim "LSH
beats softmax" was an artifact of (a) the balance loss handicapping the
softmax baseline, and (b) statistical noise with only 3 seeds.

The honest finding is: **at micro scale (G=8, homogeneous data, 500
steps), all routing strategies -- uniform, LSH, learned softmax --
produce statistically indistinguishable quality.** Routing quality is
irrelevant at this scale.

This is a useful null result. It tells us:
- LSH routing does not degrade quality (safe to use)
- The routing advantage will only emerge at larger G with diverse data
- The balance loss in the softmax router is the cause of near-uniform
  routing, not a limitation of softmax itself

The value of LSH routing for the contribution protocol (zero calibration,
zero routing parameters, instant expert addition) remains architecturally
valid, but quality advantages cannot be claimed at micro scale.

Next steps:
- Test at larger G (64, 256) where routing decisions become consequential
- Test with diverse data (multiple character distributions or synthetic domains)
- Implement binary LSH to validate the O(T*d) scaling claim
- Test composition: add new expert groups without recalibration
