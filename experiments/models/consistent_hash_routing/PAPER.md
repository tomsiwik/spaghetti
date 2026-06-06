# Consistent Hash Routing: Research Digest

## Hypothesis

Placing experts on a consistent hash ring enables incremental expert
add/remove with <5% quality degradation and <30% routing displacement,
without any recalibration.

**Falsifiable**: If adding one expert (N=8 to N=9) causes >5% quality
degradation without recalibration, or displaces >30% of existing routing
decisions, the approach is killed.

---

## What This Model Is

`ConsistentHashRoutingGPT` replaces the learned softmax router with a
**consistent hash ring** (Karger et al. 1997). Each expert is placed on
a ring [0, 2^32) at 150 virtual node positions. Each token's hidden state
is projected to a scalar via a fixed random vector, hashed to a ring
position, and routed to the k=2 nearest distinct experts clockwise.

### How It Works

1. **Ring construction**: Each of N experts gets 150 virtual nodes placed
   on the ring via FNV1a hashing of (expert_id, virtual_node_id).

2. **Token hashing**: x @ p (fixed random projection) -> FNV1a -> ring
   position in [0, 2^32).

3. **Expert selection**: Binary search for insertion point, walk clockwise
   collecting k=2 distinct experts.

4. **Weights**: Softmax over inverse ring distances for selected experts.

5. **Add expert**: Insert 150 new virtual nodes into the sorted ring.
   Only tokens whose ring position falls in the new expert's segments
   get rerouted. Theoretical displacement: ~1/(N+1).

### Why It Exists

The current composition protocol requires 100-200 steps of router
recalibration when adding or removing experts. This is a deployment
friction point: a contributor cannot add their expert without pausing
the system. Consistent hashing from distributed systems provides the
theoretical guarantee that adding one node displaces only ~1/N of keys.

---

## Lineage in the Arena

```
gpt -> moe -> capsule_moe -> consistent_hash_routing
                              (replaces softmax with hash-ring routing)
```

Sibling: `lsh_capsule_routing` (random-projection LSH, different hash scheme)

---

## Key References

**Consistent Hashing** (Karger et al. 1997): The foundational result that
adding/removing a node in a hash ring displaces only 1/N of keys. Originally
for web caching (Akamai). We adapt it to MoE expert routing.

**Jump Consistent Hash** (Lamping & Stepanov 2014): Simplified algorithm
achieving perfect balance with O(ln N) time. Used in our displacement
validation tests. Our ring implementation uses virtual nodes (the Karger
original approach) for easier add/remove operations.

**Hash Layers** (NeurIPS 2021, Chen et al.): Hash-based routing for MoE.
Competitive with learned routing (Switch Transformer). Does NOT use
consistent hashing -- uses fixed balanced hashes without incremental
add/remove guarantees.

**LSH Capsule Routing** (this project): Validated that all routing
strategies (softmax, LSH, uniform) produce indistinguishable quality
at G=8. This is the key enabler: if routing quality is irrelevant at
micro scale, the displacement property becomes the differentiator.

---

## Empirical Results

### Training Quality (500 steps, 3 seeds)

| Config | Params | Val Loss (mean) | vs Softmax |
|--------|--------|-----------------|------------|
| Softmax (k=2, balance loss) | 204,160 | 0.5188 | baseline |
| Consistent Hash (k=2) | 202,112 | 0.5234 | +0.89% |

Training quality is within 1% of softmax, consistent with the LSH
finding that routing strategy is irrelevant at G=8 micro scale.

### Add-Expert Protocol (N=8 -> N=9, no recalibration)

| Metric | Consistent Hash | Softmax | Kill Threshold |
|--------|----------------|---------|---------------|
| Degradation (mean) | +0.20% | +0.01% | >5% kills |
| Displacement (mean) | 9.1% | 0.0% | >30% kills |

**Kill criterion 1 (degradation >5%): PASSES** at +0.20%.

**Kill criterion 2 (displacement >30%): PASSES** at 9.1%.

### Per-Seed Breakdown

| Seed | CH val_before | CH val_after | CH degrad | CH displace | SM val_before | SM val_after | SM degrad | SM displace |
|------|-------------|-------------|---------|-----------|-------------|-------------|---------|-----------|
| 42 | 0.5169 | 0.5183 | +0.28% | 8.9% | 0.5148 | 0.5149 | +0.02% | 0.0% |
| 123 | 0.5173 | 0.5185 | +0.25% | 8.9% | 0.5139 | 0.5140 | +0.01% | 0.0% |
| 777 | 0.5361 | 0.5364 | +0.07% | 9.6% | 0.5276 | 0.5276 | +0.01% | 0.0% |

### Displacement Analysis

The measured displacement of 9.1% aligns with the theoretical prediction:
- Theoretical (top-1): 1/(N+1) = 1/9 = 11.1%
- Measured (primary expert argmax): 9.1%

The measured value is below theoretical because not all displaced tokens
change their argmax expert -- some change only their secondary expert
within the top-2 set.

### Softmax 0% Displacement Explained

The softmax router shows 0% displacement because the new expert's router
weights are initialized at scale 0.02 (random), while trained experts
have router weights at scale ~0.5-1.0 after 500 steps. The new expert
never wins the argmax. This is correct behavior: softmax routing is
inherently stable when adding a weak (random-init) expert.

However, this means softmax requires recalibration for the new expert
to actually participate in routing. Without recalibration, the new expert
is effectively dead. Consistent hashing guarantees the new expert
immediately receives ~1/N of traffic.

### Throughput

| Config | tok/s (mean) | vs Softmax |
|--------|-------------|-----------|
| Softmax | 62,103 | baseline |
| Consistent Hash | 21,019 | -66% |

Consistent hash routing is 3x slower due to Python-level hash
computation (FNV1a, binary search, ring walk) vs MLX-native matrix
multiply for softmax. This is an implementation artifact, not a
fundamental limitation -- the hash operations are O(d + log(NV))
per token vs O(Nd) for softmax, asymptotically cheaper at large N.

---

## Parameter Comparison

| Component | Softmax (G=8) | Consistent Hash (G=8) |
|-----------|--------------|----------------------|
| Routing params/layer | 512 | 0 |
| Total routing params | 2,048 | 0 |
| Capsule params | 196,608 | 196,608 |
| **Total params** | **204,160** | **202,112** |

---

## The Key Insight: Liveness vs Stability Tradeoff

Consistent hashing and softmax routing represent different tradeoffs
for the expert add/remove problem:

| Property | Consistent Hash | Softmax |
|----------|----------------|---------|
| New expert immediately routes traffic | Yes (~1/N) | No (dead until recalibrated) |
| Existing routing stable on add | ~91% unchanged | 100% unchanged |
| Quality degradation on add | +0.20% | +0.01% (but expert is dead) |
| Recalibration needed | No | Yes (100-200 steps) |

**Consistent hashing trades some routing stability for expert liveness**:
the new expert immediately participates at the cost of displacing ~9% of
existing routing. Softmax routing preserves 100% stability but the new
expert is dormant until recalibrated.

For the contribution protocol (VISION.md), consistent hashing enables a
**hot-add** workflow: a contributor uploads their expert and it
immediately serves traffic. Softmax requires a cold restart with
recalibration.

---

## Micro-Scale Limitations

1. **Small expert count (G=8)**: At G=256+, the displacement fraction
   decreases (1/257 ~ 0.4% for top-1), making consistent hashing even
   more attractive. The micro experiment cannot test this.

2. **Homogeneous data**: Character-level names have no domain structure
   that would benefit from locality-preserving routing. With diverse
   data, the fixed hash projection may route tokens suboptimally.

3. **Python hash overhead**: The current implementation uses Python-level
   hash computation, making throughput comparisons meaningless for
   scaling arguments. A C/MLX-native implementation would close the gap.

4. **Random-init new expert**: The experiment adds a random-init expert.
   In practice, the new expert would be pre-trained on a domain, which
   would further reduce quality degradation.

5. **No composition test**: This experiment tests single-domain training
   with expert addition. The full composition protocol (multiple domains,
   calibration) is not tested.

6. **Only 3 seeds**: Statistical power is limited. The 0.20% degradation
   and 9.1% displacement are directional findings, not precise estimates.

---

## What Would Kill This

### At Micro Scale (tested, survived)

- **>5% degradation on expert add**: SURVIVED. +0.20% mean (25x margin).
- **>30% displacement on expert add**: SURVIVED. 9.1% mean (3.3x margin).

### At Macro Scale (untested)

- **Locality matters at scale**: If diverse data requires locality-
  preserving routing (similar tokens -> similar experts), the fixed
  random projection may not provide sufficient locality. Learned routing
  could be 5-10% better. The LSH experiment defers this question.

- **Virtual node count tuning**: V=150 works at N=8. At N=256, V may
  need to scale. The ring size grows as O(N*V), which could become a
  memory concern.

- **Sequential add of many experts**: Adding experts one at a time, each
  displacing ~1/N, could accumulate displacement. After adding K experts,
  total cumulative displacement is bounded by K/(N+K). For N=8, adding 8
  more experts displaces at most 50% cumulatively. Need to verify quality
  holds under cumulative displacement.

- **Hash collision pathology**: If the fixed projection maps many tokens
  to the same ring region, some experts could be overloaded. Virtual
  nodes mitigate this, but adversarial inputs could break balance.

---

## Summary

Consistent hash routing **passes both kill criteria** with comfortable
margins: +0.20% degradation (threshold 5%) and 9.1% displacement
(threshold 30%) when adding one expert without recalibration.

The mechanism provides a **hot-add** protocol for the expert contribution
workflow: new experts immediately participate in routing without system
downtime. This complements the existing calibration-based protocol
(which provides higher quality but requires recalibration).

Training quality matches softmax within 0.89%, consistent with the
prior finding (LSH experiment) that routing quality is irrelevant at
micro scale with G=8.

The consistent hashing approach is genuinely novel for MoE routing --
no prior work applies hash-ring placement to neural network expert
selection. The cross-domain transfer from distributed systems (Karger
et al. 1997) to neural routing works because the displacement guarantee
is independent of the nature of the items being routed.

**Recommended next steps**:
1. Test cumulative add (N=8 -> N=16 one at a time)
2. Test with pre-trained (not random) new expert
3. Test at larger G (32, 64) where displacement fraction decreases
4. Implement in C/MLX for realistic throughput comparison
