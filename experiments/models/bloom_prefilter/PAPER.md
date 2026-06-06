# Bloom Filter Pre-Filtering: Research Digest

## Hypothesis

A two-stage routing pipeline -- Bloom filter per expert group for fast
pre-filtering, then softmax routing over survivors only -- will eliminate
at least 30% of expert-token pairs with a false positive rate under 20%,
reducing routing cost while preserving quality.

**Falsifiable**: Kill if (a) elimination rate is below 30%, or (b) false
positive rate exceeds 20%.

---

## What This Model Is

`BloomPrefilterGPT` adds a Bloom filter bank to each CapsulePool layer.
After training, activation patterns are profiled: for each token, which
expert groups fire above an activation threshold? Those token-group pairs
are inserted into per-group Bloom filters. At inference, each token is
hashed and queried against all Bloom filters. Groups whose filter returns
"definitely not" are excluded from softmax routing.

### How It Works

1. **Train** the model normally (Bloom filters inactive, standard softmax
   routing). Architecture and training are identical to CapsuleMoEGPT.

2. **Profile** by running training data through the trained model:
   - Compute per-group activation magnitude (L1 norm of group output)
   - For each (token, group) pair where activation > threshold:
     quantize the hidden state to an integer key, insert into that
     group's Bloom filter

3. **Activate** Bloom filters for inference:
   - For each token, query all G Bloom filters
   - Groups that return False are masked out (score set to -infinity)
   - Softmax routing runs only over surviving groups

### Why It Exists

Softmax routing costs O(G * d) per token per layer. At G=256 experts
(DeepSeek-V3 scale), this is significant. If most experts are irrelevant
for any given token, pre-filtering could eliminate them in O(G * k_hash)
time, independent of d. Bloom filters use bits (not learned weights),
providing extreme memory efficiency. And they guarantee zero false
negatives: no expert that should fire will be missed.

---

## Lineage in the Arena

```
gpt -> moe -> capsule_moe -> bloom_prefilter
                              (adds Bloom filter pre-filtering stage)
```

---

## Key References

**Bloom (1970)**: The original Bloom filter data structure. Space-efficient
probabilistic membership testing with zero false negatives.

**Switch Transformer** (Fedus et al., 2022): Standard MoE routing
baseline. Our Bloom filter approach adds a pre-filtering stage before
the Switch-style softmax routing.

**Hash Layers** (Chen et al., NeurIPS 2021): Uses hashing for MoE expert
selection (competitive with softmax). Differs from our approach: they
REPLACE routing with hashing; we use hashing to PRE-FILTER before routing.

**PEER** (He et al., DeepMind 2024): Product-key retrieval for 1M+
experts. Uses structured hashing for O(sqrt(N)) routing -- closest to
our pre-filtering motivation but different mechanism.

---

## Empirical Results

### Phase 1: Small m (m=256 bits) -- Complete Saturation

| Seed | Baseline VL | Bloom VL | Elim% | FPR | Delta |
|------|------------|----------|-------|-----|-------|
| 42 | 0.5184 | 0.5128 | 0.0% | 100% | +0.00% |
| 123 | 0.5104 | 0.5101 | 0.0% | 100% | +0.00% |
| 777 | 0.5286 | 0.5346 | 0.0% | 100% | +0.00% |

**Root cause**: With ~20K insertions per group into a 256-bit filter
(k=4), 80K bit-set operations completely saturate 256 bits. Every bit
is set, every query returns True, the filter is a no-op.

All m_bits values from 64 to 1024 are saturated. Kill criterion (a)
triggered.

### Phase 2: Large m (m=100K bits) -- High Elimination, High Damage

With sufficient Bloom filter capacity, the mechanism works:

| Seed | Threshold | m_bits | Elim% | Baseline VL | Bloom VL | Delta | FN-in-top-k |
|------|-----------|--------|-------|------------|----------|-------|-------------|
| 42 | 0.5 | 100K | 73.9% | 0.5130 | 0.5587 | +8.89% | 76.4% |
| 42 | 1.0 | 100K | 79.6% | 0.5130 | 0.5919 | +15.38% | 80.3% |
| 123 | 0.5 | 100K | 82.9% | 0.5099 | 0.5388 | +5.67% | 84.6% |
| 123 | 1.0 | 100K | 97.3% | 0.5099 | 0.5368 | +5.28% | 99.1% |
| 777 | 0.5 | 100K | 81.9% | 0.5135 | 0.5722 | +11.42% | 84.2% |
| 777 | 1.0 | 100K | 89.3% | 0.5135 | 0.5715 | +11.29% | 88.7% |

**FN-in-top-k** = fraction of experts that SHOULD be in the top-k
(per the full softmax router) but are eliminated by the Bloom filter.
This is 76-99% -- catastrophic.

### Phase 3: Activation Threshold Sweep (m=4096, seed=42)

| Threshold | Elim% | Val Loss | Delta |
|-----------|-------|----------|-------|
| 0.1 | 0.0% | 0.5220 | +0.00% |
| 0.5 | 1.4% | 0.5223 | +0.05% |
| 1.0 | 4.4% | 0.5235 | +0.28% |
| 1.5 | 16.9% | 0.5291 | +1.36% |
| 2.0 | 42.4% | 0.5486 | +5.10% |

Trade-off is clear: elimination and quality degradation are coupled.
There is no operating point that achieves >30% elimination with <2%
quality loss.

### Kill Criteria Evaluation

**Kill criterion (a): elimination rate < 30%.**

At practical Bloom filter sizes (m=256 to 4096), elimination is 0-17%.
**KILLED.** At large m (100K+), elimination reaches 74-97%, but this
comes with 5-15% quality degradation and 76-99% false negative rate
in the routing sense.

**Kill criterion (b): false positive rate > 20%.**

At m=256: FPR = 100% (filter saturated). **KILLED.**
At m=100K: Bloom FPR < 10% (filter works), but ROUTING false negatives
(experts that should fire but are excluded) are 76-99%. **KILLED** on
the routing-relevant metric.

**OVERALL VERDICT: KILLED.**

---

## Root Cause Analysis

The hypothesis fails because of a fundamental mismatch between Bloom
filters and neural routing:

### 1. Exact vs Approximate Membership

Bloom filters answer "was this exact key ever inserted?" Neural routing
needs "is this token SIMILAR to patterns this expert handles?" Similar
hidden states produce different hash keys when they fall in different
quantization bins. The Bloom filter correctly reports "this exact
quantized pattern was never profiled" -- but very similar patterns WERE
profiled.

### 2. The Quantization Boundary Problem

Hidden state x = [1.200, -0.501, ...] quantizes to key_A.
Hidden state x' = [1.200, -0.499, ...] quantizes to key_B (different bin).
These vectors are nearly identical but produce different Bloom filter
keys. The filter correctly says key_B was never inserted, even though
x and x' should route to the same experts.

### 3. The Coverage Problem

With d_h=8 dimensions at 256 bins each, the key space has 256^8 = 1.8e19
possible keys. Even at 8 bins per dimension, the space is 8^8 = 16.7M keys.
Profiling 20K tokens covers 0.12% of the key space. The filter has
seen almost nothing of the space it needs to cover.

### 4. Contrast with LSH (Which Works)

LSH routing (proven at micro scale, see lsh_capsule_routing) uses random
projections that PRESERVE angular similarity:

```
P[h(x) = h(y)] = 1 - theta(x,y) / pi
```

Similar vectors hash to the same bucket with high probability. Bloom
filters have no such locality property -- they are designed for exact
set membership, not similarity search.

---

## Micro-Scale Limitations

1. **Expert specialization is weak.** At G=8 with character-level data,
   93-100% of groups fire for every token (at threshold=0.1). Deeper
   layers show some specialization (30-45% for some groups). At macro
   scale with diverse data, specialization should be much stronger.

2. **The quantization scheme is naive.** More sophisticated feature
   hashing (SimHash, MinHash, feature hashing) might generalize better.
   However, the fundamental exact-vs-approximate mismatch remains.

3. **Only single-domain tested.** Composition (multi-domain) with Bloom
   pre-filtering was not tested because the single-domain results are
   already a kill.

---

## What Would Kill This (Already Killed)

### At Micro Scale (tested, killed)

- **Elimination < 30% at practical m.** KILLED. At m=256 to 4096,
  elimination is 0-17%. Only at m=100K+ does elimination exceed 30%,
  but with catastrophic quality impact.

- **FPR > 20%.** KILLED. At m=256, FPR=100% (saturation). At m=100K,
  Bloom FPR<10% but routing FN rate is 76-99%.

### At Macro Scale (not tested, likely still killed)

The fundamental problem (exact vs approximate membership) is
scale-independent. Stronger expert specialization at macro scale would
help, but the quantization boundary problem persists. A token that falls
on a bin boundary will hash to a different key than its nearest profiled
neighbor, regardless of scale.

### What WOULD Work Instead

For pre-filtering experts at scale, use **similarity-preserving** data
structures:

1. **LSH tables** (already validated at micro scale): random projections
   that hash similar vectors to the same bucket.

2. **KD-trees** (exp_kd_tree in HYPOTHESES.yml): spatial partitioning
   that groups similar vectors.

3. **Learned binary gates** (hierarchical tree, already validated):
   sigmoid gates that learn to route similar tokens together.

All three preserve the locality property that Bloom filters lack.

---

## What Was Learned

1. **Bloom filters are fundamentally unsuited for continuous-valued
   membership testing.** The exact-membership guarantee (zero false
   negatives on inserted keys) provides no value when the key space
   is vast and coverage is sparse. This is a general lesson applicable
   beyond MoE routing.

2. **Expert specialization at micro scale is weak.** With G=8 groups
   on homogeneous character-level data, 83-98% of expert-token pairs
   have activation above 0.1. This confirms the LSH experiment finding:
   routing quality is irrelevant at micro scale because all groups
   handle all tokens.

3. **Bloom filter capacity matters but does not save the approach.**
   At m=256, the filter is saturated (useless). At m=100K, the filter
   works mechanically but eliminates experts that should fire (because
   coverage is sparse). There is no m that fixes the coverage problem.

4. **The profiling infrastructure works.** The activation profiling
   pipeline (compute per-group L1 norms, threshold, record patterns)
   is reusable for other routing diagnostics. The Bloom filter insert/query
   machinery is correct and fast (numpy vectorized). The mechanism just
   does not match the problem.

---

## Summary

Bloom filter pre-filtering for expert routing is **KILLED at micro scale**.
The Bloom filter data structure provides exact membership testing, but
neural routing requires approximate similarity matching. Quantizing
continuous hidden states to discrete hash keys destroys the locality
needed for generalization: profiled patterns and inference patterns
hash to different keys even when they should route identically. With
sufficient filter capacity (m=100K), the filter eliminates 74-97% of
expert-token pairs, but 76-99% of those eliminations are wrong (removing
experts that should fire), causing 5-15% quality degradation.

The experiment validates that routing pre-filtering is a valid research
direction, but the data structure must preserve similarity. LSH (proven),
hierarchical trees (proven), and KD-trees (open) are the correct
alternatives.
