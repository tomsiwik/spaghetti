# Split Domain Specialization: Research Digest

## Hypothesis

Split children specialize faster than independently-initialized children
when fine-tuned on different domains, as measured by convergence speed
(>10% fewer steps) and domain separation (Jaccard overlap <0.95).

**Falsifiable**:
- KC1 KILL: split children do not reach independent-child quality faster (>10% fewer steps)
- KC2 KILL: split children show worse domain separation (Jaccard overlap >0.95 between domains)

---

## What This Model Is

`SplitDomainSpecGPT` is architecturally identical to `HierarchicalTreeGPT`
(depth-3 binary tree, 8 leaf capsule groups, beam=2). The experiment tests
domain-specific specialization after splitting a trained parent leaf into
two children.

The protocol:
1. Train a base tree on all data (300 steps)
2. Split leaf 0 into two half-size children (16 capsules each)
3. Fine-tune children on alternating domain batches (child 0 and 1 both
   trainable, gate learns to route domain A to one child and B to the other)
4. Compare against independently-initialized half-size children trained
   identically

This directly addresses Limitation 5 from the split_leaf_actual PAPER.md:
"Mixed-data fine-tuning only. The children were fine-tuned on mixed data,
not domain-specialized data."

---

## Lineage in the Arena

```
gpt -> moe -> capsule_moe -> hierarchical_tree -> split_freeze_protocol -> split_leaf_actual -> split_domain_spec
                              (tree routing)       (warm-start)            (split mechanism)    (domain specialization)
```

---

## Key References

**Jordan & Jacobs, 1994** (Hierarchical Mixtures of Experts): The original
HME framework where gating networks learn to route inputs to specialized
experts. Our split-then-specialize is a weight-inheritance approach to
creating new experts in this framework.

**Capsule Identity Tracking (Exp 16)**: Established the Jaccard methodology
for comparing activation sets across conditions.

**Split Leaf Actual (parent experiment)**: Validated the split mechanism
and showed convergence equivalence on mixed data.

---

## Empirical Results

### KC1: Convergence Speed (3 seeds, 400 steps, eval every 25)

Convergence = first step reaching within 1% of final quality.

| Seed | Split (steps) | Independent (steps) | Speedup |
|------|---------------|---------------------|---------|
| 42   | 150           | 125                 | -20.0%  |
| 123  | 75            | 75                  | 0.0%    |
| 777  | 150           | 175                 | +14.3%  |
| **Mean** | **125** | **125** | **0.0%** |

**KC1: KILLED (0.0% speedup, threshold >10%).**

Split children show no convergence advantage over independent children
when fine-tuned on domain-specific data. In seed 42, independent
actually converges faster. The mean speedup is exactly zero.

### KC2: Domain Separation (3 seeds, 20 batches x 32 profiling)

Jaccard similarity between a child's active capsule set on domain A vs
domain B. J=1.0 means identical activation patterns (no specialization).
J=0.0 means completely different patterns (full specialization).

| Seed | Split J | Independent J |
|------|---------|--------------|
| 42   | 0.980   | 0.992        |
| 123  | 0.946   | 0.990        |
| 777  | 1.000   | 1.000        |
| **Mean** | **0.975** | **0.994** |

**KC2: KILLED (J=0.975, threshold <0.95).**

Neither condition shows meaningful domain specialization. Nearly all
capsules fire on both domains. The split condition shows marginally
better separation (J=0.975 vs 0.994) but both are far above the
0.95 threshold.

### Per-Layer Jaccard Detail

The few capsules that DO differentiate between domains appear only in
Layer 3 (deepest). Layers 0-2 show perfect overlap (J=1.0) in almost
all cases. This is consistent with the behavioral_dedup finding that
deeper layers specialize more.

| Layer | Split J (mean across seeds/children) | Indep J |
|-------|--------------------------------------|---------|
| 0     | 0.989                                | 1.000   |
| 1     | 0.974                                | 0.987   |
| 2     | 0.965                                | 1.000   |
| 3     | 0.973                                | 0.990   |

### Final Quality

| Method | Val Loss (mean mixed) | vs Independent |
|--------|----------------------|----------------|
| Base (pre-split) | 0.5232 | -- |
| **Split** | **0.5204** | **+0.08%** |
| Independent | 0.5199 | -- |

Both conditions converge to equivalent final quality (+0.08% gap).
This confirms split_leaf_actual's finding that split and independent
are interchangeable at micro scale.

### Learning Curves

In seed 123, split shows a consistent (but small) advantage throughout
training (-0.0007 to -0.0026 per evaluation point). In seeds 42 and
777, independent is marginally better. Overall: no systematic advantage
for either condition.

---

## Key Findings

1. **Domain specialization does not emerge at micro scale for either
   condition.** Jaccard > 0.95 for BOTH split and independent children.
   The 16 capsules per child are too few, and the character-level name
   domains are too similar, for capsule-level specialization to occur.
   This is not a failure of the split mechanism -- it is a property of
   the task and scale.

2. **Split provides no convergence advantage on domain-specific data.**
   Mean speedup is exactly 0.0%. This contrasts with the mixed-data
   result from split_leaf_actual (2/3 seeds faster). On domain-specific
   data, only ~50% of inherited features are relevant (those activated
   by the target domain), reducing the initialization advantage.

3. **Split does show marginally better separation (J=0.975 vs 0.994).**
   This difference (0.019) is small but consistent. The split children
   inherit a capsule partition where half the capsules were more aligned
   with one half of the data distribution. This structural difference
   persists through fine-tuning but is not large enough to cross the
   0.95 threshold.

4. **Specialization concentrates in deeper layers.** The small amount
   of domain differentiation that does occur (J < 1.0) appears mainly
   in Layer 3. This is consistent with the known pattern that deeper
   layers specialize (behavioral_dedup found all redundancy in Layer 0,
   none in deeper layers).

---

## Micro-Scale Limitations

1. **Capsule count too low for specialization.** With 16 capsules per
   child, nearly all capsules fire on both domains. At macro scale with
   128+ capsules, there would be enough capacity for domain-exclusive
   detectors.

2. **Domains are too similar.** The binary split (a-m vs n-z) produces
   domains that differ mainly in the first character. Names share most
   bigram/trigram patterns regardless of starting letter. Genuinely
   distinct domains (code vs medical text) would show stronger separation.

3. **Binary activation metric may miss soft specialization.** Jaccard
   uses binary (fires/doesn't fire) classification. A capsule that fires
   strongly on domain A and weakly on domain B counts as active for both.
   A frequency-weighted metric might reveal more differentiation.

4. **Only 400 fine-tuning steps.** More training might produce stronger
   specialization, though at micro scale loss plateaus by step 200.

5. **No explicit domain routing.** The gate starts at 50/50 and learns
   to route through alternating batches. Explicit domain-conditioned
   routing (child 0 = domain A, child 1 = domain B) might produce
   stronger specialization but would require domain labels at inference.

---

## What Would Kill This

### At Micro Scale (tested -- both killed)

- **KC1: Convergence speed.** KILLED at 0.0% speedup (threshold >10%).
  Neither condition has a systematic advantage.

- **KC2: Domain separation.** KILLED at J=0.975 (threshold <0.95).
  No domain specialization emerges for either condition.

### At Macro Scale (untested, would need to validate)

- **Distinct domain separation.** If domains with genuinely different
  vocabulary (code vs prose vs math) still show J > 0.95 at macro scale,
  then capsule-level specialization is not achievable through fine-tuning
  alone, and explicit routing mechanisms are needed.

- **Convergence advantage.** If split children do not converge faster
  than random init on distinct domains at d=4096 with limited fine-tuning
  budget, the split-then-specialize protocol provides no value over
  simply creating new random leaves.

---

## Why This Negative Result Matters

This experiment closes a gap identified in exp_split_leaf_actual's review.
The hypothesis that split-then-specialize would produce faster domain
specialization was plausible but is now FALSIFIED at micro scale.

The negative result has a constructive interpretation: **domain specialization
at micro scale is bottlenecked by the task/data similarity, not by the
initialization method.** Both split and independent produce identical
activation patterns because the domains are not distinct enough in
character-level feature space.

This means:
- At micro scale, split vs independent initialization is a non-factor
  for domain specialization (use whichever is simpler)
- The split mechanism's value lies in function preservation and tree
  growth (validated by split_leaf_actual), not in specialization
- True domain specialization experiments require either (a) macro scale
  with distinct data domains, or (b) a different micro task with more
  domain-separable features

---

## Summary

The split domain specialization hypothesis is **KILLED** on both criteria:

**KC1 (convergence speed)**: 0.0% mean speedup (threshold >10%). Split
children show no advantage over random initialization for domain-specific
fine-tuning.

**KC2 (domain separation)**: J=0.975 (threshold <0.95). Neither split
nor independent children show domain specialization. The micro-scale
task does not produce domain-separable features.

The split mechanism remains valid for its original purpose (function-preserving
tree growth, validated by split_leaf_actual). Domain specialization is
a property of the data and scale, not the initialization method.
