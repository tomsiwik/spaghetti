# Domain Taxonomy Generation: Research Digest (Revised)

## Hypothesis

An automated, principled domain taxonomy can produce expert domains with
less than 5% pairwise overlap (embedding cosine > 0.5) and fewer than 5% of
domains having nearest-neighbor cosine > 0.7, while a deliberately redundant
taxonomy fails these criteria -- demonstrating that the metrics have
discriminative power.

## What This Experiment Is

This experiment addresses the fundamental question: **what ARE the domains?**
The pilot-50 used a hand-crafted 5x10 taxonomy (programming, science,
professional, writing, reasoning x 10 each). The scale-500 plan used random
SlimOrca partitioning with no semantic structure. Neither approach scales to
5,000+ expert domains.

We design a principled 3-level hierarchical taxonomy with 270 leaf domains,
measure inter-domain overlap using sentence-transformer embeddings
(all-MiniLM-L6-v2), and validate the metrics against a negative control
taxonomy (270 paraphrases of 30 domains). The taxonomy serves as domain
planning infrastructure for SOLE expert scaling.

**Revision note**: This is v2, revised after adversarial review. The original
experiment had vacuous kill criteria (3000x margins), no negative control, and
over-interpreted a null correlation (r=0.034) as positive. This revision adds
all four required fixes.

## Lineage in the Arena

```
exp_distillation_pilot_50 (50 hand-crafted domains, 98% win rate)
    |
    v
exp_domain_taxonomy_generation (THIS EXPERIMENT -- 270 principled domains)
    |
    v
exp_scale_500_experts (future -- use this taxonomy for domain selection)
```

## Key References

- BigBench (Srivastava et al., 2022) -- 204 NLP tasks, 6 broad categories
- FLAN (Wei et al., 2022) -- 62 NLP datasets categorized by task type
- MMLU (Hendrycks et al., 2021) -- 57 academic subjects organized hierarchically
- WordNet (Miller, 1995) -- lexical database with hypernym/hyponym hierarchy
- Pilot-50 taxonomy (this project) -- 50 domains, 5 categories, empirically validated

## Taxonomy Design

The taxonomy is a 3-level hierarchy:

| Level | Count | Examples |
|-------|-------|----------|
| Supercategory | 6 | programming, science, professional, writing, reasoning, domain_specific |
| Category | 35 | physics, healthcare, web_frontend, creative_writing, formal_reasoning, ... |
| Leaf domain | 270 | quantum_mechanics, clinical_medicine, react_components, fiction_novel, ... |

**Supercategory breakdown:**

| Supercategory | Domains | Categories |
|---------------|---------|------------|
| Programming | 56 | 8 (systems, app, web, scripting, data, mobile, functional, devops) |
| Science | 56 | 6 (physics, chemistry, biology, earth/space, math, statistics) |
| Professional | 61 | 7 (healthcare, legal, business, management, engineering, marketing, cybersecurity) |
| Writing | 37 | 5 (creative, professional, journalism, persuasive, translation) |
| Reasoning | 32 | 5 (formal, applied, critical thinking, problem solving, metacognition) |
| Domain-specific | 28 | 4 (education, arts/culture, social sciences, practical skills) |

Each leaf domain has a rich text description (50-100 words) specifying the
knowledge area, key concepts, and distinguishing characteristics.

## Empirical Results

### Fix 1: Negative Control Comparison

The critical addition in this revision. We constructed a deliberately bad
taxonomy: 270 descriptions formed from 30 base domains, each paraphrased 9
times. For example, "python_coding" appears 9 times with near-synonym
descriptions ("Python programming", "Python development", "Python scripting",
etc.). If our metrics pass for both the good and bad taxonomy, they are useless.

| Metric | Good Taxonomy | Negative Control | Discriminates? |
|--------|---------------|------------------|----------------|
| K1: % pairs cos>0.5 | 0.45% | 2.79% | No (both pass) |
| K2: % domains NN cos>0.7 | 3.0% | **72.2%** | **Yes** |
| Mean pairwise cosine | 0.133 | 0.148 | Weak (1.1x) |
| Mean NN cosine | 0.520 | 0.740 | Strong (1.4x) |

**Key finding**: K2 (nearest-neighbor threshold) is the discriminating metric.
The negative control has 72.2% of domains with NN cos > 0.7 (FAIL), while the
good taxonomy has only 3.0% (PASS). This is a 24.4x separation ratio. K2
catches the redundancy because paraphrases of the same domain cluster tightly,
making nearest-neighbor cosine very high.

K1 (pairwise overlap) does NOT discriminate: even the bad taxonomy has only
2.79% of pairs above cos>0.5, because 30 distinct base domains still have low
cross-domain cosine. The redundancy is visible only in the nearest-neighbor
structure, not in the bulk pairwise distribution.

### Fix 2: Tightened Kill Criteria

| Version | Criterion | Threshold | Good Taxonomy | Margin |
|---------|-----------|-----------|---------------|--------|
| Old K1 | % pairs cos>0.7 | <=30% | 0.01% | 3000x (vacuous) |
| Old K2 | % domains NN cos>0.85 | <=20% | 0.0% | infinite (vacuous) |
| **New K1** | **% pairs cos>0.5** | **<=5%** | **0.45%** | **11x** |
| **New K2** | **% domains NN cos>0.7** | **<=5%** | **3.0%** | **1.7x** |

The tightened K2 margin of 1.7x is meaningful -- the good taxonomy is within
striking distance of failure. Adding many fine-grained domains within a single
category could push K2 over the threshold. This is appropriate: it means the
metric would flag real problems at scale.

### Overlap Distribution (Good Taxonomy)

| Cosine threshold | Pairs above | Fraction |
|------------------|-------------|----------|
| > 0.3 | 2,087 / 36,315 | 5.75% |
| > 0.4 | 587 / 36,315 | 1.62% |
| > 0.5 | 165 / 36,315 | 0.45% |
| > 0.6 | 28 / 36,315 | 0.08% |
| > 0.7 | 4 / 36,315 | 0.01% |

Mean pairwise cosine: 0.133, median: 0.122, std: 0.098.

### Hierarchical Structure Validation

| Comparison | Mean cosine | N pairs | Ratio |
|------------|-------------|---------|-------|
| Within-category | 0.338 | 943 | 2.65x |
| Cross-category | 0.128 | 35,372 | 1.0x (baseline) |
| Within-supercategory | 0.204 | 6,450 | 1.73x |
| Cross-supercategory | 0.118 | 29,865 | 1.0x (baseline) |

### Top Most-Similar Domain Pairs

| Cosine | Domain 1 | Domain 2 |
|--------|----------|----------|
| 0.726 | advertising_creative | copywriting_ads |
| 0.717 | editorial_opinion | debate_argumentation |
| 0.713 | scala_functional | haskell_pure |
| 0.704 | technical_documentation | user_guides |
| 0.694 | c_programming | embedded_c |

### Fix 3: Pilot-50 Proxy Validation (Honest Assessment)

**The embedding proxy does NOT predict expert improvement quality.**

| Metric | Value | p-value | Interpretation |
|--------|-------|---------|----------------|
| Pearson r(max_sibling_cos, improvement) | 0.034 | 0.813 | Effectively zero |
| Spearman rho(max_sibling_cos, improvement) | 0.149 | 0.301 | Not significant |
| Pearson r(max_any_cos, improvement) | 0.028 | 0.846 | Effectively zero |
| Pearson r(base_ppl, improvement) | -0.084 | 0.564 | Not significant |

**What this means**: Knowing two domains have similar embeddings tells you
NOTHING about whether they produce similar or different experts. The embedding
proxy validates only that domain NAMES are semantically distinct -- not that
domain EXPERTS will be distinct or useful. You could replace the entire embedding
analysis with "I looked at the names and they seem different" and get identical
predictive power.

**Why r is near zero**: All 50 pilot domains are already sufficiently distinct in
embedding space (max pairwise cos among pilot-50 mapped domains = 0.617). The
correlation would only emerge if we had domain pairs that are near-duplicates
(cos > 0.8), which is exactly the scenario the taxonomy is designed to avoid.
The proxy is a necessary-but-not-sufficient condition: domains identical in
embedding space would definitely produce similar experts, but distinct embeddings
do not guarantee distinct or useful experts.

**Closest pilot-50 domain pairs** (showing no improvement correlation):

| Cosine | Domain 1 (impr.) | Domain 2 (impr.) |
|--------|-------------------|-------------------|
| 0.617 | debate (37.3%) | critical-analysis (50.4%) |
| 0.609 | technical-writing (42.3%) | documentation (47.8%) |
| 0.573 | marketing (47.6%) | copywriting (57.0%) |
| 0.545 | creative-fiction (48.8%) | screenplay (53.5%) |

These pairs have high embedding similarity but uncorrelated improvement rates,
confirming the proxy's inability to predict outcomes.

## Output Artifacts

1. **taxonomy_500.json**: Full 270-domain taxonomy with paths and descriptions
2. **results.json**: Original overlap metrics (v1)
3. **results_revised.json**: Revised metrics with negative control and tightened criteria (v2)
4. **distinctness_scores.json**: Per-domain distinctness scores and most-similar neighbors
5. **validate_revised.py**: Revision script implementing all 4 fixes

## Limitations

1. **Embedding proxy does not predict expert outcomes** (r=0.034). The taxonomy
   validates name distinctness only. Whether a domain produces a USEFUL expert
   depends on: (a) base model weakness in that area, (b) quality of training
   data, (c) domain's structure/learnability. None of these are captured by
   embedding cosine. This is the experiment's fundamental limitation.

2. **K1 does not discriminate good from bad taxonomies.** Only K2
   (nearest-neighbor) catches redundancy. K1 (pairwise) misses it because even
   redundant taxonomies have low cross-domain cosine. A better K1 would use
   within-cluster overlap directly.

3. **No training data generation.** This experiment designs the taxonomy but does
   not generate training data for the 220 new domains. Generating 1000 examples
   per domain at $0.19/domain would cost ~$42 for all 220 new domains.

4. **270 not 500.** The current taxonomy has 270 leaf domains, not the target
   500. K2 margin is only 1.7x -- adding 230 more fine-grained domains would
   likely increase NN cosine and could push K2 toward failure.

5. **Taxonomy is entirely hand-crafted.** The title says "generation" but there
   is no algorithmic generation -- it is a manually authored Python dictionary.
   This is fine as infrastructure but the "experiment" is: "I wrote 270 domain
   names and confirmed they are distinct by embedding cosine."

6. **Single embedding model.** all-MiniLM-L6-v2 (384-dim). Different embedding
   models might assign different similarity scores.

## What Would Kill This

**At micro scale (this experiment) -- tightened criteria:**
- K1: >5% of domain pairs have cos>0.5 (actual: 0.45%, PASS)
- K2: >5% of domains have NN cos>0.7 (actual: 3.0%, PASS, 1.7x margin)

**Negative control validation:**
- If the negative control also passed both K1 and K2, the metrics would be
  useless (it fails K2 with 72.2%, confirming discriminative power)

**At macro scale (future validation):**
- Training experts on the 270 domains and finding >20% show <2% PPL improvement
  over base (domain descriptions too vague to generate useful data)
- Finding that semantically similar domains produce LoRA experts with weight
  cosine > 0.1, indicating the taxonomy does not prevent inter-expert interference
- The scale-500 pipeline producing worse results than the pilot-50 pipeline

## Status Assessment

This experiment is **infrastructure with weak validation**. The taxonomy artifact
(270 domains, 3-level hierarchy) is useful for downstream planning. The embedding
metrics have been shown to discriminate good taxonomies from bad ones (via
negative control), but the embedding proxy has no predictive power for expert
outcomes (r=0.034). The tightened kill criteria pass with reasonable margins
(11x for K1, 1.7x for K2).

The honest conclusion is: **the 270 domain names are semantically distinct** (a
weak but real claim), **and the K2 metric can detect redundancy** (demonstrated
by the negative control). The much stronger claim -- that this taxonomy will
produce distinct, useful experts -- remains unvalidated and requires actual
training experiments.
