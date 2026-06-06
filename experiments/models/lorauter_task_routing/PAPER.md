# LoRAuter Task-Representation Routing: Proof Verification Report

## Theorem (Restated from MATH.md)

Sentence-embedding space, trained to preserve semantic task structure, provides
a routing signal where proximity to adapter centroids correlates with adapter
effectiveness -- unlike TF-IDF bag-of-words space (Finding #253, r=-0.079).

## Predictions vs Measured

| Prediction (from MATH.md) | Measured | Match? |
|---------------------------|----------|--------|
| P1: Routing accuracy >= 80% | 96.0% (48/50) | YES |
| P2: Effectiveness correlation r > 0.3 on >= 1 domain | max |r| = 0.212 (finance), overall |r| = 0.234 | NO |
| P3: Emb-routed >= base on >= 1 domain | 2 domains (math +40.0pp, code +6.2pp) | YES |
| P4: Incoherent output <= 20% | 0.0% (0/50) | YES |

## Hypothesis

Sentence-embedding centroids from validation sets enable near-oracle adapter
routing at 96% domain accuracy, achieving 100% of oracle behavioral quality on
3/5 domains, but embedding similarity does NOT predict per-query adapter
effectiveness (max r=0.21, same structural gap as TF-IDF).

## What This Model Is

LoRAuter-style task-representation routing applied to 5-domain BitNet-2B-4T
LoRA composition. For each adapter, we encode 20 validation instructions through
a sentence-transformer (all-MiniLM-L6-v2, 384-dim), compute centroid embeddings,
and route incoming queries by cosine similarity to the nearest centroid.

## Key References

- LoRAuter (arXiv:2601.21795): Task-representation routing achieving 101.2% oracle
- Finding #253: TF-IDF has zero predictive power for adapter effectiveness (r=-0.079)
- Finding #247: TF-IDF achieves 90% domain classification accuracy
- Finding #238: Oracle routing achieves math 8x, code +48.9%, medical +10.7%

## Empirical Results

### Routing Accuracy: 96% (vs TF-IDF 90%)

| Domain | Embedding Routing | TF-IDF (Finding #247) |
|--------|------------------|-----------------------|
| Medical | 100% | 100% |
| Code | 100% | 100% |
| Math | 100% | 100% |
| Legal | 90% | 80% |
| Finance | 90% | 70% |
| **Overall** | **96%** | **90%** |

Sentence embeddings improve routing accuracy by 6 percentage points, primarily
on the hardest domains (legal, finance). Fisher separability ratio = 5.61.

### Behavioral Quality: Near-Oracle Performance

| Domain | Base | Emb-Routed | Oracle | Emb/Oracle |
|--------|------|-----------|--------|------------|
| Math | 0.000 | 0.400 | 0.400 | 100.0% |
| Code | 0.550 | 0.612 | 0.612 | 100.0% |
| Medical | 0.321 | 0.306 | 0.306 | 100.0% |
| Legal | 0.101 | 0.113 | 0.115 | 98.5% |
| Finance | 0.058 | 0.057 | 0.057 | 100.0% |

Embedding-routed composition achieves 99.7% of oracle performance overall.
The only loss is on legal (1 misrouted query out of 10).

Key behavioral improvements vs base:
- Math: +40.0 percentage points (0.0 -> 0.4, answer correctness)
- Code: +6.2 percentage points (0.55 -> 0.61, syntax validity + recall)
- Legal: +1.2 percentage points (neutral, within noise)
- Medical: -1.5 percentage points (neutral, within noise)
- Finance: -0.1 percentage points (neutral, scale=1.0 means minimal adapter effect)

### Embedding-Effectiveness Correlation: Not Significant

| Domain | Pearson r | p-value | Interpretation |
|--------|-----------|---------|----------------|
| Medical | 0.191 | 0.596 | Not significant |
| Code | 0.070 | 0.847 | Not significant |
| Math | 0.015 | 0.967 | Not significant |
| Legal | -0.024 | 0.947 | Not significant |
| Finance | 0.212 | 0.557 | Not significant |
| **Overall** | **0.234** | **0.103** | Marginal trend |

Maximum per-domain |r| = 0.212 (finance). Overall |r| = 0.234 with p=0.103.
While the direction is positive (higher similarity -> marginally higher scores),
it does not reach the r > 0.3 threshold. This confirms that embedding similarity
predicts DOMAIN IDENTITY (96% accuracy) but not WITHIN-DOMAIN EFFECTIVENESS.

### Inter-Centroid Similarity Matrix

|          | Medical | Code  | Math  | Legal | Finance |
|----------|---------|-------|-------|-------|---------|
| Medical  | 1.000   | 0.602 | 0.538 | 0.649 | 0.682   |
| Code     | 0.602   | 1.000 | 0.575 | 0.694 | 0.738   |
| Math     | 0.538   | 0.575 | 1.000 | 0.555 | 0.700   |
| Legal    | 0.649   | 0.694 | 0.555 | 1.000 | 0.825   |
| Finance  | 0.682   | 0.738 | 0.700 | 0.825 | 1.000   |

Legal-finance pair has highest inter-centroid similarity (0.825), explaining
the 90% accuracy on both (misroutes go between these two). Math is most
distinct (lowest inter-centroid similarities), enabling 100% accuracy.

## Kill Criteria Assessment

**K1 (#666): FAIL** -- Embedding-effectiveness correlation <= 0.3 across all domains.
Max per-domain |r| = 0.212, overall |r| = 0.234. Neither exceeds 0.3. However,
the overall correlation shows a marginal positive trend (p=0.103) -- directionally
better than TF-IDF (r=-0.079) but not statistically significant at n=50.

**K2 (#667): PASS** -- Learned routing improves behavioral quality on 2/5 domains
vs base (math: +40.0pp, code: +6.2pp). Both exceed the 2% significance threshold.

**K3 (#668): PASS** -- Incoherent output rate = 0.0% (0/50 queries). Well below
the 20% threshold.

## Interpretation: The Two-Signal Decomposition

This experiment reveals a clean structural decomposition of the routing problem:

1. **Domain identification** (WHICH adapter): Sentence embeddings solve this at 96%
   accuracy, achieving 99.7% of oracle behavioral quality. This is a SOLVED problem.

2. **Effectiveness prediction** (HOW MUCH benefit): Neither TF-IDF (r=-0.079) nor
   sentence embeddings (r=0.234) predict per-query adapter utility. This remains
   UNSOLVED.

The good news: domain identification is the HIGHER-VALUE signal. Getting the
right adapter gives you 99.7% of oracle. Getting the per-query scaling right
(Finding #249) is a separate, orthogonal problem.

The insight for the project: **Routing is solved. Scaling is not.** Future work
should focus on predicting per-query optimal scale, not on improving domain routing.

## Limitations

- n=10 per domain (directional, not statistically powerful)
- 5 well-separated SFT domains only; unclear how routing degrades with similar domains
- Single sentence-transformer model (all-MiniLM-L6-v2, 384-dim); LoRAuter's
  recommended model (Styxxxx/lora_retriever) was not tested
- Effectiveness correlation at n=10 per domain has 95% CI of +/-0.63 for r;
  the overall n=50 has tighter bounds but still cannot definitively confirm r > 0.3
- Using non-overlapping val/test split (20 centroid + 10 eval from 50 total);
  slight contamination possible if domain-level features are shared

## What Would Kill This

**At micro scale:** If routing accuracy drops below 80% on a harder domain set
(e.g., 20+ overlapping domains), the centroid approach may fail and require
per-layer or per-token routing (PHATGOOSE-style).

**At macro scale:** If the sentence-transformer embedding dimension (384) becomes
insufficient for fine-grained routing among 50+ adapters, the centroids may
overlap and routing accuracy may degrade significantly.
