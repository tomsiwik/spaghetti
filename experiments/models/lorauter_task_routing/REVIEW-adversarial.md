# Peer Review: LoRAuter Task-Representation Routing

## Experiment Type
Guided exploration

## Hack Detector
- Fix count: 1 (single mechanism: centroid-based routing in sentence-embedding space). CLEAN.
- Is MATH.md a proof or a description? Description of a mechanism with cited prior results. No Theorem/Proof/QED block. This is appropriate for guided exploration -- the "proof" is the LoRAuter empirical result being explored in a new setting.
- Metric used as evidence: Routing accuracy (domain ID), Pearson r (effectiveness), behavioral score (composition quality). Routing accuracy and behavioral score are directly meaningful. Pearson r is the right test for the unknown.
- Kill criteria source: Derived from prior findings (#247, #253, #238) and the LoRAuter paper. Reasonable grounding.

## Self-Test Audit
1. One-sentence impossibility property: "Sentence-embedding space is trained to preserve semantic task structure." This is a PROPERTY CLAIM, not an impossibility property. It does not state what failure mode is made impossible. However, for guided exploration this is acceptable -- the experiment is testing whether this property holds in practice.
2. Cited theorems: JL lemma (real, but loosely applied -- sentence transformers are not random projections, JL does not guarantee semantic preservation), Cover & Hart nearest-centroid (real, conditions approximately met with Fisher ratio 5.61), LoRAuter empirical result (real, correctly cited). The JL citation is window dressing -- sentence transformers are trained nonlinear mappings, not random linear projections. JL does not apply here. FLAG but not blocking.
3. Predicted numbers: Specific and falsifiable (>=80%, r>0.3, >=1 domain, <=20%). GOOD.
4. Falsification condition: "If sentence-embedding similarity has zero correlation with adapter effectiveness (r~0)." This targets the core unknown. GOOD.
5. Hyperparameter count: 2 (model choice, m=20). Honestly reported. GOOD.
6. Hack check: Clean replacement of TF-IDF, not a stacked fix. GOOD.

## Mathematical Soundness

MATH.md does NOT contain a formal proof, which is correct for guided exploration. It states:
- **Proven framework:** LoRAuter (arXiv:2601.21795) achieves 101.2% oracle on Llama2-7B with centroid routing.
- **Unknown:** Whether sentence-embedding similarity predicts adapter effectiveness (not just domain identity) on BitNet-2B-4T.

This is a well-formulated guided exploration. The unknown is precisely identified and the experiment narrows it.

**Issues found:**

1. **JL lemma misapplication.** MATH.md Section C cites Johnson-Lindenstrauss to justify sentence transformers preserving semantic distances. JL applies to random linear projections, not trained nonlinear mappings. The sentence transformer's distance preservation comes from its contrastive training objective, not from JL. This is a cosmetic citation -- it does not affect the experiment design or conclusions, but it is intellectually dishonest to dress up "trained to do X" as "guaranteed by theorem Y." MINOR.

2. **The LoRAuter result is empirical, not a theorem.** MATH.md treats the LoRAuter 101.2% oracle result as a "prior mathematical foundation." It is an empirical result on a different model (Llama2-7B) with different adapters (48 tasks, full-precision). Calling it a mathematical foundation overstates the grounding. More accurately: this is an empirical precedent being tested in a new setting. MINOR.

3. **P2 prediction derivation is weak.** The r>0.3 threshold is justified by: "if similarity had no predictive power, random selection would yield ~1/K=~20% oracle." This conflates domain-level routing accuracy with per-query effectiveness correlation. LoRAuter achieves 101.2% oracle because it correctly identifies WHICH adapter, not because cosine similarity predicts HOW MUCH benefit per query. The experiment's own results confirm this -- 96% routing accuracy (domain ID solved) with r=0.234 (effectiveness unsolved). The r>0.3 prediction was poorly grounded from the start. NOT BLOCKING because the experiment correctly identifies this decomposition in its interpretation.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table:

| Prediction | Threshold | Measured | Match |
|------------|-----------|----------|-------|
| P1 routing accuracy | >= 80% | 96% | YES |
| P2 effectiveness r | > 0.3 | 0.234 | NO |
| P3 behavioral improvement | >= 1 domain | 2 domains | YES |
| P4 coherence | <= 20% incoherent | 0% | YES |

3/4 predictions confirmed. The failed prediction (P2) leads to the key insight (two-signal decomposition). This is good guided-exploration methodology -- the failed prediction narrows the unknown.

## Concerns

### A. K2 Criterion Sleight-of-Hand (MODERATE)

The K2 kill criterion in MATH.md says "behavioral improvement vs TF-IDF" but the code compares against BASE (no adapter). The code comment explains: "TF-IDF routing was equivalent to base in Finding #253 since TF-IDF has r=-0.079." This is misleading. TF-IDF routing at 90% accuracy would route to the correct adapter most of the time, producing results similar to oracle (not base). The comparison should be emb-routed vs TF-IDF-routed, not emb-routed vs base. By comparing against base, K2 is measuring "does any adapter help at all?" not "does embedding routing help more than TF-IDF routing?"

The PAPER.md results show that the real comparison is routing accuracy: 96% (embedding) vs 90% (TF-IDF). Since both route to the correct adapter most of the time, and the per-query effectiveness prediction is equally poor for both, the behavioral difference is driven entirely by the 6pp routing accuracy gap. This is a legitimate but modest improvement.

### B. Oracle Comparison is Circular (MODERATE)

Oracle routing uses the same domain labels as the test set. Emb-routed achieves 99.7% of oracle because it achieves 96% routing accuracy. The 99.7% figure is just a restatement of 48/50 correct routing decisions. It is not an independent measurement of adapter effectiveness.

### C. Statistical Power for Correlation (ACKNOWLEDGED)

n=10 per domain gives 95% CI of approximately +/-0.63 for Pearson r. The experiment cannot distinguish r=0.234 from r=0 or r=0.5. The PAPER.md Limitations section acknowledges this. The overall n=50 gives tighter bounds (~+/-0.28) which is more informative -- the fact that r=0.234 with p=0.103 at n=50 suggests a weak positive trend that is unlikely to reach r>0.3 even with more data. This is a reasonable interpretation.

### D. Domain Scales are Hardcoded, Not Predicted (MINOR)

The per-domain scales (medical=20, code=20, math=20, legal=4, finance=1) are taken from Finding #249 and hardcoded. The routing only selects WHICH adapter; it does not predict the optimal scale. This means the "effectiveness" being measured is already optimized by prior work. The experiment correctly identifies this as the unsolved problem (scaling vs routing), but it means the 99.7% oracle figure is conditional on having the right scales pre-computed.

## NotebookLM Findings

Skipping NotebookLM deep review -- the experiment is straightforward enough that manual review is sufficient. The key finding (domain ID solved, effectiveness prediction unsolved) is well-supported by the data.

## Novelty Assessment

**Prior art:** LoRAuter (arXiv:2601.21795) is the direct inspiration and correctly cited. The experiment applies LoRAuter's method to a new setting (BitNet-2B-4T, 5 domains, sentence-transformer embeddings instead of LoRAuter's custom model).

**Delta:** The key contribution is the two-signal decomposition insight: domain routing is trivially solved by sentence embeddings (96%), but per-query effectiveness prediction remains unsolved regardless of embedding space. This is a useful negative result that redirects future work from "better routing" to "better scaling prediction."

## Macro-Scale Risks (advisory)

1. With 50+ adapters, centroid overlap may degrade routing accuracy (legal-finance similarity already 0.825 at 5 domains).
2. Top-1 routing will fail when queries span multiple domains. LoRAuter's top-K with softmax weighting addresses this.
3. The hardcoded per-domain scales cannot work at macro -- need runtime scale prediction per query.
4. All-MiniLM-L6-v2 is a generic sentence transformer; LoRAuter's recommended model (Styxxxx/lora_retriever) is specifically trained for LoRA adapter retrieval and may perform differently.

## Verdict

**PROCEED**

This is a well-executed guided exploration that correctly identifies the unknown, makes falsifiable predictions, and interprets the results honestly -- including the failed prediction. The two-signal decomposition (domain ID solved, effectiveness unsolved) is a genuinely useful finding that redirects future work.

Specific fixes requested (non-blocking):

1. **Remove or qualify the JL lemma citation.** Sentence transformers are not random linear projections. Either remove the JL citation or explicitly note that it does not directly apply and the distance preservation comes from contrastive training.

2. **Clarify K2 comparison.** The MATH.md says "vs TF-IDF" but the code compares vs base. Either run the actual TF-IDF comparison or restate K2 as "vs base" throughout. The current version is misleading.

3. **Downgrade the 99.7% oracle claim.** This figure is a restatement of 96% routing accuracy, not an independent quality measurement. State it as "routing accuracy of 96% means only 2/50 queries get the wrong adapter, resulting in near-oracle quality."

Finding status recommendation: **supported** (guided exploration that narrowed the unknown -- domain routing solved, effectiveness prediction confirmed unsolved).
