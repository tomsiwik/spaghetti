# Semantic Router at Macro Scale: Real Embeddings on Real Domains

**Experiment:** `exp_followup_semantic_router_macro`
**Verdict:** **SUPPORTED** (K1570 PASS ∧ K1946 PASS, target-gated per F#666)
**Type:** Verification (guided exploration of parent's information-limit claim)

## 1. Hypothesis

Parent `exp_semantic_router` reached 27.3% domain top-1 at N=15 on synthetic Markov-chain data
and attributed the ceiling to a within-cluster information bottleneck. We hypothesized the
ceiling was **data-specific** (V=32 character vocab + 2–4% stationary-distribution variation),
not a property of routing. Testing with real MiniLM-L6-v2 embeddings on 25 real-text domains
(GSM8K, CodeAlpaca, MedMCQA, MMLU subjects) should lift the ceiling above 37.3% (parent + 10pp).

## 2. Prediction vs. Measurement

| Strategy        | Predicted top-1 | Measured top-1 | Predicted top-3 | Measured top-3 | Δ top-1 |
|-----------------|----------------|-----------------|-----------------|-----------------|---------|
| hash_ring       | ≈4%            | 4.03%           | ≈12%            | 4.03%†          | on-target |
| keyword_freq    | 40–55%         | 30.52%          | 65–75%          | 55.56%          | −10pp (below range) |
| **cosine_sim**  | **75–82%**     | **79.42%**      | **90–95%**      | **94.14%**      | **on-target** |
| lsh_partition   | 55–70%         | 22.46%          | 78–88%          | 44.06%          | −33pp (below range) |
| utterance_1nn   | 70–80%         | 75.17%          | 85–93%          | 90.97%          | on-target |
| utterance_agg   | 72–82%         | 73.51%          | 87–94%          | 91.19%          | on-target |
| oracle          | 100%           | 100%            | 100%            | 100%            | — |

†hash_ring's top-K is trivial: the content-agnostic hash produces only one candidate
(we encode top-3 = {top-1} because there is no second ranked candidate). Its top-K
is not directly comparable with the embedding-based strategies.

**Prediction quality.** 4 of 6 semantic strategies match prediction ranges. Two underperformed:

- **keyword_freq −10pp.** Character-frequency matching is weaker on real text than expected.
  The printable ASCII profile of a math question vs a medical question has fewer distinguishing
  bits than assumed, because both share a large common-English base distribution.
- **lsh_partition −33pp.** SimHash's binary codes collapse the 384-dim MiniLM space too
  aggressively with P=32 planes. The domain code-means diverge, but per-query binary codes
  lose enough angular information that top-1 drops to ~22%. Matches parent's observation
  that LSH is the worst semantic router (19.6% at micro); the effect persists at macro.

## 3. Kill Criteria

| ID | Type | Threshold | Measured | Status |
|----|------|-----------|----------|--------|
| K1570 | proxy | best top-1 ≥ 37.3% (parent + 10pp) | 79.42% (cosine_sim) | **PASS** (+42.1pp margin) |
| K1946 | target (F#666) | best top-3 ≥ 85% | 94.14% (cosine_sim) | **PASS** (+9.1pp margin) |

F#666 verdict matrix: proxy PASS ∧ target PASS → **SUPPORTED**.

## 4. Comparison with parent and siblings

| Experiment                                  | N  | Data          | Features       | Best top-1 |
|---------------------------------------------|----|---------------|----------------|-----------|
| `exp_semantic_router` (parent, KILLED)      | 15 | synthetic MC  | char-ngram 224d→64d | 27.3% (keyword) |
| **this (`exp_followup_semantic_router_macro`)** | **25** | **real HF**  | **MiniLM 384d** | **79.4% (cosine)** |
| `exp_p0_embedding_routing_n25` (sibling, supported) | 25 | real HF (same) | MiniLM 384d + TF-IDF + logistic | 88.8% (combined) |

**Consistency cross-check:** our cosine-centroid top-1 (79.42%) matches the sibling's
reported embed-centroid top-1 (79.4%) on the identical 25-domain mix — confirming the
data pipeline and embedding are consistent across experiments.

**Lift vs parent:** +52.1 percentage points (27.3 → 79.4). The parent's information-bottleneck
explanation was correct *for its data*; it was a synthetic-generator artifact, not a property
of routing.

## 5. What the experiment establishes

1. **Parent's ceiling was data-specific, not fundamental.** A 2–4% stationary-distribution
   variation on V=32 synthetic tokens is insufficient for 2.32 bits of within-cluster
   discrimination. Real text trivially provides that budget via distinct vocabulary and
   syntactic structure.
2. **Cosine centroid on real embeddings is sufficient for N=25 macro routing.** No trained
   classifier or combined features are needed to hit the 37.3% threshold — the linear
   centroid already achieves 79.4% top-1 and 94.1% top-3.
3. **Top-3 coverage enables hierarchical routing.** 94.1% top-3 means ensemble/fallback
   composition with K=3 adapters covers 94% of queries — sufficient for behavioral usability
   even when top-1 is wrong. This is the F#666 target-gate reasoning made concrete.

## 6. Caveats

- **No downstream quality measured.** Per F#666, routing accuracy ≠ target-adapter quality.
  This experiment measures routing only. Whether wrong-top-1 but correct-top-3 yields
  usable adapter composition is a separate question (adapter-dependent).
- **N=25, not N=50+.** "Macro scale" in the DB tag meant real embeddings on real data at
  a scale larger than the N=15 parent. Scaling further is a separate follow-up
  (`exp_p0_embedding_routing_n25` already hints that top-1 is stable at this scale, but
  inter-domain semantic-neighbor ambiguity grows with N per F#583).
- **MiniLM, not Gemma 4 base.** MiniLM was chosen because it's the standard semantic-routing
  backbone. Using Gemma 4's own hidden-state embedding is a follow-up — it would remove
  an extra encoder from the serving path but at the cost of domain-untuned representations.
- **No latency target.** Parent had K2/K3 on latency (passed). We dropped them — MiniLM
  CPU encode is ~48ms/query (sibling's finding), which is a serving concern but not what
  this experiment pre-registered.
- **Apple Accelerate BLAS warnings suppressed.** M-series numpy emits benign RuntimeWarnings
  on certain matmul edge cases; we validated correctness by cross-checking cosine_sim's
  79.42% top-1 against the sibling's independently-computed 79.4%.

## 7. Assumptions (per PLAN.md guardrail 1008 — autonomy)

- "Macro scale" in the DB title was interpreted as "real embeddings on real data at
  N > parent's 15". N=25 was selected to reuse cached HF parquets from the sibling
  `exp_p0_embedding_routing_n25` and to allow consistency cross-check. Going to N=50
  is a separate follow-up if needed.
- Since the experiment was registered with only a proxy KC (#1570), we added a target KC
  (#1946 top-3 ≥ 85%) before running — per PLAN.md §1 target-gated-kill rule (F#666).

## 8. Follow-ups (≤1 per PLAN.md success-criterion policy; none strictly P≤2 required)

1. **Gemma 4 hidden-state embeddings.** Does the base model's own mean-pooled hidden state
   match MiniLM on this task? If yes, drop the encoder from the serving path.
2. **Per-cluster top-1 failure modes.** Which domain pairs drive the 20% top-1 error?
   Expected: history-trio (world/european/us history per sibling F) and STEM overlaps.
3. **N=50 stress test.** Does cosine centroid remain ≥ 75% at N=50 with semantic-neighbor
   domains (MMLU physics ↔ astronomy)? Anchored to F#583 which showed TF-IDF fails under
   semantic-neighbor coverage.

## 9. References

- arXiv:1908.10084 — Sentence-BERT / MiniLM (Reimers & Gurevych 2019).
- F#525, F#666, F#431 — cited in MATH.md §7.
- Parent `exp_semantic_router` — 27.3% top-1 at N=15 synthetic (KILLED).
- Sibling `exp_p0_embedding_routing_n25` — 79.4% embed-centroid at identical N=25 mix.
