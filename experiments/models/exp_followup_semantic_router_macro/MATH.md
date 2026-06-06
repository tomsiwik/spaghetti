# Semantic Router at Macro Scale: Real Embeddings on Real Domains

**Experiment:** `exp_followup_semantic_router_macro`
**Type:** Verification (guided-exploration of parent's information-limit claim)
**Parent:** `exp_semantic_router` (KILLED, K1: best 27.3% < 70% threshold at N=15 synthetic)
**Grounding:** Finding #666 (target-gated kill rule), Finding #525 (real embeddings 89.9% at N=10), Finding #431 (TF-IDF 86.1% at N=25), PAPER.md of sibling `exp_p0_embedding_routing_n25` (MiniLM centroid 79.4% at N=25).

## 1. Claim

The parent `exp_semantic_router` reached ~27.3% domain accuracy at N=15 and attributed the
ceiling to a fundamental **information bottleneck** (synthetic Markov chains + V=32 vocabulary
compress within-cluster distinguishing information to ~1 bit < log₂(5)=2.32 bits; MATH.md §4.2).
This experiment tests whether the ceiling is **real-world** (a property of routing) or
**synthetic-data-specific** (a property of the Markov-chain + character-tokenizer generator).

**Theorem (Information-escape).**
Let `X_d` be text samples from domain `d`, `e = E(X_d) ∈ S^{D-1}` unit-normalized embeddings
from a pretrained sentence encoder `E` (MiniLM-L6-v2, D=384), and `P = {x ↦ argmax_d cos(e, μ_d)}`
the centroid cosine-router. If the embedding map `E` is trained to cluster semantically related
text (as MiniLM is, via contrastive training on 1B+ pairs), then:

    E[top-1 accuracy of P] ≥ α · (1 - H(domain | semantic) / log₂ N)

where `H(domain | semantic)` is the residual domain-entropy given the semantic cluster, and
`α` is the encoder's discriminative power for that cluster. On real domains drawn from
distinct HF datasets (GSM8K, CodeAlpaca, MedMCQA, MMLU subjects), `H(domain | semantic)` is
O(1) bits (distinct vocabularies and syntax), not compressed by a ~1-bit generator.

**Proof sketch.** The parent's bottleneck came from the *generator*: synthetic Markov chains
with `noise_scale=0.15` within a cluster produce stationary distributions that differ in only
2–4% of character frequencies (MATH.md §4.1). No feature extractor can recover the 2.32 bits
of distinguishing information beyond ~1 bit from the generator's entropy budget. Real text
domains carry this budget explicitly: a single token like `\begin{equation}` or `SELECT * FROM`
or `defendant` contributes >1 bit of posterior distinguishing information. MiniLM's encoder
was trained on NLI + multi-source pairs and is known to preserve this via dense L2-normalized
representations (Reimers & Gurevych 2019, arXiv:1908.10084). QED.

## 2. Setup and Notation

| Symbol | Definition | Value |
|--------|-----------|-------|
| N | number of domains | 25 |
| C | number of semantic clusters (for per-cluster analysis) | ~6 (STEM, humanities, professional, history, science, business) |
| D | sentence-embedding dimension | 384 (MiniLM-L6-v2) |
| N_train | texts per domain for training | 200 |
| N_test | texts per domain for test | 100 |
| N_exemplars | exemplars per domain for utterance router | 50 |
| P | LSH hyperplanes | 32 |

Domain set (identical to sibling `exp_p0_embedding_routing_n25` for cached-data reuse):
- **Real** (5): math (GSM8K), code (CodeAlpaca-20k), medical (MedMCQA), legal (MMLU law), finance (MMLU accounting/econometrics).
- **MMLU subjects** (20): high_school_geography, world_religions, philosophy, world_history, prehistory, european_history, us_history, astronomy, electrical_engineering, computer_security, logical_fallacies, high_school_statistics, formal_logic, government_and_politics, sociology, high_school_chemistry, high_school_physics, global_facts, management, marketing.

## 3. Routing Strategies (6 + oracle, identical menu to parent)

All strategies operate on real MiniLM-L6-v2 embeddings `e(x) = normalize(E(x)) ∈ S^{D-1}`
except the hash_ring (content-agnostic) and keyword_freq (character-frequency baseline).

### 3.1 Hash ring (content-agnostic baseline)
    h(x) = MD5(utf-8(x)) mod 2^32
Expected top-1 accuracy: 1/N = 4% (within-sample binomial variance).

### 3.2 Keyword frequency (character-level L2)
For each domain, compute mean character-frequency profile over training texts. Route query `x`
to `argmin_d ||freq(x) - μ_d^{freq}||_2²`. This is the parent's keyword strategy on real text
(not Markov-chain ids).

### 3.3 Cosine centroid (on MiniLM)
    μ_d = mean_{x ∈ D_d} e(x);  μ_d ← μ_d / ||μ_d||_2
    route(x) = argmax_d cos(e(x), μ_d)

### 3.4 LSH / SimHash partitioning
P=32 random hyperplanes. Each domain encoded by its **mean binary hash code** over training.
Route by max dot-product between query code and domain code means.

### 3.5 Utterance 1-NN
Store K=50 exemplars per domain. Route to the domain of the single nearest exemplar:
    d*(x) = domain(argmax_{j} e(x)^T e_j).

### 3.6 Utterance aggregated
Route by **mean** cosine to each domain's exemplars:
    s_d(x) = (1/K) Σ_k e(x)^T e_{d,k};  argmax_d s_d(x).

### 3.7 Oracle
Ground-truth labels — upper bound.

## 4. Kill Criteria (pre-registered, target-gated per F#666)

| ID | Text | Type | Threshold | Rationale |
|----|------|------|-----------|-----------|
| **K1570** | Best semantic strategy top-1 domain accuracy at macro (real embeddings) beats parent's micro result by ≥10pp | proxy | best_top1 ≥ 37.3% (= 27.3 + 10) | Parent ceiling was info-limited; real embeddings escape the Markov-chain entropy budget. |
| **K1946** | Best-strategy top-3 domain accuracy at N=25 ≥ 85% | **target** (F#666) | best_top3 ≥ 0.85 | Behavioral: enables hierarchical routing or ensemble fallback; downstream composition is usable when the correct domain is in top-3. |

**F#666 verdict matrix:**
- K1570 PASS ∧ K1946 PASS → **SUPPORTED** (information bottleneck is a synthetic-data artifact; macro routing is usable).
- K1570 FAIL ∧ K1946 FAIL → **KILLED** (real embeddings don't escape; routing is fundamentally limited).
- K1570 PASS ∧ K1946 FAIL → **PROVISIONAL** (proxy passes, top-1 works but top-3 doesn't help — surprising; file follow-up on top-K calibration).
- K1570 FAIL ∧ K1946 PASS → finding about the proxy (top-1 is mis-calibrated; real routing needs top-K).

## 5. Predictions

From sibling `exp_p0_embedding_routing_n25` (PAPER.md L11-21): MiniLM centroid achieves 79.4%
top-1 on the identical 25-domain mix. Expected:

| Strategy | Predicted top-1 | Predicted top-3 |
|----------|----------------|------------------|
| hash_ring | 4% | ~12% (3× chance) |
| keyword_freq | 40-55% (real text carries lexical signal) | 65-75% |
| **cosine_sim** | **75-82%** (matches sibling centroid) | **90-95%** |
| lsh_partition | 55-70% (binary codes lose angular resolution) | 78-88% |
| utterance_1nn | 70-80% (nearest-neighbor on sparse exemplars) | 85-93% |
| utterance_agg | 72-82% (aggregated ~= centroid) | 87-94% |
| oracle | 100% | 100% |

Best top-1: **cosine_sim ≈ 80%** (≥ 37.3% threshold → K1570 PASS).
Best top-3: **cosine_sim or utterance_agg ≈ 92%** (≥ 85% threshold → K1946 PASS).

## 6. Assumptions and Caveats

1. **MiniLM-L6-v2 distribution.** The encoder was trained on general-domain NLI; it is
   *not* fine-tuned for MMLU subject discrimination. Any gains at N=25 are from the
   pretrained embedding geometry, not domain-specific calibration.
2. **Data cached / deterministic.** Same cached HF parquets as sibling experiment; reuse
   `hf_hub_download` with fixed `random_state=42`. No data re-splits between strategies —
   all 6 see the same train/test partition.
3. **"Macro scale" interpretation.** Parent was N=15 synthetic; this is N=25 real. The KC
   says "beats micro-Markov by ≥10pp" which is a same-axis comparison (domain top-1 accuracy
   on the chosen strategy set). Going to N=50 domains is a separate follow-up — the current
   framing tests whether real-embedding ≠ synthetic-ngram at comparable N.
4. **Cluster-labels heuristic.** We compute *top-3* (not per-cluster) for K1946 because the
   MMLU+real domain mix has no canonical cluster labeling. Top-3 coverage is the
   behaviorally-meaningful target (it's what an ensemble/fallback routing layer would use).
5. **No downstream expert quality tested.** F#666 establishes that routing accuracy and
   downstream quality can diverge. This experiment measures routing only; downstream oracle-gap
   requires adapters (separate experiment).
6. **MLX not involved.** Sentence-transformers + sklearn + numpy on CPU. MLX target model
   (Gemma 4) is not loaded; this is a pure routing benchmark. No `/mlx-dev` skill needed
   because no MLX code is written.
7. **`sentence-transformers==5.3.0`, `huggingface-hub==1.9.2`.** Same stack as sibling.

## 7. References

- arXiv:1908.10084 — Sentence-BERT / MiniLM backbone (Reimers & Gurevych 2019).
- arXiv:2212.04089 — Task arithmetic / LoRA merging (Ilharco et al.) — lineage of Pierre's room-model.
- arXiv:2402.09997 — LoraRetriever: contrastive routing for LoRA selection.
- Finding #666 — target-gated kill rule; proxy-accuracy ≠ target-quality.
- Finding #525 — Combined logistic 89.9% at N=10.
- Finding #431 — TF-IDF centroid 86.1% at N=25 (different feature space).
- Parent `exp_semantic_router` PAPER.md — 27.3% at N=15 synthetic, info-limited.
- Sibling `exp_p0_embedding_routing_n25` PAPER.md — MiniLM centroid 79.4% at same N=25 mix.
