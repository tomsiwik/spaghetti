# Cross-Domain Semantic Transfer: Mathematical Foundations

## 1. Problem Statement

Given a frozen base model with weights W in R^{d x d} per layer (d = 4096 for
Qwen2.5-7B), and N = 50 domain-specialized LoRA experts {(A_i, B_i)}_{i=1}^N
where A_i in R^{r x d}, B_i in R^{d x r}, r = 16, we ask:

Does the composed model W + sum_{i in S} w_i * B_i @ A_i produce coherent
responses on queries Q that require **simultaneous** knowledge from two domains
i, j in {1, ..., N}?

**Sequential chaining** (micro precedent): Q = "compute 12 + 34, then reverse
the result." Here domain A (arithmetic) produces an intermediate result, and
domain B (reversal) operates on that output. The two domains never interact --
they compose temporally.

**Semantic transfer** (this experiment): Q = "Explain how a binary search
algorithm is analogous to differential diagnosis in medicine." Here both domains
must be active simultaneously to produce a single coherent answer that bridges
the conceptual gap between them. There is no intermediate result; the output
requires joint reasoning.

## 2. Formal Distinction: Sequential vs Semantic

### 2.1 Sequential Composition

A query Q_seq can be decomposed as:

  Q_seq = f_j(f_i(x))

where f_i is the function computed by expert i and f_j is the function computed
by expert j. The composition is **separable** -- the output of f_i is the input
to f_j. In weight space:

  y_seq = (W + B_j A_j) * (W + B_i A_i) * x

The cross term B_j A_j B_i A_i x is O(||B_j||_F ||A_j||_F ||B_i||_F ||A_i||_F)
which is small when deltas are small relative to W.

### 2.2 Semantic Composition

A query Q_sem requires a response y where:

  y_sem = g(knowledge_i, knowledge_j)

where g is a non-separable function -- neither domain alone can produce the
answer, and there is no sequential ordering. In weight space with additive
composition:

  y_sem = (W + w_i B_i A_i + w_j B_j A_j) * x

The key question: does additive perturbation of W create a model that can
perform the non-separable reasoning g? Or does it merely produce an
interpolation of two separate domain behaviors?

### 2.3 Why Additive Composition Might Work for Semantic Transfer

Hypothesis: the base model W already encodes general reasoning capabilities
(analogy-making, explanation, comparison). The LoRA experts shift the model's
attention toward domain-specific knowledge without destroying the cross-domain
reasoning circuits in W. Under this view:

  (W + B_i A_i + B_j A_j) * x
  = W * x + B_i A_i * x + B_j A_j * x

The base model provides the bridging logic (analogies, explanations), while
each expert provides domain-specific factual grounding. Since r/d = 16/4096
= 0.39%, each expert perturbs less than 0.4% of the model's effective
dimensionality, leaving the vast majority of W's cross-domain reasoning
capability intact.

### 2.4 Why Additive Composition Might Fail

Counter-hypothesis: semantic transfer requires non-linear interaction between
domain knowledge that additive composition cannot provide. The cross-domain
reasoning g(knowledge_i, knowledge_j) may require:

  1. Domain-specific vocabulary mappings that conflict
  2. Attention patterns specialized to one domain that suppress the other
  3. Knowledge that is not present in either expert alone (emergent from
     their interaction)

Under the counter-hypothesis, the composed model would produce outputs that
are domain-correct for each expert individually but fail to bridge them:
e.g., producing a medical explanation that uses no programming concepts, or
a programming explanation with no medical terminology.

## 3. Evaluation Framework

### 3.1 Query Taxonomy

We define three categories of cross-domain queries, ordered by semantic
integration difficulty:

**Level 1 -- Domain Translation:** Apply domain B's framework to domain A's
content. Example: "Explain the Python sorting algorithm as if describing a
medical triage procedure." Both domains required but the structure is A-content
expressed in B-framework.

**Level 2 -- Analogical Reasoning:** Draw structural parallels between domains.
Example: "How is recursion in programming similar to the immune system's
response cascade?" Requires understanding deep structures in both domains to
find meaningful parallels.

**Level 3 -- Creative Synthesis:** Generate novel insights at the intersection.
Example: "Design a medical diagnostic protocol inspired by database query
optimization techniques." Requires genuine integration, not just mapping.

### 3.2 Quality Metrics

For each query Q and response y, we measure:

**M1: Domain Coverage Score (0-1)**
  DC(y, i, j) = min(relevance(y, domain_i), relevance(y, domain_j))

Both domains must appear in the response. A response that discusses only domain
i scores relevance(y, domain_j) = 0, so DC = 0 regardless of how well it
covers domain i. Operationalized via LLM-as-judge.

**M2: Integration Score (0-1)**
  IS(y) = 1 - separability(y)

where separability(y) measures whether the response can be decomposed into
non-interacting domain-A and domain-B paragraphs. A fully integrated response
(concepts from both domains in the same sentences) scores IS = 1. A response
with "Part 1: Code explanation. Part 2: Medical explanation." scores IS ~ 0.
Operationalized via LLM-as-judge.

**M3: Answer-Conditioned Perplexity**
  PPL_cond(y | Q) = exp(-1/T * sum_{t=1}^T log p(y_t | y_{<t}, Q))

Standard next-token perplexity of a high-quality reference answer y given the
query Q. Lower is better. Measured for base, each single expert, and composed
model.

**M4: Win Rate via Pairwise Comparison**
For each query, a judge LLM compares responses from:
  (a) base model
  (b) single expert_i
  (c) single expert_j
  (d) composed (expert_i + expert_j)

Win rate of composed vs each alternative is reported.

### 3.3 Kill Criteria Formalization

**K1:** Let D_base be the set of domain coverage scores for the base model, and
D_comp for the composed model. Define degradation:

  degradation = 1 - mean(D_comp) / mean(D_base)

K1 KILL if degradation > 0.20 (20% worse than base).

**K2:** For each query Q, let score(composed) and score(best_single) be the
win-rate or quality score of the composed vs best single expert:

  K2_fail_rate = |{Q : score(composed) < score(best_single)}| / |queries|

K2 KILL if K2_fail_rate > 0.50 (composed worse than best single expert on
more than half the queries).

## 4. Domain Pair Selection

From the 50 pilot adapters, we select domain pairs that maximize diversity of
semantic transfer difficulty:

**Code x Science pairs** (high semantic distance):
- python x medical: "Explain this sorting algorithm as a diagnostic protocol"
- python x physics: "Describe how Python's garbage collector relates to entropy"

**Code x Code pairs** (low semantic distance, control):
- python x rust: "Compare Rust's ownership model to Python's garbage collection"
- python x sql: "Rewrite this Python data pipeline as SQL queries"

**Science x Science pairs** (medium semantic distance):
- medical x chemistry: "How do pharmaceutical drug interactions work at the
  molecular level?"
- biology x statistics: "Explain evolutionary fitness using statistical
  hypothesis testing"

**Code x Humanities pairs** (very high semantic distance):
- python x legal: "Write a Python function that models contract breach analysis"
- math x ethics: "What are the ethical implications of Bayes' theorem in
  criminal justice?"

**Minimum 8 domain pairs, 10 queries per pair = 80 queries total.**

## 5. Composition Strategies

### 5.1 Equal-Weight Merge (Baseline)

  W_comp = W + 0.5 * (B_i A_i + B_j A_j)

Known from pilot-50: this dilutes each expert to 50% contribution.

### 5.2 PPL-Probe Weighted (Micro-Validated)

  w_k = softmax(-PPL_k / tau)  for k in {i, j}

where PPL_k is the perplexity of expert k on a 10-example probe buffer.
From micro: r=0.990 oracle correlation.

### 5.3 Single Expert (Hash-Ring Scenario)

  W_single = W + B_k A_k  where k = argmin PPL_k

Tests whether one expert alone is sufficient for semantic transfer.

## 6. Statistical Design

- 8+ domain pairs
- 10 queries per pair per difficulty level (Levels 1-3)
- 3 composition strategies (equal, PPL-weighted, single-expert)
- Evaluation via LLM-as-judge (Qwen2.5-72B or GPT-4o) with structured rubric
- 3 seeds per configuration (temperature sampling)
- Total: 8 pairs * 30 queries * 3 strategies * 3 seeds = 2,160 evaluations

Judge evaluations at ~0.5s each: ~18 minutes total judge time.
Model generation at ~2s each per configuration: ~8 pairs * 30 * 4 configs * 2s
= ~32 minutes generation time.

## 7. Expected Outcomes

**If H0 (additive composition enables semantic transfer):**
- DC scores for composed model >= 0.7 (both domains present)
- IS scores for composed model > 0.3 (non-trivial integration)
- Win rate of composed vs base > 0.5 on Level 1-2 queries
- Degradation < 20% (K1 PASS)
- Composed wins vs best-single on > 50% of queries (K2 PASS)

**If H1 (composition fails for semantic transfer):**
- DC scores for composed model drop toward single-expert levels (one domain
  dominates, the other vanishes)
- IS scores near zero (separable responses)
- Composed consistently loses to the better single expert

## 8. Worked Example

**Query:** "Explain how a binary search algorithm is analogous to differential
diagnosis in medicine."

**Base model (W):** The 7B base likely produces a passable response since both
concepts are in its pretraining data, but may lack depth in either domain.

**Python expert (W + B_py A_py):** Excellent on binary search mechanics, but
medical analogies may be superficial or absent.

**Medical expert (W + B_med A_med):** Excellent on differential diagnosis, but
binary search explanation may be vague.

**Composed (W + 0.5 B_py A_py + 0.5 B_med A_med):** If semantic transfer works,
the response should:
  - Correctly describe binary search (halving the search space)
  - Correctly describe differential diagnosis (ruling out conditions)
  - Draw the structural parallel (both are divide-and-conquer on uncertainty)

**PPL-weighted composition:** If medical domain contributes more to this analogy
query, weights might be w_med=0.6, w_py=0.4, emphasizing the medical framework
while retaining enough CS knowledge.

## 9. Computational Cost

| Component | Cost |
|-----------|------|
| Load base model (4-bit) | ~30s, ~6GB VRAM |
| Generate responses (80 queries * 4 configs * 3 seeds) | ~32 min |
| PPL computation for probe weighting | ~5 min |
| LLM judge evaluation | ~18 min |
| Total | ~55 min |

Well within the 2-hour macro budget.

## 10. Assumptions

1. **Base model has cross-domain reasoning capability.** Qwen2.5-7B has seen
   enough pretraining data to bridge domains when prompted. If the base model
   cannot answer semantic transfer queries at all, the experiment reduces to
   "can experts add domain knowledge to a base model?" which is already proven.

2. **LLM-as-judge is reliable for integration scoring.** The structured rubric
   should produce consistent scores. We report inter-rater agreement if using
   multiple judges.

3. **10 queries per pair per level is sufficient.** Given 3 seeds, this gives
   30 observations per domain-pair-level, enough for reliable mean estimation
   (SE = std / sqrt(30) ~ 0.18 * std).

4. **Domain pairs from pilot-50 span representative semantic distances.** The
   pilot set includes code (python, rust, cpp, java, go, bash, sql), science
   (medical, biology, chemistry, physics, genetics, neuroscience), humanities
   (legal, ethics, marketing, finance), and reasoning (math, statistics,
   logic-puzzles). This provides diverse transfer distances.
