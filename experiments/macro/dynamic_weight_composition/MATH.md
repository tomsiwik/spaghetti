# Dynamic Weight Composition: Mathematical Foundations

## 1. Setup and Notation

**Base model:** Qwen2.5-7B with frozen weights W_base in R^{d x d} per layer, d=3584, L=28 layers.

**Expert library:** N=50 pilot LoRA adapters, each rank r=16, stored as
Delta_k = B_k A_k where A_k in R^{r x d}, B_k in R^{d x r}. Applied to all
modules (q/k/v/o/gate/up/down), total ~6MB per expert.

**Query:** Input token sequence x = (x_1, ..., x_T) with hidden states
h_t^l in R^d at layer l, position t.

**Composition:** Given expert set S = {1, ..., N} and per-expert weights
w = (w_1, ..., w_N), the composed model output at each layer is:

    y^l = W_base^l h^l + sum_{k in S} w_k (B_k^l A_k^l) h^l

**Loss:** Next-token prediction cross-entropy on domain-specific eval text:

    L(w; D) = -(1/|D|) sum_{(x,y) in D} log P(y | x; W_base + sum_k w_k Delta_k)

**PPL:** exp(L(w; D))

## 2. Three Composition Strategies

### 2.1 Strategy A: Equal-Weight Pre-Merge (Current SOLE Default)

    w_k = 1/N  for all k in S

Pre-merge all N adapters into base weights once:

    W_merged = W_base + (1/N) sum_k Delta_k

Inference is a single forward pass through W_merged.

**Complexity:**
- Merge: O(N * M * d * r) where M=7 modules per layer, done once
- Inference: O(1) -- same as base model
- Latency overhead: 0ms per query

**Known results (pilot50 at N=5):**
- All domains beat base (K3 PASS)
- Mean degradation vs single-expert: +127% (K1 FAIL)
- Medical worst: +247%

### 2.2 Strategy B: Top-k Selection

Select only the k most relevant experts per query, then pre-merge those k:

    S_k(x) = argtop_k { score(x, Delta_j) : j in {1, ..., N} }
    w_j = 1/k  if j in S_k(x), else 0

The scoring function determines relevance. We test:

**B1: PPL-probe scoring.** For each expert j, maintain a probe buffer
P_j of n=10 representative examples from the expert's training domain.
Score by cross-entropy on the probe:

    score_ppl(x, Delta_j) = -L(W_base + Delta_j; P_j)

This requires N forward passes through individual expert models on
n=10 short sequences. However, we can precompute these scores once
per expert (they are query-independent, measuring expert quality on
its own domain, not relevance to the specific query).

**Wait -- that scores expert quality, not query relevance.** For
query-conditional scoring, we need:

    score_ppl(x, Delta_j) = -L(W_base + Delta_j; x)

i.e., evaluate the query x under each expert model. This requires
N forward passes of the query.

**Optimization:** At N=50, we batch-evaluate the query under all 50
single-expert models. With PEFT adapter switching, each eval is cheap
(no weight reload, just pointer swap). But N=50 forward passes is 50x
the base inference cost.

**Cheaper alternative -- embedding-based scoring:**

    score_emb(x, Delta_j) = cos(embed(x), c_j)

where embed(x) is the mean of the last hidden state of x through the
base model (computed during the initial forward pass anyway), and c_j
is the precomputed centroid of expert j's training data embeddings.
Cost: one matrix-vector multiply of (N x d), negligible.

**B2: Embedding-based top-k.** Precompute domain centroids from
training data. At inference, compute query embedding from the base
model's hidden states, select top-k by cosine similarity.

### 2.3 Strategy C: PPL-Probe Weighted Merge

Weight each expert proportionally to its relevance score:

    w_j = exp(score_j / tau) / sum_i exp(score_i / tau)

where tau is a temperature parameter and score is one of:

**C1: PPL-probe weights (query-conditional).**
Score each expert by its PPL on the current query:

    score_j = -L(W_base + Delta_j; x)

Then softmax with temperature tau to get weights.

**C2: Embedding-cosine weights.**
    score_j = cos(embed(x), c_j)

This is ~free but micro experiments showed weight-space signals have
low oracle correlation (r=0.023 for activation, r=-0.245 for logit_diff).
However, embedding cosine was not tested at micro -- it may work better
at d=3584 where embeddings are richer.

**C3: Hybrid -- embedding top-k then PPL-probe weight.**
First select top-k by embedding similarity (cheap filter), then weight
those k experts by PPL-probe score (expensive but only k forward passes
instead of N).

    S_k(x) = argtop_k { cos(embed(x), c_j) : j }
    score_j = -L(W_base + Delta_j; x)   for j in S_k
    w_j = softmax(score_j / tau)

Cost: 1 base forward pass + k expert forward passes. At k=5, this is
6x base cost instead of 51x.

## 3. Theoretical Analysis

### 3.1 Dilution Under Equal Weight

With N experts and equal weight 1/N, each relevant expert contributes
(1/N) of its delta. For a single-domain query where expert j is the
only relevant one:

    signal = (1/N) Delta_j h
    noise = (1/N) sum_{i != j} Delta_i h

Under SOLE structural orthogonality (cos ~ 0.0002 at d=896), the noise
terms are orthogonal to the signal:

    ||noise||^2 = (1/N)^2 sum_{i != j} ||Delta_i h||^2

    SNR = ||signal||^2 / ||noise||^2 = ||Delta_j h||^2 / sum_{i != j} ||Delta_i h||^2

If all experts have similar activation norms ||Delta_i h|| ~ sigma:

    SNR = 1 / (N-1)

At N=50, SNR = 1/49 = 0.020 (-17 dB). The useful signal is 2% of the
total perturbation. This explains the +127% degradation at N=5 (SNR=25%)
and predicts catastrophic dilution at N=50.

### 3.2 Top-k Restores SNR

With top-k (k relevant experts, equal weight):

    SNR_topk = k / (k - k_noise)

where k_noise is the number of selected experts that are actually irrelevant.
With perfect selection (k_noise = 0):

    SNR_topk = 1   (if all k experts are equally relevant)

Even with imperfect selection (1 noise expert in top-5):

    SNR_topk = 4 / 1 = 4.0   (+6 dB)

vs equal-weight N=50: SNR = 1/49 = 0.020 (-17 dB). A 200x improvement.

### 3.3 Weighted Merge Further Improves

Weighted composition with oracle weights achieves:

    L(w*; D) <= L(w_topk; D) <= L(w_eq; D)

The gap between top-k and weighted depends on how much variation exists
among the k selected experts' relevance. From micro results:

- Equal weight: gap = -0.6% mean, +24.3% worst
- PPL-probe weighted: gap = -9.94% mean, +6.3% worst (33pp swing)
- Loss oracle: gap = -10.06% mean, +6.3% worst

The probe-to-oracle gap is only 0.12pp, confirming that a small probe
(10 examples) captures essentially all the information.

### 3.4 Latency Budget

Kill criterion: overhead < 50ms per query.

| Strategy | Forward passes | Est. time (A5000) | Within budget? |
|----------|---------------|-------------------|----------------|
| A: Equal pre-merge | 0 | 0ms | Yes |
| B1: PPL top-k (N=50) | 50 | ~1500ms | NO |
| B2: Embed top-k | 0 + matmul | ~0.1ms | Yes |
| C1: PPL weights (N=50) | 50 | ~1500ms | NO |
| C2: Embed weights | 0 + matmul | ~0.1ms | Yes |
| C3: Hybrid (k=5) | 5 | ~150ms | NO (but close) |

PPL-probe scoring of all N=50 experts per query is too expensive.
The viable strategies are:

1. **Embedding-based top-k** (B2): ~free, quality TBD at macro
2. **Embedding top-k + PPL rerank of top-k** (C3 with small k):
   k forward passes. At k=3 (~90ms), borderline. At k=2 (~60ms), tight.
3. **Pre-computed PPL scores** (not query-conditional): score each expert
   on its own domain probes. These are static relevance scores (how good
   is expert j at domain j), not dynamic (how relevant is expert j to
   query x). Can be precomputed once and cached.

### 3.5 Practical Compromise: Cached PPL + Embed Routing

The production-viable approach combines:

1. **Offline:** Compute per-domain PPL scores for all experts (matrix R
   of shape N x D_domains). This tells us which experts are good at which
   domains.

2. **Online:** Classify query into domain using embedding cosine (~0.1ms).
   Look up pre-computed expert-domain scores. Weight experts by domain
   relevance.

This is equivalent to embedding-based routing but uses PPL quality scores
instead of raw cosine similarity as weights. Zero per-query forward passes
beyond the one base forward pass.

## 4. Experimental Design

### 4.1 Conditions

| ID | Strategy | Merge | Per-query cost | Description |
|----|----------|-------|---------------|-------------|
| A | equal_premerge | Static 1/N | 0 | Current SOLE default |
| B2 | embed_topk | Dynamic top-k | ~0.1ms | Embed cosine selection |
| C2 | embed_weighted | Dynamic softmax | ~0.1ms | Embed cosine weights |
| C3_k3 | hybrid_k3 | Dynamic top-3+PPL | ~90ms | Embed filter + PPL rerank |
| D | ppl_precomputed | Static domain | 0 | Cached quality weights |

Strategy C1 (full PPL probe of all N) is excluded from the experiment
because it exceeds the latency budget by 30x. It serves as oracle only.

### 4.2 Metrics

For each strategy, evaluate on eval data for all 50 domains:

1. **Per-domain PPL** -- L(w; D_domain) for each domain
2. **Mean degradation vs single-expert** -- (PPL_composed - PPL_single) / PPL_single
3. **Domains worse than base** -- count where PPL_composed > PPL_base
4. **Wall-clock latency** -- time per query including routing overhead

### 4.3 Embedding Centroids

For each of the 50 experts, compute the centroid of its training data
embeddings by:

1. Sample 100 training examples from domain j
2. Forward pass through base model (no adapters)
3. Take mean of last hidden layer across tokens and examples
4. Centroid c_j = mean(h_base(x) for x in train_j), shape (d,)
5. Normalize: c_j = c_j / ||c_j||

Query embedding: same procedure on the eval query.

## 5. Kill Criteria Formalization

**K1: Dynamic > equal-weight by >= 2% (quality).**
Let PPL_best be the mean cross-domain PPL of the best dynamic strategy.
Let PPL_eq be the mean PPL of equal-weight pre-merge.

    improvement = (PPL_eq - PPL_best) / PPL_eq * 100

K1 PASSES if improvement >= 2%.

**K2: Latency < 50ms (speed).**
The winning dynamic strategy must add < 50ms per query.

**K3: Pareto optimality.**
Plot (latency, quality) for all strategies. If equal-weight pre-merge
is on the Pareto frontier AND no dynamic strategy dominates it, KILL.

## 6. Assumptions

1. Pilot 50 adapters have meaningful domain specialization (validated:
   98% win rate vs base, 42.2% avg PPL improvement).

2. Eval data exists for all 50 domains in /workspace/llm/data/distillation/.

3. 4-bit NF4 quantization does not differentially affect dynamic vs
   static composition.

4. Embedding cosine similarity correlates with domain relevance at
   d=3584 (not tested at macro; micro showed r=0.023 for activation
   but embedding cosine was untested).

5. PEFT adapter switching is fast enough for sequential expert evaluation
   (confirmed by pilot50 composition quality experiment: ~12 min for
   5 adapters x 5 domains).

## 7. Worked Example at Macro Scale

**Query:** "Write a Python function to calculate body mass index"
**Relevant domains:** python, medical
**N=50 experts total**

**Strategy A (equal pre-merge):**
- Each expert weighted 1/50 = 0.02
- Python expert contributes 0.02 * Delta_python
- Medical expert contributes 0.02 * Delta_medical
- 48 irrelevant experts contribute 0.96 * noise
- SNR = 0.04 / 0.96 = 0.042 (-14 dB)

**Strategy B2 (embed top-k=5):**
- Query embedding closest to: python, medical, data_science, bash, sql
- Each weighted 1/5 = 0.2
- Python contributes 0.2 * Delta_python
- Medical contributes 0.2 * Delta_medical
- 3 semi-relevant experts contribute 0.6 * (weak signal + some noise)
- SNR ~ 0.4 / 0.6 = 0.67 (-1.7 dB), 16x better than A

**Strategy C2 (embed weighted, all 50):**
- Softmax concentrates weight on python (~0.15), medical (~0.12),
  data_science (~0.08), with 47 others sharing ~0.65
- Better than equal-weight but still diluted by tail

**Strategy C3 (hybrid k=3):**
- Embed filter: python, medical, data_science
- PPL rerank: python (score -1.5), medical (score -2.1), data_science (score -3.8)
- Weights after softmax(tau=1): python 0.52, medical 0.31, data_science 0.17
- SNR ~ 0.83 / 0.17 = 4.9 (+6.9 dB), 117x better than A

**Expected PPL improvement (from micro scaling):**
- Strategy A: +127% degradation at N=5, likely +300-500% at N=50
- Strategy B2/C2: ~+20-50% (uncertain, embedding quality TBD)
- Strategy C3: ~+5-15% (near single-expert quality)

## 8. Computational Budget

| Phase | GPU time | Description |
|-------|----------|-------------|
| Compute 50 centroids | ~15 min | 100 examples x 50 domains forward pass |
| Equal-weight compose at N=5,25,50 | ~25 min | 3 compositions x 5 eval domains each |
| Embed top-k/weighted | ~25 min | Same composition cost, different weights |
| Hybrid k=3 PPL rerank | ~40 min | 3 expert forward passes per query x domains |
| Oracle (full PPL N=50) | ~120 min | 50 forward passes per query (reference only) |

**Total estimated: ~60 min (without oracle), ~180 min (with oracle).**
Budget: aim for 60 min main experiment + optional oracle extension.
