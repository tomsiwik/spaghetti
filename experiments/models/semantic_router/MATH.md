# Semantic Router: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Range |
|--------|-----------|-------------|
| V | Vocabulary size | 32 (micro) |
| T | Context length | 16 (micro) |
| D | Embedding dimension (post-projection) | 64 (micro) |
| D_raw | Raw n-gram feature dimension | 224 = V + 128 + 64 |
| N | Number of expert domains | 15 |
| C | Number of clusters | 3 (5 domains each) |
| P | Number of LSH hyperplanes | 32 |
| K | Exemplars per domain (utterance router) | 50 |
| R | Random projection matrix | (D_raw, D) |

## 2. Feature Extraction: Character N-gram Embeddings

### 2.1 Raw Feature Construction

Given a token sequence x = (x_1, ..., x_T) with x_t in {0, ..., V-1}:

**Unigram features** (V = 32 dimensions):
    f_uni(x)_c = (1/T) * sum_{t=1}^T 1[x_t = c],  c in {0,...,V-1}

**Bigram features** (hashed to 128 buckets):
    f_bi(x)_b = (1/(T-1)) * sum_{t=1}^{T-1} 1[(x_t * V + x_{t+1}) mod 128 = b]

**Trigram features** (hashed to 64 buckets):
    f_tri(x)_b = (1/(T-2)) * sum_{t=1}^{T-2} 1[(x_t*V^2 + x_{t+1}*V + x_{t+2}) mod 64 = b]

**Concatenated raw feature:**
    f(x) = [f_uni(x); f_bi(x); f_tri(x)]  in R^{224}

### 2.2 Random Projection to D Dimensions

Following the Johnson-Lindenstrauss lemma, a random Gaussian projection
approximately preserves pairwise distances:

    e(x) = f(x) R / ||f(x) R||_2

where R in R^{D_raw x D} has i.i.d. N(0, 1/D) entries (column-normalized).

The L2-normalized embedding e(x) in R^D lies on the unit sphere S^{D-1}.

**Why n-grams instead of bag-of-words:** The unigram feature alone (used in
content_aware_routing) captures only marginal character frequencies. Bigram
and trigram features capture transition patterns (the Markov chain structure
that defines each domain). This richer representation gives the router more
signal to discriminate domains within a cluster.

## 3. Routing Strategies

### 3.1 Hash Ring (Baseline)

Maps query to ring position via MD5:

    h(x) = MD5(bytes(x)) mod 2^32

Expert i has virtual nodes at positions MD5("name_i:v") for v in {0,...,149}.
Route to the expert whose virtual node is closest (clockwise) to h(x).

**Complexity:** O(log(N * 150)) = O(log N) per query.
**Domain accuracy:** 1/N = 6.7% (uniform random by construction).

### 3.2 Keyword Frequency Matching

For each domain d, compute the empirical character frequency profile:

    p_d(c) = (1/|D_d|) * sum_{x in D_d} (1/T) * sum_t 1[x_t = c]

Route query x to argmin_d ||f_uni(x) - p_d||_2^2.

**Complexity:** O(N * V) per query.

### 3.3 Cosine Similarity to Expert Centroids

Compute centroid embedding for each domain:

    mu_d = mean_{x in D_d} e(x),  then  mu_d <- mu_d / ||mu_d||

Route query x to argmax_d cos(e(x), mu_d) = argmax_d e(x)^T mu_d.

**Complexity:** O(N * D) per query (single matrix multiply for batch).

### 3.4 SimHash LSH Partitioning

Partition embedding space using P random hyperplanes {w_1, ..., w_P}:

    hash(e) = (sign(e^T w_1), ..., sign(e^T w_P)) in {0,1}^P

For each domain d, compute the mean hash code:

    h_d = mean_{x in D_d} hash(e(x)) in [0,1]^P

Route query x to argmax_d hash(e(x))^T h_d (fraction of matching bits).

**Complexity:** O(D * P + N * P) per query.

**Collision probability (SimHash guarantee):** For two unit vectors u, v:

    Pr[sign(u^T w) = sign(v^T w)] = 1 - theta(u,v)/pi

where theta is the angle between u and v. So P bits give Hamming distance
that approximates angular distance.

### 3.5 Utterance Matching (Semantic Router Pattern)

Store K exemplar embeddings per domain: {e_{d,1}, ..., e_{d,K}} for each d.
Total exemplar matrix E in R^{NK x D}.

**1-NN variant:** Route query x to the domain of the nearest exemplar:

    d*(x) = domain(argmax_{j=1}^{NK} e(x)^T E_j)

**Aggregated variant:** For each domain, compute mean similarity to its exemplars:

    s_d(x) = (1/K) * sum_{k=1}^K e(x)^T e_{d,k}

Route to argmax_d s_d(x).

**Complexity:** O(NK * D) per query (dominates at large K).
With K=50, N=15: 750 * 64 = 48K multiply-adds per query.

### 3.6 Oracle

Perfect routing using ground-truth domain labels. Upper bound: 100% accuracy.

## 4. Accuracy Analysis

### 4.1 Domain vs Cluster Discrimination

The synthetic data is generated from Markov transition matrices with:
- **Inter-cluster separation:** distinct transition structure across 3 clusters
- **Intra-cluster variation:** small perturbation (noise_scale=0.15) between
  5 domains in the same cluster

**Theoretical limit on within-cluster accuracy:**

Each domain within a cluster differs by O(noise_scale * 0.5) = O(0.075) in
transition probabilities. The character frequency feature captures only the
stationary distribution of the Markov chain, not the full transition matrix.

For a Markov chain with transition matrix T, stationary distribution pi satisfies:

    pi = pi * T

Two chains T_1, T_2 with ||T_1 - T_2||_F ~ epsilon have:

    ||pi_1 - pi_2|| <= epsilon / (1 - lambda_2)

where lambda_2 is the second-largest eigenvalue modulus. For near-uniform chains,
lambda_2 ~ 0.5-0.8, so stationary distributions converge quickly but the
difference is compressed: a 7.5% transition difference may yield only 2-4%
frequency difference.

With V=32 vocabulary, this gives ~ 0.5-1.3 bits of distinguishing information
between within-cluster domains -- well below the log2(5) = 2.32 bits needed
for 100% within-cluster discrimination.

### 4.2 The Fundamental Information Bottleneck

Given C=3 clusters with D/C = 5 domains each:

- **Cluster routing** requires log2(3) = 1.58 bits. Available from inter-cluster
  frequency differences (strong signal). All routers achieve >90% except LSH.

- **Domain routing** requires log2(15) = 3.91 bits. Of this, 1.58 bits come
  from cluster identity. The remaining 2.32 bits must come from within-cluster
  domain features -- which are limited by the noise_scale parameter.

**Expected domain accuracy at micro scale:**

    acc_domain ~ acc_cluster * acc_within_cluster
              ~ 0.95 * (0.20 + epsilon)  [epsilon ~ 0.05 from weak domain signal]
              ~ 0.24

This matches the observed ~25% across all strategies.

## 5. Latency Model

All routers operate on pre-computed embeddings. The embedding computation
itself (n-gram extraction + projection) is O(T * V + D_raw * D) ~ O(T * V)
per query.

| Strategy | Per-query ops | Measured (us) | Scales with |
|----------|-------------|---------------|-------------|
| Hash ring | O(T + log N) | 2.1 | log N |
| Keyword | O(T + N*V) | 5.7 | N * V |
| Cosine | O(N*D) | 0.19 | N * D |
| LSH | O(D*P + N*P) | 0.33 | N * P |
| Utterance 1-NN | O(NK*D) | 1.75 | NK * D |
| Utterance agg | O(NK*D) | 1.96 | NK * D |

At production scale (N=500, D=4096, K=50):
- Cosine: 500 * 4096 ~ 2M ops ~ 10us (vectorized)
- LSH: 4096 * 256 + 500 * 256 ~ 1.2M ops ~ 6us
- Utterance: 25000 * 4096 ~ 100M ops ~ 500us (may need FAISS)

## 6. Assumptions

1. Character n-gram features capture sufficient domain signal for routing
   (stronger assumption at micro with V=32 than macro with V=32K)
2. Random projection approximately preserves angular structure (JL lemma;
   valid for D_raw=224 -> D=64 with moderate distortion)
3. Domain separability in embedding space reflects downstream expert utility
   (not tested here -- downstream quality is vacuous at micro)
4. Latency measurements on Apple Silicon (M-series) transfer directionally
   to production hardware (absolute values differ)
5. Synthetic Markov chain domains reflect the cluster structure of real
   knowledge domains (strong assumption; real domains have richer structure)
