# N-gram Expert Mixing: Mathematical Foundations

## Overview

Mix a statistical n-gram language model with a neural model at inference time
using entropy-adaptive weighting. The n-gram model is FREE (zero training,
built from frequency counts) and provides a complementary signal: it captures
local character patterns that a small neural model may underfit.

## Notation

| Symbol | Shape/Type | Description |
|--------|-----------|-------------|
| V | scalar | Vocabulary size (28 for char-level names) |
| T | scalar | Sequence length |
| n | scalar | N-gram order (2 = bigram, 3 = trigram, etc.) |
| c(w_{t-n+1:t}) | scalar | Count of n-gram ending at position t |
| c(w_{t-n+1:t-1}) | scalar | Count of (n-1)-gram context |
| p_ng(w_t | ctx) | R^V | N-gram probability distribution |
| p_nn(w_t | x_{<t}) | R^V | Neural model probability distribution |
| p_mix(w_t) | R^V | Mixed probability distribution |
| H(p) | scalar | Shannon entropy of distribution p |
| H_max | scalar | Maximum entropy = log(V) |
| alpha(t) | [0,1] | Mixing weight for n-gram at position t |

## N-gram Model with Backoff

### Frequency Estimation

For n-gram order n, the maximum-likelihood estimate is:

    p_n(w_t | w_{t-n+1:t-1}) = c(w_{t-n+1:t}) / c(w_{t-n+1:t-1})

With add-delta smoothing (delta = 0.1):

    p_n(w_t | ctx) = (c(ctx, w_t) + delta) / (c(ctx) + delta * V)

### Stupid Backoff (Brants et al., 2007)

For a K-gram model with backoff factor gamma_bo = 0.4:

    S(w_t | ctx_K) = c(ctx_K, w_t) / c(ctx_K)               if c(ctx_K, w_t) > 0
    S(w_t | ctx_K) = gamma_bo * S(w_t | ctx_{K-1})           otherwise

The backoff factor accumulates multiplicatively: if we fall through m levels
before finding a count, the score is gamma_bo^m * c(ctx_{K-m}, w_t) / c(ctx_{K-m}).

At the unigram level with smoothing:
    S(w_t) = gamma_bo^(K-1) * (c(w_t) + delta) / (N_total + delta * V)

The scores are then normalized to probabilities:
    p_ng(w_t | ctx) = S(w_t | ctx) / sum_v S(v | ctx)

### Per-Domain Tables

For D domains, we maintain D separate n-gram tables. At inference, we select
the domain-specific table (or use a global table for the baseline).

## Entropy-Adaptive Mixing

### Entropy Computation

For the n-gram distribution at position t:

    H_ng(t) = -sum_v p_ng(v | ctx) * log p_ng(v | ctx)

Normalized entropy (0 = perfectly confident, 1 = uniform):

    h(t) = H_ng(t) / log(V)

### Mixing Rule

The mixing weight alpha determines how much to trust the n-gram:

    alpha(t) = max(0, 1 - h(t) / tau)

where tau in (0, 1] is a threshold hyperparameter. When h(t) >= tau,
alpha = 0 (fully neural). When h(t) = 0, alpha = 1 (fully n-gram).

The mixed distribution:

    p_mix(w_t) = alpha(t) * p_ng(w_t | ctx) + (1 - alpha(t)) * p_nn(w_t | x_{<t})

### Key Property

The mixing is a CONVEX COMBINATION of valid probability distributions,
so p_mix is always a valid distribution (non-negative, sums to 1).

## Loss Computation

Cross-entropy loss with the mixed distribution:

    L_mix = -(1/T) * sum_t log p_mix(w_t | ctx)

Bits per byte (BPB) conversion (for character-level data):

    BPB = L_mix / log(2)

Perplexity:

    PPL = exp(L_mix)

## Improvement Bound (Qualified)

Mixing improves the prediction at position t ONLY when the n-gram model
assigns higher probability to the correct token than the neural model:

    p_ng(w_t* | ctx) > p_nn(w_t* | ctx)  =>  log p_mix(w_t*) > log p_nn(w_t*)

Formally, at such positions:

    log p_mix(w_t*) = log(alpha * p_ng(w_t*) + (1-alpha) * p_nn(w_t*))
                    > log p_nn(w_t*)                (by convexity, since p_ng > p_nn)

However, mixing can HURT at positions where p_ng(w_t*) < p_nn(w_t*), because
the n-gram model dilutes the neural probability with mass on wrong tokens:

    p_ng(w_t*) < p_nn(w_t*)  =>  p_mix(w_t*) < p_nn(w_t*)   (mixing hurts)

The net effect depends on the balance: if the fraction of "n-gram wins"
exceeds a threshold (related to the magnitude of wins vs losses), mixing
helps overall. Our 2-gram experiments confirm the bound does NOT universally
hold: 2-gram mixing hurts because the 2-gram model wins at only 34% of
positions, and its wins are small while its losses are large.

The expected improvement is:

    E[log p_mix - log p_nn] = sum_t [log p_mix(w_t*) - log p_nn(w_t*)] / T

This is positive when the n-gram model is sufficiently accurate (>=4-gram
in our experiments, with 55%+ win rate).

## Memory Analysis

### N-gram Table Size

For vocabulary V = 28 and max n-gram order K:
- Unigrams: V entries = 28
- Bigrams: up to V^2 entries = 784
- Trigrams: up to V^3 entries = 21,952
- 4-grams: up to V^4 entries = 614,656
- 5-grams: up to V^5 entries = 17,210,368

With sparse storage (only observed n-grams), actual size depends on data.
For names.txt (~32K names, avg length ~6 chars):
- Total characters: ~200K
- Observed bigrams: ~400 (out of 784)
- Observed trigrams: ~4,000 (out of 21,952)
- Observed 4-grams: ~15,000 (out of 614,656)
- Observed 5-grams: ~30,000 (out of 17M)

At 8 bytes per count + 8 bytes per key in a Python dict:
- 5-gram table: ~50K entries * 16 bytes = 800 KB per domain
- 5 domains: ~4 MB total

This is FAR below the 2 GB kill criterion (K2 PASS by ~500x margin).

## Worked Example (V=4, trigram, 2 domains)

Suppose V = {a, b, c, d}, and we have the sequence "abcab":

N-gram counts for "abcab":
- Bigrams: ab=2, bc=1, ca=1
- Trigrams: abc=1, bca=1, cab=1

At position 4, context = "ab":
- p_ng(c | ab) = 1/2 = 0.5 (trigram "abc" seen once, "ab" seen twice)
- p_ng(a | ab) = 1/2 = 0.5 (trigram "aba" from backoff? No, "ab*" only has "abc")

With smoothing (delta=0.1):
- p_ng(c | ab) = (1+0.1)/(2+0.4) = 1.1/2.4 = 0.458
- p_ng(others) = 0.1/2.4 = 0.042 each

H_ng = -0.458*log(0.458) - 3*0.042*log(0.042) = 0.358 + 0.399 = 0.757
h = 0.757 / log(4) = 0.757 / 1.386 = 0.546

With tau = 0.7: alpha = max(0, 1 - 0.546/0.7) = max(0, 0.220) = 0.220

If neural model gives p_nn(c) = 0.3:
p_mix(c) = 0.220 * 0.458 + 0.780 * 0.3 = 0.101 + 0.234 = 0.335

## Computational Cost

- Building n-gram tables: O(N * K) where N = total tokens, K = max order
- Inference mixing per token: O(V * K) for backoff lookup + O(V) for mixing
- No GPU needed for n-gram component
- Total overhead per token: negligible (<1 microsecond at V=28)

## Assumptions

1. N-gram patterns in character-level names are COMPLEMENTARY to neural patterns
   (n-grams capture local phonotactic constraints, neural captures global structure)
2. The names dataset has sufficient n-gram regularity (many repeated bigrams/trigrams
   in names from shared etymological roots)
3. Entropy of the n-gram distribution is a reliable signal for when to trust it
4. Character-level vocabulary (V=28) is small enough that n-gram tables are sparse
   and memory-efficient
