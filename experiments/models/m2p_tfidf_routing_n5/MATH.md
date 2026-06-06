# MATH.md — exp_m2p_tfidf_routing_n5

## TYPE: Verification (Type 1)

Proof-first. Experiment confirms quantitative predictions derived from the
theorems below.

---

## A. Failure Mode Identification

**Root cause (exp_m2p_composition_n5, Finding #351):**

The per-token MLP router was trained on *base model* hidden states
`h_base(x)` but deployed on *composed model* hidden states `h_comp(x)`.
Because composition changes the representation distribution,
`P(h_base) ≠ P(h_comp)`, causing a covariate shift that directly violates
the i.i.d. assumption of supervised classification.

Additionally, per-token routing on the toy vocabulary (128 tokens) cannot
distinguish `sort`, `reverse`, and `repeat` domains because they share the
identical character alphabet `{a-h}`. Any single token drawn from this
alphabet is uninformative about which of the three domains generated it.

**Disease (one root cause, not symptoms):** Routing depends on *model
internals* (hidden states). Any change to the model distribution — whether
from composition, fine-tuning, or quantization — shifts the input
distribution of the router out-of-distribution.

**Is this a stable failure point?**

Yes. Let `φ: X → R^K` be the router (hidden-state → domain logit), and let
`h_base, h_comp` be the hidden state distributions under the base and
composed model respectively. The MLP router minimises
`L_train = -E_{h_base}[log p(d | φ(h_base))]`.
At deployment, the effective loss is `L_test = -E_{h_comp}[log p(d | φ(h_comp))]`.
When `||P(h_base) - P(h_comp)||_TV > 0`, the optimum of `L_train` is not an
optimum of `L_test`. This failure is *structural*: more training data or a
larger router cannot fix it without access to composed-model hidden states.

---

## B. Reframe — Ask the Right Question

Wrong: "How do we prevent the router from being fooled by composed-model
hidden states?"

Right: "What routing function is invariant to model distribution by
construction, such that the covariate-shift failure is geometrically
impossible?"

**Answer (from classical information theory):** Any function that depends
only on the *input sequence* `x` (and not on model activations) is
trivially invariant to all changes in model parameterisation. The simplest
such function with proven domain-separation power is TF-IDF + logistic
regression over sequence-level n-grams.

---

## C. Prior Mathematical Foundations

### Theorem C.1 (LoraRetriever, He et al. 2024, arXiv:2402.09997, §3)

For a routing function `r: X → [N]` that depends only on the *input text*
`x ∈ X`, the routing decision is independent of the adapter weights
`{W_1,...,W_N}` and the base model `W_0`. Consequently, no combination of
adapter compositions can shift the routing distribution.

This is a direct corollary of the factorisation `P(route | x, adapters) =
P(route | x)` when the route is computed before any model forward pass.

### Theorem C.2 (Fisher's Linear Discriminant, Fisher 1936)

For two classes with means `μ_1, μ_2 ∈ R^p` and shared covariance `Σ`,
the Fisher ratio `J = (μ_1 - μ_2)^T Σ^{-1} (μ_1 - μ_2)` is the maximum
achievable signal-to-noise ratio for any linear classifier.

Applied to TF-IDF space: the feature map `φ(x) = tf-idf(x) ∈ R^V` has
`J > 0` for domains with different n-gram distributions (e.g., `"sort:"`,
`"reverse:"`, `"*"` triggers). Hence logistic regression in TF-IDF space
achieves non-trivial accuracy whenever domain-specific n-grams exist —
regardless of their shared character alphabet.

### Theorem C.3 (Cover's theorem on linear separability, Cover 1965)

A randomly drawn set of `N` points in `R^d` is linearly separable with
high probability when `N < 2d`. For TF-IDF vectors (d up to 5000
features) and training sets of N ≈ 500 per domain, the data lies in a
regime where linear separability is expected a priori.

### Lemma C.4 (Sequence-level disambiguation via format tokens and ordering statistics)

Let `x = (x_1,...,x_T)` be a token sequence. The five domains produce sequences
in these exact formats:

- **arithmetic**: digit tokens `{0,...,9}` + operator tokens `{+, -, =}`.
  Unigram presence of `{+, -, =}` is unique to arithmetic.
- **parity**: binary tokens `{0, 1}` followed by the label token `"even"` or
  `"odd"`. The label tokens are unique to parity.
- **repeat**: format `"{chars}*{n}={repeated}"` — the `*` token is absent in
  all other domains. Bigrams containing `*` are unique to repeat.
- **sort**: format `"{input}>{output}"` where the output after `>` is the
  input letters sorted in ascending order (e.g., `"bca>abc"`). No prefix label.
- **reverse**: format `"{input}>{output}"` where the output after `>` is the
  input letters in reversed order (e.g., `"abc>cba"`). No prefix label.

**Syntactic separation (exact):**

`arithmetic`, `parity`, and `repeat` are exactly separable from each other
and from `{sort, reverse}` via unique format tokens:
- `arithmetic` ↔ others: presence of `{+, -, =}` with digit tokens
- `parity` ↔ others: presence of `"even"` or `"odd"` tokens
- `repeat` ↔ others: presence of `*` token

The unigram and bigram distributions of these three domains are disjoint from
each other and from `{sort, reverse}`. Fisher ratio `J >> 0` for all 6 pairs
involving `arithmetic`, `parity`, or `repeat`.

**Statistical separation of sort vs. reverse (non-exact):**

Both `sort` and `reverse` share the identical format `{input}>{output}` with
the same input alphabet `{a,...,h}` and the same separator token `>`. They
cannot be separated by format tokens alone.

Separation relies on **character-ordering statistics in the output portion**
after `>`:
- In `sort`, adjacent bigrams in the output satisfy `char_{i+1} > char_i`
  (ascending order). E.g., output `"abc"` → bigrams `(a,b)`, `(b,c)` both
  ascending.
- In `reverse`, adjacent bigrams in the output satisfy `char_{i+1} < char_i`
  (descending order). E.g., output `"cba"` → bigrams `(c,b)`, `(b,a)` both
  descending.

For sequences of length `n ≥ 3`, these bigram distributions have non-zero
Fisher ratio: `μ_sort(bigram_ascending) > μ_reverse(bigram_ascending)`.
TF-IDF captures these ordering statistics as n-gram frequencies.

**Predicted ambiguity for short sequences:**

For `n = 2` character sequences (e.g., input `"ab"`), the sort output is
`"ab"` and the reverse output is `"ba"`. When the two input characters happen
to be in ascending order, `sort → "ab>ab"` and `reverse → "ab>ba"` have only
a single bigram difference in the output. For the specific case where input
equals its own reverse (e.g., input `"aa"` — not possible with distinct
chars), output is identical. More generally, the bigram signal is weakest for
length-2 sequences, predicting ~10-15% confusion for sort vs. reverse at the
short-sequence end of the distribution.

Hence TF-IDF over n-grams (1,2) has non-zero Fisher ratio for all 10 pairs in
the 5-way classification, with syntactically exact separation for
`{arithmetic, parity, repeat}` and statistically reliable (non-exact)
separation for `{sort, reverse}`.

---

## D. Theorem 1 — Distribution-Agnostic Routing Guarantee

**Theorem 1.** Let `r_TFIDF: X → [N]` be a logistic regression classifier
trained on input strings. Let `M_base` and `M_comp` be the base and
composed models respectively. Then for all `x ∈ X`:

```
r_TFIDF(x) = r_TFIDF(x)    (trivially: r_TFIDF does not depend on M)
```

More precisely:

```
P(r_TFIDF(x) = d | M_base) = P(r_TFIDF(x) = d | M_comp) = P(r_TFIDF(x) = d)
```

**Proof.**

`r_TFIDF` is defined as:

```
r_TFIDF(x) = argmax_d  [W_LR · φ_{TFIDF}(text(x))]_d
```

where:
- `text(x)` = decode tokens `x` to string (deterministic, model-independent)
- `φ_{TFIDF}` = TF-IDF vectoriser fitted on training strings (fixed after training)
- `W_LR` = logistic regression weights (fixed after training)

None of `text`, `φ_{TFIDF}`, or `W_LR` depend on any model parameters
(`W_0`, `{W_i}`, composition weights, or adapter scales). The function
`r_TFIDF` is a pure function of `x` and fixed parameters. Therefore:

```
P(r_TFIDF(x) = d | M_base) = P(r_TFIDF(x) = d | M_comp)    (independence)
```

The train/test distribution of `r_TFIDF` is always the distribution over
input strings `P(x)`, which is the same at training and deployment.
Covariate shift from model composition cannot affect routing decisions.

**QED.**

---

## D. Theorem 2 — Sequence-Level Disambiguation

**Theorem 2.** TF-IDF representation of full input sequences (not
individual tokens) provides sufficient Fisher discriminant power to
separate all 5 domains including `sort`, `reverse`, and `repeat` which
share the character alphabet `{a,...,h}`.

**Proof sketch.**

Per-token routing fails because token `"a"` appears in sort, reverse, and
repeat with equal unigram probability (each domain draws uniformly from
`{a,...,h}`). For any single token, the posterior `P(domain | token)` is
uniform over the three ambiguous domains.

For sequence-level routing, consider the joint unigram+bigram distribution
over a sequence `x = (x_1,...,x_T)`:

1. **sort** generates sequences of the form `"bca>abc"` — letter tokens,
   then `>` separator, then the same letters in ascending sorted order.
   No label prefix. Bigrams in the output after `>` are predominantly
   ascending-ordered character pairs (e.g., `(a,b)`, `(b,c)`).

2. **reverse** generates `"abc>cba"` — same format `{input}>{output}`, same
   alphabet, same separator `>`, but output is the input reversed. Bigrams
   in the output after `>` are predominantly descending-ordered character
   pairs (e.g., `(c,b)`, `(b,a)`). No label prefix; not syntactically
   distinguishable from sort by format tokens alone.

3. **repeat** generates `"ab*3=ababab"` — contains `*` and `=` tokens not
   present in sort/reverse. These format tokens are exact discriminators.

For `arithmetic` and `parity`, unique format tokens (`+`, `-`, `=` with
digits; `even`/`odd`) provide exact TF-IDF support. For `repeat`, the `*`
token provides exact support. For `sort` vs. `reverse`, TF-IDF captures the
character-ordering statistics in the output bigrams. Therefore for any pair
`(d_i, d_j)` in `{sort, reverse, repeat}`:

```
μ_i - μ_j ≠ 0    in R^V (TF-IDF space)
```

By Fisher's theorem (C.2), there exists a linear classifier achieving
`J > 0`, i.e., strictly better than chance. The empirical evidence
(Finding #207: 90% on SFT domains with the same TF-IDF+LR architecture)
confirms this. On our toy domains, the separation is *even stronger*
because the data generators use explicit format tokens (`*`, `=`, `even`/`odd`, digit operators)
that provide syntactically exact separation for arithmetic, parity, and repeat, with only sort/reverse
requiring the statistical ordering-statistics argument.

**QED (directional; format-string argument exact for arithmetic, repeat, parity;
character-ordering argument statistical for sort, reverse — predicting ~10-15%
confusion for short sequences).**

---

## E. Corollary — Kill Criteria Derivation

Let `α_route` = TF-IDF routing accuracy, `q_m2p` = M2P generation quality
ratio (vs SFT, Finding #351 measured 93.3% median), `q_sft` = SFT quality.

**Expected composed quality under TF-IDF routing:**

When the router selects the correct domain with probability `α_route`, and
the wrong domain with probability `1 - α_route`, the expected quality is:

```
q_composed = α_route · q_m2p + (1 - α_route) · q_wrong
```

where `q_wrong < q_sft` for misrouted sequences (applying wrong adapter).
Lower bound: assume `q_wrong = 0` (worst case).

```
q_composed ≥ α_route · q_m2p
```

**Bound from Finding #207 + Finding #351:**
- `α_route ≥ 0.90` (TF-IDF routing accuracy, Finding #207)
- `q_m2p = 0.933` (per-domain M2P quality, Finding #351)

```
q_composed ≥ 0.90 × 0.933 = 0.84    (lower bound, worst-case misrouting)
```

**Kill criterion derivation:**

- **K867 (routing accuracy > 70%):** Derived from `0.70 × 0.933 = 0.65 >
  SFT-floor`. If routing accuracy < 70%, even perfect M2P generation yields
  < 65% quality, below the composition-adds-value threshold. Threshold 70%
  is conservative vs. the 90% Finding #207 baseline.

- **K868 (composition quality > 70% SFT):** Derived from K867 (70%) ×
  q_m2p (0.933) = 0.653 ≈ 70%. This is the predicted floor assuming ~5%
  margin. The 70% threshold directly follows from the kill-criterion
  composition argument above.

- **K869 (oracle routing quality ceiling > 80% SFT):** Oracle routing
  means `α_route = 1.0`, so `q_oracle = q_m2p = 0.933 ≥ 0.80`.
  This confirms M2P generation quality (not routing) is the bottleneck at
  oracle routing. If oracle routing < 80%, the M2P adapters themselves have
  regressed (Finding #351 would be invalidated for this run's adapters).

---

## F. Assumptions & Breaking Conditions

| Assumption | Content | Consequence if violated |
|---|---|---|
| A1 | arithmetic/parity/repeat have unique format tokens; sort/reverse have statistically distinct output-bigram ordering | Theorem 2 fails; routing accuracy < 70%; K867 FAIL |
| A2 | M2P generation quality ≥ 90% of SFT (Finding #351 reproducible) | K869 may FAIL even at oracle routing |
| A3 | TF-IDF vocabulary of toy data has sufficient discriminative n-grams | Fisher ratio J ≈ 0; logistic regression degrades to random baseline |
| A4 | Routing and generation are independent at inference (no feedback loop) | Theorem 1 proof holds only if routing is pre-model |

**Breaking A1:** For `arithmetic`, `parity`, and `repeat`: if the data generator
removes their unique format tokens (`+`/`-`/`=`, `even`/`odd`, `*`), TF-IDF
separability of those domains collapses. For `sort` vs. `reverse`: if sequence
lengths are restricted to ≤ 2 characters throughout, the output-bigram ordering
signal disappears and confusion rises toward 50%. Either case would drive K867
toward FAIL, triggering a new direction (e.g., domain-aware tokenisation).

**Breaking A2:** If the base model or M2P adapters are retrained with
different seeds/hyperparameters, `q_m2p` may differ from 93.3%.
K869 serves as a direct guard on this.

**Breaking A3:** Toy vocab=128 encodes all chars modulo 128.
Format strings (`>`, `*`, `=`) map to ASCII 62, 42, 61 — all < 128,
so they survive encoding. This assumption holds for the specific
`encode_text` function in `m2p_composition_n5`.

---

## G. Worked Example (d=5, 2-way toy)

Consider 2 domains, 3 training strings each:

- **sort**: `"ab>ab"`, `"ba>ab"`, `"ca>ac"`
- **repeat**: `"a*2=aa"`, `"b*3=bbb"`, `"ab*2=abab"`

Character-level unigrams:
- sort: `{a, b, c, >}` — `>` appears, `*` and `=` absent
- repeat: `{a, b, *,=}` — `*` and `=` appear, `>` absent

TF-IDF vocabulary (size = 4 unique chars): `{">", "*", "=", "a"}`.
TF-IDF for `"ab>ab"` ≈ `[1.0, 0, 0, ...]`, for `"a*2=aa"` ≈ `[0, 1.0, 1.0, ...]`.

Fisher ratio J for `>` feature: `μ_sort(">") ≈ 0.25` vs `μ_repeat(">") = 0`.
`J = (0.25 - 0)^2 / σ^2 > 0` — perfectly linearly separable.

For bigrams including `"sort"` string (encoded from ord('s')%128=115):
The 4-gram `"sort"` → TF-IDF weight → cleanly separates `sort` from all
other domains. Same logic applies at token level for all format strings.

**Conclusion:** At micro-scale, the toy data format strings guarantee
TF-IDF separability without any learned representations.

---

## H. Complexity & Architecture Connection

| Operation | FLOPs | Notes |
|---|---|---|
| TF-IDF vectorisation | O(T·V) | T=48 tokens, V=5000 features |
| Logistic regression predict | O(V·N) | N=5 domains |
| M2P forward (per domain) | O(N_MEM · D_M2P^2) | Same as m2p_composition_n5 |
| Composed forward | O(D·T) per layer | Base forward only (no router MLP) |

Total routing overhead vs. base forward: `O(T·V) / O(T·D^2) = V/D^2 ≈ 0.076`
(5000 / 65536). TF-IDF routing is computationally negligible.

**Architecture integration:**

```
Input string x
    ↓  [text(x) — decode to string]
    ↓  [φ_TFIDF(text(x)) — sklearn TF-IDF, frozen after training]
    ↓  [W_LR · feature → domain logit → argmax]
domain d
    ↓  [load B-matrices for domain d]
    ↓  [M2P composed forward: base + LoRA with A_d, B_d]
output
```

No dependency on any model hidden state. Routing is a pure text transform.

---

## Self-Test (MANDATORY)

**1. What is the ONE mathematical property that makes the failure mode impossible?**
TF-IDF routing is a pure function of the input string, computed before any
model forward pass, making it trivially invariant to all changes in model
parameterisation (Theorem 1). Covariate shift from model composition cannot
reach the routing function.

**2. Which existing theorem(s) does the proof build on?**
- LoraRetriever (He et al., arXiv:2402.09997, §3): text-based routing
  decouples routing from adapter NLL and model internals.
- Fisher (1936): linear discriminant theory guarantees J > 0 for
  non-identical class means in TF-IDF space.
- Cover (1965): linear separability expected when N < 2d.

**3. What specific numbers does the proof predict?**
- Routing accuracy ≥ 90% (by analogy with Finding #207, same TF-IDF+LR
  architecture, even stronger signal due to format prefixes)
- Composed quality ≥ 0.70 × 0.933 = 0.653 (K868 derived lower bound)
- Oracle quality ≈ 0.933 (K869 = M2P generation quality, independent of routing)

**4. What would FALSIFY the proof (not just the experiment)?**
If TF-IDF routing accuracy < 70% on these toy sequences, either:
(a) The format strings are not preserved in the `encode_text` encoding
    (breaking A3 — checkable analytically), OR
(b) The logistic regression overfits (too few training samples), OR
(c) The data generator was changed and format strings removed.
None of these would falsify Theorem 1 (the distribution-invariance
guarantee), only Theorem 2 (the separability guarantee for THIS specific
dataset). Theorem 1 is a tautology for any text-only routing function.

**5. How many hyperparameters does this approach add?**
2: TF-IDF max_features (5000) and logistic regression C (1.0).
Both are borrowed exactly from Finding #207 (contrastive_routing_n5).
They cannot be derived from the proof (Fisher ratio doesn't prescribe them),
but they are empirically validated on the same architecture family.

**6. Hack check: Am I adding fix #N to an existing stack?**
No. This replaces the MLP router entirely with a simpler, theoretically
cleaner mechanism. The stack goes from [base + M2P + MLP router] to
[base + M2P + TF-IDF text router]. No additive fix.

---

## Prediction Table (for PAPER.md)

| Prediction (from proof) | Source | K-criterion | Expected value |
|---|---|---|---|
| TF-IDF routing accuracy ≥ 70% | Theorem 2 + Finding #207 | K867 | ≥ 0.90 |
| TF-IDF composed quality ≥ 70% SFT | Corollary (K868) | K868 | ≥ 0.65 |
| Oracle routing quality ≥ 80% SFT | Finding #351 reproducibility | K869 | ≈ 0.933 |
| Routing invariant to model distribution | Theorem 1 (tautology) | — | 100% guaranteed |
