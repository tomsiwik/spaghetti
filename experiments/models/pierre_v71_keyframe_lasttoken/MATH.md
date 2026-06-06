# MATH — Pierre v7.1 Keyframe Last-Token Verifier

## Setup

Base model: BitNet-b1.58-2B-4T (frozen ternary base), hidden dim H=2560.

Task: arithmetic verification. For expression/answer pair (e, a) with
e ∈ Σ* (expression text ending in "="), a ∈ Σ (answer digits),
label y = 1[a = eval(e)]. Dataset is class-balanced, E[y] = ½.

Feature: h(e, a) = hidden_state(model(encode(e ⊕ a)))[−1, :]
— **last token** of the encoded sequence e ⊕ a. Hypothesis motivating v7.1
(vs v7 mean-pool): the last token is where the model has committed to its
next-token prediction, so its hidden state should carry the answer-relevant
signal undiluted.

Verifier: ternary MLP (down ∈ {-1,0,+1}^{H×r}, up ∈ {-1,0,+1}^{r×1}) with
STE, trained by balanced BCE on binary labels, decoded by sign threshold.

## Theorem (Last-Token Hidden State is a Commit Vector, Not a Verifier)

Let h = model_last_hidden(e ⊕ a). In a decoder-only causal LM, h is the
state from which the model predicts the **next** token (position |e ⊕ a|+1),
not a verification of position |e ⊕ a|. Therefore h does not natively encode
"is a the correct answer to e". When trained with balanced BCE under sign
decoding, the verifier converges to the class prior, yielding deterministic
single-class collapse.

### Proof

**(1) Causal hidden-state semantics.** By construction of a causal decoder,
layer_l(h)[t, :] depends only on tokens at positions ≤ t. The model's next-
token distribution at position t is p(x_{t+1} | x_{≤t}) = softmax(W_unembed
· LN(h_L[t])). The terminal hidden state h_L[T−1] (where T = |e ⊕ a|) is
therefore the representation used to predict position T — i.e. the token
**after** the displayed answer a, not a. Nothing in the forward pass
executes a verification of "is x_T the right continuation of x_{<T}".

**(2) Class-conditional representations are near-degenerate for balanced a.**
Consider pairs (e, a_+) and (e, a_−) with a_+ correct and a_− incorrect,
differing only in the last token. Because causal attention makes h_L[T−1]
depend on x_{<T} (not x_T itself beyond embedding + self-attention on x_T's
own query), and because the model's internal "commitment" at position T−1
(final "=") has already encoded its own answer belief independent of a,
the component of h_L[T−1] that differs between a_+ and a_− is exactly the
token-T self-attention contribution — a single head-projection of emb(a).
This is a tiny fraction of ‖h‖. Empirically (Phase 2 sanity check)
relative_diff = ‖μ_+ − μ_−‖ / ((‖μ_+‖+‖μ_−‖)/2) is expected to be ≪ 1.

**(3) BCE + balanced labels → class prior.** Let p̂ = σ(logit(x)). With
y ∼ Bernoulli(½) and x approximately label-independent (by step 2),
min_θ E[BCE] ⇒ p̂(x) → ½ for all x. Equivalently logit(x) → 0.

**(4) Sign decoding of ~0 logits collapses to one class.** With
float arithmetic, sign(logit) ∈ {−1, +1} is determined by infinitesimal
perturbation. For our initialization and optimizer trajectory, the sign
lands deterministically on one class across the full test set ⇒
pos_acc ∈ {0, 1} and neg_acc = 1 − pos_acc. Balanced accuracy = 0.5. ∎

### Corollary (Why v7.1 ≠ v7 hypothesized improvement)

v7 used mean-pool features and was killed. The v7.1 hypothesis
("last-token concentrates the signal") assumed the signal exists somewhere
in hidden space and pooling was the bug. Theorem shows the signal is not
in hidden state at all — it's in the **logits** (the un-embedding matrix
multiplies h, collapsing it onto the vocabulary simplex where arithmetic
consistency appears as a token-probability ordering). Base-model logit
accuracy is 80% (Phase 6) because the right answer sits as the argmax
token in p(x_T | x_{<T}), not because the pre-unembed h has a
separable "correctness" direction.

### Predictions

P1. relative_diff in Phase 2 sanity check ≪ 1 (features nearly class-independent).
P2. Final BCE loss → −log(½) ≈ 0.693.
P3. pos_accuracy ∈ {0, 1}, neg_accuracy = 1 − pos_accuracy (single-class collapse).
P4. Overall accuracy ≈ 0.5 (random).
P5. Phase 6 base_accuracy ≥ 70% (logit-level signal exists even though hidden-state signal does not).
P6. Phase 5 degradation ≈ 0 **by construction** (Phase 5 code path
    injects domain adapter only on both branches — no verifier — so
    "composition" is a measurement of same-adapter-twice, not a genuine
    composition check). Ghost-composition (F#157 family).

## Kill criteria (pre-registered, unchanged)

- K#748: Verifier accuracy < 60%. FAIL here kills.
- K#749: Training divergence. PASS here (no divergence) is a sanity check.
