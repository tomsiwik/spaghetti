# MATH — Pierre v7 Keyframe POC (Arithmetic Verifier)

## Setup

Base model: BitNet-b1.58-2B-4T (BitNet, ternary base) with hidden dim H=2560.
Binary verifier classifier over hidden state h ∈ ℝ^H at terminal token:
  f(h) = σ( STE_q(U)ᵀ · ReLU( STE_q(D)ᵀ h ) ), D ∈ ℝ^{H×r}, U ∈ ℝ^{r×1}, r=16.
STE_q(W) = (α · clip(round(W/α), -1, 1)) with α = mean|W| + ε, forward ternary / backward identity.

Data: N=2000 balanced (expr, answer, label) with label=1 if answer is arithmetically
correct and label=0 otherwise (perturbation ∈ {±1..±5}). Features extracted by running
the base model once over the full text "expr + answer" and taking the final hidden
state via `extract_hidden`. Loss = BCE on σ(f(h)), 500 Adam steps, BS=32, LR=1e-3.

## Theorem (Capacity-Sufficient, Information-Insufficient)

Let h_c, h_w ∈ ℝ^H be the hidden states for the SAME expression with the correct and
a wrong answer respectively. For the verifier f to achieve accuracy > 1/2 + ε on
balanced data, the feature map h(expr, ans) must satisfy a minimal separability
condition: E_c,w[ ‖h_c − h_w‖₂ / (‖h_c‖₂ + ‖h_w‖₂) ] ≥ δ(ε, r) > 0.

**Claim.** Under the protocol used here — single hidden state at the terminal token
of a BitNet pass with no supervision signal injected, and a ternary rank-r probe —
this condition is not met for arithmetic correctness, so the verifier degenerates to
the constant class-prior solution even when training loss decreases.

**Sketch.** The base model is not arithmetic-accurate (Phase-6 measured base
accuracy 80% and most perturbations land on high-logit distractors). Its terminal
hidden state encodes what the base model thinks comes next, not whether the
supplied answer is correct. With balanced labels and no causal separation in h,
the BCE-minimizing predictor is the class-constant p̂ = ½, which under the
threshold-at-0 decision rule collapses to a single class (here: negative).

## Predictions (pre-registered)

P1. Terminal-hidden-state features are approximately class-independent:
    cos( mean(h | label=1), mean(h | label=0) ) > 0.99.
P2. Training loss trends toward BCE of the class prior: L_final ≈ −log(½) = 0.693.
P3. Classifier collapses: pos_acc=0% and neg_acc=100% (or the mirror), accuracy
    within ±3pp of the majority class.
P4. K745 (acc ≥ 60%) FAILS; K746 (composition PPL drift ≤ 10%) PASSES trivially
    because the runner's Phase 5 code path applies only the domain adapter on both
    branches (no verifier action is injected). K747 (divergence) PASSES.

## Cross-references

- F#291 (ternary saturation family): this is the dual of ternary merge — ternary
  *classification* with STE also degenerates when the upstream feature is uninformative.
- AnLLM / anchor-token lit: supervision signal must be injected; unsupervised terminal
  states do not carry arithmetic truth.
- Omlin & Giles: DFAs are encodable, but require the encoding to be trained end-to-end,
  not learned as a linear probe over an untrained-for-the-task base representation.
