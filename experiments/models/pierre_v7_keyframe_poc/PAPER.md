# PAPER — Pierre v7 Keyframe POC

## Verdict: KILLED (K745 FAIL)

Ternary rank-16 probe over BitNet-2B terminal hidden states cannot verify
arithmetic correctness. Measured acc = 48.6% at random-ish majority collapse
(pos_acc = 0.0%, neg_acc = 100.0%). Hypothesis: terminal hidden state does
not encode arithmetic-correctness-given-answer; it encodes the base model's
next-token belief. The classifier optimum under BCE is therefore the class
prior, which under sign-threshold decoding collapses to a single class.

## Prediction vs. Measurement

| # | Prediction                                                     | Measurement                                  | Hit |
|---|----------------------------------------------------------------|----------------------------------------------|-----|
| 1 | L_final ≈ -log(½) = 0.693                                      | final_loss = 0.6933                           | ✓   |
| 2 | pos_acc = 0% or neg_acc = 0% (single-class collapse)           | pos_acc = 0.0%, neg_acc = 100.0%              | ✓   |
| 3 | Accuracy within ±3pp of majority class                         | acc = 48.6% (majority class ≈ 50%)           | ✓   |
| 4 | K745 FAIL (acc ≥ 60%)                                          | 48.6% < 60% ⇒ FAIL                           | ✓   |
| 5 | K746 "PASS" but tautological (runner omits verifier injection) | degradation ∈ {+0.01, -0.01, -0.00}% — trivial | ✓  |
| 6 | K747 PASS (no divergence)                                      | initial=0.7865, final=0.6933                  | ✓   |

## Kill Criteria

- K#745 FAIL — acc = 48.6% < 60%.
- K#746 "PASS" — reported max |degradation| = 0.01% ≤ 10%; see antipattern note.
- K#747 PASS — no divergence; final loss ≈ log(2).

## Antipattern note

K#746 in this runner is tautologically satisfied: Phase 5 builds the domain-only
and "domain + verifier" PPL by running the **same** code path (domain adapter only,
no verifier composition) — the verifier is trained and evaluated but never injected
into the base model for PPL measurement. The logged degradation is measurement noise
from re-loading the model, not a test of composition. This is the F#157 family
"ghost composition" antipattern (composition KC passes without exercising the
composed operator). The kill verdict does not depend on K#746; it is delivered
entirely by K#745.

## Discussion

- The base model's *own* arithmetic accuracy (Phase 6) is 80%, so the difficulty is
  not that BitNet can't do the arithmetic. The difficulty is that a ternary
  **linear probe** over the terminal hidden state cannot read off
  "answer-is-correct" because that property is not represented there — the
  representation encodes "what I would predict next", not "is what the user
  supplied correct".
- Supervising on (h, label) with balanced labels gives BCE its optimum at the class
  prior (p̂ = ½), which under a sign threshold collapses deterministically to one
  class. That is the observed behaviour (pos_acc = 0%).
- This refutes the hypothesis that DFA-style verification can be trained as a
  post-hoc linear probe on a frozen BitNet without injecting a supervision signal
  into the representation pipeline. Training loss touching -log(½) is a reliable
  tripwire for this failure mode.

## Limitations

- N=500 test is small; confidence interval on acc = 48.6% is ≈ ±4.4pp. Even the
  upper end does not cross the K745 threshold.
- The runner's Phase 5 composition is a placeholder; a true Grassmannian-orthogonal
  composition was not measured. This does not salvage K745.
- r=16 may undercapacity, but the collapse-to-prior behaviour indicates the
  bottleneck is information content of h, not capacity.

## Reusable finding candidates

- "Frozen-base linear probes of terminal hidden states cannot read answer-correctness
  under balanced BCE training; the BCE optimum is the class prior, which under
  sign-threshold decoding collapses to a single class." Generalises across
  post-hoc verifier proposals on frozen LMs.
- "Composition KCs that measure PPL under two identical injection code paths are
  tautologies; they must exercise the operator under test."
