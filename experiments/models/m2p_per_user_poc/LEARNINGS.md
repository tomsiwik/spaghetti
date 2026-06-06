# LEARNINGS.md — exp_m2p_per_user_poc

## Core Finding

M2P hypernetworks trained on style demonstrations produce measurable behavioral
differentiation (K940 PASS, d=1.262 for CODE) and compose safely with domain adapters
(K941 PASS, 0% degradation). However, EOS termination is a separate learned behavior
from token-level style — style-copying shifts P(token|context) but cannot reliably
increase P(EOS) unless EOS appears strongly in the training continuation signal.

## Why

Code-style output terminates via natural format structure (`# computed` → answer → stop),
giving the M2P a clear EOS anchor. Concise-style ("#### N") has EOS at the very end of a
short sequence, but 300 steps / 50 examples is insufficient to overcome the base model's
prior against early termination. This matches the STE gradient-flow argument: B-matrix
updates perturb the activation manifold but cannot amplify low-prior tokens without
dense supervision.

## Implications for Next Experiment

1. **EOS-aware training**: Explicitly include `<EOS>` as a target token in concise
   training continuations, or train with negative-example penalty for continuation tokens.
2. **Behavioral eval**: Replace token-length proxy with "format adherence" regex
   (matches "#### N" with no continuation) — more precise and avoids degenerate-cap artifacts.
3. **Composition scaling**: K941's safe composition result (same-domain adapters) is
   the credible signal here; the +4pp improvement is within noise at n=50. Next test
   should use n=200+ for composition eval.
