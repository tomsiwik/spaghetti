# LEARNINGS.md — exp_q_wrong_real_domains

## Core Finding

The GSM8K M2P adapter causes ~58% relative accuracy drop on language-structure tasks
(sort, reverse) while remaining neutral on numeric tasks (count_even). Q_wrong is
determined by output format compatibility, not semantic similarity.

## Why

The M2P B-matrix conditions on GSM8K-style hidden states and biases attention toward
the "#### N" output format. Language tasks (sequential word manipulation) are
format-incompatible: the adapter injects math answer tokens mid-sequence, inducing
loops and corrupting word order. Numeric tasks are format-compatible ("Answer: N" ≈
"#### N"), so the adapter's perturbation is benign.

This is a structural result: the Grassmannian A-matrices span the GSM8K-relevant
subspace; projecting language-task hidden states onto this subspace produces a B-matrix
biased toward math output format (connects to Aghajanyan 2021, 2012.13255 — low-rank
subspaces capture task-specific structure).

## Implications for Next Experiment

- Routing is confirmed NON-OPTIONAL. Without routing, wrong-domain application halves
  accuracy on language tasks (-57% sort, -58% reverse).
- TF-IDF routing (Finding #354, 100% accuracy on math/sort) is exactly the mechanism
  that prevents this harm — the Q_wrong measurement provides the quantitative stakes.
- Next: exp_m2p_composition_n5_qwen3 (multi-domain composition with TF-IDF routing).
  K927 quality_ratio ≥ 0.75 is now understood as a routing-protected bound: with 100%
  routing accuracy, Q_wrong never fires, and composition quality is isolated to the
  correct adapter path.
- Per-user adapters (Finding #384) showed Q_wrong > 0 (composition helped): confirms
  that format similarity between user adapters prevents format injection.
