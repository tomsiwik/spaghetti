# LEARNINGS: Energy Gap Top-1 Routing

## Core Finding

Energy gap top-1 routing (argmin NLL reduction) achieves 88% adapter selection accuracy
and +133% math answer correctness over uniform composition. The ranking signal from
Finding #182 (AUC=0.942) translates directly to generation quality on structured tasks
(math, code), but provides no benefit on prose tasks where the base model is already
competent. This validates the ranking-to-routing pipeline while revealing a two-world
pattern: routing matters for structured output, not for prose.

## Why This Happened

### Ranking vs Gating: The Critical Distinction

Finding #184 killed binary gating (all adapters reduce NLL universally). This experiment
proved the RANKING signal survives: while all adapters help, they help *by different
amounts*. The magnitude ordering |Delta_E_correct| >> |Delta_E_wrong| is preserved
because domain-matched adapters have much larger NLL reductions (math adapter on math:
-1.90 nats vs code adapter on math: -0.60 nats). The argmin operation discards the
absolute threshold problem entirely, using only relative ordering.

### Why Structured Tasks Benefit Disproportionately

Math answer correctness jumped from 30% (uniform) to 70% (top-1) because GSM8K requires
specific output formatting (chain-of-thought, boxed answers) that only the math adapter
learned. Uniform composition dilutes this formatting signal with 80% irrelevant adapter
contributions. On prose tasks (medical, legal, finance), the base BitNet model already
produces reasonable text — adapters provide marginal or negative benefit, making routing
irrelevant. This matches the "two-world" pattern seen across the project's experiments.

### The O(N) Overhead Problem

Energy gap routing requires N+1 forward passes on the prompt to compute all gaps.
MATH.md predicted ~18% overhead (Section G), the kill criterion was set at 10% (Section E),
and the measurement was 29.5% (inflated by disk loading). Even the theoretical ~5%
(with caching) grows linearly: at N=25 adapters, this becomes ~26 forward passes per
query. This is the fundamental scalability limitation of loss-based routing — confirmed
by our own NotebookLM sources documenting the "K+1 bottleneck" as a known failure mode
of NLL-based routing.

## Confirming Evidence

- **LoRAuter (arxiv 2601.21795):** Task-aware vector retrieval for LoRA routing. Uses
  sentence embeddings instead of NLL to avoid the O(N) forward pass problem. Achieves
  competitive routing without any test-time loss computation. Confirms our finding that
  ranking works but offers a cheaper routing signal.

- **MoLoRA (arxiv 2603.15965):** Per-token routing of LoRA experts. Qwen3-1.7B + MoLoRA
  exceeds Qwen3-8B on GSM8K (+14%), MATH (+8%). Confirms that correct adapter routing
  produces large gains on structured reasoning tasks, matching our +133% math improvement.
  Uses learned gating rather than energy gap, avoiding the O(N) problem.

- **Finding #182 (this project):** Energy gap AUC=0.942 on math. The ranking signal that
  this experiment converts into routing decisions. Without this prior result, the
  experiment would have no theoretical motivation.

## Contradicting Evidence

- **SOLE project PPL-probe routing:** Our own prior work showed NLL-based routing achieves
  r=0.990 correlation with oracle at micro-scale, but **collapses to r=-0.63 at macro-scale**
  with real converged adapters. The current experiment operates at the boundary — 5 adapters,
  10 prompts per domain. The strong math result (AUC=0.942 → 100% routing accuracy) may not
  survive scaling to more domains or more similar adapters.

- **Finding #178:** PPL correlates at r=0.08 with task accuracy. Energy gap partially
  resolves this (difficulty cancellation), but the prose-domain results (all configurations
  within noise) suggest the discrimination fails when there's no strong structured signal.

- **Legal-finance confusion (Finding #186):** Energy gap difference of only 0.041 nats
  between legal and finance adapters on legal queries. At N=10+ similar domains, routing
  accuracy will degrade from pairwise confusions. The 88% accuracy is partly an artifact of
  3 perfectly separable domains (med/code/math) masking 2 confused ones (legal/finance).

## Alternative Approaches

- **LoRAuter (arxiv 2601.21795):** Embedding-based routing via sentence encoder + vector DB.
  O(1) routing cost (single embedding lookup) vs our O(N) forward passes. Training-free.
  Would solve the overhead kill criterion failure while potentially matching routing quality.
  Directly applicable to our adapter set.

- **MoLoRA (arxiv 2603.15965):** Per-token learned routing. Routes individual tokens to
  different adapters within a single sequence. Solves the cross-domain query problem
  (Limitation #4 in PAPER.md) that single-adapter top-1 routing cannot handle. Requires
  router training but achieves amortized O(1) routing per token.

- **L-MoE (arxiv 2510.17898):** End-to-end lightweight MoE with LoRA experts. Learned
  gating network computes per-token weighted average of adapter parameters. Avoids the
  discretization error of top-1 selection.

- **LD-MoLE (arxiv 2509.25684):** Learnable dynamic routing replacing non-differentiable
  top-k with differentiable routing function. Token-dependent, layer-wise allocation.

- **CLONE (arxiv 2506.02847):** MoE router for dynamic LoRA selection at edge. Specifically
  designed for edge deployment — directly relevant to our M5 Pro target.

## Implications for Next Experiments

1. **Energy gap routing is validated as a PROOF-OF-CONCEPT but not a production mechanism.**
   The O(N) cost and legal-finance confusion are structural limitations. Production routing
   needs either learned routers (MoLoRA, L-MoE) or embedding-based routing (LoRAuter).

2. **The two-world pattern is now a robust finding across 3+ experiments:** structured tasks
   (math, code) benefit enormously from correct routing; prose tasks are routing-insensitive.
   Future experiments should focus on structured-output domains.

3. **Ranking is the right abstraction.** Finding #184 killed gating. This experiment
   validated ranking→routing. The energy gap is a valid ranking signal, but cheaper ranking
   signals (embeddings, learned routers) should replace it for deployment.

4. **Cross-domain queries remain untested.** The single-adapter routing paradigm breaks
   for queries requiring multiple domains ("write Python to solve a math problem"). Per-token
   routing (MoLoRA) is the natural solution.

## Recommended Follow-Up

**Embedding-based routing (LoRAuter-style):** Replace energy gap routing with sentence
embedding similarity. Motivation: LoRAuter (arxiv 2601.21795) achieves competitive routing
at O(1) cost. We already have 5 trained adapters with labeled training data — compute
adapter "centroids" in embedding space, route by nearest centroid. This would fix the
overhead kill criterion (K577) while preserving routing accuracy, and scale to N=25 without
the O(N) forward pass problem. Alternative: train a tiny learned router (Finding #179 showed
100% accuracy with 2.32% overhead routing heads — this was already proven in our project).
