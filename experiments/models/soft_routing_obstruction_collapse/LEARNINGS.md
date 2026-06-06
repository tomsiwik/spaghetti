# LEARNINGS: Soft-Routing Obstruction Collapse (KILLED)

## Core Finding

BCE domain-classification loss mathematically guarantees K=1 sparse routing by
construction. The optimal logits under BCE with soft labels (primary=0.80,
others=0.05) are +1.39 for the primary adapter and -2.94 for all others, making
multi-adapter activation (K>=3) impossible. The H¹=3 topological obstructions
from Finding #242 persist under learned routing because Gumbel-sigmoid with
domain-classification loss is a discriminative objective — it optimizes for
selecting ONE correct adapter, which is antithetical to multi-adapter composition.

## Why This Happened

### 1. BCE loss is a discriminative objective — it optimizes for selection, not composition

Binary Cross Entropy with soft domain labels treats routing as a multi-label
classification problem where each gate independently predicts "is this adapter
relevant?" The optimal solution is to maximize the primary gate and suppress
all others. With label smoothing of 0.05, the Bayes-optimal logits are:
- Primary: logit(0.80) = +1.39 → sigmoid = 0.80 → above threshold
- Others: logit(0.05) = -2.94 → sigmoid = 0.05 → far below threshold

This is not a training failure — it is the CORRECT solution to the posed
optimization problem. The disease is the objective, not the optimizer.

### 2. The kill was predictable analytically from MATH.md's own worked example

MATH.md Section F computed E[K] = 2.2 for moderately informative logits
[1.0, 0.8, 0.5, -0.3, -1.0], already below the K1 threshold of 2.5. The
adversarial review correctly noted that the prediction E[K]>=2.5 contradicted
the proof's own analysis. Future experiments must verify that worked examples
satisfy kill criteria before running code.

### 3. Theorem 1 was unfalsifiable by this experiment

Theorem 1 states: IF K>=3 THEN H¹=0. This is a conditional — the experiment
tested the antecedent (does K>=3 happen?), not the theorem. The theorem remains
mathematically valid but architecturally irrelevant under BCE routing because
the precondition is never satisfied.

## Confirming Evidence

- **SCMoE (2405.14507):** "Increasing the number of activated experts does not
  necessarily improve and can even degrade the output quality." Different experts
  do not inherently act synergistically when forced to activate together. Their
  self-contrast approach leverages unchosen experts at inference time instead of
  forcing multi-activation during routing.

- **Switch Transformers (Fedus et al., 2021):** Strict K=1 routing with 128
  experts achieved 400% pretraining speedup at trillion-parameter scale. Sparse
  activation is not a bug — it is the established paradigm for efficiency.

- **Expert Choice Routing (2202.09368, NeurIPS 2022):** Flips the routing paradigm
  — experts select top-k tokens rather than tokens selecting experts. Each token
  ends up routed to a variable number of experts. Demonstrates that the
  token→expert direction inherently produces sparse activation.

- **AdaMoE (2406.13233, EMNLP 2024):** Introduces null experts to enable
  token-adaptive K without changing the routing mechanism. Reduces FLOPs by 14.5%
  while improving accuracy by 1.69% on ARC-C. Shows that variable K is achievable
  but requires architectural support, not just loss modification.

- **Rethinking Inter-LoRA Orthogonality (Zhang et al., 2025):** Even perfectly
  orthogonal LoRA modules in weight space produce semantically interfering outputs
  when merged. Multi-adapter composition has fundamental non-linear interference
  that weight-space metrics cannot predict — consistent with our four-level proxy
  chain (Finding #238, #240).

## Contradicting Evidence

- **LoRAuter (2602.21222):** Dynamic linear merging via cosine similarity retrieval
  achieves 70.95% on PIQA by composing multiple LoRA adapters with continuous
  weights from nucleus (top-p) sampling. This is composition-aware routing without
  BCE — the router optimizes similarity to task representations, not domain
  classification. Demonstrates multi-adapter composition IS achievable with the
  right objective.

- **RDLC (Router-Driven LoRA Compaction):** Continuous coefficient regression
  replaces discrete routing. Output-space distillation loss with L1/L2
  regularization trains unconstrained mixing weights. Multi-adapter activation
  is natural because the objective is output quality, not domain selection.

- **L-MoE (2510.17898):** End-to-end training of lightweight LoRA experts with
  jointly-trained gating. The gating network computes weighted averages of adapter
  parameters per token — composition-aware by construction since the loss flows
  through the composed output.

- **MoLoRA (2603.15965):** Per-token routing where Qwen3-1.7B + 4 adapters
  beats 8B. Shows that per-token multi-adapter composition works at our scale,
  but requires the router to be trained jointly with composition quality.

## Alternative Approaches (with paper evidence)

### 1. Composition-Aware Loss (Output-Space Distillation)
Replace BCE with a loss that measures quality of the COMPOSED output, not
domain classification accuracy. RDLC uses distillation loss: train the router
to produce mixing coefficients that minimize ||composed_output - target||.
L-MoE (2510.17898) trains gating jointly end-to-end. This makes multi-adapter
activation the optimal solution when multiple adapters contribute to output quality.

### 2. Similarity-Based Retrieval Routing (LoRAuter, 2602.21222)
Replace learned classification with cosine similarity retrieval against task
representations in a vector database. Nucleus sampling naturally selects a
variable number of adapters based on similarity distribution. No BCE, no
domain labels — composition weights are continuous and input-dependent.

### 3. Forced Top-K with K>=3 (Architectural Constraint)
Given Finding #242's proof that H¹ collapses at exactly K=3, the simplest
architectural fix is to always activate the top-3 highest-gate adapters with
proper weight normalization. AdaMoE (2406.13233) shows that variable-K routing
is achievable with minimal architectural modification (null experts).

### 4. Self-Contrast at Inference (SCMoE, 2405.14507)
Rather than forcing multi-activation during routing, use unchosen experts as
negative contrast at inference time. This extracts value from non-active experts
without the quality degradation of forced multi-activation. Training-free,
applicable to any existing MoE system.

## Implications for Next Experiments

### The SIGReg diagnosis is clear: the DISEASE is the training objective

BCE domain-classification loss cannot produce multi-adapter activation because
it is solving the wrong problem. The five-level proxy chain now extends:

1. PPL doesn't predict MMLU accuracy (Finding #236, r=0.08)
2. MMLU accuracy doesn't predict behavioral quality (Finding #238)
3. PPL improvement sets don't predict specialization (Finding #240)
4. Cosine similarity doesn't predict functional disagreement (Finding #240)
5. **Domain classification doesn't predict composition quality (Finding #243)**

Each level adds a broken link between what we optimize and what we want. The
fix is to optimize composition quality directly — not through proxy metrics.

### Forced top-3 remains viable but untested behaviorally

The phase transition at K=3 (Finding #242) is topologically proven, but
Finding #238 showed metrics can mislead (+700% behavioral / -20pp MMLU).
Before implementing forced top-3, we need behavioral evidence that
obstruction collapse actually improves generated text quality.

### Bridge adapters are still needed under current routing

Since BCE routing produces K=1, the H¹=3 obstructions from Finding #242
persist at inference time. Any composition system using the current
domain-classification router must account for these obstructions — either
through bridge adapters (rank >= 3) or by replacing the routing objective.

## Recommended Follow-Up

### Priority 1: Generation Quality Test (exp_generation_quality_perscale)
**Motivation:** Finding #243 + Finding #238. Before changing the routing objective,
we must establish whether current routed composition produces useful text AT ALL.
This is the P0 existential test from the strategic priorities.
**Literature:** MMLU-Pro (2406.01574) showed 16-33% accuracy drops from format
changes — benchmarks are unreliable proxies for generation quality.
**Why:** If current composition already generates useful text despite H¹=3
obstructions, then obstruction collapse may be unnecessary (and forced
multi-activation may degrade quality per SCMoE's finding). If generation
quality is poor, we know the disease is in the composition, not just the routing.

### Priority 2: Composition-Aware Router Training
**Motivation:** Finding #243 (BCE is the disease) + RDLC/L-MoE (composition-aware
loss works elsewhere).
**Literature:** L-MoE (2510.17898), RDLC, LoRAuter (2602.21222).
**Why:** Train router to optimize quality of composed output, not domain
classification. This makes multi-adapter activation the OPTIMAL solution rather
than forcing it architecturally.

### Priority 3: Self-Contrast Inference (Training-Free)
**Motivation:** SCMoE (2405.14507) extracts value from unchosen experts without
quality degradation.
**Literature:** SCMoE improved Mixtral GSM8K from 61.79 to 66.94 (training-free).
**Why:** If our K=1 routing is fundamentally correct for quality, self-contrast
can extract additional value from non-active adapters without changing the
routing mechanism. Zero training cost, immediate applicability.
