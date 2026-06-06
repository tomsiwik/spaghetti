# LEARNINGS: Energy Gap Routing at N=24 Scaling

## Core Finding

Energy gap argmin routing collapses from 88% accuracy (N=5) to 8.3% (N=24) due to
adapter strength disparity, NOT domain similarity confusion. Two "loud" adapters
(health_fitness +2.95 nats, code +1.90 nats) absorb nearly all routing decisions
regardless of query domain. The N=5 result (Finding #185) was an artifact of five
well-matched adapters, not evidence of a robust routing mechanism. Argmin over
uncalibrated scores is a well-known MoE failure mode.

## Why This Happened

### The Loudest-Voice Problem (Well-Known in MoE Literature)

Energy gap routing computes DeltaE_i = NLL(adapted_i) - NLL(base) and selects
argmin_i DeltaE_i. This implicitly assumes all adapters have comparable NLL
reduction magnitudes on their target domains. When adapter strengths are
heterogeneous (trained on different data quality, different iteration counts,
different domain difficulty), the adapter with the largest absolute NLL reduction
wins regardless of domain relevance.

This is equivalent to the "dominant expert" or "expert collapse" problem documented
extensively in MoE research. NotebookLM sources confirm that MoE routers
inherently exhibit magnitude bias, consistently selecting experts that produce
larger output norms. Industrial evaluations found output distribution discrepancies
cause >90% zero activations on peripheral experts while dominant experts absorb
the workload.

### The Gumbel Analysis Failed Because Its Precondition Was Wrong

MATH.md used Fisher-Tippett/Gumbel extreme value theory to predict gradual
degradation to 60-75% accuracy. The Gumbel analysis assumes i.i.d. competitor
gaps (common mean mu_other and variance sigma^2). The actual adapter mean gaps
ranged from ~0 nats (math) to ~3 nats (health_fitness) -- a 10x range that
catastrophically violates the i.i.d. assumption. The worked example used
fabricated parameters (sigma=0.3, delta=0.5) rather than measured values from
the N=5 experiment, further disconnecting theory from reality.

### N=5 Was Accidentally Calibrated

The 5 adapters (medical/code/math/legal/finance) happened to have comparable
training data quality and specialization strength. This made energy gaps
comparable in magnitude across domains, so argmin selected by domain relevance.
At N=24, the 24 Grassmannian-initialized adapters trained for 200 iterations on
domains with vastly different data characteristics, producing heterogeneous
specialization strengths. The mechanism was never robust -- it was calibrated
by coincidence.

## Confirming Evidence

- **MoE "Standing Committee" phenomenon (NotebookLM sources):** Despite the
  assumption that MoE routing achieves fine-grained specialization, audits reveal
  a compact coalition of experts consistently captures the vast majority of routing
  mass across domains. Our two-attractor collapse (health_fitness + code absorbing
  22/24 domains) is an extreme instance of this documented pattern.

- **MoA: Heterogeneous Mixture of Adapters (arxiv 2506.05928):** Homogeneous
  MoE-LoRA architectures suffer from "representation collapse and expert load
  imbalance." MoA proposes heterogeneous adapter structures specifically to avoid
  this. Confirms that adapter heterogeneity is a known failure trigger for standard
  routing.

- **Our own Finding #186 (legal-finance confusion):** Predicted domain similarity
  would cause confusion at scale. While the actual failure mechanism was different
  (magnitude disparity, not similarity), both are consequences of uncalibrated
  argmin routing.

- **SOLE PPL-probe routing (macro-scale kill):** NLL-based routing collapsed to
  r=-0.63 at macro-scale with real adapters. This prior kill foreshadowed the
  current result: loss-based routing signals degrade with heterogeneous adapters.

## Contradicting Evidence

- **Finding #185 (N=5 success):** 88% accuracy with energy gap routing, now
  reinterpreted as an artifact of matched adapter strengths rather than a robust
  mechanism. The finding remains factually correct for N=5 but should not be
  extrapolated to larger N without calibration.

- **Energy gap AUC=0.942 (Finding #182):** The ranking signal itself is valid.
  The problem is converting ranking to routing via uncalibrated argmin. A
  calibrated routing mechanism using the same energy gap signal could still work.

## Alternative Approaches

- **LoRAuter (arxiv 2601.21795):** Embedding-based routing via task representations.
  Routes queries by sentence embedding similarity to adapter "centroids" computed
  from small validation sets. Scales to 1500+ adapters without adapter-magnitude
  dependence. Training-free, O(1) routing cost. Directly solves the calibration
  problem by operating in embedding space where adapter strength is irrelevant.

- **MoA (arxiv 2506.05928):** Heterogeneous Mixture of Adapters. Uses diverse
  adapter structures (not just LoRA) with learned routing that explicitly handles
  heterogeneous expert capacities. Soft MoA fuses all expert outputs with learned
  weights; Sparse MoA activates experts by contribution, not raw magnitude.

- **Z-score normalization (statistical calibration):** DeltaE_normalized =
  (DeltaE - mu_d) / sigma_d per adapter, using pre-computed statistics from
  adapter training data. Zero new parameters, preserves simplicity of energy gap
  routing. Standard approach in calibration literature but untested in our setting.

- **Z-loss / logit control (DeepSeek-style):** Penalize excessively high router
  logits during training to prevent magnitude dominance. Used alongside auxiliary
  load-balancing losses. Requires router training, not applicable to
  training-free energy gap routing.

- **Tiny routing heads (Finding #179):** Our own project achieved 100% routing
  accuracy with 2.32% overhead using small learned routing heads. These are
  calibrated by construction (trained on labeled routing decisions, not raw NLL
  magnitudes). Already validated at N=5 -- the natural next step is testing at N=24.

- **CoMoL (arxiv 2603.00573):** Dynamic core space merging of LoRA experts.
  Merges adapter parameters in a shared subspace, avoiding routing entirely for
  shared knowledge while routing only specialized components.

## Implications for Next Experiments

1. **Energy gap routing is dead for production.** Three kills now: (a) O(N) overhead
   (#185 LEARNINGS), (b) SFT-routing incompatibility (#187), (c) magnitude collapse
   at N=24 (#189). The ranking SIGNAL survives but needs a calibrated routing MECHANISM.

2. **Tiny routing heads (Finding #179) are the strongest routing candidate remaining.**
   100% accuracy at N=5, learned calibration, 2.32% overhead, no forward-pass cost.
   Testing at N=24 is the natural next experiment. If they maintain accuracy at scale,
   they solve routing definitively.

3. **The N=5 → N=24 scaling gap is diagnostic.** Any routing mechanism must be tested
   at N>=20 to validate scaling. N=5 results are unreliable indicators of mechanism
   robustness -- too easy to succeed by coincidence.

4. **Z-score normalization is the cheapest fix to test.** If it restores routing
   accuracy at N=24 with zero new parameters, it validates the energy gap ranking
   signal while fixing the calibration problem. But it still has the O(N) forward
   pass problem, so it's diagnostic (is the signal real?) not practical.

## Recommended Follow-Up

**Tiny routing heads at N=24 (Finding #179 scaling test):** Test whether the learned
routing heads that achieved 100% accuracy at N=5 maintain accuracy at N=24 with
heterogeneous adapters. Motivation: Finding #179 showed learned heads bypass all
three energy gap failure modes (O(N) cost, SFT incompatibility, magnitude collapse).
Literature support: LoRAuter (arxiv 2601.21795) demonstrates that embedding-based
(not NLL-based) routing scales to 1500+ adapters; our tiny routing heads are
analogous (learned representations, not raw loss). If routing heads work at N=24,
they become the canonical routing mechanism for the architecture.
