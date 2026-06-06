# LEARNINGS: Sheaf Cohomology Dimension Estimation

## Core Finding

The Čech nerve of a top-2 specialization cover over 4 active LoRA adapters on
BitNet-2B-4T has first Betti number β₁ = 3, meaning three independent pairwise
incompatibility cycles exist that cannot be transitively reconciled. At k=3,
all cycles collapse (H¹ = 0), revealing a sharp topological phase transition
in adapter compatibility as the routing granularity relaxes.

## Why This Happened

### 1. The k=2 nerve is a complete graph K₄ with no triangles — a known topological obstruction generator

When each sample appears in exactly 2 cover sets and 4 adapters are active, the
nerve is necessarily the complete graph K₄ (6 edges, 0 triangles). The first Betti
number β₁ = E - V + c = 6 - 5 + 2 = 3 (including isolated finance vertex) follows
from elementary graph theory. The three independent cycles correspond to three
independent bases of the cycle space of K₄.

This is NOT a surprise in hindsight: any complete graph K_n on n ≥ 3 vertices
without filled triangles has β₁ = C(n,2) - n + 1 = n(n-1)/2 - n + 1. For n=4,
β₁ = 3. The topological result is a combinatorial consequence of having 4 specialized
adapters with strict top-2 routing.

### 2. The phase transition at k=3 is the "transitive reconciliation" threshold

At k=3, every sample appears in 3 cover sets, creating triangular faces that fill
all cycles. H¹ drops from 3 to 0. This is the topological signature of a well-known
phenomenon in adapter merging: pairwise conflicts can be resolved when a third adapter
provides a "bridge" pathway. The LoRI framework (2504.07448) exploits exactly this
principle — fixed random projections create an implicit shared subspace that
transitively reconciles pairwise conflicts.

### 3. Full-rank edge difference matrices confirm genuine incompatibility

All 6 edge difference matrices are full-rank (rank = number of overlap samples,
13-68), meaning every shared sample contributes a unique incompatibility direction.
This rules out the possibility that adapter disagreements are low-dimensional
artifacts. The adapters genuinely represent shared samples differently — consistent
with findings from "Adapter Merging Reactivates Latent Reasoning Traces" (2601.18350),
which localized interference to a specific subspace in the final 6-10 layers.

### 4. Finance degeneracy is a scale artifact, not domain property

Finance (scale=1.0) vs others at 20.0 creates complete domination. Finding #235
established binary on/off scale behavior for ternary adapters. At equal scales,
finance would participate and the nerve topology would change (potentially K₅
with β₁ = 6 if still triangle-free at k=2).

## Confirming Evidence

- **Sheaf Cohomology of LPC Networks** (2511.11092, Seely/Sakana AI 2025): Showed
  that sheaf cohomology characterizes irreducible error patterns in predictive coding
  networks that inference cannot remove. Uses Hodge decomposition to determine when
  internal contradictions cause learning to stall. Our H¹ = 3 is analogous —
  irreducible incompatibility patterns that naive merging cannot resolve.

- **Hansen & Ghrist Knowledge Sheaves** (2110.03789): The theoretical framework we
  applied. dim(H¹) counts linearly independent obstruction directions where local
  sections cannot be globally reconciled. Our experiment validates that this framework
  produces non-trivial results on real adapter composition (correcting the degenerate
  result from Finding #240).

- **OSRM** (2505.22934): "Unraveling LoRA Interference: Orthogonal Subspaces for
  Robust Model Merging" — constrains LoRA subspaces pre-training to prevent
  cross-task interference. Validates our finding that pairwise adapter conflicts
  are real and structurally meaningful, not noise. Their orthogonal subspace approach
  implicitly addresses the same obstruction we measure topologically.

- **Finding #240** (our project): Established that specialization sets + L2 norm
  are the correct inputs for sheaf analysis. Without that correction, the cover
  was degenerate and H¹ trivially 0. This experiment is a direct validation of
  the predecessor's recommended follow-up.

- **Finding #68** (our project): Weight-space orthogonality ≠ data-space orthogonality.
  The full-rank edge difference matrices confirm this: adapters may share weight-space
  directions but produce genuinely different representations on shared samples.

## Contradicting Evidence

- **Tensorized Clustered LoRA Merging** (2508.03999): Uses CP decomposition to
  disentangle task-specific and shared factors, achieving successful merging WITHOUT
  bridge adapters. If CP decomposition resolves conflicts without additional rank,
  our "rank budget" conjecture may overestimate what's needed — the obstruction
  might be resolvable by better decomposition rather than additional parameters.

- **LoRAuter** (Adsul et al.): Simple linear merging SURPASSES individual adapters
  on PIQA (70.95% vs 46%). If linear merging works, the topological obstructions
  we measure may not translate to behavioral degradation — the full-rank differences
  might be in dimensions irrelevant to task performance. However, LoRAuter used
  FP16 bases, not ternary.

- **Ortho-LoRA** (2601.09684): Dynamically projects conflicting gradients onto
  orthogonal complements during training. This prevents conflicts from forming in
  the first place, making post-hoc sheaf analysis unnecessary. If adapters are
  trained with orthogonal constraints, H¹ would be 0 by construction.

- **The k=3 collapse itself**: H¹ = 0 at k=3 means the obstruction vanishes when
  routing considers top-3 instead of top-2. If the production routing regime uses
  soft weights over ≥3 adapters (as in our Gumbel-sigmoid router), the topological
  obstruction may be irrelevant in practice.

## Alternative Approaches

### For measuring adapter incompatibility (alternatives to sheaf H¹):

1. **Orthogonal Subspace Rank Merging (OSRM)** (2505.22934): Pre-training
   orthogonalization that prevents conflicts. Measures incompatibility as
   subspace overlap between LoRA update matrices. More actionable than our
   topological approach because it directly informs training constraints.

2. **Activation-Guided Consensus Merging (ACM)** (2505.14009): Layer-specific
   merging coefficients from activation mutual information. Would give a
   DATA-WEIGHTED incompatibility measure rather than our topological (unweighted)
   one. Directly comparable to our H¹ but more informative.

3. **Tensorized Clustered Merging** (2508.03999): CP decomposition identifies
   shared vs task-specific factors across adapters. Would reveal whether our
   3 topological cycles correspond to 3 independent task-specific factors or
   are artifacts of the cover construction.

### For resolving adapter incompatibility (alternatives to bridge adapters):

1. **LoRI** (2504.07448): Fixed random projections + sparse B matrices. Creates
   approximately orthogonal subspaces by design, eliminating interference at
   training time rather than patching it post-hoc. Published result: reduces
   cross-task interference while maintaining single-task performance.

2. **Mediator** (2502.04411): Layer-wise conflict routing. Average low-conflict
   layers, route high-conflict layers to task-specific experts. Our L2 data
   (peaking at layer 15) directly informs which layers need routing.

3. **Twin-Merging** (2406.15479): Dynamically integrates modular expertise by
   separating shared and task-specific knowledge, then selectively activating
   task-specific components at inference. Sidesteps the fixed-bridge approach.

## Implications for Next Experiments

### 1. The topological obstruction is REAL but may not be LOAD-BEARING

H¹ = 3 proves three independent conflict cycles exist. But Finding #238 showed
that behavioral quality can improve DESPITE metric regression (math: +700%
behavioral, -20pp MMLU). The key unanswered question: do these topological
obstructions cause behavioral degradation, or are they in irrelevant dimensions?

**Test:** Compose adapters on samples from each of the 3 cycles. Measure BEHAVIORAL
output quality (not PPL). If quality is fine despite topological conflict, the
obstructions are not load-bearing.

### 2. The k=2 vs k=3 phase transition maps directly to routing strategy

k=2 (strict top-2): H¹ = 3, bridge adapters needed.
k=3 (soft top-3): H¹ = 0, conflicts resolve transitively.

This has a direct architectural implication: if the router activates ≥3 adapters
per token (even with small weights), topological obstructions vanish. The Gumbel-
sigmoid router already does this. So bridge adapters may be unnecessary if routing
is sufficiently soft.

### 3. The scalar-vs-vector H¹ gap is the critical open question

Scalar H¹ = 3 (cycle count) vs edge difference rank 13-68 (per edge). The Rank
Budget Bound conjecture maps scalar cycles to vector-space rank, but this is
unproven. Resolving this determines whether rank-3 bridges suffice or rank-68+
is needed. The sheaf cohomology of LPC networks paper (2511.11092) uses full
vector-valued sheaf Laplacians — their approach would give the data-weighted
answer.

### 4. The four-level proxy chain extends

Finding #236: PPL ↛ MMLU accuracy (r=0.08)
Finding #238: MMLU accuracy ↛ behavioral quality (+700%/-20pp)
Finding #240: PPL improvement sets ↛ specialization structure (degenerate)
Finding #242: Topological obstruction ↛ ??? behavioral impact (UNTESTED)

Each proxy level loses information. The sheaf result is mathematically clean
but its behavioral relevance is unknown.

## Recommended Follow-Up

### Priority 1: Bridge Adapter Rank Ablation (behavioral test)

- **Motivation**: H¹ = 3 predicts rank ≥ 3 bridge needed (conjecture). Edge ranks
  13-68 suggest possibly much more. The scalar-vs-vector gap must be resolved
  empirically.
- **Literature**: OSRM (2505.22934) showed orthogonal subspace rank determines
  merging quality. LoRI (2504.07448) showed fixed projections at any rank reduce
  interference.
- **Design**: Train bridge adapters at rank {1, 3, 8, 16, 32} on overlap samples
  from the 3 conflict cycles. Measure BEHAVIORAL quality (not PPL) on composed
  output. If rank 3 suffices → conjecture supported. If rank >> 3 needed →
  scalar H¹ is an undercount.
- **Kill criterion**: If rank-1 bridge matches rank-32 → topological prediction
  is uninformative.

### Priority 2: Soft-Routing Obstruction Collapse Verification

- **Motivation**: k=3 gives H¹ = 0. If Gumbel-sigmoid router already activates
  ≥3 adapters, bridge adapters are unnecessary.
- **Literature**: MoLoRA (2603.15965) per-token routing; LD-MoLE (2509.25684)
  learnable dynamic routing.
- **Design**: Measure effective k (number of adapters with weight > threshold)
  under current Gumbel-sigmoid routing. If effective k ≥ 3 for most tokens,
  obstructions are already resolved.
- **Kill criterion**: If effective k < 3 for >50% of tokens → bridge adapters
  are needed for those tokens.

### Priority 3: Vector-Valued Sheaf Laplacian (full H¹)

- **Motivation**: Scalar H¹ = 3 is a lower bound. Full vector-valued sheaf H¹
  with R^{2560} stalks gives the actual obstruction dimension.
- **Literature**: Sheaf cohomology of LPC networks (2511.11092) computes
  vector-valued sheaf Laplacians. Sheaf theory survey (2502.15476) provides
  algorithms for sheaf cohomology on arbitrary posets.
- **Design**: Construct full coboundary matrix with vector-valued stalks
  (R^{2560} per vertex), compute H¹ via Hodge Laplacian nullity.
- **Kill criterion**: If vector H¹ = 3 (same as scalar) → scalar suffices.
  If vector H¹ >> 3 → bridge rank budget is much larger than predicted.

## References Added

- 2511.11092: Sheaf Cohomology of Linear Predictive Coding Networks (Seely, Sakana AI)
- 2502.15476: Sheaf Theory: From Deep Geometry to Deep Learning (Ayzenberg & Magai)
- 2508.03999: Tensorized Clustered LoRA Merging for Multi-Task Interference
- 2504.07448: LoRI: Reducing Cross-Task Interference in Multi-Task Low-Rank Adaptation
- 2601.09684: Ortho-LoRA: Orthogonal Gradient Projection for Multi-Task LoRA
- 2601.18350: Adapter Merging Reactivates Latent Reasoning Traces
- 2505.22934: OSRM: Orthogonal Subspaces for Robust Model Merging (already in DB)
- 2410.14837: Topological Obstruction to Training of Shallow ReLU Networks
