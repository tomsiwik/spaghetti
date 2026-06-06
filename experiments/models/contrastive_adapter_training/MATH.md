# Contrastive Orthogonal Adapter Training: Mathematical Framework

## Type: Guided Exploration (Type 2)

The mathematical framework for contrastive orthogonality is established (LoRACLR, 2412.09622;
NeuroLoRA, 2603.12378; InfLoRA, 2404.00228). The unknown is: does forcing weight-space
decorrelation during training produce genuinely domain-specialized adapters (not just
format-specialized ones)?

## A. Failure Mode: Format Specialization Masquerading as Domain Expertise

**Disease (not symptom):** SFT adapters learn instruction-following format, not domain
knowledge. Evidence:
- Finding #208: Code adapter beats domain-specific on 4/5 domains
- Finding #212: Code adapter DEGRADES standardized benchmarks (GSM8K -18pp, HumanEval -15pp)
- LIMA hypothesis (2305.11206): SFT teaches format alignment, knowledge comes from pre-training

**Why this is a stable fixed point:** Standard SFT minimizes cross-entropy on domain text.
The easiest gradient direction is shared formatting patterns (instruction parsing, response
structure). With limited training (200-300 steps), adapters converge to similar format-teaching
weight directions, explaining why code (most structured format) appears universal.

Formally: Let Delta_i = B_i A_i be the weight update for adapter i. Under standard SFT:

    Delta_i = argmin_Delta L_task(W + Delta; D_i)

where D_i is domain data. If D_1, ..., D_N share format structure F and differ in content C_i:

    Delta_i approx Delta_F + delta_C_i

where Delta_F >> delta_C_i (format gradient dominates content gradient in early training).
This produces cos(Delta_i, Delta_j) >> 0 for all pairs.

## B. The Right Question

**Wrong:** "How do we prevent adapter interference?"
**Right:** "What training loss forces adapter weight deltas to encode domain-specific
information orthogonal to the shared format direction?"

The answer: a contrastive penalty that makes high inter-adapter similarity COSTLY, forcing
each adapter to find a distinct direction even if the dominant shared gradient is format.

## C. Prior Mathematical Foundations

### Contrastive Orthogonality Loss (NeuroLoRA, 2603.12378)

NeuroLoRA defines a Contrastive Orthogonality Loss (COL) between expert parameters:

    L_COL = (1/N(N-1)) * sum_{i != j} |tr(Delta_i^T Delta_j)| / (||Delta_i||_F * ||Delta_j||_F)

This penalizes the cosine similarity between flattened weight deltas, forcing experts to
occupy orthogonal subspaces.

### LoRACLR Contrastive Merging (2412.09622)

LoRACLR uses contrastive alignment in weight space to merge multiple LoRA models while
preserving concept-specific representations. The key insight: align positive pairs (same
concept) and push apart negative pairs (different concepts) in weight space.

### InfLoRA Orthogonal Projection (2404.00228)

InfLoRA projects new task parameters onto the orthogonal complement of previous task
subspaces, preventing catastrophic forgetting. For task t:

    A_t = (I - P_{t-1}) * A_t_init

where P_{t-1} is the projection onto the subspace of tasks 1..t-1.

### Critical Caveat: Orthogonality != Semantic Disentanglement (2510.03262)

Rethinking Inter-LoRA Orthogonality (2510.03262) proves that guaranteed weight-space
orthogonality does NOT ensure semantic compositionality. Our OSRM finding confirms this
empirically (100% pairs fail OSRM).

**Implication for this experiment:** We use contrastive loss not to guarantee semantic
disentanglement (which weight-space methods cannot provide), but to force adapters to
learn DIFFERENT features from their domain data. The hypothesis is that with sufficient
decorrelation pressure, the "different features" will be domain-specific rather than
format-specific, because format is the SHARED direction that the contrastive loss pushes
away from.

## D. Mechanism Description (Not a Formal Proof)

**Conjecture 1 (Contrastive Decorrelation of Format Component).**
Let Delta_i = Delta_F + delta_C_i for adapters i = 1,...,N, where Delta_F is the shared
format direction and delta_C_i is the domain-specific component. Let the training loss be:

    L_total = L_task(Delta_i; D_i) + lambda * L_COL(Delta_1, ..., Delta_N)

where L_COL = (1/N(N-1)) sum_{i != j} cos^2(vec(Delta_i), vec(Delta_j)).

If lambda is sufficiently large, then at convergence:

    ||Delta_F|| <= epsilon(lambda)

where epsilon(lambda) -> 0 as lambda -> infinity.

*Mechanism sketch (not a proof).* The contrastive loss L_COL is minimized when all
Delta_i are mutually orthogonal. Since Delta_F is the shared component, it contributes
to L_COL for all N(N-1)/2 pairs. Specifically, the cosine between Delta_i and Delta_j
when both share Delta_F depends on the ratio ||Delta_F||^2 / (||Delta_F + delta_C_i|| *
||Delta_F + delta_C_j||), NOT on ||Delta_F||^4 as previously stated (that was a
dimensionality error). The gradient of L_COL with respect to Delta_F pushes the shared
component toward zero. At a critical point of L_total, the task loss gradient pulling
Delta_F positive is balanced by the contrastive gradient pushing it negative:

    dL_task/dDelta_F = lambda * dL_COL/dDelta_F

For large lambda, the contrastive term dominates, shrinking Delta_F.

**Unstated assumptions:**
1. Format is a single shared direction. In reality, format could be multi-dimensional.
   The contrastive loss cannot distinguish "shared format" from "shared beneficial
   transfer" — it suppresses ALL shared structure indiscriminately.
2. Domain content components delta_C_i are assumed naturally decorrelated. But domain
   content may share structure (e.g., medical and biology overlap). This assumption
   is never verified.
3. No convergence guarantee. The claim "at convergence" assumes L_total converges
   under the round-robin optimization used in implementation (see Section G2 below),
   which is not proven.

**This is a mechanism description, not a formal proof.** It describes the intuitive
dynamics of the joint loss but lacks: stated assumptions with conditions, convergence
analysis, and tight bounds. This is a Type 2 guided exploration: the mechanism is
grounded in established techniques (NeuroLoRA COL, LoRACLR), but the quantitative
predictions require empirical calibration.

### LoRA Scale and the Format-Knowledge Tradeoff

**Observation (Scale-Dependent Degradation).**
At LoRA scale s, the adapter contribution is s * B @ A. Finding #212 showed that at s=20,
the adapter OVERWRITES pre-trained capability. The effective perturbation magnitude is:

    ||s * Delta||_F = s * ||Delta||_F

This is a trivial property of scalar multiplication, not a derived result. At s=20, the
perturbation is 20x base, explaining why capability degrades. At s=2, the perturbation
is 10x smaller, preserving more pre-trained knowledge. No bound is given on what scale
is "safe" — the choice of s=2.0 is informed by Finding #212 but not mathematically derived.

**Prediction (empirical, not proof-derived):** At s=2.0, adapters should preserve base
capability while adding domain specialization.

## D. Predictions

### Behavioral Predictions (from Theorem 1)
1. With contrastive loss, inter-adapter cosine similarity will be significantly lower
   than without (~0.3 -> <0.1)
2. Domain-specific adapter will beat code adapter on its own domain (the format direction
   being suppressed means domain content dominates)
3. Low LoRA scale (2.0) will preserve base capability (GSM8K >= 50%)

### Quantitative Predictions
| Prediction | Expected | Kill threshold |
|-----------|----------|----------------|
| Inter-adapter cos sim (contrastive) | < 0.1 | > 0.3 would indicate contrastive loss failed |
| Code adapter universality (alpha on 4+ domains) | < 0.9 | >= 0.9 triggers K617 |
| Training convergence | loss < 2x baseline | > 2x triggers K618 |
| Domain adapter vs base PPL ratio | < 1.0 on own domain | > 1.0 on ALL domains triggers K619 |
| Domain adapter beats code on own domain | >= 15% advantage | < 15% = hypothesis not supported |

## E. Assumptions and Breaking Conditions

1. **Format is a low-rank shared direction.** If format is high-dimensional and varied per
   domain, the contrastive loss won't selectively suppress it.
   - Breaking: If cos(Delta_i, Delta_j) is already low without contrastive loss, the
     format-sharing hypothesis is wrong.

2. **Domain content is orthogonal to format.** If domain knowledge requires the SAME
   weight directions as format, contrastive loss will suppress both.
   - Breaking: If contrastive adapters perform worse than standard on ALL domains, this
     assumption fails.

3. **200-300 training steps are sufficient.** With contrastive loss adding a penalty,
   the effective learning rate for task loss is reduced.
   - Breaking: If training doesn't converge (K618), may need more steps.

4. **LoRA scale 2.0 is sufficient for domain learning.** If scale is too low, adapters
   may not learn anything distinguishable.
   - Breaking: If all adapters have near-zero effect, scale is too low.

## F. Worked Example (d=16, r=4, N=2 adapters)

Two adapters (code, math) with shared format direction:
- Delta_code = Delta_F + delta_code, Delta_math = Delta_F + delta_math
- Delta_F = [[0.1, 0.1], [0.1, 0.1], ...] (uniform format direction)
- delta_code = [[0.05, -0.02], ...], delta_math = [[-0.01, 0.03], ...]

Without contrastive loss:
- cos(Delta_code, Delta_math) = cos(Delta_F + delta_code, Delta_F + delta_math)
- Since ||Delta_F|| >> ||delta_C||: cos approx 1.0

With contrastive loss (lambda = 1.0):
- L_COL = cos^2(Delta_code, Delta_math)
- Gradient pushes Delta_F -> 0, leaving:
- cos(delta_code, delta_math) approx 0.0 (naturally decorrelated domain content)

Result: Adapters become domain-specific rather than format-specific.

## G2. Implementation-Theory Gap

**IMPORTANT:** The implementation diverges from the mechanism described above in several
ways that may affect conclusions:

1. **Round-robin vs joint optimization.** The mechanism describes joint training where all
   adapters are updated simultaneously with a shared contrastive penalty. The implementation
   uses round-robin: each step trains ONE adapter on its domain data, cycling through domains.
   This is a different optimization problem.

2. **Stale contrastive gradients.** The contrastive penalty uses stored parameters from other
   adapters (from their last update step), not current parameters. Between updates, up to
   N-1=4 other adapters may have changed. The gradient is computed with respect to stale
   neighbors.

3. **Frequency.** Contrastive gradients are only computed every 5 steps (batch_idx % 5 == 0),
   not every step. For 4 out of 5 steps, only the task loss drives optimization.

4. **Partial parameters.** Contrastive loss is computed only on lora_b parameters, not the
   full Delta = B @ A. This means the contrastive signal does not account for lora_a.

These approximations make the optimization cheaper but less faithful to the stated mechanism.
The 99.6% cosine reduction suggests the approximation is sufficient for achieving weight
decorrelation, but may not achieve the same convergence properties as joint optimization.

## G. Complexity and Architecture Connection

**Training overhead:**
- Standard SFT per adapter: O(T * L * d * r) per step (T tokens, L layers, d model dim, r rank)
- Contrastive loss: O(N^2 * L * d * r) per step (comparing all adapter pairs per layer)
- For N=5, L=30, d=2560, r=16: contrastive adds ~25 Frobenius inner products = negligible

**Memory overhead:**
- Must hold all N adapter parameter sets simultaneously: N * L * 7 * (d*r + r*d) params
- For N=5: 5 * 30 * 7 * 2 * 2560 * 16 = ~108M extra params (bfloat16 = ~216MB)
- Within M5 Pro 48GB budget

**Implementation:** Joint training loop where all N adapters are updated each step on their
respective domain data, with the contrastive penalty computed across all pairs.

## Self-Test (MANDATORY)

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   **Honest answer: no impossibility property is proven.** The contrastive orthogonality
   loss makes high inter-adapter cosine similarity costly, which we conjecture forces the
   shared format component to shrink. This is a mechanism description, not a proven
   impossibility guarantee. A true impossibility property would bound ||Delta_F|| <= epsilon
   under stated conditions — we do not have this.

2. **Which existing theorem(s) does the proof build on?**
   NeuroLoRA Contrastive Orthogonality Loss (2603.12378), LoRACLR contrastive merging
   (2412.09622), LIMA Superficial Alignment Hypothesis (2305.11206).

3. **What specific numbers does the proof predict?**
   Inter-adapter cosine < 0.1 (vs ~0.3 without). Domain adapter beats code on own domain
   by >= 15%. Training loss < 2x baseline.

4. **What would FALSIFY the proof (not just the experiment)?**
   If format is NOT a shared low-rank direction (i.e., standard SFT adapters already have
   low inter-adapter cosine), the format-sharing hypothesis is wrong.

5. **How many hyperparameters does this approach add?**
   Count: 2 (lambda for contrastive weight, LoRA scale). Lambda could be derived from
   the ratio of format-to-content gradient norms, but this is unknown a priori (Type 2
   exploration). LoRA scale is informed by Finding #212 (scale=20 too high).

6. **Hack check: Am I adding fix #N to an existing stack?**
   No. This is a single mechanism (contrastive loss during training) that addresses the
   root cause (format dominance in adapter specialization). It replaces the previous
   approach (independent SFT per domain) rather than adding to it.
