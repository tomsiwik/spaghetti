# Learnings: exp_adapter_specialization_emergence

## Core Finding

Orthogonal Grassmannian A matrices do NOT induce adapter specialization when all
adapters are trained on the same mixed data. All 10 adapters converge to identical
PPL profiles (CV 0.1-0.4% across adapters per domain, silhouette exactly 0.0).

## Why This Happened

### Orthogonality prevents interference but does not create competition

The A_i^T A_j = 0 guarantee means adapters operate in independent subspaces. This
is the exact property that makes composition safe -- but it also means there is no
inter-adapter competition. In MoE, experts self-specialize because the gating
function creates competitive pressure (one expert winning for a sample reduces
gradient signal to others). Without gating, all adapters receive identical gradient
signal from all data.

### The projection rotation is invisible to the optimizer

When A is frozen and B starts at zero, the gradient for B_i is dL/dy * x^T * A_i.
Since A_i is orthogonal to A_j, these are different gradients -- but they produce
equivalent weight updates in the sense that delta_W_i = B_i @ A_i^T has the same
effect on the loss landscape. The optimization finds the same basin rotated into
each adapter's subspace.

### Training convergence is nearly deterministic

All 10 adapters converge to the same loss (1.65-1.66) despite different A matrices.
The loss trajectory difference is < 0.01 at every checkpoint. This means the
initial A rotation does not create meaningfully different optimization landscapes.

## Confirming Evidence

1. **exp_softmax_router_scaling LEARNINGS:** Showed LoRA activation magnitudes are
   domain-independent (in/out ratio 1.08x). This is consistent with our finding that
   adapters learn similar perturbations regardless of which subspace they operate in.

2. **exp_cross_adapter_knowledge_transfer (KILLED):** Found 0/20 pairwise transfers
   >2%. This is also consistent -- adapters are not learning domain-specific features
   that could transfer.

## Contradicting Evidence

1. **FlyLoRA (arxiv 2510.08396):** Claims frozen random A produces comparable quality
   to learned A. Our experiment does not contradict this -- all mixed adapters DO
   improve PPL (mean -30% vs base). The claim was about quality equivalence of random
   vs learned A, not about specialization induction.

## Alternative Approaches (if revisiting)

1. **Different data ordering per adapter.** Give each adapter a different shuffle of
   the mixed data. This tests data-ordering specialization, not projection-geometry.

2. **Competitive gating during training.** Add a lightweight gating layer that routes
   samples to adapters based on current adapter confidence. This reintroduces the MoE
   competition mechanism that our setup deliberately removed.

3. **Much longer training (2000+ steps).** With more gradient signal, small initial
   differences from A projections might amplify. But the < 0.01 loss difference at
   200 steps suggests amplification would be negligible.

## Implications for SOLE

1. **Grassmannian A matrices serve one purpose: interference prevention.** They do
   NOT serve as implicit routers or specialization inducers.

2. **Domain-specific training data is mandatory.** There is no shortcut to
   specialization -- each adapter must see domain-focused data.

3. **The training pipeline (curate data -> train per-domain -> compose) is correct.**
   Attempts to simplify by mixing all data fail completely.
