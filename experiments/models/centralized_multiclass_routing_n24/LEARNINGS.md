# LEARNINGS: Centralized Multi-Class Routing at N=24

## Core Finding

Centralized softmax routing achieves 39.4% top-1 accuracy at N=24 — virtually identical
to independent binary heads (39.6%). Eliminating the false positive cascade (proven by
Theorem 1) does NOT improve routing accuracy. The bottleneck is mean-pooled hidden state
representation quality: ~6/24 domains have distinctive representations, ~18/24 do not,
producing a ~40% accuracy ceiling that is routing-architecture-independent.

## Why This Happened

### The Representation Bottleneck (Not the Routing Architecture)

The experiment was designed to test whether FPR cascade (Finding #191) was the disease
or a symptom. By replacing 24 independent binary heads with a single softmax head that
structurally eliminates false positive cascade (Theorem 1 verified), we isolated the
variable. Result: identical accuracy on the same 6 succeeding and 18 failing domains.

The real disease is that mean-pooled hidden states from BitNet-2B-4T do not separate
overlapping domains. Domains with distinctive vocabulary (finance, legal, medical, math,
health_fitness, psychology) achieve ~95-100% routing in BOTH architectures. Generic
domains (cooking, philosophy, economics, education, etc.) are near-random in BOTH.

This is consistent with the information bottleneck principle: mean pooling over all tokens
collapses per-token domain cues into a single vector that preserves only the strongest
statistical signatures (specialized vocabulary), losing the distributional information
that distinguishes overlapping domains.

### Five Routing Kills Converge on One Root Cause

| Kill | Mechanism | Accuracy | Apparent Cause | Real Cause |
|------|-----------|----------|----------------|------------|
| #28 | Softmax router (early) | Low | Architecture | Representation |
| #184 | Energy gating | Impossible | Non-negative energy | N/A (structural) |
| #189 | Energy gap argmin | 8.3% | Adapter magnitude | Representation + magnitude |
| #191 | Binary routing heads | 39.6% | FPR cascade | Representation |
| #192/193 | Centralized softmax | 39.4% | None eliminated | Representation |

The progression is illuminating: after eliminating magnitude disparity (#189) and
FPR cascade (#191), the ~40% ceiling persists. This ceiling is the representation
quality limit, not any routing architecture artifact.

### Cover's Theorem Misapplication

The MATH.md cited Cover's function counting theorem to argue linear separability in
d=2560. The reviewer correctly identified this as misapplied: Cover's theorem guarantees
existence of a separating hyperplane for finite point sets in general position, but says
nothing about learnability from 40 samples per class, population generalization, or points
NOT in general position (overlapping domain representations). VC dimension and linear
separability are existence theorems, not learnability guarantees.

## Confirming Evidence

- **HMoRA (ICLR 2025)**: Hierarchical Mixture of LoRA Experts explicitly addresses the
  failure of flat routing by combining token-level and task-level routing in a hierarchical
  manner. Their core insight: different LLM layers capture features at varying granularity,
  and task-level routing from a single representation loses fine-grained information.
  This directly validates our finding that mean-pooled features are insufficient.

- **MoLA (NAACL 2025, arxiv 2402.08562)**: MoE LoRA with layer-wise expert allocation
  found that optimal expert counts vary by layer — deeper layers need more experts because
  representations become more task-specific. Implication: routing from a single layer's
  mean pool misses the layer-dependent information structure.

- **MoE-Sieve (arxiv 2603.24044)**: Routing-guided LoRA confirmed that routing is highly
  skewed in practice, with dominant experts absorbing disproportionate mass.

- **Prior Finding #191 (binary heads N=24)**: Per-head accuracy 87.2% but ranking accuracy
  39.6%. Now proven to be representation-limited, not cascade-limited.

- **Prior Finding #189 (energy gap N=24)**: 8.3% accuracy from magnitude disparity.
  Different symptom (magnitude vs representation), but same underlying weak signal.

## Contradicting Evidence

- **DeepSeekMoE (arxiv 2401.06066)**: Uses normalized sigmoid gating at 256 experts with
  success. However, their experts are trained jointly with the router, and their tokens
  carry much richer per-token features than our mean-pooled post-hoc features. The
  normalization helps, but the jointly-trained feature space is the key difference.

- **Switch Transformer (arxiv 2101.03961)**: Achieves >90% routing accuracy with softmax
  over a linear router. However, Switch routes at the per-token level within a jointly
  trained model — the router learns features specifically optimized for routing, unlike
  our post-hoc extraction from frozen hidden states.

- **NotebookLM found ET (Expert Threshold) routing succeeds** with per-expert EMA
  thresholds calibrated against global token distribution. Key difference: ET calibrates
  independently but against a SHARED reference distribution, and operates per-token.

The common thread in all "contradicting" evidence: successful routing at N>20 uses either
(a) per-token routing, (b) jointly-trained features, or (c) embedding-based retrieval.
None succeed with post-hoc mean-pooled features from a frozen model.

## Alternative Approaches

### 1. Embedding-Based Routing (Strongest Evidence)
- **LoRAuter (arxiv 2601.21795)**: Training-free routing via task representations, scales
  to 1500+ adapters. Achieves 101.2% of oracle adapter performance. Routes via task
  embeddings derived from validation examples, NOT hidden state features. O(T) complexity
  where T = number of tasks. This completely bypasses the representation bottleneck by
  routing in a different feature space entirely.
- **LoraRetriever (arxiv 2402.09997)**: Retrieve-then-compose framework encoding LoRA
  adapters into a vector space using an instruction-tuned retriever. Ranks adapters by
  input-adapter embedding similarity.
- **Task-Aware Composition (arxiv 2602.21222)**: Vector database approach with nucleus
  sampling for adapter selection. Scales to 22+ diverse tasks.

### 2. Per-Token Routing
- **LoRA-Flow (arxiv 2402.11455)**: Dynamic per-token mixture weights via lightweight
  gating. Routes at token level, preserving fine-grained domain cues that mean pooling
  destroys.
- **MoLoRA (arxiv 2603.15965)**: Per-token learned routing, Qwen3-1.7B + 4 adapters
  beats 8B model. Confirmed in our own reference list.
- **HMoRA (ICLR 2025)**: Hierarchical token+task routing with layer-wise granularity.

### 3. Hierarchical Routing
- **H-MoE (arxiv 2410.02935)**: Two-stage coarse-to-fine gating. First routes to
  super-expert groups, then to specific experts within groups. Reduces the effective
  class count at each stage.
- **HiLoRA (arxiv 2510.12266)**: Adaptive hierarchical LoRA routing for domain
  generalization.

## Implications for Next Experiments

1. **Mean-pooled hidden states are a dead end for routing at N>10.** Five kills confirm
   this. No further routing architecture experiments on mean-pooled features are warranted.

2. **The ~40% ceiling is a representation ceiling.** The 6 domains that route correctly
   have specialized vocabulary (medical, legal, finance, code, math, psychology). The 18
   that fail share generic language patterns. This is a property of the base model's
   representation, not the routing mechanism.

3. **LoRAuter is the strongest candidate for our system.** It (a) scales to 1500+ adapters,
   (b) is training-free, (c) operates on task embeddings not hidden states, (d) achieves
   >100% oracle performance. It completely sidesteps our representation bottleneck.

4. **Per-token routing is the alternative path** but requires training infrastructure that
   mean-pooled approaches avoid. MoLoRA and LoRA-Flow both show strong results.

5. **Adapter quality is suspect.** Individual adapter PPL (10.119) is WORSE than base PPL
   (10.057). If adapters don't specialize, no routing mechanism can extract signal that
   doesn't exist. This should be investigated separately.

## Recommended Follow-Up

1. **LoRAuter-style embedding routing (HIGHEST PRIORITY)**: Implement task-embedding-based
   routing using a pre-trained sentence encoder. Motivation: LoRAuter (arxiv 2601.21795)
   achieves 101.2% oracle performance at 1500+ adapters. This bypasses the representation
   bottleneck entirely — routes on input text embeddings, not hidden state features.
   Would fix: the mean-pooled representation ceiling.

2. **Adapter quality audit**: Before any more routing experiments, verify that the 24
   adapters actually specialize. If individual PPL ≈ base PPL, routing is moot regardless
   of mechanism. Quick diagnostic: compare per-domain PPL reduction for each adapter.

3. **Hierarchical domain clustering**: Cluster the 24 domains into ~6 groups based on
   embedding similarity. Use proven N=5 mechanisms (100% accuracy) within clusters.
   Motivation: H-MoE (arxiv 2410.02935) shows two-stage routing reduces effective class
   count and improves accuracy. Would fix: high N by reducing it to manageable groups.
