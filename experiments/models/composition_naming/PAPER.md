# Composition Naming: Research Digest

## Hypothesis

A clear, distinctive name and glossary for the composable architecture will
reduce ambiguity in communication, differentiate from existing work, and
provide a stable terminology for all future papers and code.

**(No kill criteria -- this is a framing/naming task, not an experiment.)**

## What This Survey Is

A systematic review of naming conventions across five research communities
(MoE, PEFT, model merging, continual learning, modular deep learning) to
identify: (1) which terms are taken, (2) which are available, and (3) what
name best captures the three defining properties of our architecture --
orthogonal independence, additive composition, and evolutionary lifecycle.

## Lineage in the Arena

```
(all prior experiments) --> composition_naming (this survey)
```

This is a meta-task. It depends on the full body of findings, not a single
parent experiment.

## Key References

| Paper | Abbreviation | Year | Key Terms Introduced |
|-------|-------------|------|---------------------|
| Shazeer et al. | MoE | 2017 | experts, gating network, sparsely-gated |
| Fedus et al. | Switch | 2022 | switch routing, capacity factor, expert parallelism |
| Jiang et al. | Mixtral | 2024 | sparse mixture of experts (SMoE), top-2 routing |
| DeepSeek-AI | DeepSeek-V3 | 2024 | auxiliary-loss-free load balancing, fine-grained experts |
| Hu et al. | LoRA | 2021 | low-rank adaptation, trainable rank decomposition matrices |
| Yadav et al. | TIES | 2023 | task vector, trim-elect-merge, sign conflict |
| Lialin et al. | ReLoRA | 2023 | merge-and-restart, high-rank from low-rank updates |
| Liang et al. | InfLoRA | 2024 | interference-free LoRA, orthogonality constraints |
| Yu et al. | MoE-Adapters4CL | 2024 | MoE adapters, frozen backbone, task-incremental |
| Pfeiffer et al. | Modular DL | 2023 | modules, routing, aggregation, composition function |
| Li et al. | BTM | 2022 | branch-train-merge, embarrassingly parallel training |
| Huang et al. | LoRAHub | 2023 | cross-task LoRA composition, gradient-free optimization |
| Yang et al. | Union of Experts | 2024 | intrinsic routing, expert self-routing via SVD |

## 1. Literature Survey: Terminology Landscape

### 1.1 Individual Components

| Term | Used By | Meaning | Conflict with Us? |
|------|---------|---------|-------------------|
| Expert | MoE, Switch, Mixtral, DeepSeek | Specialized FFN block within a layer | **Partial.** We use "expert" similarly but ours are LoRA deltas, not full FFN copies. |
| Adapter | Houlsby 2019, LoRA | Bottleneck module injected into transformer | **Partial.** We use LoRA adapters but "adapter" is generic. |
| Module | Modular DL survey | Autonomous unit of computation | **Low conflict.** Very generic. |
| Task vector | TIES-Merging | Delta between fine-tuned and pretrained weights | **Useful analog.** Our expert deltas ARE task vectors. |
| Branch | BTM | Independently trained copy of the model | **Low conflict.** Different mechanism. |

### 1.2 Composition Mechanisms

| Term | Used By | Meaning | Conflict with Us? |
|------|---------|---------|-------------------|
| Routing / Gating | MoE, Switch, Mixtral | Learned softmax router selects experts | **High conflict.** Our routing is hash-based, not learned. |
| Merging | TIES, Model Soups | Combining multiple model weights | **Medium.** We do additive composition, not weight averaging. |
| Aggregation | Modular DL survey | Combining module outputs | **Medium.** Generic enough to reuse. |
| Composition function | Modular DL survey | f(module_1, ..., module_k) | **Good fit.** This is exactly what we do. |
| Injection | LoRA | Adding low-rank matrices to frozen weights | **Low conflict.** Describes a single adapter, not N. |

### 1.3 Architecture Patterns

| Term | Used By | Meaning | Conflict with Us? |
|------|---------|---------|-------------------|
| Mixture of Experts (MoE) | Widely used | Conditional computation with learned routing | **High conflict.** Implies learned routing, joint training. |
| Sparse MoE (SMoE) | Mixtral | MoE with top-k sparsity | **Same issue.** |
| Modular deep learning | Survey paper | Broad framework for composable modules | **Too broad.** Covers everything from adapters to neural module networks. |
| Parameter-efficient fine-tuning (PEFT) | LoRA, InfLoRA | Small trainable additions to frozen model | **Describes ingredients, not architecture.** |
| Branch-Train-Merge | Li et al. 2022 | Train branches independently, merge at end | **Closest analog.** But BTM merges permanently; we compose at runtime. |

### 1.4 Evolution/Lifecycle

| Term | Used By | Meaning | Conflict with Us? |
|------|---------|---------|-------------------|
| Continual learning | InfLoRA, MoE-Adapters4CL | Sequential task learning without forgetting | **Partial.** We avoid forgetting by design, not by constraint. |
| Interference-free | InfLoRA | Orthogonal subspaces prevent task conflict | **Close analog.** Our cos=0.0002 achieves this naturally. |

**Key gap identified:** No existing term captures "runtime-composable, independently-trained, orthogonal experts with evolutionary lifecycle." The closest is Branch-Train-Merge, but it lacks the runtime composition and evolution aspects.

## 2. Analysis: What Names Are Taken vs Available

### 2.1 TAKEN (avoid as primary name)

| Name | Why Taken |
|------|-----------|
| MoE / Mixture of Experts | Implies learned routing, joint training |
| SMoE / Sparse MoE | Same as MoE |
| Switch/Mixtral | Specific architectures |
| LoRAHub | Existing paper (gradient-free LoRA composition) |
| Branch-Train-Merge | Existing paper (permanent merge, no runtime composition) |
| TIES / DARE | Specific merging algorithms |
| Modular Deep Learning | Survey term, too broad |
| InfLoRA / ReLoRA | Specific LoRA variants |
| MoE-Adapters | Existing paper |

### 2.2 AVAILABLE (no conflicts found in literature)

| Name | Notes |
|------|-------|
| Composable Expert Architecture (CEA) | Descriptive but generic |
| Orthogonal Expert Composition (OEC) | Captures the key mechanism |
| Expert Fabric | Metaphor: woven from independent threads |
| Adapter Mesh | Metaphor: like service mesh in distributed systems |
| Neural Lego | Metaphor: snap-together components |
| Living Model | Captures evolution but not mechanism |
| Evolvable Model | Similar to Living Model |

### 2.3 "Model Soups" and "Expert Soups"

Worth noting: Wortsman et al. (2022) coined "Model Soups" for averaging
fine-tuned model weights. The metaphor is charming but describes permanent
averaging, not runtime composition. "Expert Soups" would be confusingly
close. Avoid soup metaphors.

## 3. Recommended Name

### Primary: **Structurally Orthogonal Latent Experts (SOLE)**

**Pronunciation:** "sole" (one syllable, like the word)

**Justification:**

The name encodes the properties that make this architecture unique, while
aligning with existing literature and acknowledging prior work:

1. **Structurally** -- orthogonality is a structural property of high-dimensional
   low-rank subspaces, not enforced via Gram-Schmidt (MDM-OC), constrained
   initialization (OSRM/InfLoRA), or penalty terms. This is our key
   differentiator: orthogonality is free at sufficient d, not trained.

2. **Orthogonal** -- experts occupy independent subspaces (cos=0.0002),
   guaranteeing non-interference. This is the safety property. Prior work
   (InfLoRA, OSRM) enforces orthogonality; we observe it emerges naturally.

3. **Latent** -- experts exist as latent low-rank deltas in weight space,
   not as separate FFN blocks (MoE) or full model copies (BTM). This
   aligns with "latent experts" terminology in the MoE literature (e.g.,
   SMILE's zero-shot SVD experts) and distinguishes from explicit routing.

4. **Experts** -- the standard MoE term for specialized components.
   This grounds the work in the MoE literature while the "Structurally
   Orthogonal Latent" prefix signals the departure.

**Why renamed from OAE (Orthogonal Additive Experts):**
- "Additive" described the composition mechanism but is not distinctive --
  all LoRA composition is technically additive. The real insight is that
  the orthogonality is structural, not enforced.
- "Latent" better captures the nature of the experts (weight-space deltas
  that are never materialized as separate models) and connects to existing
  literature (latent experts, latent skills).
- SOLE aligns with prior work terminology: MDM-OC (orthogonal delta
  merging), OSRM (constrained LoRA init), SMILE (zero-shot SVD experts),
  InfLoRA (interference-free LoRA). Our differentiator: orthogonality is
  structural at sufficient d, not enforced -- no Gram-Schmidt, no
  constrained init, no penalty terms needed.

What it deliberately omits:
- "LoRA" -- the architecture is broader than LoRA. Any orthogonal
  adapter (IA3, VeRA, etc.) would work in principle, though LoRA is
  currently the only practical choice.
- "Living" / "Evolvable" -- evolution is a lifecycle feature, not a
  structural property. The clone-and-compete mechanism deserves its own
  name within the glossary, not in the architecture name.
- "Mixture" -- we are NOT a mixture. There is no learned gating. The
  experts are selected (by hash ring, semantic search, or manual
  assignment), not mixed.

### Why not the other candidates?

| Candidate | Verdict | Reason |
|-----------|---------|--------|
| CEA (Composable Expert Architecture) | Rejected | "Composable" is too generic -- every MoE is composable. |
| OEC (Orthogonal Expert Composition) | Close second | "Composition" describes the operation, not the experts themselves. SOLE is noun-centric ("what are they?") while OEC is verb-centric ("what do they do?"). |
| Adapter Mesh | Rejected | "Mesh" implies interconnection. Our experts are independent, not meshed. |
| Neural Lego | Rejected | Too informal for papers. Works for blog posts. |
| Expert Fabric | Rejected | "Fabric" implies weaving/interdependence. Our key property is independence. |
| Living Model | Deferred | Better as the name for the SYSTEM (infrastructure + lifecycle), not the architecture pattern. |

### The Full Naming Hierarchy

```
Living Composable Model (the system)
  |
  +-- SOLE: Structurally Orthogonal Latent Experts (the architecture pattern)
  |     |
  |     +-- Skeleton: frozen base model (attention + embeddings)
  |     +-- Expert: rank-r LoRA delta on FFN layers
  |     +-- Expert Library: collection of all available experts
  |     +-- Composition: additive combination of selected expert deltas
  |     |
  |     +-- Routing (how experts are selected):
  |     |     +-- Hash routing: deterministic, zero-recalibration
  |     |     +-- Semantic routing: embedding similarity
  |     |     +-- Manual assignment: human-specified
  |     |
  |     +-- Evolution (how experts improve):
  |           +-- Clone-and-compete: fork expert, fine-tune clone, tournament
  |           +-- Shadow scoring: parallel evaluation without serving impact
  |           +-- Lineage: version chain of an expert through evolution
  |
  +-- Distillation pipeline (how experts are created)
  +-- Compose CLI (how experts are managed)
  +-- vLLM serving (how inference runs)
```

## 4. Glossary of Terms

### Architecture Components

| Term | Definition | Formal Notation |
|------|-----------|-----------------|
| **Skeleton** | The frozen base model parameters (attention layers, embeddings, layer norms). Shared by all experts. Never modified after initial training. | W_s in R^{d_out x d_in} |
| **Expert** | A rank-r LoRA delta applied to FFN layers only. Independently trained, independently deployable. | dW_i = (alpha/r) * B_i @ A_i where A_i in R^{r x d_in}, B_i in R^{d_out x r} |
| **Expert Library** | The full collection of available experts (stored on NVMe). Only k experts are active per token. | E = {dW_1, ..., dW_N} |
| **Composition** | The additive combination of skeleton + selected expert deltas. No learned weights, no optimization. | y = (W_s + sum_{i in S} w_i * dW_i) @ x where S is the selected subset |
| **Orthogonality guarantee** | The structural property that independently-trained LoRA experts occupy near-orthogonal subspaces, ensuring non-interference under addition. | cos(vec(dW_i), vec(dW_j)) ~ O(1/d) for i != j |
| **Active parameters** | The parameters used per token: skeleton + k expert deltas. | P_active = P_skeleton + k * r * (d_in + d_out) * L |
| **Stored parameters** | Total parameters across all experts. | P_stored = P_skeleton + N * r * (d_in + d_out) * L |

### Routing Terms

| Term | Definition |
|------|-----------|
| **Hash routing** | Deterministic expert selection using consistent hashing on query features. Zero-recalibration: adding/removing an expert displaces only 1/N of traffic. |
| **Semantic routing** | Expert selection using embedding similarity between query and expert description/centroid. |
| **Selection set** | The k experts chosen for a given input. S subset {1, ..., N}, |S| = k. |
| **Displacement** | The fraction of queries that change their selected expert when a new expert is added or removed. |

### Evolution Terms

| Term | Definition |
|------|-----------|
| **Clone-and-compete** | The evolution mechanism: clone an expert, fine-tune the clone with corrections, run both in parallel, prune the loser. |
| **Shadow scoring** | Evaluating a candidate expert on real traffic without serving its outputs to users. Comparison by next-token perplexity. |
| **Tournament** | The parallel evaluation period during which original and clone compete on shared traffic. |
| **Lineage** | The version history of an expert: python_v1 -> python_v2 -> python_v3. |
| **Correction** | A training signal (unit test failure, teacher judgment, human feedback) that triggers a clone-and-compete cycle. |

### Lifecycle Terms

| Term | Definition |
|------|-----------|
| **Distillation** | Creating an expert by fine-tuning a LoRA adapter on teacher-generated training data. |
| **Registration** | Adding an expert to the library and routing table. Immediate, no recalibration. |
| **Pruning** | Removing a losing expert after a tournament. Immediate, no recalibration. |
| **Hot-swapping** | Replacing one version of an expert with another during serving (zero downtime). |

## 5. Positioning Against Related Work

| Dimension | Traditional MoE | BTM | LoRAHub | MDM-OC / OSRM / InfLoRA | SMILE | **SOLE (ours)** |
|-----------|----------------|-----|---------|-------------------------|-------|----------------|
| Expert type | Full FFN block | Full model branch | LoRA adapter | LoRA adapter | SVD factors | LoRA delta |
| Training | Joint (all experts + router) | Independent (per branch) | Independent | Independent (constrained) | Zero-shot (post-hoc SVD) | Independent (unconstrained) |
| Router | Learned softmax | N/A (merge permanently) | Gradient-free optimization | Task ID / sequential | Learned gate | Hash ring (deterministic) |
| Composition | Weighted sum at runtime | Permanent weight averaging | Learned coefficients | Sequential / merged | Weighted sum | Additive (unit weights) |
| Orthogonality | Not guaranteed | Not analyzed | Not analyzed | **Enforced** (Gram-Schmidt / constrained init / penalty) | Not guaranteed | **Structural** (emerges from d >> r^2) |
| Add/remove expert | Retrain router | Retrain merge | Re-optimize | Retrain constraints | Re-decompose | Instant (zero recalibration) |
| Evolution | Not supported | Not supported | Not supported | Not supported | Not supported | Clone-and-compete |
| Interference control | Balance loss, capacity factor | Linear mode connectivity | Task similarity | Orthogonal projection / penalty terms | SVD separation | Structural orthogonality (no enforcement needed) |

### What Makes SOLE Distinct

1. **Orthogonality is structural, not enforced.** MDM-OC uses Gram-Schmidt orthogonalization at merge time. OSRM constrains LoRA initialization to orthogonal subspaces. InfLoRA adds interference-free penalty terms during training. SOLE requires none of this: independently-trained rank-r LoRA experts in dimension d are structurally near-orthogonal (cos ~ O(r/sqrt(d))), verified empirically at cos=0.0002 for d=896. No Gram-Schmidt, no constrained init, no penalty terms needed.

2. **No learned routing.** MoE requires training a router. BTM requires merge optimization. LoRAHub requires gradient-free search. SOLE uses deterministic hash routing or semantic matching -- no training whatsoever.

3. **Instant expert management.** Adding or removing an expert requires no retraining, no re-optimization, no recalibration. This is a direct consequence of structural orthogonality + additive composition.

4. **Evolutionary lifecycle.** No other architecture supports runtime evolution of experts through a competitive mechanism. MoE experts are fixed after training. BTM branches are merged once. SOLE experts evolve continuously.

5. **Composition safety by structure, not by training.** MoE uses balance losses. TIES uses sign-conflict resolution. InfLoRA uses orthogonality constraints during training. SOLE achieves orthogonality structurally (cos ~ O(1/d) for random low-rank subspaces), requiring no special training procedure.

## Micro-Scale Limitations

This is a naming/framing task, not an experiment. The terminology is scale-independent. However:

- The orthogonality guarantee (cos ~ O(1/d)) improves with model dimension, so "Structurally Orthogonal" in SOLE is MORE justified at macro scale than micro.
- Additive composition has only been tested at micro scale (d=64, N=4-20). At macro scale with hundreds of simultaneously active experts, higher-order interactions may require composition weights.
- The glossary assumes FFN-only experts. If attention-layer experts become viable at scale, the glossary would need updating.

## What Would Kill This

### The Name
- Discovery of an existing paper or system already named "SOLE" or "Structurally Orthogonal Latent Experts" in the ML literature.
- Community adoption of a different name for the same concept (e.g., if "Composable LoRA" becomes standard).

### The Framing
- If the orthogonality guarantee breaks at macro scale (cos >> O(1/d) for production-scale models), the "Orthogonal" claim becomes misleading.
- If structural orthogonality does not hold without enforcement (i.e., constrained init or penalties are actually needed at scale), the "Structurally" claim is invalidated.
- If the architecture requires components beyond LoRA (making "Experts" too narrow), the name would need broadening.

## Artifacts

- `micro/models/composition_naming/PAPER.md` -- this document
- `micro/models/composition_naming/MATH.md` -- formal notation for composition operation
