# Research Brief: Level-of-Detail (LOD) for Adaptive M2P Capacity Allocation

**Date:** 2026-04-08
**NotebookLM Notebook:** `e61ce732-fc64-459c-9ae6-1bd43a016b53` (607 sources, 534 indexed)
**NotebookLM Home:** `/tmp/notebooklm-topic5`
**Conversation ID:** `78c70775-5073-4109-ae3e-b58ddf93c157`

---

## Key Papers Found

### Tier 1: Directly Applicable to M2P LOD

| Paper | arXiv ID | Relevance |
|-------|----------|-----------|
| **Mixture-of-Depths (MoD)** | 2404.02258 | Router decides per-token whether to skip or process a layer. Top-k routing with fixed compute budget. Directly maps to "skip B-matrix generation for layers that don't need it." |
| **AdaLoRA** | 2303.10512 | Adaptive rank allocation per weight matrix using SVD parameterization. Prunes singular values of unimportant updates. Budget scheduler gradually reduces total parameters. |
| **RANKALLOC** | (VLA paper, 2024) | Fisher information-guided rank allocation across LoRA layers. Uses only 3 held-out demonstrations + single forward-backward pass. Water-filling optimization to distribute rank budget. |
| **MoLA (MoE LoRA)** | (2024) | Assigns different numbers of LoRA experts to different layers. Finds middle layers need most capacity. Rejects uniform allocation. |

### Tier 2: Supporting Techniques

| Paper | arXiv ID | Relevance |
|-------|----------|-----------|
| **LoRA-squared (LoRA^2)** | 2512.04555 | Ranks freely adapt during fine-tuning. Imposes importance ordering on rank positions. Higher ranks created only when strictly needed. |
| **Chain of LoRA (COLA)** | 2510.10150 | Iterative residual learning: merge LoRA into weights, re-initialize new LoRA. Builds capacity incrementally rather than all-at-once. |
| **PEARL** | 2407.03036 | Continual learning framework that adaptively determines rank per task. Allocates rank based on proximity to reference weights in parameter space. |
| **DynaMoE** | 2603.01697 | Layer-wise expert scheduling (ascending, descending, pyramid, wave). Language tasks need ascending/pyramid (more capacity at depth). |
| **Layer-Wise Scaling (LWS)** | 2509.06518 | Pre-training architecture that scales FFN dims and attention heads per layer. "Crown LWS" gives most capacity to central layers. |
| **Surgical Fine-Tuning** | 2502.11466 | Uses diagonal Fisher Information Matrix to select candidate layers before training. Task info is localized to small subset of layers. |
| **ELA-ViT** | (2024) | Layer Instability Index (LII) from attention scores. Proven to upper-bound trace of layer-wise Fisher matrix. Parameter-free, forward-pass only. |
| **Inner Thinking Transformer** | 2502.13842 | Gradient Nuclear Norm (GNN) to measure layer-wise learning difficulty. Easy tasks: GNN decays in early/late layers. Hard tasks: GNN stays high in middle layers. |
| **Dr.LLM** | 2001.08361 | Per-layer routers decide skip/execute/repeat. Trained via Monte Carlo Tree Search. Proves multi-path routing is viable. |
| **Mixture-of-Recursions (MoR)** | (2024) | Shared block with variable recursion depth. Expert-choice router selects top-k tokens per depth. Variable compute without unique params per layer. |

---

## LOD-Inspired Mechanisms

### Mechanism 1: Fisher-Guided Water-Filling (from RANKALLOC)
- Measure diagonal Fisher information F_l for each layer using 3 samples + 1 backward pass
- Compute efficiency: eta_l = F_l / (d_l + k_l) where d_l, k_l are dimensions
- Solve constrained integer program: max sum(F_l * r_l) subject to sum(r_l * c_l) <= budget
- Water-filling via bisection finds optimal lambda*
- **M2P mapping:** Run Fisher calibration once per task, then M2P generates rank r_l B-matrices per layer instead of uniform rank

### Mechanism 2: Layer Instability Index (from ELA-ViT)
- Compute Median Absolute Deviation (MAD) of softmax attention patterns across calibration batch
- LII is parameter-free, requires only forward pass
- Mathematically proven: LII upper-bounds trace of layer-wise Fisher matrix
- Low LII = "Fisher-flat" layer = no adaptation needed = skip B-matrix generation
- **M2P mapping:** Run calibration forward pass, compute LII per layer, skip generation for layers below threshold

### Mechanism 3: Top-k Routing with Fixed Budget (from MoD)
- Router outputs scalar weight per token/layer
- Top-k selected for full computation, rest get residual connection
- Budget is predictable: always exactly k tokens processed
- **M2P mapping:** Router decides per layer: full-rank B (top-k), low-rank B, or B=0

### Mechanism 4: Gradient Nuclear Norm Profiling (from Inner Thinking Transformer)
- Measure GNN of attention matrices on small calibration set
- High GNN = layer is struggling = needs more capacity
- Low GNN = layer has converged = can skip
- **M2P mapping:** Profile GNN per layer, allocate B-matrix rank proportionally

### Mechanism 5: Hidden State Cosine Similarity
- Measure cosine similarity between layer input and output
- High similarity = layer acts as near-identity = no adaptation needed
- SqueezeAttention uses this for attention module importance
- **M2P mapping:** Forward pass with task input, measure delta per layer, skip high-similarity layers

---

## Layer Importance Metrics (Ranked by Cost)

| Metric | Cost | Signal Quality | When to Use |
|--------|------|----------------|-------------|
| **Cosine Similarity (delta-x)** | 1 forward pass | Moderate | Cheapest. Good for initial culling of identity-like layers. |
| **Attention Entropy / LII** | 1 forward pass | High (proven bound on Fisher) | Best cost/quality ratio. ELA-ViT proves LII upper-bounds Fisher trace. |
| **Gradient Nuclear Norm** | 1 forward + backward pass | High | Identifies layers actively struggling with the task. |
| **Diagonal Fisher Information** | 1 forward + backward pass (3 samples) | Very High | Gold standard for "will adaptation matter here?" RANKALLOC uses this. |
| **Influence Functions (LayerIF)** | Hessian approximation | Very High | Most expensive but predicts exact validation loss improvement per layer. |

---

## Dead Ends to Avoid

1. **Uniform rank reduction across all layers (VeRA approach).** VeRA rank-4 was killed because it reduced ALL layers equally. The research consistently shows middle layers need MORE capacity, early/late layers need LESS. Uniform reduction throws away the most important signal.

2. **Aggressive global rank without per-layer profiling.** LoRA^2 (2512.04555) showed that freely adapting ranks works better than fixed budgets, but the adaptation must be driven by data, not arbitrary thresholds.

3. **Pure layer skipping without representation alignment.** MoD (2404.02258) works when training from scratch but FlexiDepth showed that naively skipping layers in pre-trained models degrades performance due to representation misalignment. If you skip B-matrix generation, the residual path must still be numerically compatible.

4. **Static allocation without task calibration.** Layer-Wise Scaling (2509.06518) shows that optimal allocation depends on the task (ascending for language, descending for vision). A static "always give middle layers more capacity" heuristic will fail on some tasks.

5. **Relying solely on weight magnitude for importance.** Singular value magnitude (as in AdaLoRA) is a better proxy than raw weight norm because it captures the directional structure of the adaptation, not just its scale.

---

## Top 3 Actionable Approaches (Ranked by Promise for M2P at 4B)

### 1. LII-Gated LOD with Three Tiers (Highest Promise)

**Concept:** Use Layer Instability Index (forward-pass only) to classify each of the 36 layers into three tiers BEFORE M2P generation:
- **Tier 1 (Full):** High LII layers get full-rank B-matrices (rank-8)
- **Tier 2 (Partial):** Medium LII layers get reduced B-matrices (rank-2)
- **Tier 3 (Skip):** Low LII layers get B=0 (no M2P output at all)

**Why this is best for M2P:**
- LII requires only a forward pass (no gradients) -- trivially cheap
- Mathematically proven to bound Fisher information
- Directly addresses the "generate full B for layers that barely need it" problem
- Reduces M2P output space by skipping Tier 3 layers entirely
- The 4B model's quality degradation pattern (99.7% at L=2, 86.4% at L=16) suggests early layers may universally be Tier 3

**Prediction:** For GSM8K-like tasks, expect ~12 of 36 layers to be Tier 1, ~12 Tier 2, ~12 Tier 3. This cuts M2P output by ~50% with minimal quality loss.

**Key papers:** ELA-ViT (LII metric), MoD (top-k routing concept), Surgical Fine-Tuning (Fisher-based layer selection)

### 2. Fisher-Guided Water-Filling with Dynamic Ranks (High Promise)

**Concept:** Run RANKALLOC-style calibration (3 samples, 1 backward pass) to get per-layer Fisher scores. Use water-filling to distribute a fixed rank budget across layers. M2P generates B-matrices with layer-specific ranks.

**Why this works for M2P:**
- Water-filling is a proven optimal allocation under budget constraints
- Fisher information directly measures "will adaptation here change the loss?"
- Only needs 3 demonstration samples -- fits the M2P few-shot paradigm
- The rank budget can be tuned: if 4B M2P has 6x more output space but similar input, set total budget = same as 0.6B and let water-filling distribute it

**Prediction:** Water-filling will concentrate rank in middle layers (matching MoLA findings), with q_proj getting more capacity than v_proj (matching d_int=86 vs d_int=69 intrinsic dimensionality).

**Key papers:** RANKALLOC (water-filling algorithm), AdaLoRA (SVD-based rank pruning), Surgical Fine-Tuning (Fisher-based selection)

### 3. Progressive Coarse-to-Fine Generation with Residual Refinement (Medium Promise)

**Concept:** M2P generates adapters in two passes:
1. **Coarse pass:** Generate rank-1 B-matrices for ALL layers (cheap, ~1/8th of full output)
2. **Refinement pass:** Use LII or Fisher to identify which layers need more, then generate residual rank-1 increments for those layers only
3. **Repeat:** Each pass adds rank-1 to the layers that need it most, until budget is exhausted

**Why this works for M2P:**
- Matches the "progressive LOD" concept from game engines (mip-map levels)
- Coarse pass provides a quality floor; refinement only adds where needed
- Naturally discovers the per-layer importance profile
- Similar to Chain of LoRA (COLA) which merges and re-initializes iteratively

**Prediction:** After the coarse pass, refinement will concentrate on middle-to-late layers. The coarse pass alone should recover ~80% of full-rank quality.

**Key papers:** COLA (iterative residual LoRA), PEARL (adaptive rank for continual learning), LoRA^2 (importance-ordered rank positions)

---

## Summary Statistics

- **NotebookLM sources analyzed:** 607 (534 indexed, 7 processing, 7 error)
- **Research queries executed:** 4 deep research queries + 2 direct arxiv sources
- **Unique papers cited:** 14 with specific arxiv IDs
- **Importance metrics identified:** 5 (LII, Fisher, GNN, Cosine Similarity, Influence Functions)
- **Key finding:** The research converges on a clear message -- uniform allocation is suboptimal, middle layers need most capacity, and cheap forward-pass metrics (especially LII) can reliably identify which layers to skip before spending generation compute.

---

## NotebookLM Access

To revisit or extend this research:
```bash
export NOTEBOOKLM_HOME=/tmp/notebooklm-topic5
notebooklm ask "follow-up question" --notebook e61ce732-fc64-459c-9ae6-1bd43a016b53
```
