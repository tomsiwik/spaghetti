# Pathway Preservation — Experiment Design

## Grounding: Key Papers

### Persistent Homology (TDA)
| Paper | arXiv | Why it matters |
|-------|-------|----------------|
| Neural Persistence | 1812.09764 | Foundational: weight-space filtrations → PH complexity measure. Our pathway graph extends this to co-activation. |
| Deep Graph Persistence | 2307.10865 | Fixes caveats of neural persistence (dominated by weight variance). Extends filtration to whole network — we need this for cross-layer pathways. |
| Persistent Topological Features in LLMs | 2410.11042 | Zigzag persistence across LLM layers. Introduces "persistence similarity" metric — candidate for our decomposition diff. |
| PH for BERT Compression | 2312.10702 | Uses 0-dim PH for neuron importance → pruning. Achieves 58% retention. Proves PH-guided compression works. We extend from pruning to decomposition. |
| TDA for NN Survey | 2312.05840 | Comprehensive survey. Covers activation space, weight space, training dynamics. Essential reference. |
| Probing Neural Topology of LLMs | 2506.01042 | Graph probing of LLM neuron connectivity. Topology outperforms activation probing 130%. Identifies hub neurons — our "major bridges." |
| TDA Beyond PH Review | 2507.19504 | Covers sheaf theory, persistent Laplacians, de Rham cohomology. State of the art beyond classical PH. |

### Sheaf Theory
| Paper | arXiv | Why it matters |
|-------|-------|----------------|
| Knowledge Sheaves | 2110.03789 | **Critical.** Recasts knowledge graph embedding as finding global sections of a knowledge sheaf. Consistency constraints from schema. Directly maps to our adapter composition as sheaf gluing. |
| Neural Sheaf Diffusion | 2202.04579 | Foundational sheaf NN. Shows non-trivial sheaves control asymptotic diffusion behavior. Our restriction maps between adapters are sheaf morphisms. |
| Sheaf Theory → Deep Learning | 2502.15476 | 2025 survey connecting sheaf Laplacians, heat diffusion, message passing to deep learning. Accessible entry point. |
| Copresheaf TNNs | 2505.21251 | NeurIPS 2025. Unifying framework: each local region gets its own feature space with learnable maps. This IS our adapter-as-local-section idea, formalized differently. |
| Sheaf HyperNetworks for FL | 2405.20882 | Uses sheaf-structured hypernetworks for personalized federated learning. Direct analogue: each "client" = adapter, sheaf coordinates composition. |
| Federated Multi-Task via Sheaves | 2502.01145 | Sheaf Laplacian regularization for heterogeneous clients. Our adapters ARE heterogeneous "clients" in a composition system. |

### Bridge / Interaction Terms
| Paper | arXiv | Why it matters |
|-------|-------|----------------|
| TuckA | 2511.06859 | Tucker decomposition → tensor experts. Core tensor captures multi-linear interactions between expert slices. Our pairwise C_ij bridges are a special case. |
| CRAFT | 2602.17510 | Cross-layer Tucker (HOSVD) on attention weights. Frozen factors + lightweight transforms. Template for our bridge adapter architecture. |
| Partial Information Decomposition for KD | 2411.07483 | **Critical.** Uses PID to quantify transferred knowledge: redundant, unique, synergistic. Our "lost pathways" = synergistic information not captured by any single adapter. |
| Crosscoders (Anthropic) | transformer-circuits.pub/2024 | Cross-layer shared features. Resolves cross-layer superposition. Our cross-adapter pathways are analogous to cross-layer features. |
| TC-LoRA | 2508.03999 | CP decomposition on stacked LoRA tensors. Disentangles task-specific vs shared. Our framework explains WHY shared factors matter (they're the bridges). |

### Composition Limits (the disease we're treating)
| Paper | arXiv | Why it matters |
|-------|-------|----------------|
| Pause Recycling LoRAs | 2506.13479 | **Critical negative result.** LoRA unlikely to yield compositional behavior for two-hop reasoning. Our bridge terms are exactly the fix — they encode the hop. |
| Understanding LoRA as Knowledge Memory | 2603.01097 | Maps LoRA design space: misrouting and parameter interference as key bottlenecks. Our sheaf gluing conditions prevent misrouting. |
| Forgetting of Task-Specific Knowledge | 2507.23311 | Shared knowledge preserved during merging, unshared rapidly degrades. Our persistence diagram distinguishes shared (low persistence, redundant) from unshared (high persistence, critical). |
| LoRA Learns Less, Forgets Less | 2405.09673 | Full finetuning learns 10-100x higher rank perturbations. Our rank budget from dim(H^1) predicts exactly how much is lost. |

---

## Experiment Pipeline

### Phase 1: Can We See the Pathways? (Measurement)

#### Exp 1: Pathway Graph Construction on BitNet-2B
**Type:** Verification
**Cites:** 1812.09764 (neural persistence), 2506.01042 (neural topology probing)
**Hypothesis:** The co-activation pathway graph of BitNet-2B has non-trivial topological structure — high-persistence features exist that are not captured by top-k singular vectors.

**Method:**
1. Sample 10K inputs from mixed-domain corpus (5 domains × 2K)
2. For each FFN layer, record activation vectors h(x)
3. Compute SVD of activation matrix → singular directions V
4. Build co-activation graph: edge (v_i, v_j) weighted by co-activation frequency
5. Compute 0-dimensional persistent homology (connected components)
6. Plot persistence diagram

**Kill criteria:**
- K1: Persistence diagram has ≥10 features with persistence > 0.1 (non-trivial topology exists)
- K2: High-persistence features are NOT simply the top-k singular vectors (correlation < 0.5 between persistence rank and singular value rank)

**Predictions from MATH.md:**
- The pathway graph should show a power-law distribution of persistence values
- Cross-domain inputs should create the longest-persisting bridges

**Scale:** micro (single layer first, then extend)
**Compute:** ~30 min on M5 Pro (10K forward passes + PH computation)

---

#### Exp 2: Persistence Diagram Diff — Before vs After Decomposition
**Type:** Verification
**Cites:** 2312.10702 (PH for compression), 2410.11042 (persistent features in LLMs)
**Hypothesis:** Decomposing BitNet-2B into 5 domain adapters creates a measurable bottleneck distance in the persistence diagram — pathways are lost.

**Method:**
1. Use existing 5-domain adapters from Finding #44
2. Compute pathway graph G_full (full model)
3. Compute pathway graph G_composed (base + 5 adapters, uniform 1/N)
4. Compute bottleneck distance d_B(Dgm_full, Dgm_composed)
5. Identify the high-persistence features in Dgm_full that are absent in Dgm_composed — these are the lost bridges

**Kill criteria:**
- K1: d_B > 0 (decomposition is NOT topologically lossless)
- K2: ≥3 lost features have persistence > median persistence (important pathways were lost, not just noise)
- K3: Lost features correlate with cross-domain inputs (Jaccard overlap > 0.3 between lost-feature activation set and multi-domain inputs)

**Predictions:**
- d_B should be significantly > 0 (based on Finding #68: weight orth ≠ data orth)
- The lost features should disproportionately come from domain-overlap regions

**Scale:** micro
**Compute:** ~1hr (two full pathway graph constructions + PH)

---

### Phase 2: Do Sheaf Overlaps Exist? (Structure Discovery)

#### Exp 3: Knowledge Region Overlap Mapping
**Type:** Guided exploration
**Cites:** 2110.03789 (knowledge sheaves), 2502.15476 (sheaf theory survey)
**Hypothesis:** Domain adapters have non-trivial knowledge overlaps — inputs where multiple adapters contribute — and these overlaps are structured (not random).

**Method:**
1. For each of 5 domain adapters, compute "improvement set" U_i = {x : PPL_adapter(x) < PPL_base(x)}
2. Compute all pairwise overlaps U_i ∩ U_j
3. For each overlap, measure representation compatibility: cos(h_i(x), h_j(x)) for x in overlap
4. Build the Čech nerve of the cover {U_i} — this is the simplicial complex whose topology determines sheaf cohomology feasibility

**Kill criteria:**
- K1: At least 3 non-empty pairwise overlaps exist (|U_i ∩ U_j| > 50 samples)
- K2: Compatibility varies within overlaps (std of cosine similarity > 0.1) — i.e., some overlap points are compatible, some aren't. Uniform compatibility means no bridge is needed.

**Predictions:**
- Medical-legal, code-math overlaps should be largest (semantic proximity)
- Compatibility should be bimodal: some overlap inputs are well-served by either adapter, others are "contested"

**Scale:** micro
**Compute:** ~45 min

---

#### Exp 4: Sheaf Cohomology Dimension Estimation
**Type:** Frontier extension
**Cites:** 2202.04579 (neural sheaf diffusion), 2502.01145 (sheaf for federated multi-task)
**Hypothesis:** The first sheaf cohomology H^1 of the adapter knowledge cover is non-zero and low-dimensional — bridge adapters are needed but cheap.

**Method:**
1. From Exp 3, construct the adapter cover nerve
2. For each overlap U_i ∩ U_j, compute the representation difference δ_ij(x) = h_i(x) - h_j(x)
3. Collect all δ_ij vectors → matrix D
4. Compute rank of D → this approximates dim(H^1)
5. Compare rank(D) to adapter rank r — if rank(D) << r, bridges are cheap

**Kill criteria:**
- K1: rank(D) > 0 (non-trivial cohomology — bridges ARE needed)
- K2: rank(D) < 2r (bridges are cheaper than a full adapter)

**Predictions from MATH.md §5.4:**
- dim(H^1) should be O(number of incompatible overlap pairs)
- If finding #38 is correct (orthogonality ≠ specialization), H^1 should be non-trivial

**Scale:** micro
**Compute:** ~20 min (reuses Exp 3 data)

---

### Phase 3: Can We Build the Bridges? (Construction)

#### Exp 5: Persistence-Guided Bridge Extraction
**Type:** Guided exploration
**Cites:** 2411.07483 (PID for KD), 2312.10702 (PH for compression)
**Hypothesis:** High-persistence lost features from Exp 2 can be encoded as a sparse low-rank correction that restores topological fidelity.

**Method:**
1. From Exp 2, take the lost high-persistence features
2. For each lost feature, identify the weight directions it corresponds to
3. Construct a sparse bridge matrix B by projecting the residual R onto these directions
4. Measure: does base + adapters + B restore the persistence diagram?
5. Measure: does it improve PPL on cross-domain inputs?

**Kill criteria:**
- K1: Bridge matrix B reduces d_B by ≥50% (topological restoration works)
- K2: Bridge matrix B has rank < r (it's cheaper than another adapter)
- K3: PPL on overlap inputs improves ≥5% with bridge vs without

**Predictions:**
- B should be very sparse — most weight directions are NOT high-persistence lost features
- The rank of B estimates the practical cost of pathway preservation

**Scale:** micro
**Compute:** ~1hr

---

#### Exp 6: Pairwise Bridge Adapters (C_ij)
**Type:** Guided exploration
**Cites:** 2511.06859 (TuckA), 2508.03999 (TC-LoRA), 2506.13479 (LoRA composition limits)
**Hypothesis:** The bridge correction B_S decomposes into pairwise terms C_ij, and most C_ij are near-zero (sparse bridge graph).

**Method:**
1. From Exp 5, decompose bridge B into pairwise contributions:
   - For each pair (i,j), compute C_ij = projection of B onto the subspace activated by U_i ∩ U_j inputs
2. Measure ||C_ij|| for all pairs → build the bridge graph
3. Threshold at noise level → sparse bridge adjacency matrix
4. Compare to the sheaf nerve from Exp 3 — do the non-trivial bridges match the non-trivial overlaps?

**Kill criteria:**
- K1: ≥60% of C_ij are below noise threshold (bridge graph is sparse)
- K2: Non-trivial C_ij correspond to non-trivial overlaps from Exp 3 (correlation > 0.5)
- K3: Using only top-k C_ij (by norm) recovers ≥80% of full bridge B's benefit

**Predictions from MATH.md §4.3:**
- Finding #31 killed pair-level caching at 3.9% hit rate. But that was for routing, not for knowledge correction. Bridge sparsity should be different — we predict ~20-40% non-trivial pairs at N=5.

**Scale:** micro
**Compute:** ~1hr

---

### Phase 4: Does It Scale? (Validation)

#### Exp 7: Bridge Cost vs N Scaling
**Type:** Frontier extension
**Cites:** 2405.09673 (LoRA learns less forgets less), 2507.23311 (forgetting task-specific knowledge)
**Hypothesis:** Bridge parameter budget scales sub-quadratically with N — despite O(N^2) possible pairs, the sparse bridge graph grows slowly.

**Method:**
1. Repeat Exp 6 at N=5, 10, 15, 24 (using existing adapter pools)
2. At each N, measure: total bridge parameters, number of non-trivial C_ij, dim(H^1)
3. Fit scaling curve: bridge_params = f(N)
4. Extrapolate to N=50, N=100 — does it fit in 48GB?

**Kill criteria:**
- K1: Bridge params at N=24 < 1 full adapter (bridge is cheap relative to composition)
- K2: Growth is sub-quadratic (exponent < 2.0 in power law fit)
- K3: Extrapolated N=100 bridge fits in 48GB memory budget

**Depends on:** Exp 5, Exp 6 results

**Scale:** micro → macro
**Compute:** ~4hrs

---

#### Exp 8: Persistence-Weighted Distillation
**Type:** Frontier extension
**Cites:** 2411.07483 (PID for KD), 2510.13182 (IT criteria for KD)
**Hypothesis:** Weighting the distillation loss by pathway persistence (penalize more for losing high-persistence features) produces adapters that preserve bridges natively — without needing explicit bridge adapters.

**Method:**
1. Compute persistence diagram of teacher (full model)
2. For each training batch, identify which singular directions activate
3. Weight the KD loss by the persistence of those directions:
   L = Σ_i persistence(v_i) · KL(teacher_i || student_i)
4. Train 5 domain adapters with persistence-weighted KD
5. Compare pathway preservation (d_B) vs standard KD adapters

**Kill criteria:**
- K1: d_B(persistence-weighted) < 0.5 × d_B(standard) — better pathway preservation
- K2: Domain PPL not worse than standard adapters (±2%)
- K3: Overlap-region PPL improves ≥5% vs standard

**Predictions:**
- This should produce adapters that naturally allocate rank budget to bridges
- The persistence weighting is information-theoretically grounded: high-persistence = high structural information

**Scale:** micro
**Compute:** ~6hrs (training 5 adapters with modified loss)

---

## Dependency Graph

```
Exp 1 (pathway graph) ──→ Exp 2 (decomposition diff) ──→ Exp 5 (bridge extraction)
                                                              │
Exp 3 (overlap mapping) ──→ Exp 4 (cohomology dim)          │
        │                                                     ↓
        └──────────────────────────────────────→ Exp 6 (pairwise C_ij)
                                                     │
                                                     ↓
                                                Exp 7 (N scaling)

Exp 1 ──→ Exp 8 (persistence-weighted KD)  [independent track]
```

**Critical path:** Exp 1 → Exp 2 → Exp 5 → Exp 6 → Exp 7
**Independent track:** Exp 1 → Exp 8 (can run in parallel after Exp 1)
**Support track:** Exp 3 → Exp 4 (feeds into Exp 6 but not blocking)

## Tool Requirements

- `ripser` or `giotto-tda` for persistent homology computation
- `gudhi` for bottleneck distance
- Existing BitNet-2B + 5 domain adapters (Finding #44)
- MLX for forward passes and activation collection

## Total Estimated Compute

~14 hours on M5 Pro for full pipeline (critical path ~9hrs)
