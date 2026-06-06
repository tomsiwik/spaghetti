# Pathway Preservation Theory for Adapter Decomposition

## Status: Frontier Extension (brainstorm → formalization)

## 1. The Problem

When a pretrained model W is decomposed into N domain adapters, the decomposition
captures high-magnitude directions (highways) but discards low-magnitude cross-domain
interactions (footpaths). These footpaths are:

- Individually small in Frobenius norm
- Collectively essential for cross-domain inference
- Not owned by any single adapter
- Lost silently during composition when only a subset of adapters is active

**Metaphor (load-bearing):** A city network with rivers. Major bridges (highways) are
replicated in any reconstruction. But small bridges between minor cities — low traffic,
structurally critical for connectivity — vanish. Residents of small cities must now
route through major hubs, losing efficiency and introducing distortion.

**The question:** What mathematics makes it impossible to lose structurally important
pathways during decomposition?

---

## 2. Formal Setup

### 2.1 Weight Decomposition

A pretrained weight matrix decomposes as:

$$W = W_{\text{base}} + \sum_{i=1}^{N} A_i + R$$

where:
- $W_{\text{base}}$ is the shared scaffold (base model or learned skeleton)
- $A_i = B_i C_i^\top$ is a rank-r adapter for domain i
- $R$ is the **residual** — everything no adapter captured

**Standard assumption (wrong):** R is noise. Prune it.

**Our claim:** R contains structurally critical cross-domain pathways. Some of R is
noise, some is load-bearing. We need a principled way to distinguish them.

### 2.2 Pathway Graph

For a given corpus $\mathcal{D}$, define the **pathway graph** $G = (V, E, w)$:

- **Vertices** $V$: singular directions of W (the right singular vectors $v_1, \ldots, v_d$)
- **Edges** $E$: co-activation links. An edge $(v_i, v_j)$ exists if both directions
  activate (above threshold $\epsilon$) for some input $x \in \mathcal{D}$
- **Weights** $w(v_i, v_j) = \frac{|\{x \in \mathcal{D} : |v_i^\top h(x)| > \epsilon \wedge |v_j^\top h(x)| > \epsilon\}|}{|\mathcal{D}|}$

This gives the "usage map" of the model's internal pathways.

---

## 3. Persistent Homology for Pathway Importance

### 3.1 Core Idea

Persistent homology (Edelsbrunner et al., 2000; Zomorodian & Carlsson, 2005) provides
a threshold-free method to identify structurally important features in a weighted graph.

### 3.2 Filtration

Build a **sublevel filtration** on G by decreasing edge weight:

$$G_t = (V, \{e \in E : w(e) \geq t\}), \quad t \in [1, 0]$$

At $t = 1$: only the most-used pathways (highways).
At $t = 0$: all pathways included.

As $t$ decreases, connected components merge. Each merge event records:
- **Birth**: the threshold at which a component appeared
- **Death**: the threshold at which it merged into a larger component
- **Persistence**: death - birth (how long the feature survived)

### 3.3 The Persistence Diagram

The persistence diagram $\text{Dgm}(G)$ is the multiset of (birth, death) pairs.

**Key insight:** A pathway that *connects two otherwise-disconnected knowledge regions*
has high persistence — it's the last edge that merges two components, so it dies late.
This is true regardless of the pathway's magnitude/usage frequency.

A rarely-used pathway between medical and legal knowledge that is the *only* connection
between those regions will have high persistence. A frequently-used pathway within
medical knowledge that has many redundant alternatives will have low persistence.

### 3.4 Structural Importance Score

For each edge $e$ in the pathway graph, define:

$$\text{importance}(e) = \max_{[\text{birth}, \text{death}] \ni w(e)} (\text{death} - \text{birth})$$

i.e., the persistence of the longest-lived feature that this edge participates in.

**Guarantee:** If we preserve all edges with importance above threshold $\delta$,
then the bottleneck distance between the original and preserved persistence diagrams
is at most $\delta$:

$$d_B(\text{Dgm}(G_{\text{original}}), \text{Dgm}(G_{\text{preserved}})) \leq \delta$$

This is a **metric guarantee** — not an empirical threshold but a topological bound.

---

## 4. Decomposition with Pathway Accounting

### 4.1 Instrumented Decomposition

1. **Pre-decomposition:** Compute pathway graph $G_{\text{full}}$ and its persistence
   diagram $\text{Dgm}_{\text{full}}$ from a representative corpus.

2. **Decompose** $W$ into adapters $\{A_i\}$ and residual $R$.

3. **Post-decomposition:** For any composition subset $S \subseteq \{1, \ldots, N\}$,
   compute the pathway graph $G_S$ of $W_{\text{base}} + \sum_{i \in S} A_i$.

4. **Diff:** Compute bottleneck distance $d_B(\text{Dgm}_{\text{full}}, \text{Dgm}_S)$.
   If $d_B = 0$, decomposition is topologically lossless for this subset.
   If $d_B > 0$, the high-persistence features in $\text{Dgm}_{\text{full}} \setminus \text{Dgm}_S$
   are the lost bridges.

### 4.2 The Bridge Adapter

The lost bridges from step 4 define a sparse correction term:

$$W_S = W_{\text{base}} + \sum_{i \in S} A_i + B_S$$

where $B_S$ encodes only the high-persistence pathways missing from the composed model.

**Key property:** $B_S$ depends on which adapters are active. This is the "context-dependent
adapter" from the brainstorm — the correction that reshapes based on composition context.

**Parameter budget:** If most bridge terms have low persistence (noise), $B_S$ is sparse.
The persistence diagram directly tells us the rank budget needed.

### 4.3 Pairwise Interaction Decomposition

If bridge terms are dominated by pairwise interactions, we can factor:

$$B_S \approx \sum_{i,j \in S, i < j} C_{ij}$$

where $C_{ij}$ is a low-rank "bridge adapter" activating when adapters $i$ and $j$ are
co-selected. Total terms: $\binom{N}{2}$, but sparsity (most $C_{ij} \approx 0$) makes
this tractable. The persistence diagram predicts which pairs have non-trivial bridges.

---

## 5. Sheaf Theory for Cross-Adapter Coherence

### 5.1 Motivation

Persistent homology tells us *which* pathways are lost. Sheaf theory tells us how to
*guarantee consistency* when gluing adapter knowledge back together — even when the
adapters are orthogonal in weight space.

The core tension: orthogonality in weight space prevents destructive interference but
also prevents constructive sharing. Two adapters can be perfectly orthogonal yet share
knowledge pathways in activation/data space. Sheaf theory provides the language to
formalize this.

### 5.2 The Knowledge Sheaf

Define a **sheaf** $\mathcal{F}$ over the adapter composition space:

**Base space (topology):** Let $\mathcal{U} = \{U_1, \ldots, U_N\}$ be a cover where
$U_i$ is the "knowledge region" of adapter $i$ — the set of inputs on which adapter $i$
materially improves over base.

**Stalks:** For each point $x$ in input space, the stalk $\mathcal{F}_x$ is the model's
internal representation at $x$ — the full activation vector.

**Sections:** A section over $U_i$ is the function $\sigma_i : U_i \to \mathcal{F}_x$ —
the adapter's contribution to the representation for all inputs it covers.

**Restriction maps:** For overlaps $U_i \cap U_j$ (inputs where both adapters contribute),
the restriction maps $\rho_{ij}$ describe how adapter $i$'s representation relates to
adapter $j$'s representation on shared inputs.

### 5.3 The Gluing Axiom

A sheaf satisfies the **gluing axiom**: if local sections agree on overlaps, they glue
into a unique global section.

**For adapter composition:** If adapters $A_i$ and $A_j$ produce compatible representations
on their shared inputs ($U_i \cap U_j$), their composition produces the correct global
representation. The gluing conditions are the "bridge constraints."

**When gluing fails:** The obstruction to gluing is precisely the cross-domain knowledge
that lives in the overlap but isn't captured by either adapter individually. This is a
**sheaf cohomology class** — it measures the "incompatibility" between local adapter
knowledge on shared regions.

### 5.4 Cohomological Obstruction to Lossless Composition

The first sheaf cohomology group $H^1(\mathcal{U}, \mathcal{F})$ measures the space of
"sections that are locally consistent but fail to glue globally."

$$H^1(\mathcal{U}, \mathcal{F}) = 0 \implies \text{lossless composition is possible}$$
$$H^1(\mathcal{U}, \mathcal{F}) \neq 0 \implies \text{bridge terms are necessary}$$

**The dimension of $H^1$ tells you the rank budget for bridge adapters.**

If $\dim H^1 = k$, you need at most $k$ additional parameters to encode the
cross-adapter pathways. This is a hard mathematical bound — not an empirical guess.

### 5.5 Computing Compatibility on Overlaps

For adapters $A_i, A_j$ and an input $x \in U_i \cap U_j$, the compatibility condition is:

$$\|\sigma_i(x) - \sigma_j(x)\|_{\mathcal{F}_x} < \delta$$

where $\sigma_i(x)$ is the representation contribution of adapter $i$ at $x$.

If this fails for some subset of the overlap, those inputs are the "disconnected small
cities" — they need explicit bridge terms.

### 5.6 Practical Implication: Marked Pathways

During distillation/decomposition, for each pair of adapters $(i, j)$:

1. Identify the overlap region $U_i \cap U_j$ (inputs where both improve over base)
2. Compute the compatibility condition on the overlap
3. Where compatibility fails, extract the **cocycle** — the minimal correction needed
4. Store the cocycle as a sparse bridge term $C_{ij}$

These bridge terms are the "marked pathways" from the brainstorm — cross-domain
connections that are orthogonal in weight space but coupled in knowledge space,
explicitly retained so composition doesn't lose them.

---

## 6. Unified Framework: Topological Decomposition with Sheaf Gluing

### 6.1 The Full Pipeline

```
Full Model W
    │
    ├── [1] Compute pathway graph G_full
    │       Compute persistence diagram Dgm_full
    │       Identify high-persistence features (structural bridges)
    │
    ├── [2] Decompose W = base + Σ adapters + residual
    │
    ├── [3] Build knowledge sheaf F over adapter cover
    │       Compute overlaps U_i ∩ U_j
    │       Check gluing conditions
    │       Compute H^1 (obstruction dimension)
    │
    ├── [4] For each composition subset S:
    │       Compute Dgm_S
    │       Diff: lost_bridges = Dgm_full \ Dgm_S
    │       Extract cocycles from sheaf on overlaps
    │       Construct bridge adapter B_S from cocycles
    │
    └── [5] Serving:
            W_S = base + Σ_{i∈S} A_i + B_S
            Bridge adapter B_S is sparse, context-dependent
            Topological guarantee: d_B(Dgm_full, Dgm_S) ≤ δ
```

### 6.2 What Makes This Impossible to Fail

Three mathematical guarantees:

1. **Persistence gives a metric.** The bottleneck distance is a provable bound on
   topological distortion. If $d_B = 0$, no structural pathway was lost. Period.

2. **Sheaf cohomology gives a hard rank bound.** $\dim H^1$ tells you exactly how many
   bridge parameters you need. You don't guess — the math tells you.

3. **Cocycles give the exact correction.** The sheaf cocycle on each overlap is the
   minimal repair. No more, no less. You're not training a bridge — you're computing it.

### 6.3 What Could Still Go Wrong

- **Computational cost**: Persistent homology on the full activation graph may require
  sampling. The guarantee degrades with sampling quality.
- **Activation threshold $\epsilon$**: The pathway graph depends on this choice. Persistence
  is threshold-free *within* the filtration, but the initial graph construction isn't.
- **Sheaf construction**: Defining "knowledge regions" $U_i$ requires a corpus that
  covers the input space well. Distribution shift breaks the guarantee.
- **Rank of $H^1$**: If the obstruction space is high-dimensional, bridge adapters may
  be expensive. The theory predicts the cost but doesn't reduce it.

---

## 7. Connection to Existing Findings

| Finding | Connection |
|---------|-----------|
| #3 (LoRA orthogonality structural) | Orthogonality prevents destructive interference but allows omission interference. Sheaf theory addresses the omission case. |
| #38 (orthogonality ≠ specialization) | Orthogonal in weight space, coupled in knowledge space. Sheaf overlaps capture this. |
| #68 (weight orth ≠ data orth) | Directly motivates the knowledge sheaf: weight-space geometry ≠ data-space topology. |
| #94 (scaffold replacement killed) | The residual R is massive and structural. Persistence analysis can decompose R into important vs noise. |
| #208 (format not knowledge) | Highways are format, footpaths are knowledge. Persistence can distinguish them by structural role, not magnitude. |
| #212 (high scale destroys capability) | Scaling amplifies highways (format) and drowns footpaths (knowledge). Bridge adapters preserve footpaths explicitly. |
| #33 (hypernetwork killed for zero-shot) | Context-dependent bridges (B_S varies with S) are a structured version of hypernetworks with mathematical backing. |

---

## 8. Key References

### Foundational
- Edelsbrunner, Letscher & Zomorodian (2000). Topological persistence and simplification.
- Carlsson (2009). Topology and data. Bulletin of the AMS.
- Curry (2014). Sheaves, cosheaves and applications. arXiv:1303.3255.
- Robinson (2014). Topological Signal Processing. Springer. (sheaves on networks)

### Persistent Homology in Neural Networks
- Rieck et al. (2019). Neural persistence. arXiv:1812.09764. [weight-space filtrations → PH complexity]
- Girrbach et al. (2023). Deep graph persistence. arXiv:2307.10865. [whole-network filtration, fixes variance dominance]
- Brunello et al. (2024). Persistent topological features in LLMs. arXiv:2410.11042. [zigzag persistence across layers]
- Brunello et al. (2023). PH for BERT compression. arXiv:2312.10702. [PH-guided pruning: 58% retention]
- Ballester et al. (2023). TDA for NN survey. arXiv:2312.05840. [comprehensive survey]
- Zheng et al. (2025). Probing neural topology of LLMs. arXiv:2506.01042. [topology outperforms activation probing 130%]
- Su, Liu et al. (2025). TDA beyond PH review. arXiv:2507.19504. [sheaf theory, persistent Laplacians]

### Sheaf Theory in ML
- Gebhart, Hansen & Schrater (2021). Knowledge sheaves. arXiv:2110.03789. [KG embedding as sheaf sections]
- Bodnar et al. (2022). Neural sheaf diffusion. arXiv:2202.04579. [foundational sheaf NN]
- Ayzenberg & Magai (2025). Sheaf theory → deep learning. arXiv:2502.15476. [survey]
- Hajij et al. (2025). Copresheaf TNNs. arXiv:2505.21251. [NeurIPS 2025, unifying framework]
- Nguyen et al. (2024). Sheaf hypernetworks for FL. arXiv:2405.20882. [sheaf-structured hypernetworks]
- (2025). Sheaf for federated multi-task. arXiv:2502.01145. [sheaf Laplacian regularization]

### Bridge / Interaction Terms
- (2025). TuckA: Tucker tensor experts. arXiv:2511.06859. [core tensor captures interactions]
- (2026). CRAFT: cross-layer Tucker. arXiv:2602.17510. [frozen HOSVD + lightweight transforms]
- (2024). PID for knowledge distillation. arXiv:2411.07483. [synergistic information = our lost bridges]
- Anthropic (2024). Crosscoders. transformer-circuits.pub/2024. [cross-layer shared features]
- (2025). TC-LoRA: tensorized clustered merging. arXiv:2508.03999. [CP decomposition, shared vs task-specific]

### Composition Limits (the disease)
- (2025). Pause recycling LoRAs. arXiv:2506.13479. [compositional reasoning fundamentally limited]
- Back et al. (2026). LoRA as knowledge memory. arXiv:2603.01097. [misrouting + interference bottlenecks]
- (2025). Forgetting task-specific knowledge in merging. arXiv:2507.23311. [unshared knowledge degrades]
- Biderman et al. (2024). LoRA learns less forgets less. arXiv:2405.09673. [10-100x rank gap]

---

## 9. Open Questions for Experimentation

1. **Empirical**: What does the persistence diagram of a real BitNet-2B pathway graph
   look like? How many high-persistence features exist?
2. **Scaling**: Does $\dim H^1$ grow with N (number of adapters)? If sub-linearly,
   bridge adapters scale well.
3. **Sparsity**: What fraction of pairwise bridge terms $C_{ij}$ are non-trivial?
   Finding #31 killed pair-level caching at 3.9% hit rate — does the sheaf predict this?
4. **Distillation signal**: Can persistence-weighted distillation (teacher penalizes
   student more for losing high-persistence pathways) outperform standard KD?
5. **Fluid adapters**: Can bridge terms be parameterized as a function of the active
   adapter set (a learned $B_S$ rather than pre-computed), and does the sheaf cohomology
   bound still hold?
