# Adapter Pruning Lifecycle: Mathematical Foundations

## 1. Mechanism Definition

### Problem Setting

Given a pool of N = 24 domain adapters, each parameterized as:

$$\Delta W_i = B_i A_i^T, \quad A_i \in \mathbb{R}^{d \times r}, \; B_i \in \mathbb{R}^{r \times d_{out}}$$

where $A_i$ are Grassmannian-initialized (frozen) and $B_i$ are trained with STE ternary
quantization. The question: can we remove $k$ adapters from the pool with minimal quality
degradation?

### Pruning Metric 1: Leave-One-Out PPL Delta

For adapter $i$, define:

$$\Delta_{\text{LOO}}(i) = \text{PPL}(\mathcal{P} \setminus \{i\}) - \text{PPL}(\mathcal{P})$$

where $\text{PPL}(\mathcal{S})$ is the average composed perplexity across all domains in set $\mathcal{S}$.

Adapters with small $\Delta_{\text{LOO}}(i)$ are safely removable: removing them barely affects
composed quality.

**Complexity:** $O(N)$ full model evaluations (each across $N-1$ domains). At 24 domains with
25 val batches each, this is $24 \times 23 \times 25 = 13,800$ forward passes. With ~0.15s
per forward pass on M5 Pro: ~35 minutes. Feasible but dominant cost.

**Approximation:** We use per-domain oracle PPL (each adapter evaluated on its own domain) rather
than full routed composition for LOO, since we proved in exp_softmax_router_scaling that softmax
routing matches oracle quality (0% gap). This reduces cost to $24 \times 25 = 600$ forward passes
(~1.5 min per domain load cycle).

### Pruning Metric 2: Routing Frequency

Given a trained softmax router $r: \mathbb{R}^d \to \Delta^{N-1}$ and a held-out evaluation set
$\mathcal{D} = \{x_1, \ldots, x_M\}$ drawn uniformly from all domains:

$$f_i = \frac{1}{M} \sum_{j=1}^{M} \mathbb{1}[\arg\max_k r_k(x_j) = i]$$

This is the fraction of tokens/sequences for which adapter $i$ is selected as top-1.

Under perfect classification with balanced evaluation data: $f_i = 1/N$ for all $i$.

Deviation below $1/N$ indicates the domain is either:
(a) underrepresented in evaluation data, or
(b) semantically subsumed by a neighboring domain (router sends its queries elsewhere).

Case (b) is exactly what makes an adapter prunable: if the router never selects it, removing it
has zero impact on routed composition quality.

**Complexity:** Single forward pass through router for all val samples. $O(M \cdot N)$ where
$M \approx 1200$ (50 per domain). Trivial cost (~seconds).

### Pruning Metric 3: Effective Delta Magnitude

The effective weight perturbation from adapter $i$ at layer $l$, module $m$:

$$\|\Delta W_i^{(l,m)}\|_F = \|B_i^{(l,m)} A_i^{(l,m)T}\|_F$$

Aggregated across all layers and modules:

$$\delta_i = \sqrt{\sum_{l,m} \|B_i^{(l,m)} A_i^{(l,m)T}\|_F^2}$$

Adapters with small $\delta_i$ have learned minimal perturbation from the base model. If
$\delta_i \approx 0$, the adapter is effectively a no-op and safely removable.

**Approximation:** Since $A_i$ is frozen with orthonormal columns (Grassmannian init),
$\|B_i A_i^T\|_F = \|B_i\|_F$ (orthonormal $A$ preserves Frobenius norm). So:

$$\delta_i = \sqrt{\sum_{l,m} \|B_i^{(l,m)}\|_F^2}$$

This avoids materializing the full $d \times d_{out}$ delta matrix.

**Proof:** $\|BA^T\|_F^2 = \text{tr}(AB^TBA^T) = \text{tr}(B^TBA^TA)$. Since $A^TA = I_r$
(orthonormal), $= \text{tr}(B^TB) = \|B\|_F^2$.

**Complexity:** Load each adapter, sum squared B-matrix entries. $O(N \cdot L \cdot M_{mod} \cdot r \cdot d_{out})$.
Trivial cost.

### Pruning Metric 4: Cross-Adapter Similarity

**Effective delta cosine** (what we'd ideally compute):

$$\text{cos}_{\text{eff}}(i, j) = \frac{\sum_{l,m} \text{tr}(B_i^{(l,m)T} B_j^{(l,m)} \cdot A_i^{(l,m)T} A_j^{(l,m)})}{\delta_i \cdot \delta_j}$$

Since the Grassmannian skeleton ensures $A_i^T A_j \approx 0$ for $i \neq j$, the
effective delta cosine is near-zero by construction. This is the 17x decorrelation filter.

**B-matrix cosine** (what we actually compute):

$$\text{cos}_B(i, j) = \frac{\text{vec}(B_i)^T \text{vec}(B_j)}{\|B_i\|_F \cdot \|B_j\|_F}$$

where $\text{vec}(B_i)$ concatenates all B-matrices across layers and modules.

We intentionally compute B-matrix cosine rather than effective delta cosine because:
1. Effective delta cosine is near-zero by Grassmannian design — it tests AP convergence, not
   adapter similarity.
2. B-matrix cosine asks: did two adapters learn similar perturbation patterns despite being
   projected into orthogonal subspaces? This reveals functional redundancy that the A-matrix
   orthogonality might mask.

If $\text{cos}_B(i,j)$ is high, the adapters learned similar B-matrices. One is redundant
(the Grassmannian ensures they don't interfere, but they provide similar value).
Remove the one with lower $\delta$ (smaller effective perturbation).

**Complexity:** $O(N^2)$ pairwise comparisons, each requiring loading two adapters and computing
inner products across all layers/modules. $C(24,2) = 276$ pairs, each ~milliseconds. Total: seconds.

## 2. Why It Works

### Redundancy in Domain-Specialized Pools

With 24 diverse domains, some are semantically adjacent:
- philosophy/history (humanistic reasoning)
- economics/finance (quantitative social science)
- health_fitness/medical (health domain)
- politics/sociology (social systems)

If two domains share substantial overlap, their adapters learn similar perturbations. The
Grassmannian A-matrices force them into orthogonal subspaces, but the B-matrices may converge
to capture the same underlying knowledge pattern. This manifests as:
- Similar routing frequency (router sends related queries to either)
- Similar PPL improvement on each other's data
- Low LOO delta (removing one barely hurts because the other covers its queries)

### Connection to MoE Expert Pruning

Unchosen Experts (arXiv 2402.05858) showed that in production MoE models:
- Expert utilization follows a long-tail distribution
- Some experts are rarely selected but still matter for tail queries
- Pruning based purely on frequency can miss important rare-case experts

This motivates using multiple complementary metrics rather than frequency alone.

### LoRA-Hub Evidence

LoRA-Hub (Huang et al., 2023, arXiv 2307.13269) demonstrated that for any given task, only a
small subset of LoRA adapters contribute positively. Their gradient-free optimization over
composition weights often zeros out most adapters. This suggests our pool likely contains
adapters irrelevant to most queries.

## 3. What Breaks It

### K1 Failure: All Adapters Equally Important

If $\Delta_{\text{LOO}}(i) > 0.05 \cdot \text{PPL}(\mathcal{P})$ for ALL $i$, then every adapter
is load-bearing and pruning is unsafe. This would mean:
- No domain redundancy (all 24 domains are maximally distinct)
- No semantic overlap between neighboring domains
- The pool is already minimal

**When this happens:** When domains are chosen to be maximally diverse with no overlap (e.g.,
code vs. cooking vs. music). Our 24-domain pool was chosen for breadth, so some overlap is
expected.

### K2 Failure: All Metrics Agree

If all four pruning strategies rank adapters identically for removal, then the metrics are
measuring the same underlying signal. This would mean:
- Routing frequency perfectly predicts LOO delta
- Delta magnitude perfectly predicts routing frequency
- Cross-adapter similarity adds no information

This is unlikely because the metrics capture different aspects:
- LOO: end-to-end quality impact (most comprehensive but expensive)
- Routing: demand-side (what queries need)
- Delta magnitude: supply-side (what adapters provide)
- Similarity: redundancy structure (pairwise relationships)

### S1 Failure: Cannot Prune 20%

If removing any 5 adapters causes >2% quality loss, this means the pool is denser than expected.
Possible causes:
- Grassmannian orthogonality prevents functional redundancy even when domains overlap semantically
- Each adapter captures unique non-overlapping features despite domain similarity

## 4. Assumptions

1. **Oracle PPL approximates routed PPL** — Justified by exp_softmax_router_scaling finding
   that softmax router matches oracle (0% gap at N=24). If this assumption fails, LOO metrics
   are unreliable.

2. **Uniform composition is representative** — We evaluate quality under uniform (1/N) composition.
   If routing dramatically changes the quality landscape, pruning decisions under uniform may
   not transfer to routed serving.

3. **Per-adapter evaluation suffices** — We measure each adapter on its own domain rather than
   cross-domain. If an adapter's primary value is cross-domain transfer (e.g., math adapter
   improving code quality), LOO on own-domain misses this.

4. **Grassmannian orthogonality holds** — The skeleton guarantees $A_i^T A_j \approx 0$.
   If AP convergence was incomplete for some pairs, cross-adapter similarity may be non-trivial.
   Prior experiments confirmed mean $|\cos| = 0.0238$ at N=24.

## 5. Complexity Analysis

| Metric | Forward Passes | Time (est.) | Memory |
|--------|---------------|-------------|---------|
| Base PPL (24 domains) | 600 | ~3 min | 1 model load |
| LOO PPL (24 leave-outs) | 24 x 575 = 13,800 | ~35 min | 1 model load per LOO |
| Routing frequency | ~1,200 | seconds | Router only |
| Delta magnitude | 0 (just load weights) | seconds | 1 adapter at a time |
| Cross-similarity | 0 (just load weights) | seconds | 2 adapters at a time |
| Pruned evaluation | $S \times 24 \times 25$ | ~5 min per strategy | 1 model load per eval |

**Optimization:** LOO is the bottleneck. We can reduce cost by:
- Computing per-adapter oracle PPL once (24 model loads)
- For LOO(i), approximate composed PPL as average of remaining 23 oracle PPLs
- This avoids reloading the model 24 times with N-1 adapters each time

**Total estimated runtime:** 30-40 minutes with the per-adapter oracle approach.

## 6. Worked Example (Micro Scale)

Consider N=4 adapters: {medical, code, math, finance}

**Oracle PPLs:** medical=3.2, code=4.1, math=3.8, finance=4.5

**Full-pool average PPL:** (3.2 + 4.1 + 3.8 + 4.5) / 4 = 3.90

**LOO deltas:**
- Remove medical: avg(4.1, 3.8, 4.5) = 4.13, delta = +0.23 (+5.9%)
- Remove code: avg(3.2, 3.8, 4.5) = 3.83, delta = -0.07 (-1.8%)
- Remove math: avg(3.2, 4.1, 4.5) = 3.93, delta = +0.03 (+0.8%)
- Remove finance: avg(3.2, 4.1, 3.8) = 3.70, delta = -0.20 (-5.1%)

Pruning order (smallest absolute delta first): finance, code, math, medical.

But wait: removing finance actually IMPROVES average PPL (negative delta). This can happen when
an adapter adds noise to composition. These are prime pruning candidates.

**Delta magnitudes:** medical=12.3, code=8.7, math=15.1, finance=3.2

finance has smallest delta magnitude AND negative LOO delta. Both metrics agree: it is the
weakest adapter. If routing frequency also shows finance as least-selected, all three metrics
converge, triggering K2 concern. But if routing shows code as least-selected (disagreement),
the metrics capture genuinely different signals.

## 7. Connection to Architecture

### Pruning and the Evolve Track

This experiment directly addresses the "Evolve" track (10% readiness). As the adapter pool
grows, lifecycle management becomes essential:
- **Which adapters to keep?** (this experiment: pruning strategies)
- **When to retrain?** (future: quality monitoring over time)
- **How to replace?** (future: retrain-from-scratch with quality gate, per exp_bitnet_clone_compete)

The Grassmannian skeleton has capacity $N_{\max} = d^2/r^2 = 2560^2/16^2 = 25,600$ adapters.
At 853 adapters fitting in 48GB, memory is the binding constraint, not skeleton capacity.
Pruning extends the effective memory budget by removing low-value adapters.

### Pruning and Softmax Routing

The softmax router (exp_softmax_router_scaling) provides a natural pruning signal: routing
frequency. A key finding was that 40% classification accuracy still matched oracle PPL,
because within-cluster misrouting is quality-benign. This implies that if the router rarely
selects an adapter, the adapter's domain is either:
1. Semantically covered by a close neighbor (prunable), or
2. Genuinely rare in the evaluation distribution (keep for tail coverage)

Distinguishing (1) from (2) requires cross-domain PPL evaluation, which LOO provides.

### Pruning and Production Serving

From exp_memory_budget_analysis: per-adapter cost is 45.2 MB. Pruning 5 adapters saves 226 MB.
At N=24, total adapter memory is ~1.08 GB. Pruning 20% saves ~216 MB.

More importantly, pruning reduces routing complexity (fewer classes in softmax) and composition
cost (fewer expert terms to sum in forward pass). At N=19 vs N=24, per-token composition cost
drops by ~20% (linear in N for the LoRA sum loop).
