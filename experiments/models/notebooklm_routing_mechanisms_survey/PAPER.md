# Routing Mechanisms for Composable Ternary Experts: Research Survey

## Hypothesis

A systematic comparison of 13 published routing mechanisms against our proven BitNet-SOLE architecture (Gumbel-sigmoid + entropy gating + Grassmannian skeleton) will identify at least 3 concrete, implementable improvements for scaling from N=25 to N=100+ experts on Apple Silicon.

## What This Survey Covers

We analyze routing mechanisms from 13 papers across four axes:
1. Mathematical routing function and complexity
2. Failure mode prevention (collapse, imbalance, overhead)
3. Apple Silicon / Metal GPU compatibility
4. Relevance to our specific architecture constraints

### Our Proven Baseline

| Mechanism | Status | Result |
|-----------|--------|--------|
| Gumbel-sigmoid routing | PROVEN | 44% better than softmax, 0.58% overhead, diversity 2.42 |
| Softmax router (N=24) | PROVEN | Matches oracle quality (0.0% gap), 0% fallback, 330K params |
| Entropy gating pre-filter | PROVEN | 63% tokens skip at 1.13% PPL cost |
| Tiny routing heads | PROVEN | 100% accuracy on 5 domains, 2.32% overhead |
| Per-sequence routing | PROVEN | Correct granularity (per-layer KILLED: -18.3%) |
| Speculative expert selection | KILLED | Non-problem: router is 0.46% of inference |

### Key Constraints

- Platform: Apple M5 Pro, 48GB unified memory (Metal GPU)
- Architecture: Ternary base + ternary LoRA adapters + frozen orthogonal A matrices
- Serving: Runtime LoRA (h @ A_i @ B_i), pre-merge for always-on adapters
- Scale target: N=25 current, N=100+ goal
- Memory budget: 853 max adapters at 45.2 MB/adapter (proven)

## Key References

| Paper | Arxiv | Key Contribution |
|-------|-------|------------------|
| Switch Transformer | 2101.03961 | Top-1 routing + load balance loss |
| Hash Layers | 2106.04426 | Zero-parameter deterministic routing |
| PHATGOOSE | 2402.05859 | Post-hoc per-token per-layer gating |
| X-LoRA | 2402.07148 | Per-layer token-level LoRA mixing |
| MixLoRA | 2404.15159 | Top-k router in FFN, 30% latency reduction |
| SpectR | 2504.03454 | Training-free spectral routing |
| CLONE | 2506.02847 | Hardware-algorithm co-design for edge |
| LD-MoLE | 2509.25684 | Learnable dynamic expert count |
| L2R | 2601.21349 | SIPS: Lipschitz-controlled low-rank routing |
| Grassmannian MoE | 2602.17798 | Bingham distribution collapse bounds |
| Task-Aware LoRA | 2602.21222 | Vector DB retrieval for adapter selection |
| CoMoL | 2603.00573 | Core-space merging for memory efficiency |
| MoLoRA | 2603.15965 | Per-token routing, 1.7B beats 8B |

## Systematic Comparison

### Routing Function Taxonomy

| Mechanism | Gate Type | Granularity | Collapse Prevention | Metal-Friendly |
|-----------|-----------|-------------|---------------------|----------------|
| Switch Transformer | top-1 softmax | per-token | Load balance loss | YES |
| Hash Layers | deterministic hash | per-token | Perfect balance by construction | YES |
| PHATGOOSE | sigmoid per-layer | per-token per-layer | None specified | NO (per-layer) |
| X-LoRA | dense per-layer | per-token per-layer | None specified | NO (per-layer) |
| MixLoRA | top-k softmax | per-token | Auxiliary load balance | YES |
| SpectR | SVD spectral | per-token per-layer | None (training-free) | NO (SVD) |
| CLONE | HW-algorithm co-design | not specified | Not specified | NO (custom HW) |
| LD-MoLE | differentiable threshold | per-token per-layer | Analytical sparsity | MODERATE |
| L2R (SIPS) | clamped inner product | per-token | Lipschitz control | YES |
| Grassmannian MoE | Bingham distribution | per-token | Exponential collapse bound | MODERATE |
| Task-Aware LoRA | vector DB retrieval | per-sequence | None | NO (k-NN) |
| CoMoL | soft merge in r x r | per-token | Not specified | YES |
| MoLoRA | learned gating | per-token | Not specified | YES |

### Complexity at N=100, d=2560, r=16

| Mechanism | Routing FLOPs | Router Params | Memory Overhead |
|-----------|---------------|---------------|-----------------|
| Softmax | 256,000 | 256K | O(N) = 400 bytes |
| Gumbel-sigmoid (ours) | 256,000 | 256K | O(N) = 400 bytes |
| SIPS (L2R) r_route=16 | 42,560 | 42.6K | O(r_route + N) = 464 bytes |
| Hash | ~10 | 0 | O(1) |
| CoMoL core | 41,472 | 25.6K cores + 40.9K shared | O(r^2) = 1 KB |
| Grassmannian Bingham | 4,096,000 | 4.1M | O(N*r*d) = 16 MB |

Key insight: SIPS reduces routing from O(d*N) to O(d*r_route + N*r_route), yielding
6x savings at N=100 and 30x at N=500.

## Empirical Results (Prior Findings from Our Experiments)

| Finding ID | Result | Implication for This Survey |
|------------|--------|-----------------------------|
| #27 | Binary routing heads collapse at N>10 | Softmax/Gumbel superior, per-layer harmful |
| #28 | Softmax matches oracle at N=24 | Current routing is GOOD but untested at N>25 |
| #29 | Depth routing degrades -18.3% | Per-layer routing is DEAD for our architecture |
| #32 | Speculative selection is non-problem | Router overhead (0.46%) already negligible |
| #58 | Per-token top-2 beats uniform by 13.9% | Routing provides genuine value vs uniform |
| #72 | N=50 Gumbel routing works (86.33% acc) | Gumbel-sigmoid scales to at least N=50 |
| #115 | Content-aware routing killed at micro | Domain accuracy 26.5% < 60% threshold |
| #116 | Cluster-level routing 96% accuracy | Hierarchical (cluster first) is viable |
| #118 | Routing moot without specialization | Expert quality matters more than routing |
| #144 | Inference routing strategies killed | Quality capture 41.5% < 90% threshold |
| #145 | Routing latency solved at N=1000 | 20.7us at N=1000, not a bottleneck |
| #168 | MoE scaling laws don't apply to LoRA | Our LoRA IS output-space MoE |

## What Doesn't Work For Us (Eliminated Mechanisms)

### CLONE (arxiv 2506.02847) — ELIMINATED
CLONE's 11.92x speedup relies on a custom 28nm hardware accelerator. Its system-level
optimizations (algorithm-hardware co-design) are silicon-specific and cannot be ported
to Apple Silicon Metal. The algorithmic routing insights are not separable from the
hardware design. **Verdict: completely irrelevant for our deployment target.**

### X-LoRA (arxiv 2402.07148) — ELIMINATED
X-LoRA uses "deep layer-wise token-level gating" — exactly the per-layer routing we
proved harmful (-18.3% degradation, Finding #29). X-LoRA recomputes mixing weights at
every layer using hidden states, creating L sequential gate evaluations. On Metal, this
prevents layer-level parallelism. **Verdict: incompatible with our per-sequence finding.**

### PHATGOOSE (arxiv 2402.05859) — ELIMINATED
Same per-layer routing architecture as X-LoRA. Post-hoc nature is useful (no joint
training needed) but we already have post-hoc routing (Gumbel-sigmoid trained separately).
**Verdict: per-layer routing is harmful for our adapters.**

### SpectR (arxiv 2504.03454) — ELIMINATED
Requires per-token per-layer SVD of expert weight matrices at runtime. SVD is iterative
and not Metal-friendly. Also per-layer, which we've killed. Training-free is nice but
our routing head training is already cheap (330K params). **Verdict: per-layer + SVD = double kill.**

### Task-Aware LoRA Composition (arxiv 2602.21222) — LOW PRIORITY
Vector DB retrieval (k-NN search) at inference creates irregular memory access patterns
hostile to Metal. However, the per-SEQUENCE granularity matches our architecture. Could
work as an offline adapter selection mechanism for deployment configuration, not runtime
routing. **Verdict: useful for adapter catalog management, not runtime routing.**

## What Works For Us: Ranked Recommendations

### R1: SIPS + Gumbel-Sigmoid Hybrid (L2R, arxiv 2601.21349)

**Priority: HIGHEST. Addresses N>25 scaling bottleneck.**

**What:** Add Saturated Inner-Product Scoring to our existing Gumbel-sigmoid router.
Replace g_i(x) = sigma(w_i^T x + eps_i) with g_i(x) = sigma(clamp(w_i^T W_proj x, -C, C) + eps_i).

**Why:** Angular concentration in high-dimensional routing spaces degrades discriminability
as N grows. At N=100 with d_route=64, max pairwise cosine >= 0.075 (Welch bound). SIPS's
Lipschitz control prevents score explosion and concentration, maintaining stable routing
geometry at high N.

**Implementation on MLX:**
```python
# Current Gumbel-sigmoid (proven, N<=50)
z = mx.matmul(h, W_router)  # [batch, N]
g = mx.sigmoid((z + gumbel_noise) / tau)

# SIPS-enhanced (proposed, N=100+)
z_proj = mx.matmul(h, W_proj)  # [batch, r_route]  -- low-rank projection
scores = mx.matmul(z_proj, W_experts.T)  # [batch, N]  -- score in latent space
scores = mx.clip(scores, -C, C)  # SIPS saturation
g = mx.sigmoid((scores + gumbel_noise) / tau)
```

**Complexity:** O(d * r_route + N * r_route) vs current O(d * N). At N=100, r_route=16: 6x cheaper.
**Metal compatibility:** Two dense matmuls + elementwise clamp + sigmoid. Fully Metal-friendly.
**Router params:** d * r_route + N * r_route = 2560*16 + 100*16 = 42.6K (vs current 256K at N=100).
**Risk:** C hyperparameter needs tuning. Too small = information loss, too large = no benefit.
**Kill criterion:** If routing accuracy at N=100 is not within 5% of oracle, SIPS is insufficient.

### R2: Hierarchical Cluster-then-Route (Grassmannian MoE, arxiv 2602.17798 + Finding #116)

**Priority: HIGH. Proven cluster accuracy (96%) + mathematical collapse guarantees.**

**What:** Two-stage routing: (1) assign token to one of C clusters using cheap cosine similarity,
(2) route within cluster using Gumbel-sigmoid over N/C experts. Grassmannian MoE's Bingham
distribution provides mathematical guarantees against collapse at each level.

**Why:** Cluster-level routing is already proven at 96% accuracy (Finding #116). By reducing
the per-token routing problem from N=100 to N_cluster=10-20, we avoid angular concentration
entirely. The Bingham distribution provides P(collapse) <= exp(-c * min_gap(Lambda)), making
collapse exponentially unlikely.

**Implementation sketch:**
1. Precompute C cluster centroids from expert A_i matrices (offline, on Grassmannian)
2. At runtime: cluster_id = argmax_c cos(h, centroid_c) — O(d * C) FLOPs
3. Within cluster: apply SIPS + Gumbel-sigmoid over N/C experts — O(d * r_route + (N/C) * r_route)

**Complexity:** O(d * C + d * r_route + (N/C) * r_route). At C=10, N=100: O(25,600 + 40,960 + 160) = 66,720.
Compare flat routing: O(d * N) = 256,000. Hierarchical is 3.8x cheaper.
**Metal compatibility:** Two matmuls (cluster scores + within-cluster scores). Fully Metal-friendly.
**Risk:** Cluster boundaries may not align with semantic boundaries for mixed-domain inputs.
**Kill criterion:** If hierarchical routing accuracy < flat routing accuracy by >3%, clusters are wrong.

### R3: LD-MoLE Dynamic Expert Count (arxiv 2509.25684) as Entropy Gating Upgrade

**Priority: MEDIUM. Upgrades binary entropy gating to continuous expert budget.**

**What:** Replace our binary entropy gating (skip all experts or use top-k) with LD-MoLE's
differentiable dynamic-k mechanism. Instead of a fixed Otsu threshold, learn a per-sequence
function that outputs the optimal number of experts k*(x) in [0, N].

**Why:** Our current entropy gating is all-or-nothing: 63% of tokens use 0 experts, 37% use
all top-k. This leaves quality on the table for tokens that need 1 expert but get k=2, or
tokens that are borderline but get 0. LD-MoLE's analytical sparsity control learns the
optimal k per input.

**Critical adaptation:** LD-MoLE is designed per-layer, which we've killed. We MUST adapt it
to per-SEQUENCE granularity: learn k*(sequence) not k*(token, layer).

**Implementation approach:**
1. Train a lightweight network: k_pred = round(sigmoid(MLP(h_cls)) * k_max)
2. Use Gumbel-softmax trick to make k differentiable during training
3. Analytical sparsity objective: L_sparsity = lambda * ||k_pred - k_target||^2
   where k_target is the expected sparsity level

**Complexity:** One additional forward pass through small MLP: O(d * d_hidden + d_hidden * 1).
With d_hidden=64: O(2560*64 + 64) = 163,904 FLOPs. Comparable to routing itself.
**Metal compatibility:** Dense MLP + sigmoid + round. Fully Metal-friendly.
**Risk:** Requires retraining entropy gate. Current Otsu method needs zero training.
**Kill criterion:** If dynamic-k does not improve PPL by >2% over binary entropy gating at same
skip rate, the added complexity is not justified.

### R4: CoMoL-Inspired Core-Space Routing for N>50 (arxiv 2603.00573)

**Priority: MEDIUM-LOW. Memory scaling solution, but requires architectural change.**

**What:** For each expert, decompose B_i = B_shared @ C_i where C_i in R^{r x r}. Route in the
compact r x r core space instead of d x r weight space.

**Why:** At N>50, even ternary adapters consume significant runtime buffer memory. Core-space
routing reduces per-expert overhead from O(d * r) to O(r^2), a 160x reduction at d=2560, r=16.

**Incompatibility note:** Our architecture uses distinct orthogonal A_i (Grassmannian skeleton).
CoMoL assumes shared A. A hybrid approach: group experts into C clusters sharing A_cluster,
with core-space routing within each cluster. This preserves inter-cluster orthogonality while
gaining intra-cluster memory efficiency.

**Implementation would require:**
1. Cluster A_i matrices by Grassmannian geodesic distance
2. Within each cluster, SVD-decompose B_i relative to shared A_cluster
3. Route in core space C_i during inference

**Complexity:** Per-cluster: O(d * r + k * r^2) instead of O(k * d * r). At k=2, d=2560, r=16:
O(40,960 + 512) vs O(81,920). 2x savings per cluster.
**Metal compatibility:** Dense matmuls in small space. YES.
**Risk:** Forcing shared A within clusters may degrade the Grassmannian orthogonality guarantee.
**Kill criterion:** If within-cluster interference (|cos(deltaW_i, deltaW_j)|) exceeds 0.05.

### R5: MoLoRA Per-Token for Mixed-Domain Sequences (arxiv 2603.15965)

**Priority: LOW. Only relevant when we encounter mixed-domain inputs.**

**What:** Per-token routing allows different tokens in the same sequence to use different experts.
MoLoRA shows 1.7B model with 4 adapters beating 8B on mixed benchmarks.

**Why our current finding says "low priority":** On clean single-domain text, per-token routing
is equivalent to per-sequence (-0.46% difference, Finding from exp_molora_per_token_mlx).
But for mixed-domain sequences (e.g., a legal document with code snippets), per-token routing
could activate the code expert for code tokens and legal expert for legal tokens.

**When to revisit:** When we have mixed-domain evaluation data showing per-sequence routing
fails on heterogeneous inputs.
**Kill criterion:** If per-token improves <1% over per-sequence on mixed-domain eval.

## Limitations

1. **Abstract-level analysis only.** Most papers were analyzed from arxiv abstracts, not full
   implementations. Exact parameter counts, convergence properties, and failure modes may differ
   from what abstracts report.

2. **No empirical validation in this survey.** Recommendations are based on mathematical analysis
   and compatibility assessment, not experimental verification on our specific architecture.

3. **Scaling projections are theoretical.** The angular concentration bounds and complexity
   reductions at N=100+ are derived from theory; actual behavior may differ due to data
   distribution, adapter quality variation, and Metal-specific performance characteristics.

4. **CoMoL adaptation is speculative.** The hybrid clustered-CoMoL approach (R4) has not been
   published or tested. It is our novel proposal based on combining CoMoL's core-space idea
   with our Grassmannian skeleton.

5. **LD-MoLE per-sequence adaptation untested.** LD-MoLE is designed per-layer; our proposed
   per-sequence adaptation (R3) changes its core operating granularity.

## What Would Kill This Survey's Recommendations

- **R1 killed if:** Gumbel-sigmoid alone maintains oracle-matching quality at N=100 without SIPS.
  This would mean angular concentration is not a real problem at our scale.

- **R2 killed if:** Cluster-level routing accuracy degrades below 90% on real data with
  overlapping domains (e.g., science-medical overlap noted in Finding #53).

- **R3 killed if:** Binary entropy gating (0 or k experts) is already optimal — i.e., the
  distribution of "how many experts does this token need" is bimodal, not continuous.

- **R4 killed if:** Grassmannian orthogonality cannot be preserved within clusters (shared A
  forces cos > 0.05 within cluster).

- **R5 killed if:** Mixed-domain sequences are rare in deployment (<5% of queries) or per-sequence
  routing handles them adequately.

## Summary of Actionable Recommendations

| Rank | Recommendation | Paper | Key Metric | Implementation Effort |
|------|---------------|-------|------------|----------------------|
| R1 | SIPS + Gumbel-sigmoid hybrid | L2R (2601.21349) | 6x routing FLOPs reduction at N=100 | LOW: add W_proj + clamp to existing router |
| R2 | Hierarchical cluster-then-route | Grassmannian MoE (2602.17798) | 3.8x routing FLOPs + collapse bound | MEDIUM: cluster A_i offline, 2-stage routing |
| R3 | Dynamic expert count (LD-MoLE adapted) | LD-MoLE (2509.25684) | Upgrade binary to continuous gating | MEDIUM: retrain gating network |
| R4 | Core-space routing within clusters | CoMoL (2603.00573) | 160x per-expert overhead reduction | HIGH: requires B_i factorization |
| R5 | Per-token for mixed domains | MoLoRA (2603.15965) | Mixed-domain quality | LOW: already implemented, needs eval data |

All 5 recommendations cite arxiv papers. R1-R3 are highest-confidence implementable improvements
with clear kill criteria and Metal-compatible implementations.
