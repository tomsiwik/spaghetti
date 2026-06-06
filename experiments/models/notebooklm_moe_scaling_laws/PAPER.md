# MoE Scaling Laws for LoRA Composition: Research Brief

## Hypothesis

Traditional MoE scaling laws (interpreted as showing MoE becomes suboptimal
below ~500M params) do NOT apply to LoRA-based expert composition because LoRA
experts are fundamentally different from FFN experts in parameter efficiency,
interference properties, and minimum viable scale.

## What This Research Is

A literature review and mathematical analysis answering three questions:
1. Are there scaling laws for LoRA composition specifically?
2. At what scale does sparse routing beat dense composition?
3. What is the optimal expert count for our parameter budget (2B ternary base)?

## Key References

| Paper | ArXiv | Key Finding |
|-------|-------|-------------|
| Apple MoE Scaling | 2501.12370 | L(N,D,S) formula; optimal sparsity -> 0 at small N |
| Joint MoE Scaling | 2502.05171 | MoE needs 2-4x more data than dense; 32-expert beats 1.7B dense at 10^20 FLOPs |
| MoLoRA | 2603.15965 | Per-token LoRA routing: 1.7B+adapters > 8B on reasoning |
| LD-MoLE | 2509.25684 | Dynamic differentiable routing > static TopK |
| CoMoL | 2603.00573 | Core-space merging: MoE routing with 1/N per-expert cost |
| Naive LoRA Summation | 2508.11985 | Orthogonal adapters compose at 117M (GPT-2); RMS cosine predicts interference |
| LoRI | 2504.07448 | Frozen random A + sparse B eliminates cross-task interference |
| OSRM | 2505.22934 | Pre-training orthogonal constraints reduce merging interference |
| LoRA Soups | 2410.13025 | CAT composition: k=2 skills super-linear, k>=3 loses to data mixing |
| CLONE | 2506.02847 | Edge MoE: algorithm-hardware co-design, 11.92x acceleration |
| X-LoRA | 2402.07148 | Dynamic layer-wise token-level LoRA scaling via learned head |
| Sparse-BitNet | 2603.05168 | Ternary models tolerate higher N:M sparsity than FP16 |

## Empirical Results

### Finding 1: MoE Scaling Laws Do NOT Apply to LoRA Composition

The Apple scaling law L(N,D,S) was derived for traditional MoE with full FFN
experts (O(d^2) params each). At our scale:
- FFN expert (full up+down projection): 35.4M params/layer (53.1M with SwiGLU)
- LoRA expert (r=16): 81.9K params/layer
- **Ratio: 432:1 to 648:1**

Our interpretation of Apple's scaling law suggests full FFN experts need minimum
scale (~500M) to learn useful transformations. LoRA experts don't need to learn
complete transformations -- they learn rank-16 perturbations to an existing
transformation. This works at ANY scale where the base model is competent.

**Evidence:** Naive LoRA Summation works at 117M params (GPT-2 Small) with
rank-4 adapters (Cao et al., 2508.11985). Our 2B base is 17x above this minimum.

### Finding 2: Runtime LoRA IS Output-Space MoE

Our serving architecture computes:
```
h = W*x + sum_{i in topK} g_i * B_i * A_i * x
```

This is mathematically identical to MoE output-space composition where each
"expert" is the low-rank function f_i(x) = B_i * A_i * x. Key implications:
- **No cross-terms within each layer** (each adapter acts independently on x)
- **No parameter-space interference** (unlike pre-merge)
- Cross-layer interference through nonlinearities is empirically small (gamma=0.982)
- MoE properties (routing, load balancing, expert specialization) all apply
- But scaling law coefficients need empirical refitting for low-rank experts

This means: our architecture already IS an MoE system. The "should we use MoE
or LoRA composition?" question is a false dichotomy -- we're doing both.

### Finding 3: Sparse Routing Beats Dense at N >= 3

From our experiments and the literature:

| N (experts) | Best Strategy | Evidence |
|-------------|--------------|----------|
| 1 | Single best adapter | Trivially optimal |
| 2 | Weighted merge (CAT) | LoRA Soups: super-linear gains at k=2 |
| 3-5 | Top-2 routing | Our data: 1/N scaling resolves PPL explosion |
| 5-25 | Softmax top-2 routing | Oracle-matching at N=24 (0% quality gap) |
| 25+ | Softmax top-2 + pruning | LOO removes 20% with <1% impact |

**The crossover is at N=3.** At N=2, parameter merging gives super-linear gains
(LoRA Soups CAT). At N>=3, routing dominates because cross-term interference
grows quadratically with N while routing overhead is constant.

### Finding 4: Optimal Expert Count = 8-25 for 2B Base

Based on converging evidence:

**Lower bound (N >= 8):**
- MoLoRA validates LoRA-MoE at similar scale (Qwen3-1.7B)
- Mixtral proves 8-expert MoE works at production scale
- Our softmax router handles N=24 at oracle quality

**Upper bound (N <= 25-50):**
- Near-linear scaling: gamma = 0.982 at N=25, meaning composed PPL is 98.2%
  of base (1.8% improvement), all 25 domains benefit
- Memory: 2.26 GB at N=25, 2.79 GB at N=50 (fits easily in 48GB)
- Adapter quality bounded by base model's per-domain capability
- LOO pruning shows 5/24 (20.8%) are removable at <1% cost

**Routing strategy: top-2 softmax.**
- Matches DeepSeek-V3 and Mixtral's proven top-2 selection
- Our softmax router already matches oracle at N=24
- Per-token routing (MoLoRA) worth exploring for mixed-domain inputs
- LD-MoLE's dynamic activation count is the next frontier

### Finding 5: The Binding Constraint is Base Model Quality

From our Oracle PPL analysis: quality spread across 24 adapters is 6.2x, and
this spread is driven by the base model's per-domain capability. An adapter
cannot fix what the base model doesn't know.

This means:
- Adding more experts has diminishing returns once all "competent domains" are covered
- The priority should be improving the base model, not adding more experts
- For domains where the base is weak, larger rank (r=32) helps more than more experts

## 5 Concrete Recommendations

### R1: Use top-2 routing, not uniform composition
**Citation:** DeepSeek-V3 (2412.19437), Mixtral (2401.04088), our softmax
router experiment (0% gap to oracle at N=24).
**Implementation:** Already have softmax router. Ensure k=2 at inference, not
k=N with 1/N weights. k=2 gives each selected expert 50% influence instead of
1/N = 4% influence (at N=25).

### R2: Target N=8-16 high-quality experts, not N=100
**Citation:** MoLoRA (2603.15965) beats 8B with just 4 adapters. Our N=25
gamma=0.982. LOO prunes 20% of experts.
**Implementation:** Invest in fewer, better-trained adapters on domains where
the base model has strong priors. Prune low-quality adapters via LOO. Use
rank-32 for reasoning domains (math, code), rank-16 for knowledge domains.

### R3: Replace pre-merge with runtime LoRA for routed experts
**Citation:** LoRI (2504.07448) shows output-space composition eliminates
cross-terms. Our result: pre-merge -36% worse for ternary.
**Implementation:** Already using runtime LoRA (addmm) at 97 tok/s. Pre-merge
only for always-on adapters (instruction adapter). This split is confirmed
optimal by our batched pre-merge throughput experiment.

### R4: Explore CoMoL core-space routing for N>25
**Citation:** CoMoL (2603.00573) routes in r x r space (r=16: 256-dim) instead
of d x d, reducing per-expert routing overhead by N_experts/1.
**Implementation:** Decompose each LoRA via SVD: B = U_B * Sigma_B * V_B^T,
A = U_A * Sigma_A * V_A^T. Route only the r x r core matrix M = Sigma_B *
V_B^T * U_A * Sigma_A. Shared U_B, V_A amortized across all experts.
FLOPs: standard MoE-LoRA = 2*L*r*(2d+r)*N, CoMoL = 2*L*r*(2d+r) + L*r^2*(N-1).

### R5: Investigate LD-MoLE dynamic expert count per token
**Citation:** LD-MoLE (2509.25684) on Qwen3-1.7B: differentiable routing
adaptively determines expert count per token.
**Implementation:** Replace fixed top-k with learned threshold: activate
adapter i if g_i(x) > tau, where tau is learned via analytical sparsity
control. Some tokens may use 0 experts (base-only), others may use 3+.
This subsumes our entropy-adaptive gating (which achieves 63% skip rate).

## Limitations

1. **No empirical LoRA scaling law exists.** The Apple MoE scaling law was not
   derived for LoRA experts. Our analysis is mathematical reasoning about why
   the law doesn't transfer, not an empirical validation of LoRA-specific
   scaling behavior. An empirical study sweeping (N, r, d, D) would be needed
   to derive actual LoRA composition scaling coefficients.

2. **All composition results are PPL-based.** Task-specific metrics (accuracy
   on benchmarks) may behave differently from PPL. LoRA Soups found k=2 skill
   composition works for math but k>=3 doesn't -- this is task-dependent.

3. **Router quality has not been stress-tested on truly ambiguous inputs.**
   Our softmax router matches oracle on clean domain-labeled data. Performance
   on mixed-domain or out-of-distribution inputs is unknown.

4. **Base model quality is the binding constraint** and this analysis cannot
   predict how adapter quality scales with base model improvements.

## What Would Kill This

1. **If LoRA composition quality degrades faster than O(1/sqrt(N))** as N
   grows beyond 25, the "many experts" premise fails. This would mean
   interference accumulates despite orthogonality. Currently gamma=0.982
   at N=25 (1.8% improvement over base), which is sustainable.

2. **If per-token routing provides >5% improvement over per-sequence routing
   on mixed-domain data**, then our per-sequence softmax router is inadequate
   and we need MoLoRA-style per-token routing. Current result: per-token is
   equivalent to per-sequence on clean domains (-0.46%).

3. **If the Apple scaling law is empirically confirmed to apply to LoRA experts**
   (not just FFN experts), then our architecture would need much larger active
   LoRA parameters, which is impractical at r=16.
