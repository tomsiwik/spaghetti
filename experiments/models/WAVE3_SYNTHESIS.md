# Wave 3 Micro Synthesis: Cross-Experiment Learnings

_9 experiments, 5 supported, 4 killed. Generated 2026-03-25._

## Executive Summary

Wave 3 resolved all micro-scale unknowns for BitNet-SOLE. The architecture is validated: ternary base + independent LoRA adapters + per-token routing + retrain-from-scratch evolution. The remaining questions are all scale-dependent and require GPU.

## The Five Things That Work

### 1. KR-Test as Quality Gate (exp_bitnet_kr_test_evaluation — SUPPORTED)
- Perfect rank correlation with task accuracy (rho=1.0, n=4 domains)
- Catches degenerate adapters that PPL misses (legal adapter: 4.4x PPL gain, zero KR-Test improvement)
- Cross-item pairing is critical; rule-based perturbation yields zero discrimination
- **Scale gap**: n=50 gives only 12% statistical power; needs n>=500 for rigorous gating

### 2. GaLore Scaffold (exp_bitnet_galore_scaffold — SUPPORTED)
- GaLore produces better FP32 models (0.81x PPL vs Adam) but 2-2.9x worse after ternary PTQ
- Quantization is the bottleneck, not training quality
- **Key insight**: GaLore warmup → STE/QAT transition is the most promising base-free path
- Composition ratio 1.045x — GaLore scaffolds are naturally composition-friendly

### 3. llama.cpp Multi-Adapter Serving (exp_bitnet_llamacpp_serving — SUPPORTED)
- TQ2_0 + 5 runtime LoRA adapters works on CPU
- Affine overhead model: 9.5% + 7.5%×N (memory-bandwidth bound, not compute)
- Hot-swap via API without corruption
- **Scale gap**: 10x higher overhead than FLOP predictions; custom kernels needed for N>5

### 4. Per-Sequence Routing (exp_bitnet_per_token_routing — SUPPORTED)
- Top-2 routing beats uniform 1/N by 13.9% (15 adapters)
- Top-1 fails (-10.9%) due to adapter overshoot — convex combination from top-2 provides regularization
- 659K-param router achieves 91.7% domain accuracy
- **Scale gap**: true per-token routing (MoLoRA) is strictly better but needs N forward passes

### 5. Retrain-from-Scratch Evolve (exp_bitnet_retrain_evolve — SUPPORTED)
- 4.4x PPL improvement validates retrain as Evolve primitive
- PPL and KR-Test diverge on legal domain: style learning ≠ knowledge acquisition
- Quality gate revised: PPL primary + KR non-regression + cos<0.05
- **Scale gap**: convergence over multiple Evolve cycles untested

## The Four Things That Don't Work (And Why)

### 1. Effective-Delta Cosine (KILLED)
**Why it fails**: Concatenating per-module B@A vectors into a mega-vector inflates cosine 19x due to concentration of measure in 2B dimensions. The A-filtering is real per-module but doesn't survive aggregation across 210 modules.
**Implication**: Raw parameter cosine is the correct operational metric. tools/orthogonality.py needs no changes.

### 2. LoRI Sparse B (KILLED)
**Why it fails**: Floor effect. Ternary weights already provide |cos|=0.0016 (114x below FP16). B-sparsity is solving a problem that doesn't exist. Magnitude pruning actually *increases* cosine by 1.46x through signal concentration.
**Implication**: Interference reduction is solved on BitNet-2B. No further orthogonality experiments needed.

### 3. Scaffold Fresh Adapters (KILLED)
**Why it fails**: Information-theoretic bottleneck. Rank-16 LoRA (0.98% of params) can adapt a pretrained base but cannot reconstruct language modeling from scratch on a random scaffold (36-642x worse PPL).
**Implication**: Pretrained base is load-bearing. >99% of model utility comes from pretrained weights.

### 4. Meta Scaffold / MAML (KILLED)
**Why it fails**: Triple failure — FOMAML's irreducible bias at K=50, gradient-disconnected composition penalty, unconstrained outer-loop destroying ternary quantization (12x PPL degradation). GaLore scaffold already achieves good composition without meta-optimization.
**Implication**: Bilevel optimization for scaffold is deprioritized. GaLore is the right base-free path.

## Cross-Cutting Insights

### Insight 1: The ternary base solves interference for free
Three experiments (effective-delta cosine, LoRI sparse B, scaffold fresh adapters) all confirmed that ternary weight constraints produce near-zero adapter interference (|cos| ~ 0.001-0.002) as a geometric property of high-dimensional ternary spaces, independent of training. This means:
- Grassmannian skeleton provides per-module guarantees but the global metric is inherently safe
- All parameter-space interference reduction techniques are redundant on ternary bases
- The binding constraint shifts from *interference* to *individual adapter quality*

### Insight 2: PPL is necessary but not sufficient
The retrain-evolve and KR-Test experiments together prove that PPL captures stylistic adaptation while missing factual knowledge. This has architectural consequences:
- The Evolve quality gate must use a two-signal approach (PPL + KR-Test)
- Adapter training budget must be sufficient for knowledge acquisition (>300 steps)
- Domain difficulty calibration is essential (medical ceiling effect at base 100%)

### Insight 3: Quantization is the universal bottleneck
GaLore scaffold (2-2.9x degradation from PTQ), meta scaffold (12x from PTQ), and the literature survey all point to the same conclusion: any training method that doesn't integrate ternary quantization into the training loop (QAT/STE) will produce quantization-hostile weights. The base-free path MUST use STE-aware training.

### Insight 4: Top-k routing with k=2 is the composition sweet spot
Top-1 overshoot, 1/N signal dilution, and top-2's convex regularization converge on k=2 as the minimum viable routing configuration. This is consistent with Mixtral-8x7B's production choice and suggests k=2 may be universal for LoRA-MoE.

### Insight 5: Scale is the remaining variable
Every supported experiment identified a specific scale gap:
- KR-Test: n=50 → n=500 for statistical power
- GaLore: PTQ → QAT transition at scale
- llama.cpp: 5 adapters → custom kernels for N>5
- Routing: per-sequence → per-token with fused kernels
- Evolve: single cycle → multi-cycle convergence

## Macro-Scale Priorities (Derived from Micro Learnings)

1. **P0: GaLore → QAT transition** — Train scaffold with GaLore warmup then STE/QAT. This is the critical path for base-free viability.
2. **P1: KR-Test at n>=500** — Statistical power for the Evolve quality gate. Without this, the gate is unreliable.
3. **P1: Multi-cycle Evolve with two-signal gate** — Validate retrain-from-scratch convergence over 3+ cycles with PPL+KR quality gate.
4. **P2: Per-token routing on GPU** — True per-token (MoLoRA-style) with load balancing (LD-MoLE). Requires fused kernels.
5. **P2: llama.cpp at scale** — Test with N=10-20 adapters, measure Metal/GPU overhead vs CPU affine model.

## Key References Discovered in Wave 3

| Paper | arXiv | Primary Finding for BitNet-SOLE |
|-------|-------|-------------------------------|
| MoLoRA | 2603.15965 | Per-token LoRA routing: 1.7B matches 8B |
| LD-MoLE | 2509.25684 | Adaptive expert count via differentiable routing |
| CoLD | 2505.14620 | Contrastive LoRA decoding as knowledge signal |
| LoRALib | 2509.18137 | Standardized LoRA-MoE benchmark (40 tasks, 680 LoRAs) |
| Q-GaLore | 2407.08296 | GaLore weights need in-training quantization |
| Continual QAT | 2502.11895 | 16→1.58-bit transition strategy |
| LoRA vs Full FT | 2410.21228 | LoRA learns style, not facts (intruder dimensions) |
| Naive LoRA Summation | 2508.11985 | Superposition Principle for independently trained LoRAs |
| LoRI | 2504.07448 | B-sparsity for interference (redundant on ternary) |
| Compress then Serve | 2407.00066 | Joint LoRA compression for 1000+ adapter serving |
