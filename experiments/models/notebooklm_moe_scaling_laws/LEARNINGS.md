# LEARNINGS: MoE Scaling Laws for LoRA Composition

## Core Finding

MoE scaling laws (Apple 2501.12370, Joint MoE 2502.05171) were derived for full FFN experts (432-648x larger than LoRA) and do not transfer to LoRA composition. Runtime LoRA is mathematically identical to output-space MoE — our architecture already IS an MoE system, making the "MoE vs LoRA" framing a false dichotomy.

## Why This Happened

The key insight is dimensional: traditional MoE experts must learn *complete* functions (O(d^2) params), while LoRA experts learn *perturbations* to an existing function (O(d*r) params). The Apple scaling law's "minimum viable scale" constraint applies to the former — an underparameterized FFN expert can't learn a useful transformation. But a rank-16 perturbation to an already-competent base model needs no minimum scale; it just needs the base to be good enough.

**Mathematical structure:** h = Wx + Σ g_i B_i A_i x is output-space MoE where each "expert" f_i(x) = B_i A_i x contributes independently. No cross-terms within a layer (confirmed by LoRI, arxiv:2504.07448). Cross-layer interference through LayerNorm and attention is empirically bounded (gamma=0.982 at N=25).

**Joint MoE Scaling Laws (arxiv:2502.05171)** explicitly state their model applies to "the standard MoE variant, where the size of each expert matches the size of the feed-forward layer in the corresponding dense model." The formula L(N,D,S) was never intended for rank-16 perturbations.

## Confirming Evidence

- **Cao et al. (arxiv:2508.11985)**: LoRA composition works at 117M params (GPT-2 Small) with rank-4 adapters. Math+Medicine improves PPL by -9.10%. This is 4x below the ~500M MoE threshold interpretation.
- **MoLoRA (arxiv:2603.15965)**: Qwen3-1.7B + 4 LoRA adapters beats Qwen3-8B on reasoning. Per-token routing at sub-2B scale.
- **Our own experiments**: 24 adapters compose with gamma=0.982 on BitNet-2B-4T. Softmax router matches oracle at N=24. Pre-merge -36% worse confirms runtime LoRA is the right serving strategy.
- **LoRA Soups (arxiv:2410.13025)**: Dynamic CAT composition outperforms model-merging by 43% average. k=2 skill composition shows superlinear gains.

## Contradicting Evidence

- **GOAT framework (ICML 2026)**: Derives a theoretical scaling factor that aligns LoRA MoE optimization with full fine-tuned MoE, achieving state-of-the-art across 25 datasets. This suggests MoE scaling *concepts* (not coefficients) may transfer to LoRA — you just need a different scaling factor. The gap between LoRA MoE and full MoE is bridgeable with proper SVD-structured optimization alignment.
- **OSRM (arxiv:2505.22934)**: Weight-space orthogonality doesn't guarantee semantic disentanglement. "Existing merging methods often fail for LoRA due to significant performance degradation." Parameter interference is real even at small scale — it's not just a large-N problem.
- **Cao et al. (arxiv:2508.11985)** (same paper, different finding): Finance+Medicine composition *degrades* PPL by +27.56%. Orthogonality alone doesn't prevent interference when domains conflict. The "LoRA works at small scale" claim requires domain compatibility.
- **LoRI (arxiv:2504.07448)**: Standard LoRA "suffers from parameter interference in multi-task scenarios" without forced orthogonal constraints. Our Grassmannian skeleton addresses this, but the problem is real for unconstrained LoRA.

## Alternative Approaches

1. **LD-MoLE (arxiv:2509.25684)**: Differentiable routing with dynamic expert count per token. Replaces fixed top-k with learned threshold — some tokens use 0 experts, others use 3+. Subsumes our entropy-adaptive gating. Proven on Qwen3-1.7B.

2. **CoMoL (arxiv:2603.00573)**: Core-space merging routes in r x r space (256-dim at r=16) instead of d x d, reducing per-expert routing overhead dramatically. Retains MoE-LoRA adaptability at standard LoRA parameter cost. Relevant when scaling to N>25.

3. **D-MoLE (ICML 2026)**: Dynamic layer-wise expert allocation under parameter budget. Automatically distributes LoRA experts across layers. 15% improvement over baselines for continual learning. Relevant for our adapter growth story.

4. **GOAT SVD-structured MoE (ICML 2026)**: Derives scaling factor aligning LoRA MoE with full fine-tuned MoE. Achieves full-FT performance across 25 datasets without modifying architecture. Could inform how we calibrate routing weights.

## Implications for Next Experiments

1. **"LoRA vs MoE" is resolved** — runtime LoRA IS output-space MoE. Stop framing them as alternatives. The real question is routing strategy (per-sequence vs per-token, static vs dynamic top-k).

2. **No empirical LoRA scaling law exists yet.** The Apple law doesn't transfer, but we haven't derived our own L(N, r, d, D) for LoRA composition. An empirical sweep of (N, r) would be valuable but is a large undertaking.

3. **Base model quality is the binding constraint.** Our oracle PPL analysis shows 6.2x quality spread driven by the base's per-domain capability. Adding more experts past the base's domain coverage has diminishing returns. Improving the base (P1 track) is more impactful than N=50+ experts.

4. **GOAT's scaling factor is the most interesting contradicting signal.** If LoRA MoE optimization can be aligned with full MoE via a single scaling factor, our Grassmannian skeleton might benefit from SVD-structured initialization (connecting to PiSSA, arxiv:2404.02948).

5. **Domain compatibility matters more than scale.** Cao's +27.56% degradation on Finance+Medicine (117M) shows that "LoRA works at small scale" needs qualification: *orthogonal adapters on compatible domains* work at small scale. Our routing heads (100% accuracy at N=5) address this by not composing incompatible adapters.

## Recommended Follow-Up

**exp_lambda_scaling_law** (already recommended from lora_soups_cat): Sweep lambda {0.5, 0.7, 1.0, 1.5, 2.0} for Task Arithmetic composition. Motivated by: monotonic improvement 0.1→0.5 in lora_soups_cat, Cao et al. theory predicting lambda=1.0 optimal, OSRM concern about function-space interference wall at high lambda. ~2 min compute.

No new experiment recommended from this literature review — its value is in settling the MoE-vs-LoRA framing and providing concrete recommendations (top-2 routing, N=8-16, runtime LoRA for routed) that inform the deployment track.
