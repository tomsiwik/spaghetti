# Learnings: exp_depth_routed_adapters

## Core Finding

Per-layer depth routing (AttnRes pseudo-queries + expert embeddings) cannot improve over token-level softmax routing when the token router already achieves oracle performance. Depth weights remain near-uniform (entropy ratio 0.992, threshold <0.95) and the routing degrades quality by -18.3% (3/3 seeds consistent). The optimization landscape has no gradient toward depth specialization at L=4 with N=5 domains.

## Why This Happened (Literature-Grounded)

**Token-level routing saturates the oracle gap.** At N=5 with 100% router accuracy, every input gets its optimal adapter. The depth router's only possible contribution is per-layer weighting of an already-correct adapter — but the trained adapter weights are optimized for uniform application across layers. Any non-uniform scaling disrupts this.

**L=4 is too shallow for depth effects.** This confirms exp_attnres_depth_composition: at L=4, each layer contributes ~25% of the residual. The Kimi AttnRes paper (arXiv 2603.15031) showed benefits at L=48 (each layer ~2%). The adapter norm gradient (L0/L3 ratio 1.1x-2.8x) exists but is insufficient for depth routing to exploit without causing instability.

**The "mixed" domain catastrophe reveals the mechanism.** Mixed domain has the steepest per-layer norm gradient (1.06→2.99 at seed 42, 2.8x ratio). When depth routing applies non-uniform weights (even small deviations from 0.200), it amplifies this gradient. The result is PPL blow-up on mixed (1.02→1.84 at seed 42, 1.02→4.96 at seed 314) while other domains with flatter gradients (1.1-1.3x ratio) are barely affected.

**Gradient-free optimization is adequate for this search space.** The space has only 4×32 + 5×32 = 288 parameters. 40 perturbation iterations with sigma decay found improvements in seeds 42 and 137 (reducing PPL by 0.06% and 0.19% respectively), proving the search is not stuck. The optimizer correctly discovers that near-uniform is optimal.

## Confirming Evidence

1. **exp_attnres_depth_composition (SUPPORTED):** AttnRes learns non-uniform weights (entropy 0.775) but composition improvement is 0.39% at L=4. Our depth router similarly cannot find useful non-uniform weights (entropy 0.992). Same root cause: insufficient depth.

2. **exp_softmax_router_scaling (KILLED on K1, but S1 PASS):** Softmax token routing matches oracle (gamma 0.625 = oracle at N=24). When token routing is this good, there is no room for a second routing axis to improve.

3. **exp_more_adapters_is_better (KILLED):** Binary routing heads collapsed at N>10. The softmax fix (exp_softmax_router_scaling) resolved the token-routing bottleneck. Depth routing addresses a non-existent bottleneck.

4. **Kimi AttnRes (arXiv 2603.15031):** Benefits at L=48 (1.25x compute-equivalent improvement). Not tested for LoRA composition. Our micro result (no benefit at L=4) is consistent with their scale-dependent finding.

## Contradicting Evidence

1. **MoDA (arXiv 2603.15619):** Achieves +2.11% at 1.5B with unified sequence+depth attention. However, MoDA operates at scale (1.5B, many layers) and on the base model, not LoRA adapters. Our experiment is adapter-specific and micro-scale.

2. **Adapter norm gradient exists but is unexploitable.** Per-layer ΔW norms increase monotonically (1.3→1.7 typical, 1.1→3.0 for mixed). Theory predicts depth routing could exploit this. Practice shows that scaling trained adapter weights by non-uniform factors destroys more quality than it gains.

## Alternative Approaches

1. **Gradient-based depth router training.** Our perturbation search found the same optimum as uniform. Gradient-based training through the per-sample routing loop might find better solutions, but the fundamental issue (no oracle gap to exploit) would remain.

2. **Training adapters WITH depth routing.** If adapters are trained with non-uniform per-layer scaling from the start, they would adapt to non-uniform application. This changes the experiment from "can depth routing help existing adapters?" to "can depth-aware adapters be better?" — a more expensive question.

3. **Test at L≥16.** The attnres_depth_composition LEARNINGS.md recommends L≥16 for depth effects. At L=16, each layer contributes ~6%, and the norm gradient would be steeper, giving the depth router more signal. However, the micro transformer at L=16 would need >4x more compute and the character-level task may still be too simple.

## Implications for Next Experiments

1. **Depth routing is a dead end when token routing is optimal.** The architecture should focus on improving token routing quality (e.g., at harder tasks where 100% accuracy is not achievable) rather than adding routing axes.

2. **The token-only softmax router is the standard.** Combined with the softmax_router_scaling result (oracle-matching at N=24), depth routing adds complexity without benefit. The routing architecture is: softmax over adapter pool, top-1 selection, full-weight application.

3. **The interesting regime is token routing with oracle gap.** If future experiments find domains where the softmax router has significant oracle gap (e.g., ambiguous inputs, domain boundaries), depth routing MIGHT help. But the current evidence suggests the router's job is done once it identifies the right semantic cluster.

4. **Per-layer adapter effects exist but are not routable.** The monotonic norm increase (L0 < L1 < L2 < L3) is a genuine per-layer effect, but it's baked into the adapter during training. Changing it at inference hurts. If per-layer specialization is desired, it should be a training-time design choice (e.g., layer-specific LoRA rank or training schedule), not a routing-time modulation.
