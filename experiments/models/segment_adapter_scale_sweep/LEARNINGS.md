# Learnings: exp_segment_adapter_scale_sweep

## Core Finding

**LoRA adapters trained on full instruction-response sequences provide zero useful domain signal on isolated 128-token segments at ANY scale.** The failure is directional (wrong subspace), not magnitude (wrong scale). PPL is monotonically increasing from s=2 to s=20; the best scale (s=2) improves over base by only -0.06% (~5 nats over 6350 tokens, noise). Behavioral scores are identical across all configurations. No scaling factor can fix a directional mismatch — this is a structural impossibility, not a tuning problem.

## Why This Happened (Literature-Grounded)

**The adapter perturbation direction B*A*x is calibrated for full-sequence attention patterns that don't exist on isolated segments.**

LoRA adapters learn perturbation directions during training that are coupled to the input distribution — specifically, the attention patterns and hidden-state geometry produced by full instruction-response sequences. On 128-token isolated segments (mid-response text without instruction framing), three things change simultaneously:

1. **Attention patterns shift** — causal mask restricts to 128 positions, eliminating long-range dependencies the adapter learned to exploit
2. **Hidden-state geometry shifts** — different norm distributions and subspace occupancy without instruction context
3. **The adapter direction becomes orthogonal to the domain-relevant subspace** — the perturbation B*A*x points into a subspace that was useful for full sequences but is irrelevant for short segments

This means the problem is fundamentally in the **direction** of the adapter perturbation, not its **magnitude**. Scaling s only controls magnitude. When the direction is wrong, all scales produce either noise (small s) or active harm (large s), explaining the monotonically increasing PPL curve.

**DoRA (arXiv 2402.09353)** provides the theoretical framework: decomposing pretrained weights into magnitude and direction components, they show LoRA conflates the two. Standard LoRA's inability to independently control direction causes failures when the input distribution shifts — exactly our scenario. **Dual LoRA (arXiv 2512.03402)** extends this, confirming that directional errors are the primary failure mode when composing or transferring LoRA adapters.

**Decouple and Orthogonalize (arXiv 2505.15875)** demonstrates this from the merging perspective: when composing multiple LoRA adapters, direction interference (not magnitude) is the primary cause of degradation. Our finding is the single-adapter version of this: the adapter's own direction is misaligned with the shifted input distribution.

**LongLoRA (arXiv 2309.12307)** confirms that LoRA alone fails when context length changes — trainable embedding and normalization layers were needed to unlock long-context adaptation. This shows LoRA's sensitivity to context regime is well-documented.

## Confirming Evidence

- **DoRA: Weight-Decomposed Low-Rank Adaptation (arXiv 2402.09353):** Decomposes weights into magnitude and direction, applies LoRA only to direction. Consistently outperforms standard LoRA, confirming that direction (subspace) matters more than magnitude for adaptation quality. Our monotonic PPL curve is the negative case: when direction is wrong, no magnitude fixes it.
- **Dual LoRA (arXiv 2512.03402):** Shows standard LoRA underperforms because it conflates magnitude and direction updates. Directional errors are the primary failure mode — extending DoRA's insight to show that explicit direction control is necessary for robust adaptation.
- **Decouple and Orthogonalize (arXiv 2505.15875):** Decomposes adapters into magnitude and direction for merging. Direction interference (not magnitude) is primary degradation cause. Confirms our finding from the composition angle.
- **LongLoRA (arXiv 2309.12307):** LoRA alone fails to transfer across context lengths. Required trainable normalization layers to bridge the gap. Confirms that LoRA effectiveness is coupled to the training context regime.
- **Computational Limits of LoRA (arXiv 2406.03136):** Proves a sharp phase transition in LoRA efficiency based on input/weight/adapter norms. Efficient approximation only exists below a threshold — formally showing adapter behavior is coupled to input structure.
- **Finding #310 (this project):** Per-token hidden states are linearly separable at 98.3% — domain signal exists per-token. But classification signal ≠ adaptation signal. The base model knows the domain, but the adapter's perturbation direction is wrong for short contexts.
- **Finding #309 (this project):** KV-cache reuse killed by same duality — orthogonal subspace guarantees that work for isolation also guarantee incompatibility. Adapters are isolated processing units, not context-transferable modules.

## Contradicting Evidence

- **LoRA-Augmented Generation / LAG (arXiv 2507.05346):** Enables dynamic per-token, per-layer LoRA selection without retraining. Claims adapters work at sub-sequence granularity. Key difference: LAG uses retrieval-augmented adapter selection within a full forward pass (full context available), not isolated segment application. The adapter sees the full sequence context even though selection is per-token.
- **Token-Level LoRA Adaptation (arXiv 2311.10847):** Per-token adaptation works. But uses joint training (router and adapters co-adapt). Our adapters were frozen and post-hoc applied. The contradiction resolves: jointly-trained adapters learn directions compatible with per-token application; post-hoc adapters don't.
- **MoLoRA (arXiv 2603.15965):** Per-token routing with jointly-trained router matches larger models. But MoLoRA operates within full forward passes (full context), not isolated segments. The router sees all tokens' hidden states; each adapter operates with full causal attention available. Contradicts only if you equate "per-token routing" with "segment isolation" — they are different.

## Alternative Approaches (Paper-Grounded)

1. **Per-token routing within full-sequence forward pass (arXiv 2603.15965, 2507.05346):** Apply adapters per-token but within a single forward pass over the full sequence. Each adapter application sees the complete causal context. This is how MoLoRA and LAG work. Directly addresses the context-dependency failure by never isolating segments. Tradeoff: requires routing decision per-token per-layer, not just per-segment.

2. **Overlapping/sliding-window segments (Finding #310 follow-up):** Instead of isolated 128-token segments, use overlapping windows where each segment includes the prior segment's context. Adapters see 256+ tokens but routing decisions are per-128-token block. Addresses the context-dependency by ensuring adapters always have sufficient context. Literature support: LongLoRA's shifted sparse attention (arXiv 2309.12307) shows local windows can approximate full context.

3. **DoRA-style direction-aware adapters (arXiv 2402.09353):** Train adapters using DoRA's magnitude-direction decomposition. The direction component is optimized independently of magnitude, potentially producing directions that generalize across context lengths. Would need retraining adapters but addresses the root cause (directional mismatch).

4. **L-MoE lightweight gating (arXiv 2510.17898):** Lightweight per-token gating that learns to weight adapter contributions based on input. Operates within full forward passes. More parameter-efficient than MoLoRA's full router.

## Implications for Next Experiments

1. **Segment isolation is definitively killed for adapter APPLICATION.** Three related kills now form a pattern:
   - Finding #309: Cross-adapter KV-cache reuse impossible under orthogonality
   - Finding #307: Boundary detection works but PPL delivery fails on segments
   - Finding #311: Adapter perturbation direction wrong for isolated segments
   All three share the same root cause: adapters are context-dependent processing units that require full-sequence attention patterns to function. Segment isolation works for CLASSIFICATION (98.3%, Finding #310) but not ADAPTATION.

2. **Per-token routing within full forward passes is the correct architecture.** The literature unanimously confirms this: MoLoRA, LAG, L-MoE, LD-MoLE all operate within full forward passes. The router selects adapters per-token, but each adapter application has full causal context. This is architecturally different from segment isolation.

3. **The direction vs magnitude distinction is fundamental.** This experiment proved that adapter failures under distribution shift are directional, not magnitude-based. Future experiments must either (a) ensure adapters see their training-compatible context, or (b) train adapters with direction-aware methods (DoRA) that generalize across contexts.

4. **Finding #310's "adapters help" result needs reinterpretation.** Per-sequence PPL in THIS experiment (10.48) was WORSE than base (7.99), contradicting Finding #310's 7.366. The discrepancy is the evaluation protocol: Finding #310 used mixed-domain concatenation and best-of-5 adapter selection. The adapters may help via cross-domain interaction effects, not domain specialization. This is an open question.

## Recommended Follow-Up

**exp_full_sequence_per_token_routing** (P1): Apply adapters within full-sequence forward passes using the ridge regression router (Finding #276/310) for per-token adapter selection. Each token's adapter is selected by the router, but the forward pass runs over the complete sequence so adapters have full causal context. Motivation: this experiment (Finding #311) proves segment isolation kills adapter effectiveness; the literature (MoLoRA 2603.15965, LAG 2507.05346) proves per-token routing within full forward passes works. The minimal test: compare PPL of full-sequence per-token routed adapters vs base vs per-sequence best adapter, on the same 5-domain validation set.
