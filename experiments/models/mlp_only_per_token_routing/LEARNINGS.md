# Learnings: exp_mlp_only_per_token_routing

## Core Finding

MLP adapters contribute **~6x more per-token domain signal** than attention adapters (3.3% vs 0.5% PPL improvement over per-sequence, t(9)=4.69, p<0.001). This converges with Finding #304's perturbation energy split (MLP ~69%, attention ~31%) from an independent measurement angle — per-token NLL selection vs perturbation norm decomposition. However, the contamination elimination thesis that motivated MLP-only routing was **circumvented, not tested**: the multi-pass oracle methodology (5 separate forward passes) avoids contamination by construction for ALL strategies, and full-module per-token (4.500) beats MLP-only (4.656) in this regime.

## Why This Happened (Literature-Grounded)

**MLP layers are the natural locus of domain-specific knowledge in transformers.**

The MoE architecture principle — shared attention + routed FFN — is not a design choice but a reflection of where domain signal concentrates. In standard transformers, MLP/FFN layers store factual knowledge and domain-specific patterns (Geva et al., 2021; Dai et al., 2022), while attention layers learn general relational structure that transfers across domains. When LoRA adapts both, the MLP perturbation carries the domain-discriminative signal; the attention perturbation carries cross-domain relational adjustment that is less token-specific.

This explains the 6x ratio: per-token NLL selection rewards token-level domain discrimination, which is exactly what MLP stores. Attention adapters contribute less per-token signal because their effect is distributed across the sequence via the attention mechanism — each token's attention output depends on all tokens in the causal window, not just the current token's domain.

**Why full-module beats MLP-only in multi-pass:** When contamination is eliminated by design (each forward pass uses one consistent adapter), attention adapters provide an additional ~3.2% improvement (4.500 vs 4.656). This is the attention contribution that would be lost to contamination in a single-pass mixed-adapter regime, but is safely captured in multi-pass. The multi-pass regime is an upper bound on what any single-pass strategy can achieve.

**Why Finding #305's null was methodological:** The original experiment applied one globally-selected adapter to all tokens in a single forward pass. This cannot differentiate per-token from per-sequence selection — by definition, the best single adapter for the whole sequence IS the per-sequence choice. Multi-pass oracle (one forward pass per adapter, per-token NLL selection) was needed to reveal per-token signal.

## Confirming Evidence

- **MixLoRA (arXiv 2404.15159):** Applies top-K MoE routing to FFN layers only, with a single shared LoRA on attention projections. Directly confirms the architectural principle: FFN is where domain routing belongs, attention is shared. Our 6x MLP signal ratio independently validates this design choice.
- **Brainstacks (arXiv 2604.01152):** Extends MoE-LoRA to all seven projections including attention, finding this outperforms FFN-only MoE. However, their routed-attention uses jointly trained gating, not post-hoc per-token selection. In the post-hoc regime (our setting), attention contributes minimally per-token.
- **Finding #304 (per-domain module selection):** Perturbation norm split: attention 31%, MLP 69%. Attn-only outperforms full-module on medical/math MMLU. Our per-token NLL decomposition (attention 0.5%, MLP 3.3%, full 6.5%) provides a complementary measurement from NLL selection rather than norm analysis. Two independent methods converge on MLP >> attention for domain signal.
- **PHATGOOSE (arXiv 2402.05859):** Post-hoc per-token gating on frozen LoRA modules. Per-token, per-layer routing outperforms explicit multitask training. Confirms that post-hoc per-token routing yields genuine improvement — consistent with our multi-pass oracle showing non-null per-token signal.
- **MoE-Sieve (arXiv 2603.24044):** Profiles per-layer expert activations and applies LoRA only to top-25% most-routed experts per layer, competitive with full LoRA. Confirms that adapter signal is concentrated in a subset of modules, consistent with our finding that MLP modules carry most domain signal.

## Contradicting Evidence

- **Brainstacks (arXiv 2604.01152) full-projection routing:** When joint training is used, routing all seven projections (including attention) outperforms FFN-only MoE. This contradicts MLP-only as the optimal architecture — but only in the jointly-trained regime. Our post-hoc setting lacks gradient co-adaptation between router and attention adapters, which may explain the discrepancy.
- **Full-module per-token beats MLP-only (this experiment):** 4.500 vs 4.656 (3.4% gap). In the multi-pass regime where contamination cannot occur, more adapter signal is strictly better. MLP-only is ONLY advantageous if single-pass contamination is real — which remains untested. This is the experiment's most important negative result.
- **Token-Level LoRA Adaptation (arXiv 2311.10847):** Token-level routing with gradient-free weighting outperforms per-sequence across tasks. Uses full-module adapters, not MLP-only. Suggests full-module per-token routing works when the routing mechanism is appropriate — the issue may be single-pass mixing, not which modules are adapted.

## Alternative Approaches (Paper-Grounded)

1. **Single-pass mixed-adapter MLP-only routing (MATH.md proof):** Apply domain A's MLP adapter to tokens 0-127 and domain B's MLP adapter to tokens 128-255 in ONE forward pass. The proof guarantees no cross-attention contamination for MLP-only. This is the experiment the proof calls for and would upgrade Finding #312 from PROVISIONAL to SUPPORTED (or kill it if the proof's prediction fails).
2. **MoLoRA joint per-token routing (arXiv 2603.15965):** Per-token adapter routing with jointly trained lightweight MLP router. Achieves 98.6%+ routing accuracy per domain. Trades our "$2 per domain" post-hoc goal for better adaptation quality. At K=5, post-hoc ridge routing suffices; at K=50+, joint training may be necessary.
3. **L-MoE per-token gating within full forward passes (arXiv 2510.17898):** Per-token expert gating in MLP layers during standard (single-pass) forward passes. This is the production-viable version of what our multi-pass oracle simulates. Key difference: L-MoE uses a learned gate, our proof says a fixed per-token domain assignment suffices for MLP.
4. **Attention context isolation (segment boundaries in full sequence):** Instead of MLP-only routing, use block-diagonal attention masks to prevent cross-domain attention while keeping all adapter modules. Addresses the 15% segment-isolation gap by combining full-module adaptation with context isolation. No paper reference for LoRA-specific implementation, but block-diagonal attention is standard in multi-document settings.

## Implications for Next Experiments

1. **The experiment-proof gap must be closed.** The single-pass mixed-adapter MLP-only experiment is the natural next step. If MLP token-independence holds (which it must, structurally), single-pass MLP-only should match multi-pass MLP-only oracle. This would be a Type 1 verification experiment.

2. **Full-module multi-pass is the current best.** At 4.500 PPL (6.5% over per-seq), full-module per-token multi-pass oracle is the upper bound. Any production-viable approach must close the gap to this, not to MLP-only (4.656). The question is whether single-pass MLP-only (proven safe) is close enough to multi-pass full-module (the oracle).

3. **The 15% segment-isolation gap is fundamental.** Segment isolation (4.042) beats all per-token strategies because it eliminates cross-domain context, not just cross-adapter contamination. Four killed experiments (#302, #307, #309, #311) confirm that adapter perturbation within cross-domain context is fundamentally limited. Closing this gap requires context isolation (attention masking), not better routing.

4. **Two independent measurements now converge on MLP >> attention for domain signal.** Finding #304 (perturbation norms: 69/31 MLP/attention) and Finding #312 (per-token NLL: 6x MLP signal). This is robust enough to inform architecture: MLP-only serving is a valid optimization for domains where attention adapters provide negligible benefit (medical, math, legal, finance per Finding #304).

## Recommended Follow-Up

**exp_single_pass_mlp_mixed_adapter (P1)** — Apply different MLP-only adapters to different token positions in a SINGLE forward pass. Compare against multi-pass MLP-only oracle (4.656). The proof guarantees these should match (MLP token-independence). Motivation: closes the experiment-proof gap (Finding #312), would be a Type 1 verification. Literature: MLP token-independence is a structural property of SiLU FFN (Vaswani et al. 2017, any standard transformer MLP). Implementation: modify forward pass to apply adapter_A's MLP LoRA to tokens 0-127 and adapter_B's MLP LoRA to tokens 128-255, base attention everywhere.
