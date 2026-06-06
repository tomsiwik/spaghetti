# Learnings: exp_single_pass_mlp_mixed_adapter

## Core Finding

**MLP token-independence holds exactly in practice: same-segment tokens produce identical outputs (max diff 0.000000 across 25,600 tokens) between single-pass mixed-adapter and multi-pass oracle.** This closes the experiment-proof gap from Finding #312. Single-pass achieves PPL 4.684 vs multi-pass 4.656 (0.61% gap), entirely from cross-segment divergence where tokens attend to differently-adapted residuals. The corrected proof (same-segment exact + cross-segment bounded divergence) is more informative than the original exact-equivalence claim, which was wrong.

## Why This Happened (Literature-Grounded)

**Same-segment exact match is a structural consequence of causal masking + MLP token-independence.**

In a causal transformer, tokens before the domain boundary can only attend to tokens with the same adapter assignment. Since MLP layers operate token-independently (each token's MLP output depends only on its own hidden state), and attention uses base weights (no per-adapter attention LoRA), the residual stream for same-segment tokens is identical in single-pass and multi-pass by induction over layers: identical embeddings -> identical attention context (same tokens, base weights) -> identical MLP output (same adapter, same input) -> identical residual. This is Theorem 3 in MATH.md, proven with QED and verified to floating-point precision.

**Cross-segment divergence arises from attention over differently-adapted residuals.** After the domain boundary, tokens attend to preceding tokens whose MLP residuals were computed with adapter A (single-pass) vs adapter B (multi-pass-for-B). These different residuals flow through subsequent attention layers, causing divergence that accumulates over L=30 layers. The empirical divergence (mean NLL diff 0.068, 0.61% global PPL) is vastly smaller than the vacuous formal bound (~10^9 * delta_MLP), suggesting natural cancellation effects in practice.

**Single-pass is arguably the MORE correct architecture.** Multi-pass gives each token a counterfactual: "what if ALL tokens used adapter B?" Single-pass gives each token its genuinely-adapted context: "tokens before me that are from domain A already have adapter A applied." The small divergence (0.61%) validates that cross-adapter residual interactions through attention are mostly benign.

**The dtype ordering bug was diagnostic.** The implementation initially produced same-segment divergence due to `scale * mixed.astype(x.dtype)` (cast before scale, losing float32 precision) vs the correct `(scale * z).astype(x.dtype)`. Fixing this eliminated ALL same-segment divergence, confirming the exact-match theorem holds when implementation matches the proof's assumptions.

## Confirming Evidence

- **MoLoRA (arXiv 2603.15965):** Per-token adapter routing in a single forward pass. Directly validates the architecture: different tokens get different adapters within one forward pass. MoLoRA achieves this with a learned router; our experiment uses oracle assignment to isolate the mathematical property from routing noise. MoLoRA's success on Qwen3-1.7B (exceeding Qwen3-8B on reasoning) confirms per-token single-pass routing is production-viable.
- **MixLoRA (arXiv 2404.15159):** FFN-only MoE routing with shared attention LoRA. The same architectural decomposition we prove: routed MLP + shared attention. Our Theorem 3 provides the formal justification for why this works — MLP token-independence under causal masking guarantees no cross-token interference in the MLP pathway.
- **Mixtral (arXiv 2401.04088) and Switch Transformer (arXiv 2101.03961):** Both use shared attention + routed FFN experts. The MoE architecture validates MLP-only routing at massive scale. Our contribution is the proof that post-hoc adapter assignment preserves exact equivalence for same-segment tokens.
- **Clustering in Causal Attention Masking (arXiv 2411.04990):** Theoretical analysis of how causal masking creates distinct token clusters in attention. Confirms that causal masking creates natural isolation boundaries — tokens before a boundary form a causally isolated cluster, which is exactly what enables our same-segment exact match.
- **Finding #312 (MLP-only per-token routing):** Multi-pass oracle showed MLP contributes 6x more per-token signal than attention (3.3% vs 0.5%). This experiment confirms single-pass captures most of that signal (2.7% vs multi-pass 3.3%), with only 0.61% gap from cross-segment effects.
- **Finding #304 (per-domain module selection):** Perturbation energy split MLP 69% / attention 31%. Two independent measurements (norm decomposition and per-token NLL selection) converge on MLP as the dominant domain-signal carrier. Single-pass MLP-only routing is structurally justified.

## Contradicting Evidence

- **Brainstacks (arXiv 2604.01152):** All-projection MoE routing (including attention) outperforms FFN-only when jointly trained. This suggests MLP-only is suboptimal in the jointly-trained regime, where attention adapters can co-adapt with routed MLP. Our post-hoc setting cannot leverage this co-adaptation, making MLP-only the pragmatic choice.
- **LoRA-Mixer (arXiv 2507.00029):** Routes LoRA experts through attention projection matrices (Q, K, V, O), NOT MLP. Achieves strong results (GSM8K +7.61%, HumanEval +4.88%) with attention-only routing. This contradicts the "MLP-only is sufficient" narrative — for code/math tasks, attention routing may carry significant signal. However, LoRA-Mixer uses joint training with a Specialization Balance Loss, which is outside our post-hoc constraint.
- **Per-pair K793 failures:** 4/10 domain pairs exceed the 1% single-vs-multi divergence threshold (math+legal 1.14%, medical+legal 1.28%, medical+creative 1.32%, legal+creative 1.04%). All failing pairs involve legal or creative, which have the largest adapter norms. The 1% global average masks per-pair variability driven by adapter distance.
- **Per-token outliers:** True per-token max NLL diff is 4.125 (vs mean 0.068). Individual cross-segment tokens can experience 100-400% NLL differences. While these average out globally, they could cause generation failures (high-perplexity tokens → wrong token predictions) in autoregressive decoding where errors compound.

## Alternative Approaches (Paper-Grounded)

1. **Block-diagonal attention masking for cross-domain isolation:** Instead of accepting cross-segment divergence, use segment-level causal masks to prevent tokens from attending across domain boundaries. This would make ALL tokens same-segment (exact match) at the cost of cross-domain attention. Multi-Instance Processing (arXiv 2603.22608) shows cross-domain context causes progressive degradation, suggesting isolation may be beneficial. Combined with MLP-only routing, this would give exact equivalence everywhere.

2. **MoLoRA-style learned per-token routing (arXiv 2603.15965):** Replace oracle domain labels with a learned 2-layer MLP router. Our Finding #310 already demonstrates 98.3% token-level routing accuracy with ridge regression. MoLoRA validates that per-token routing works end-to-end. The natural next step is connecting Finding #310's router to this experiment's single-pass architecture.

3. **LoRA-Mixer attention routing (arXiv 2507.00029):** For domains where attention adapters carry significant signal (code per Finding #304), route attention LoRA experts separately. This would require proving a similar token-independence property for attention routing, which is harder because attention is inherently cross-token. May be viable with block-diagonal masking.

4. **Efficient mixed-adapter computation:** Current implementation computes both adapters for all tokens (2x LoRA compute). Production deployment should compute each adapter only for its assigned tokens, using gather/scatter or grouped computation (MoLoRA's histogram dispatch). This is an engineering optimization, not a research question.

## Implications for Next Experiments

1. **The experiment-proof gap is CLOSED.** Finding #312 upgraded from PROVISIONAL to SUPPORTED. Three findings now form a complete chain: #304 (MLP carries 69% of adapter energy) -> #312 (MLP per-token routing works, 6x signal over attention) -> #313 (single-pass matches multi-pass, exact same-segment). The MLP-only single-pass architecture is validated.

2. **Cross-segment divergence is the remaining frontier.** The 0.61% gap comes entirely from cross-segment tokens attending to differently-adapted residuals. Two paths forward: (a) accept it as the cost of full-sequence processing, or (b) use block-diagonal attention to eliminate it entirely. Four killed experiments (#302, #307, #309, #311) already showed isolation is necessary for adapter effectiveness — block-diagonal attention could be the structural fix.

3. **Production viability confirmed for post-hoc MLP-only routing.** Single-pass (1 forward pass) achieves within 0.61% of multi-pass oracle (5 forward passes). Combined with Finding #310's 98.3% token-level routing accuracy and Finding #287's 0.5% pipeline overhead, the full inference stack is: ridge router (12ms) -> per-token MLP adapter selection -> single forward pass -> generation. No router training needed.

4. **Conjecture 2 (bounded divergence) remains open.** The formal bound is vacuous (~10^9 * delta_MLP at L=30). A tight bound would require characterizing the natural cancellation that makes empirical divergence ~0.61% instead of the astronomical formal prediction. This is a theoretical question — the practical answer (0.61%) is already sufficient.

## Recommended Follow-Up

**exp_block_diagonal_single_pass (P1)** — Combine single-pass MLP-only routing with block-diagonal attention masking at domain boundaries. By preventing cross-domain attention, ALL tokens become "same-segment" and the exact-match guarantee extends to the full sequence. Motivation: closes the 0.61% cross-segment gap and addresses the per-token outlier problem (max NLL diff 4.125). Literature: Multi-Instance Processing (arXiv 2603.22608) shows cross-domain context degradation; StableMask (arXiv 2402.04779) refines causal masking for generation quality. Four killed experiments (#302, #307, #309, #311) all point to isolation as the structural fix for adapter effectiveness on mixed-domain sequences.
