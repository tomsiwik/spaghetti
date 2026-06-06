# Learnings: exp_hidden_state_probe_router

## Core Finding

Per-token hidden states from a frozen ternary base model are **linearly separable** at 98.3% accuracy across 5 domains — making the MLP probe (the experiment's original hypothesis) unnecessary. Ridge regression applied per-token matches the MLP within 0.2pp, extending Finding #276 from sequence-level to token-level granularity with negligible degradation. The SNR model's prediction of 16x degradation from losing mean-pooling was falsified: domain signal is structured per-token, not noise reduced by averaging.

## Why This Happened (Literature-Grounded)

**The isotropic noise assumption was wrong — domain information is encoded per-token, not per-sequence.**

The SNR analysis assumed token-level hidden states are noisy copies of a domain centroid (h_t = mu_k + epsilon, epsilon ~ N(0, sigma^2 I)). This model predicts sqrt(T) = 16x degradation when classifying individual tokens vs mean-pooled sequences. The actual degradation was 0.2pp because the base model builds domain-specific representations at every token position — each h_t independently carries domain signal, not just aggregated statistics.

This is consistent with a growing body of evidence that LLMs encode high-level semantic information (including domain) in linearly separable low-dimensional subspaces of hidden states, accessible even at individual token positions:

- **Guo et al. (arXiv 2507.09709)** demonstrate that LLMs encode semantics in low-dimensional linear subspaces, with separability increasing in deeper layers. Domain-level distinctions are among the most robustly separated — a simple linear classifier suffices.
- **Wendler et al. (arXiv 2504.16871)** show that hidden-state activations during prefill capture domain-specific trajectories that are robust across architectures and prompt variations. Domain signal is not emergent from aggregation — it's present in the raw activations.
- **Meyoyan & Del Corro (arXiv 2601.13288)** frame classification as representation selection over the full token-layer hidden-state tensor, finding that lightweight probes on hidden states can predict labels in the same forward pass used for generation. Their token-and-layer-selective approach confirms that per-token hidden states carry sufficient signal for classification.

**Why ridge matches MLP at K=5:** Cover's theorem (1965) guarantees that random labelings in d=2560 are linearly separable with probability approaching 1 when d/N >> 1. With K=5 structured (not random) domain labels, the separation margin is large enough that no nonlinear correction is needed. The MLP's extra capacity is wasted.

**Why legal-finance confusion persists at cos=0.981:** Legal and finance text share vocabulary (contracts, regulations, monetary terms) and syntactic structures. This is a genuine domain overlap, not a representational failure. At K=5, this is the only confusion axis. At K=50+, similar overlaps (e.g., health-fitness vs medical) will multiply.

## Confirming Evidence

- **LLMs Encode Semantics in Linearly Separable Representations (arXiv 2507.09709):** Large-scale study of 11 decoder-only models across 6 topics. High-level semantic information lies in low-dimensional linear subspaces. Separability increases in deeper layers. Directly confirms our finding that last-layer hidden states are linearly separable for domain classification.
- **Exploring How LLMs Capture Domain-Specific Knowledge (arXiv 2504.16871):** Hidden-state activations capture domain-specific information robust across architectures. Domain signal is present per-token in the prefill phase — not just after aggregation. Confirms that our per-token linear separability is not an artifact of our specific model.
- **Token-and-Layer-Selective Probes (arXiv 2601.13288):** Lightweight probes (100K–35M params) on hidden states achieve classification within a single forward pass. A two-stage aggregator over token-layer tensor is sufficient. Confirms the probe-on-hidden-states paradigm works for production classification tasks.
- **PHATGOOSE (arXiv 2402.05859):** Post-hoc sigmoid gating on frozen LoRA modules using hidden-state dot products. Per-token, per-layer routing. Outperforms explicit multitask training in some cases. Confirms that post-hoc hidden-state routing (no joint training) can be effective.
- **Finding #276:** Ridge regression on mean-pooled hidden states: 96% accuracy, closed-form, 23s init. Our result extends this to 98.3% per-token — the mean-pooling step was unnecessary.
- **Finding #305:** Segment-isolated routing +16% over per-sequence. Validated the architecture that this probe now provides a practical router for.

## Contradicting Evidence

- **MoLoRA (arXiv 2603.15965):** Per-token routing with a jointly-trained router on Qwen3-1.7B matches Qwen3-8B. Uses a learned gating router, not a post-hoc probe. Key difference: MoLoRA's router is trained jointly with adapters via shared gradients, enabling co-adaptation. Our post-hoc probe cannot replicate this. However, at K=5 with strong linear separability, the distinction is moot — both approaches would route correctly. The difference matters at K=50+ where domain boundaries blur.
- **Token-Level LoRA Adaptation (arXiv 2311.10847):** Token-level routing outperforms per-sequence, contradicting our earlier null result (Finding #305 showed per-token full-sequence routing was null). Reconciliation: their positive result uses joint training; our null result was for post-hoc routing within a single forward pass where cross-attention contamination dominated. Segment isolation resolves this.
- **Linear probe limitations at scale:** Linear probes are fundamentally limited by the expressiveness of the underlying model's representation (Hewitt & Liang 2019, control tasks). At K=50+ domains with overlapping vocabularies, linear separability may degrade below useful thresholds. The legal-finance confusion (cos=0.981) at K=5 is the canary.

## Alternative Approaches (Paper-Grounded)

1. **PHATGOOSE per-layer gating (arXiv 2402.05859):** Train a sigmoid gate per adapter per layer, routing at per-layer per-token granularity. More expressive than our single last-layer probe. Tradeoff: N_adapters × N_layers gates vs our single ridge regression matrix.
2. **MoLoRA joint router training (arXiv 2603.15965):** Train router and adapters jointly via shared gradients. Enables co-adaptation and better routing at scale. Tradeoff: requires retraining all adapters together, breaking our "$2 and 10 minutes per domain" goal.
3. **Task representation routing — LORAUTER (arXiv 2601.21795):** Route via task embeddings rather than hidden states. Scales with task count, not adapter count. Could complement hidden-state routing at K=50+ where per-token classification degrades.
4. **Two-stage token-layer aggregation (arXiv 2601.13288):** Use multiple layers (not just last) for probe input. Their scoring-attention gate selects informative token-layer combinations. Could improve robustness at K=50+ where single-layer features may not suffice.

## Implications for Next Experiments

1. **The production routing pipeline is now complete for K=5.** Ridge regression W* (Finding #276) applied per-token gives 98.3% token accuracy → 100% segment accuracy via majority vote → segment-isolated evaluation (Finding #305). No MLP needed. No training needed. Closed-form, incremental (Woodbury updates for new domains).

2. **Segment-isolated PPL degrades vs per-sequence and vs no-adapter.** Oracle segment-isolated PPL (7.636) is WORSE than per-sequence best (7.366, -3.5%) and base only (7.465, -2.2%). The adapters trained on full sequences hurt when applied to 128-token segments. This is the next problem to solve — not routing accuracy, which is solved.

3. **K=5 is trivially separable; K=50+ is the real test.** At K=5, d/K = 512 — Cover's theorem predicts near-certain linear separability. At K=50, d/K = 51 — still high but domain overlap (legal-finance-type pairs) will multiply. The ridge router may need augmentation (multi-layer features, task embeddings) at scale.

4. **The SNR model's failure reveals structure.** Domain signal is per-token, not per-sequence. This means the base model's representations are more structured than a centroid-plus-noise model suggests. Each token's hidden state encodes not just "what domain am I in" but "what domain-specific concept am I processing." This structural insight could inform adapter design (adapters don't need to "discover" domain — the base model already knows).

## Recommended Follow-Up

**exp_segment_adapter_scale_sweep** (P1): Test LORA_SCALE values {5, 10, 15, 20} on 128-token isolated segments. Motivation: segment-isolated PPL (7.636) is WORSE than base (7.465), and Limitation 5 in PAPER.md identifies LORA_SCALE=20.0 as potentially too aggressive for short segments without full context. The adapters were trained on full sequences at s=20; segment application is a distribution shift. A scale sweep is the minimal experiment to determine if the PPL degradation is a scale confound or structural. ~30 minutes, single-seed, 5 domains. Literature support: Finding #308 showed scale confounds in purpose-trained adapters (21-37% B-matrix norm variation).
