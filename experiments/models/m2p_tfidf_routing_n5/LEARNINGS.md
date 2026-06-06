# LEARNINGS — exp_m2p_tfidf_routing_n5

**Experiment type:** Verification (Type 1)
**Finding #354:** supported
**Status:** ALL PASS (K867, K868, K869)

---

## Core Finding

Text-based TF-IDF routing eliminates the covariate-shift failure mode in M2P composition: routing accuracy jumped from 36.6% (MLP on hidden states) to 95.0% (TF-IDF on input strings). With routing solved, the remaining 7.8% quality gap vs SFT is attributable entirely to M2P generation quality — the composition bottleneck has shifted from routing to generation.

---

## Why This Happened

### Root cause of old failure

The per-token MLP router (Finding #351) was trained on base-model hidden states `h_base(x)` but deployed against composed-model hidden states `h_comp(x)`. Composition changes the representation distribution (`P(h_base) ≠ P(h_comp)`), violating the i.i.d. assumption. This is structural — no amount of training data fixes it without access to composed hidden states at training time. The same failure mode appears in PHATGOOSE (arxiv:2402.15987) and other hidden-state routing approaches when the deployment model differs from the training model.

### Why TF-IDF routing is immune

TF-IDF is a pure function of the input string, computed before any model forward pass. By construction, `P(route | x, model) = P(route | x)` — routing is factored out of the model. This is the core insight of LoraRetriever (He et al., arxiv:2402.09997, §3): text-based routing decouples selection from adapter NLL and model internals. Composition cannot shift the routing distribution because routing never touches the model.

### Why TF-IDF achieves near-oracle quality

Oracle routing (ground-truth domain labels) achieves 92.2% of SFT quality. TF-IDF routing also achieves 92.2% — they are equal within measurement noise. This is because: (1) the three domains with 100% routing accuracy (arithmetic, parity, repeat) have unique format tokens that are exactly discriminative, and (2) even the two confusable domains (sort/reverse, 89%/86% accuracy) are structurally interchangeable — their adapters produce similar composition quality, so routing errors don't degrade the output.

### The sort/reverse confusion is informative

The 11-14% sort/reverse confusion rate is a prototype of real-world domain overlap. Both domains share format `{input}>{output}` and character alphabet `{a-h}`, forcing TF-IDF to rely on character-ordering statistics in the output bigrams rather than format tokens. This is statistically reliable for long sequences but degrades at length 2. At macro scale, many real domain pairs (e.g., different legal sub-domains, different code styles) will have this level of surface similarity or worse, requiring semantic routing beyond TF-IDF.

---

## Confirming Evidence (arxiv IDs)

- **LoraRetriever** (He et al., 2024, arxiv:2402.09997): Text-based retrieval-augmented composition with routing decoupled from model internals. Reports strong routing accuracy on real-domain tasks without any model forward pass.
- **Finding #207** (exp_contrastive_routing_n5): TF-IDF + logistic regression achieves 90% routing accuracy on real SFT domains (medical, code, math, legal, finance) using the same architecture. This experiment exceeds that baseline (95%) on toy domains with stronger format signal.
- **X-LoRA** (Buehler et al., 2024, arxiv:2402.07148): Input-dependent scaling without a separate router; routing emerges from model forward pass. Contrast: our approach explicitly avoids model-internal routing to prevent covariate shift.

---

## Contradicting Evidence

- **Finding #192** (exp_centralized_multiclass_routing_n24): Mean-pooled hidden states (base model) achieved only 39.4% accuracy at N=24 — consistent with our failure at N=5, suggesting hidden-state routing is fragile even before composition.
- **Switch Transformer** (Fedus et al., 2022, arxiv:2101.03961) and MoE literature generally: routing is computed from hidden states in-flight. This works when the router is trained jointly with the model; it fails in our setting because the router is trained on the base but deployed on the composed model. The MoE literature does not address post-hoc adapter routing.

---

## Alternative Approaches

### Semantic routing (when TF-IDF fails on overlapping domains)
- **DePT** (Shi et al., 2023, arxiv:2309.02869): Task-agnostic embedding routing — route on semantic embeddings rather than format tokens. Necessary when real domains share vocabulary.
- **PHATGOOSE** (Muqeeth et al., 2024, arxiv:2402.15987): Per-head activation-based gating trained to recognize domain-relevant activations. Addresses covariate shift by training gates to recognize inputs rather than model states — closer to TF-IDF in spirit, but operates on activations.
- **Soft routing with k > 1 adapters**: When domain boundaries are ambiguous, top-2 or top-3 weighted composition (Finding #58: +13.9% over uniform 1/N) can absorb routing uncertainty. The current architecture uses hard argmax, which fails at confidence boundaries.

### Semantic TF-IDF extensions
- Sentence-level TF-IDF over embedding spaces: Replace character n-grams with BM25 retrieval on domain representative documents (LoraRetriever approach). Preserves text-only routing while capturing semantics beyond surface format.
- Domain-discriminative prefix tokens: If sequences in real data can be prefixed with a domain cue (e.g., `[MEDICAL]`), TF-IDF separation becomes exact — same guarantee as arithmetic/parity/repeat in this experiment.

---

## Implications for Next Experiments

### 1. Generation quality is the new bottleneck (7.8% gap vs SFT)
Routing is solved (95% accuracy, oracle-quality composition). The remaining gap is M2P generation quality. Finding #354 and #351 both converge on 91-93% of SFT quality, suggesting this is a structural ceiling of the current M2P adapter training regime — not a measurement artifact.

**Open question:** What limits M2P generation quality? Hypothesis: the M2P bottleneck MLP (d_M2P=64) is too narrow to faithfully reconstruct SFT adapter behavior from base hidden states. This is supported by the JL bound violation noted in the teacher distillation review (d_M2P=64 < O(110) needed for lossless projection at N=5).

### 2. Sort/reverse confusion at macro scale
11-14% confusion between structurally similar domains is the irreducible floor for surface-level text routing. At real scale, domain pairs like `medical-clinical` vs `medical-research`, or `legal-contracts` vs `legal-litigation`, will exhibit similar or higher overlap. TF-IDF must be augmented with semantic routing for real-domain deployment.

### 3. Parity adapter seed-specificity
M2P loss on parity (2.68) is 4.6x worse than SFT (0.58) despite the base model already handling parity well (0.59 base loss). The reused parity adapter was trained with a different random seed. This suggests M2P adapters are seed-specific — they capture the particular weight trajectory of one training run, not the abstract domain knowledge. Implication: M2P adapters trained on one seed may not compose with adapters trained on another seed.

### 4. Hard routing vs. soft routing
Current architecture uses hard argmax routing (one adapter per sequence). The sort/reverse confusion (11-14%) is tolerable now because the domains are structurally interchangeable. For real domains where adapters are NOT interchangeable, misrouting will cause quality degradation proportional to the routing error rate. Top-k weighted routing (Finding #58) would provide a hedge.

---

## Recommended Follow-Up

### P0: M2P generation quality improvement
**Motivation:** Finding #354 confirms routing is solved; generation quality (7.8% gap) is the only remaining bottleneck.
**Literature:** JL-Lemma (Johnson-Lindenstrauss, 1984) — lossless projection requires d_M2P ≥ O(log N / ε²). At N=5, ε=0.1, this gives d_M2P ≈ 110. Current d_M2P=64 is 40% below the JL bound. Hypothesis: increasing d_M2P to 128 or 256 should close the 7.8% gap.
**What to measure:** Compare M2P generation quality (K-criterion: quality ratio ≥ 97% of SFT) at d_M2P ∈ {64, 128, 256} on all 5 domains.

### P1: Semantic routing for real-domain overlap
**Motivation:** Sort/reverse confusion (11-14%) is the TF-IDF ceiling for similar domains.
**Literature:** LoraRetriever (arxiv:2402.09997) for semantic retrieval-based routing; DePT (arxiv:2309.02869) for task-embedding routing.
**Only after:** M2P generation quality is improved (P0 above) — routing is already solved for toy domains.

---

## References Added

- #517 — LoraRetriever (He et al., 2024, arxiv:2402.09997): Text-based retrieval routing, routing decoupled from model internals
- #518 — PHATGOOSE (Muqeeth et al., 2024, arxiv:2402.15987): Activation-gating routing that avoids deployment-time covariate shift
- #519 — DePT (Shi et al., 2023, arxiv:2309.02869): Task-agnostic embedding routing for semantic domain separation
