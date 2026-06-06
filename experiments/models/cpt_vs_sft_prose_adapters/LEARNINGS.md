# LEARNINGS: CPT vs SFT Prose Adapters (KILLED)

## Core Finding

**CPT does not inject domain knowledge at 200 iters / 80K tokens / rank-16 — it acts as a
near-identity adapter.** The legal CPT adapter (0.097) matched the base model (0.098) exactly,
meaning zero knowledge was injected. The +74% over SFT is entirely because SFT at scale=20
destroys legal prose capability (0.056 vs base 0.098), confirming Finding #209's causal mechanism:
format patterns overwrite base model knowledge.

## Why This Happened

### 1. Optimization Bottleneck, Not Capacity Bottleneck (Primary Cause)

The adversarial review correctly identified that rank-16 LoRA has ample capacity for 80K tokens
(~10.9M ternary params >> ~400K bits of legal corpus entropy). The bottleneck is **200 iterations
of SGD** — insufficient to find the relevant low-rank subspace for legal knowledge.

**"Learning Beyond the Surface" (arXiv:2501.17840)** directly investigates how far CPT with LoRA
can enhance domain-specific insight learning. Their key finding: CPT on original documents has
**marginal effect** — the model must be trained on modified documents retaining only essential
information to significantly enhance insight learning. Raw legal text contains enormous amounts
of redundant formatting, citation boilerplate, and discourse structure that dilute the signal.
At 80K tokens, the signal-to-noise ratio for legal knowledge is too low for 200 iterations to
extract anything meaningful.

**Gururangan et al. (arXiv:2004.10964)** showed DAPT works but used millions of tokens with
full model fine-tuning, not 80K with rank-16. The scale gap is 2-3 orders of magnitude in data
and unconstrained model capacity vs low-rank.

### 2. LoRA as Knowledge Memory: Capacity ≠ Accessibility

**"Understanding LoRA as Knowledge Memory" (arXiv:2603.01097)** provides systematic empirical
investigation mapping the design space of LoRA-based memory. Key insight: LoRA modules have
measurable storage capacity, but **internalization** (getting knowledge into the module
efficiently) is a separate challenge. Storage capacity being sufficient does not mean the
optimization landscape allows efficient knowledge injection in limited training steps.

**"The Scaling Law for LoRA Based on Mutual Information Upper Bound" (arXiv:2501.03152)**
formalizes this: LoRA's learned knowledge has mutual information with both the pre-training
knowledge (frozen) and the new data. When the LoRA module learns LESS that relies on the base
model, it captures MORE specific knowledge of new data. At 200 iters, the LoRA barely moved
from initialization — it learned nothing specific, defaulting to base model behavior.

### 3. Format-Knowledge Divergence Determines SFT Damage

The experiment cleanly confirmed the capacity-allocation theory for the SFT direction:
- **Legal** (high format-knowledge divergence): SFT degrades -43% vs base. The instruction-
  response format is alien to legal prose structure.
- **Medical** (low format-knowledge divergence): SFT ≈ CPT ≈ base+10%. Clinical Q&A format
  naturally aligns with medical communication patterns.

This is consistent with **LIMA (arXiv:2305.11206)**: knowledge comes from pre-training, alignment
is about style/format. When SFT format conflicts with domain prose style, the adapter destroys
more than it adds.

### 4. The Rank Trade-off as Regularizer

**"How Much is Too Much? Exploring LoRA Rank Trade-offs" (arXiv:2512.15634)** found that LoRA's
constrained adaptation space acts as a regularizer — low rank limits degrees of freedom, preventing
drastic alteration of the core knowledge base. This is exactly what we observed: rank-16 CPT on
legal text couldn't alter the base model's behavior at all, even when we wanted it to. The same
property that prevents catastrophic forgetting also prevents knowledge injection at small data scale.

## Confirming Evidence

- **Finding #209:** SFT adapters degrade prose domains vs base (legal -30%, finance -14%).
  Our experiment confirmed this with causal mechanism: format patterns overwrite base knowledge.
- **Finding #211:** Prose domains show negligible adapter benefit even with execution-based eval.
  Medical +5.8%, legal -16.6%. Consistent with our medical tie and legal no-improvement.
- **Finding #249:** Scale=4 for legal, scale=20 for math/code. The two-regime solution
  correctly predicts that scale=20 maximizes SFT damage on legal.
- **Finding #216:** All 5 SFT adapters have 0.97 inter-cosine. Adapters learn format, not
  knowledge — LIMA hypothesis confirmed at the weight level.

## Contradicting Evidence

- **Gururangan et al. (arXiv:2004.10964)** showed DAPT works at scale. Our failure may be
  purely a data quantity issue, not a fundamental impossibility. With 10-100x more legal text,
  CPT might succeed.
- **arXiv:2501.17840** showed CPT with LoRA CAN enhance insight learning when documents are
  pre-processed to retain only essential information. Our raw legal text may have been too noisy.

## Alternative Approaches

1. **Curated CPT data (arXiv:2501.17840):** Instead of raw legal text, extract and train on
   essential legal facts/principles only. The paper shows this significantly outperforms CPT
   on original documents. Would require a preprocessing pipeline.

2. **Higher rank + more data:** The most obvious fix. Rank-64 with 1M+ tokens of legal text
   could cross the optimization threshold. But this conflicts with the composition architecture
   (rank-16 is already our per-adapter budget).

3. **Retrieval-Augmented Generation:** Don't inject prose knowledge into adapters at all.
   Use adapters for FORMAT (structured tasks where format=capability) and RAG for KNOWLEDGE.
   LIMA's insight: if knowledge comes from pre-training, and we can't re-pretrain, retrieval
   is the principled alternative.

4. **LoRA-Augmented Generation (arXiv:2507.05346):** Combines LoRA with retrieval for
   knowledge-intensive tasks. Adapters provide task-specific behavior while retrieval provides
   knowledge — a hybrid that avoids the knowledge-injection bottleneck entirely.

5. **Doc-to-LoRA (arXiv:2602.15902):** A hypernetwork that converts documents directly into
   LoRA weights in a single forward pass. Bypasses the optimization bottleneck entirely by
   learning the document→adapter mapping end-to-end.

## Implications for Next Experiments

1. **The training objective question (CPT vs SFT) is settled for our operating point.**
   At 80K tokens / rank-16 / 200 iters, neither CPT nor SFT can inject prose knowledge.
   CPT is a no-op; SFT can actively harm. The difference only matters at much larger data scale.

2. **Adapters should be reserved for FORMAT tasks (math, code) where format IS capability.**
   Prose domains (legal, finance) should use scale=4 (Finding #249) to avoid damaging the
   base model. If prose knowledge improvement is needed, look to retrieval, not adaptation.

3. **The P0 deployment track is unaffected.** The critical path (generation quality test,
   real benchmarks) uses math/code/medical adapters where SFT works. Legal/finance adapters
   should be treated as "preservative" (don't harm base) rather than "injective" (add knowledge).

4. **Meta-pattern: 200 iters is an optimization wall, not a capacity wall.** Future
   experiments claiming to inject knowledge must either (a) train longer, (b) use curated
   data, or (c) bypass SGD optimization entirely (Doc-to-LoRA, hypernetworks).

## Recommended Follow-Up

No immediate follow-up experiment recommended. The training-objective question is resolved
for our operating point, and the P0 track (generation quality existential test) is higher
priority. If prose domain improvement becomes blocking for P0, the first experiment should
be curated CPT data (arXiv:2501.17840 approach) rather than scaling raw CPT.
