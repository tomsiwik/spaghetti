# Text Classification Routing for SFT Adapters

> **Naming note:** This experiment was originally titled "contrastive routing" after the
> LoraRetriever motivation (contrastive retrieval). The actual method implemented is
> TF-IDF + logistic regression — a standard text classification pipeline. The InfoNCE
> and supervised contrastive citations below motivated the *principle* of input-based
> routing but are not directly applied in this experiment.

## Type: Guided Exploration

The motivating framework is input-based adapter retrieval (LoraRetriever, Zhao et al.
2024). The unknown is whether 5 SFT domains on a micro-scale model produce sufficiently
distinct vocabulary distributions for reliable TF-IDF text classification routing.

## A. Failure Mode Identification

**Disease:** Energy gap routing (argmin NLL reduction) collapses when SFT training
makes one adapter universally dominant. Finding #205 showed the code adapter reduces
NLL more than domain-matched adapters on 3/5 domains, yielding 36% routing accuracy.

**Root cause:** NLL is a *generation quality* signal, not a *domain identity* signal.
SFT instruction-tuning on structured data (code) produces a general conditional
generation capability that reduces NLL everywhere. The energy gap
Delta_E_k(q) = NLL(q | adapter_k) - NLL(q | base) conflates "this adapter generates
well" with "this adapter matches this domain."

**Why this is a stable failure mode:** The code adapter's universal NLL reduction is
not noise --- it reflects genuine capability improvement on structured generation.
Any NLL-based routing will select it because NLL measures generation quality, and the
code adapter genuinely is the best generator for most prompts.

## B. The Right Question (Reframe)

**Wrong question:** "How do we fix energy gap routing to not always pick code?"

**Right question:** "What routing signal is invariant to adapter generation quality
and depends only on domain identity?"

**Answer:** The *text content* of the input query carries domain identity regardless
of which adapter generates best. A medical question is recognizable as medical from
its vocabulary alone, independent of any adapter's NLL. Routing should operate on
input features, not on model outputs.

This is exactly the insight of LoraRetriever (Zhao et al. 2024, arXiv:2402.09997):
decouple routing from the LLM's output distribution by learning a separate retrieval
model that maps inputs to adapter labels.

## C. Motivation from Prior Work

### Motivation: InfoNCE and Input-Based Routing (van den Oord et al. 2018)

The InfoNCE framework motivates the idea of learning representations that capture
mutual information between inputs and labels. LoraRetriever (Zhao et al. 2024)
applied this principle to adapter routing: use input features to select adapters
instead of relying on model outputs.

### Motivation: Supervised Contrastive Learning (Khosla et al. 2020, arXiv:2004.11362)

With known labels, supervised contrastive loss can produce well-separated embeddings.
This motivates the general principle that domain-discriminative features exist and
can be learned.

### Simplification for Our Setting

With only K=5 discrete classes and known labels, we do not need contrastive learning
or a retrieval system. The problem reduces to **multi-class text classification**.

This is a crucial simplification: LoraRetriever's contrastive retrieval is designed
for open-ended adapter libraries. With K=5 fixed adapters, the optimal approach
is a supervised classifier trained on the same domain data used for SFT training.

**Approach chosen:** Train a lightweight TF-IDF + logistic regression classifier on
the instruction text from our 5 domains. This is the simplest possible implementation
of "input-based routing" and provides a clean comparison against NLL-based routing.

If even this simple baseline achieves >70% accuracy, it validates the core hypothesis
that domain identity is in the input text, not in model NLL.

## D. Design Principles

**Design Principle 1 (Domain Separability Hypothesis).** Our 5 domains (medical,
code, math, legal, finance) have sufficiently distinct vocabulary distributions that
TF-IDF features produce linearly separable representations.

*Rationale.* TF-IDF features capture term frequency weighted by inverse document
frequency. For domains with distinct vocabularies (medical: "patient", "diagnosis";
code: "function", "variable"; math: "equation", "solve"; legal: "statute", "court";
finance: "portfolio", "investment"), the TF-IDF vectors should concentrate in distinct
regions. A linear classifier partitions feature space into K convex regions.

This is a hypothesis, not a proof — whether our specific 5 domains are sufficiently
separated is the empirical question this experiment answers. Note that legal/finance
may share vocabulary (contracts, regulations), reducing separability for that pair.

**Design Principle 2 (NLL-independence).** A classifier operating on input text
only is by definition independent of adapter generation quality.

*Rationale.* The classifier f: text -> {1,...,K} operates on input text x only. It
does not access the model, any adapter, or any NLL computation. Therefore, if adapter k
has universally lower NLL (as the code adapter does), this has no effect on f(x).
This is a property of the function signature — any f(x) that does not take adapter
outputs as input is independent of adapter outputs. It is a design choice, not a
mathematical discovery.

## D'. Quantitative Predictions

1. **Routing accuracy >= 90%**: Our 5 domains have maximally distinct vocabularies.
   A TF-IDF classifier on well-separated domains typically achieves 95%+ accuracy.
   We predict >= 90% conservatively, far exceeding the 70% kill criterion.

2. **Energy gap accuracy ~36%**: Unchanged from Finding #205 (this is the baseline).

3. **Per-domain routing accuracy**: All 5 domains should achieve >= 70% individually.
   Energy gap failed on math (0%), legal (0%), finance (0%).

4. **Math correctness with correct routing**: When the math adapter is correctly
   routed (instead of code adapter), we expect math correctness similar to or
   better than the 70% achieved by the code adapter (Finding #204).
   Prediction: >= 60% (conservative).

5. **Prose domains (medical, legal, finance) not degraded**: With correct routing,
   domain-matched adapters should maintain or improve over base model.
   Prediction: 0/5 domains worse than base (vs 3/5 worse in Finding #204).

## E. Assumptions and Breaking Conditions

1. **Domain vocabulary distinctness**: If domains share most vocabulary (e.g., if
   "legal" and "finance" both discuss contracts), TF-IDF separation degrades.
   **If violated:** accuracy drops for confused domain pairs but remains above
   random (20%). We can check the confusion matrix.

2. **Training data representativeness**: The 400 train samples per domain must
   cover the vocabulary distribution of the 10 test prompts per domain.
   **If violated:** test prompts may contain unseen vocabulary, reducing accuracy.

3. **Adapter quality**: The domain-matched adapter must actually be better than a
   mismatched adapter for the behavioral outcomes (K606, K607) to pass.
   **If violated:** Correct routing does not guarantee better generation ---
   the code adapter might genuinely be better even for math.

## F. Worked Example (K=5, d=TF-IDF dim)

Consider 3 example inputs:
- x1: "What is the treatment for hypertension?" -> TF-IDF: high weight on "treatment", "hypertension" -> medical features dominate -> route to medical
- x2: "Write a Python function to sort a list" -> TF-IDF: high weight on "Python", "function", "sort" -> code features dominate -> route to code
- x3: "Calculate 15% of $240" -> TF-IDF: high weight on "calculate", "%" -> math features dominate -> route to math

Nearest centroid: cos(phi(x1), mu_medical) >> cos(phi(x1), mu_code), so route to medical.

## G. Complexity and Architecture Connection

- **Router training:** O(N * |V|) for TF-IDF on N samples, O(|V| * K) for linear classifier.
  With N=2000, |V|~5000, K=5: trivially fast (< 1 second).
- **Router inference:** O(|V|) per query for TF-IDF transform + O(|V| * K) for classification.
  Negligible compared to LLM inference.
- **No additional model loading:** Router is a scikit-learn classifier, not a neural network.
  Zero additional GPU memory.
- **Production connection:** This validates the LoraRetriever principle (input-based routing)
  at micro scale. At macro scale, the router could be a small transformer (e.g., BERT-tiny)
  trained with InfoNCE loss for open-ended adapter libraries.

## Self-Test

1. **One mathematical property making failure impossible:**
   Routing depends only on input text features, which are independent of adapter NLL,
   so NLL-dominance of one adapter cannot corrupt routing decisions.

2. **Prior work motivating the approach:**
   LoraRetriever (Zhao et al. 2024) established input-based adapter routing.
   InfoNCE (van den Oord et al. 2018) and supervised contrastive learning
   (Khosla et al. 2020) motivate the principle; our implementation simplifies
   to TF-IDF + logistic regression for K=5 fixed domains.

3. **Specific numbers predicted:**
   Routing accuracy >= 90%. Math correctness >= 60%. 0/5 prose domains degraded.

4. **What would falsify:**
   If domains are NOT separable in TF-IDF space (accuracy < 50%), the assumption
   of vocabulary distinctness is wrong. This would mean our 5 domains actually share
   too much vocabulary for text-based routing to work.

5. **Hyperparameters added:** 0 for routing (TF-IDF + linear classifier is parameter-free
   in the sense that regularization is standard). The lora_scale=20.0 is inherited
   from prior experiments.

6. **Hack check:** No. This replaces the energy gap mechanism entirely with a
   fundamentally different signal (input features vs model NLL). Single change.
