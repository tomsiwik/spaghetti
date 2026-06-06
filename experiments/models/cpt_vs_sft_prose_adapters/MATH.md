# MATH.md: CPT vs SFT Adapters for Prose Domains

## Type: Guided Exploration (Type 2)

The proven framework is the two-regime model (Finding #249): adapters operate in
FORMAT regime (s<=4) or CAPABILITY regime (s>=20). The unknown is whether changing
the training objective from SFT (instruction-response CLM) to CPT (raw text CLM)
produces adapters that inject domain knowledge rather than format patterns.

## Step A: Diagnose the Disease

**Symptom:** SFT adapters degrade prose domains (legal -30%, finance -14% per
Finding #209).

**Disease:** SFT training on instruction-response pairs teaches the adapter to
reproduce the FORMAT of responses (sentence structure, hedging patterns, discourse
markers) rather than domain KNOWLEDGE (terminology, factual relationships, legal
doctrines, medical protocols). This is structural, not a hyperparameter problem.

**Why it is a stable failure mode:** The SFT loss L_SFT = -sum log p(y_t | y_<t, x)
is computed over the response tokens y conditioned on instruction x. The adapter
learns the conditional distribution p(response | instruction). For prose domains
where the "correct" response requires domain knowledge NOT in the base model, the
adapter compensates by learning format patterns that reduce loss without encoding
the needed facts.

**Evidence this is the disease:** LIMA (2305.11206) proves that all knowledge comes
from pre-training; SFT teaches only format/style. Finding #249 confirms two distinct
behavioral regimes corresponding to this split.

## Step B: The Right Question

**Wrong question:** "How do we make SFT adapters work better on prose domains?"

**Right question:** "Under what training objective does a LoRA adapter provably
encode domain-specific token co-occurrence statistics (knowledge) rather than
conditional response patterns (format)?"

**Answer from classical NLP:** Causal language modeling on raw domain text. The
CPT objective L_CPT = -sum log p(x_t | x_<t) computed over raw domain text
forces the adapter to model the unconditional distribution p(x) of domain text.
This necessarily encodes domain-specific token co-occurrences -- the definition
of domain knowledge at the token level.

## Step C: Prior Mathematical Foundations

**Theorem (DAPT effectiveness, Gururangan et al. 2020, Theorem via empirical
demonstration across 4 domains):** Continued pre-training on unlabeled domain
text D_domain improves downstream task performance across all tested domains
(biomedical, CS, news, reviews), with gains ranging from +1.0 to +8.3 F1 points.
The effect is additive: DAPT followed by task-adaptive pretraining (TAPT)
combines both benefits.

**Observation (LIMA, Zhou et al. 2023):** A model fine-tuned on only 1000
carefully curated examples matches GPT-4 preferences 43% of the time, despite
having orders of magnitude less SFT data. This demonstrates that SFT primarily
teaches output format, not knowledge.

**Information-theoretic framing:** Let H(D) be the entropy of domain corpus D.
The adapter parameter update delta_theta encodes information about D bounded by:

  I(delta_theta; D) <= C(r, d)

where C(r, d) is the capacity of a rank-r LoRA perturbation in d-dimensional
space. Under SFT, this capacity is split between format information I_format
and knowledge information I_knowledge:

  I(delta_theta; D) = I_format + I_knowledge

Under CPT, since raw text has no separate format component:

  I(delta_theta; D) = I_knowledge

Thus CPT allocates ALL adapter capacity to domain knowledge.

## Step D: Predictions

**Proven framework:** Two-regime model (Finding #249).

**Unknown:** The behavioral quality delta between CPT and SFT adapters on prose
domains.

**Predictions:**

| ID  | Prediction | Derived From |
|-----|-----------|-------------|
| P1  | CPT adapters improve factual recall on legal by >= 15% vs SFT | Capacity allocation: CPT uses full capacity for knowledge |
| P2  | CPT adapters improve factual recall on medical by >= 15% vs SFT | Same mechanism |
| P3  | CPT training converges (last_50_loss < first_50_loss * 0.95) | CLM is a well-conditioned objective |
| P4  | CPT output coherence >= 80% (not incoherent) | Base model provides language structure; adapter adds domain knowledge |
| P5  | CPT training completes in < 2 hours on M5 Pro | Same architecture as SFT, 200 iters |

**Kill criteria mapping:**
- K1 (#672): CPT WORSE than SFT on legal AND medical -> KILL. Tests P1 and P2.
- K2 (#673): Incoherent output > 20% -> KILL. Tests P4.
- K3 (#674): Fails to converge or > 2 hours -> KILL. Tests P3 and P5.

## Step E: Assumptions and Breaking Conditions

**A1:** The existing SFT training data contains domain text that can be
reformatted for CPT (stripping instruction/response markers).
*If violated:* Would need separate raw domain corpus, adding a confound.

**A2:** 400 training samples of raw domain text provide sufficient signal
for CPT at rank-16.
*If violated:* CPT might need more data than SFT (Gururangan used millions of
tokens). This is the most likely failure mode. However, even 400 samples x ~200
tokens = 80K tokens should provide some domain signal at rank-16.

**A3:** The base model (BitNet-2B-4T) has a reasonable prior for legal and
medical language from its pre-training corpus.
*If violated:* CPT cannot fill knowledge gaps that are too large for the
adapter capacity.

**A4:** Behavioral evaluation (factual recall) accurately measures domain
knowledge injection.
*If violated:* We might see CPT adapters with better knowledge but worse
factual recall due to generation style differences.

## Step F: Worked Example (Conceptual)

Consider a legal question: "Is it stalking to pay someone's bills?"

**SFT adapter learning signal:** The adapter sees (instruction, response) pairs.
It learns patterns like "I am paying...", "restraining order", response structure.
The gradient updates lora_b to reproduce these patterns.

**CPT adapter learning signal:** The adapter sees raw legal text including the
full context of legal discussions. It learns co-occurrence patterns: "restraining
order" near "stalking", "harassment" near "contact". The gradient updates lora_b
to increase probability of legal terminology in legal contexts.

**At inference:** When asked a legal question, the CPT adapter increases the
probability of relevant legal terms and reasoning patterns, while the SFT adapter
primarily adjusts response format.

## Step G: Complexity and Architecture

**Training cost:** Identical to SFT (same model, same iterations, same LoRA rank).
Only the data format changes.

**Inference cost:** Zero additional cost. CPT adapter is structurally identical
to SFT adapter (same rank, same projection keys). Uses existing Grassmannian
skeleton and pre-merge infrastructure.

**Composition:** CPT adapters compose identically to SFT adapters via the
existing pre-merge pathway: W_new = W_base + scale * B^T @ A^T.

## Self-Test

1. **One mathematical property:** CPT allocates full adapter capacity to domain
   knowledge (I_knowledge) rather than splitting with format (I_format + I_knowledge).

2. **Existing theorems:** DAPT effectiveness (Gururangan 2020, 2004.10964);
   LIMA observation (Zhou 2023, 2305.11206).

3. **Specific numbers:** P1: >=15% factual recall improvement on legal vs SFT.
   P2: >=15% on medical. P4: >=80% coherent outputs.

4. **Falsification:** The proof is wrong if CPT adapters score WORSE than SFT on
   factual recall for BOTH legal AND medical. This would mean the format component
   is actually helpful for prose generation (i.e., format IS knowledge for prose).

5. **Hyperparameters added:** 0. Same training config as SFT (rank-16, 200 iters,
   lr=1e-4). Only the data format changes.

6. **Hack check:** No fixes stacked. Single variable changed: training data format
   (instruction-response -> raw text).
