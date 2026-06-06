# LEARNINGS: Self-Contrast Decoding (KILLED)

## Core Finding

**Self-contrast decoding is structurally incompatible with Grassmannian-orthogonal LoRA adapters.**
The same mathematical property (orthogonality) that guarantees interference-free composition
also guarantees that non-primary adapters carry zero contrastive signal. These are dual properties.

## Literature Grounding

### Directly Relevant

- **SCMoE (2405.14507):** Self-contrast on MoE experts works because experts are co-trained FFN
  blocks sharing embeddings and attention. The unchosen experts have meaningful per-token preferences
  that correlate with generic tokens. Our adapters share only the frozen base weights and were trained
  independently — the SCMoE mechanism does not transfer.

- **Contrastive Decoding (Li et al., 2210.15097):** The original CD paper uses a small amateur model
  vs. large expert model. The amateur is "reasonably competent but less specialized." Our non-primary
  adapters are neither competent in the primary domain nor systematically less specialized — they
  operate in orthogonal subspaces and produce uncorrelated noise.

- **DExperts (2105.03023):** Expert/anti-expert pairs for controlled generation. Works when the
  anti-expert has a known, correlated relationship with the target distribution (e.g., toxic language).
  Independent orthogonal adapters have no such relationship.

### Contradicting Evidence / Alternative Approaches

- **LoRAuter (2602.21222):** Cosine similarity retrieval + nucleus sampling achieves multi-adapter
  composition (70.95% PIQA) WITHOUT contrastive decoding. The routing signal comes from output
  similarity, not adapter interference. This suggests the right composition approach is better
  selection, not cross-adapter signal extraction.

- **RDLC (Retrieval-Driven LoRA Composition):** Similarity-based retrieval over adapter outputs
  avoids the orthogonality problem by choosing adapters based on output quality, not subspace
  relationships.

- **Zhang et al. 2025 (Inter-LoRA Orthogonality):** Weight-space orthogonality ≠ semantic
  disentanglement. Our adapters are orthogonal in weight space but this tells us nothing about
  their semantic relationship. This further explains why contrastive decoding (a semantic
  operation) cannot leverage weight-space orthogonality.

### Analytical Kill Argument (from Adversarial Review)

The adversarial review correctly notes this experiment could have been killed in 5 minutes
without code:

1. Grassmannian skeleton guarantees A_i^T A_j ≈ 0 for i ≠ j
2. Non-primary adapter logit contributions project h onto subspaces orthogonal to primary
3. avg(Delta_{Q_j}(x)) captures components of h orthogonal to primary domain's specialization
4. Subtracting an orthogonal signal cannot sharpen the primary signal — only adds noise
5. QED: MATH.md Proposition 2 (noise condition) holds by construction

**Lesson: Before running frontier extensions, check whether the base architecture's structural
guarantees analytically preclude the mechanism.** This is a 5-minute derivation that would have
saved ~43 minutes of compute.

## Key Insight: The Proxy Chain Extends to Six Levels

The research has now documented a six-level cascade where each proxy fails to predict the next:

1. PPL doesn't predict MMLU accuracy (Finding #236, r=0.08)
2. MMLU accuracy doesn't predict behavioral quality (Finding #238)
3. PPL improvement sets don't predict specialization structure (Finding #240)
4. Cosine similarity doesn't predict functional disagreement (Finding #240)
5. Domain classification doesn't predict composition quality (Finding #243)
6. **Adapter orthogonality doesn't predict contrastive value (Finding #245)**

Each level was discovered by assuming the previous proxy chain was sufficient.

## Key Insight: Duality of Orthogonality

| Property | Grassmannian Orthogonal | Co-trained (MoE) |
|----------|------------------------|-------------------|
| Interference-free composition | YES (by construction) | NO (shared representations) |
| Contrastive value extraction | NO (noise, not signal) | YES (correlated preferences) |
| Independent training | YES | NO (joint optimization) |
| Bridge adapters needed | YES (H¹=3, Finding #242) | NO (implicit reconciliation) |

This is a fundamental trade-off. Our architecture chose the left column. The benefits
(zero interference, independent training, composability) come at the cost of no cross-adapter
information flow. Future work must accept this trade-off, not try to circumvent it.

## What This Rules Out

1. **Any training-free method that relies on cross-adapter signal:** Self-contrast, DExperts-style
   ensemble, adapter interpolation — all require correlated representations that our architecture
   explicitly eliminates.

2. **Adapter-averaging composition:** Finding #23 already showed uniform composition produces
   trillion-level PPL. This experiment confirms the mechanism: averaged orthogonal adapters
   produce noise, whether used additively or contrastively.

3. **Post-hoc extraction from unchosen adapters:** The value of unchosen adapters in our
   architecture is exactly zero for the current query. Better routing (Finding #243's conclusion)
   is the only viable path.

## What Remains Open

1. **Composition-aware routing training:** L-MoE (2510.17898) and RDLC approaches train
   routers with composition quality as the objective, not domain classification. This addresses
   the disease identified in Finding #243 (BCE loss induces sparse activation).

2. **Generation quality test (P0 existential):** Finding #238 showed K=1 routing already
   produces behavioral improvements for math (+700%) and code (+49%). The system may be
   deployable without multi-adapter composition. This is the most important open question.

3. **Contrastive training (not decoding):** Contrastive adapter training (learning representations
   that contrast between adapters during training) is structurally different from contrastive
   decoding (combining pre-trained adapter outputs at inference). The former modifies the
   adapters' subspaces; the latter assumes fixed subspaces.
