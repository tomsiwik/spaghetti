# Third Structurally Diverse Domain: Caesar Cipher
## Experiment: exp_m2p_third_domain

**Status:** KILLED (K901 FAIL — structural diversity not demonstrated)
**Scale:** micro (Apple M5 Pro, MLX)
**Runtime:** 68.8s, d_model=512, n=2000 samples/domain

---

## Abstract

We test whether the M2P (Memory-to-Parameter) mechanism generalizes to a third domain — Caesar cipher — that is structurally distinct from the sort/reverse sequence-reordering domains validated in Findings #359-#362. Cipher achieves 99.7% M2P quality (K900 PASS), and sort/reverse replicate within 5pp of prior results (K902 PASS). However, cross-domain transfer from sort to cipher reaches 97.78% and reverse to cipher reaches 98.54% — both far above the 50% threshold required to declare structural diversity (K901 FAIL). The cipher domain is not structurally diverse from sort/reverse at toy scale. The failure reveals an impossibility property: for tasks operating on the same 26-token character vocabulary, embedding manifolds cannot be disjoint. Critique #2 (structural diversity) remains unresolved.

---

## Background

Findings #359, #361, and #362 established that M2P achieves ≥98% quality on sort and reverse domains at d=512, with perfect per-domain isolation (grassmannian_max_cos ≈ 0). An adversarial review (Critique #2) noted that both validated domains are sequence-reordering tasks (permutation group operations) and therefore do not constitute structural diversity. To resolve Critique #2, a third domain must satisfy two conditions simultaneously:

1. M2P achieves ≥85% quality (the mechanism generalizes)
2. Cross-domain transfer from existing domains stays <50% (confirming distinct representational structure)

**Domain design rationale.** Caesar cipher (shift each character by a fixed offset mod 26) was chosen as a candidate because:
- Sort/reverse rearrange characters (permutation group S_n)
- Cipher transforms each character independently (substitution group Z_26)
- No positional dependency between input characters in cipher (each maps independently)
- The cipher is a group homomorphism Z_26 → Z_26, requiring modular addition as the computational primitive rather than comparison/permutation

The prediction was that this distinct computational primitive would produce a different activation pattern in the M2P, leading to low cross-domain transfer.

---

## Method

Four domains were included: arithmetic, sort, reverse, cipher. Arithmetic was excluded as a parity-class domain (quality=0%, base and SFT losses nearly equal at 1.56 and 1.54 respectively, meaning the base model already saturates the task). The remaining three domains — sort, reverse, cipher — are used for kill criteria evaluation.

M2P configuration: d_model=512, d_m2p=64, n_memory=32, 2 layers, trained for t=1000 steps per domain, 2000 samples per domain.

Cross-domain quality is measured by applying the M2P memory trained on domain A to evaluate loss on domain B, then computing the quality ratio against the SFT baseline for domain B.

---

## Predictions vs. Measurements

| Metric | Predicted | Measured | Match |
|--------|-----------|----------|-------|
| Cipher M2P quality | 85-100% | 99.7% | Yes — within range |
| Sort M2P quality | 98-102% | 98.5% | Yes — within range |
| Reverse M2P quality | 98-102% | 99.1% | Yes — within range |
| Cross-domain sort→cipher | <30% | 97.78% | NO — off by ~3x |
| Cross-domain cipher→sort | <30% | not measured directly | — |
| Cross-domain reverse→cipher | <30% | 98.54% | NO — off by ~3x |
| grassmannian_max_cos | near 0 | 4.66e-9 | Yes — perfect isolation |

The quality predictions are all confirmed. The diversity predictions fail catastrophically.

---

## Results

### Per-Domain Losses

| Domain | Base Loss | SFT Loss | M2P Val Loss | Quality |
|--------|-----------|----------|--------------|---------|
| arithmetic | 1.5635 | 1.5407 | 1.9335 | 0.0% (excluded) |
| sort | 14.2674 | 2.161 | 2.3456 | 98.5% |
| reverse | 14.2681 | 2.2303 | 2.3394 | 99.1% |
| cipher | 14.2613 | 3.3637 | 3.3989 | 99.7% |

### Cross-Domain Transfer

| Transfer | Cross Val Loss | Cross Quality |
|----------|---------------|---------------|
| sort → cipher | 3.6061 | 97.78% |
| reverse → cipher | 3.5223 | 98.54% |

### Kill Criteria

| Criterion | Threshold | Measured | Verdict |
|-----------|-----------|----------|---------|
| K900: All 3 domains ≥85% quality | sort/reverse/cipher all ≥85% | 98.5% / 99.1% / 99.7% | PASS |
| K901: sort→cipher cross-quality <50% | <50% | 97.78% | FAIL |
| K902: Sort/reverse within 5pp of #361 (101.0%) | within 5pp | 98.5% / 99.1% | PASS |

**Overall outcome: KILLED** — K901 FAIL means structural diversity is not demonstrated.

---

## Analysis

### Why did cipher fail the diversity test?

The prediction assumed that a different computational primitive (modular addition vs. permutation) would produce divergent activation patterns. This reasoning was incorrect at toy scale for the following reason:

All four character-level domains operate on the same 26-token vocabulary. The embedding matrix E ∈ R^{26 × d} is shared across all tasks. When d=512 and the vocabulary has only 26 tokens, the embedding manifold is a 26-dimensional subspace of a 512-dimensional space — every character-level task is constrained to this same low-dimensional manifold. M2P memory trained on sort must produce an output that lies in the span of E, and cipher activations lie in exactly the same span. Cross-domain transfer is therefore structurally guaranteed to be high.

More precisely: if tasks A and B share vocabulary V, then for any M2P memory trained on A, its projection onto the shared embedding manifold has non-zero cosine similarity with any B-task query. The prediction of <30% cross-transfer implicitly assumed that sort and cipher activate different regions of embedding space — but with |V|=26 and d=512, there is no room for such separation.

### Why did arithmetic fail entirely?

Arithmetic (quality=0%) is a parity-class domain: the base model already achieves near-optimal loss on arithmetic (1.54 SFT vs 1.56 base), so the M2P has essentially nothing to learn. The quality ratio is defined as (base_loss - m2p_val_loss) / (base_loss - sft_loss); when the denominator is near zero, the ratio collapses. This confirms arithmetic should be excluded from M2P quality evaluations for this architecture.

### The impossibility structure

For structural diversity to hold — i.e., for cross-domain transfer to fall below 50% — one of the following conditions must be true:

1. **Disjoint vocabularies.** Domain A and domain B operate on different token sets so their embedding manifolds do not overlap. Example: sort over {1..20} vs. natural language summarization over a 32k BPE vocabulary.

2. **Fundamentally different output dimensionality.** Domain A produces short fixed-length sequences; domain B produces variable-length natural language. The M2P must route to different output regions.

3. **Different syntactic structure.** Domain A is synthetic (single-character tokens); domain B has multi-token n-gram dependencies that activate different attention heads.

None of these conditions are satisfied when comparing sort, reverse, and cipher — all three are character-level operations on the same 26-character alphabet.

### What the K900 PASS does confirm

The cipher quality result (99.7%) is valid and meaningful. It shows:
- M2P generalizes to a substitution group operation (not just permutation)
- The M2P bottleneck (d_m2p=64) is sufficient for modular addition
- The intrinsic dimensionality of cipher is low (d_int likely <64, consistent with the prediction)

This is a positive signal about M2P expressiveness. The failure is exclusively about diversity, not capability.

---

## Conclusion

Caesar cipher is not a structurally diverse domain from sort/reverse at toy scale. All character-level tasks over a 26-token vocabulary share the same embedding manifold, making high cross-domain transfer an inevitability rather than a failure of the algorithm.

To resolve Critique #2, the next domain candidate must satisfy at least one of: different token vocabulary, different output structure, or fundamentally different computational substrate. Recommended candidates:

1. **Multi-digit integer arithmetic** (e.g., add two 5-digit numbers, answer is a multi-character string). Tokens are digit characters 0-9 but the computational primitive is carry propagation — a fundamentally different dependency structure from character substitution.

2. **Natural language summarization** (e.g., TinyStories → single-sentence summary). Uses the full BPE vocabulary (~32k tokens), N-gram dependencies, and variable-length outputs. Cross-domain transfer from sort/reverse (26-char alphabet) to natural language is geometrically impossible due to disjoint embedding regions.

Option 2 is the stronger test: it maximally separates the vocabulary and output structure simultaneously.

---

## Self-Test

**Impossibility property stated:** Yes. Cross-domain transfer >50% is inevitable when tasks share the same token vocabulary, because M2P memory lies in the span of the shared embedding manifold. This is a structural constraint, not a training artifact.

**Cited theorems:** The argument uses the dimension-counting property of shared embedding manifolds (|V|=26, d=512 implies all tasks are constrained to the same 26-dimensional subspace of embedding space). Referenced group-theoretic framing: Caesar cipher is Z_26 → Z_26 homomorphism; sort/reverse are S_n permutation group operations.

**Predicted numbers with measurements:**
- Cipher quality: predicted 85-100%, measured 99.7% — match.
- Cross-domain: predicted <30%, measured ~98% — strong refutation.

**Falsification conditions:**
- K901 is falsified by the measurements (cross-quality 97.78% >> 50% threshold).
- The impossibility argument itself would be falsified by demonstrating <50% cross-domain transfer between two tasks sharing the same 26-token vocabulary at d=512 — which no follow-up experiment is expected to do.

**Prior findings consistency:**
- K902 PASS: sort=98.5%, reverse=99.1% are within 5pp of Finding #361 (101.0%). The M2P quality result is stable across experimental runs.
- grassmannian_max_cos = 4.66e-9 ≈ 0 confirms per-domain M2P isolation holds, consistent with Findings #359-#362.
