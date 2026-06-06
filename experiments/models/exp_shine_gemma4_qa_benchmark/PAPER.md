# SHINE S4: Document QA Without Context in Prompt (vs ICL)

## Summary

First behavioral test of the SHINE pipeline: encode document → generate
session LoRA → answer questions WITHOUT document in prompt. Architecture:
S2 + multi-projection q+v+o (no meta LoRA). Result: **KILLED**. CE ratio
of 0.0804 (92% reduction) is NOT document encoding — it's a universal LM
shift. The centroid trap (cos=0.9986) makes all generated LoRA identical,
producing training passage fragments regardless of input document.

## Architecture

| Component | Config |
|-----------|--------|
| Base model | Gemma 4 E4B 4-bit (42 layers, 2560 hidden) |
| Memory extraction | Pre-cached (S2 approach, no meta LoRA) |
| M2P | dim=128, 2 blocks, 4 heads, 8.2M params |
| M2P output | q_proj + v_proj + o_proj LoRA (rank 2 each) |
| Training | 1000 steps, Adam lr=3e-4, reconstruction loss |
| QA eval | 7 documents, 21 questions, greedy decode |

## Prediction vs Measurement

| ID | Prediction | Expected | Measured | Result |
|----|-----------|----------|----------|--------|
| P1 | F1 < 10% (centroid blocks encoding) | < 0.10 | **0.006** | CONFIRMED |
| P2 | Factual F1 ≈ ICL F1 (base knows facts) | close | **0.015 vs 0.196** | REFUTED |
| P3 | Detail F1 << ICL F1 (novel facts need I(ΔW;D)>0) | << | **0.000 vs 0.196** | CONFIRMED |
| P4 | Adapter generation < 5s | < 5s | **0.133s** | CONFIRMED |
| P5 | CE ratio < 0.20 | < 0.20 | **0.0804** | CONFIRMED |

## Kill Criteria

| ID | Criterion | Measured | Result |
|----|-----------|----------|--------|
| K1261 | QA F1 > 30% | 0.006 (0.6%) | **FAIL** |
| K1262 | QA F1 >= 50% of ICL | 0.029 (2.9%) | **FAIL** |
| K1263 | Adapter gen < 5s | 0.133s | **PASS** |

## Key Results

### 1. CE Ratio is Misleading (0.0804 = 92% Reduction, But F1 = 0.6%)

The M2P achieves a CE ratio of 0.0804 on test chunks — even BETTER than
S2's 0.134 and S3's 0.151. Training loss drops from 13.16 to 0.62 (95.3%).
By all CE metrics, this is the best SHINE model yet.

But QA F1 is 0.006 (0.6%). The generated text is incoherent fragments from
training passages: "4 BC, his adopted heir", "28 when he noticed", "2017 BC.
The self-produce the membrane". The LoRA is NOT encoding document knowledge.

**This validates Theorem 2 (MATH.md):** CE ratio and QA ability are orthogonal.
The optimization found the centroid LoRA that minimizes average CE across ALL
training passages simultaneously, not a document-specific encoding.

### 2. Centroid Trap Confirmed at cos=0.9986

Pairwise cosine between LoRA generated from 7 completely different documents:
- Mean: 0.9986
- Max: 0.9994

This is WORSE than S3 (0.988) and comparable to S2 (0.998). Without the
diversity regularizer from S3 (which we dropped), the centroid trap is deeper.
All 7 documents produce effectively identical LoRA weights.

### 3. The Universal LoRA Explanation

The centroid LoRA acts as a "language model improvement patch" — it shifts the
model's distribution toward the training passage distribution. This explains:
- CE ratio < 1 on ANY text (including unseen test chunks): the model is better
  at predicting "typical English prose" after applying the LoRA
- F1 ≈ 0 on QA: the LoRA doesn't encode ANY document-specific information
- Gibberish output: the LoRA strongly biases toward training passage tokens,
  overriding the question-answering mode

### 4. ICL Works, SHINE Doesn't

ICL baseline (document + question in prompt) achieves F1 = 0.196 (19.6%).
This is modest but functional — the model can extract answers from passages
when given the text directly. The bottleneck is NOT the base model's QA
ability, it's the LoRA's failure to encode the document.

Notable ICL successes:
- "72 flights" (Ingenuity): F1 = 0.667
- "150 degrees Celsius" (Maillard): F1 = 0.667
- "2.2 percent caffeine" (Robusta): F1 = 0.667
- "1000 nucleotides per second" (E. coli): F1 = 0.444

### 5. P2 Refuted: Base Model Can't Answer Either

Predicted that factual F1 would be close to ICL F1 (base knows these facts).
Actually: no-adapter F1 = 0.002, ICL F1 = 0.196. The base model struggles
with the QA format even for well-known facts (Napoleon, Corsica, helicase).

This is likely a format issue: the 4-bit quantized model generates `<turn|>`
tokens and Japanese text instead of concise English answers. The instruct
formatting may not be optimal for this model.

### 6. Adapter Generation is Fast (K1263 PASS)

Mean adapter generation time: 0.133s (memory extraction + M2P forward).
This is well under the 5s threshold and comparable to SHINE paper's 0.3s
on A100. The architecture is fast — the problem is what it generates.

## Timing

| Phase | Time |
|-------|------|
| Model load | ~5s |
| Memory extraction (50 chunks) | 4.7s |
| M2P build | <1s |
| Training (1000 steps) | 388.3s |
| CE evaluation | ~10s |
| QA evaluation (21 questions × 3 conditions) | ~220s |
| Context specificity | ~5s |
| **Total** | **627.5s** |

## Impossibility Structure

### Why CE reduction ≠ Document encoding

For a session LoRA to encode document D, the M2P must map distinct memory
states to distinct LoRA: ΔW(D_1) ≠ ΔW(D_2). The centroid trap prevents this:

1. **Loss landscape**: The NTP loss has a single deep basin at ΔW_centroid
   that minimizes average CE across all training passages
2. **No contrastive signal**: The reconstruction loss only says "make CE low
   for this passage" — it doesn't say "make CE high for OTHER passages"
3. **Data homogeneity**: 8 similar English prose passages don't provide enough
   distributional diversity to create separate basins

### What would fix it

The disease is **lack of contrastive learning**. The symptoms are:
- Centroid trap (cos > 0.99)
- CE reduction without QA ability
- Identical LoRA for different documents

The SIGReg question: "What is the optimal M2P output distribution such that
centroid collapse is geometrically impossible?"

Answer: **InfoNCE contrastive loss** (arXiv:1807.03748). If the loss explicitly
requires that ΔW(D_i) produces low CE on D_i AND high CE on D_j≠i, then the
centroid is no longer a minimum. This is the same insight as SIGReg (uniform
distribution on the hypersphere prevents collapse) applied to the LoRA space.

Requirements:
1. Large diverse document corpus (100+ documents, not 8)
2. Contrastive loss: CE(D_i | ΔW(D_i)) - log Σ_j exp(-CE(D_j | ΔW(D_i)))
3. Each step evaluates ΔW on both positive and negative documents
4. This 2-4x the compute per step but attacks the root cause

Alternative: abandon document-specific encoding entirely. The "universal LoRA"
IS useful — it's a 92% CE reduction. It just doesn't do what SHINE promises.
Frame it as "generic context enhancement" rather than "session adapter".

## Status: KILLED

K1261 and K1262 fail catastrophically (F1 = 0.6%, SHINE/ICL = 2.9%).
The centroid trap makes session adapters non-functional for QA.
CE metrics are confirmed misleading for behavioral evaluation.

## References

- arXiv:2602.06358 (SHINE) — M2P architecture, 63.6% SQuAD F1
- arXiv:1807.03748 (CPC/InfoNCE) — Contrastive predictive coding
- Finding #484 — S2: CE ratio 0.134, centroid trap cos=0.998
- S3 PAPER.md — Multi-projection 7.7x validated, meta LoRA killed
- Finding #480 — v_proj + o_proj format priors
- Finding #345 — Algebraic proof of centroid trap
