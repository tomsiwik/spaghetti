# MTP Composition: Research Digest

## Hypothesis

Multi-Token Prediction (MTP) training forces capsule groups to learn more
coherent multi-step patterns, improving composition quality because experts
that predict multiple tokens must capture richer structure.

## Verdict: KILL (criterion 2)

MTP does NOT improve composed model quality. Composed MTP models are
0.8-2.1% WORSE in absolute loss than composed NTP models. MTP does not
hurt the composition gap (criterion 1 passes), but provides no benefit
for the composed model's final quality (criterion 2 kills).

## What This Model Is

MTPCapsuleMoEGPT extends CapsuleMoEGPT with D-1 auxiliary prediction
heads following the DeepSeek-V3 MTP architecture. Each MTP head at
depth k sequentially predicts token t+k+1 from a chained hidden state:

```
h_k = RMSNorm(W_k @ h_{k-1} + emb(token_{t+k}))
logits_k = lm_head(h_k)       -- shared lm_head (parameter-efficient)
```

The total training loss combines NTP + weighted MTP:

```
L = L_ntp + 0.3 * mean(L_1, ..., L_{D-1}) + L_balance
```

At inference, only NTP logits are used. MTP heads are discarded.

## Lineage in the Arena

```
gpt (dense baseline)
 |-- capsule_moe (routed capsule groups)
      |-- mtp_capsule_moe (+ MTP training heads) <-- THIS
```

## Key References

- DeepSeek-V3 (2024): MTP as auxiliary training objective, sequential
  chaining of prediction heads, shared lm_head across depths
- Qwen3-Coder-Next (2026): MTP for both training and inference
  (speculative decoding), 512 experts with top-10 routing

## Protocol

Follows the established capsule_moe composition protocol:

1. Pretrain shared base on all data (300 steps)
2. Fine-tune only capsule groups per domain (freeze attention) -- 300 steps
3. Compose: concatenate domain groups, double top-k
4. Calibrate: train only router on mixed data (100 steps)
5. Evaluate: NTP-only val loss on per-domain val sets

MTP affects step 2 only (changes fine-tuning objective). Identical protocol
across all conditions enables clean comparison.

## Empirical Results

### Main Results (3 seeds, mean +/- std)

| Depth | Single Avg | Joint Avg | Composed Avg | Gap (%) | Std (%) |
|-------|-----------|-----------|-------------|---------|---------|
| 1 (NTP) | 0.4863 | 0.5267 | 0.5141 | -2.40 | 1.19 |
| 2 (MTP-2) | 0.4917 | 0.5352 | 0.5184 | -3.15 | 0.46 |
| 3 (MTP-3) | 0.5006 | 0.5347 | 0.5250 | -1.82 | 0.56 |

### Per-Seed Composition Gaps

| Depth | Seed 42 | Seed 123 | Seed 777 |
|-------|---------|----------|----------|
| 1 (NTP) | -3.76% | -1.87% | -1.57% |
| 2 (MTP-2) | -2.83% | -2.92% | -3.68% |
| 3 (MTP-3) | -2.20% | -1.18% | -2.10% |

### Kill Criteria Evaluation

**Kill 1: MTP-trained groups compose >5% worse than NTP groups**

| Depth | NTP Gap | MTP Gap | Difference | Verdict |
|-------|---------|---------|------------|---------|
| 2 | -2.40% | -3.15% | -0.75pp | PASS |
| 3 | -2.40% | -1.82% | +0.57pp | PASS |

MTP-2 actually shows a slightly SMALLER (better) composition gap than NTP,
though the difference is within noise. MTP-3 shows a slightly larger gap
but well within the 5pp threshold.

**Kill 2: MTP provides <2% quality improvement for composed models**

| Depth | NTP Composed | MTP Composed | Improvement | Verdict |
|-------|-------------|-------------|-------------|---------|
| 2 | 0.5141 | 0.5184 | -0.84% | KILL |
| 3 | 0.5141 | 0.5250 | -2.12% | KILL |

MTP composed models have HIGHER loss (worse quality) than NTP composed models.
The "improvement" is negative -- MTP makes composed quality slightly worse.

## Analysis

### Why MTP does not help at micro scale

1. **Training overhead without benefit**: MTP adds auxiliary loss terms that
   make training harder. The training losses for MTP conditions are 30-45%
   higher than NTP during fine-tuning (e.g., 0.70 vs 0.52). This indicates
   the MTP objective competes with NTP for gradient bandwidth in the capsule
   parameters.

2. **Character-level sequences are too simple**: At character level with
   T=32, predicting t+2 or t+3 is often trivial (common bigrams/trigrams).
   The "richer structure" hypothesis assumes that multi-step prediction
   requires qualitatively different representations -- but at character
   level, next-character prediction already captures the relevant patterns.

3. **MTP increases joint loss, proportionally increases composed loss**: Both
   joint and composed losses increase with MTP depth, but the composition
   gap (relative difference) stays similar. MTP uniformly degrades model
   quality rather than selectively improving composition.

4. **Composition gap is already negative**: The calibrated composition
   protocol already produces composed models that BEAT joint training
   (negative gaps). There is no composition deficit for MTP to fix.

### What is notable

- **MTP-2 reduces gap variance**: Std drops from 1.19% (NTP) to 0.46%
  (MTP-2). MTP-2 produces more consistent composition gaps across seeds.
  This could indicate that MTP does regularize capsule specialization,
  even though it doesn't improve absolute quality.

- **MTP-3 shows diminishing returns**: Both joint and composed quality
  degrade further at depth 3 vs depth 2, consistent with the hypothesis
  that character-level MTP provides limited signal beyond depth 2.

- **Single-domain quality is similar**: Single-domain losses are close
  across all depths (0.486 vs 0.492 vs 0.501), suggesting MTP's cost
  is primarily in the fine-tuning efficiency, not in final representation
  quality.

## Parameter Overhead

| Depth | Params | Overhead |
|-------|--------|----------|
| 1 (NTP) | 202,560 | baseline |
| 2 (MTP-2) | 206,656 | +4,096 (2.0%) |
| 3 (MTP-3) | 210,752 | +8,192 (4.0%) |

The parameter overhead is modest (d^2 per MTP depth). At macro scale
(d=2048), each MTP depth adds 4.2M params -- small relative to a
multi-billion parameter model.

## Micro-Scale Limitations

This experiment deliberately does NOT test:

1. **Token-level MTP**: At subword tokenization, predicting t+2 means
   predicting the next word (not the next character). The structural
   richness of MTP is fundamentally different at token level.

2. **MTP with longer sequences**: T=32 characters limits the MTP
   horizon. At T=2048+ with subword tokens, MTP-3 could capture
   paragraph-level structure that NTP cannot.

3. **MTP interaction with fine-grained routing**: With 512 experts
   (Qwen3-Coder-Next), MTP may force more discriminative routing
   patterns. At G=4 groups, routing is too coarse.

4. **MTP during pretraining**: We only used MTP during fine-tuning.
   DeepSeek-V3 and Qwen3-Coder-Next use MTP during pretraining,
   where it can shape the base model's representations.

5. **Speculative decoding benefit**: MTP's inference-time value
   (speculative decoding for faster generation) is not tested here.
   This is a real production benefit orthogonal to composition.

## What Would Kill This

**Already killed at micro scale** (criterion 2): MTP does not improve
composed model quality at character level with d=64, T=32.

**What would resurrect it at macro scale**:
- Token-level MTP with T >= 2048 showing >2% composed quality improvement
- MTP during pretraining (not just fine-tuning) changing base representations
- MTP interaction with fine-grained routing (512 experts, top-10)

**Definitive kill at macro scale**:
- MTP-pretrained base + MTP-finetuned capsules compose >5% worse than
  NTP-only equivalents on HumanEval/MBPP
- MTP provides <2% quality improvement on any downstream task for composed
  models (not just perplexity)

## Key Takeaway

At micro scale, MTP is a training-time cost that does not translate to
composition benefit. The composition gap is already favorable (-2.4% for
NTP), and MTP cannot improve it further. However, MTP does not HURT
composition either (kill criterion 1 passes), suggesting it is neutral
for the composition mechanism. The real question -- whether MTP improves
composition at token level with real-world data -- remains open and is
the appropriate macro-scale follow-up.
