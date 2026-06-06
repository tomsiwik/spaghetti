# MTP Composition: Mathematical Foundations

## Notation

| Symbol | Shape/Type | Definition |
|--------|-----------|------------|
| d | scalar | Hidden dimension (d=64 at micro scale) |
| V | scalar | Vocabulary size (V=28 for char-level) |
| T | scalar | Sequence length (T=32) |
| D | scalar | MTP depth (D=1 is standard NTP, D=2,3 are MTP variants) |
| G | scalar | Number of capsule groups per layer (G=4) |
| P | scalar | Capsules per group (P=64) |
| k | scalar | Top-k groups for routing (k=2) |
| L | scalar | Number of transformer layers (L=4) |
| B | scalar | Batch size (B=32) |
| lambda | scalar | MTP loss weight (lambda=0.3) |

## Standard Next-Token Prediction (NTP)

The base model maps tokens to hidden states through the transformer stack:

```
h = Transformer(tokens)   -- h in R^{B x T x d}
logits = lm_head(h)       -- logits in R^{B x T x V}
```

NTP loss at position t predicts token_{t+1}:

```
L_ntp = (1/T) * sum_{t=1}^{T} CE(logits_t, token_{t+1})
```

where CE is cross-entropy loss.

## Multi-Token Prediction (MTP)

Following DeepSeek-V3's MTP formulation, we add D-1 auxiliary prediction
heads that sequentially predict future tokens.

### MTP Module Architecture

For depth k in {1, ..., D-1}, each MTP module computes:

```
h_k = RMSNorm(W_k * h_{k-1} + emb(token_{t+k}))
logits_k = lm_head(h_k)
```

where:
- h_0 = h (the main transformer output)
- W_k in R^{d x d} is a learned projection for depth k
- emb(token_{t+k}) is the token embedding at offset k
- lm_head is **shared** with the main model (parameter-efficient)

The sequential chaining is critical: h_k depends on h_{k-1}, which means
depth-k prediction builds on all previous depths. This forces the model
to learn coherent multi-step structure, not independent predictions.

### MTP Loss

```
L_mtp = lambda * (1/(D-1)) * sum_{k=1}^{D-1} L_k
```

where:

```
L_k = (1/(T-k)) * sum_{t=1}^{T-k} CE(logits_k[t], token_{t+k+1})
```

Note: for depth k, we can only compute loss at positions 1..T-k
(need k additional future tokens as input context).

### Total Training Loss

```
L_total = L_ntp + L_mtp + L_balance
```

where L_balance is the capsule routing balance loss (same as CapsuleMoEGPT).

## Parameter Overhead

Each MTP module adds:
- W_k: d^2 parameters (linear projection)
- RMSNorm: 0 parameters (no learnable params in our implementation)

Total MTP overhead: (D-1) * d^2

### Worked Example at Micro Scale

d=64, D=3:
- MTP overhead: 2 * 64^2 = 8,192 params
- Base CapsuleMoEGPT: ~202K params
- Overhead ratio: 8,192 / 202,000 = 4.1%

The overhead is modest. At macro scale (d=2048, D=2):
- MTP overhead: 1 * 2048^2 = 4.2M params
- This scales as O(d^2) per depth, identical to one attention head's QK projection.

## Hypothesis: Why MTP Should Improve Composition

### Argument

In standard NTP, each capsule group optimizes for:

```
argmin_{A_g, B_g} E[CE(lm_head(B_g @ ReLU(A_g @ h)), token_{t+1})]
```

This objective only requires capturing the **immediate next character**. At
character level, this is often a local pattern (e.g., after "th" -> predict "e"
or "a"). The capsule specialization may be shallow.

With MTP, each capsule group's hidden representation must additionally support:

```
argmin_{A_g, B_g} E[CE(lm_head(MTP_k(B_g @ ReLU(A_g @ h))), token_{t+k+1})]
for all k in 1..D-1
```

This requires the capsule output to encode not just the immediate next token
but a **multi-step plan**. The richer representation may lead to:

1. **More structured specialization**: Capsules learn sequential patterns
   (e.g., common bigrams/trigrams), not just character frequencies
2. **Better compositional alignment**: Multi-step patterns are more domain-
   specific than single-character patterns, leading to less cross-domain
   interference when composing

### Counter-argument

MTP could also:
1. Make training harder (higher loss, worse convergence)
2. Force capsules toward generic multi-step patterns rather than domain-specific
   ones (reducing specialization, hurting composition)
3. Have negligible effect at character level where sequences are short and
   patterns are simple

## Composition Protocol

Composition follows the established capsule_moe protocol:

1. **Pretrain** shared base on all data (attention + capsules + embeddings)
2. **Fine-tune** only capsule groups per domain (freeze attention/embeddings)
3. **Compose**: concatenate domain capsule groups into 2G-group model
4. **Calibrate**: train only router on mixed-domain data (100 steps)

The MTP training signal affects **step 2 only** -- it changes how capsule
groups specialize during domain fine-tuning. Steps 1, 3, 4 are identical.

At evaluation, only NTP logits are used (MTP heads discarded). The hypothesis
is that MTP training produces better-specializing capsules that compose more
cleanly, even though composition uses NTP inference only.

## Computational Cost Analysis

### Training Time

Per-step cost increase from MTP (D depths):
- Main transformer: unchanged
- MTP depth k: one Linear(d,d) + one RMSNorm + one lm_head(d,V)
  = d^2 + d + d*V FLOPs per position
- Total MTP FLOPs: (D-1) * (d^2 + d + d*V) * T * B

At micro scale (d=64, V=28, T=32, B=32, D=3):
- MTP cost per step: 2 * (4096 + 64 + 1792) * 32 * 32 = 12.2M FLOPs
- Main transformer cost per step: ~L * (12*d^2 + 2*d*V) * T * B
  = 4 * (49152 + 3584) * 1024 = 215M FLOPs
- MTP overhead: 12.2/215 = 5.7%

### Inference Time

Zero overhead. MTP modules are not used at inference. The composed model
is identical in architecture to a standard CapsuleMoE.

## Kill Criteria Formalization

Let gap(depth) = (composed_loss(depth) - joint_loss(depth)) / joint_loss(depth) * 100

**Kill 1**: gap(D>1) > gap(1) + 5.0
- MTP makes composition worse by more than 5 percentage points

**Kill 2**: (composed_loss(1) - composed_loss(D>1)) / composed_loss(1) * 100 < 2.0
- MTP does not meaningfully improve composed model quality
- This is the absolute composed loss comparison, not the gap

Both kills must be checked independently. If kill 2 triggers (MTP < 2%
improvement) but kill 1 does not (gap is similar), the conclusion is:
MTP doesn't help composition but doesn't hurt either. If both trigger,
MTP is actively harmful for composition.

## Assumptions

1. **MTP signal propagates through capsule groups**: The MTP loss gradients
   flow through the shared lm_head and the capsule group outputs. Since we
   only fine-tune capsule groups (A, B weights), MTP must create useful
   gradient signal for these specific parameters.

2. **Character-level MTP is meaningful**: At character level, predicting
   t+2 and t+3 tokens captures bigram/trigram patterns. This is less rich
   than token-level MTP in real LLMs (where t+2 might be a whole word
   away), but should still test the mechanism.

3. **Composition protocol isolates the MTP effect**: By using identical
   pretraining, composition, and calibration for NTP and MTP conditions,
   the only variable is the fine-tuning objective. Any difference in
   composition quality is attributable to MTP.

4. **D=3 is sufficient depth**: DeepSeek-V3 uses D=2 (one auxiliary head).
   Testing D=3 provides a dose-response curve. Deeper MTP at micro scale
   may not help due to short sequence lengths (T=32).
