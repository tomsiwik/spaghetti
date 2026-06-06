# Cohere Tiny Aya: Parallel Transformer Blocks

**Source:** https://huggingface.co/CohereLabs/tiny-aya-base
**From-scratch implementation:** `references/LLMs-from-scratch/ch05/15_tiny-aya/`

**Key Insight:** Uses parallel transformer blocks where attention and MLP compute
from the SAME normalized input in parallel, rather than sequentially. This is
architecturally different from standard pre-norm residual connections.

```
Standard:   x → norm → attn → + → norm → mlp → +
Parallel:   x → norm → [attn, mlp] → + (both from same norm output)
```

## Novel Techniques
- **Parallel attention + MLP**: reduces sequential dependency, potentially
  faster training/inference on parallel hardware
- **Dropped QK-Norm**: removed QK normalization for better long-context performance
  (opposite of Qwen3.5 which uses QK-Norm)

## Relevance to our work
- Parallel blocks change where capsule groups would be injected — the MLP
  runs alongside attention, not after it
- At 3B params, this is a practical baseline for small model composition
- From-scratch implementation already available in LLMs-from-scratch repo
