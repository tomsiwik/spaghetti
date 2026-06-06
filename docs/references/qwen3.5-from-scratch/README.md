# Qwen 3.5 0.8B From Scratch (Sebastian Raschka)

**Source:** https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/16_qwen3.5
**Full repo:** `references/LLMs-from-scratch/` (git submodule)

**Key Insight:** Complete educational reimplementation of Qwen3.5-0.8B showing
the hybrid attention architecture (alternating linear + full attention layers).
Clean, readable code suitable as a baseline for understanding modern LLM internals.

**Architecture highlights:**
- Hybrid attention: 18 linear attention layers + 6 full attention layers (3:1 ratio)
- GQA with gated Q projection (2x output dim for gate)
- RoPE with partial rotary factor (0.25)
- SiLU-gated MLP (gate_proj * up_proj pattern)
- RMSNorm with (1 + weight) scaling and zero-init
- Linear attention via Gated DeltaNet (chunk-based recurrence)

**Relevance to our work:**
- **Baseline model architecture** — our macro experiments use Qwen2.5-Coder-0.5B;
  understanding the Qwen family architecture from scratch is essential
- **SiLU activation** — Qwen uses SiLU not ReLU, directly relevant to
  `exp15_non_relu_pruning` (our micro pruning results assume ReLU hard zeros)
- **Capsule surgery target** — knowing exactly how the MLP layers work
  informs where to inject capsule groups
- **Linear attention** — Qwen3.5's hybrid attention could interact with
  composition differently than pure full attention

**Files:**
- `miniqwen.py` (repo root) — standalone PyTorch reimplementation, copied from this reference
- Original notebook: `references/LLMs-from-scratch/ch05/16_qwen3.5/qwen3.5.ipynb`
- Linear attention helper: `references/LLMs-from-scratch/ch05/16_qwen3.5/qwen3_5_transformers.py`
