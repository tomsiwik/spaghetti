# LLMs-from-scratch (Sebastian Raschka)

**Source:** https://github.com/rasbt/LLMs-from-scratch
**Type:** Git submodule at `references/LLMs-from-scratch/`

**Key Insight:** Complete educational resource for building LLMs from the ground
up. Every concept explained clearly with working code. Invaluable as a reference
for understanding the architectural building blocks we compose with.

**Relevant chapters:**
- `ch04/` — GPT from scratch (our `minigpt.py` baseline is similar)
- `ch05/16_qwen3.5/` — Qwen3.5 0.8B reimplementation (hybrid attention, GQA, RoPE)
- `ch05/` — Various model implementations (Llama, Phi, etc.)
- `ch06/` — Fine-tuning (LoRA, etc. — relevant to our capsule/adapter approach)
- `ch07/` — Preference tuning (DPO, RLHF)

**How to use:**
When implementing a new mechanism or debugging an architectural question,
check this repo first. The code is clean, well-commented, and educational.
Adapt existing implementations rather than reinventing.
