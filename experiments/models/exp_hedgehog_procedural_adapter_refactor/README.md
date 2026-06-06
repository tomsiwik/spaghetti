# exp_hedgehog_procedural_adapter_refactor

## Paper
Stage 1 of [arXiv:2604.14191](https://arxiv.org/abs/2604.14191) (Moudgil/Apple+MILA, 2026-04-01). Parallel to `exp_hedgehog_behavior_adapter_politeness` but the axis is **procedural knowledge** (refactoring) rather than behavior (politeness).

## Teacher corpus
Martin Fowler, *Refactoring: Improving the Design of Existing Code*, 2nd ed.:
- Extract 66 catalog entries (Extract Function, Inline Variable, Replace Conditional with Polymorphism, ...)
- For each entry: 3-5 before/after code pairs with the book's narrative explanation
- Total: ~200-300 refactor examples across languages (Java, JavaScript, TypeScript)

Teacher = larger model (Claude 3.7 / Gemma 4 26B-A4B) with the catalog entry **in context** performing the refactor.
Student = Gemma 4 E4B + rank-8 LoRA, sees only the code without the catalog entry.

## Reference implementations
- `HazyResearch/lolcats` — Hedgehog φ_MLP feature map
- No official code for 2604.14191

## MLX translation notes
Same attention-output capture pattern as `exp_hedgehog_behavior_adapter_politeness`. One difference: prompts are code blocks, longer context — set `SEQLEN=1024` minimum.

## Dataset acquisition
The Fowler book is copyrighted. Do NOT redistribute. Options:
1. Use `openai/humaneval-refactor` / `refactoring-challenges` on HuggingFace as proxy dataset
2. Synthesize from public Fowler catalog summaries (fowler.com lists refactor names + one-line descriptions; expand with LLM-generated examples)
3. Draw from Git history of large OSS repos (GitHub Archive): commits labeled "refactor" give before/after pairs

Option 2 + 3 is recommended for reproducibility.

## Quick start
```bash
experiment claim <worker-id> --id exp_hedgehog_procedural_adapter_refactor
experiment run exp_hedgehog_procedural_adapter_refactor
```
