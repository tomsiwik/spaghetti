# exp_memento_gemma4_replication

## Paper
**MEMENTO: Teaching LLMs to Manage Their Own Context** (Kontonis et al., Microsoft+UT+OpenAI, [arXiv:2604.09852](https://arxiv.org/abs/2604.09852), 2026-04-10).

## Gate experiment
Replicate MEMENTO 2-stage SFT at Gemma 4 E4B (~4B active) — smaller than the paper's 8B–32B range. If replication succeeds, it unlocks `exp_memento_cross_session_persistence` and `exp_user_adapter_from_memento_distillation`. If it fails, that itself is a publishable finding ("MEMENTO requires ≥8B") — pivot to 26B-A4B base.

## Reference implementation
- **Repo:** [github.com/microsoft/memento](https://github.com/microsoft/memento)
  - `data/` — pipeline that converts raw CoT traces → block-annotated mementos for SFT
  - `vllm/` — vLLM overlay with KV cache block masking (`bash install_overlay.sh`)
- **Dataset:** [huggingface.co/datasets/microsoft/OpenMementos](https://huggingface.co/datasets/microsoft/OpenMementos) — 228K traces, 54% math / 19% code / 27% science, MIT-licensed, SFT-ready
- **Special tokens:** `<|block_start|>`, `<|block_end|>`, `<|summary_start|>`, `<|summary_end|>` (+ `<think>`/`</think>` for thinking mode)
- **Format:** `<|block_start|> reasoning <|block_end|> <|summary_start|> memento <|summary_end|>`
- **Compression target:** 15-25% of original tokens per memento; ~6× trace-level compression (11k → <2k)

## MLX translation notes
1. **No native block-masking in mlx-lm.** The paper's key infra contribution is a vLLM fork with data-dependent KV masking. For Gemma 4 E4B 4-bit MLX, we have two options:
   - **(Path A, simpler)** Use InftyThink-style discard — mask blocks by *re-prompting from summaries only*, losing the implicit KV channel (paper's 15pp AIME24 channel ablation finding says this loses accuracy). This is exactly the ablation K3 tests.
   - **(Path B, faithful)** Implement custom block-mask attention in MLX. Attention mask is `[L, L]` bool; set to `False` for evicted block tokens after their corresponding `<|summary_end|>` token is generated. Apply as `mx.fast.scaled_dot_product_attention(q, k, v, mask=block_mask)`.
   - **Recommended:** Path B for the gate experiment (faithful replication), Path A as control.

2. **Add special tokens:** extend `tokenizer.add_special_tokens({...})` with the 4 boundary tokens; resize embedding matrix.

3. **Training data:** load OpenMementos via `datasets.load_dataset("microsoft/OpenMementos")`. Tokenize with added special tokens. Standard SFT (next-token CE on response only, loss mask on prompt).

4. **Memory:** 4-bit Gemma 4 E4B + SFT + block-masked attention on M5 Pro 48GB. Batch size 1, SEQLEN=4096. Should fit.

## Prerequisites
- Invoke `/mlx-dev` and `/fast-mlx` skills
- Clone github.com/microsoft/memento for reference (don't redistribute vLLM fork — MLX path is separate)
- `uv pip install "datasets>=2.20"` if not already

## Quick start
```bash
experiment claim <worker-id> --id exp_memento_gemma4_replication
experiment run exp_memento_gemma4_replication
```
