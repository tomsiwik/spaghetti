# exp_hedgehog_behavior_adapter_politeness

## Paper
**Attention to Mamba: A Recipe for Cross-Architecture Distillation** (Moudgil et al., Apple+MILA, [arXiv:2604.14191](https://arxiv.org/abs/2604.14191), 2026-04-01). We use only Stage 1 (Hedgehog per-layer cos-sim distillation), not Stage 2 (Mamba conversion).

Underlying primitive: Zhang et al., *The Hedgehog & the Porcupine*, [arXiv:2402.04347](https://arxiv.org/abs/2402.04347), ICLR 2024.

## Reference implementations (PyTorch)
- `HazyResearch/lolcats` — Hedgehog φ_MLP feature map + attention-output matching (`src/model/linear_attention/*`)
- No public repo yet for 2604.14191 (3 weeks old)

## MLX translation
See `run_experiment.py`. Key differences from PyTorch reference:
- `mx.fast.scaled_dot_product_attention` replaces manual softmax attention where possible
- Attention-output capture via hook-equivalent: subclass the attention module, stash pre-output tensor on the module instance, `mx.eval` it per step
- `nn.value_and_grad` pattern; `mx.clear_cache()` between teacher/student forward to cap peak memory on M5 Pro 48GB

## Prerequisites
- Invoke `/mlx-dev` and `/fast-mlx` skills BEFORE writing code (see ralph guardrail)
- Base model: `mlx-community/gemma-4-e4b-it-4bit`
- Adapter: rank-8 LoRA on `v_proj + o_proj` (per Pierre F#627)

## Quick start
```bash
experiment claim <worker-id> --id exp_hedgehog_behavior_adapter_politeness
# researcher hat writes MATH.md details + run_experiment.py
experiment run exp_hedgehog_behavior_adapter_politeness
```
