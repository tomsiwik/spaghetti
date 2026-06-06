# Reproduction Notes: Brainstacks

Paper: [Brainstacks: Cross-Domain Cognitive Capabilities via Frozen MoE-LoRA Stacks for Continual LLM Learning](https://arxiv.org/abs/2604.01152)

Official code: https://github.com/achelousace/brainstacks

## Implementation Framework

This implementation uses **MLX** (Apple Silicon native) instead of the paper's PyTorch + bitsandbytes.
Key differences from paper:
- No QLoRA 4-bit quantization via bitsandbytes — use MLX native quantization instead
- No CPU/GPU offloading shuttle — Apple Silicon unified memory means frozen stacks stay in same memory pool
- `mx.stop_gradient()` replaces `torch.no_grad()` for frozen stack computation
- SVD via `mx.linalg.svd` instead of `torch.svd_lowrank` (no randomized variant in MLX)

## Unspecified Choices

| Item | Our Choice | Alternatives | Impact |
|------|-----------|-------------|--------|
| Router W_r initialization | MLX default (Glorot uniform) | Kaiming, normal(0.01) | Low — router converges quickly |
| Router W_n initialization | MLX default (Glorot uniform) | Zeros, small normal | Low |
| Router bias | No bias | With bias | Low |
| Meta-router dropout rate | 0.1 | 0.05, 0.2 | Medium — affects router generalization |
| Meta-router LR | 1e-3 | 1e-4, 5e-4 | Medium — router training stability |
| Meta-router optimizer | AdamW | Adam, SGD | Low |
| Meta-router confidence margin | 0.1 | 0.05, 0.2 | Medium — pushes predictions toward 0/1 |
| SFT optimizer | AdamW | Adam | Low |
| SFT weight decay | 0.01 | 0.0, 0.1 | Low |
| SFT warmup steps | 0 | Linear warmup | Low-Medium |
| Frozen stack precision | float16 | bfloat16 | Low on Apple Silicon (both fast) |

## Scope Decisions

### Implemented
- MoELoRADelta module (§3.1) — core architecture
- StackedMoELoRALayer (§3.2) — additive composition
- Inner loop residual boosting (§3.3, Algorithm 1) — core training algorithm
- Outer loop continual learning (§3.4, Algorithm 2) — core training algorithm
- Null-space projection (§3.5) — zero-forgetting mechanism
- Meta-router with outcome-based sigmoid gating (§3.6) — composition mechanism
- Loss functions (§3.1 aux loss, §3.6 router BCE)

### Intentionally Excluded
- PSN pretraining experiment (§6.1) — boundary experiment, not core
- Per-domain RL / DPO / GRPO (§6.2) — optional post-SFT step
- Dataset decontamination subsystem — engineering detail
- Specific benchmark evaluations — validation infrastructure
- BestStackCallback weight snapshot/restore — simplified in scaffold

### Would Need for Full Reproduction
- Base model: TinyLlama-1.1B or Gemma 3 12B IT loaded via mlx-lm
- Datasets: alpaca, GSM8K, python_code_18k, medalpaca, etc.
- Full null-space delta collection with forward hooks
- Meta-router outcome discovery (greedy domain combination search)
