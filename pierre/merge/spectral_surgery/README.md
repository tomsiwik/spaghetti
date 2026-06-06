# Spectral Surgery (MLX)

MLX implementation of "Spectral Surgery: Training-Free Refinement of LoRA via Gradient-Guided Singular Value Reweighting" (Tian et al., 2026, arxiv 2603.03995).

## What it does

Given a **trained LoRA adapter**, Spectral Surgery improves it *without retraining* by:

1. **Decompose** -- SVD of the LoRA delta: `DeltaW = U Sigma V^T`
2. **Estimate** -- per-component sensitivity via gradient projections on a small calibration set: `g_k = u_k^T G v_k`
3. **Reweight** -- adjust singular values under magnitude/energy constraints while keeping U, V fixed

Only ~1000 scalar coefficients are modified (r scalars per edited module, default r=16, editing o_proj + down_proj across all layers).

## Usage

```python
from pierre.merge.spectral_surgery.src.surgery import spectral_surgery_pipeline, load_config

config = load_config("spectral_surgery/configs/base.yaml")

# Collect LoRA modules from your model
lora_modules = {
    "model.layers.0.self_attn.o_proj": {
        "lora_B": layer.self_attn.o_proj.lora_B,
        "lora_A": layer.self_attn.o_proj.lora_A,
    },
    # ... for all o_proj and down_proj modules
}

# Prepare calibration data (small set, ~128 examples)
calib_tokens = [...]   # list of (batch, seq_len) mx.arrays
calib_labels = [...]   # list of (batch, seq_len) mx.arrays, -100 for masked positions

# Run surgery
edited = spectral_surgery_pipeline(
    model=model,
    lora_modules=lora_modules,
    calibration_tokens=calib_tokens,
    calibration_labels=calib_labels,
    loss_fn=your_loss_fn,
    config=config,
)

# Apply edited factors back
for name, (B_new, A_new) in edited.items():
    # set lora_B and lora_A on the module
    ...
```

## Editing policies

| Policy | Description |
|--------|-------------|
| `smooth_abs` | Continuous sigmoid-gated reweighting (default, safest) |
| `abs_select` | Hard three-level gating by sensitivity rank |
| `grad_direction` | Signed multiplicative update (highest gains, highest risk) |
| `random_index` | Matched-random control baseline |

## Key paper results (Table 2)

- Llama-3.1-8B CSQA: +4.4 points (grad_direction)
- Qwen3-8B HumanEval: +2.4 pass@1 (grad_direction)
- smooth_abs is the robust default across all tasks

## Configuration

See `configs/base.yaml` for all parameters with documentation.
