# Fisher-Rao Karcher Mean Merging (MLX)

MLX implementation of "Functionality-Oriented LLM Merging on the Fisher-Rao Manifold" (arXiv:2603.04972).

Merges N fine-tuned models by computing a weighted Karcher mean on the unit sphere per parameter block, then rescaling by mean source norm. For N=2, this reduces to SLERP. For N>2, it uses fixed-point iteration on S^(d-1).

## Usage

```python
import mlx.core as mx
from pierre.merge.fisher_rao_merging.src.merge import merge_parameters

# params_list: list of N flat dicts {param_name: mx.array}
merged = merge_parameters(
    params_list=[model1_params, model2_params, model3_params],
    weights=[0.4, 0.3, 0.3],  # optional, defaults to equal
)
```

For nested state dicts (e.g., from `mlx.nn.Module`):

```python
from pierre.merge.fisher_rao_merging.src.merge import merge_state_dicts

merged_state = merge_state_dicts(
    [model1.parameters(), model2.parameters(), model3.parameters()],
    weights=[1/3, 1/3, 1/3],
)
model.load_weights(list(mx.utils.tree_flatten(merged_state)))
```

## Files

```
fisher_rao_merging/
  configs/base.yaml      # All hyperparameters with section citations
  src/
    __init__.py
    model.py             # Spherical geometry: normalize, log/exp maps, Karcher mean, SLERP
    merge.py             # Merging pipeline: blockwise normalize -> Karcher -> rescale
    loss.py              # Diagnostics: FR proxy loss, activation variance, effective rank
  REPRODUCTION_NOTES.md  # Unspecified choices, coverage gaps
  requirements.txt       # mlx, pyyaml
```

## Configuration

Edit `configs/base.yaml`. All parameters cite their paper section. Key settings:

- `step_size`: eta for fixed-point iteration (default 1.0)
- `max_iterations`: Karcher convergence limit (default 100)
- `norm_rescale`: "mean" (paper default), "median", or "max"

## Requirements

- Apple Silicon Mac with MLX >= 0.22.0
- Python 3.10+
