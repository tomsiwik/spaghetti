# DC-Merge (MLX)

MLX implementation of **DC-Merge: Improving Model Merging with Directional Consistency**
([arXiv 2603.06242](https://arxiv.org/abs/2603.06242)).

DC-Merge improves multi-task model merging by:
1. Smoothing the singular value energy distribution of task vectors so weak-but-important
   knowledge components survive merging.
2. Constructing a shared orthonormal "cover space" that captures the directional geometry
   of all task vectors.
3. Merging in that aligned space, then projecting back with a structural mask.

## Usage

```python
import mlx.core as mx
from pierre.merge.dc_merge.src.merge import dc_merge

# Suppose you have base weights and fine-tuned weights for T tasks
# task_vectors[i] = W_finetuned_i - W_base, each shape (m, n)
task_vectors = [ft_weights[i] - base_weights for i in range(T)]

# Merge a single weight matrix
merged = dc_merge(
    task_vectors=task_vectors,
    rank=16,              # SVD truncation rank (= LoRA rank for LoRA)
    smoothing="average",  # or "linear" or "none"
    merge_method="ties",  # or "ta"
    alpha=1.0,            # rescaling coefficient
    base_weights=base_weights,
)
```

### Merge full model state dicts

```python
from pierre.merge.dc_merge.src.merge import dc_merge_state_dicts

merged_state = dc_merge_state_dicts(
    base_state=base_model_weights,    # dict[str, mx.array]
    task_states=list_of_ft_weights,   # list of dicts
    rank=16,
    smoothing="average",
    merge_method="ties",
    alpha=1.0,
)
```

### Use YAML config

```python
from pierre.merge.dc_merge.src.merge import dc_merge_from_config

merged = dc_merge_from_config(
    task_vectors=task_vectors,
    config_path="dc_merge/configs/base.yaml",
    base_weights=base_weights,
)
```

## Configuration

See `configs/base.yaml` for all parameters and their documentation.

## Files

```
dc_merge/
  configs/base.yaml          # Default hyperparameters
  src/
    __init__.py
    model.py                 # SVD, energy smoothing, cover space, projection, DirSim
    merge.py                 # Full DC-Merge pipeline, TIES, state dict merging
  REPRODUCTION_NOTES.md      # Equation-to-code mapping and ambiguity notes
  README.md
  requirements.txt
```
