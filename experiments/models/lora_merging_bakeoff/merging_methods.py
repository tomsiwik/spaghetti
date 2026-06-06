"""LoRA delta merging methods: TIES, DARE, DARE-TIES, simple average.

Each method takes a list of N delta dictionaries (keyed by (layer, sublayer))
and returns a single merged delta dictionary.

Delta format: dict[(int, str), mx.array]
  key = (layer_index, 'fc1' | 'fc2')
  value = delta matrix of shape (in_dim, out_dim)

References:
- TIES: Yadav et al., "Resolving Interference When Merging Models", NeurIPS 2023
- DARE: Yu et al., "Language Models are Super Mario", 2023
- Task Arithmetic: Ilharco et al., "Editing Models with Task Arithmetic", ICLR 2023
"""

import mlx.core as mx
import mlx.nn as nn


# ── Delta extraction helpers ──────────────────────────────────────────────────

def extract_deltas(model) -> dict:
    """Extract LoRA deltas from a LoRAGPT model as {(layer, sublayer): delta}."""
    deltas = {}
    for l_idx, layer in enumerate(model.layers):
        for name, fc in [('fc1', layer.mlp.fc1), ('fc2', layer.mlp.fc2)]:
            delta = (fc.alpha / fc.rank) * (fc.A @ fc.B)
            deltas[(l_idx, name)] = delta
    return deltas


# ── Simple Average (Task Arithmetic with lambda = 1/N) ────────────────────────

def merge_simple_average(delta_list: list[dict]) -> dict:
    """Average all deltas element-wise: merged[k] = (1/N) * sum(delta_i[k]).

    This is equivalent to task arithmetic with scaling_coefficient = 1/N.
    """
    N = len(delta_list)
    keys = list(delta_list[0].keys())
    merged = {}
    for k in keys:
        merged[k] = sum(d[k] for d in delta_list) / N
    return merged


# ── TIES-Merging ──────────────────────────────────────────────────────────────

def merge_ties(delta_list: list[dict], density: float = 0.2) -> dict:
    """TIES-Merging: Trim + Elect Sign + Merge.

    Args:
        delta_list: list of N delta dicts
        density: fraction of parameters to keep (top-k% by magnitude)
                 density=0.2 means keep top 20%, trim bottom 80%

    Returns:
        merged delta dict

    Algorithm (per-parameter, applied to flattened delta vectors):
      1. TRIM: For each task vector, zero out all but top-k% magnitude entries
      2. ELECT SIGN: For each parameter position, the elected sign is
         sign(sum of trimmed values across all tasks)
      3. MERGE: Average only the trimmed values whose sign matches the elected sign
    """
    N = len(delta_list)
    keys = list(delta_list[0].keys())
    merged = {}

    for k in keys:
        # Stack all deltas for this weight matrix: (N, *shape)
        stacked = mx.stack([d[k] for d in delta_list])  # (N, in_dim, out_dim)
        shape = stacked.shape[1:]
        flat = stacked.reshape(N, -1)  # (N, D) where D = in_dim * out_dim

        # Step 1: TRIM -- keep top density% by magnitude per task
        D = flat.shape[1]
        n_keep = max(1, int(D * density))
        trimmed_list = []
        for i in range(N):
            abs_vals = mx.abs(flat[i])
            # Find threshold: n_keep-th largest value
            sorted_vals = mx.sort(abs_vals)  # ascending
            threshold = sorted_vals[D - n_keep]  # the n_keep-th from the top
            mask = (abs_vals >= threshold).astype(mx.float32)
            trimmed_list.append(flat[i] * mask)
        trimmed = mx.stack(trimmed_list)  # (N, D)

        # Step 2: ELECT SIGN -- majority vote via sum
        aggregate = mx.sum(trimmed, axis=0)  # (D,)
        elected_sign = mx.sign(aggregate)  # +1, -1, or 0

        # Step 3: MERGE -- average only values matching elected sign
        # For each position, only include task vectors whose sign matches
        signs = mx.sign(trimmed)  # (N, D)
        match_mask = (signs == elected_sign[None, :]).astype(mx.float32)  # (N, D)

        # Where elected_sign is 0, only include tasks with nonzero trimmed values
        # (Previously included ALL tasks, diluting any surviving nonzero values)
        zero_elected = (elected_sign == 0).astype(mx.float32)  # (D,)
        has_nonzero = (trimmed != 0).astype(mx.float32)  # (N, D)
        match_mask = match_mask * (1 - zero_elected[None, :]) + zero_elected[None, :] * has_nonzero

        numerator = mx.sum(trimmed * match_mask, axis=0)  # (D,)
        denominator = mx.sum(match_mask, axis=0) + 1e-8  # (D,)
        merged_flat = numerator / denominator

        merged[k] = merged_flat.reshape(shape)

    return merged


# ── DARE (Drop And REscale) ──────────────────────────────────────────────────

def dare_sparsify(delta_list: list[dict], drop_rate: float = 0.9,
                  seed: int = 42) -> list[dict]:
    """Apply DARE sparsification to each delta dict.

    For each delta matrix, randomly drop drop_rate fraction of parameters
    and rescale remaining by 1/(1-drop_rate).

    Args:
        delta_list: list of N delta dicts
        drop_rate: fraction to drop (0.9 = drop 90%)
        seed: random seed for reproducibility

    Returns:
        list of N sparsified delta dicts
    """
    mx.random.seed(seed)
    result = []
    for i, deltas in enumerate(delta_list):
        sparse_deltas = {}
        for k, v in deltas.items():
            # Generate binary mask: keep with probability (1 - drop_rate)
            mask = (mx.random.uniform(shape=v.shape) >= drop_rate).astype(mx.float32)
            rescale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 0.0
            sparse_deltas[k] = v * mask * rescale
        result.append(sparse_deltas)
    return result


def merge_dare(delta_list: list[dict], drop_rate: float = 0.9,
               seed: int = 42) -> dict:
    """DARE + simple average: sparsify then average.

    Args:
        delta_list: list of N delta dicts
        drop_rate: fraction to drop
        seed: random seed

    Returns:
        merged delta dict
    """
    sparse = dare_sparsify(delta_list, drop_rate, seed)
    return merge_simple_average(sparse)


# ── DARE-TIES (DARE sparsification + TIES sign election) ────────────────────

def merge_dare_ties(delta_list: list[dict], drop_rate: float = 0.9,
                    density: float = 1.0, seed: int = 42) -> dict:
    """DARE-TIES: DARE sparsification followed by TIES sign election and merge.

    The DARE step replaces TIES's magnitude-based trimming with random dropping.
    Then we apply the elect-sign and disjoint-merge steps from TIES.

    With density=1.0, we skip the TIES trim step (DARE already handles sparsification).
    """
    sparse = dare_sparsify(delta_list, drop_rate, seed)
    # Apply TIES on the DARE-sparsified deltas (density=1.0 means no additional trimming)
    return merge_ties(sparse, density=density)


# ── Apply merged deltas to base model ────────────────────────────────────────

def apply_merged_deltas(base_model, merged_deltas: dict, vocab_size: int):
    """Create a GPT model with base weights + merged LoRA deltas baked in.

    Returns a standard GPT (no LoRA) with modified MLP weights.
    """
    from experiments.models.gpt.gpt import GPT

    BASE = dict(
        n_embd=base_model.wte.weight.shape[1],
        n_head=base_model.layers[0].attn.n_head,
        n_layer=len(base_model.layers),
        block_size=base_model.wpe.weight.shape[0],
    )
    model = GPT(vocab_size=vocab_size, **BASE)
    mx.eval(model.parameters())

    # Copy all base weights
    pairs = list(zip(
        [k for k, _ in nn.utils.tree_flatten(base_model.parameters())],
        [v for _, v in nn.utils.tree_flatten(base_model.parameters())]
    ))
    model.load_weights(pairs)
    mx.eval(model.parameters())

    # Apply merged deltas
    for (l_idx, name), delta in merged_deltas.items():
        layer = model.layers[l_idx]
        if name == 'fc1':
            layer.mlp.fc1.weight = layer.mlp.fc1.weight + delta.T
        elif name == 'fc2':
            layer.mlp.fc2.weight = layer.mlp.fc2.weight + delta.T

    mx.eval(model.parameters())
    return model
