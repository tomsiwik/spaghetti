"""Model catalog and load/unload helpers for pretrained LLMs.

Two backends:
  - "local": loaded via mlx-lm, runs on Apple Silicon GPU
  - "api":   evaluated via Together AI logprobs endpoint
"""

import gc
import time

MODEL_CATALOG = {
    # --- Local models (mlx-lm) ---
    "smollm-135m": {
        "hf_id": "HuggingFaceTB/SmolLM-135M",
        "tier": "tiny",
        "backend": "local",
    },
    "smollm-1.7b": {
        "hf_id": "HuggingFaceTB/SmolLM2-1.7B",
        "tier": "small",
        "backend": "local",
    },
    "qwen-coder-0.5b": {
        "hf_id": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        "tier": "small",
        "backend": "local",
    },
    "qwen-coder-1.5b": {
        "hf_id": "mlx-community/Qwen2.5-Coder-1.5B-4bit",
        "tier": "medium",
        "backend": "local",
    },
    "qwen-coder-7b": {
        "hf_id": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "tier": "large",
        "backend": "local",
    },
    "deepseek-coder-6.7b": {
        "hf_id": "mlx-community/deepseek-coder-6.7b-instruct-hf-4bit-mlx",
        "tier": "medium",
        "backend": "local",
    },
    "codestral-22b": {
        "hf_id": "mlx-community/Codestral-22B-v0.1-4bit",
        "tier": "large",
        "backend": "local",
        "note": "12.5 GB — tight on 16 GB RAM",
    },
    # --- API models (Together AI, MoE heavyweights) ---
    "minimax-m2.5": {
        "together_id": "MiniMaxAI/MiniMax-M2.5",
        "tier": "king",
        "backend": "api",
        "arch": "MoE",
        "params": "229B total / 46B active",
    },
    "deepseek-v3.1": {
        "together_id": "deepseek-ai/DeepSeek-V3-0324",
        "tier": "king",
        "backend": "api",
        "arch": "MoE",
        "params": "671B total / 37B active",
    },
    "mixtral-8x7b": {
        "together_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "tier": "large",
        "backend": "api",
        "arch": "MoE",
        "params": "46.7B total / 12.9B active",
    },
}


def list_models() -> list[str]:
    """List all model names in the catalog."""
    return list(MODEL_CATALOG.keys())


def get_model_info(name: str) -> dict:
    """Get catalog entry for a model name."""
    if name in MODEL_CATALOG:
        return MODEL_CATALOG[name]
    raise KeyError(f"Unknown model: {name}. Use --list or pass --eval-hf <hf_id>.")


def count_params(model) -> int:
    """Count total parameters in a model."""
    import mlx.nn as nn

    return sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))


def load_model(hf_id: str) -> tuple:
    """Load a pretrained model via mlx-lm. Returns (model, tokenizer, load_time_s)."""
    from mlx_lm import load

    t0 = time.time()
    model, tokenizer = load(hf_id)
    load_time = time.time() - t0
    return model, tokenizer, load_time


def unload_model():
    """Free GPU memory after unloading a model."""
    import mlx.core as mx

    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    else:
        mx.metal.clear_cache()

from typing import Dict, Any

def load_and_apply_weights(model_name: str, expert_weights: Dict[str, float]) -> Any:
    """
    Loads a model and applies expert weights.
    This is a placeholder function.
    """
    print(f"Loading model: {model_name}")
    print(f"Applying weights: {expert_weights}")
    # In a real implementation, this would load the actual model
    # and apply the weights using a specific framework (e.g., PyTorch, TensorFlow).
    return {"message": f"Model {model_name} loaded and weights applied."}
