"""
SHINE S1: Memory Token Extraction on Gemma 4 E4B

Extracts per-layer hidden states from M learnable memory tokens appended
to the input sequence. This is the first step of the SHINE hypernetwork
pipeline (arXiv:2602.06358 S3.2), ported to Gemma 4 on MLX.

Kill criteria:
  K1252: shape == (42, M, 2560)
  K1253: mean cross-layer cosine < 0.95
  K1254: latency < 500ms for T=1024 on M5 Pro
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Memory extraction module
# ---------------------------------------------------------------------------

class MemoryExtractor(nn.Module):
    """Wraps a frozen Gemma 4 text model to extract per-layer memory states."""

    def __init__(self, text_model, num_mem_tokens: int = 32):
        super().__init__()
        self.text_model = text_model
        self.num_mem_tokens = num_mem_tokens
        hidden_size = text_model.config.hidden_size  # 2560
        # Learnable memory embeddings (will be scaled by sqrt(d) like normal embeds)
        self.mem_tokens = mx.random.normal(
            shape=(1, num_mem_tokens, hidden_size)
        ) * 0.02

    def _build_graph(self, input_ids: mx.array):
        """Build the lazy computation graph (no eval)."""
        tm = self.text_model
        config = tm.config
        M = self.num_mem_tokens
        L = config.num_hidden_layers  # 42

        # 1. Embed context tokens
        ctx_embeds = tm.embed_tokens(input_ids)  # (1, T, 2560)

        # 2. Concatenate memory token embeddings
        mem_embeds = mx.broadcast_to(
            self.mem_tokens, (1, M, config.hidden_size)
        )
        h = mx.concatenate([ctx_embeds, mem_embeds], axis=1)  # (1, T+M, 2560)
        h = h * config.hidden_size**0.5  # embed_scale

        # 3. Per-layer inputs — context tokens get normal per-layer embeddings,
        # memory tokens get zeros (no valid token IDs; the model will learn
        # to use them through the main residual stream instead).
        if tm.hidden_size_per_layer_input:
            ctx_pli = tm._get_per_layer_inputs(input_ids)  # (1, T, L, 256)
            # Pad memory positions with zeros
            mem_pli = mx.zeros((1, M, L, tm.hidden_size_per_layer_input))
            full_pli = mx.concatenate([ctx_pli, mem_pli], axis=1)  # (1, T+M, L, 256)
            full_pli = tm._project_per_layer_inputs(h, full_pli)
            per_layer_inputs = [
                full_pli[:, :, i, :]
                for i in range(L)
            ]
        else:
            per_layer_inputs = [None] * L

        # 4. Build attention masks (no cache)
        from mlx_lm.models.base import create_attention_mask
        mask = {}
        masks = []
        for layer in tm.layers:
            if layer.layer_type not in mask:
                if layer.layer_type == "full_attention":
                    mask["full_attention"] = create_attention_mask(h, None)
                elif layer.layer_type == "sliding_attention":
                    mask["sliding_attention"] = create_attention_mask(
                        h, None, window_size=tm.window_size
                    )
            masks.append(mask[layer.layer_type])

        # 5. Forward through layers, extracting memory states
        memory_states = []
        intermediates = [(None, None)] * L
        cache = [None] * L
        for idx in range(L):
            layer = tm.layers[idx]
            prev_idx = tm.previous_kvs[idx]
            kvs, offset = intermediates[prev_idx]

            h, kvs, offset = layer(
                h, masks[idx], cache[idx],
                per_layer_input=per_layer_inputs[idx],
                shared_kv=kvs,
                offset=offset,
            )
            intermediates[idx] = (kvs, offset)

            # Extract memory token hidden states (last M positions)
            memory_states.append(h[:, -M:, :])

        # Stack into (1, L, M, d)
        return mx.stack(memory_states, axis=1)  # (1, L, M, d)

    def extract(self, input_ids: mx.array):
        """
        Args:
            input_ids: (1, T) token IDs
        Returns:
            memory_states: (num_layers, num_mem_tokens, hidden_size)
            latency_ms: float
        """
        memory_states = self._build_graph(input_ids)

        # Time only the evaluation (actual GPU compute)
        start = time.perf_counter()
        mx.eval(memory_states)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return memory_states[0], elapsed_ms  # (L, M, d), float


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_cross_layer_cosine(memory_states: mx.array) -> dict:
    """
    Args:
        memory_states: (L, M, d)
    Returns:
        dict with cosine statistics
    """
    L, M, d = memory_states.shape

    # Flatten each layer's memory to (L, M*d) for layer-level comparison
    flat = memory_states.reshape(L, -1)  # (L, M*d)
    norms = mx.sqrt((flat * flat).sum(axis=-1, keepdims=True))  # (L, 1)
    normed = flat / (norms + 1e-8)  # (L, M*d)

    # Pairwise cosine: (L, L)
    cos_matrix = normed @ normed.T
    mx.eval(cos_matrix)

    # Extract upper triangle (exclude diagonal)
    cos_np = cos_matrix.tolist()
    pairwise = []
    for i in range(L):
        for j in range(i + 1, L):
            pairwise.append(cos_np[i][j])

    mean_cos = sum(pairwise) / len(pairwise)

    # Cosine by layer distance
    by_distance = {}
    for i in range(L):
        for j in range(i + 1, L):
            dist = j - i
            by_distance.setdefault(dist, []).append(cos_np[i][j])
    avg_by_distance = {d: sum(v) / len(v) for d, v in sorted(by_distance.items())}

    # Global vs sliding layer distinction
    # Full-attention layers at indices: 5, 11, 17, 23, 29, 35, 41
    full_attn_indices = {5, 11, 17, 23, 29, 35, 41}
    sliding_indices = set(range(L)) - full_attn_indices

    full_cos = []
    sliding_cos = []
    cross_cos = []
    for i in range(L):
        for j in range(i + 1, L):
            c = cos_np[i][j]
            if i in full_attn_indices and j in full_attn_indices:
                full_cos.append(c)
            elif i in sliding_indices and j in sliding_indices:
                sliding_cos.append(c)
            else:
                cross_cos.append(c)

    return {
        "mean_cosine": mean_cos,
        "min_cosine": min(pairwise),
        "max_cosine": max(pairwise),
        "std_cosine": (sum((x - mean_cos) ** 2 for x in pairwise) / len(pairwise)) ** 0.5,
        "avg_by_distance_sample": {
            k: round(v, 4)
            for k, v in list(avg_by_distance.items())[:5]
            + list(avg_by_distance.items())[-3:]
        },
        "full_attn_mean_cos": sum(full_cos) / len(full_cos) if full_cos else None,
        "sliding_mean_cos": sum(sliding_cos) / len(sliding_cos) if sliding_cos else None,
        "cross_type_mean_cos": sum(cross_cos) / len(cross_cos) if cross_cos else None,
    }


def compute_norm_stats(memory_states: mx.array) -> dict:
    """Per-layer norm statistics."""
    L, M, d = memory_states.shape
    # L2 norm per memory token per layer
    norms = mx.sqrt((memory_states * memory_states).sum(axis=-1))  # (L, M)
    mx.eval(norms)
    norms_list = norms.tolist()
    per_layer_mean = [sum(row) / len(row) for row in norms_list]
    return {
        "global_mean_norm": sum(per_layer_mean) / len(per_layer_mean),
        "min_layer_norm": min(per_layer_mean),
        "max_layer_norm": max(per_layer_mean),
        "first_5_layer_norms": [round(x, 3) for x in per_layer_mean[:5]],
        "last_5_layer_norms": [round(x, 3) for x in per_layer_mean[-5:]],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from mlx_lm import load

    results = {}
    M = 32  # number of memory tokens
    T = 1024  # context length

    print("Loading Gemma 4 E4B 4-bit...")
    model, tokenizer = load("mlx-community/gemma-4-e4b-it-4bit")
    text_model = model.language_model.model
    print(f"  Layers: {len(text_model.layers)}")
    print(f"  Hidden size: {text_model.config.hidden_size}")

    # Freeze model
    model.freeze()

    # Create memory extractor
    extractor = MemoryExtractor(text_model, num_mem_tokens=M)

    # Prepare input: 1024 tokens of real text
    context = (
        "The mitochondria is the powerhouse of the cell. It generates ATP through "
        "oxidative phosphorylation. The electron transport chain consists of four "
        "protein complexes embedded in the inner mitochondrial membrane. Complex I "
        "accepts electrons from NADH, Complex II from succinate, Complex III transfers "
        "electrons to cytochrome c, and Complex IV transfers electrons to oxygen. "
        "Protons are pumped across the membrane, creating a gradient that drives ATP "
        "synthase. This process is fundamental to aerobic respiration in eukaryotic "
        "cells and produces approximately 30-32 ATP molecules per glucose molecule. "
        "The Krebs cycle, also known as the citric acid cycle, occurs in the "
        "mitochondrial matrix and produces the electron carriers NADH and FADH2. "
    )
    # Repeat to reach ~1024 tokens
    context = context * 5
    tokens = tokenizer.encode(context)
    if len(tokens) > T:
        tokens = tokens[:T]
    elif len(tokens) < T:
        # Pad by repeating
        while len(tokens) < T:
            tokens = tokens + tokens
        tokens = tokens[:T]

    input_ids = mx.array([tokens])
    actual_T = input_ids.shape[1]
    print(f"  Input tokens: {actual_T}")
    print(f"  Memory tokens: {M}")

    # Warmup
    print("\nWarmup run...")
    _ = extractor.extract(input_ids)

    # Timed runs
    print("Timed runs (3x)...")
    latencies = []
    memory_states = None
    for i in range(3):
        ms, lat = extractor.extract(input_ids)
        latencies.append(lat)
        memory_states = ms
        print(f"  Run {i+1}: {lat:.1f}ms, shape={ms.shape}")

    # --- K1252: Shape check ---
    shape = tuple(int(x) for x in memory_states.shape)
    expected_shape = (42, M, 2560)
    k1252_pass = shape == expected_shape
    print(f"\nK1252 (shape): {shape} == {expected_shape} -> {'PASS' if k1252_pass else 'FAIL'}")
    results["k1252_shape"] = {
        "actual": list(shape),
        "expected": list(expected_shape),
        "pass": k1252_pass,
    }

    # --- K1253: Non-degeneracy ---
    print("\nComputing cross-layer cosine similarity...")
    cos_stats = compute_cross_layer_cosine(memory_states)
    k1253_pass = cos_stats["mean_cosine"] < 0.95
    print(f"K1253 (non-degenerate): mean_cos={cos_stats['mean_cosine']:.4f} < 0.95 -> {'PASS' if k1253_pass else 'FAIL'}")
    print(f"  min={cos_stats['min_cosine']:.4f}, max={cos_stats['max_cosine']:.4f}, std={cos_stats['std_cosine']:.4f}")
    print(f"  full_attn layers mean: {cos_stats['full_attn_mean_cos']:.4f}" if cos_stats['full_attn_mean_cos'] else "")
    print(f"  sliding layers mean: {cos_stats['sliding_mean_cos']:.4f}" if cos_stats['sliding_mean_cos'] else "")
    print(f"  cross-type mean: {cos_stats['cross_type_mean_cos']:.4f}" if cos_stats['cross_type_mean_cos'] else "")
    print(f"  Distance samples: {cos_stats['avg_by_distance_sample']}")
    results["k1253_cosine"] = {
        "mean_cosine": cos_stats["mean_cosine"],
        "pass": k1253_pass,
        **cos_stats,
    }

    # --- K1254: Latency ---
    median_lat = sorted(latencies)[len(latencies) // 2]
    k1254_pass = median_lat < 500.0
    print(f"\nK1254 (latency): {median_lat:.1f}ms < 500ms -> {'PASS' if k1254_pass else 'FAIL'}")
    print(f"  All: {[f'{l:.1f}' for l in latencies]}")
    results["k1254_latency"] = {
        "median_ms": median_lat,
        "all_ms": latencies,
        "pass": k1254_pass,
    }

    # --- Norm statistics ---
    print("\nNorm statistics...")
    norm_stats = compute_norm_stats(memory_states)
    print(f"  Global mean norm: {norm_stats['global_mean_norm']:.3f}")
    print(f"  Range: [{norm_stats['min_layer_norm']:.3f}, {norm_stats['max_layer_norm']:.3f}]")
    print(f"  First 5: {norm_stats['first_5_layer_norms']}")
    print(f"  Last 5: {norm_stats['last_5_layer_norms']}")
    results["norm_stats"] = norm_stats

    # --- Summary ---
    all_pass = k1252_pass and k1253_pass and k1254_pass
    results["summary"] = {
        "all_pass": all_pass,
        "k1252": "PASS" if k1252_pass else "FAIL",
        "k1253": "PASS" if k1253_pass else "FAIL",
        "k1254": "PASS" if k1254_pass else "FAIL",
        "context_tokens": actual_T,
        "memory_tokens": M,
        "model": "gemma-4-e4b-it-4bit",
    }

    print(f"\n{'='*60}")
    print(f"SUMMARY: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print(f"  K1252 shape:         {results['summary']['k1252']}")
    print(f"  K1253 non-degenerate: {results['summary']['k1253']} (mean_cos={cos_stats['mean_cosine']:.4f})")
    print(f"  K1254 latency:       {results['summary']['k1254']} ({median_lat:.1f}ms)")
    print(f"{'='*60}")

    # Save results
    out_dir = Path(__file__).parent
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
