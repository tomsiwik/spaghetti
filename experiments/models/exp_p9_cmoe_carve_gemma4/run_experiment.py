#!/usr/bin/env python3
"""
CMoE Carve Gemma 4 E4B: Dense FFN → Shared + Routed Experts on MLX.

Port of CMoE (arXiv:2502.04416) to Gemma 4 E4B 4-bit on Apple Silicon.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
N_EXPERTS = 8
N_SHARED = 1
N_ACTIVATED = 3  # 50% activation: 1 shared + 3 routed = 4/8 experts
K_ACT = 128
N_CALIB_BATCH = 4
SEQ_LEN = 256
OUTPUT_DIR = Path(__file__).parent


def dequantize_linear(ql: nn.QuantizedLinear) -> mx.array:
    w = mx.dequantize(ql.weight, ql.scales, ql.biases, ql.group_size, ql.bits)
    mx.eval(w)
    return w


def analyze_activations_mlx(h_ffn, gate_w, up_w, k_act):
    gate_out = h_ffn @ gate_w.T
    up_out = h_ffn @ up_w.T
    h_inter = nn.gelu_approx(gate_out) * up_out
    mx.eval(h_inter)

    n_tokens, D = h_inter.shape
    abs_h = mx.abs(h_inter).astype(mx.float32)
    mx.eval(abs_h)

    markers = np.zeros((n_tokens, D), dtype=np.float32)
    for start in range(0, n_tokens, 256):
        end = min(start + 256, n_tokens)
        topk = mx.argpartition(-abs_h[start:end], kth=k_act, axis=-1)[:, :k_act]
        mx.eval(topk)
        topk_np = np.array(topk)
        for i in range(end - start):
            markers[start + i, topk_np[i]] = 1.0

    return markers.mean(axis=0), markers


def balanced_kmeans(markers_np, rates_np, n_experts, n_shared):
    D = rates_np.shape[0]
    npe = D // n_experts

    shared = np.argsort(-rates_np)[:npe * n_shared].tolist()
    groups = [shared]

    remaining = sorted(set(range(D)) - set(shared))
    remaining_arr = np.array(remaining)
    k = n_experts - n_shared

    remaining_rates = rates_np[remaining_arr]
    init_idx = np.argsort(-remaining_rates)[:k]
    centroid_neurons = [remaining[i] for i in init_idx]

    mr = markers_np[:, remaining_arr].T
    centroids = markers_np[:, centroid_neurons].T

    dists = np.zeros((len(remaining), k), dtype=np.float32)
    for j in range(k):
        dists[:, j] = np.abs(mr - centroids[j:j+1]).sum(axis=1)

    cost = np.zeros((len(remaining), len(remaining)), dtype=np.float32)
    for i in range(k):
        cost[:, i*npe:(i+1)*npe] = dists[:, i:i+1]

    _, col = linear_sum_assignment(cost)
    assignments = col // npe

    reps = []
    for i in range(k):
        mask = assignments == i
        ids = remaining_arr[mask].tolist()
        groups.append(ids)
        gm = mr[mask]
        d = np.abs(gm - centroids[i:i+1]).sum(axis=1)
        reps.append(ids[int(np.argmin(d))])

    return groups, reps


class ExpertMLP(nn.Module):
    def __init__(self, gate_w, up_w, down_w):
        super().__init__()
        inter, hidden = gate_w.shape
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.gate_proj.weight = gate_w
        self.up_proj.weight = up_w
        self.down_proj.weight = down_w

    def __call__(self, x):
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class CMoERouter(nn.Module):
    def __init__(self, hidden_size, n_routed, n_activated, gate_w, up_w):
        super().__init__()
        self.n_activated = n_activated
        self.classifier = nn.Linear(hidden_size, n_routed, bias=False)
        self.gate_linear = nn.Linear(hidden_size, n_routed, bias=False)
        up_norm = up_w / (mx.linalg.norm(up_w, axis=1, keepdims=True) + 1e-8)
        gate_norm = gate_w / (mx.linalg.norm(gate_w, axis=1, keepdims=True) + 1e-8)
        self.classifier.weight = up_norm
        self.gate_linear.weight = gate_norm

    def __call__(self, x):
        scores = mx.abs(self.classifier(x) * nn.silu(self.gate_linear(x)))
        scores = mx.softmax(scores.astype(mx.float32), axis=-1)
        n_routed = scores.shape[-1]
        if self.n_activated >= n_routed:
            # All experts active — bypass routing to avoid argpartition OOB
            return mx.broadcast_to(
                mx.arange(n_routed), (x.shape[0], n_routed)
            )
        top = mx.argpartition(-scores, kth=self.n_activated, axis=-1)
        return top[..., :self.n_activated]


class CMoELayer(nn.Module):
    """Carved MoE: shared (always-on) + routed (top-k, weight=1) experts."""

    def __init__(self, shared, routed, router):
        super().__init__()
        self.shared_expert = shared
        self.routed_experts = routed
        self.router = router
        self.n_routed = len(routed)

    def __call__(self, x):
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])

        y = self.shared_expert(x_flat)

        indices = self.router(x_flat)  # (n_tokens, n_activated)

        for i in range(self.n_routed):
            mask = mx.any(indices == i, axis=-1)  # (n_tokens,)
            out = self.routed_experts[i](x_flat)
            y = y + out * mask[:, None].astype(x_flat.dtype)

        return y.reshape(shape)


def build_carved_moe(gate_w, up_w, down_w, groups, reps, n_activated):
    """Build CMoE from weight arrays and neuron groupings.
    gate_w, up_w: (inter, hidden) mx.array
    down_w: (hidden, inter) mx.array
    """
    experts = []
    for g_indices in groups:
        idx = mx.array(g_indices)
        g = gate_w[idx]
        u = up_w[idx]
        d = down_w[:, idx]
        experts.append(ExpertMLP(g, u, d))

    shared = experts[0]
    routed = experts[1:]

    rep_idx = mx.array(reps)
    router = CMoERouter(
        gate_w.shape[1], len(routed), n_activated,
        gate_w[rep_idx], up_w[rep_idx],
    )

    moe = CMoELayer(shared, routed, router)
    mx.eval(moe.parameters())
    return moe


def evaluate_ppl(model, tokenizer, texts, seq_len=512):
    total_nll = 0.0
    total_tokens = 0
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:seq_len + 1]
        input_ids = mx.array(tokens[:-1])[None, :]
        labels = mx.array(tokens[1:])
        logits = model(input_ids).squeeze(0)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        nll = -mx.take_along_axis(log_probs, labels[:, None], axis=-1).squeeze(-1)
        mx.eval(nll)
        total_nll += float(mx.sum(nll).item())
        total_tokens += len(tokens) - 1
    return float(np.exp(total_nll / total_tokens))


def benchmark_inference(model, tokenizer, n_tokens=50, n_warmup=2):
    from mlx_lm import generate
    prompt = "The quick brown fox jumps over the lazy dog."
    for _ in range(n_warmup):
        generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
    start = time.time()
    generate(model, tokenizer, prompt=prompt, max_tokens=n_tokens, verbose=False)
    return n_tokens / (time.time() - start)


def collect_activations(model, calib_input):
    """Single forward pass collecting pre-MLP hidden states."""
    text_model = model.language_model.model
    layers = model.layers
    n_layers = len(layers)

    h = text_model.embed_tokens(calib_input) * text_model.embed_scale
    mx.eval(h)

    pli_all = None
    if text_model.hidden_size_per_layer_input:
        pli_all = text_model._get_per_layer_inputs(calib_input)
        pli_all = text_model._project_per_layer_inputs(h, pli_all)
        mx.eval(pli_all)

    from mlx_lm.models.base import create_attention_mask
    masks = {}
    kvs = [(None, None)] * n_layers
    states = []

    for idx in range(n_layers):
        layer = layers[idx]
        lt = layer.layer_type
        if lt not in masks:
            if lt == "full_attention":
                masks[lt] = create_attention_mask(h, None)
            else:
                masks[lt] = create_attention_mask(h, None, window_size=text_model.window_size)

        pli = pli_all[:, :, idx, :] if pli_all is not None else None
        prev = text_model.previous_kvs[idx]
        kv, off = kvs[prev]

        # Manual unroll to capture pre-MLP state
        res = h
        h_n = layer.input_layernorm(h)
        h_a, kv_o, off_o = layer.self_attn(h_n, masks[lt], None, shared_kv=kv, offset=off)
        h_a = layer.post_attention_layernorm(h_a)
        h = res + h_a
        kvs[idx] = (kv_o, off_o)

        # THIS is the pre-MLP state
        h_pre = layer.pre_feedforward_layernorm(h)
        mx.eval(h_pre)
        B, S, D = h_pre.shape
        states.append(h_pre.reshape(-1, D))

        # Continue through MLP
        res2 = h
        h_m = layer.mlp(h_pre)
        h_m = layer.post_feedforward_layernorm(h_m)
        h = res2 + h_m

        if layer.per_layer_input_gate is not None and pli is not None:
            r = h
            g = nn.gelu_approx(layer.per_layer_input_gate(h))
            g = mx.multiply(g, pli)
            g = layer.per_layer_projection(g)
            g = layer.post_per_layer_input_norm(g)
            h = r + g

        if layer.layer_scalar is not None:
            h = h * layer.layer_scalar
        mx.eval(h)

    return states


# ── VERIFICATION ────────────────────────────────────────────────────────

def verify_carving_one_layer(model, layer_idx, calib_input):
    """Verify that all-experts-active carved MoE matches dense output.

    Returns (dense_output, all_experts_output, max_diff).
    """
    layer = model.layers[layer_idx]
    mlp = layer.mlp

    # Dequantize
    gate_w = dequantize_linear(mlp.gate_proj)
    up_w = dequantize_linear(mlp.up_proj)
    down_w = dequantize_linear(mlp.down_proj)

    # Get a test input (just random for verification)
    test_x = mx.random.normal(shape=(1, 2560)).astype(mx.float16)

    # Dense output via dequantized weights
    h = nn.gelu_approx(test_x @ gate_w.T) * (test_x @ up_w.T)
    dense_out = h @ down_w.T
    mx.eval(dense_out)

    # Compute per-group outputs and sum
    inter_size = gate_w.shape[0]
    npe = inter_size // N_EXPERTS
    all_indices = list(range(inter_size))

    # Just split evenly for this test
    group_out_sum = mx.zeros_like(dense_out)
    for g in range(N_EXPERTS):
        idx = list(range(g * npe, (g + 1) * npe))
        idx_mx = mx.array(idx)
        g_gate = gate_w[idx_mx]
        g_up = up_w[idx_mx]
        g_down = down_w[:, idx_mx]

        h_g = nn.gelu_approx(test_x @ g_gate.T) * (test_x @ g_up.T)
        out_g = h_g @ g_down.T
        group_out_sum = group_out_sum + out_g
    mx.eval(group_out_sum)

    diff = mx.max(mx.abs(dense_out - group_out_sum)).item()
    return float(diff)


def main():
    from mlx_lm import load

    results = {"experiment": "exp_p9_cmoe_carve_gemma4", "model": MODEL_ID}

    print("=" * 60)
    print("CMoE Carve Gemma 4 E4B → Shared + Routed Experts")
    print("=" * 60)

    # Load
    print("\n[1/8] Loading model...")
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    print(f"  Mem: {mx.get_active_memory()/1e9:.2f}GB")

    # Verify carving correctness on layer 0
    print("\n[2/8] Verifying carving correctness (layer 0)...")
    max_diff = verify_carving_one_layer(model, 0, None)
    print(f"  Max diff (all-experts vs dense): {max_diff:.6e}")
    results["verification_max_diff_layer0"] = max_diff
    if max_diff > 0.01:
        print("  WARNING: Carving does NOT reproduce dense output!")
    else:
        print("  OK: Carving reproduces dense output within tolerance.")

    # Verify on layer 20 too
    max_diff_20 = verify_carving_one_layer(model, 20, None)
    print(f"  Layer 20 max diff: {max_diff_20:.6e}")

    # Calibration data
    print("\n[3/8] Calibration data...")
    calib_texts = [
        "The theory of general relativity describes gravity as geometry. "
        "Albert Einstein published this in 1915, revolutionizing physics.",
        "Algorithms are building blocks for solving computational problems. "
        "Sorting and searching form the core of computer science.",
        "The immune system protects through innate and adaptive responses. "
        "Antibodies provide targeted defense against pathogens.",
        "Financial markets are influenced by indicators and behavior. "
        "Asset pricing models capture risk and return relationships.",
    ]
    calib_tokens = []
    for text in calib_texts:
        toks = tokenizer.encode(text * 5)[:SEQ_LEN]
        if len(toks) < SEQ_LEN:
            toks += [0] * (SEQ_LEN - len(toks))
        calib_tokens.append(toks)
    calib_input = mx.array(calib_tokens[:N_CALIB_BATCH])

    # Base PPL
    print("\n[4/8] Base PPL...")
    eval_texts = [
        "Mathematics is the language of the universe, providing precise descriptions of natural phenomena. "
        "From Euler's identity to the Riemann hypothesis, mathematical structures reveal deep truths. "
        "Number theory, once abstract, now underlies modern cryptography and security.",
        "Artificial intelligence has accelerated dramatically. Large language models generate coherent text, "
        "translate languages, and answer questions. The implications for society are profound, "
        "from scientific discovery to creative expression and beyond.",
        "Biodiversity is essential for ecosystem resilience. Each species maintains balance in nature. "
        "Loss of species cascades through ecosystems, affecting pollination, water, and nutrients.",
        "Computing spans from abacuses to quantum computers. Each generation builds on previous "
        "innovations, creating exponential growth in computational power.",
    ]
    base_ppl = evaluate_ppl(model, tokenizer, eval_texts, seq_len=SEQ_LEN)
    print(f"  Base PPL: {base_ppl:.2f}")
    results["base_ppl"] = base_ppl

    # Collect activations
    print("\n[5/8] Collecting activations...")
    t0 = time.time()
    pre_mlp = collect_activations(model, calib_input)
    print(f"  {len(pre_mlp)} layers in {time.time()-t0:.1f}s")

    # Carve
    print("\n[6/8] Carving FFN layers...")
    carve_t0 = time.time()
    layers = model.layers

    for li in range(len(layers)):
        layer = layers[li]
        mlp = layer.mlp
        t = time.time()

        gate_w = dequantize_linear(mlp.gate_proj)
        up_w = dequantize_linear(mlp.up_proj)
        down_w = dequantize_linear(mlp.down_proj)

        h_ffn = pre_mlp[li]
        rates, markers = analyze_activations_mlx(h_ffn, gate_w, up_w, K_ACT)
        del h_ffn

        groups, reps = balanced_kmeans(markers, rates, N_EXPERTS, N_SHARED)
        del markers, rates

        moe = build_carved_moe(gate_w, up_w, down_w, groups, reps, N_ACTIVATED)
        del gate_w, up_w, down_w

        # NO re-quantization for now — keep float16 to isolate quality
        layer.mlp = moe
        mx.clear_cache()

        if li % 7 == 0 or li == len(layers) - 1:
            print(f"  L{li:2d}/{len(layers)}: {time.time()-t:.1f}s mem={mx.get_active_memory()/1e9:.1f}GB")

    del pre_mlp
    carve_time = time.time() - carve_t0
    print(f"  Total carve: {carve_time:.1f}s ({carve_time/60:.1f}min)")
    results["total_carve_time_s"] = carve_time

    # Carved PPL
    print("\n[7/8] Carved PPL...")
    carved_ppl = evaluate_ppl(model, tokenizer, eval_texts, seq_len=SEQ_LEN)
    deg = (carved_ppl - base_ppl) / base_ppl * 100
    print(f"  Carved PPL: {carved_ppl:.2f} ({deg:+.1f}%)")
    results["carved_ppl"] = carved_ppl
    results["ppl_degradation_pct"] = deg

    # Speed
    print("\n[8/8] Speed benchmark...")
    try:
        carved_tps = benchmark_inference(model, tokenizer)
        print(f"  Carved: {carved_tps:.1f} tok/s")
    except Exception as e:
        print(f"  Failed: {e}")
        carved_tps = 0.0
    results["carved_tok_per_s"] = carved_tps

    del model
    mx.clear_cache()
    model_d, _ = load(MODEL_ID)
    mx.eval(model_d.parameters())
    dense_tps = benchmark_inference(model_d, tokenizer)
    print(f"  Dense: {dense_tps:.1f} tok/s")
    results["dense_tok_per_s"] = dense_tps

    speedup = carved_tps / dense_tps if dense_tps > 0 else 0
    print(f"  Speedup: {speedup:.2f}x")
    results["speedup"] = speedup

    # Kill criteria
    k1 = abs(deg) <= 5.0
    k2 = carve_time <= 600
    k3 = speedup >= 1.3

    results["kill_criteria"] = {
        "K1342": {"pass": k1, "value": f"{deg:+.1f}%"},
        "K1343": {"pass": k2, "value": f"{carve_time:.0f}s"},
        "K1344": {"pass": k3, "value": f"{speedup:.2f}x"},
    }

    print("\n" + "=" * 60)
    for k, v in results["kill_criteria"].items():
        print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} ({v['value']})")

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
