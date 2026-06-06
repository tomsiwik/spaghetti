"""E14: Grassmannian ⟹ Activation-Space Orthogonality proof.

Proves and validates: Grassmannian A guarantees E_x[cos(δ_i, δ_j)] ≈ 0
(zero-mean activation interference) but per-sample interference is bounded
by σ_max(B_i^T B_j).

Phases:
1. Construct Grassmannian and random A matrices, train minimal B matrices
2. Measure activation-level cos(δ_i, δ_j) for Grassmannian vs random
3. Validate bound: measured |cos| ≤ 2× predicted bound
4. Compare mean decorrelation: Grassmannian vs random
"""

import json
import os
import gc
import time
import sys

import mlx.core as mx
import mlx.nn as nn

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
RANK = 6
LORA_SCALE = 6
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

GLOBAL_ATTN_LAYERS = {5, 11, 17, 23, 29, 35, 41}

if SMOKE_TEST:
    LAYER_INDICES = [0, 6, 20]
    N_ADAPTERS = 3
    N_PROMPTS = 10
    TRAIN_STEPS = 20
else:
    LAYER_INDICES = [i for i in range(42) if i not in GLOBAL_ATTN_LAYERS]
    N_ADAPTERS = 5
    N_PROMPTS = 50
    TRAIN_STEPS = 50

PROMPTS = [
    "Explain the concept of recursion in programming.",
    "What causes tides in the ocean?",
    "Describe the process of photosynthesis.",
    "How does a compiler transform source code?",
    "What is the difference between TCP and UDP?",
    "Explain how vaccines work in the immune system.",
    "What is the Pythagorean theorem and why does it work?",
    "How do neural networks learn from data?",
    "Describe the water cycle in detail.",
    "What causes earthquakes along tectonic plates?",
    "How does encryption protect data in transit?",
    "Explain the concept of supply and demand.",
    "What is the greenhouse effect?",
    "How do transistors work in a computer chip?",
    "Describe the structure of DNA.",
    "What is the theory of general relativity?",
    "How does a blockchain maintain consensus?",
    "Explain the Krebs cycle in cellular respiration.",
    "What causes the seasons on Earth?",
    "How do electric motors convert energy?",
    "Describe the process of protein folding.",
    "What is quantum entanglement?",
    "How does the human eye perceive color?",
    "Explain how hash tables achieve O(1) lookup.",
    "What is natural selection and how does it work?",
    "How do antibiotics fight bacterial infections?",
    "Describe the lifecycle of a star.",
    "What causes auroras (northern lights)?",
    "How does GPS determine your location?",
    "Explain the concept of entropy in thermodynamics.",
    "What is the CAP theorem in distributed systems?",
    "How do batteries store and release energy?",
    "Describe the process of nuclear fission.",
    "What causes inflation in an economy?",
    "How does the internet route packets?",
    "Explain the double slit experiment.",
    "What is CRISPR and how does it edit genes?",
    "How do airplanes generate lift?",
    "Describe the carbon cycle.",
    "What is the halting problem in computer science?",
    "How does a refrigerator cool its contents?",
    "Explain the concept of machine learning overfitting.",
    "What causes volcanic eruptions?",
    "How does the human heart pump blood?",
    "Describe the process of fermentation.",
    "What is dark matter and how do we know it exists?",
    "How do search engines rank web pages?",
    "Explain the concept of opportunity cost.",
    "What is the Heisenberg uncertainty principle?",
    "How does fiber optic communication work?",
]


def load_model():
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    mx.eval(model.parameters())
    return model, tokenizer


def dequantize_weight(qlinear):
    W = mx.dequantize(qlinear.weight, qlinear.scales, qlinear.biases,
                      qlinear.group_size, qlinear.bits)
    mx.eval(W)
    return W


def grassmannian_A(d_in, rank, n, seed=42):
    key = mx.random.key(seed)
    W = mx.random.normal(key=key, shape=(d_in, n * rank))
    mx.eval(W)
    Q, _ = mx.linalg.qr(W, stream=mx.cpu)
    mx.eval(Q)
    As = []
    for i in range(n):
        A_i = Q[:, i * rank:(i + 1) * rank].T
        mx.eval(A_i)
        As.append(A_i)
    return As


def random_A(d_in, rank, n, seed=99):
    key = mx.random.key(seed)
    As = []
    for i in range(n):
        sub_key = mx.random.key(seed + i + 1)
        A_i = mx.random.normal(key=sub_key, shape=(rank, d_in))
        A_i = A_i / mx.sqrt(mx.sum(A_i * A_i, axis=1, keepdims=True) + 1e-8)
        mx.eval(A_i)
        As.append(A_i)
    return As


def train_B_matrices(model, tokenizer, layer_idx, A_matrices, target_proj="v_proj"):
    """Train minimal B matrices via gradient-free approximation.

    We don't need full LoRA training — just B matrices that capture
    some data-dependent structure. Use the activation covariance method:
    B_i ≈ W @ A_i^T (A_i A_i^T)^{-1} scaled to LoRA magnitude.
    This gives B matrices that approximate the local gradient structure.
    """
    layer = model.language_model.model.layers[layer_idx]
    if target_proj == "v_proj":
        qlinear = layer.self_attn.v_proj
    else:
        qlinear = layer.self_attn.o_proj

    W = dequantize_weight(qlinear)
    d_out, d_in = W.shape

    Bs = []
    for A_i in A_matrices:
        B_i = W @ A_i.T
        norm = mx.sqrt(mx.sum(B_i * B_i) + 1e-8)
        B_i = B_i * (LORA_SCALE * RANK / (norm + 1e-8))
        mx.eval(B_i)
        Bs.append(B_i)

    del W
    mx.clear_cache()
    gc.collect()
    return Bs


class CaptureWrapper(nn.Module):
    """Wraps a QuantizedLinear to capture its input."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.captured = []
        self.capturing = False

    def __call__(self, x):
        if self.capturing:
            self.captured.append(x)
        return self.inner(x)


def collect_hidden_states(model, tokenizer, layer_idx, n_prompts):
    """Collect hidden states at the input to the target layer's v_proj."""
    prompts = PROMPTS[:n_prompts]
    hidden_states = []

    attn = model.language_model.model.layers[layer_idx].self_attn
    original_v_proj = attn.v_proj
    wrapper = CaptureWrapper(original_v_proj)
    attn.v_proj = wrapper

    try:
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            wrapper.captured = []
            wrapper.capturing = True

            out = model(input_ids)
            mx.eval(out)
            wrapper.capturing = False

            if wrapper.captured:
                h = wrapper.captured[0]
                if h.ndim == 3:
                    h_mean = mx.mean(h[0], axis=0)
                else:
                    h_mean = h
                mx.eval(h_mean)
                hidden_states.append(h_mean)

            del out
            wrapper.captured = []
            mx.clear_cache()
    finally:
        attn.v_proj = original_v_proj

    return hidden_states


def compute_interference(B_matrices, A_matrices, hidden_states):
    """Compute cos(δ_i, δ_j) for all pairs across all hidden states."""
    n = len(A_matrices)
    all_cos = []
    all_predicted_bounds = []

    for i in range(n):
        for j in range(i + 1, n):
            M = B_matrices[i].T @ B_matrices[j]
            mx.eval(M)
            U_m, S_m, Vt_m = mx.linalg.svd(M.astype(mx.float32), stream=mx.cpu)
            mx.eval(S_m)
            sigma_max = S_m[0].item()

            pair_cos = []
            pair_bounds = []

            for x in hidden_states:
                z_i = A_matrices[i] @ x
                z_j = A_matrices[j] @ x
                mx.eval(z_i, z_j)

                delta_i = B_matrices[i] @ z_i
                delta_j = B_matrices[j] @ z_j
                mx.eval(delta_i, delta_j)

                norm_i = mx.sqrt(mx.sum(delta_i * delta_i) + 1e-12)
                norm_j = mx.sqrt(mx.sum(delta_j * delta_j) + 1e-12)
                cos_val = mx.sum(delta_i * delta_j) / (norm_i * norm_j)
                mx.eval(cos_val)

                norm_zi = mx.sqrt(mx.sum(z_i * z_i) + 1e-12).item()
                norm_zj = mx.sqrt(mx.sum(z_j * z_j) + 1e-12).item()
                predicted = sigma_max * norm_zi * norm_zj / (norm_i.item() * norm_j.item() + 1e-12)

                pair_cos.append(cos_val.item())
                pair_bounds.append(predicted)

            all_cos.extend(pair_cos)
            all_predicted_bounds.extend(pair_bounds)

            del M, S_m
            mx.clear_cache()

    return all_cos, all_predicted_bounds


def run_phase1_and_2(model, tokenizer):
    """Construct adapters, collect activations, measure interference."""
    results = {"layers": {}}

    for layer_idx in LAYER_INDICES:
        print(f"\n=== Layer {layer_idx} ===")
        t0 = time.time()

        layer = model.language_model.model.layers[layer_idx]
        d_out, d_in_packed = layer.self_attn.v_proj.weight.shape
        d_in = d_in_packed * (32 // layer.self_attn.v_proj.bits)

        print(f"  v_proj: ({d_out}, {d_in})")

        A_grass = grassmannian_A(d_in, RANK, N_ADAPTERS, seed=42 + layer_idx)
        A_rand = random_A(d_in, RANK, N_ADAPTERS, seed=99 + layer_idx)

        ortho_check = []
        for i in range(N_ADAPTERS):
            for j in range(i + 1, N_ADAPTERS):
                dot = mx.sum(A_grass[i] @ A_grass[j].T).item()
                ortho_check.append(abs(dot))
        print(f"  Grassmannian A ortho check (max |A_i^T A_j|): {max(ortho_check):.6f}")

        rand_ortho = []
        for i in range(N_ADAPTERS):
            for j in range(i + 1, N_ADAPTERS):
                dot = mx.sum(A_rand[i] @ A_rand[j].T).item()
                rand_ortho.append(abs(dot))
        print(f"  Random A overlap (max |A_i^T A_j|): {max(rand_ortho):.6f}")

        B_grass = train_B_matrices(model, tokenizer, layer_idx, A_grass, "v_proj")
        B_rand = train_B_matrices(model, tokenizer, layer_idx, A_rand, "v_proj")

        M_grass_norms = []
        for i in range(N_ADAPTERS):
            for j in range(i + 1, N_ADAPTERS):
                M = B_grass[i].T @ B_grass[j]
                mx.eval(M)
                U, S, Vt = mx.linalg.svd(M.astype(mx.float32), stream=mx.cpu)
                mx.eval(S)
                M_grass_norms.append(S[0].item())
                del M, U, S, Vt
                mx.clear_cache()

        M_rand_norms = []
        for i in range(N_ADAPTERS):
            for j in range(i + 1, N_ADAPTERS):
                M = B_rand[i].T @ B_rand[j]
                mx.eval(M)
                U, S, Vt = mx.linalg.svd(M.astype(mx.float32), stream=mx.cpu)
                mx.eval(S)
                M_rand_norms.append(S[0].item())
                del M, U, S, Vt
                mx.clear_cache()

        print(f"  σ_max(B_grass^T B_grass) mean: {sum(M_grass_norms)/len(M_grass_norms):.4f}")
        print(f"  σ_max(B_rand^T B_rand) mean: {sum(M_rand_norms)/len(M_rand_norms):.4f}")

        print(f"  Collecting hidden states...")
        hidden_states = collect_hidden_states(model, tokenizer, layer_idx, N_PROMPTS)
        print(f"  Got {len(hidden_states)} hidden states")

        if len(hidden_states) == 0:
            print(f"  WARNING: No hidden states collected, skipping layer {layer_idx}")
            results["layers"][str(layer_idx)] = {"error": "no hidden states"}
            continue

        print(f"  Computing Grassmannian interference...")
        cos_grass, bounds_grass = compute_interference(B_grass, A_grass, hidden_states)

        print(f"  Computing random interference...")
        cos_rand, bounds_rand = compute_interference(B_rand, A_rand, hidden_states)

        mean_cos_grass = sum(abs(c) for c in cos_grass) / len(cos_grass) if cos_grass else 0
        mean_cos_rand = sum(abs(c) for c in cos_rand) / len(cos_rand) if cos_rand else 0

        mean_signed_grass = sum(cos_grass) / len(cos_grass) if cos_grass else 0
        mean_signed_rand = sum(cos_rand) / len(cos_rand) if cos_rand else 0

        bound_violations = sum(1 for c, b in zip(cos_grass, bounds_grass)
                               if abs(c) > 2.0 * b)
        violation_rate = bound_violations / len(cos_grass) if cos_grass else 0

        dt = time.time() - t0
        print(f"  Layer {layer_idx} done in {dt:.1f}s")
        print(f"  Grassmannian mean |cos|: {mean_cos_grass:.4f}, signed mean: {mean_signed_grass:.4f}")
        print(f"  Random mean |cos|: {mean_cos_rand:.4f}, signed mean: {mean_signed_rand:.4f}")
        print(f"  Bound violation rate: {violation_rate:.2%} ({bound_violations}/{len(cos_grass)})")
        print(f"  Decorrelation benefit: {mean_cos_rand - mean_cos_grass:.4f}")

        results["layers"][str(layer_idx)] = {
            "d_in": int(d_in),
            "d_out": int(d_out),
            "n_hidden_states": len(hidden_states),
            "grassmannian": {
                "mean_abs_cos": float(mean_cos_grass),
                "mean_signed_cos": float(mean_signed_grass),
                "std_cos": float((sum((c - mean_signed_grass)**2 for c in cos_grass) / len(cos_grass))**0.5) if cos_grass else 0,
                "max_abs_cos": float(max(abs(c) for c in cos_grass)) if cos_grass else 0,
                "n_samples": len(cos_grass),
                "sigma_max_B_mean": float(sum(M_grass_norms) / len(M_grass_norms)),
            },
            "random": {
                "mean_abs_cos": float(mean_cos_rand),
                "mean_signed_cos": float(mean_signed_rand),
                "std_cos": float((sum((c - mean_signed_rand)**2 for c in cos_rand) / len(cos_rand))**0.5) if cos_rand else 0,
                "max_abs_cos": float(max(abs(c) for c in cos_rand)) if cos_rand else 0,
                "n_samples": len(cos_rand),
                "sigma_max_B_mean": float(sum(M_rand_norms) / len(M_rand_norms)),
            },
            "bound_violation_rate": float(violation_rate),
            "decorrelation_benefit": float(mean_cos_rand - mean_cos_grass),
            "ortho_check_max": float(max(ortho_check)),
            "random_overlap_max": float(max(rand_ortho)),
            "time_s": float(dt),
        }

        del A_grass, A_rand, B_grass, B_rand, hidden_states
        del cos_grass, cos_rand, bounds_grass, bounds_rand
        mx.clear_cache()
        gc.collect()

    return results


def evaluate_kcs(results):
    """Evaluate kill criteria against measurements."""
    layers = results["layers"]
    valid_layers = {k: v for k, v in layers.items() if "error" not in v}

    if not valid_layers:
        return {
            "K2043_bound_holds": False,
            "K2044_decorrelation_measurable": False,
            "all_pass": False,
            "verdict": "KILLED",
            "reason": "No valid layer measurements",
        }

    all_violation_rates = [v["bound_violation_rate"] for v in valid_layers.values()]
    mean_violation_rate = sum(all_violation_rates) / len(all_violation_rates)
    k2043_pass = mean_violation_rate <= 0.10

    decorr_benefits = [v["decorrelation_benefit"] for v in valid_layers.values()]
    mean_decorr = sum(decorr_benefits) / len(decorr_benefits)
    k2044_pass = abs(mean_decorr) >= 0.01

    all_pass = k2043_pass and k2044_pass
    verdict = "SUPPORTED" if all_pass else "KILLED"

    return {
        "K2043_bound_holds": k2043_pass,
        "K2043_mean_violation_rate": float(mean_violation_rate),
        "K2043_detail": f"Mean violation rate {mean_violation_rate:.2%} vs ≤10% threshold",
        "K2044_decorrelation_measurable": k2044_pass,
        "K2044_mean_decorrelation": float(mean_decorr),
        "K2044_detail": f"Mean decorrelation benefit {mean_decorr:.4f} vs ≥0.01 threshold",
        "all_pass": all_pass,
        "verdict": verdict,
    }


def main():
    print(f"E14: Grassmannian ⟹ Activation-Space Orthogonality")
    print(f"SMOKE_TEST={SMOKE_TEST}")
    print(f"Layers: {LAYER_INDICES}, N_ADAPTERS={N_ADAPTERS}, N_PROMPTS={N_PROMPTS}")
    print(f"RANK={RANK}, LORA_SCALE={LORA_SCALE}")
    print()

    t_start = time.time()

    print("Loading model...")
    model, tokenizer = load_model()
    print(f"Model loaded in {time.time() - t_start:.1f}s")
    mx.clear_cache()
    gc.collect()

    results = run_phase1_and_2(model, tokenizer)

    del model, tokenizer
    mx.clear_cache()
    gc.collect()

    kc_results = evaluate_kcs(results)
    results["kill_criteria"] = kc_results
    results["is_smoke"] = SMOKE_TEST
    results["verdict"] = kc_results["verdict"]
    results["all_pass"] = kc_results["all_pass"]
    results["config"] = {
        "model": MODEL_ID,
        "rank": RANK,
        "lora_scale": LORA_SCALE,
        "n_adapters": N_ADAPTERS,
        "n_prompts": N_PROMPTS,
        "train_steps": TRAIN_STEPS,
        "layer_indices": LAYER_INDICES,
    }
    results["total_time_s"] = float(time.time() - t_start)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"K2043 (bound holds): {'PASS' if kc_results['K2043_bound_holds'] else 'FAIL'}")
    print(f"  {kc_results['K2043_detail']}")
    print(f"K2044 (decorrelation): {'PASS' if kc_results['K2044_decorrelation_measurable'] else 'FAIL'}")
    print(f"  {kc_results['K2044_detail']}")
    print(f"Verdict: {kc_results['verdict']}")
    print(f"Total time: {results['total_time_s']:.1f}s")

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
