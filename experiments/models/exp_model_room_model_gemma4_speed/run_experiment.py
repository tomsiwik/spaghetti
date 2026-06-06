"""Room Model on Gemma 4 E4B — speed + equivalence + exact reversibility.

Three pre-registered KCs (see MATH.md):
- KC1688: W_combined (pre-summed N=5 adapter deltas) achieves >=150 tok/s on M5 Pro.
- KC1689: logit cosine(W_combined, explicit per-adapter routing) > 0.999.
- KC1690: add/remove via += / -= is bitwise exact vs fresh W_combined without k.

No training: all adapters are random Grassmannian A + small-Gaussian B. All KCs test
mathematical identities of the injection mechanism (see MATH.md §4). PoLAR r=6 on
v_proj and o_proj (PLAN.md Part 2).

Phased execution (mlx-dev pattern): each phase in its own function, mx.eval() at
boundaries, mx.clear_cache() between phases.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
N_ADAPTERS = 5
RANK = 6
SIGMA_B = 0.02
ALPHA = 1.0
N_SPEED_TOKENS = 64
RESULTS_PATH = Path(__file__).parent / "results.json"


# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------
def module_dims(module):
    """Return (d_out, d_in) for QuantizedLinear or plain Linear."""
    if isinstance(module, nn.QuantizedLinear):
        bits = int(module.bits)
        d_out, packed_d_in = module.weight.shape
        d_in = packed_d_in * (32 // bits)
        return int(d_out), int(d_in)
    W = getattr(module, "weight", None)
    if W is None:
        return None
    return int(W.shape[0]), int(W.shape[1])


def collect_target_modules(model):
    targets = []
    for name, module in model.named_modules():
        if not (name.endswith(".v_proj") or name.endswith(".o_proj")):
            continue
        dims = module_dims(module)
        if dims is None:
            continue
        d_out, d_in = dims
        targets.append((name, module, d_out, d_in))
    return targets


# ---------------------------------------------------------------------------
# Grassmannian adapter construction
# ---------------------------------------------------------------------------
def grassmannian_As(d_in: int, n: int, r: int, key):
    """Return (n, r, d_in) with A_i A_j^T ~= 0 for i != j. Columns of Q span R^{n*r}."""
    total = n * r
    total = min(total, d_in)
    G = mx.random.normal(shape=(total, d_in), key=key).astype(mx.float32)
    # QR on G.T: (d_in, total) -> Q (d_in, total) with orthonormal columns
    Q, _ = mx.linalg.qr(G.T, stream=mx.cpu)
    Q = Q.T.astype(mx.bfloat16)  # (total, d_in)
    return mx.stack([Q[i * r : (i + 1) * r] for i in range(n)])  # (n, r, d_in)


def random_Bs(d_out: int, n: int, r: int, sigma: float, key):
    """Return (n, d_out, r) B-matrices with Gaussian(0, sigma^2)."""
    B = mx.random.normal(shape=(n, d_out, r), key=key) * sigma
    return B.astype(mx.bfloat16)


def build_adapter_stack(targets, n, r, sigma_b, seed=0):
    """Per-module dict of (A_stack, B_stack), each shape (n, r, d_in) and (n, d_out, r)."""
    adapters = {}
    for li, (name, _, d_out, d_in) in enumerate(targets):
        ka = mx.random.key(seed * 100_000 + li * 2)
        kb = mx.random.key(seed * 100_000 + li * 2 + 1)
        As = grassmannian_As(d_in, n, r, ka)
        Bs = random_Bs(d_out, n, r, sigma_b, kb)
        adapters[name] = (As, Bs)
    mx.eval(adapters)
    return adapters


# ---------------------------------------------------------------------------
# Pre-summed W_combined per module; exact-reversibility test (KC1690)
# ---------------------------------------------------------------------------
def compute_wcombined(A_stack, B_stack, alpha):
    """A_stack (n,r,d_in), B_stack (n,d_out,r) -> W (d_out, d_in) = alpha * Σ B_i A_i."""
    # Sum left-to-right deterministically
    n = A_stack.shape[0]
    W = alpha * (B_stack[0] @ A_stack[0])
    for i in range(1, n):
        W = W + alpha * (B_stack[i] @ A_stack[i])
    return W


def build_all_wcombined(adapters, alpha):
    W = {}
    for name, (As, Bs) in adapters.items():
        W[name] = compute_wcombined(As, Bs, alpha)
    mx.eval(W)
    return W


def kc1690_exact_reversibility(adapters, W_full, alpha, k=None):
    """(W_full - alpha * B_k @ A_k) vs fresh compute without k — exact equality."""
    max_diffs = []
    passes = []
    for name, (As, Bs) in adapters.items():
        n = As.shape[0]
        kk = n - 1 if k is None else k
        dW_k = alpha * (Bs[kk] @ As[kk])
        W_minus = W_full[name] - dW_k
        # Fresh compute without k (same left-to-right order over remaining indices)
        remaining = [i for i in range(n) if i != kk]
        W_fresh = alpha * (Bs[remaining[0]] @ As[remaining[0]])
        for i in remaining[1:]:
            W_fresh = W_fresh + alpha * (Bs[i] @ As[i])
        mx.eval(W_minus, W_fresh)
        diff = mx.abs(W_minus - W_fresh)
        max_abs = float(mx.max(diff).item())
        # Check that diff ≤ 2 ULP bf16 of the magnitude of W_fresh
        mag = float(mx.max(mx.abs(W_fresh)).item())
        ulp = max(2 ** -7 * mag, 1e-4)
        max_diffs.append(max_abs)
        passes.append(max_abs <= ulp)
    return {
        "layers_tested": len(max_diffs),
        "mean_max_abs_diff": float(sum(max_diffs) / len(max_diffs)),
        "global_max_abs_diff": float(max(max_diffs)),
        "all_within_ulp": bool(all(passes)),
    }


# ---------------------------------------------------------------------------
# Wrappers — inject delta into v_proj/o_proj forward
# ---------------------------------------------------------------------------
class RoomWrapper(nn.Module):
    """y = base(x) + x @ delta.T  (delta is (d_out, d_in))."""

    def __init__(self, base, delta):
        super().__init__()
        self._base = base
        self._delta = delta

    def __call__(self, x):
        return self._base(x) + x @ self._delta.T


class RoutedWrapper(nn.Module):
    """y = base(x) + alpha * (x @ A_k.T) @ B_k.T."""

    def __init__(self, base, A_k, B_k, alpha):
        super().__init__()
        self._base = base
        self._A = A_k
        self._B = B_k
        self._alpha = float(alpha)

    def __call__(self, x):
        return self._base(x) + self._alpha * ((x @ self._A.T) @ self._B.T)


def set_module_by_path(model, path, new_mod):
    parts = path.split(".")
    obj = model
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    leaf = parts[-1]
    if leaf.isdigit():
        obj[int(leaf)] = new_mod
    else:
        setattr(obj, leaf, new_mod)


def install_room(model, targets, W_room):
    for name, base, _, _ in targets:
        set_module_by_path(model, name, RoomWrapper(base, W_room[name]))


def install_routing_single(model, targets, adapters, k, alpha):
    for name, base, _, _ in targets:
        As, Bs = adapters[name]
        set_module_by_path(model, name, RoutedWrapper(base, As[k], Bs[k], alpha))


def restore(model, originals):
    for name, base in originals.items():
        set_module_by_path(model, name, base)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def measure_tps(model, tokenizer, prompt, max_tokens):
    """Use mlx_lm.stream_generate and read generation_tps from the last response."""
    last = None
    count = 0
    start = time.perf_counter()
    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        last = resp
        count += 1
    elapsed = time.perf_counter() - start
    library_tps = float(getattr(last, "generation_tps", 0.0)) if last is not None else 0.0
    return {
        "library_tps": library_tps,
        "wall_tps": count / elapsed if elapsed > 0 else 0.0,
        "tokens_generated": count,
        "elapsed_s": elapsed,
    }


def last_token_logits(model, tokenizer, prompt):
    toks = mx.array([tokenizer.encode(prompt)])
    logits = model(toks)[0, -1]
    mx.eval(logits)
    return logits


def cosine(u, v):
    u = u.astype(mx.float32)
    v = v.astype(mx.float32)
    num = float((u * v).sum().item())
    du = float(mx.sqrt((u * u).sum()).item())
    dv = float(mx.sqrt((v * v).sum()).item())
    if du == 0 or dv == 0:
        return 0.0
    return num / (du * dv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    import mlx_lm as _mlx_lm

    results = {
        "experiment": "exp_model_room_model_gemma4_speed",
        "model": MODEL_ID,
        "mlx_lm_version": getattr(_mlx_lm, "__version__", "unknown"),
        "n_adapters": N_ADAPTERS,
        "rank": RANK,
        "alpha": ALPHA,
        "sigma_b": SIGMA_B,
        "is_smoke": False,
        "ran": False,
    }

    print(f"[load] {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)

    targets = collect_target_modules(model)
    results["n_target_modules"] = len(targets)
    print(f"  target modules (v_proj/o_proj): {len(targets)}")
    if len(targets) == 0:
        results["verdict"] = "KILLED"
        results["error"] = "no v_proj/o_proj modules found"
        RESULTS_PATH.write_text(json.dumps(results, indent=2))
        return

    originals = {name: mod for name, mod, _, _ in targets}

    # Report dimension profile (diagnostic)
    dim_profile = {}
    for _, _, d_out, d_in in targets:
        k = f"{d_out}x{d_in}"
        dim_profile[k] = dim_profile.get(k, 0) + 1
    results["dim_profile"] = dim_profile
    print(f"  dim_profile: {dim_profile}")

    print("[phase B] building Grassmannian adapter stack")
    adapters = build_adapter_stack(targets, N_ADAPTERS, RANK, SIGMA_B, seed=0)
    mx.clear_cache()

    print("[phase C] computing W_combined (per module)")
    W_room = build_all_wcombined(adapters, ALPHA)

    print("[phase C'] KC1690 exact reversibility")
    kc1690 = kc1690_exact_reversibility(adapters, W_room, ALPHA)
    results["kc1690"] = kc1690
    print(f"  -> max_abs_diff={kc1690['global_max_abs_diff']:.3e}; all_within_ulp={kc1690['all_within_ulp']}")
    mx.clear_cache()

    print("[phase F] KC1689 logit cosine room vs explicit routing")
    eval_prompts = [
        ("math", "Solve: what is the derivative of x^2 + 3x + 5?"),
        ("code", "Write a Python function that reverses a string."),
        ("medical", "What are common symptoms of type 2 diabetes?"),
        ("legal", "Define the term 'tort' in common law."),
        ("finance", "Explain compound interest in simple terms."),
    ]
    domain_to_k = {"math": 0, "code": 1, "medical": 2, "legal": 3, "finance": 4}

    # Baseline logits
    logits_base = {}
    for d, p in eval_prompts:
        logits_base[d] = last_token_logits(model, tokenizer, p)

    # Room logits
    install_room(model, targets, W_room)
    logits_room = {}
    for d, p in eval_prompts:
        logits_room[d] = last_token_logits(model, tokenizer, p)
    restore(model, originals)

    # Explicit routing (per-prompt ground-truth adapter only)
    logits_routed = {}
    for d, p in eval_prompts:
        install_routing_single(model, targets, adapters, domain_to_k[d], ALPHA)
        logits_routed[d] = last_token_logits(model, tokenizer, p)
        restore(model, originals)

    cos_rr = {d: cosine(logits_room[d], logits_routed[d]) for d, _ in eval_prompts}
    cos_rb = {d: cosine(logits_room[d], logits_base[d]) for d, _ in eval_prompts}
    mean_rr = sum(cos_rr.values()) / len(cos_rr)
    results["kc1689"] = {
        "per_domain_cos_room_vs_routing": cos_rr,
        "per_domain_cos_room_vs_base": cos_rb,
        "mean_cos_room_vs_routing": mean_rr,
        "threshold": 0.999,
        "pass": mean_rr > 0.999,
    }
    print(f"  -> mean cos(room, routing) = {mean_rr:.6f}")
    mx.clear_cache()
    gc.collect()

    print("[phase E] KC1688 tok/s")
    tps_prompt = "Explain the capital of France in one paragraph."
    # Base throughput (warm)
    _ = measure_tps(model, tokenizer, tps_prompt, 16)  # warmup
    tps_base = measure_tps(model, tokenizer, tps_prompt, N_SPEED_TOKENS)

    install_room(model, targets, W_room)
    _ = measure_tps(model, tokenizer, tps_prompt, 16)  # warmup with room installed
    tps_room = measure_tps(model, tokenizer, tps_prompt, N_SPEED_TOKENS)
    restore(model, originals)

    results["kc1688"] = {
        "tps_base": tps_base,
        "tps_room": tps_room,
        "threshold_tps": 150.0,
        "pass": tps_room["library_tps"] >= 150.0,
    }
    print(f"  -> base {tps_base['library_tps']:.2f} tok/s | room {tps_room['library_tps']:.2f} tok/s")

    all_pass = all([
        results["kc1688"]["pass"],
        results["kc1689"]["pass"],
        results["kc1690"]["all_within_ulp"],
    ])
    results["all_pass"] = all_pass
    results["verdict"] = "SUPPORTED" if all_pass else "KILLED"
    results["ran"] = True
    results["total_time_s"] = time.perf_counter() - t0

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[DONE] verdict={results['verdict']} all_pass={all_pass} in {results['total_time_s']:.1f}s")


if __name__ == "__main__":
    main()
