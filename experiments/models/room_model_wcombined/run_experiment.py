#!/usr/bin/env python3
"""Room Model Piece A: W_combined = Σ ΔW_i from one matmul.

Sum all 5 SFT adapter deltas into a SINGLE matrix per module.
Inject once. Every domain should get its correct output because
the orthogonal deltas don't interfere.

Kill criteria:
  K802: Any domain PPL > 2x single-adapter PPL
  K803: Speed < 90 tok/s
"""

import gc, json, math, os, time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from pierre import attach_adapter, detach_adapters, load_adapter, load_frozen_A, encode, fit_router, route
from pierre.pierre import RuntimeLoRA, ADAPTER_TARGETS
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

SFT_SOURCE = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3" / "sft_adapters"
SKELETON_PATH = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "adapters" / "grassmannian_skeleton.npz"
DATA_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_SCALE = 20.0
MAX_SEQ = 256
SEED = 42
DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_TEST = 50


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

def log(m): print(m, flush=True)
def cleanup(*o):
    for x in o: del x
    gc.collect(); mx.clear_cache(); mx.reset_peak_memory()

def load_data(d, split="valid", n=None):
    s = []
    with open(DATA_DIR / d / f"{split}.jsonl") as f:
        for l in f:
            s.append(json.loads(l)["text"])
            if n and len(s) >= n: break
    return s

def compute_ppl(model, tok, texts):
    loss, n = 0.0, 0
    for text in texts:
        toks = tok.encode(text)[:MAX_SEQ]
        if len(toks) < 4: continue
        x = mx.array(toks)[None, :]
        logits = model(x); mx.eval(logits)
        targets = x[:, 1:]
        lp = mx.log(mx.softmax(logits[:, :-1, :], axis=-1) + 1e-10)
        tlp = mx.take_along_axis(lp, targets[:,:,None], axis=-1).squeeze(-1)
        mx.eval(tlp)
        loss += -tlp.sum().item(); n += targets.shape[1]
        del logits, lp, tlp, x
    return math.exp(loss / n) if n else float('inf')


# ── Room Model: compute and inject W_combined ────────────────────────────

def compute_w_combined(frozen_A, adapters, domains, alpha, n_layers):
    """Compute W_combined = Σ α · B_i^T @ A_i^T for all modules.

    Returns dict of (layer_idx, module_key) → combined delta matrix.
    """
    t0 = time.time()
    combined = {}
    for li in range(n_layers):
        for key in ADAPTER_TARGETS:
            delta_sum = None
            for di, d in enumerate(domains):
                bk = f"model.layers.{li}.{key}.lora_b"
                ak = f"layer_{li}_{key}_domain_{di}"
                if bk not in adapters[d] or ak not in frozen_A:
                    continue
                A = mx.array(frozen_A[ak]).astype(mx.bfloat16)
                B = adapters[d][bk].astype(mx.bfloat16)
                delta = alpha * (B.T @ A.T)  # (out, in)
                if delta_sum is None:
                    delta_sum = delta
                else:
                    delta_sum = delta_sum + delta
            if delta_sum is not None:
                combined[(li, key)] = delta_sum

    # Eval all at once
    mx.eval(*combined.values())
    dt = time.time() - t0

    mem_mb = sum(v.size * 2 for v in combined.values()) / 1e6
    log(f"  W_combined: {len(combined)} modules, {mem_mb:.0f}MB, computed in {dt:.2f}s")
    return combined


class WCombinedLayer(nn.Module):
    """Base module + pre-baked W_combined delta. One matmul for ALL domains."""
    def __init__(self, base, w_combined_transposed):
        super().__init__()
        self.base = base
        # Store as (in, out) for x @ W convention
        self._delta = w_combined_transposed
        self.freeze()

    def __call__(self, x):
        y = self.base(x)
        return y + (x @ self._delta).astype(y.dtype)


def inject_w_combined(model, w_combined):
    """Inject W_combined into model. One extra matmul per module for ALL domains."""
    from mlx.utils import tree_unflatten
    count = 0
    for li in range(len(model.model.layers)):
        updates = []
        for key in ADAPTER_TARGETS:
            if (li, key) not in w_combined:
                continue
            m = model.model.layers[li]
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None: break
            if m is None: continue

            # W_combined is (out, in). For x @ W we need (in, out).
            delta_T = w_combined[(li, key)].T
            wrapped = WCombinedLayer(m, delta_T)
            updates.append((key, wrapped))
            count += 1

        if updates:
            model.model.layers[li].update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())
    return count


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("Room Model Piece A: W_combined = Σ ΔW_i")
    log("=" * 60)
    mx.random.seed(SEED)

    frozen_A = load_frozen_A(str(SKELETON_PATH))
    adapters = {d: load_adapter(str(SFT_SOURCE / d / "adapter.npz")) for d in DOMAINS}
    val = {d: load_data(d, "valid", N_TEST) for d in DOMAINS}

    # Phase 1: Single-adapter PPL (baseline)
    log("\n=== Phase 1: Single-adapter PPL (baseline) ===")
    single_ppls = {}
    for di, d in enumerate(DOMAINS):
        model, tok = load(MODEL_ID)
        attach_adapter(model, frozen_A, adapters[d], di, LORA_SCALE)
        single_ppls[d] = round(compute_ppl(model, tok, val[d]), 3)
        log(f"  single/{d}: {single_ppls[d]}")
        cleanup(model, tok)

    # Phase 2: W_combined PPL (room model)
    log("\n=== Phase 2: W_combined PPL (all domains in one matrix) ===")
    model, tok = load(MODEL_ID)
    n_layers = len(model.model.layers)
    w_combined = compute_w_combined(frozen_A, adapters, DOMAINS, LORA_SCALE, n_layers)
    n_modules = inject_w_combined(model, w_combined)
    log(f"  Injected W_combined into {n_modules} modules")

    room_ppls = {}
    for d in DOMAINS:
        room_ppls[d] = round(compute_ppl(model, tok, val[d]), 3)
        ratio = room_ppls[d] / single_ppls[d]
        log(f"  room/{d}: {room_ppls[d]} (ratio vs single: {ratio:.3f}x)")

    # Phase 3: Speed
    log("\n=== Phase 3: Speed ===")
    prompt = "### Instruction:\nExplain photosynthesis.\n\n### Response:\n"
    sampler = make_sampler(temp=0.0)
    for _ in range(3):
        mlx_generate(model, tok, prompt=prompt, max_tokens=32, sampler=sampler, verbose=False)
    times = []
    for _ in range(5):
        t1 = time.time()
        out = mlx_generate(model, tok, prompt=prompt, max_tokens=128, sampler=sampler, verbose=False)
        dt = time.time() - t1
        n = len(tok.encode(out)) - len(tok.encode(prompt))
        times.append({"s": dt, "toks": n})
    tps = sum(t["toks"] for t in times) / sum(t["s"] for t in times)
    log(f"  Room model speed: {tps:.1f} tok/s")
    cleanup(model, tok, w_combined)

    # Phase 4: Compare against base (no adapter)
    log("\n=== Phase 4: Base model (no adapter) ===")
    model, tok = load(MODEL_ID)
    base_ppls = {}
    for d in DOMAINS:
        base_ppls[d] = round(compute_ppl(model, tok, val[d]), 3)
        log(f"  base/{d}: {base_ppls[d]}")

    for _ in range(3):
        mlx_generate(model, tok, prompt=prompt, max_tokens=32, sampler=sampler, verbose=False)
    base_times = []
    for _ in range(5):
        t1 = time.time()
        out = mlx_generate(model, tok, prompt=prompt, max_tokens=128, sampler=sampler, verbose=False)
        dt = time.time() - t1
        n = len(tok.encode(out)) - len(tok.encode(prompt))
        base_times.append({"s": dt, "toks": n})
    base_tps = sum(t["toks"] for t in base_times) / sum(t["s"] for t in base_times)
    log(f"  Base speed: {base_tps:.1f} tok/s")
    cleanup(model, tok)

    # Results
    results = {
        "experiment": "room_model_wcombined",
        "total_time_s": round(time.time() - t0, 1),
        "base_ppl": base_ppls,
        "single_adapter_ppl": single_ppls,
        "room_model_ppl": room_ppls,
        "ppl_ratios": {d: round(room_ppls[d] / single_ppls[d], 3) for d in DOMAINS},
        "speed": {"room_tps": round(tps, 1), "base_tps": round(base_tps, 1),
                  "overhead_pct": round((base_tps - tps) / base_tps * 100, 2)},
        "n_modules": n_modules,
    }

    worst_ratio = max(results["ppl_ratios"].values())
    k802 = worst_ratio <= 2.0
    k803 = tps >= 90.0

    results["kill_criteria"] = {
        "K802": {"pass": k802, "value": worst_ratio, "threshold": 2.0},
        "K803": {"pass": k803, "value": round(tps, 1), "threshold": 90.0},
    }
    results["all_pass"] = k802 and k803

    log(f"\n{'='*60}")
    log("Per-domain PPL comparison:")
    log(f"  {'Domain':<12} {'Base':>8} {'Single':>8} {'Room':>8} {'Ratio':>8}")
    for d in DOMAINS:
        log(f"  {d:<12} {base_ppls[d]:>8.3f} {single_ppls[d]:>8.3f} {room_ppls[d]:>8.3f} {results['ppl_ratios'][d]:>8.3f}x")
    log(f"\nSpeed: room={tps:.1f} tok/s, base={base_tps:.1f} tok/s, overhead={results['speed']['overhead_pct']}%")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} — {v}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
