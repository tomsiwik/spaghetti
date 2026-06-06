#!/usr/bin/env python3
"""Multi-tenant serving: different adapter stacks per user, shared base.

Simulates 4 users with different domain adapters on one loaded model.
Measures swap latency, memory, throughput.

Kill criteria:
  K830: Adapter swap > 1s per request
  K831: Memory per user > 5GB
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

from pierre import attach_adapter, detach_adapters, load_adapter, load_frozen_A
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

SFT_SOURCE = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3" / "sft_adapters"
SKELETON_PATH = EXPERIMENT_DIR.parent / "real_data_domain_experts" / "adapters" / "grassmannian_skeleton.npz"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_SCALE = 20.0
SEED = 42


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


# Simulate 4 users with different domain needs
USERS = [
    {"name": "doctor", "domain": "medical", "di": 0,
     "prompt": "### Instruction:\nWhat are the symptoms of diabetes?\n\n### Response:\n"},
    {"name": "developer", "domain": "code", "di": 1,
     "prompt": "### Instruction:\nWrite a Python function to sort a list.\n\n### Response:\n"},
    {"name": "student", "domain": "math", "di": 2,
     "prompt": "### Instruction:\nSolve: what is the integral of x^2?\n\n### Response:\n"},
    {"name": "lawyer", "domain": "legal", "di": 3,
     "prompt": "### Instruction:\nExplain the concept of habeas corpus.\n\n### Response:\n"},
]


def main():
    t0 = time.time()
    log("Multi-Tenant Serving: 4 Users, Shared Base")
    log("=" * 60)
    mx.random.seed(SEED)

    frozen_A = load_frozen_A(str(SKELETON_PATH))

    # Pre-load all adapters into memory
    adapters = {}
    for user in USERS:
        adapters[user["domain"]] = load_adapter(str(SFT_SOURCE / user["domain"] / "adapter.npz"))

    adapter_mem = sum(sum(v.size * 2 for v in a.values()) for a in adapters.values()) / 1e6
    log(f"All adapters loaded: {adapter_mem:.1f}MB total")

    # Load model once
    model, tok = load(MODEL_ID)
    base_mem = mx.get_active_memory() / 1e9
    log(f"Base model memory: {base_mem:.2f}GB")

    sampler = make_sampler(temp=0.0)
    results = {"users": {}, "swap_times": [], "generation_times": []}

    # Simulate round-robin serving: each user gets one request
    log("\n=== Simulating 4 users ===")
    for round_num in range(3):  # 3 rounds
        log(f"\n  Round {round_num + 1}:")
        for user in USERS:
            # Detach previous adapter
            t_swap_start = time.time()
            detach_adapters(model)

            # Attach this user's adapter
            attach_adapter(model, frozen_A, adapters[user["domain"]], user["di"], LORA_SCALE)
            swap_time = time.time() - t_swap_start

            # Generate
            t_gen_start = time.time()
            out = mlx_generate(model, tok, prompt=user["prompt"],
                               max_tokens=64, sampler=sampler, verbose=False)
            gen_time = time.time() - t_gen_start
            n_toks = len(tok.encode(out)) - len(tok.encode(user["prompt"]))

            tps = n_toks / gen_time if gen_time > 0 else 0
            log(f"    {user['name']}: swap={swap_time*1000:.0f}ms gen={gen_time:.2f}s ({tps:.0f} tok/s)")

            results["swap_times"].append(swap_time)
            results["generation_times"].append(gen_time)

            if user["name"] not in results["users"]:
                results["users"][user["name"]] = {"swaps": [], "gens": [], "tps": []}
            results["users"][user["name"]]["swaps"].append(round(swap_time, 4))
            results["users"][user["name"]]["gens"].append(round(gen_time, 3))
            results["users"][user["name"]]["tps"].append(round(tps, 1))

    # Memory measurement
    peak_mem = mx.get_peak_memory() / 1e9
    per_user_mem = (peak_mem - base_mem) / len(USERS)

    cleanup(model, tok)

    # Aggregate
    mean_swap = float(np.mean(results["swap_times"]))
    max_swap = float(np.max(results["swap_times"]))
    mean_tps = float(np.mean([t for u in results["users"].values() for t in u["tps"]]))

    results["summary"] = {
        "mean_swap_s": round(mean_swap, 4),
        "max_swap_s": round(max_swap, 4),
        "mean_tps": round(mean_tps, 1),
        "base_memory_gb": round(base_mem, 2),
        "peak_memory_gb": round(peak_mem, 2),
        "per_user_memory_gb": round(per_user_mem, 2),
        "adapter_memory_mb": round(adapter_mem, 1),
    }
    results["total_time_s"] = round(time.time() - t0, 1)

    k830 = max_swap <= 1.0
    k831 = per_user_mem <= 5.0

    results["kill_criteria"] = {
        "K830": {"pass": k830, "value": round(max_swap, 4), "threshold": 1.0},
        "K831": {"pass": k831, "value": round(per_user_mem, 2), "threshold": 5.0},
    }
    results["all_pass"] = k830 and k831

    log(f"\n{'='*60}")
    log(f"Swap: mean={mean_swap*1000:.0f}ms max={max_swap*1000:.0f}ms")
    log(f"Speed: {mean_tps:.1f} tok/s")
    log(f"Memory: base={base_mem:.2f}GB peak={peak_mem:.2f}GB per_user={per_user_mem:.2f}GB")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} — {v}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
