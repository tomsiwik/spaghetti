#!/usr/bin/env python3
"""Pierre v6.2: hybrid precomputed attention + factored MLP.

Combines the optimal parts of v6 and v6.1:
  - Attention (QKV, O): precomputed concat deltas (2 dispatches/layer)
  - MLP (gate, up, down): factored RuntimeLoRA (6 dispatches/layer)
  Total: 8 dispatches/layer x 30 = 240 dispatches

Kill criteria:
  K759: Speed >= 75 tok/s
  K760: Code behavioral >= 0.80
  K761: Overall behavioral >= 0.35
  K762: Peak memory <= 6.0 GB
"""

import ast
import gc
import json
import math
import os
import re
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from pierre import load_adapter, load_frozen_A, encode, fit_router, route
from pierre.pierre import RuntimeLoRA, ADAPTER_TARGETS, _mask_cache
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

NTP_SOURCE = EXPERIMENT_DIR.parent / "real_data_domain_experts"
SFT_SOURCE = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3" / "sft_adapters"
SKELETON_PATH = NTP_SOURCE / "adapters" / "grassmannian_skeleton.npz"
DATA_DIR = NTP_SOURCE / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_SCALE = 20.0
MAX_SEQ = 256
SEED = 42
DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_CAL, N_TEST, N_GEN, MAX_TOK = 50, 50, 5, 128

# Attention groups for precomputed concat deltas
ATTN_QKV = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
ATTN_O = ["self_attn.o_proj"]
ATTN_GROUPS = [("qkv", ATTN_QKV), ("o", ATTN_O)]

# MLP modules for factored RuntimeLoRA
MLP_MODULES = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


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

STOP_WORDS = {'the','a','an','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might','can',
    'to','of','in','for','on','with','at','by','from','as','and','but','or','not',
    'so','yet','both','either','each','every','all','any','few','more','most','other',
    'some','such','no','only','own','same','than','too','very','just','because','if',
    'when','where','how','what','which','who','this','that','these','those','it','its',
    'i','me','my','we','our','you','your','he','him','his','she','her','they','them','their'}

def factual_recall(g, r):
    def t(x): return set(w for w in re.findall(r'\b[a-z]+\b', x.lower()) if w not in STOP_WORDS and len(w)>2)
    gt, rt = t(g), t(r)
    return len(gt & rt) / len(rt) if rt else 0.0

def eval_response(g, r, d):
    if d == "code":
        blocks = re.findall(r'```(?:python)?\s*\n(.*?)\n```', g, re.DOTALL)
        code = '\n'.join(blocks) if blocks else '\n'.join(l for l in g.split('\n') if l.strip() and not l.startswith('#'))
        try: ast.parse(code); ok=True
        except SyntaxError: ok=False
        return 0.7*float(ok) + 0.3*factual_recall(g,r)
    return factual_recall(g, r)


# -- Precomputed concat delta for attention modules --------------------------

class _GroupCache:
    """Shared cache for concatenated group matmul."""
    result = None

class ConcatDeltaLinear(nn.Module):
    """Precomputed delta with group concatenation.

    For a group of modules sharing the same input (e.g., QKV):
    - First module computes x @ DeltaW_concat (one dispatch for all)
    - Other modules slice from the cached result

    For single-module groups (O): just x @ DeltaW (one dispatch).
    """
    def __init__(self, base, concat_delta, my_slice, role, cache):
        super().__init__()
        self.base = base
        self._delta = concat_delta     # (in, out_total) shared
        self._slice = my_slice
        self._role = role              # "first" computes, others read cache
        self._cache = cache
        self.freeze()

    def __call__(self, x):
        y = self.base(x)
        if self._role == "first":
            self._cache.result = x @ self._delta
        correction = self._cache.result[..., self._slice]
        return y + correction.astype(y.dtype)


# -- Hybrid injection: precomputed attention + factored MLP ------------------

def inject_hybrid(model, frozen_A, adapter_B, domain_idx, scale):
    """Inject hybrid adapters: precomputed attention + factored MLP.

    Attention modules (QKV, O): ConcatDeltaLinear (precomputed DeltaW = scale * A @ B)
    MLP modules (gate, up, down): RuntimeLoRA (factored x@A, h@B)

    Returns dict with dispatch counts and memory stats.
    """
    from mlx.utils import tree_unflatten

    attn_dispatches = 0
    mlp_dispatches = 0
    n_layers = len(model.model.layers)

    for li in range(n_layers):
        layer = model.model.layers[li]
        updates = []

        # --- Attention: precomputed concat deltas ---
        for group_name, group_keys in ATTN_GROUPS:
            deltas = []
            valid_keys = []
            for key in group_keys:
                bk = f"model.layers.{li}.{key}.lora_b"
                ak = f"layer_{li}_{key}_domain_{domain_idx}"
                if bk in adapter_B and ak in frozen_A:
                    A = mx.array(frozen_A[ak]).astype(mx.bfloat16)
                    B = adapter_B[bk].astype(mx.bfloat16)
                    delta_W = (scale * (B.T @ A.T)).T  # (in, out) for x @ DeltaW
                    deltas.append(delta_W)
                    valid_keys.append(key)

            if not deltas:
                continue

            if len(deltas) == 1:
                # Single module group (O)
                key = valid_keys[0]
                m = layer
                for part in key.split("."):
                    m = getattr(m, part, None)
                if m is None: continue
                cache = _GroupCache()
                wrapped = ConcatDeltaLinear(m, deltas[0], slice(None), "first", cache)
                updates.append((key, wrapped))
                attn_dispatches += 1
            else:
                # Multi-module group (QKV)
                concat_delta = mx.concatenate(deltas, axis=1)
                mx.eval(concat_delta)
                splits = [d.shape[1] for d in deltas]
                cumsum = [0] + [sum(splits[:i+1]) for i in range(len(splits))]
                cache = _GroupCache()

                for i, key in enumerate(valid_keys):
                    m = layer
                    for part in key.split("."):
                        m = getattr(m, part, None)
                    if m is None: continue
                    s = slice(cumsum[i], cumsum[i+1])
                    role = "first" if i == 0 else "read"
                    wrapped = ConcatDeltaLinear(m, concat_delta, s, role, cache)
                    updates.append((key, wrapped))

                attn_dispatches += 1  # one dispatch for the whole group

        # --- MLP: factored RuntimeLoRA ---
        for key in MLP_MODULES:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_B or ak not in frozen_A:
                continue

            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue

            A = mx.array(frozen_A[ak]).astype(mx.bfloat16)
            B = adapter_B[bk].astype(mx.bfloat16)
            wrapped = RuntimeLoRA(m, A, B, scale)
            updates.append((key, wrapped))
            mlp_dispatches += 2  # x@A and h@B per module

        if updates:
            layer.update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())

    total_dispatches = attn_dispatches + mlp_dispatches
    log(f"  Injected: {attn_dispatches} attn dispatches (precomputed) + "
        f"{mlp_dispatches} MLP dispatches (factored) = {total_dispatches} total")

    return {
        "attn_dispatches": attn_dispatches,
        "mlp_dispatches": mlp_dispatches,
        "total_dispatches": total_dispatches,
    }


# -- Phases ------------------------------------------------------------------

def phase_calibrate():
    log("\n=== Phase 1: Router ===")
    model, tok = load(MODEL_ID)
    W = fit_router(model, tok, {d: load_data(d, "train", N_CAL) for d in DOMAINS}, max_seq=MAX_SEQ)
    correct, total = 0, 0
    for di, d in enumerate(DOMAINS):
        dc = sum(1 for t in load_data(d, "valid", N_TEST) if route(model, tok, t, W, MAX_SEQ) == di)
        correct += dc; total += N_TEST
    acc = correct / total
    log(f"  Routing: {acc:.1%}")
    np.save(str(EXPERIMENT_DIR / "router_W.npy"), np.array(W))
    cleanup(model, tok)
    return {"accuracy": round(acc, 4)}


def phase_behavioral():
    log("\n=== Phase 2: Behavioral ===")
    frozen_A = load_frozen_A(str(SKELETON_PATH))
    W = mx.array(np.load(str(EXPERIMENT_DIR / "router_W.npy")))
    results = {"per_domain": {}}

    for di, d in enumerate(DOMAINS):
        model, tok = load(MODEL_ID)
        test = load_data(d, "valid", N_GEN)
        ri = route(model, tok, test[0], W, MAX_SEQ)
        rd = DOMAINS[ri]
        adapter = load_adapter(str(SFT_SOURCE / rd / "adapter.npz"))
        stats = inject_hybrid(model, frozen_A, adapter, ri, LORA_SCALE)

        scores = []
        sampler = make_sampler(temp=0.0)
        for text in test:
            if "### Response:" in text:
                prompt = text.split("### Response:")[0].strip() + "\n### Response:\n"
                ref = text.split("### Response:")[-1].strip()
            else:
                prompt, ref = text[:200], text
            try:
                gen = mlx_generate(model, tok, prompt=prompt, max_tokens=MAX_TOK, sampler=sampler, verbose=False)
            except Exception:
                gen = ""
            scores.append(eval_response(gen, ref, d))

        mean = float(np.mean(scores)) if scores else 0.0
        results["per_domain"][d] = {
            "score": round(mean, 3),
            "routed_to": rd,
            "dispatches": stats["total_dispatches"],
            "attn_dispatches": stats["attn_dispatches"],
            "mlp_dispatches": stats["mlp_dispatches"],
        }
        log(f"  {d} -> {rd}: {mean:.3f} ({stats['total_dispatches']} dispatches: "
            f"{stats['attn_dispatches']} attn + {stats['mlp_dispatches']} MLP)")
        cleanup(model, tok, adapter)

    results["overall"] = round(float(np.mean([v["score"] for v in results["per_domain"].values()])), 3)
    results["code_behavioral"] = results["per_domain"].get("code", {}).get("score", 0.0)
    log(f"  Overall: {results['overall']}")
    log(f"  Code: {results['code_behavioral']}")
    return results


def phase_latency():
    log("\n=== Phase 3: Latency ===")
    frozen_A = load_frozen_A(str(SKELETON_PATH))
    prompt = "### Instruction:\nExplain photosynthesis.\n\n### Response:\n"
    sampler = make_sampler(temp=0.0)

    def bench(model, tok, label):
        # Warmup
        for _ in range(3):
            mlx_generate(model, tok, prompt=prompt, max_tokens=32, sampler=sampler, verbose=False)
        mx.reset_peak_memory()
        times = []
        for _ in range(5):
            t0 = time.time()
            out = mlx_generate(model, tok, prompt=prompt, max_tokens=MAX_TOK, sampler=sampler, verbose=False)
            dt = time.time() - t0
            n = len(tok.encode(out)) - len(tok.encode(prompt))
            times.append({"s": dt, "toks": n})
        tps = sum(t["toks"] for t in times) / sum(t["s"] for t in times)
        peak = mx.get_peak_memory() / 1e9
        log(f"  {label}: {tps:.1f} tok/s, peak={peak:.2f}GB")
        return round(tps, 1), round(peak, 2)

    # Base model (no adapters)
    model, tok = load(MODEL_ID)
    base_tps, base_mem = bench(model, tok, "native BitLinear")
    cleanup(model, tok)

    # Hybrid: precomputed attention + factored MLP
    model, tok = load(MODEL_ID)
    adapter = load_adapter(str(SFT_SOURCE / "medical" / "adapter.npz"))
    stats = inject_hybrid(model, frozen_A, adapter, 0, LORA_SCALE)
    hybrid_tps, hybrid_mem = bench(model, tok,
        f"v6.2 hybrid ({stats['total_dispatches']} dispatches: "
        f"{stats['attn_dispatches']} attn + {stats['mlp_dispatches']} MLP)")
    overhead = round((base_tps - hybrid_tps) / base_tps * 100, 2)
    log(f"  Overhead: {overhead}%")
    cleanup(model, tok, adapter)

    return {
        "base_tps": base_tps,
        "hybrid_tps": hybrid_tps,
        "overhead_pct": overhead,
        "base_peak_gb": base_mem,
        "hybrid_peak_gb": hybrid_mem,
        "attn_dispatches": stats["attn_dispatches"],
        "mlp_dispatches": stats["mlp_dispatches"],
        "total_dispatches": stats["total_dispatches"],
    }


def main():
    t0 = time.time()
    log("Pierre v6.2 -- Hybrid Precomputed Attention + Factored MLP")
    log("=" * 60)
    mx.random.seed(SEED)

    r1 = phase_calibrate()
    r2 = phase_behavioral()
    r3 = phase_latency()

    # Kill criteria assessment
    k759 = r3["hybrid_tps"] >= 75.0
    k760 = r2["code_behavioral"] >= 0.80
    k761 = r2["overall"] >= 0.35
    k762 = r3["hybrid_peak_gb"] <= 6.0

    results = {
        "experiment": "pierre_v62_hybrid",
        "total_time_s": round(time.time() - t0, 1),
        "routing": r1,
        "behavioral": r2,
        "latency": r3,
        "kill_criteria": {
            "K759": {"pass": k759, "value": r3["hybrid_tps"], "threshold": 75.0,
                     "description": "Speed >= 75 tok/s"},
            "K760": {"pass": k760, "value": r2["code_behavioral"], "threshold": 0.80,
                     "description": "Code behavioral >= 0.80"},
            "K761": {"pass": k761, "value": r2["overall"], "threshold": 0.35,
                     "description": "Overall behavioral >= 0.35"},
            "K762": {"pass": k762, "value": r3["hybrid_peak_gb"], "threshold": 6.0,
                     "description": "Peak memory <= 6.0 GB"},
        },
        "all_pass": k759 and k760 and k761 and k762,
        "comparison": {
            "v3_factored": "73 tok/s, 420 dispatches, behavioral 0.41, code 0.844",
            "v6_attn_only": "86.8 tok/s, 60 dispatches, behavioral 0.315, code 0.281",
            "v6.1_full_precomp": "42.1 tok/s, 120 dispatches, behavioral 0.419, code 0.844",
            "v6.2_hybrid": (f"{r3['hybrid_tps']} tok/s, {r3['total_dispatches']} dispatches, "
                           f"behavioral {r2['overall']}, code {r2['code_behavioral']}"),
            "native_bitlinear": f"{r3['base_tps']} tok/s",
        },
        "predictions_vs_actual": {
            "dispatches": {"predicted": 240, "actual": r3["total_dispatches"]},
            "speed_tok_s": {"predicted": "80-90", "actual": r3["hybrid_tps"]},
            "code_behavioral": {"predicted": 0.84, "actual": r2["code_behavioral"]},
            "overall_behavioral": {"predicted": 0.41, "actual": r2["overall"]},
            "peak_memory_gb": {"predicted": "2.5-3.5", "actual": r3["hybrid_peak_gb"]},
        },
    }

    log(f"\n{'='*60}")
    log("Version comparison:")
    for k, v in results["comparison"].items():
        log(f"  {k}: {v}")
    log(f"\nPredictions vs actual:")
    for k, v in results["predictions_vs_actual"].items():
        log(f"  {k}: predicted={v['predicted']}, actual={v['actual']}")
    log(f"\nKill criteria:")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} -- {v['description']}: {v['value']}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'} in {results['total_time_s']}s")
    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
