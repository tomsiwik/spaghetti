#!/usr/bin/env python3
"""Pierre v6.1: precomputed concat deltas with ALL modules (QKV+GateUp groups).

v6 was attention-only (60 dispatches, 87 tok/s) but code domain suffered.
v6.1 restores MLP gate+up adapters via concatenation:

  Group 1: QKV concat (3→1 dispatch)
  Group 2: O (1 dispatch)
  Group 3: Gate+Up concat (2→1 dispatch)
  Group 4: Down (1 dispatch)
  Total: 4 dispatches × 30 layers = 120

Kill criteria:
  K756: Speed < 75 tok/s (must beat v3's 73)
  K757: Behavioral < 0.35
  K758: Memory > 6GB peak
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

# Group definitions for concatenation
# Modules sharing the same input can be concatenated
ATTN_QKV = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
ATTN_O = ["self_attn.o_proj"]
MLP_GATE_UP = ["mlp.gate_proj", "mlp.up_proj"]
MLP_DOWN = ["mlp.down_proj"]

ALL_GROUPS = [
    ("qkv", ATTN_QKV),
    ("o", ATTN_O),
    ("gate_up", MLP_GATE_UP),
    ("down", MLP_DOWN),
]


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


# ── Precomputed concat delta injection ───────────────────────────────────

class _GroupCache:
    """Shared cache for concatenated group matmul."""
    result = None

class ConcatDeltaLinear(nn.Module):
    """Precomputed delta with group concatenation.

    For a group of modules sharing the same input (e.g., QKV):
    - First module computes x @ ΔW_concat (one dispatch for all)
    - Other modules slice from the cached result

    For single-module groups (O, Down): just x @ ΔW (one dispatch).
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


def inject_precomputed_full(model, frozen_A, adapter_B, domain_idx, scale):
    """Inject precomputed concatenated deltas for ALL modules (4 groups per layer)."""
    from mlx.utils import tree_unflatten

    dispatch_count = 0
    n_layers = len(model.model.layers)

    for li in range(n_layers):
        layer = model.model.layers[li]
        updates = []

        for group_name, group_keys in ALL_GROUPS:
            # Precompute ΔW = scale * B^T @ A^T for each module, then concatenate
            deltas = []
            valid_keys = []
            for key in group_keys:
                bk = f"model.layers.{li}.{key}.lora_b"
                ak = f"layer_{li}_{key}_domain_{domain_idx}"
                if bk in adapter_B and ak in frozen_A:
                    A = mx.array(frozen_A[ak]).astype(mx.bfloat16)
                    B = adapter_B[bk].astype(mx.bfloat16)
                    delta_W = (scale * (B.T @ A.T)).T  # (in, out) for x @ ΔW
                    deltas.append(delta_W)
                    valid_keys.append(key)

            if not deltas:
                continue

            if len(deltas) == 1:
                # Single module group — no concatenation needed
                key = valid_keys[0]
                m = layer
                for part in key.split("."):
                    m = getattr(m, part, None)
                if m is None: continue
                cache = _GroupCache()
                wrapped = ConcatDeltaLinear(m, deltas[0], slice(None), "first", cache)
                updates.append((key, wrapped))
                dispatch_count += 1
            else:
                # Multi-module group — concatenate along output dimension
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

                dispatch_count += 1  # one dispatch for the whole group

        if updates:
            layer.update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())
    return dispatch_count


# ── Phases ───────────────────────────────────────────────────────────────

def phase_calibrate():
    log("\n=== Phase 1: Router ===")
    model, tok = load(MODEL_ID)
    W = fit_router(model, tok, {d: load_data(d,"train",N_CAL) for d in DOMAINS}, max_seq=MAX_SEQ)
    correct, total = 0, 0
    for di, d in enumerate(DOMAINS):
        dc = sum(1 for t in load_data(d,"valid",N_TEST) if route(model,tok,t,W,MAX_SEQ)==di)
        correct += dc; total += N_TEST
    log(f"  Routing: {correct/total:.1%}")
    np.save(str(EXPERIMENT_DIR/"router_W.npy"), np.array(W))
    cleanup(model, tok)
    return {"accuracy": round(correct/total, 4)}


def phase_behavioral():
    log("\n=== Phase 2: Behavioral ===")
    frozen_A = load_frozen_A(str(SKELETON_PATH))
    W = mx.array(np.load(str(EXPERIMENT_DIR/"router_W.npy")))
    results = {"per_domain": {}}

    for di, d in enumerate(DOMAINS):
        model, tok = load(MODEL_ID)
        test = load_data(d,"valid",N_GEN)
        ri = route(model,tok,test[0],W,MAX_SEQ)
        rd = DOMAINS[ri]
        adapter = load_adapter(str(SFT_SOURCE/rd/"adapter.npz"))
        n = inject_precomputed_full(model, frozen_A, adapter, ri, LORA_SCALE)

        scores = []
        sampler = make_sampler(temp=0.0)
        for text in test:
            if "### Response:" in text:
                prompt = text.split("### Response:")[0].strip()+"\n### Response:\n"
                ref = text.split("### Response:")[-1].strip()
            else: prompt, ref = text[:200], text
            try: gen = mlx_generate(model,tok,prompt=prompt,max_tokens=MAX_TOK,sampler=sampler,verbose=False)
            except: gen = ""
            scores.append(eval_response(gen, ref, d))

        mean = float(np.mean(scores)) if scores else 0.0
        results["per_domain"][d] = {"score": round(mean,3), "routed_to": rd, "dispatches": n}
        log(f"  {d} → {rd}: {mean:.3f} ({n} dispatches)")
        cleanup(model, tok, adapter)

    results["overall"] = round(float(np.mean([v["score"] for v in results["per_domain"].values()])), 3)
    log(f"  Overall: {results['overall']}")
    return results


def phase_latency():
    log("\n=== Phase 3: Latency ===")
    frozen_A = load_frozen_A(str(SKELETON_PATH))
    prompt = "### Instruction:\nExplain photosynthesis.\n\n### Response:\n"
    sampler = make_sampler(temp=0.0)

    def bench(model, tok, label):
        for _ in range(3):
            mlx_generate(model,tok,prompt=prompt,max_tokens=32,sampler=sampler,verbose=False)
        mx.reset_peak_memory()
        times = []
        for _ in range(5):
            t0 = time.time()
            out = mlx_generate(model,tok,prompt=prompt,max_tokens=MAX_TOK,sampler=sampler,verbose=False)
            dt = time.time()-t0
            n = len(tok.encode(out))-len(tok.encode(prompt))
            times.append({"s":dt,"toks":n})
        tps = sum(t["toks"] for t in times)/sum(t["s"] for t in times)
        peak = mx.get_peak_memory()/1e9
        log(f"  {label}: {tps:.1f} tok/s, peak={peak:.2f}GB")
        return round(tps, 1), round(peak, 2)

    model, tok = load(MODEL_ID)
    base_tps, _ = bench(model,tok,"native BitLinear")
    cleanup(model, tok)

    model, tok = load(MODEL_ID)
    adapter = load_adapter(str(SFT_SOURCE/"medical"/"adapter.npz"))
    n = inject_precomputed_full(model, frozen_A, adapter, 0, LORA_SCALE)
    v61_tps, v61_mem = bench(model,tok,f"v6.1 full precomputed ({n} dispatches)")
    overhead = round((base_tps-v61_tps)/base_tps*100, 2)
    log(f"  Overhead: {overhead}%")
    cleanup(model, tok, adapter)

    return {"base_tps": base_tps, "v61_tps": v61_tps, "overhead_pct": overhead,
            "peak_gb": v61_mem, "dispatch_count": n}


def main():
    t0 = time.time()
    log("Pierre v6.1 — Full Precomputed Concat")
    log("=" * 60)
    mx.random.seed(SEED)

    r1 = phase_calibrate()
    r2 = phase_behavioral()
    r3 = phase_latency()

    results = {
        "experiment": "pierre_v6_full_precomputed",
        "total_time_s": round(time.time()-t0, 1),
        "routing": r1, "behavioral": r2, "latency": r3,
        "kill_criteria": {
            "K756": {"pass": r3["v61_tps"] >= 75.0, "value": r3["v61_tps"]},
            "K757": {"pass": r2["overall"] >= 0.35, "value": r2["overall"]},
            "K758": {"pass": r3["peak_gb"] <= 6.0, "value": r3["peak_gb"]},
        },
        "all_pass": r3["v61_tps"] >= 75.0 and r2["overall"] >= 0.35 and r3["peak_gb"] <= 6.0,
    }

    log(f"\n{'='*60}")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} — {v}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'} in {results['total_time_s']}s")
    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
