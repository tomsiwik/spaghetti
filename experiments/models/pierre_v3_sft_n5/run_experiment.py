#!/usr/bin/env python3
"""Pierre v3: SFT adapters + BitLinear side-path at N=5.

Key changes from v2:
  1. SFT adapters (Finding #187: SFT > NTP on 4/5 domains)
  2. No BitLinear unpacking — LoRA runs as side computation
  3. Uses pierre.v3 inject_lora() instead of premerge()

Kill criteria:
  K718: Behavioral score < 0.4 mean (must beat NTP 0.332)
  K719: Routing accuracy < 80%
  K720: Inference speed < 80% of native BitLinear

Platform: Apple M5 Pro 48GB, MLX
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

# Memory safety
device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from pierre.v3 import (
    calibrate_router, route, extract_hidden,
    inject_lora, strip_lora, load_adapter, load_skeleton,
)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source data
NTP_SOURCE = EXPERIMENT_DIR.parent / "real_data_domain_experts"
SFT_SOURCE = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3" / "sft_adapters"
SKELETON_PATH = NTP_SOURCE / "adapters" / "grassmannian_skeleton.npz"
DATA_DIR = NTP_SOURCE / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_SCALE = 20.0
MAX_SEQ = 256
SEED = 42
DOMAINS = ["medical", "code", "math", "legal", "finance"]

N_CAL = 50
N_TEST = 50
N_GEN = 5
MAX_NEW_TOKENS = 128

from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler


# ── Utilities ────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def log(msg): print(msg, flush=True)

def log_memory(label=""):
    a = mx.get_active_memory()/1e9
    log(f"[MEM {label}] active={a:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")

def cleanup(*objects):
    for o in objects: del o
    gc.collect(); mx.clear_cache(); mx.reset_peak_memory()

def load_data(domain, split="valid", n=None):
    samples = []
    with open(DATA_DIR / domain / f"{split}.jsonl") as f:
        for line in f:
            samples.append(json.loads(line)["text"])
            if n and len(samples) >= n: break
    return samples


# ── Behavioral eval ──────────────────────────────────────────────────────

STOP_WORDS = {'the','a','an','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might','can',
    'to','of','in','for','on','with','at','by','from','as','and','but','or','not',
    'so','yet','both','either','each','every','all','any','few','more','most','other',
    'some','such','no','only','own','same','than','too','very','just','because','if',
    'when','where','how','what','which','who','this','that','these','those','it','its',
    'i','me','my','we','our','you','your','he','him','his','she','her','they','them','their'}

def factual_recall(gen, ref):
    def toks(t):
        return set(w for w in re.findall(r'\b[a-z]+\b', t.lower())
                   if w not in STOP_WORDS and len(w) > 2)
    g, r = toks(gen), toks(ref)
    return len(g & r) / len(r) if r else 0.0

def eval_response(gen, ref, domain):
    if domain == "code":
        blocks = re.findall(r'```(?:python)?\s*\n(.*?)\n```', gen, re.DOTALL)
        code = '\n'.join(blocks) if blocks else '\n'.join(
            l for l in gen.split('\n') if l.strip() and not l.startswith('#'))
        try: ast.parse(code); ok = True
        except SyntaxError: ok = False
        return 0.7 * float(ok) + 0.3 * factual_recall(gen, ref)
    return factual_recall(gen, ref)


# ── Phase 1: Calibrate router (on base model, no adapters) ──────────────

def phase_calibrate():
    log("\n=== Phase 1: Calibrate router ===")
    model, tok = load(MODEL_ID)
    log_memory("model loaded (native BitLinear)")

    cal = {d: load_data(d, "train", N_CAL) for d in DOMAINS}
    W = calibrate_router(model, tok, cal, lam=1.0, max_seq=MAX_SEQ)
    log(f"  Router: H={W.shape[0]}, D={W.shape[1]}")

    correct, total = 0, 0
    per_domain = {}
    for di, d in enumerate(DOMAINS):
        dc = sum(1 for t in load_data(d, "valid", N_TEST)
                 if route(model, tok, t, W, MAX_SEQ) == di)
        correct += dc; total += N_TEST
        per_domain[d] = round(dc / N_TEST, 3)
        log(f"    {d}: {per_domain[d]:.1%}")

    acc = correct / total
    log(f"  Overall: {acc:.1%}")
    np.save(str(EXPERIMENT_DIR / "router_W.npy"), np.array(W))
    cleanup(model, tok)
    return {"accuracy": round(acc, 4), "per_domain": per_domain}


# ── Phase 2: PPL — base vs NTP-single vs SFT-single vs SFT-Pierre ──────

def phase_ppl():
    log("\n=== Phase 2: PPL comparison ===")
    skeleton = load_skeleton(str(SKELETON_PATH))
    val = {d: load_data(d, "valid", N_TEST) for d in DOMAINS}
    W = mx.array(np.load(str(EXPERIMENT_DIR / "router_W.npy")))

    def ppl(model, tok, texts):
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

    results = {"base": {}, "ntp_single": {}, "sft_single": {}, "sft_pierre": {}}

    # Base PPL
    model, tok = load(MODEL_ID)
    for d in DOMAINS:
        results["base"][d] = round(ppl(model, tok, val[d]), 3)
        log(f"  base/{d}: {results['base'][d]}")
    cleanup(model, tok)

    # SFT single-adapter PPL (gold standard)
    for di, d in enumerate(DOMAINS):
        model, tok = load(MODEL_ID)
        adapter = load_adapter(str(SFT_SOURCE / d / "adapter.npz"))
        n = inject_lora(model, skeleton, adapter, di, LORA_SCALE)
        results["sft_single"][d] = round(ppl(model, tok, val[d]), 3)
        log(f"  sft_single/{d}: {results['sft_single'][d]} ({n} wrapped)")
        cleanup(model, tok, adapter)

    # SFT Pierre: route → inject
    for di, d in enumerate(DOMAINS):
        model, tok = load(MODEL_ID)
        ri = route(model, tok, val[d][0], W, MAX_SEQ)
        rd = DOMAINS[ri]
        adapter = load_adapter(str(SFT_SOURCE / rd / "adapter.npz"))
        inject_lora(model, skeleton, adapter, ri, LORA_SCALE)
        results["sft_pierre"][d] = round(ppl(model, tok, val[d]), 3)
        deg = (results["sft_pierre"][d] - results["sft_single"][d]) / results["sft_single"][d] * 100
        log(f"  sft_pierre/{d}: {results['sft_pierre'][d]} (→{rd}, {deg:+.1f}%)")
        cleanup(model, tok, adapter)

    results["degradation"] = {}
    for d in DOMAINS:
        results["degradation"][d] = round(
            (results["sft_pierre"][d] - results["sft_single"][d]) / results["sft_single"][d] * 100, 2)

    return results


# ── Phase 3: Behavioral eval with SFT adapters ──────────────────────────

def phase_behavioral():
    log("\n=== Phase 3: Behavioral eval (SFT) ===")
    skeleton = load_skeleton(str(SKELETON_PATH))
    W = mx.array(np.load(str(EXPERIMENT_DIR / "router_W.npy")))
    results = {"per_domain": {}}

    for di, d in enumerate(DOMAINS):
        model, tok = load(MODEL_ID)
        test = load_data(d, "valid", N_GEN)

        ri = route(model, tok, test[0], W, MAX_SEQ)
        rd = DOMAINS[ri]
        adapter = load_adapter(str(SFT_SOURCE / rd / "adapter.npz"))
        inject_lora(model, skeleton, adapter, ri, LORA_SCALE)

        scores = []
        sampler = make_sampler(temp=0.0)
        for text in test:
            if "### Response:" in text:
                prompt = text.split("### Response:")[0].strip() + "\n### Response:\n"
                ref = text.split("### Response:")[-1].strip()
            else:
                prompt, ref = text[:200], text
            try:
                gen = mlx_generate(model, tok, prompt=prompt,
                                   max_tokens=MAX_NEW_TOKENS, sampler=sampler, verbose=False)
            except Exception:
                gen = ""
            scores.append(eval_response(gen, ref, d))

        mean = float(np.mean(scores)) if scores else 0.0
        results["per_domain"][d] = {"score": round(mean, 3), "routed_to": rd}
        log(f"  {d} → {rd}: {mean:.3f}")
        cleanup(model, tok, adapter)

    results["overall"] = round(float(np.mean([v["score"] for v in results["per_domain"].values()])), 3)
    log(f"  Overall: {results['overall']}")
    return results


# ── Phase 4: Latency — native BitLinear vs BitLinear+LoRA side-path ─────

def phase_latency():
    log("\n=== Phase 4: Latency (native BitLinear) ===")
    skeleton = load_skeleton(str(SKELETON_PATH))
    prompt = "### Instruction:\nExplain photosynthesis.\n\n### Response:\n"
    sampler = make_sampler(temp=0.0)

    def bench(model, tok, label):
        for _ in range(2):
            mlx_generate(model, tok, prompt=prompt, max_tokens=32, sampler=sampler, verbose=False)
        times = []
        for _ in range(5):
            t0 = time.time()
            out = mlx_generate(model, tok, prompt=prompt, max_tokens=MAX_NEW_TOKENS,
                               sampler=sampler, verbose=False)
            dt = time.time() - t0
            n = len(tok.encode(out)) - len(tok.encode(prompt))
            times.append({"s": dt, "toks": n})
        tps = sum(t["toks"] for t in times) / sum(t["s"] for t in times)
        log(f"  {label}: {tps:.1f} tok/s")
        return round(tps, 1)

    # Native BitLinear (no unpacking!)
    model, tok = load(MODEL_ID)
    base_tps = bench(model, tok, "native BitLinear")
    cleanup(model, tok)

    # BitLinear + LoRA side-path
    model, tok = load(MODEL_ID)
    adapter = load_adapter(str(SFT_SOURCE / "medical" / "adapter.npz"))
    inject_lora(model, skeleton, adapter, 0, LORA_SCALE)
    pierre_tps = bench(model, tok, "BitLinear + LoRA side-path")
    overhead = round((base_tps - pierre_tps) / base_tps * 100, 2)
    log(f"  Overhead: {overhead}%")
    cleanup(model, tok, adapter)

    return {"base_tps": base_tps, "pierre_tps": pierre_tps, "overhead_pct": overhead}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("Pierre v3 — SFT adapters + BitLinear side-path")
    log("=" * 60)
    mx.random.seed(SEED)

    r1 = phase_calibrate()
    r2 = phase_ppl()
    r3 = phase_behavioral()
    r4 = phase_latency()

    k1 = r3["overall"] >= 0.40
    k2 = r1["accuracy"] >= 0.80
    k3 = r4["overhead_pct"] <= 20.0

    results = {
        "experiment": "pierre_v3_sft_n5",
        "total_time_s": round(time.time() - t0, 1),
        "routing": r1, "ppl": r2, "behavioral": r3, "latency": r4,
        "kill_criteria": {
            "K718": {"pass": k1, "value": r3["overall"], "threshold": 0.40,
                     "detail": f"Behavioral {r3['overall']} (target >= 0.4, NTP was 0.332)"},
            "K719": {"pass": k2, "value": r1["accuracy"], "threshold": 0.80},
            "K720": {"pass": k3, "value": r4["overhead_pct"], "threshold": 20.0,
                     "detail": f"Side-path overhead {r4['overhead_pct']}% (limit 20%)"},
        },
        "all_pass": k1 and k2 and k3,
    }

    log("\n" + "=" * 60)
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} — {v.get('detail', v['value'])}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'} in {results['total_time_s']}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
