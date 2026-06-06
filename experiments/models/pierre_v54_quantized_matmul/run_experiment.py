#!/usr/bin/env python3
"""Pierre v5.4: quantized_matmul side-path.

2-bit quantized adapter via mx.quantized_matmul — 2x faster than bf16, lazy eval.
Benchmark: 3.93ms (quantized) vs 7.79ms (bf16) vs 40.6ms (per-module eval).

Kill criteria:
  K739: Speed < 100 tok/s
  K740: Behavioral < 0.30
  K741: Memory > 8GB peak
"""

import ast, gc, json, math, os, re, time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from pierre.v54 import (
    calibrate_router, route, inject_quantized_lora,
    load_adapter, load_skeleton,
)
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


def phase_calibrate():
    log("\n=== Phase 1: Router ===")
    model, tok = load(MODEL_ID)
    W = calibrate_router(model, tok, {d: load_data(d,"train",N_CAL) for d in DOMAINS}, max_seq=MAX_SEQ)
    correct, total, per = 0, 0, {}
    for di, d in enumerate(DOMAINS):
        dc = sum(1 for t in load_data(d,"valid",N_TEST) if route(model,tok,t,W,MAX_SEQ)==di)
        correct += dc; total += N_TEST
        per[d] = round(dc/N_TEST, 3)
        log(f"  {d}: {per[d]:.1%}")
    acc = correct/total
    log(f"  Overall: {acc:.1%}")
    np.save(str(EXPERIMENT_DIR/"router_W.npy"), np.array(W))
    cleanup(model, tok)
    return {"accuracy": round(acc,4), "per_domain": per}


def phase_ppl():
    log("\n=== Phase 2: PPL ===")
    skel = load_skeleton(str(SKELETON_PATH))
    val = {d: load_data(d,"valid",N_TEST) for d in DOMAINS}
    W = mx.array(np.load(str(EXPERIMENT_DIR/"router_W.npy")))
    results = {"base": {}, "lazy_single": {}, "lazy_pierre": {}, "degradation": {}}

    model, tok = load(MODEL_ID)
    for d in DOMAINS:
        results["base"][d] = round(compute_ppl(model,tok,val[d]), 3)
        log(f"  base/{d}: {results['base'][d]}")
    cleanup(model, tok)

    for di, d in enumerate(DOMAINS):
        model, tok = load(MODEL_ID)
        adapter = load_adapter(str(SFT_SOURCE/d/"adapter.npz"))
        n = inject_quantized_lora(model, skel, adapter, di, LORA_SCALE)
        results["lazy_single"][d] = round(compute_ppl(model,tok,val[d]), 3)
        log(f"  lazy_single/{d}: {results['lazy_single'][d]} ({n} wrapped)")
        cleanup(model, tok, adapter)

    for di, d in enumerate(DOMAINS):
        model, tok = load(MODEL_ID)
        ri = route(model,tok,val[d][0],W,MAX_SEQ)
        rd = DOMAINS[ri]
        adapter = load_adapter(str(SFT_SOURCE/rd/"adapter.npz"))
        inject_quantized_lora(model, skel, adapter, ri, LORA_SCALE)
        results["lazy_pierre"][d] = round(compute_ppl(model,tok,val[d]), 3)
        deg = (results["lazy_pierre"][d]-results["lazy_single"][d])/results["lazy_single"][d]*100
        results["degradation"][d] = round(deg, 2)
        log(f"  pierre/{d}: {results['lazy_pierre'][d]} (→{rd}, {deg:+.1f}%)")
        cleanup(model, tok, adapter)

    return results


def phase_behavioral():
    log("\n=== Phase 3: Behavioral ===")
    skel = load_skeleton(str(SKELETON_PATH))
    W = mx.array(np.load(str(EXPERIMENT_DIR/"router_W.npy")))
    results = {"per_domain": {}}

    for di, d in enumerate(DOMAINS):
        model, tok = load(MODEL_ID)
        test = load_data(d,"valid",N_GEN)
        ri = route(model,tok,test[0],W,MAX_SEQ)
        rd = DOMAINS[ri]
        adapter = load_adapter(str(SFT_SOURCE/rd/"adapter.npz"))
        inject_quantized_lora(model, skel, adapter, ri, LORA_SCALE)

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
        results["per_domain"][d] = {"score": round(mean,3), "routed_to": rd}
        log(f"  {d} → {rd}: {mean:.3f}")
        cleanup(model, tok, adapter)

    results["overall"] = round(float(np.mean([v["score"] for v in results["per_domain"].values()])), 3)
    log(f"  Overall: {results['overall']}")
    return results


def phase_latency():
    log("\n=== Phase 4: Latency + Memory ===")
    skel = load_skeleton(str(SKELETON_PATH))
    prompt = "### Instruction:\nExplain photosynthesis.\n\n### Response:\n"
    sampler = make_sampler(temp=0.0)

    def bench(model, tok, label):
        for _ in range(2):
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
    base_tps, base_mem = bench(model,tok,"native BitLinear")
    cleanup(model, tok)

    model, tok = load(MODEL_ID)
    adapter = load_adapter(str(SFT_SOURCE/"medical"/"adapter.npz"))
    inject_quantized_lora(model, skel, adapter, 0, LORA_SCALE)
    lazy_tps, lazy_mem = bench(model,tok,"quantized_matmul side-path")
    overhead = round((base_tps-lazy_tps)/base_tps*100, 2)
    log(f"  Overhead: {overhead}%")
    cleanup(model, tok, adapter)

    return {"base_tps": base_tps, "lazy_tps": lazy_tps, "overhead_pct": overhead,
            "base_peak_gb": base_mem, "lazy_peak_gb": lazy_mem}


def main():
    t0 = time.time()
    log("Pierre v5.3 — Quantized Matmul LoRA Side-Path")
    log("=" * 60)
    mx.random.seed(SEED)

    r1 = phase_calibrate()
    r2 = phase_ppl()
    r3 = phase_behavioral()
    r4 = phase_latency()

    k1 = r4["lazy_tps"] >= 100.0
    k2 = r3["overall"] >= 0.30
    k3 = r4["lazy_peak_gb"] <= 8.0

    results = {
        "experiment": "pierre_v54_quantized_matmul",
        "total_time_s": round(time.time()-t0, 1),
        "routing": r1, "ppl": r2, "behavioral": r3, "latency": r4,
        "kill_criteria": {
            "K739": {"pass": k1, "value": r4["lazy_tps"], "threshold": 100.0},
            "K740": {"pass": k2, "value": r3["overall"], "threshold": 0.30},
            "K741": {"pass": k3, "value": r4["lazy_peak_gb"], "threshold": 8.0},
        },
        "all_pass": k1 and k2 and k3,
        "comparison": {
            "v2_bf16_premerge": "47 tok/s, 0% overhead",
            "v3_bf16_sidepath": "73 tok/s, 48% overhead",
            "v5_ternary_sidepath": "77 tok/s, 45% overhead",
            "v54_quantized_matmul": f"{r4['lazy_tps']} tok/s, {r4['overhead_pct']}% overhead",
            "native_bitlinear": f"{r4['base_tps']} tok/s",
        },
    }

    log("\n" + "=" * 60)
    log("Version comparison:")
    for k, v in results["comparison"].items(): log(f"  {k}: {v}")
    log("\nKill criteria:")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} ({v['value']} vs {v['threshold']})")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'} in {results['total_time_s']}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
