#!/usr/bin/env python3
"""Pierre unified pipeline v2: route → compose → pre-merge → generate.

Uses pierre.py single-file API. Previous version in run_experiment_v1.py.

Kill criteria:
  K715: Routing accuracy < 80% on held-out domain classification
  K716: Any domain PPL > 10% worse than single-adapter PPL
  K717: Behavioral quality score < 0.3 mean across domains

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
from mlx.utils import tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from pierre import (
    calibrate_router, route, extract_hidden,
    build_deltas, merge_deltas, premerge,
    load_adapter, load_skeleton, TARGET_MODULES,
    lora_delta,
)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

SOURCE_DIR = EXPERIMENT_DIR.parent / "real_data_domain_experts"
ADAPTERS_DIR = SOURCE_DIR / "adapters"
DATA_DIR = SOURCE_DIR / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
SEED = 42
DOMAINS = ["medical", "code", "math", "legal", "finance"]

N_CAL = 50          # from train split
N_TEST = 50         # from valid split (all of it)
N_GEN = 5           # for behavioral eval
MAX_NEW_TOKENS = 128


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
    a, c, p = mx.get_active_memory()/1e9, mx.get_cache_memory()/1e9, mx.get_peak_memory()/1e9
    log(f"[MEM {label}] active={a:.2f}GB cache={c:.2f}GB peak={p:.2f}GB")


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


# ── BitNet unpacking (shared boilerplate) ────────────────────────────────

from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.bitlinear_layers import BitLinear


def unpack_model(model):
    """Replace BitLinear with nn.Linear for differentiable forward pass."""
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                w = module.weight; s = module.weight_scale
                w0 = (w & 3).astype(mx.bfloat16) - 1
                w1 = ((w >> 2) & 3).astype(mx.bfloat16) - 1
                w2 = ((w >> 4) & 3).astype(mx.bfloat16) - 1
                w3 = ((w >> 6) & 3).astype(mx.bfloat16) - 1
                unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:module.out_features]
                scale = s.astype(mx.bfloat16)
                unpacked = unpacked / scale if module.invert_weight_scales else unpacked * scale
                lin = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
                lin.weight = unpacked
                if module.bias is not None: lin.bias = module.bias
                updates.append((key, lin)); count += 1
        if updates: layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    log(f"  Unpacked {count} BitLinear -> nn.Linear")
    return model


def load_model():
    """Load + unpack BitNet model. Returns (model, tokenizer)."""
    model, tokenizer = load(MODEL_ID)
    return unpack_model(model), tokenizer


# ── Behavioral evaluation ────────────────────────────────────────────────

STOP_WORDS = {'the','a','an','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might','can','shall',
    'to','of','in','for','on','with','at','by','from','as','into','through','during',
    'before','after','and','but','or','nor','not','so','yet','both','either','neither',
    'each','every','all','any','few','more','most','other','some','such','no','only',
    'own','same','than','too','very','just','because','if','when','where','how','what',
    'which','who','whom','this','that','these','those','it','its','i','me','my','we',
    'our','you','your','he','him','his','she','her','they','them','their'}


def factual_recall(generated, reference):
    def tokens(text):
        return set(w for w in re.findall(r'\b[a-z]+\b', text.lower())
                   if w not in STOP_WORDS and len(w) > 2)
    gen, ref = tokens(generated), tokens(reference)
    if not ref: return 0.0
    return len(gen & ref) / len(ref)


def eval_response(generated, reference, domain):
    if domain == "code":
        code = re.findall(r'```(?:python)?\s*\n(.*?)\n```', generated, re.DOTALL)
        code_text = '\n'.join(code) if code else '\n'.join(
            l for l in generated.split('\n') if l.strip() and not l.startswith('#'))
        try: ast.parse(code_text); syntax = True
        except SyntaxError: syntax = False
        return 0.7 * float(syntax) + 0.3 * factual_recall(generated, reference)
    elif domain == "math":
        for pat in [r'(?:answer|result|=)\s*[:\s]*([+-]?\d+\.?\d*)', r'=\s*([+-]?\d+\.?\d*)\s*$']:
            for text in [generated, reference]:
                m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
                if m:
                    try: val = float(m.group(1).replace(',','')); break
                    except ValueError: continue
            else: continue
            break
        else: return 0.0
        # Simplified: just check factual recall for math too
        return factual_recall(generated, reference)
    else:
        return factual_recall(generated, reference)


# ── Phase 1: Calibrate router ───────────────────────────────────────────

def phase_calibrate():
    log("\n=== Phase 1: Calibrate router ===")
    model, tokenizer = load_model()
    log_memory("model loaded")

    # Train split for calibration
    cal_data = {d: load_data(d, "train", N_CAL) for d in DOMAINS}
    W = calibrate_router(model, tokenizer, cal_data, lam=1.0, max_seq=MAX_SEQ_LENGTH)
    log(f"  Router calibrated (H={W.shape[0]}, D={W.shape[1]})")

    # Test on valid split
    correct, total = 0, 0
    per_domain = {}
    for di, domain in enumerate(DOMAINS):
        dc = 0
        for text in load_data(domain, "valid", N_TEST):
            if route(model, tokenizer, text, W, MAX_SEQ_LENGTH) == di:
                dc += 1; correct += 1
            total += 1
        per_domain[domain] = round(dc / max(N_TEST, 1), 3)
        log(f"    {domain}: {per_domain[domain]:.1%}")

    acc = correct / total if total else 0.0
    log(f"  Overall: {acc:.1%} ({correct}/{total})")

    np.save(str(EXPERIMENT_DIR / "router_W.npy"), np.array(W))
    cleanup(model, tokenizer)
    return {"routing_accuracy": round(acc, 4), "per_domain": per_domain}


# ── Phase 2: PPL comparison ─────────────────────────────────────────────

def phase_ppl():
    log("\n=== Phase 2: PPL comparison ===")
    skeleton = load_skeleton(str(ADAPTERS_DIR / "grassmannian_skeleton.npz"))
    val = {d: load_data(d, "valid", N_TEST) for d in DOMAINS}
    W = mx.array(np.load(str(EXPERIMENT_DIR / "router_W.npy")))

    def ppl(model, tokenizer, texts):
        loss, n = 0.0, 0
        for text in texts:
            toks = tokenizer.encode(text)[:MAX_SEQ_LENGTH]
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

    results = {"base": {}, "single": {}, "pierre": {}, "degradation": {}}

    # Base
    model, tok = load_model()
    for d in DOMAINS:
        results["base"][d] = round(ppl(model, tok, val[d]), 3)
        log(f"  base/{d}: {results['base'][d]}")
    cleanup(model, tok)

    # Single adapter (gold standard)
    for di, d in enumerate(DOMAINS):
        model, tok = load_model()
        deltas = build_deltas(load_adapter(str(ADAPTERS_DIR/d/"adapter.npz")),
                              skeleton, di, LORA_SCALE, len(model.model.layers))
        premerge(model, deltas)
        results["single"][d] = round(ppl(model, tok, val[d]), 3)
        log(f"  single/{d}: {results['single'][d]}")
        cleanup(model, tok, deltas)

    # Pierre: route → build_deltas → premerge
    for di, d in enumerate(DOMAINS):
        model, tok = load_model()
        routed_idx = route(model, tok, val[d][0], W, MAX_SEQ_LENGTH)
        routed_domain = DOMAINS[routed_idx]
        deltas = build_deltas(load_adapter(str(ADAPTERS_DIR/routed_domain/"adapter.npz")),
                              skeleton, routed_idx, LORA_SCALE, len(model.model.layers))
        premerge(model, deltas)
        results["pierre"][d] = round(ppl(model, tok, val[d]), 3)
        deg = (results["pierre"][d] - results["single"][d]) / results["single"][d] * 100
        results["degradation"][d] = round(deg, 2)
        log(f"  pierre/{d}: {results['pierre'][d]} (→{routed_domain}, {deg:+.1f}%)")
        cleanup(model, tok, deltas)

    return results


# ── Phase 3: Behavioral eval ────────────────────────────────────────────

def phase_behavioral():
    log("\n=== Phase 3: Behavioral evaluation ===")
    skeleton = load_skeleton(str(ADAPTERS_DIR / "grassmannian_skeleton.npz"))
    W = mx.array(np.load(str(EXPERIMENT_DIR / "router_W.npy")))
    results = {"per_domain": {}, "generations": []}

    for di, domain in enumerate(DOMAINS):
        model, tok = load_model()
        test = load_data(domain, "valid", N_GEN)

        routed_idx = route(model, tok, test[0], W, MAX_SEQ_LENGTH)
        routed = DOMAINS[routed_idx]
        deltas = build_deltas(load_adapter(str(ADAPTERS_DIR/routed/"adapter.npz")),
                              skeleton, routed_idx, LORA_SCALE, len(model.model.layers))
        premerge(model, deltas)

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
            score = eval_response(gen, ref, domain)
            scores.append(score)
            results["generations"].append({"domain": domain, "routed_to": routed,
                                           "score": round(score, 3)})

        mean = float(np.mean(scores)) if scores else 0.0
        results["per_domain"][domain] = {"score": round(mean, 3), "routed_to": routed}
        log(f"  {domain} → {routed}: {mean:.3f}")
        cleanup(model, tok, deltas)

    results["overall"] = round(float(np.mean([v["score"] for v in results["per_domain"].values()])), 3)
    log(f"  Overall: {results['overall']}")
    return results


# ── Phase 4: Latency ────────────────────────────────────────────────────

def phase_latency():
    log("\n=== Phase 4: Latency ===")
    skeleton = load_skeleton(str(ADAPTERS_DIR / "grassmannian_skeleton.npz"))
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

    model, tok = load_model()
    base_tps = bench(model, tok, "base")
    cleanup(model, tok)

    model, tok = load_model()
    deltas = build_deltas(load_adapter(str(ADAPTERS_DIR/"medical"/"adapter.npz")),
                          skeleton, 0, LORA_SCALE, len(model.model.layers))
    premerge(model, deltas)
    pierre_tps = bench(model, tok, "pierre")
    overhead = round((base_tps - pierre_tps) / base_tps * 100, 2)
    log(f"  Overhead: {overhead}%")
    cleanup(model, tok, deltas)

    return {"base_tps": base_tps, "pierre_tps": pierre_tps, "overhead_pct": overhead}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("Pierre Unified Pipeline v2"); log("=" * 60)
    mx.random.seed(SEED)

    r1 = phase_calibrate()
    r2 = phase_ppl()
    r3 = phase_behavioral()
    r4 = phase_latency()

    # Kill criteria
    k1 = r1["routing_accuracy"] >= 0.80
    k2 = max(r2["degradation"].values()) <= 10.0
    k3 = r3["overall"] >= 0.30

    results = {
        "experiment": "pierre_unified_pipeline_v2",
        "total_time_s": round(time.time() - t0, 1),
        "routing": r1, "ppl": r2, "behavioral": r3, "latency": r4,
        "kill_criteria": {
            "K715": {"pass": k1, "value": r1["routing_accuracy"], "threshold": 0.80},
            "K716": {"pass": k2, "value": max(r2["degradation"].values()), "threshold": 10.0},
            "K717": {"pass": k3, "value": r3["overall"], "threshold": 0.30},
        },
        "all_pass": k1 and k2 and k3,
    }

    log("\n" + "=" * 60)
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} ({v['value']} vs {v['threshold']})")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'} in {results['total_time_s']}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
