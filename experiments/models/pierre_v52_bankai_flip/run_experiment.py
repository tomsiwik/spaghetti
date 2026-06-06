#!/usr/bin/env python3
"""Pierre v5.2: Bankai-inspired ternary row flips.

Adapts Bankai (github.com/nikshepsvn/bankai) for ternary BitLinear:
  - Bankai: XOR flips entire rows in 1-bit models (840 bytes per patch)
  - Ternary: can't XOR, but CAN increment/decrement rows by 1, clip to {-1,0,+1}

Approach:
  1. For each target domain, greedily search for beneficial row flips
  2. A "flip" = increment or decrement all entries in a row by 1, clip to {-1,0,+1}
  3. Measure domain PPL improvement after each flip
  4. Store adapter as list of (layer, module, row, direction) — sub-1KB
  5. After apply: model is native BitLinear, zero inference overhead

Key Bankai insight: scale-guided targeting — high-scale rows have 3.88x more impact.
We adapt this: sort rows by their weight_scale contribution, try high-impact rows first.

Kill criteria:
  K733: Zero domain signal
  K734: Search > 30 min per domain
  K735: Speed < 120 tok/s
"""

import gc, json, math, os, re, time, ast
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from pierre.v5.pierre import (
    pack_ternary, unpack_ternary, extract_hidden,
    calibrate_router, route, load_skeleton,
    TARGET_MODULES,
)
from mlx_lm import load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.bitlinear_layers import BitLinear

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

NTP_SOURCE = EXPERIMENT_DIR.parent / "real_data_domain_experts"
SFT_SOURCE = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3" / "sft_adapters"
SKELETON_PATH = NTP_SOURCE / "adapters" / "grassmannian_skeleton.npz"
DATA_DIR = NTP_SOURCE / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
MAX_SEQ = 256
SEED = 42
DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_CAL, N_TEST, N_GEN, MAX_TOK = 50, 50, 5, 128

# Bankai-style search parameters
MAX_FLIPS = 100        # max flips per domain patch
MAX_SEARCH_ITERS = 200 # greedy search iterations
N_CAL_PPL = 10         # samples for fast PPL during search


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

def fast_ppl(model, tok, texts):
    """Fast PPL on small sample for search."""
    loss, n = 0.0, 0
    for text in texts[:N_CAL_PPL]:
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


# ── Ternary Row Flip Operations ──────────────────────────────────────────

def get_row_candidates(model):
    """Build list of (layer_idx, module_key, row_idx, weight_scale) for all rows.

    Sorted by weight_scale (descending) — Bankai insight: high-scale = high impact.
    """
    candidates = []
    for li, layer in enumerate(model.model.layers):
        for key in TARGET_MODULES:
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None: break
            if not isinstance(m, BitLinear):
                continue
            scale = m.weight_scale.item() if m.weight_scale.size == 1 else mx.mean(m.weight_scale).item()
            for row in range(m.out_features):
                candidates.append((li, key, row, scale))

    # Sort by scale descending (high impact first)
    candidates.sort(key=lambda x: -x[3])
    return candidates


def apply_row_flip(model, layer_idx, module_key, row_idx, direction):
    """Flip a single row in a BitLinear module.

    direction=+1: increment each entry by 1, clip to [-1,1]
    direction=-1: decrement each entry by 1, clip to [-1,1]
    """
    m = model.model.layers[layer_idx]
    for part in module_key.split("."):
        m = getattr(m, part, None)
    if not isinstance(m, BitLinear):
        return

    W = unpack_ternary(m.weight, m.out_features)  # (out, in) int8
    row = W[row_idx].astype(mx.int32) + direction
    row = mx.clip(row, -1, 1).astype(mx.int8)

    # Write back the modified row
    W_modified = W.at[row_idx].add(mx.zeros_like(W[row_idx]))  # copy
    # MLX doesn't support row assignment easily, so rebuild the full tensor
    rows = []
    for i in range(m.out_features):
        if i == row_idx:
            rows.append(row[None, :])
        else:
            rows.append(W[i:i+1])
    W_new = mx.concatenate(rows, axis=0)

    m.weight = pack_ternary(W_new, m.out_features)
    mx.eval(m.weight)


def revert_row_flip(model, layer_idx, module_key, row_idx, direction):
    """Undo a flip by applying the opposite direction."""
    apply_row_flip(model, layer_idx, module_key, row_idx, -direction)


# ── Greedy Patch Search ──────────────────────────────────────────────────

def search_patch(model, tok, domain_cal, domain_name):
    """Greedy hill-climbing search for beneficial row flips.

    Returns list of (layer, module, row, direction) flips and search stats.
    """
    t0 = time.time()
    candidates = get_row_candidates(model)
    log(f"    {len(candidates)} candidate rows (sorted by scale)")

    baseline_ppl = fast_ppl(model, tok, domain_cal)
    log(f"    Baseline PPL: {baseline_ppl:.3f}")

    flips = []
    current_ppl = baseline_ppl
    n_tried = 0
    n_improved = 0

    # Sample from top candidates (scale-guided)
    mx.random.seed(SEED)
    for iteration in range(min(MAX_SEARCH_ITERS, len(candidates))):
        if len(flips) >= MAX_FLIPS:
            break
        if time.time() - t0 > 1800:  # 30 min timeout
            log(f"    Timeout at {len(flips)} flips")
            break

        li, key, row, scale = candidates[iteration]

        # Try both directions
        best_dir = None
        best_ppl = current_ppl

        for direction in [+1, -1]:
            apply_row_flip(model, li, key, row, direction)
            ppl = fast_ppl(model, tok, domain_cal)
            n_tried += 1

            if ppl < best_ppl - 0.001:  # meaningful improvement
                best_ppl = ppl
                best_dir = direction

            revert_row_flip(model, li, key, row, direction)

        if best_dir is not None:
            apply_row_flip(model, li, key, row, best_dir)
            current_ppl = best_ppl
            flips.append({"layer": li, "module": key, "row": row, "direction": best_dir})
            n_improved += 1
            if n_improved % 10 == 0:
                log(f"    Flip {n_improved}: PPL {current_ppl:.3f} ({(current_ppl/baseline_ppl-1)*100:+.1f}%)")

    elapsed = time.time() - t0
    log(f"    Search done: {len(flips)} flips, {n_tried} tried, {elapsed:.1f}s")
    log(f"    Final PPL: {current_ppl:.3f} ({(current_ppl/baseline_ppl-1)*100:+.1f}% vs base)")

    return {
        "flips": flips,
        "n_flips": len(flips),
        "n_tried": n_tried,
        "search_time_s": round(elapsed, 1),
        "baseline_ppl": round(baseline_ppl, 3),
        "final_ppl": round(current_ppl, 3),
        "ppl_change_pct": round((current_ppl/baseline_ppl - 1) * 100, 2),
        "patch_size_bytes": len(flips) * 16,  # ~16 bytes per flip (4 fields)
    }


def apply_patch(model, patch_flips):
    """Apply a saved patch to a model."""
    for flip in patch_flips:
        apply_row_flip(model, flip["layer"], flip["module"], flip["row"], flip["direction"])


# ── Phases ───────────────────────────────────────────────────────────────

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
    np.save(str(EXPERIMENT_DIR/"router_W.npy"), np.array(W))
    cleanup(model, tok)
    return {"accuracy": round(acc,4)}


def phase_search_patches():
    log("\n=== Phase 2: Greedy patch search ===")
    results = {}

    for di, d in enumerate(DOMAINS):
        log(f"\n  Domain: {d}")
        model, tok = load(MODEL_ID)
        cal = load_data(d, "train", N_CAL)

        patch = search_patch(model, tok, cal, d)
        results[d] = patch

        # Save patch
        patch_file = EXPERIMENT_DIR / f"patch_{d}.json"
        patch_file.write_text(json.dumps(patch, indent=2, cls=NumpyEncoder))
        log(f"    Saved: {patch_file.name} ({patch['patch_size_bytes']} bytes)")

        cleanup(model, tok)

    return results


def phase_eval_patches(patches):
    log("\n=== Phase 3: Evaluate patches ===")
    val = {d: load_data(d,"valid",N_TEST) for d in DOMAINS}
    results = {"base_ppl": {}, "patched_ppl": {}, "behavioral": {"per_domain": {}}}

    # Base PPL
    model, tok = load(MODEL_ID)
    for d in DOMAINS:
        results["base_ppl"][d] = round(compute_ppl(model,tok,val[d]), 3)
    cleanup(model, tok)

    # Patched PPL + behavioral
    for di, d in enumerate(DOMAINS):
        model, tok = load(MODEL_ID)
        apply_patch(model, patches[d]["flips"])

        results["patched_ppl"][d] = round(compute_ppl(model,tok,val[d]), 3)
        deg = (results["patched_ppl"][d] - results["base_ppl"][d]) / results["base_ppl"][d] * 100
        log(f"  {d}: PPL {results['patched_ppl'][d]} ({deg:+.1f}% vs base)")

        # Behavioral
        test = load_data(d,"valid",N_GEN)
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
        results["behavioral"]["per_domain"][d] = round(mean, 3)
        log(f"    Behavioral: {mean:.3f}")
        cleanup(model, tok)

    results["behavioral"]["overall"] = round(float(np.mean(
        list(results["behavioral"]["per_domain"].values()))), 3)
    return results


def phase_latency(patches):
    log("\n=== Phase 4: Latency ===")
    prompt = "### Instruction:\nExplain photosynthesis.\n\n### Response:\n"
    sampler = make_sampler(temp=0.0)

    def bench(model, tok, label):
        for _ in range(2):
            mlx_generate(model,tok,prompt=prompt,max_tokens=32,sampler=sampler,verbose=False)
        times = []
        for _ in range(5):
            t0 = time.time()
            out = mlx_generate(model,tok,prompt=prompt,max_tokens=MAX_TOK,sampler=sampler,verbose=False)
            dt = time.time()-t0
            n = len(tok.encode(out))-len(tok.encode(prompt))
            times.append({"s":dt,"toks":n})
        tps = sum(t["toks"] for t in times)/sum(t["s"] for t in times)
        log(f"  {label}: {tps:.1f} tok/s")
        return round(tps, 1)

    model, tok = load(MODEL_ID)
    base_tps = bench(model,tok,"native BitLinear")
    cleanup(model, tok)

    model, tok = load(MODEL_ID)
    apply_patch(model, patches["medical"]["flips"])
    is_bl = isinstance(model.model.layers[0].self_attn.q_proj, BitLinear)
    patched_tps = bench(model,tok,f"Bankai patched (BitLinear={is_bl})")
    overhead = round((base_tps-patched_tps)/base_tps*100, 2)
    cleanup(model, tok)

    return {"base_tps": base_tps, "patched_tps": patched_tps, "overhead_pct": overhead}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("Pierre v5.2 — Bankai-Inspired Ternary Row Flips")
    log("=" * 60)
    mx.random.seed(SEED)

    r1 = phase_calibrate()
    patches = phase_search_patches()
    r3 = phase_eval_patches(patches)
    r4 = phase_latency(patches)

    # Kill criteria
    has_signal = any(r3["patched_ppl"][d] < r3["base_ppl"][d] * 0.99 for d in DOMAINS)
    max_search = max(patches[d]["search_time_s"] for d in DOMAINS)
    k1 = has_signal
    k2 = max_search < 1800
    k3 = r4["patched_tps"] >= 120.0

    results = {
        "experiment": "pierre_v52_bankai_flip",
        "total_time_s": round(time.time()-t0, 1),
        "routing": r1, "patches": {d: {k:v for k,v in p.items() if k != "flips"} for d,p in patches.items()},
        "eval": r3, "latency": r4,
        "kill_criteria": {
            "K733": {"pass": k1, "detail": f"Domain signal: {has_signal}"},
            "K734": {"pass": k2, "detail": f"Max search time: {max_search:.0f}s"},
            "K735": {"pass": k3, "detail": f"Speed: {r4['patched_tps']} tok/s"},
        },
        "all_pass": k1 and k2 and k3,
        "comparison": {
            "v5_ternary_sidepath": "77 tok/s, 45% overhead, behavioral 0.317",
            "v52_bankai_flip": f"{r4['patched_tps']} tok/s, {r4['overhead_pct']}% overhead, behavioral {r3['behavioral']['overall']}",
            "patch_sizes": {d: f"{patches[d]['n_flips']} flips, {patches[d]['patch_size_bytes']} bytes" for d in DOMAINS},
        },
    }

    log("\n" + "=" * 60)
    log("Patch summary:")
    for d in DOMAINS:
        p = patches[d]
        log(f"  {d}: {p['n_flips']} flips, {p['ppl_change_pct']:+.1f}% PPL, {p['search_time_s']:.0f}s search, {p['patch_size_bytes']} bytes")
    log(f"\nBehavioral: {r3['behavioral']['overall']}")
    log("\nKill criteria:")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} — {v['detail']}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'} in {results['total_time_s']}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
