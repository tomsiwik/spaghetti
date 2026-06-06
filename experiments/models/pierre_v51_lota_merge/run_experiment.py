#!/usr/bin/env python3
"""Pierre v5.1: LoTA-QAF lossless ternary merge.

Adapts LoTA-QAF (arXiv:2505.18724) for BitNet:
  1. Take existing ternary A (Grassmannian) and FP32 B (SFT adapters)
  2. PTQ B to ternary: B_T = clip(round(B / mean(|B|)), -1, 1)
  3. Compute integer delta: ΔW = A_T @ B_T ∈ Z^{out×in}, values in [-r, r]
  4. Threshold to ternary: Ŵ = sign(ΔW) · I(|ΔW| > ω)
  5. Merge into BitLinear: W_new = clip(W_base + Ŵ, -1, 1), repack

After merge: native BitLinear, zero inference overhead, ~140 tok/s target.

Key difference from v4 (KILLED): v4 added CONTINUOUS delta then re-quantized.
v5.1 keeps delta as INTEGER from the start, thresholds to ternary. The adapter
speaks the same discrete language as the base weights.

Kill criteria:
  K730: PPL > 2x base
  K731: Speed < 120 tok/s
  K732: Behavioral < 0.10
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

from pierre.v5.pierre import (
    pack_ternary, unpack_ternary, extract_hidden,
    calibrate_router, route, load_adapter, load_skeleton,
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


# ── LoTA-QAF Lossless Ternary Merge ─────────────────────────────────────

def lota_merge(model, skeleton: dict, adapter_b: dict[str, mx.array],
               domain_idx: int, omega: int = 1) -> dict:
    """LoTA-QAF-style lossless ternary merge into BitLinear.

    1. Quantize B to ternary: B_T = clip(round(B / mean(|B|)), -1, 1)
    2. Quantize A to ternary (from skeleton, already near-ternary from Grassmannian)
    3. Compute integer product: ΔW = A_T^T @ B_T^T (integer matrix, range [-r, r])
    4. Threshold: Ŵ = sign(ΔW) · I(|ΔW| > ω) → ternary {-1, 0, +1}
    5. Merge: W_new = clip(W_base_ternary + Ŵ, -1, 1)
    6. Repack with ORIGINAL scale (base scale unchanged)

    Returns merge statistics.
    """
    count = 0
    n_layers = len(model.model.layers)
    total_flips = 0
    total_weights = 0
    boundary_clips = 0

    for li in range(n_layers):
        for key in TARGET_MODULES:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_b or ak not in skeleton:
                continue

            m = model.model.layers[li]
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None: break
            if not isinstance(m, BitLinear):
                continue

            # Step 1-2: Get ternary A and quantize B to ternary
            A_fp = mx.array(skeleton[ak]).astype(mx.float32)  # (in, rank)
            B_fp = adapter_b[bk].astype(mx.float32)           # (rank, out)

            # PTQ to ternary
            A_alpha = mx.mean(mx.abs(A_fp)) + 1e-7
            A_T = mx.clip(mx.round(A_fp / A_alpha), -1, 1).astype(mx.int32)

            B_alpha = mx.mean(mx.abs(B_fp)) + 1e-7
            B_T = mx.clip(mx.round(B_fp / B_alpha), -1, 1).astype(mx.int32)

            # Step 3: Integer product ΔW = A_T^T @ B_T^T → (out, in)
            # A_T is (in, rank), B_T is (rank, out)
            # We need delta for weight matrix which is (out, in):
            # delta = B_T^T @ A_T^T = (out, rank) @ (rank, in) = (out, in)
            # MLX requires float for matmul — compute in f32, result is integer-valued
            delta_int = (B_T.astype(mx.float32).T @ A_T.astype(mx.float32).T).astype(mx.int32)
            mx.eval(delta_int)

            # Step 4: Threshold to ternary
            abs_delta = mx.abs(delta_int)
            W_hat = mx.where(abs_delta > omega,
                             mx.sign(delta_int),
                             mx.zeros_like(delta_int)).astype(mx.int8)

            # Step 5: Unpack base, add, clip
            W_base = unpack_ternary(m.weight, m.out_features)  # int8 {-1,0,+1}
            W_merged = W_base.astype(mx.int32) + W_hat.astype(mx.int32)

            # Count boundary clips before clipping
            over = mx.sum(mx.abs(W_merged) > 1).item()
            boundary_clips += over

            W_clipped = mx.clip(W_merged, -1, 1).astype(mx.int8)

            # Count actual flips
            flips = mx.sum(W_hat != 0).item()
            total_flips += flips
            total_weights += W_base.size

            # Step 6: Repack with ORIGINAL scale (lossless — scale unchanged)
            packed = pack_ternary(W_clipped, m.out_features)
            m.weight = packed
            # Scale stays the same — this is the lossless part
            count += 1

            del A_fp, B_fp, A_T, B_T, delta_int, W_hat, W_base, W_merged, W_clipped

    mx.eval(model.parameters())
    return {
        "modules_merged": count,
        "total_flips": total_flips,
        "total_weights": total_weights,
        "flip_rate": round(total_flips / max(total_weights, 1) * 100, 4),
        "boundary_clips": boundary_clips,
        "omega": omega,
    }


# ── Phase 1: Router ─────────────────────────────────────────────────────

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


# ── Phase 2: PPL with omega sweep ───────────────────────────────────────

def phase_ppl():
    log("\n=== Phase 2: PPL (omega sweep) ===")
    skel = load_skeleton(str(SKELETON_PATH))
    val = {d: load_data(d,"valid",N_TEST) for d in DOMAINS}
    W = mx.array(np.load(str(EXPERIMENT_DIR/"router_W.npy")))
    results = {"base": {}, "by_omega": {}}

    # Base PPL
    model, tok = load(MODEL_ID)
    for d in DOMAINS:
        results["base"][d] = round(compute_ppl(model,tok,val[d]), 3)
        log(f"  base/{d}: {results['base'][d]}")
    cleanup(model, tok)

    # Sweep omega values
    for omega in [0, 1, 2, 4]:
        log(f"\n  --- omega={omega} ---")
        results["by_omega"][omega] = {"ppl": {}, "merge_stats": {}}

        for di, d in enumerate(DOMAINS):
            model, tok = load(MODEL_ID)
            adapter = load_adapter(str(SFT_SOURCE/d/"adapter.npz"))
            stats = lota_merge(model, skel, adapter, di, omega=omega)
            ppl = compute_ppl(model, tok, val[d])
            results["by_omega"][omega]["ppl"][d] = round(ppl, 3)
            results["by_omega"][omega]["merge_stats"][d] = stats
            deg = (ppl - results["base"][d]) / results["base"][d] * 100
            log(f"    {d}: PPL={ppl:.3f} ({deg:+.1f}% vs base), flips={stats['flip_rate']:.2f}%, clips={stats['boundary_clips']}")
            cleanup(model, tok, adapter)

    return results


# ── Phase 3: Best omega → behavioral + routed ───────────────────────────

def phase_behavioral(best_omega):
    log(f"\n=== Phase 3: Behavioral (omega={best_omega}) ===")
    skel = load_skeleton(str(SKELETON_PATH))
    W = mx.array(np.load(str(EXPERIMENT_DIR/"router_W.npy")))
    results = {"per_domain": {}, "omega": best_omega}

    for di, d in enumerate(DOMAINS):
        model, tok = load(MODEL_ID)
        test = load_data(d,"valid",N_GEN)
        ri = route(model,tok,test[0],W,MAX_SEQ)
        rd = DOMAINS[ri]
        adapter = load_adapter(str(SFT_SOURCE/rd/"adapter.npz"))
        lota_merge(model, skel, adapter, ri, omega=best_omega)

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


# ── Phase 4: Latency ────────────────────────────────────────────────────

def phase_latency(best_omega):
    log(f"\n=== Phase 4: Latency (omega={best_omega}) ===")
    skel = load_skeleton(str(SKELETON_PATH))
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
    adapter = load_adapter(str(SFT_SOURCE/"medical"/"adapter.npz"))
    stats = lota_merge(model, skel, adapter, 0, omega=best_omega)
    is_bl = isinstance(model.model.layers[0].self_attn.q_proj, BitLinear)
    merged_tps = bench(model,tok,f"LoTA merged (BitLinear={is_bl})")
    overhead = round((base_tps-merged_tps)/base_tps*100, 2)
    log(f"  Overhead: {overhead}%")
    cleanup(model, tok, adapter)

    return {"base_tps": base_tps, "merged_tps": merged_tps, "overhead_pct": overhead,
            "still_bitlinear": is_bl, "merge_stats": stats}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("Pierre v5.1 — LoTA-QAF Lossless Ternary Merge")
    log("=" * 60)
    mx.random.seed(SEED)

    r1 = phase_calibrate()
    r2 = phase_ppl()

    # Pick best omega: lowest mean PPL across domains
    best_omega = min(r2["by_omega"].keys(),
                     key=lambda o: np.mean(list(r2["by_omega"][o]["ppl"].values())))
    log(f"\n  Best omega: {best_omega}")

    r3 = phase_behavioral(best_omega)
    r4 = phase_latency(best_omega)

    # v5 baseline PPL for comparison
    v5_ppl = {"medical": 5.512, "code": 4.029, "math": 3.412, "legal": 20.498, "finance": 18.647}
    ppl_vs_v5 = {}
    best_ppls = r2["by_omega"][best_omega]["ppl"]
    for d in DOMAINS:
        ppl_vs_v5[d] = round((best_ppls[d] - v5_ppl[d]) / v5_ppl[d] * 100, 2)

    k1 = max(best_ppls[d] / r2["base"][d] for d in DOMAINS) < 2.0
    k2 = r4["merged_tps"] >= 120.0
    k3 = r3["overall"] >= 0.10

    results = {
        "experiment": "pierre_v51_lota_merge",
        "total_time_s": round(time.time()-t0, 1),
        "best_omega": best_omega,
        "routing": r1, "ppl_sweep": r2, "behavioral": r3, "latency": r4,
        "ppl_vs_v5": ppl_vs_v5,
        "kill_criteria": {
            "K730": {"pass": k1, "detail": f"Worst PPL/base ratio: {max(best_ppls[d]/r2['base'][d] for d in DOMAINS):.2f}x"},
            "K731": {"pass": k2, "detail": f"Speed: {r4['merged_tps']} tok/s"},
            "K732": {"pass": k3, "detail": f"Behavioral: {r3['overall']}"},
        },
        "all_pass": k1 and k2 and k3,
        "comparison": {
            "v4_ternary_premerge": "141 tok/s, 0% overhead, KILLED (PPL +475-1348%)",
            "v5_ternary_sidepath": "77 tok/s, 45% overhead, behavioral 0.317",
            "v51_lota_merge": f"{r4['merged_tps']} tok/s, {r4['overhead_pct']}% overhead, behavioral {r3['overall']}",
        },
    }

    log("\n" + "=" * 60)
    log(f"Best omega: {best_omega}")
    log("PPL vs v5 ternary side-path:")
    for d, v in ppl_vs_v5.items(): log(f"  {d}: {v:+.1f}%")
    log("\nKill criteria:")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} — {v['detail']}")
    log(f"\n{'ALL PASS' if results['all_pass'] else 'KILLED'} in {results['total_time_s']}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
