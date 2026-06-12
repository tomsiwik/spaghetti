#!/usr/bin/env python3
"""exp_wildcat_static_velocity_surrogate — recover the F#862 velocity core from FINAL weights only.

F#862: trajectory velocity mask (step 200 vs 1000, thresh 0.80, sign match) on the thinking adapter's
v/o_proj recovers GSM8K 0.44 -> 0.74 when composed with the math q_proj adapter. Undeployable: needs
checkpoints. Here we build 3 STATIC surrogate masks from the final ckpt 0001000 alone, at matched global
sparsity f* (the measured ground-truth core fraction), and test:
  (a) Jaccard vs the ground-truth trajectory mask,
  (b) behavioral recovery in the exact F#862 composed setting (GSM8K n=50, greedy).

Surrogates (per (layer,proj), keep top-f* entries by score):
  S1 magnitude      : |dW|
  S2 top-SVD agree  : (dW_top4 * dW) / (dW^2 + eps), dW_top4 = exact rank-4 SVD truncation (QR-of-factors)
  S3 factor energy  : ||A_row_i|| * ||B_col_j||
Null: random mask at same per-(layer,proj) fraction. Anchor: ground-truth core re-run in this harness.

Conditions (each: fresh model, math LoRA on q_proj scale 6.0 + dense masked thinking delta scale 1.0):
  B  math + full-thinking (low-rank, interference baseline; F#862: 0.44)
  R  math + random-mask thinking (null)
  S1/S2/S3 math + surrogate-masked thinking
  G  math + ground-truth velocity core (anchor; F#862: 0.74)

KILL 2334 (pre-registered): KILL if best surrogate EM <= EM(R) + 2pp OR best surrogate EM < 0.59.
GATE (success): best surrogate EM >= 0.68 AND >= EM(R) + 6pp.

Composition is Sum_i (B_i A_i) on disjoint projections, never (SumB)(SumA). Scales <= 8.
Wrappers via subclass nn.Module + setattr (F#831). NO MOCKS. is_smoke=False. mlx-lm == 0.31.2.
"""

import gc
import json
import os
import re
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 6 * 1024**3)

EXP_DIR = Path(__file__).resolve().parent
RESULTS_FILE = EXP_DIR / "results.json"
REPO = EXP_DIR.parent.parent.parent

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
THINK_DIR = REPO / "data" / "adapters" / "thinking-openthoughts-universal-v0"
THINK_EARLY = THINK_DIR / "0000200_adapters.safetensors"
THINK_FINAL = THINK_DIR / "0001000_adapters.safetensors"
MATH_AD = REPO / "data" / "adapters" / "math" / "adapters.safetensors"

THINK_SCALE = 1.0
MATH_SCALE = 6.0
THINK_PROJS = ("v_proj", "o_proj")
MATH_PROJ = "q_proj"
VEL_THRESH = 0.80
N_GSM8K = 50
MAX_NEW_TOKENS = 1024
SEED = 42
K_SVD = 4                 # rank-4 truncation (half of LoRA rank 8)
EPS = 1e-12

KILL_NULL_MARGIN = 0.02   # best surrogate must beat random null by > 2pp
KILL_ABS_EM = 0.59        # best surrogate must reach >= 0.59 (50% of 0.44->0.74)
GATE_ABS_EM = 0.68        # success gate: >= 80% of the gap
GATE_NULL_MARGIN = 0.06


def log(msg):
    print(msg, flush=True)


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def layer_keys(weights, li, proj):
    ak = f"language_model.model.layers.{li}.self_attn.{proj}.lora_a"
    bk = f"language_model.model.layers.{li}.self_attn.{proj}.lora_b"
    return (ak, bk) if ak in weights and bk in weights else (None, None)


# ----------------------------------------------------------------------------
# Masks: ground-truth (trajectory) + static surrogates (final ckpt only)
# ----------------------------------------------------------------------------

def topk_mask(score, k):
    """Boolean mask keeping the k largest entries of `score` (any shape)."""
    flat = score.reshape(-1)
    if k <= 0:
        return mx.zeros(score.shape, dtype=mx.bool_)
    if k >= flat.size:
        return mx.ones(score.shape, dtype=mx.bool_)
    part = mx.partition(flat, flat.size - k)        # ascending; kth from top at [size-k]
    thresh = part[flat.size - k]
    return score >= thresh


def svd_top_truncation(A, B, k):
    """Exact rank-k truncation of dW = A@B via QR-of-factors. A:(in,r) B:(r,out)."""
    Qa, Ra = mx.linalg.qr(A, stream=mx.cpu)                 # (in,r),(r,r)
    Qb, Rb = mx.linalg.qr(B.T, stream=mx.cpu)               # (out,r),(r,r)
    M = Ra @ Rb.T                                           # (r,r)
    U, S, Vt = mx.linalg.svd(M, stream=mx.cpu)
    left = Qa @ (U[:, :k] * S[:k])                          # (in,k)
    right = Vt[:k, :] @ Qb.T                                # (k,out)
    return left @ right                                     # (in,out)


def build_all(w_early, w_final, rng_key):
    """Per (li,proj): ground-truth core mask + 3 surrogate masks + random mask, all at matched
    sparsity (global ground-truth core fraction f*, applied uniformly per entry-block).
    Returns masks dict, dense final deltas dict, f*, and Jaccard tallies."""
    layers = sorted({
        int(re.search(r"layers\.(\d+)\.", k).group(1))
        for k in w_final if "lora_a" in k
    })
    entries = []
    gt_masks, dWs = {}, {}
    tot_core, tot = 0, 0
    for li in layers:
        for proj in THINK_PROJS:
            ak, bk = layer_keys(w_final, li, proj)
            if ak is None:
                continue
            A1 = w_final[ak].astype(mx.float32)
            B1 = w_final[bk].astype(mx.float32)
            A2 = w_early[ak].astype(mx.float32)
            B2 = w_early[bk].astype(mx.float32)
            dW1 = A1 @ B1
            dW2 = A2 @ B2
            gt = (mx.abs(dW2) >= VEL_THRESH * mx.abs(dW1)) & (mx.sign(dW1) == mx.sign(dW2))
            mx.eval(gt, dW1)
            gt_masks[(li, proj)] = gt
            dWs[(li, proj)] = dW1
            tot_core += int(gt.sum().item())
            tot += gt.size
            entries.append((li, proj, A1, B1))
    f_star = tot_core / tot
    log(f"  ground-truth core fraction f* = {f_star:.4f} ({tot_core}/{tot})")

    masks = {"gt": gt_masks, "s1_mag": {}, "s2_svd": {}, "s3_energy": {}, "random": {}}
    inter = {k: 0 for k in masks if k != "gt"}
    union = {k: 0 for k in masks if k != "gt"}
    for li, proj, A1, B1 in entries:
        dW1 = dWs[(li, proj)]
        k = int(round(f_star * dW1.size))
        # S1: magnitude
        s1 = topk_mask(mx.abs(dW1), k)
        # S2: top-SVD agreement ratio
        dW_top = svd_top_truncation(A1, B1, K_SVD)
        s2 = topk_mask((dW_top * dW1) / (dW1 * dW1 + EPS), k)
        # S3: factor energy ||A_i|| * ||B_j||
        ra = mx.sqrt(mx.sum(A1 * A1, axis=1, keepdims=True))    # (in,1)
        cb = mx.sqrt(mx.sum(B1 * B1, axis=0, keepdims=True))    # (1,out)
        s3 = topk_mask(ra * cb, k)
        # random null at same fraction
        rng_key, sub = mx.random.split(rng_key)
        rnd = topk_mask(mx.random.uniform(shape=dW1.shape, key=sub), k)
        for name, m in (("s1_mag", s1), ("s2_svd", s2), ("s3_energy", s3), ("random", rnd)):
            masks[name][(li, proj)] = m
            gt = gt_masks[(li, proj)]
            inter[name] += int((m & gt).sum().item())
            union[name] += int((m | gt).sum().item())
        mx.eval(s1, s2, s3, rnd)
    jaccard = {name: inter[name] / max(union[name], 1) for name in inter}
    for name, j in sorted(jaccard.items()):
        log(f"  Jaccard({name}, gt) = {j:.4f}")
    return masks, dWs, f_star, jaccard


def masked_deltas(masks_for_cond, dWs):
    """Dense bf16 deltas mask ⊙ dW1000 for one condition."""
    out = {}
    for key, m in masks_for_cond.items():
        out[key] = (m.astype(mx.float32) * dWs[key]).astype(mx.bfloat16)
    mx.eval(list(out.values()))
    return out


# ----------------------------------------------------------------------------
# Wrappers (subclass nn.Module + setattr — F#831)
# ----------------------------------------------------------------------------

class LoRAProj(nn.Module):
    def __init__(self, base_linear, a, b, scale):
        super().__init__()
        self.linear = base_linear
        self.a = a
        self.b = b
        self.scale = scale
        self.linear.freeze()

    def __call__(self, x):
        y = self.linear(x)
        d = (x @ self.a) @ self.b
        return y + (self.scale * d).astype(x.dtype)


class DenseDeltaProj(nn.Module):
    def __init__(self, base_linear, dW, scale):
        super().__init__()
        self.linear = base_linear
        self.dW = dW
        self.scale = scale
        self.linear.freeze()

    def __call__(self, x):
        y = self.linear(x)
        d = x @ self.dW.astype(x.dtype)
        return y + (self.scale * d).astype(x.dtype)


def attach_math(model, math_ad):
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        ak, bk = layer_keys(math_ad, li, MATH_PROJ)
        if ak is None:
            continue
        a = math_ad[ak].astype(mx.float32)
        b = math_ad[bk].astype(mx.float32)
        setattr(layer.self_attn, "q_proj", LoRAProj(layer.self_attn.q_proj, a, b, MATH_SCALE))
        count += 1
    return count


def attach_thinking_lowrank(model, think_ad):
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        for proj in THINK_PROJS:
            ak, bk = layer_keys(think_ad, li, proj)
            if ak is None:
                continue
            a = think_ad[ak].astype(mx.float32)
            b = think_ad[bk].astype(mx.float32)
            base = getattr(layer.self_attn, proj)
            setattr(layer.self_attn, proj, LoRAProj(base, a, b, THINK_SCALE))
            count += 1
    return count


def attach_thinking_dense(model, deltas):
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        for proj in THINK_PROJS:
            if (li, proj) not in deltas:
                continue
            base = getattr(layer.self_attn, proj)
            setattr(layer.self_attn, proj, DenseDeltaProj(base, deltas[(li, proj)], THINK_SCALE))
            count += 1
    return count


# ----------------------------------------------------------------------------
# Generation (greedy) + GSM8K — identical harness to F#862
# ----------------------------------------------------------------------------

def format_chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def generate(model, tokenizer, prompt, max_new=MAX_NEW_TOKENS):
    ids = mx.array(tokenizer.encode(prompt))
    cache = make_prompt_cache(model)
    logits = model(ids[None], cache=cache)[:, -1, :]
    tok = mx.argmax(logits, axis=-1)
    mx.eval(tok)
    out = [tok.item()]
    eos = tokenizer.eos_token_id
    eot_enc = tokenizer.encode("<end_of_turn>")
    eot = eot_enc[-1] if eot_enc else eos
    for _ in range(max_new - 1):
        if out[-1] in (eos, eot):
            break
        logits = model(mx.array([[out[-1]]]), cache=cache)[:, -1, :]
        tok = mx.argmax(logits, axis=-1)
        mx.eval(tok)
        out.append(tok.item())
    del cache
    return tokenizer.decode(out), len(out)


def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    probs = []
    for i in range(min(n, len(ds))):
        it = ds[i]
        gold = it["answer"].split("####")[-1].strip().replace(",", "")
        probs.append({"question": it["question"], "gold": gold})
    log(f"  Loaded {len(probs)} GSM8K problems")
    return probs


_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def extract_answer(text):
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"####\s*(-?\$?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace("$", "").replace(",", "")
    nums = _NUM.findall(text)
    if nums:
        return nums[-1].replace("$", "").replace(",", "")
    return None


def num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-4
    except (TypeError, ValueError):
        return False


def gsm8k_prompt(q):
    return (
        f"{q}\n\nSolve step by step, then give the final numeric answer on its own "
        "line in the form '#### <number>'."
    )


def eval_gsm8k(model, tokenizer, problems):
    passed, details = 0, []
    for p in problems:
        prompt = format_chat(tokenizer, gsm8k_prompt(p["question"]))
        text, ntok = generate(model, tokenizer, prompt)
        pred = extract_answer(text)
        ok = num_eq(pred, p["gold"])
        passed += int(ok)
        details.append({"gold": p["gold"], "pred": pred, "ok": ok, "ntok": ntok})
    acc = passed / len(problems)
    log(f"    GSM8K acc = {acc:.4f} ({passed}/{len(problems)})")
    return acc, details


def run_condition(label, problems, math_ad, think_setup):
    log(f"\n=== COND {label} ===")
    model, tok = load(MODEL_ID)
    nm = attach_math(model, math_ad)
    nt = think_setup(model) if think_setup else 0
    mx.eval(model.parameters())
    gc.collect(); mx.clear_cache()
    log(f"  attached math={nm} thinking={nt}")
    acc, det = eval_gsm8k(model, tok, problems)
    log(f"  [mem] peak={mx.get_peak_memory()/1024**3:.2f}GB")
    del model, tok
    gc.collect(); mx.clear_cache()
    return {"acc": acc, "n_math": nm, "n_think": nt, "details": det}


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 72)
    log("exp_wildcat_static_velocity_surrogate")
    log(f"Base: {MODEL_ID}  early={THINK_EARLY.name} final={THINK_FINAL.name}")
    log(f"n_gsm8k={N_GSM8K} think_scale={THINK_SCALE} math_scale={MATH_SCALE} k_svd={K_SVD}")
    log("=" * 72)
    for pth in (THINK_EARLY, THINK_FINAL, MATH_AD):
        assert pth.exists(), f"missing {pth}"

    log("\n=== Phase 1: build ground-truth + static surrogate masks ===")
    w_early = mx.load(str(THINK_EARLY))
    w_final = mx.load(str(THINK_FINAL))
    rng_key = mx.random.key(SEED)
    masks, dWs, f_star, jaccard = build_all(w_early, w_final, rng_key)
    del w_early
    gc.collect(); mx.clear_cache()

    math_ad = mx.load(str(MATH_AD))
    log("\n=== Load GSM8K ===")
    problems = load_gsm8k(N_GSM8K)

    # Phase 2: behavioral conditions (fresh model each; dense deltas built/freed per cond)
    results_cond = {}

    results_cond["B_full_thinking"] = run_condition(
        "B_full_thinking", problems, math_ad, lambda m: attach_thinking_lowrank(m, w_final))

    for name in ("random", "s1_mag", "s2_svd", "s3_energy", "gt"):
        dd = masked_deltas(masks[name], dWs)
        results_cond[name] = run_condition(
            name, problems, math_ad, lambda m, dd=dd: attach_thinking_dense(m, dd))
        del dd
        gc.collect(); mx.clear_cache()

    acc = {k: v["acc"] for k, v in results_cond.items()}
    em_null = acc["random"]
    surrogates = {k: acc[k] for k in ("s1_mag", "s2_svd", "s3_energy")}
    best_name = max(surrogates, key=surrogates.get)
    em_best = surrogates[best_name]

    # ---- KILL 2334 (pre-registered) ----
    clause_null = em_best <= em_null + KILL_NULL_MARGIN
    clause_abs = em_best < KILL_ABS_EM
    killed = clause_null or clause_abs
    verdict = "killed" if killed else "supported"
    gate_pass = (em_best >= GATE_ABS_EM) and (em_best >= em_null + GATE_NULL_MARGIN)

    results = {
        "experiment_id": "exp_wildcat_static_velocity_surrogate",
        "config": {
            "base_model": MODEL_ID,
            "thinking_early": str(THINK_EARLY),
            "thinking_final": str(THINK_FINAL),
            "math_adapter": str(MATH_AD),
            "think_projs": list(THINK_PROJS),
            "math_proj": MATH_PROJ,
            "think_scale": THINK_SCALE,
            "math_scale": MATH_SCALE,
            "vel_thresh": VEL_THRESH,
            "k_svd": K_SVD,
            "matched_sparsity_f_star": f_star,
            "n_gsm8k": N_GSM8K,
            "max_new_tokens": MAX_NEW_TOKENS,
            "seed": SEED,
            "mlx_lm": "0.31.2",
            "f862_reference": {"B": 0.44, "gt_core": 0.74, "math_solo": 0.70},
        },
        "jaccard_vs_gt": jaccard,
        "acc": acc,
        "best_surrogate": {"name": best_name, "em": em_best},
        "random_null_em": em_null,
        "best_minus_null": em_best - em_null,
        "gt_anchor_em": acc["gt"],
        "conditions": {k: {"acc": v["acc"], "n_think": v["n_think"], "details": v["details"]}
                       for k, v in results_cond.items()},
        "kill_criteria": {
            "2334": {
                "text": "best surrogate EM <= random null + 2pp OR best EM < 0.59 (n=50, F#862 harness)",
                "type": "target_behavioral",
                "clause_null_fail": bool(clause_null),
                "clause_abs_fail": bool(clause_abs),
                "result": "fail" if killed else "pass",
            }
        },
        "gate": {
            "text": "best surrogate EM >= 0.68 AND >= random null + 6pp",
            "pass": bool(gate_pass),
        },
        "verdict": verdict,
        "all_pass": (not killed) and gate_pass,
        "is_smoke": False,
        "total_wall_clock_sec": time.time() - t0,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    log("\n" + "=" * 72)
    log(f"f*={f_star:.4f}  Jaccard: " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(jaccard.items())))
    log("acc: " + "  ".join(f"{k}={v:.3f}" for k, v in acc.items()))
    log(f"best surrogate = {best_name} EM={em_best:.3f}  null={em_null:.3f}  gt anchor={acc['gt']:.3f}")
    log(f"KILL 2334: null_fail={clause_null} abs_fail={clause_abs}  GATE pass={gate_pass}")
    log(f"VERDICT: {verdict}  all_pass={(not killed) and gate_pass}")
    log(f"Wrote {RESULTS_FILE}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
