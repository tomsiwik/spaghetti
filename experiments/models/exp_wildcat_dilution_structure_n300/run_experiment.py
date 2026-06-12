#!/usr/bin/env python3
"""exp_wildcat_dilution_structure_n300 — WHICH structure carries the +5pp mask-dilution gap?

Follow-up to exp_wildcat_dilution_norm_vs_structure (F#882, n=100, provisional: B-C gap exactly
5.0pp, knife-edge). Six arms at ONE matched Frobenius norm N* = sqrt(f)*||dW||_F ~ 2.081 on the
SAME fixed 300 GSM8K test items (no-thinking, greedy, paired McNemar):

  B   random mask f=0.4335, seeds {0,1,2}            (norm ~ N* by construction, residual rescale)
  C   dense alpha=sqrt(f)=0.6584                     (norm = N* by construction)
  D'  keep-largest-|dW| mask, rescaled to N*         (~ x0.676)
  E'  keep-smallest-|dW| mask, rescaled to N*        (~ x4.83 — the F#882 anomaly, norm-deconfounded)
  P   keep-largest mask with WITHIN-ROW permutation of mask bits (same per-row density as D',
      destroys coordinates, keeps sparsity statistics), rescaled to N*

KILL 2336 (pre-registered): pooled mean EM(B) - EM(C) < 3pp -> dilution is pure norm reduction,
mask arc dies -> KILLED. SUPPORTED: gap >= 5pp AND pooled paired McNemar p < 0.05.
Secondary kill flag: EM(E') <= EM(C) -> F#882 E=0.92 anomaly was a norm artifact.
Read-outs: E' >= mean(B)+3pp -> small-|dW| subspace carries it; |P-D'| <= 3pp -> topology
irrelevant (sparsity per se); D'-P >= 5pp -> specific coordinates matter (lottery-ticket rung).

Harness adapted as-is from exp_wildcat_dilution_norm_vs_structure/run_experiment.py (F#831
wrappers, Sum_i(B_i A_i) composition on disjoint projections, scales <= 8). NO MOCKS. is_smoke=False.
Runtime guard: after first condition, if projected total > 2h, drop B to seeds {0,1}.
"""

import gc
import json
import math
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
THINK_FINAL = REPO / "data" / "adapters" / "thinking-openthoughts-universal-v0" / "0001000_adapters.safetensors"
MATH_AD = REPO / "data" / "adapters" / "math" / "adapters.safetensors"

F_KEEP = 0.4335
ALPHA_NORM = F_KEEP ** 0.5          # 0.65841 — matches E[Frobenius norm] of the random mask
THINK_SCALE = 1.0
MATH_SCALE = 6.0
THINK_PROJS = ("v_proj", "o_proj")
MATH_PROJ = "q_proj"
N_GSM8K = 300
MAX_NEW_TOKENS = 1024
RANDOM_SEEDS = (0, 1, 2)
PERM_SEED = 1234

KILL_GAP = 0.03                     # B-C < 3pp -> killed (pure norm)
SUPPORT_GAP = 0.05                  # B-C >= 5pp (and p<0.05) -> supported
READOUT_MARGIN = 0.03               # E' >= B+3pp; |P-D'| <= 3pp bands
COORD_GAP = 0.05                    # D'-P >= 5pp -> coordinates matter
RUNTIME_GUARD_SEC = 2 * 3600        # projected > 2h -> drop B to 2 seeds


def log(msg):
    print(msg, flush=True)


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def layer_keys(weights, li, proj):
    ak = f"language_model.model.layers.{li}.self_attn.{proj}.lora_a"
    bk = f"language_model.model.layers.{li}.self_attn.{proj}.lora_b"
    return (ak, bk) if ak in weights and bk in weights else (None, None)


# ----------------------------------------------------------------------------
# Deltas, masks, norm matching
# ----------------------------------------------------------------------------

def topk_mask(score, k):
    flat = score.reshape(-1)
    if k <= 0:
        return mx.zeros(score.shape, dtype=mx.bool_)
    if k >= flat.size:
        return mx.ones(score.shape, dtype=mx.bool_)
    part = mx.partition(flat, flat.size - k)
    thresh = part[flat.size - k]
    return score >= thresh


def build_dense_deltas(w_final):
    layers = sorted({
        int(re.search(r"layers\.(\d+)\.", k).group(1))
        for k in w_final if "lora_a" in k
    })
    dWs = {}
    for li in layers:
        for proj in THINK_PROJS:
            ak, bk = layer_keys(w_final, li, proj)
            if ak is None:
                continue
            dW = w_final[ak].astype(mx.float32) @ w_final[bk].astype(mx.float32)
            mx.eval(dW)
            dWs[(li, proj)] = dW
    log(f"  built {len(dWs)} dense deltas")
    return dWs


def frob_norm_f32(deltas):
    s = 0.0
    for v in deltas.values():
        v32 = v.astype(mx.float32)
        s += float(mx.sum(v32 * v32).item())
    return s ** 0.5


def to_bf16(deltas):
    out = {k: v.astype(mx.bfloat16) for k, v in deltas.items()}
    mx.eval(list(out.values()))
    return out


def rescale_to(deltas_f32, target):
    """Single global scalar so the concatenated Frobenius norm equals target."""
    cur = frob_norm_f32(deltas_f32)
    s = target / cur
    out = {k: s * v for k, v in deltas_f32.items()}
    log(f"  rescale x{s:.4f}  (norm {cur:.4f} -> {frob_norm_f32(out):.4f})")
    return out, s


def deltas_random_mask_f32(dWs, seed):
    key = mx.random.key(seed)
    out = {}
    for k, dW in dWs.items():
        key, sub = mx.random.split(key)
        kk = int(round(F_KEEP * dW.size))
        m = topk_mask(mx.random.uniform(shape=dW.shape, key=sub), kk)
        out[k] = m.astype(mx.float32) * dW
    mx.eval(list(out.values()))
    return out


def magnitude_masks(dWs, keep_largest):
    masks = {}
    for k, dW in dWs.items():
        kk = int(round(F_KEEP * dW.size))
        score = mx.abs(dW) if keep_largest else -mx.abs(dW)
        masks[k] = topk_mask(score, kk)
    mx.eval(list(masks.values()))
    return masks


def apply_masks_f32(dWs, masks):
    out = {k: masks[k].astype(mx.float32) * dW for k, dW in dWs.items()}
    mx.eval(list(out.values()))
    return out


def permute_mask_within_rows(mask, key):
    """Shuffle mask bits within each row: identical per-row density, random columns."""
    k_per_row = mx.sum(mask.astype(mx.int32), axis=1)        # (rows,)
    u = mx.random.uniform(shape=mask.shape, key=key)
    order = mx.argsort(u, axis=1)
    ranks = mx.argsort(order, axis=1)
    return ranks < k_per_row[:, None]


def permuted_masks(masks, seed):
    key = mx.random.key(seed)
    out = {}
    for k, m in masks.items():
        key, sub = mx.random.split(key)
        pm = permute_mask_within_rows(m, sub)
        # sanity: identical per-row density
        d0 = mx.sum(m.astype(mx.int32)).item()
        d1 = mx.sum(pm.astype(mx.int32)).item()
        assert d0 == d1, f"permutation changed density for {k}: {d0} vs {d1}"
        out[k] = pm
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
# Generation (greedy) + GSM8K — no-thinking harness
# ----------------------------------------------------------------------------

def format_chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
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
    log(f"  Loaded {len(probs)} GSM8K problems (fixed first {n} of test)")
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
    t0 = time.time()
    for i, p in enumerate(problems):
        prompt = format_chat(tokenizer, gsm8k_prompt(p["question"]))
        text, ntok = generate(model, tokenizer, prompt)
        pred = extract_answer(text)
        ok = num_eq(pred, p["gold"])
        passed += int(ok)
        details.append({"gold": p["gold"], "pred": pred, "ok": ok, "ntok": ntok})
        if (i + 1) % 50 == 0:
            log(f"    [{i+1}/{len(problems)}] acc so far {passed/(i+1):.3f} "
                f"({(time.time()-t0)/(i+1):.1f}s/item)")
    acc = passed / len(problems)
    log(f"    GSM8K acc = {acc:.4f} ({passed}/{len(problems)})")
    return acc, details


def run_condition(label, problems, math_ad, deltas_bf16):
    log(f"\n=== COND {label}  (delta frob norm = {frob_norm_f32(deltas_bf16):.4f}) ===")
    t0 = time.time()
    model, tok = load(MODEL_ID)
    nm = attach_math(model, math_ad)
    nt = attach_thinking_dense(model, deltas_bf16)
    mx.eval(model.parameters())
    gc.collect(); mx.clear_cache()
    log(f"  attached math={nm} thinking={nt}")
    acc, det = eval_gsm8k(model, tok, problems)
    log(f"  [mem] peak={mx.get_peak_memory()/1024**3:.2f}GB  cond_time={time.time()-t0:.0f}s")
    del model, tok
    gc.collect(); mx.clear_cache()
    return {"acc": acc, "n_math": nm, "n_think": nt,
            "delta_frob": frob_norm_f32(deltas_bf16), "details": det,
            "cond_sec": time.time() - t0}


# ----------------------------------------------------------------------------
# Paired McNemar (exact binomial on discordant pairs, normal fallback)
# ----------------------------------------------------------------------------

def mcnemar(oks_x, oks_y):
    """Two-sided McNemar on paired correctness lists. Returns dict."""
    b = sum(1 for x, y in zip(oks_x, oks_y) if x and not y)   # x-only
    c = sum(1 for x, y in zip(oks_x, oks_y) if y and not x)   # y-only
    n = b + c
    if n == 0:
        return {"x_only": b, "y_only": c, "n_discordant": 0, "p_two_sided": 1.0, "method": "degenerate"}
    try:
        from scipy.stats import binomtest
        p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
        method = "exact_binomial"
    except Exception:
        # normal approximation with continuity correction
        z = (abs(b - c) - 1) / math.sqrt(n)
        p = math.erfc(z / math.sqrt(2))
        method = "normal_approx_cc"
    return {"x_only": b, "y_only": c, "n_discordant": n, "p_two_sided": float(p), "method": method}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    t0 = time.time()
    log("=" * 72)
    log("exp_wildcat_dilution_structure_n300")
    log(f"f={F_KEEP}  alpha=sqrt(f)={ALPHA_NORM:.4f}  n={N_GSM8K} no-thinking greedy")
    log(f"think_scale={THINK_SCALE} math_scale={MATH_SCALE}")
    log("=" * 72)
    for pth in (THINK_FINAL, MATH_AD):
        assert pth.exists(), f"missing {pth}"

    log("\n=== Phase 1: dense deltas + norm-matched arms ===")
    w_final = mx.load(str(THINK_FINAL))
    dWs = build_dense_deltas(w_final)
    del w_final
    gc.collect(); mx.clear_cache()

    dense_frob = frob_norm_f32(dWs)
    target = ALPHA_NORM * dense_frob
    log(f"  dense ||dW||_F = {dense_frob:.4f}  target N* = {target:.4f}")

    math_ad = mx.load(str(MATH_AD))
    log("\n=== Load GSM8K ===")
    problems = load_gsm8k(N_GSM8K)

    # Pre-build all delta sets (bf16) with exact norm matching; record scale factors
    arms = {}      # name -> (deltas_bf16, scale_factor, raw_norm)
    log("\n  arm C: dense alpha")
    c_f32 = {k: ALPHA_NORM * v for k, v in dWs.items()}
    arms["C_dense_alpha"] = (to_bf16(c_f32), ALPHA_NORM, dense_frob)
    del c_f32

    for s in RANDOM_SEEDS:
        log(f"  arm B seed={s}: random mask + residual rescale")
        b_f32 = deltas_random_mask_f32(dWs, s)
        raw = frob_norm_f32(b_f32)
        b_f32, sc = rescale_to(b_f32, target)
        arms[f"B_random_s{s}"] = (to_bf16(b_f32), sc, raw)
        del b_f32
        gc.collect(); mx.clear_cache()

    log("  arm D': keep-largest mask, rescaled")
    masks_large = magnitude_masks(dWs, keep_largest=True)
    d_f32 = apply_masks_f32(dWs, masks_large)
    raw_d = frob_norm_f32(d_f32)
    d_f32, sc_d = rescale_to(d_f32, target)
    arms["Dp_keep_largest"] = (to_bf16(d_f32), sc_d, raw_d)
    del d_f32
    gc.collect(); mx.clear_cache()

    log("  arm E': keep-smallest mask, rescaled UP")
    masks_small = magnitude_masks(dWs, keep_largest=False)
    e_f32 = apply_masks_f32(dWs, masks_small)
    raw_e = frob_norm_f32(e_f32)
    e_f32, sc_e = rescale_to(e_f32, target)
    arms["Ep_keep_smallest"] = (to_bf16(e_f32), sc_e, raw_e)
    del e_f32, masks_small
    gc.collect(); mx.clear_cache()

    log("  arm P: within-row permuted keep-largest mask, rescaled")
    masks_perm = permuted_masks(masks_large, PERM_SEED)
    p_f32 = apply_masks_f32(dWs, masks_perm)
    raw_p = frob_norm_f32(p_f32)
    p_f32, sc_p = rescale_to(p_f32, target)
    arms["P_row_permuted"] = (to_bf16(p_f32), sc_p, raw_p)
    del p_f32, masks_perm, masks_large, dWs
    gc.collect(); mx.clear_cache()

    # Phase 2: run conditions sequentially; runtime guard after first
    order = ["B_random_s0", "B_random_s1", "B_random_s2",
             "C_dense_alpha", "Dp_keep_largest", "Ep_keep_smallest", "P_row_permuted"]
    conds = {}
    dropped_seed2 = False
    for idx, name in enumerate(order):
        if name == "B_random_s2" and dropped_seed2:
            log("\n  [guard] skipping B_random_s2 (projected runtime > 2h)")
            continue
        deltas, sc, raw = arms[name]
        conds[name] = run_condition(name, problems, math_ad, deltas)
        conds[name]["rescale_factor"] = sc
        conds[name]["raw_norm_before_rescale"] = raw
        arms[name] = None  # free
        gc.collect(); mx.clear_cache()
        if idx == 0:
            projected = conds[name]["cond_sec"] * len(order)
            log(f"  [guard] first cond {conds[name]['cond_sec']:.0f}s -> projected {projected/60:.0f}min")
            if projected > RUNTIME_GUARD_SEC:
                dropped_seed2 = True

    # ---- Analysis ----
    b_names = [n for n in conds if n.startswith("B_random")]
    b_accs = [conds[n]["acc"] for n in b_names]
    em_b = sum(b_accs) / len(b_accs)
    em_c = conds["C_dense_alpha"]["acc"]
    em_d = conds["Dp_keep_largest"]["acc"]
    em_e = conds["Ep_keep_smallest"]["acc"]
    em_p = conds["P_row_permuted"]["acc"]
    gap_bc = em_b - em_c

    oks = {n: [d["ok"] for d in conds[n]["details"]] for n in conds}
    # pooled B vs C: every (seed, item) pair against the same C item result
    pooled_b = [ok for n in b_names for ok in oks[n]]
    pooled_c = oks["C_dense_alpha"] * len(b_names)
    mc_bc = mcnemar(pooled_b, pooled_c)
    mc_ec = mcnemar(oks["Ep_keep_smallest"], oks["C_dense_alpha"])
    mc_eb = mcnemar(oks["Ep_keep_smallest"] * len(b_names), pooled_b)
    mc_pd = mcnemar(oks["P_row_permuted"], oks["Dp_keep_largest"])

    # ---- KILL 2336 (pre-registered) ----
    kill_fired = gap_bc < KILL_GAP
    structural = (gap_bc >= SUPPORT_GAP) and (mc_bc["p_two_sided"] < 0.05)
    secondary_kill_e = em_e <= em_c          # E=0.92 anomaly was norm artifact

    if kill_fired:
        verdict = "killed"
    elif structural:
        verdict = "supported"
    else:
        verdict = "provisional"

    # Read-out tree (meaningful only if not killed)
    small_subspace = em_e >= em_b + READOUT_MARGIN
    topology_irrelevant = abs(em_p - em_d) <= READOUT_MARGIN
    coordinates_matter = (em_d - em_p) >= COORD_GAP
    if kill_fired:
        structure_readout = "n/a (norm-only, killed)"
    elif small_subspace:
        structure_readout = "small_dW_subspace_carries_benefit"
    elif coordinates_matter:
        structure_readout = "specific_coordinates_matter (lottery-ticket rung opens)"
    elif topology_irrelevant:
        structure_readout = "sparsity_per_se (topology irrelevant)"
    else:
        structure_readout = "mixed"

    results = {
        "experiment_id": "exp_wildcat_dilution_structure_n300",
        "config": {
            "base_model": MODEL_ID,
            "thinking_final": str(THINK_FINAL),
            "math_adapter": str(MATH_AD),
            "think_projs": list(THINK_PROJS),
            "math_proj": MATH_PROJ,
            "think_scale": THINK_SCALE,
            "math_scale": MATH_SCALE,
            "f_keep": F_KEEP,
            "alpha_norm_matched": ALPHA_NORM,
            "target_frob": target,
            "dense_frob": dense_frob,
            "n_gsm8k": N_GSM8K,
            "max_new_tokens": MAX_NEW_TOKENS,
            "random_seeds_run": [int(n[-1]) for n in b_names],
            "perm_seed": PERM_SEED,
            "enable_thinking": False,
            "dropped_seed2_runtime_guard": dropped_seed2,
            "f882_reference": "n=100: B=0.84 C=0.79 D=0.68 E=0.92 (E unrescaled norm 0.431)",
        },
        "acc": {n: conds[n]["acc"] for n in conds},
        "em_b_mean": em_b,
        "b_seed_accs": b_accs,
        "em_c": em_c,
        "em_dprime": em_d,
        "em_eprime": em_e,
        "em_p_permuted": em_p,
        "gap_b_minus_c": gap_bc,
        "mcnemar": {
            "B_pooled_vs_C": mc_bc,
            "Eprime_vs_C": mc_ec,
            "Eprime_vs_B_pooled": mc_eb,
            "P_vs_Dprime": mc_pd,
        },
        "readouts": {
            "small_subspace_Ep_ge_B_plus_3pp": bool(small_subspace),
            "topology_irrelevant_P_within_3pp_of_Dp": bool(topology_irrelevant),
            "coordinates_matter_Dp_minus_P_ge_5pp": bool(coordinates_matter),
            "structure_readout": structure_readout,
        },
        "kill_criteria": {
            "2336": {
                "text": "pooled mean EM(B) - EM(C) < 3pp on same 300 items -> pure norm reduction, "
                        "mask arc dies; secondary: E' <= C -> F#882 E=0.92 was a norm artifact",
                "type": "target_behavioral",
                "primary_kill_fired": bool(kill_fired),
                "secondary_kill_Eprime_le_C_fired": bool(secondary_kill_e),
                "result": "fail" if kill_fired else "pass",
            }
        },
        "verdict": verdict,
        "all_pass": bool(structural and not kill_fired),
        "is_smoke": False,
        "conditions": {n: {k: v for k, v in conds[n].items()} for n in conds},
        "total_wall_clock_sec": time.time() - t0,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    log("\n" + "=" * 72)
    log("acc: " + "  ".join(f"{n}={conds[n]['acc']:.3f}" for n in conds))
    log(f"B mean={em_b:.3f}  C={em_c:.3f}  gap(B-C)={gap_bc:.3f}  "
        f"McNemar p={mc_bc['p_two_sided']:.4g} ({mc_bc['x_only']}/{mc_bc['y_only']} discordant)")
    log(f"D'={em_d:.3f}  E'={em_e:.3f}  P={em_p:.3f}")
    log(f"KILL 2336: primary={kill_fired}  secondary(E'<=C)={secondary_kill_e}")
    log(f"readout: {structure_readout}")
    log(f"VERDICT: {verdict}  all_pass={structural and not kill_fired}")
    log(f"Wrote {RESULTS_FILE}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
