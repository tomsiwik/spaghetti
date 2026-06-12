#!/usr/bin/env python3
"""exp_wildcat_dilution_norm_vs_structure — is F#881 mask dilution pure norm reduction or structure?

F#881: random mask at keep-fraction f=0.4335 on the dense thinking delta (v/o_proj, ckpt 0001000)
composed with the math q_proj adapter recovers +22pp of F#862's +30pp with NO core selection.
Single-point 5-arm cross at matched f on GSM8K n=100 (greedy, NO-thinking harness):

  A   dense thinking delta (f=1, interference baseline)
  B   random Bernoulli-topk mask at f, 3 seeds (the F#881 condition)
  C   dense scaled alpha=sqrt(f)=0.6584 — matches E[Frobenius norm] of the random mask
  C2  dense scaled alpha=f=0.4335 — mean-field control (exploratory, not verdict-bearing)
  D   keep-largest-|dW| at f (outliers kept)
  E   keep-smallest-|dW| at f (outliers removed)

KILL 2335 (pre-registered): if EM(C) >= mean EM(B) - 3pp -> dilution is pure norm reduction,
mask/sparsity arc collapses to a scalar alpha knob -> verdict KILLED.
SUPPORTED: mean EM(B) - EM(C) >= 5pp (structural). Gap in (3pp,5pp) -> provisional.
Secondary: 3-seed spread of B (max-min) > 8pp -> flag F#881 estimate as noise.
Structure read-out if structural: E>=B>D outlier-driven; |D-B|<=3pp distributed collision.

Harness adapted from exp_wildcat_static_velocity_surrogate (F#862 composed setting), except
enable_thinking=False and n=100 per spec. Composition Sum_i(B_i A_i) on disjoint projections,
scales <= 8. Wrappers via subclass nn.Module + setattr (F#831). NO MOCKS. is_smoke=False.
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
THINK_FINAL = REPO / "data" / "adapters" / "thinking-openthoughts-universal-v0" / "0001000_adapters.safetensors"
MATH_AD = REPO / "data" / "adapters" / "math" / "adapters.safetensors"

F_KEEP = 0.4335
ALPHA_NORM = F_KEEP ** 0.5          # 0.65841... matches expected Frobenius norm of random mask
ALPHA_MEAN = F_KEEP                 # exploratory mean-field control
THINK_SCALE = 1.0
MATH_SCALE = 6.0
THINK_PROJS = ("v_proj", "o_proj")
MATH_PROJ = "q_proj"
N_GSM8K = 100
MAX_NEW_TOKENS = 1024
RANDOM_SEEDS = (0, 1, 2)

KILL_NORM_MARGIN = 0.03             # C >= mean(B) - 3pp -> killed (pure norm)
SUPPORT_GAP = 0.05                  # mean(B) - C >= 5pp -> supported (structural)
NOISE_SPREAD = 0.08                 # B spread > 8pp -> F#881 estimate flagged as noise
STRUCT_MARGIN = 0.03                # D ~ B band for "distributed" classification


def log(msg):
    print(msg, flush=True)


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def layer_keys(weights, li, proj):
    ak = f"language_model.model.layers.{li}.self_attn.{proj}.lora_a"
    bk = f"language_model.model.layers.{li}.self_attn.{proj}.lora_b"
    return (ak, bk) if ak in weights and bk in weights else (None, None)


# ----------------------------------------------------------------------------
# Deltas and masks
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
    """Per (li,proj) dense fp32 delta dW = A@B for the thinking adapter."""
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


def deltas_scaled(dWs, alpha):
    out = {k: (alpha * v).astype(mx.bfloat16) for k, v in dWs.items()}
    mx.eval(list(out.values()))
    return out


def deltas_random_mask(dWs, seed):
    """Random mask keeping exactly round(f*size) entries per (li,proj)."""
    key = mx.random.key(seed)
    out = {}
    kept, tot = 0, 0
    for k, dW in dWs.items():
        key, sub = mx.random.split(key)
        kk = int(round(F_KEEP * dW.size))
        m = topk_mask(mx.random.uniform(shape=dW.shape, key=sub), kk)
        out[k] = (m.astype(mx.float32) * dW).astype(mx.bfloat16)
        kept += kk
        tot += dW.size
    mx.eval(list(out.values()))
    log(f"  random mask seed={seed}: keep {kept}/{tot} = {kept/tot:.4f}")
    return out


def deltas_magnitude_mask(dWs, keep_largest):
    out = {}
    for k, dW in dWs.items():
        kk = int(round(F_KEEP * dW.size))
        score = mx.abs(dW) if keep_largest else -mx.abs(dW)
        m = topk_mask(score, kk)
        out[k] = (m.astype(mx.float32) * dW).astype(mx.bfloat16)
    mx.eval(list(out.values()))
    return out


def frob_norm(deltas):
    s = 0.0
    for v in deltas.values():
        v32 = v.astype(mx.float32)
        s += float(mx.sum(v32 * v32).item())
    return s ** 0.5


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
# Generation (greedy) + GSM8K — no-thinking harness per spec
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


def run_condition(label, problems, math_ad, deltas):
    log(f"\n=== COND {label}  (delta frob norm = {frob_norm(deltas):.4f}) ===")
    model, tok = load(MODEL_ID)
    nm = attach_math(model, math_ad)
    nt = attach_thinking_dense(model, deltas)
    mx.eval(model.parameters())
    gc.collect(); mx.clear_cache()
    log(f"  attached math={nm} thinking={nt}")
    acc, det = eval_gsm8k(model, tok, problems)
    log(f"  [mem] peak={mx.get_peak_memory()/1024**3:.2f}GB")
    del model, tok
    gc.collect(); mx.clear_cache()
    return {"acc": acc, "n_math": nm, "n_think": nt,
            "delta_frob": frob_norm(deltas), "details": det}


def main():
    t0 = time.time()
    log("=" * 72)
    log("exp_wildcat_dilution_norm_vs_structure")
    log(f"f={F_KEEP}  alpha_norm=sqrt(f)={ALPHA_NORM:.4f}  alpha_mean={ALPHA_MEAN}")
    log(f"n_gsm8k={N_GSM8K} no-thinking, greedy, think_scale={THINK_SCALE} math_scale={MATH_SCALE}")
    log("=" * 72)
    for pth in (THINK_FINAL, MATH_AD):
        assert pth.exists(), f"missing {pth}"

    log("\n=== Phase 1: dense deltas ===")
    w_final = mx.load(str(THINK_FINAL))
    dWs = build_dense_deltas(w_final)
    del w_final
    gc.collect(); mx.clear_cache()

    math_ad = mx.load(str(MATH_AD))
    log("\n=== Load GSM8K ===")
    problems = load_gsm8k(N_GSM8K)

    # Phase 2: conditions — build deltas, run, free, in sequence
    conds = {}

    conds["A_dense"] = run_condition("A_dense (f=1)", problems, math_ad, deltas_scaled(dWs, 1.0))
    gc.collect(); mx.clear_cache()

    for s in RANDOM_SEEDS:
        dd = deltas_random_mask(dWs, s)
        conds[f"B_random_s{s}"] = run_condition(f"B_random seed={s}", problems, math_ad, dd)
        del dd
        gc.collect(); mx.clear_cache()

    conds["C_alpha_sqrt_f"] = run_condition(
        f"C alpha={ALPHA_NORM:.4f}", problems, math_ad, deltas_scaled(dWs, ALPHA_NORM))
    gc.collect(); mx.clear_cache()

    conds["C2_alpha_f"] = run_condition(
        f"C2 alpha={ALPHA_MEAN:.4f} (exploratory)", problems, math_ad, deltas_scaled(dWs, ALPHA_MEAN))
    gc.collect(); mx.clear_cache()

    dd = deltas_magnitude_mask(dWs, keep_largest=True)
    conds["D_keep_largest"] = run_condition("D keep-largest-|dW|", problems, math_ad, dd)
    del dd
    gc.collect(); mx.clear_cache()

    dd = deltas_magnitude_mask(dWs, keep_largest=False)
    conds["E_keep_smallest"] = run_condition("E keep-smallest-|dW|", problems, math_ad, dd)
    del dd
    gc.collect(); mx.clear_cache()

    acc = {k: v["acc"] for k, v in conds.items()}
    b_accs = [acc[f"B_random_s{s}"] for s in RANDOM_SEEDS]
    em_b_mean = sum(b_accs) / len(b_accs)
    b_spread = max(b_accs) - min(b_accs)
    em_c = acc["C_alpha_sqrt_f"]
    gap = em_b_mean - em_c

    # ---- KILL 2335 (pre-registered) ----
    pure_norm = em_c >= em_b_mean - KILL_NORM_MARGIN          # kill fires: mask arc = alpha knob
    structural = gap >= SUPPORT_GAP                            # supported: structure matters
    noise_flag = b_spread > NOISE_SPREAD                       # secondary: F#881 estimate is noise

    if pure_norm:
        verdict = "killed"
    elif structural:
        verdict = "supported"
    else:
        verdict = "provisional"   # ambiguous 3-5pp zone

    # structure classification (only meaningful if structural)
    em_d, em_e = acc["D_keep_largest"], acc["E_keep_smallest"]
    if structural:
        if em_e >= em_b_mean and em_b_mean > em_d:
            structure_class = "outlier_driven"      # removing big entries helps
        elif abs(em_d - em_b_mean) <= STRUCT_MARGIN:
            structure_class = "distributed_subspace_collision"
        else:
            structure_class = "mixed"
    else:
        structure_class = "n/a (not structural)"

    results = {
        "experiment_id": "exp_wildcat_dilution_norm_vs_structure",
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
            "alpha_mean_field": ALPHA_MEAN,
            "n_gsm8k": N_GSM8K,
            "max_new_tokens": MAX_NEW_TOKENS,
            "random_seeds": list(RANDOM_SEEDS),
            "enable_thinking": False,
            "mlx_lm": "0.31.2",
            "f881_reference": "random mask f=0.4335 recovered +22pp of F#862 +30pp",
        },
        "acc": acc,
        "em_b_mean": em_b_mean,
        "b_seed_accs": b_accs,
        "b_seed_spread": b_spread,
        "em_c_norm_matched": em_c,
        "gap_b_minus_c": gap,
        "structure_class": structure_class,
        "noise_flag_f881": bool(noise_flag),
        "kill_criteria": {
            "2335": {
                "text": "EM(C alpha=sqrt(f)) >= mean EM(B random) - 3pp -> pure norm reduction, "
                        "kill mask/sparsity arc; secondary: B 3-seed spread > 8pp -> F#881 is noise",
                "type": "target_behavioral",
                "pure_norm_fired": bool(pure_norm),
                "noise_flag_fired": bool(noise_flag),
                "result": "fail" if pure_norm else "pass",
            }
        },
        "verdict": verdict,
        "all_pass": bool(structural and not pure_norm),
        "is_smoke": False,
        "conditions": {k: {"acc": v["acc"], "n_think": v["n_think"],
                           "delta_frob": v["delta_frob"], "details": v["details"]}
                       for k, v in conds.items()},
        "total_wall_clock_sec": time.time() - t0,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    log("\n" + "=" * 72)
    log("acc: " + "  ".join(f"{k}={v:.3f}" for k, v in acc.items()))
    log(f"B mean={em_b_mean:.3f} spread={b_spread:.3f}  C={em_c:.3f}  gap(B-C)={gap:.3f}")
    log(f"KILL 2335: pure_norm={pure_norm}  structural={structural}  noise_flag={noise_flag}")
    log(f"structure_class={structure_class}")
    log(f"VERDICT: {verdict}  all_pass={structural and not pure_norm}")
    log(f"Wrote {RESULTS_FILE}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
