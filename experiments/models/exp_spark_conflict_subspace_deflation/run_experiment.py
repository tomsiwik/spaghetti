#!/usr/bin/env python3
"""exp_spark_conflict_subspace_deflation — composition damage lives in a tiny GLOBAL shared subspace.

Frozen base mlx-community/gemma-4-e4b-it-4bit + REAL r=6 q_proj adapters:
  data/adapters/math  (GSM8K)   and   data/adapters/medical (MedQA-domain).
Identical key set verified (84 keys, 42 layers). Native train scale s=6.0.

Per layer the LoRA delta-WEIGHT (row-vector x; y = W x + x @ Δ):
  Δ_math = s·(A_math B_math),  Δ_med = s·(A_med B_med),  D = Δ_math + Δ_med  (rank <= 12).

Arms (matched total scale, SAME merged model on the SAME mixed off-domain eval):
  base          : y = W x                                  (context)
  uniform_1N    : y = W x + c·D x,  c = 1/N = 1/2          (standing baseline F#863/867)
  deflate_k     : SVD(D)=UΣVᵀ; null top-k<=4 RIGHT-singular dirs; y = W x + c·(D (I-V_k V_kᵀ)) x

Δacc = acc_aggregate(best deflate_k) − acc_aggregate(uniform_1N), percentage points.

KILL 2307 (pre-registered, verbatim):
  "Deflating the top-k<=4 SVD directions of the SUMMED delta (BA_math+BA_med) per layer recovers
   <+3pp behavioral accuracy on a mixed off-domain eval vs the uniform-1/N merge baseline -> KILLED"
  Numeric: Δacc < +3.0 pp -> KILLED ; Δacc >= +3.0 pp -> SUPPORTED.

Eval is REAL exact-match: GSM8K (#### integer) + MedQA-USMLE-4-options (answer LETTER). NO PROXY.
Composition is Σ_i (A_i B_i), never (ΣA)(ΣB). c=1/2, s=6.0 <= 8. is_smoke=False.
Wrapper: subclass nn.Module + setattr (never __call__ override on instance, F#831). mlx-lm 0.31.2.
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
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 6 * 1024**3)

EXP_DIR = Path(__file__).resolve().parent
RESULTS_FILE = EXP_DIR / "results.json"
REPO = EXP_DIR.parent.parent.parent          # .../llm

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
ADAPTER_MATH = REPO / "data" / "adapters" / "math" / "adapters.safetensors"
ADAPTER_MED = REPO / "data" / "adapters" / "medical" / "adapters.safetensors"

LORA_SCALE = 6.0          # native train scale (<= 8 guard OK)
MERGE_C = 0.5             # uniform-1/N, N=2  -> matched total scale across arms
K_LIST = [1, 2, 3, 4]     # top-k SVD directions to deflate (kill says k<=4)
N_MATH = 80               # GSM8K problems
N_MED = 80                # MedQA problems
MAX_NEW_MATH = 320        # NTP adapters, no thinking
MAX_NEW_MED = 8           # only need the answer letter
SEED = 42
N_LAYERS_EXPECTED = 42
KILL_THRESH_PP = 3.0      # Δacc >= +3pp to survive


def log(msg):
    print(msg, flush=True)


def log_mem(label=""):
    log(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB "
        f"cache={mx.get_cache_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")


# ----------------------------------------------------------------------------
# Build per-layer summed delta D and its deflated / undeflated weight variants.
# All SVDs computed ONCE on CPU; cached. Returns {li: {"uniform": W, "k1":W,...}}
# plus the singular-value spectra for reporting.
# ----------------------------------------------------------------------------

def build_delta_weights(math_ad, med_ad):
    """For each layer, D = s(A_math B_math) + s(A_med B_med); shape (d_in, d_out).

    deflate_k weight = c · D (I - V_k V_kᵀ) ; uniform weight = c · D.
    """
    per_layer = {}
    spectra = {}
    for li in range(64):  # over-range; break on missing
        ak = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
        bk = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
        if ak not in math_ad:
            continue
        with mx.stream(mx.cpu):
            Am = math_ad[ak].astype(mx.float32)   # (d_in, r)
            Bm = math_ad[bk].astype(mx.float32)   # (r, d_out)
            Ad = med_ad[ak].astype(mx.float32)
            Bd = med_ad[bk].astype(mx.float32)
            D = LORA_SCALE * (Am @ Bm) + LORA_SCALE * (Ad @ Bd)   # (d_in, d_out)
            # SVD of D: U (d_in, k') , S (k',) , Vt (k', d_out)
            U, S, Vt = mx.linalg.svd(D, stream=mx.cpu)
            S_list = [float(x) for x in S.tolist()]
            variants = {"uniform": (MERGE_C * D)}
            for k in K_LIST:
                Vk = Vt[:k, :]                      # (k, d_out) right-singular rows
                # projector onto complement: I - Vk^T Vk  (d_out x d_out)
                P_perp_applied = D - (D @ Vk.T) @ Vk    # = D (I - Vk^T Vk)
                variants[f"k{k}"] = (MERGE_C * P_perp_applied)
            for v in variants.values():
                mx.eval(v)
        per_layer[li] = variants
        spectra[li] = S_list
    n = len(per_layer)
    assert n == N_LAYERS_EXPECTED, f"expected {N_LAYERS_EXPECTED} layers, got {n}"
    log(f"  Built delta weights for {n} layers; variants per layer: "
        f"{['uniform'] + [f'k{k}' for k in K_LIST]}")
    return per_layer, spectra


# ----------------------------------------------------------------------------
# q_proj wrapper: y = base(x) + x @ delta_weight   (delta_weight is None for base)
# subclass nn.Module + setattr (NEVER __call__ override on instance — F#831)
# ----------------------------------------------------------------------------

class MergedQProj(nn.Module):
    def __init__(self, base_linear, delta_weight):
        super().__init__()
        self.linear = base_linear            # frozen QuantizedLinear
        self.delta_weight = delta_weight     # (d_in, d_out) float32, or None
        self.linear.freeze()

    def __call__(self, x):
        y = self.linear(x)
        if self.delta_weight is not None:
            d = x @ self.delta_weight        # (..., d_out)
            y = y + d.astype(x.dtype)
        return y


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def attach(model, per_layer, arm):
    """arm in {'base','uniform', 'k1'..'k4'}. base => no delta."""
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        if li not in per_layer:
            continue
        dw = None if arm == "base" else per_layer[li][arm]
        wrapper = MergedQProj(layer.self_attn.q_proj, dw)
        setattr(layer.self_attn, "q_proj", wrapper)
        count += 1
    mx.eval(model.parameters())
    assert count == N_LAYERS_EXPECTED, f"expected {N_LAYERS_EXPECTED}, got {count}"
    log(f"  Attached {count} MergedQProj (arm={arm})")
    return model


# ----------------------------------------------------------------------------
# Generation (greedy)
# ----------------------------------------------------------------------------

def format_chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def generate(model, tokenizer, prompt, max_new):
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
    return tokenizer.decode(out)


# ----------------------------------------------------------------------------
# Eval data + exact-match scoring (REAL behavioral, no proxy)
# ----------------------------------------------------------------------------

def load_eval():
    from datasets import load_dataset
    g = load_dataset("gsm8k", "main", split="test")
    math_items = []
    for i in range(N_MATH):
        q = g[i]["question"]
        gold = g[i]["answer"].split("####")[-1].strip().replace(",", "")
        math_items.append({"q": q, "gold": gold})
    md = load_dataset("GBaker/med_qa-usmle-4-options", split="test")
    med_items = []
    for i in range(N_MED):
        it = md[i]
        opts = it["options"]
        med_items.append({
            "q": it["question"], "options": opts,
            "gold": it["answer_idx"].strip().upper(),
        })
    log(f"  Loaded {len(math_items)} GSM8K + {len(med_items)} MedQA problems")
    return math_items, med_items


def math_prompt(q):
    return (q + "\n\nThink step by step, then give the final answer on a new line "
                "as '#### <number>'.")


def parse_math(text):
    if "####" in text:
        tail = text.split("####")[-1]
    else:
        tail = text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")


def med_prompt(it):
    lines = [it["q"], ""]
    for letter in ["A", "B", "C", "D"]:
        lines.append(f"{letter}. {it['options'][letter]}")
    lines.append("")
    lines.append("Answer with the single letter (A, B, C, or D) of the correct option.")
    return "\n".join(lines)


def parse_med(text):
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    return m.group(1) if m else None


def num_eq(a, b):
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-4
    except ValueError:
        return a.strip() == b.strip()


def eval_arm(model, tok, math_items, med_items, arm):
    log(f"  --- eval arm={arm} ---")
    mc = 0
    for it in math_items:
        text = generate(model, tok, format_chat(tok, math_prompt(it["q"])), MAX_NEW_MATH)
        pred = parse_math(text)
        mc += int(num_eq(pred, it["gold"]))
    math_acc = mc / len(math_items)
    dc = 0
    for it in med_items:
        text = generate(model, tok, format_chat(tok, med_prompt(it)), MAX_NEW_MED)
        pred = parse_med(text)
        dc += int(pred == it["gold"])
    med_acc = dc / len(med_items)
    agg = (mc + dc) / (len(math_items) + len(med_items))
    log(f"    arm={arm}: math={math_acc:.4f} ({mc}/{len(math_items)})  "
        f"med={med_acc:.4f} ({dc}/{len(med_items)})  agg={agg:.4f}")
    return {"math_acc": math_acc, "med_acc": med_acc, "agg_acc": agg,
            "math_correct": mc, "med_correct": dc}


def run_arm(arm, per_layer, math_items, med_items):
    model, tok = load(MODEL_ID)
    attach(model, per_layer, arm)
    gc.collect(); mx.clear_cache()
    res = eval_arm(model, tok, math_items, med_items, arm)
    log_mem(f"{arm}-done")
    del model, tok
    gc.collect(); mx.clear_cache()
    return res


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 72)
    log("exp_spark_conflict_subspace_deflation")
    log(f"Base: {MODEL_ID}")
    log(f"math adapter: {ADAPTER_MATH}")
    log(f"med  adapter: {ADAPTER_MED}")
    log(f"scale={LORA_SCALE} merge_c={MERGE_C} K={K_LIST} N_math={N_MATH} N_med={N_MED}")
    log("=" * 72)
    assert ADAPTER_MATH.exists(), f"missing {ADAPTER_MATH}"
    assert ADAPTER_MED.exists(), f"missing {ADAPTER_MED}"
    log_mem("start")

    math_ad = mx.load(str(ADAPTER_MATH))
    med_ad = mx.load(str(ADAPTER_MED))
    assert set(math_ad.keys()) == set(med_ad.keys()), "adapter key sets differ"

    log("\n=== Build summed-delta weights + SVD spectra ===")
    per_layer, spectra = build_delta_weights(math_ad, med_ad)

    # spectrum summary: mean σ_i / σ_1 across layers, and mean top-k energy fraction
    import statistics as st
    n_sv = min(len(v) for v in spectra.values())
    mean_norm = []
    for j in range(n_sv):
        ratios = [spectra[li][j] / spectra[li][0] for li in spectra]
        mean_norm.append(st.mean(ratios))
    energy_topk = {}
    for k in K_LIST:
        fracs = []
        for li, S in spectra.items():
            tot = sum(s * s for s in S)
            top = sum(s * s for s in S[:k])
            fracs.append(top / tot if tot > 0 else 0.0)
        energy_topk[f"k{k}"] = st.mean(fracs)
    log(f"  mean normalized spectrum σ_i/σ_1 (first {n_sv}): "
        f"{[round(x,3) for x in mean_norm]}")
    log(f"  mean top-k energy fraction: {energy_topk}")

    log("\n=== Eval data ===")
    math_items, med_items = load_eval()

    arms = {}
    arms["base"] = run_arm("base", per_layer, math_items, med_items)
    arms["uniform_1N"] = run_arm("uniform", per_layer, math_items, med_items)
    for k in K_LIST:
        arms[f"deflate_k{k}"] = run_arm(f"k{k}", per_layer, math_items, med_items)

    uni = arms["uniform_1N"]["agg_acc"]
    deflate_aggs = {k: arms[f"deflate_k{k}"]["agg_acc"] for k in K_LIST}
    best_k = max(deflate_aggs, key=deflate_aggs.get)
    best_agg = deflate_aggs[best_k]
    delta_pp = (best_agg - uni) * 100.0

    killed = delta_pp < KILL_THRESH_PP
    verdict = "KILLED" if killed else "SUPPORTED"
    all_pass = not killed

    results = {
        "experiment_id": "exp_spark_conflict_subspace_deflation",
        "config": {
            "base_model": MODEL_ID,
            "adapter_math": str(ADAPTER_MATH),
            "adapter_med": str(ADAPTER_MED),
            "lora_scale": LORA_SCALE,
            "merge_c_1overN": MERGE_C,
            "k_list": K_LIST,
            "n_math": N_MATH, "n_med": N_MED,
            "max_new_math": MAX_NEW_MATH, "max_new_med": MAX_NEW_MED,
            "kill_thresh_pp": KILL_THRESH_PP,
            "mlx_lm": "0.31.2",
        },
        "svd_spectrum": {
            "mean_normalized_singular_values": [round(x, 4) for x in mean_norm],
            "mean_topk_energy_fraction": {k: round(v, 4) for k, v in energy_topk.items()},
            "per_layer_singular_values": spectra,
        },
        "arms": arms,
        "aggregate_acc": {
            "base": arms["base"]["agg_acc"],
            "uniform_1N": uni,
            **{f"deflate_k{k}": deflate_aggs[k] for k in K_LIST},
        },
        "best_k": best_k,
        "best_deflate_agg": best_agg,
        "delta_acc_pp_best_minus_uniform": delta_pp,
        "kill_criteria": {
            "2307": {
                "text": ("Deflating the top-k<=4 SVD directions of the SUMMED delta "
                         "(BA_math+BA_med) per layer recovers <+3pp behavioral accuracy on a "
                         "mixed off-domain eval vs the uniform-1/N merge baseline -> KILLED"),
                "type": "target_behavioral",
                "delta_acc_pp": delta_pp,
                "threshold_pp": KILL_THRESH_PP,
                "result": "fail" if killed else "pass",
            }
        },
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "total_wall_clock_sec": time.time() - t0,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    log("\n" + "=" * 72)
    log(f"agg  base={arms['base']['agg_acc']:.3f}  uniform_1N={uni:.3f}  "
        + "  ".join(f"k{k}={deflate_aggs[k]:.3f}" for k in K_LIST))
    log(f"best_k={best_k}  Δacc = {delta_pp:+.2f} pp (best deflate − uniform)")
    log(f"KILL 2307 threshold = +{KILL_THRESH_PP}pp  ->  VERDICT: {verdict}  all_pass={all_pass}")
    log(f"Wrote {RESULTS_FILE}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
