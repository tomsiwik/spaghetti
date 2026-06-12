#!/usr/bin/env python3
"""exp_spark_interference_eos — off-domain LoRA delta crossing on-domain delta is a free EOS.

Frozen base gemma-4-e4b-it-4bit + r=6 q_proj math(GSM8K) and medical(PubMedQA) adapters (F#627).
GSM8K test[0:50], greedy temp=0, no training.

Per decode step t, on the SAME committed context (the math stream's own argmax tokens), we compute three
full next-token logit vectors via three independent forwards sharing committed tokens:
    l_base = logits(base only)
    l_on   = logits(base + math   adapter)     <- this stream DRIVES generation (x_t = argmax l_on)
    l_off  = logits(base + medical adapter)
and the per-step delta magnitudes
    on_delta(t)  = || l_on(t)  - l_base(t) ||_2
    off_delta(t) = || l_off(t) - l_base(t) ||_2.

Early-stop (free EOS): halt at the first t where off_delta > on_delta (T_cross). The math stream still
runs to natural EOS (T_eos) as the BASELINE arm; we compare early-stop vs to-EOS using the SAME math
adapter config (F#866 like-for-like).

One model, a switchable single-adapter q_proj wrapper (mode in {base, math, medical}) and three separate
KV caches. Wrapper attaches via subclass nn.Module + setattr (never __call__ override on instance, F#831).
Composition is NOT used: three separate SINGLE-adapter forwards, never (sum B)(sum A), never a 2-adapter sum.

KILL K2304 (target, behavioral). KILL if ANY:
  1. exact_match(early_stop) < exact_match(math_on_to_EOS) - 2pp           (baseline = math-on-to-EOS, F#866)
  2. median(1 - T_cross/T_eos) < 0.15                                       (median token savings < 15%)
  3. crossover precedes the gold answer number in > 20% of correct cases.

NO MOCKS. Real model, real adapters, real GSM8K scoring. is_smoke=False. mlx-lm == 0.31.2.
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

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
ADAPTER_DIR = EXP_DIR.parent / "exp_composition_residual_analysis"
ADAPTER_MATH = ADAPTER_DIR / "adapter_math.safetensors"
ADAPTER_MEDICAL = ADAPTER_DIR / "adapter_medical.safetensors"

LORA_SCALE = 6.0          # <= 8 guard OK (F#627 recipe)
LORA_RANK = 6
N_GSM8K = 50
MAX_NEW_TOKENS = 1024     # thinking-mode headroom
SEED = 42
N_LAYERS_EXPECTED = 42

# Kill thresholds
K_ACC_DROP = 0.02         # early-stop must be within 2pp of math-on-to-EOS
K_MED_SAVINGS = 0.15      # median token savings must be >= 15%
K_EARLY_CROSS = 0.20      # crossover may precede answer in at most 20% of correct cases


def log(msg):
    print(msg, flush=True)


def log_mem(label=""):
    log(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB "
        f"cache={mx.get_cache_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")


# ----------------------------------------------------------------------------
# Switchable single-adapter q_proj wrapper.
# mode in {"base","math","medical"} selects which (a,b) delta (or none) is added.
# subclass nn.Module + setattr (NEVER __call__ override on instance — F#831)
# ----------------------------------------------------------------------------

class SwitchQProj(nn.Module):
    """y = linear(x) [+ scale*(x@a_math)@b_math]  OR  [+ scale*(x@a_med)@b_med], per global mode."""

    def __init__(self, base_linear, am, bm, amed, bmed, scale, mode_box):
        super().__init__()
        self.linear = base_linear
        self.am, self.bm = am, bm           # math:    (in,r),(r,out)
        self.amed, self.bmed = amed, bmed   # medical: (in,r),(r,out)
        self.scale = scale
        self.mode_box = mode_box            # 1-element list holding current mode string
        self.linear.freeze()

    def __call__(self, x):
        y = self.linear(x)
        mode = self.mode_box[0]
        if mode == "math":
            d = (x @ self.am) @ self.bm
            y = y + (self.scale * d).astype(x.dtype)
        elif mode == "medical":
            d = (x @ self.amed) @ self.bmed
            y = y + (self.scale * d).astype(x.dtype)
        # mode == "base": no delta
        return y


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def attach_switch(model, math_ad, med_ad, scale, mode_box):
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        ak = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
        bk = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
        if ak not in math_ad or bk not in math_ad:
            continue
        assert ak in med_ad and bk in med_ad, f"medical adapter missing {ak}"
        base_linear = layer.self_attn.q_proj
        am = math_ad[ak].astype(mx.float32)
        bm = math_ad[bk].astype(mx.float32)
        amed = med_ad[ak].astype(mx.float32)
        bmed = med_ad[bk].astype(mx.float32)
        wrapper = SwitchQProj(base_linear, am, bm, amed, bmed, scale, mode_box)
        setattr(layer.self_attn, "q_proj", wrapper)
        count += 1
    mx.eval(model.parameters())
    log(f"  Attached {count} SwitchQProj wrappers")
    assert count == N_LAYERS_EXPECTED, f"expected {N_LAYERS_EXPECTED} wrapped, got {count}"
    return model


# ----------------------------------------------------------------------------
# GSM8K data + scoring
# ----------------------------------------------------------------------------

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


def gsm8k_prompt(q):
    return ("Solve this math problem. Show your reasoning, then give the final "
            "answer on its own line as '#### <number>'.\n\n" + q)


def format_chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize_num(s):
    s = s.strip().replace(",", "").rstrip(".")
    if s.endswith(".0"):
        s = s[:-2]
    return s


def extract_pred(text):
    """Final numeric answer: prefer '#### N', else last number in text."""
    if not text:
        return None
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if m:
        return normalize_num(m.group(1))
    nums = _NUM_RE.findall(text)
    if nums:
        return normalize_num(nums[-1])
    return None


def correct(pred, gold):
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(normalize_num(gold))) < 1e-6
    except ValueError:
        return pred == normalize_num(gold)


def first_answer_token_index(token_strs, gold):
    """Index (0-based, in the committed token list) at/after which the gold number first appears
    as a contiguous substring of the running decoded text. Returns len if never found."""
    gold_n = normalize_num(gold)
    running = ""
    for i, ts in enumerate(token_strs):
        running += ts
        # compare on comma-stripped running text so '1,234' matches gold '1234'
        if gold_n in running.replace(",", ""):
            return i
    return len(token_strs)


# ----------------------------------------------------------------------------
# Triple-stream greedy generation with crossover early-stop detection.
# ----------------------------------------------------------------------------

def logits_step(model, mode_box, mode, cache, ids):
    """Run one forward in the given adapter mode, return last-position logits (1D, float32)."""
    mode_box[0] = mode
    out = model(ids, cache=cache)[:, -1, :]
    return out[0].astype(mx.float32)


def generate_triple(model, tokenizer, mode_box, prompt):
    """Math stream drives generation to natural EOS. At each committed step compute base/math/medical
    logits on the SAME context and record on_delta, off_delta, crossover. Returns dict."""
    prompt_ids = mx.array(tokenizer.encode(prompt))[None]
    cache_base = make_prompt_cache(model)
    cache_on = make_prompt_cache(model)
    cache_off = make_prompt_cache(model)

    eos = tokenizer.eos_token_id
    eot_enc = tokenizer.encode("<end_of_turn>")
    eot = eot_enc[-1] if eot_enc else eos
    stops = {eos, eot}

    committed = []          # token ids emitted by the math (on) stream
    token_strs = []         # per-token decoded string
    on_deltas = []
    off_deltas = []
    t_cross = None          # first step index where off_delta > on_delta

    # First step: feed full prompt to each cache (in its own mode).
    cur = prompt_ids
    step = 0
    while step < MAX_NEW_TOKENS:
        l_base = logits_step(model, mode_box, "base", cache_base, cur)
        l_on = logits_step(model, mode_box, "math", cache_on, cur)
        l_off = logits_step(model, mode_box, "medical", cache_off, cur)
        mx.eval(l_base, l_on, l_off)

        on_d = float(mx.linalg.norm(l_on - l_base).item())
        off_d = float(mx.linalg.norm(l_off - l_base).item())
        on_deltas.append(on_d)
        off_deltas.append(off_d)
        if t_cross is None and off_d > on_d:
            t_cross = step      # crossover at this committed-token step

        tok = int(mx.argmax(l_on).item())   # math stream drives generation
        committed.append(tok)
        token_strs.append(tokenizer.decode([tok]))
        step += 1
        if tok in stops:
            break
        cur = mx.array([[tok]])

    del cache_base, cache_on, cache_off
    full_text = tokenizer.decode(committed)
    t_eos = len(committed)
    if t_cross is None:
        t_cross = t_eos     # never crossed -> no savings
    early_text = tokenizer.decode(committed[: t_cross + 1])  # include the crossover token

    return {
        "committed": committed,
        "token_strs": token_strs,
        "full_text": full_text,
        "early_text": early_text,
        "t_eos": t_eos,
        "t_cross": t_cross,
        "on_deltas": on_deltas,
        "off_deltas": off_deltas,
    }


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 72)
    log("exp_spark_interference_eos")
    log(f"Base: {MODEL_ID}")
    log(f"Math adapter:    {ADAPTER_MATH}")
    log(f"Medical adapter: {ADAPTER_MEDICAL}")
    log(f"n_gsm8k={N_GSM8K} scale={LORA_SCALE} rank={LORA_RANK}")
    log("=" * 72)
    assert ADAPTER_MATH.exists(), f"missing {ADAPTER_MATH}"
    assert ADAPTER_MEDICAL.exists(), f"missing {ADAPTER_MEDICAL}"
    log_mem("start")

    problems = load_gsm8k(N_GSM8K)
    math_ad = mx.load(str(ADAPTER_MATH))
    med_ad = mx.load(str(ADAPTER_MEDICAL))

    model, tokenizer = load(MODEL_ID)
    mode_box = ["base"]
    attach_switch(model, math_ad, med_ad, LORA_SCALE, mode_box)
    gc.collect(); mx.clear_cache()

    details = []
    n_eos_correct = 0
    n_early_correct = 0
    savings_list = []
    early_cross_flags = []   # per correct(to-EOS) case: True if crossover precedes answer

    for pi, p in enumerate(problems):
        prompt = format_chat(tokenizer, gsm8k_prompt(p["question"]))
        g = generate_triple(model, tokenizer, mode_box, prompt)

        pred_eos = extract_pred(g["full_text"])
        pred_early = extract_pred(g["early_text"])
        eos_ok = correct(pred_eos, p["gold"])
        early_ok = correct(pred_early, p["gold"])
        n_eos_correct += int(eos_ok)
        n_early_correct += int(early_ok)

        savings = 1.0 - (g["t_cross"] / g["t_eos"] if g["t_eos"] > 0 else 0.0)
        savings_list.append(savings)

        ans_idx = first_answer_token_index(g["token_strs"], p["gold"])
        crossover_before_answer = g["t_cross"] < ans_idx
        if eos_ok:   # KC-3 measured over math-on-to-EOS correct cases
            early_cross_flags.append(crossover_before_answer)

        details.append({
            "idx": pi, "gold": p["gold"],
            "pred_eos": pred_eos, "pred_early": pred_early,
            "eos_correct": eos_ok, "early_correct": early_ok,
            "t_eos": g["t_eos"], "t_cross": g["t_cross"],
            "savings": round(savings, 4),
            "answer_token_idx": ans_idx,
            "crossover_before_answer": crossover_before_answer,
            "on_delta_mean": round(sum(g["on_deltas"]) / max(1, len(g["on_deltas"])), 4),
            "off_delta_mean": round(sum(g["off_deltas"]) / max(1, len(g["off_deltas"])), 4),
        })
        if (pi + 1) % 5 == 0:
            log(f"  [{pi+1}/{len(problems)}] eos_acc={n_eos_correct/(pi+1):.3f} "
                f"early_acc={n_early_correct/(pi+1):.3f} "
                f"t_eos={g['t_eos']} t_cross={g['t_cross']} sav={savings:.2f}")

    n = len(problems)
    acc_eos = n_eos_correct / n
    acc_early = n_early_correct / n
    delta_acc = acc_early - acc_eos

    sv = sorted(savings_list)
    median_savings = sv[len(sv) // 2] if len(sv) % 2 else (sv[len(sv)//2 - 1] + sv[len(sv)//2]) / 2

    n_correct_cases = len(early_cross_flags)
    early_cross_rate = (sum(early_cross_flags) / n_correct_cases) if n_correct_cases else 0.0

    # ---- Kill criterion K2304 (target, behavioral) ----
    clause_acc = delta_acc < -K_ACC_DROP                  # >2pp drop vs math-on-to-EOS
    clause_savings = median_savings < K_MED_SAVINGS       # median savings < 15%
    clause_cross = early_cross_rate > K_EARLY_CROSS       # crossover precedes answer in >20% of correct cases
    killed = clause_acc or clause_savings or clause_cross
    verdict = "killed" if killed else "supported"
    all_pass = not killed

    log("\n" + "=" * 72)
    log(f"acc_eos(math-on-to-EOS) = {acc_eos:.4f}   acc_early(crossover-stop) = {acc_early:.4f}")
    log(f"delta_acc = {delta_acc:+.4f}  (KILL if < -{K_ACC_DROP})")
    log(f"median_savings = {median_savings:.4f}  (KILL if < {K_MED_SAVINGS})")
    log(f"early_crossover_rate(over {n_correct_cases} correct) = {early_cross_rate:.4f}  (KILL if > {K_EARLY_CROSS})")
    log(f"clauses: acc={clause_acc} savings={clause_savings} cross={clause_cross}")
    log(f"VERDICT: {verdict}")
    log("=" * 72)

    results = {
        "experiment_id": "exp_spark_interference_eos",
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "config": {
            "base_model": MODEL_ID,
            "adapter_math": str(ADAPTER_MATH),
            "adapter_medical": str(ADAPTER_MEDICAL),
            "lora_scale": LORA_SCALE,
            "lora_rank": LORA_RANK,
            "n_gsm8k": n,
            "max_new_tokens": MAX_NEW_TOKENS,
            "baseline": "math-adapter-on greedy-to-EOS (F#866 like-for-like)",
            "k_acc_drop": K_ACC_DROP,
            "k_median_savings": K_MED_SAVINGS,
            "k_early_cross": K_EARLY_CROSS,
            "mlx_lm": "0.31.2",
        },
        "measured": {
            "acc_eos_baseline": acc_eos,
            "acc_early_stop": acc_early,
            "delta_acc": delta_acc,
            "median_token_savings": median_savings,
            "early_crossover_rate": early_cross_rate,
            "n_correct_cases_for_kc3": n_correct_cases,
        },
        "kill": {
            "clause_acc_drop_gt_2pp": clause_acc,
            "clause_median_savings_lt_15pct": clause_savings,
            "clause_early_crossover_gt_20pct": clause_cross,
            "killed": killed,
        },
        "details": details,
        "runtime_sec": round(time.time() - t0, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nWrote {RESULTS_FILE}  ({results['runtime_sec']}s)")
    log_mem("end")


if __name__ == "__main__":
    main()
