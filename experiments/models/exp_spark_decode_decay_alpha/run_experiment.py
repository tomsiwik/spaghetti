#!/usr/bin/env python3
"""exp_spark_decode_decay_alpha — off-domain LoRA interference as a decode-time long tail.

Frozen base gemma-4-e4b-it-4bit + ONE trained domain adapter (GSM8K math, r=6, q_proj only,
42 layers, F#627). Runtime scale-space per-decode-step gate alpha(t); NO retraining, NO
checkpoints. The composed q_proj at decode step t:

    y_t = W h_t + alpha(t) * s * (h_t @ Am) @ Bm        (Sigma Bi Ai form; single adapter)

with base LoRA scale s = LORA_SCALE = 6.0 <= 8. alpha(t) is a fixed schedule of the position
counter ONLY (content-independent, never learned).

Three conditions x two domains (on=GSM8K numeric, off=ARC-Easy letter), exact match, n=50 each:
  OFF   alpha(t) == 0            (base; adapter never acts)
  ON    alpha(t) == 1            (always-on; the standard interference baseline)
  DECAY alpha(t)=1 for t<8, then linear ->0 over 24 steps, then 0.

Metrics (fractions):
  on_lift_always = acc_on(ON)-acc_on(OFF); on_lift_decay = acc_on(DECAY)-acc_on(OFF)
  off_deg_always = acc_off(OFF)-acc_off(ON); off_deg_decay = acc_off(OFF)-acc_off(DECAY)
  lift_retention = on_lift_decay/on_lift_always       (want > 0.70)
  degradation_recovery = off_deg_decay/off_deg_always (want < 0.50)

KILL K2302: KILL if degradation_recovery >= 0.50 OR lift_retention <= 0.70 (or premise dead:
off_deg_always<=0 or on_lift_always<=0 -> KILLED, recovery/retention = NaN treated as fail).

Wrapper: subclass nn.Module + setattr, never __call__ override on instance (F#831). The
decode-step counter lives in a shared Schedule object incremented once per generated token.
Composition is Sigma Bi Ai (single delta here), never (Sigma B)(Sigma A). LORA_SCALE=6.0 <= 8.

NO MOCKS. Real model, real adapter, real GSM8K + ARC scoring. is_smoke=False. mlx-lm 0.31.2.
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

LORA_SCALE = 6.0          # <= 8 guard OK (F#627 recipe)
LORA_RANK = 6
N_PER_DOMAIN = 50
MAX_NEW_TOKENS = 1024     # thinking-mode headroom
SEED = 42
N_LAYERS_EXPECTED = 42

# DECAY schedule parameters (decode-step gate; fixed, content-independent)
DECAY_K = 8               # full authority for t < K
DECAY_W = 24              # linear ramp width K..K+W -> 0

# Kill thresholds (pre-registered, MATH.md sec 4)
K_RECOVERY_MAX = 0.50     # degradation_recovery must be < 0.50
K_RETENTION_MIN = 0.70    # lift_retention must be > 0.70


def log(msg):
    print(msg, flush=True)


def log_mem(label=""):
    log(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB "
        f"cache={mx.get_cache_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")


# ----------------------------------------------------------------------------
# Decode-step schedule. A single shared mutable object holds the current step.
# alpha(step) depends ONLY on the step counter -> content-independent, never learned.
# ----------------------------------------------------------------------------

class Schedule:
    MODE_OFF = "off"
    MODE_ON = "on"
    MODE_DECAY = "decay"

    def __init__(self, mode):
        self.mode = mode
        self.step = 0   # current decode-token index; 0 = first generated token / prompt pass

    def reset(self):
        self.step = 0

    def advance(self):
        self.step += 1

    def alpha(self):
        if self.mode == self.MODE_OFF:
            return 0.0
        if self.mode == self.MODE_ON:
            return 1.0
        # DECAY: 1 for t<K, then linear ->0 over W, then 0
        t = self.step
        if t < DECAY_K:
            return 1.0
        a = 1.0 - (t - DECAY_K) / float(DECAY_W)
        if a < 0.0:
            return 0.0
        return a


# ----------------------------------------------------------------------------
# Composed q_proj wrapper: base + alpha(t)*s*(x@Am)@Bm
# subclass nn.Module + setattr (NEVER __call__ override on instance — F#831)
# ----------------------------------------------------------------------------

class GatedQProj(nn.Module):
    """y = linear(x) + alpha(step) * scale * (x @ Am) @ Bm.

    The gate alpha is read from a SHARED Schedule object at call time. During prefill the
    wrapper sees the full prompt (seq>1) and uses alpha at step 0 (== ON for prefill of all
    schedules except OFF, which is exactly correct: the scaffold should be present for the
    prompt). During decode each call is seq==1 and reads the current decayed alpha.
    """
    def __init__(self, base_linear, am, bm, scale, schedule):
        super().__init__()
        self.linear = base_linear            # frozen QuantizedLinear
        self.am, self.bm = am, bm            # math: (in,r),(r,out)
        self.scale = scale
        self.schedule = schedule
        self.linear.freeze()

    def __call__(self, x):
        y = self.linear(x)
        a = self.schedule.alpha()
        if a != 0.0:
            dm = (x @ self.am) @ self.bm
            y = y + (a * self.scale * dm).astype(x.dtype)
        return y


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def attach_gated(model, math_ad, scale, schedule):
    """Wrap q_proj on every layer with GatedQProj. Returns model. Asserts 42 wrapped."""
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        ak = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
        bk = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
        if ak not in math_ad or bk not in math_ad:
            continue
        base_linear = layer.self_attn.q_proj
        am = math_ad[ak].astype(mx.float32)
        bm = math_ad[bk].astype(mx.float32)
        wrapper = GatedQProj(base_linear, am, bm, scale, schedule)
        setattr(layer.self_attn, "q_proj", wrapper)   # canonical: setattr
        count += 1
    mx.eval(model.parameters())
    log(f"  Attached {count} GatedQProj (mode={schedule.mode})")
    assert count == N_LAYERS_EXPECTED, f"expected {N_LAYERS_EXPECTED} wrapped, got {count}"
    return model


# ----------------------------------------------------------------------------
# Generation (greedy). The schedule step advances once per generated token so the
# per-decode-step alpha is realized exactly. Prefill is step 0.
# ----------------------------------------------------------------------------

def format_chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def generate(model, tokenizer, prompt, schedule, max_new=MAX_NEW_TOKENS):
    schedule.reset()                       # step 0: prefill (scaffold present for ON/DECAY)
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
        schedule.advance()                 # next decode token -> next alpha(t)
        logits = model(mx.array([[out[-1]]]), cache=cache)[:, -1, :]
        tok = mx.argmax(logits, axis=-1)
        mx.eval(tok)
        out.append(tok.item())
    del cache
    return tokenizer.decode(out), len(out)


def strip_thinking(text):
    if not text:
        return text
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


# ----------------------------------------------------------------------------
# On-domain: GSM8K (numeric exact match)
# ----------------------------------------------------------------------------

def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    probs = []
    for i in range(min(n, len(ds))):
        it = ds[i]
        m = re.search(r"####\s*([\-\d,]+(?:\.\d+)?)", it["answer"])
        gt = float(m.group(1).replace(",", "")) if m else None
        probs.append({"question": it["question"], "gt": gt})
    log(f"  Loaded {len(probs)} GSM8K problems")
    return probs


def gsm8k_prompt(p):
    return (
        "Solve this math problem. Show your reasoning, then give the final numeric "
        "answer on its own line in the exact form '#### <number>'.\n\n" + p["question"]
    )


def extract_gsm8k_answer(text):
    text = strip_thinking(text)
    m = re.findall(r"####\s*\$?([\-\d,]+(?:\.\d+)?)", text)
    if m:
        try:
            return float(m[-1].replace(",", ""))
        except ValueError:
            pass
    m = re.search(r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*\$?([\-\d,]+(?:\.\d+)?)",
                  text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    nums = re.findall(r"[\-]?[\d,]+(?:\.\d+)?", text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def gsm8k_correct(gen, gt, eps=1e-3):
    if gen is None or gt is None:
        return False
    if gt == 0:
        return abs(gen) < eps
    return abs(gen - gt) / abs(gt) < eps


def eval_gsm8k(model, tokenizer, problems, schedule):
    passed, details = 0, []
    for p in problems:
        prompt = format_chat(tokenizer, gsm8k_prompt(p))
        text, ntok = generate(model, tokenizer, prompt, schedule)
        ga = extract_gsm8k_answer(text)
        ok = gsm8k_correct(ga, p["gt"])
        passed += int(ok)
        details.append({"gt": p["gt"], "gen": ga, "ok": ok, "ntok": ntok})
    acc = passed / len(problems)
    log(f"    GSM8K acc = {acc:.4f} ({passed}/{len(problems)})")
    return acc, details


# ----------------------------------------------------------------------------
# Off-domain: ARC-Easy (single-letter exact match; stylistic-drift victim)
# ----------------------------------------------------------------------------

def load_arc(n):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    probs = []
    for i in range(len(ds)):
        if len(probs) >= n:
            break
        it = ds[i]
        labels = it["choices"]["label"]
        texts = it["choices"]["text"]
        key = it["answerKey"]
        # keep only clean A-D 4-choice items with letter keys for unambiguous exact match
        if key not in ("A", "B", "C", "D"):
            continue
        if list(labels) != ["A", "B", "C", "D"]:
            continue
        probs.append({"question": it["question"], "labels": labels,
                      "texts": texts, "key": key})
    log(f"  Loaded {len(probs)} ARC-Easy problems")
    return probs


def arc_prompt(p):
    opts = "\n".join(f"{l}. {t}" for l, t in zip(p["labels"], p["texts"]))
    return (
        "Answer this multiple-choice science question. Think briefly, then end with the "
        "final answer on its own line in the exact form 'Answer: <letter>'.\n\n"
        + p["question"] + "\n" + opts
    )


def extract_arc_answer(text):
    text = strip_thinking(text)
    m = re.findall(r"answer\s*(?:is|:)?\s*\(?\s*([ABCD])\b", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    # fallback: last standalone option letter
    m = re.findall(r"\b([ABCD])\b", text)
    if m:
        return m[-1].upper()
    return None


def eval_arc(model, tokenizer, problems, schedule):
    passed, details = 0, []
    for p in problems:
        prompt = format_chat(tokenizer, arc_prompt(p))
        text, ntok = generate(model, tokenizer, prompt, schedule)
        ga = extract_arc_answer(text)
        ok = (ga == p["key"])
        passed += int(ok)
        details.append({"key": p["key"], "gen": ga, "ok": ok, "ntok": ntok})
    acc = passed / len(problems)
    log(f"    ARC-Easy acc = {acc:.4f} ({passed}/{len(problems)})")
    return acc, details


# ----------------------------------------------------------------------------
# One condition = (mode) evaluated on both domains with a fresh model load.
# ----------------------------------------------------------------------------

def run_condition(mode, math_ad, gsm, arc):
    log(f"\n=== CONDITION mode={mode} ===")
    schedule = Schedule(mode)
    model, tok = load(MODEL_ID)
    if mode != Schedule.MODE_OFF:
        attach_gated(model, math_ad, LORA_SCALE, schedule)
    else:
        # OFF still wraps so all conditions share identical code path; alpha==0 -> no delta
        attach_gated(model, math_ad, LORA_SCALE, schedule)
    gc.collect(); mx.clear_cache()
    on_acc, on_det = eval_gsm8k(model, tok, gsm, schedule)
    off_acc, off_det = eval_arc(model, tok, arc, schedule)
    log_mem(f"{mode}-done")
    del model, tok
    gc.collect(); mx.clear_cache()
    return {"on_acc": on_acc, "off_acc": off_acc,
            "on_details": on_det, "off_details": off_det}


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 72)
    log("exp_spark_decode_decay_alpha")
    log(f"Base: {MODEL_ID}")
    log(f"Math adapter: {ADAPTER_MATH}")
    log(f"n_per_domain={N_PER_DOMAIN} scale={LORA_SCALE} rank={LORA_RANK} "
        f"decay_k={DECAY_K} decay_w={DECAY_W}")
    log("=" * 72)
    assert ADAPTER_MATH.exists(), f"missing {ADAPTER_MATH}"
    log_mem("start")

    log("\n=== Load data + adapter ===")
    gsm = load_gsm8k(N_PER_DOMAIN)
    arc = load_arc(N_PER_DOMAIN)
    math_ad = mx.load(str(ADAPTER_MATH))

    OFF = run_condition(Schedule.MODE_OFF, math_ad, gsm, arc)
    ON = run_condition(Schedule.MODE_ON, math_ad, gsm, arc)
    DECAY = run_condition(Schedule.MODE_DECAY, math_ad, gsm, arc)

    on_off, off_off = OFF["on_acc"], OFF["off_acc"]
    on_on, off_on = ON["on_acc"], ON["off_acc"]
    on_dec, off_dec = DECAY["on_acc"], DECAY["off_acc"]

    on_lift_always = on_on - on_off
    on_lift_decay = on_dec - on_off
    off_deg_always = off_off - off_on
    off_deg_decay = off_off - off_dec

    premise_off_ok = off_deg_always > 1e-9    # always-on actually hurts off-domain
    premise_on_ok = on_lift_always > 1e-9     # adapter actually lifts on-domain

    lift_retention = (on_lift_decay / on_lift_always) if premise_on_ok else float("nan")
    degradation_recovery = (off_deg_decay / off_deg_always) if premise_off_ok else float("nan")

    # ---- Kill K2302 (pre-registered, MATH.md sec 4) ----
    # Dead premise -> KILLED. Otherwise both inequalities must hold for support.
    if not (premise_off_ok and premise_on_ok):
        killed = True
        kill_reason = ("dead premise: "
                       + ("off_deg_always<=0 " if not premise_off_ok else "")
                       + ("on_lift_always<=0" if not premise_on_ok else "")).strip()
    else:
        recovery_fail = degradation_recovery >= K_RECOVERY_MAX
        retention_fail = lift_retention <= K_RETENTION_MIN
        killed = recovery_fail or retention_fail
        kill_reason = ""
        if recovery_fail:
            kill_reason += f"degradation_recovery {degradation_recovery:.3f} >= {K_RECOVERY_MAX} "
        if retention_fail:
            kill_reason += f"lift_retention {lift_retention:.3f} <= {K_RETENTION_MIN}"
        kill_reason = kill_reason.strip()

    verdict = "killed" if killed else "supported"
    all_pass = not killed

    def f(x):
        return None if (isinstance(x, float) and x != x) else x

    results = {
        "experiment_id": "exp_spark_decode_decay_alpha",
        "is_smoke": False,
        "verdict": verdict,
        "all_pass": all_pass,
        "kill_reason": kill_reason,
        "config": {
            "base_model": MODEL_ID,
            "adapter_math": str(ADAPTER_MATH),
            "lora_scale": LORA_SCALE,
            "lora_rank": LORA_RANK,
            "n_per_domain": N_PER_DOMAIN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "decay_k": DECAY_K,
            "decay_w": DECAY_W,
            "k_recovery_max": K_RECOVERY_MAX,
            "k_retention_min": K_RETENTION_MIN,
            "on_domain": "gsm8k_test_numeric",
            "off_domain": "arc_easy_test_letter",
            "mlx_lm": "0.31.2",
            "seed": SEED,
        },
        "measured": {
            "acc_on_OFF": on_off, "acc_on_ON": on_on, "acc_on_DECAY": on_dec,
            "acc_off_OFF": off_off, "acc_off_ON": off_on, "acc_off_DECAY": off_dec,
            "on_lift_always": on_lift_always, "on_lift_decay": on_lift_decay,
            "off_deg_always": off_deg_always, "off_deg_decay": off_deg_decay,
            "lift_retention": f(lift_retention),
            "degradation_recovery": f(degradation_recovery),
            "premise_off_ok": premise_off_ok, "premise_on_ok": premise_on_ok,
        },
        "conditions": {
            "OFF": {"on_acc": on_off, "off_acc": off_off,
                    "on_details": OFF["on_details"], "off_details": OFF["off_details"]},
            "ON": {"on_acc": on_on, "off_acc": off_on,
                   "on_details": ON["on_details"], "off_details": ON["off_details"]},
            "DECAY": {"on_acc": on_dec, "off_acc": off_dec,
                      "on_details": DECAY["on_details"], "off_details": DECAY["off_details"]},
        },
        "runtime_sec": round(time.time() - t0, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log("\n" + "=" * 72)
    log(f"acc_on:  OFF={on_off:.3f} ON={on_on:.3f} DECAY={on_dec:.3f}")
    log(f"acc_off: OFF={off_off:.3f} ON={off_on:.3f} DECAY={off_dec:.3f}")
    log(f"on_lift_always={on_lift_always:+.3f} on_lift_decay={on_lift_decay:+.3f} "
        f"-> lift_retention={lift_retention:.3f} (want > {K_RETENTION_MIN})")
    log(f"off_deg_always={off_deg_always:+.3f} off_deg_decay={off_deg_decay:+.3f} "
        f"-> degradation_recovery={degradation_recovery:.3f} (want < {K_RECOVERY_MAX})")
    log(f"VERDICT: {verdict.upper()}  all_pass={all_pass}  {kill_reason}")
    log("=" * 72)


if __name__ == "__main__":
    main()
