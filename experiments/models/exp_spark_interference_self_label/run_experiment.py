#!/usr/bin/env python3
"""exp_spark_interference_self_label — the SIGN of an adapter's top-1 logit shift is a free domain detector.

Frame-break: off-domain interference is not a bug to suppress but a SELF-LABELING signal. When adapter_i is
applied OFF-domain it lowers the base model's OWN top-1 logit on its OWN greedy pick (s_i<0); ON-domain it
preserves/raises it (s_i>=0). So the SIGN of the mean top-1 logit shift over the first K=8 base-greedy tokens
is a zero-training, zero-calibration, router-free domain detector.

Frozen base mlx-community/gemma-4-e4b-it-4bit. Adapters {code,math,medical}: self_attn.q_proj LoRA rank 6,
scale 6.0 (<=8 guard OK). ~30 held-out valid prompts/domain. Pure forward passes — no training, no merge,
no router.

Per (prompt, adapter_i):
  1. BASE greedy-generates K tokens; at each pos t record base top-1 token y_t and base logit_t(y_t).
  2. For adapter_i, ONE teacher-forced forward over [prompt + first K-1 base tokens] yields logits at the
     same K positions; read logit^{B+A_i}_t(y_t).
  3. s_i = mean_t [ logit^{B+A_i}_t(y_t) - logit^B_t(y_t) ].

AUROC(score=s_i, positive=on-domain) via Mann-Whitney U (exact, no sklearn). Positives = s_i on adapter_i's
own domain prompts; negatives = s_i on the other two domains' prompts.

KILL 2300 (pre-registered): mean AUROC over the 3 adapters < 0.70 => killed.

Composition is the single low-rank delta per projection, never (ΣB)(ΣA). Wrappers attach via subclass
nn.Module + setattr — never __call__ override on instance (F#831). NO MOCKS. is_smoke=False.
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

device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 6 * 1024**3)

EXP_DIR = Path(__file__).resolve().parent
RESULTS_FILE = EXP_DIR / "results.json"
REPO = EXP_DIR.parent.parent.parent  # .../llm

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
ADAPTER_ROOT = REPO / "experiments" / "models" / "exp_p1_t2_single_domain_training" / "adapters"
DATA_ROOT = REPO / "experiments" / "models" / "exp_p1_t2_single_domain_training" / "data"
DOMAINS = ("code", "math", "medical")

MATH_PROJ = "q_proj"
LORA_SCALE = 6.0          # adapters' trained scale (<= 8 guard OK)
K_TOKENS = 8              # first 8 base-greedy tokens
N_PROMPTS = 30            # held-out prompts per domain
SEED = 42

KILL_AUROC = 0.70         # mean AUROC over 3 adapters must be >= 0.70


def log(msg):
    print(msg, flush=True)


def log_mem(label=""):
    try:
        peak = mx.get_peak_memory() / 1024**3
        log(f"  [mem {label}] peak={peak:.2f}GB")
    except Exception:
        pass


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


# ----------------------------------------------------------------------------
# LoRA wrapper (subclass nn.Module + setattr, never __call__ on instance — F#831)
# ----------------------------------------------------------------------------

class LoRAProj(nn.Module):
    """y = linear(x) + scale * (x@a)@b — low-rank additive delta on one projection."""
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


def layer_keys(weights, li, proj):
    ak = f"language_model.model.layers.{li}.self_attn.{proj}.lora_a"
    bk = f"language_model.model.layers.{li}.self_attn.{proj}.lora_b"
    return (ak, bk) if ak in weights and bk in weights else (None, None)


def attach_adapter(model, ad):
    """Wrap q_proj with the low-rank delta on every layer that has it. Returns count."""
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        ak, bk = layer_keys(ad, li, MATH_PROJ)
        if ak is None:
            continue
        a = ad[ak].astype(mx.float32)
        b = ad[bk].astype(mx.float32)
        setattr(layer.self_attn, MATH_PROJ, LoRAProj(layer.self_attn.q_proj, a, b, LORA_SCALE))
        count += 1
    return count


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------

def load_prompts(domain, n):
    path = DATA_ROOT / domain / "valid.jsonl"
    out = []
    with open(path) as f:
        for line in f:
            if len(out) >= n:
                break
            rec = json.loads(line)
            msgs = rec["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            out.append(user)
    return out


def format_chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ----------------------------------------------------------------------------
# Base greedy continuation: returns (prompt_ids list, base_pick_tokens, base_pick_logits)
# ----------------------------------------------------------------------------

def base_continuation(model, tokenizer, prompt_text, k):
    """Greedy-decode k tokens with the (base) model. Record at each step the argmax token y_t and the
    logit assigned to y_t. Returns prompt_ids (list), picks (list[int]), pick_logits (list[float])."""
    from mlx_lm.models.cache import make_prompt_cache
    prompt_ids = tokenizer.encode(prompt_text)
    ids = mx.array(prompt_ids)
    cache = make_prompt_cache(model)
    logits = model(ids[None], cache=cache)[:, -1, :]  # (1, V)
    picks, pick_logits = [], []
    for _ in range(k):
        tok = int(mx.argmax(logits, axis=-1).item())
        lv = float(logits[0, tok].item())
        picks.append(tok)
        pick_logits.append(lv)
        logits = model(mx.array([[tok]]), cache=cache)[:, -1, :]
    del cache
    return prompt_ids, picks, pick_logits


def adapter_pick_logits(model, prompt_ids, picks):
    """ONE teacher-forced forward over [prompt + picks[:-1]]. At position giving the logits BEFORE token
    picks[t], read the logit assigned to picks[t]. Returns list[float] of length len(picks).

    Sequence fed: prompt_ids + picks[:-1].  Output logits L has shape (1, T, V) where T = len(seq).
    The logit predicting picks[0] is at index len(prompt_ids)-1; predicting picks[t] at len(prompt_ids)-1+t.
    """
    k = len(picks)
    seq = list(prompt_ids) + list(picks[:-1])
    ids = mx.array(seq)[None]
    L = model(ids)  # (1, T, V)
    p0 = len(prompt_ids) - 1
    out = []
    for t in range(k):
        out.append(float(L[0, p0 + t, picks[t]].item()))
    return out


# ----------------------------------------------------------------------------
# AUROC via Mann-Whitney U (exact, no sklearn)
# ----------------------------------------------------------------------------

def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                s += 1.0
            elif p == n:
                s += 0.5
    return s / (len(pos) * len(neg))


# ----------------------------------------------------------------------------

def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 72)
    log("exp_spark_interference_self_label")
    log(f"Base: {MODEL_ID}")
    log(f"Adapters: {[str(ADAPTER_ROOT / d / 'adapters.safetensors') for d in DOMAINS]}")
    log(f"K_TOKENS={K_TOKENS} N_PROMPTS={N_PROMPTS} LORA_SCALE={LORA_SCALE}")
    log("=" * 72)

    for d in DOMAINS:
        assert (ADAPTER_ROOT / d / "adapters.safetensors").exists(), f"missing adapter {d}"
        assert (DATA_ROOT / d / "valid.jsonl").exists(), f"missing prompts {d}"

    # ---- load prompts ----
    prompts = {d: load_prompts(d, N_PROMPTS) for d in DOMAINS}
    for d in DOMAINS:
        log(f"  {d}: {len(prompts[d])} prompts")
    log_mem("prompts")

    # ---- Phase 1: base greedy continuations (token + base logit at each of first K positions) ----
    log("\n=== Phase 1: BASE greedy continuations ===")
    base_model, tokenizer = load(MODEL_ID)
    base_data = {}  # (domain, idx) -> {prompt_ids, picks, base_logits}
    for d in DOMAINS:
        for i, ptext in enumerate(prompts[d]):
            chat = format_chat(tokenizer, ptext)
            pid, picks, blog = base_continuation(base_model, tokenizer, chat, K_TOKENS)
            base_data[(d, i)] = {"prompt_ids": pid, "picks": picks, "base_logits": blog}
        log(f"  base continuations done: {d}")
    log_mem("phase1-base")
    del base_model
    gc.collect()
    mx.clear_cache()

    # ---- Phase 2: per adapter, teacher-forced logit at base picks; score s_i ----
    log("\n=== Phase 2: per-adapter logit shifts ===")
    # scores[adapter_domain][(prompt_domain, idx)] = s_i
    scores = {a: {} for a in DOMAINS}
    for a in DOMAINS:
        ad = mx.load(str(ADAPTER_ROOT / a / "adapters.safetensors"))
        model, tok = load(MODEL_ID)
        n = attach_adapter(model, ad)
        mx.eval(model.parameters())
        gc.collect()
        mx.clear_cache()
        log(f"\n  adapter={a} attached q_proj on {n} layers")
        for d in DOMAINS:
            for i in range(len(prompts[d])):
                bd = base_data[(d, i)]
                alog = adapter_pick_logits(model, bd["prompt_ids"], bd["picks"])
                shifts = [alog[t] - bd["base_logits"][t] for t in range(K_TOKENS)]
                s_i = sum(shifts) / len(shifts)
                scores[a][(d, i)] = s_i
        # quick per-domain means for this adapter
        for d in DOMAINS:
            vals = [scores[a][(d, i)] for i in range(len(prompts[d]))]
            tag = "ON " if d == a else "off"
            log(f"    {a} on {d:8s} [{tag}] mean_s={sum(vals)/len(vals):+.4f}")
        log_mem(f"adapter-{a}")
        del model, tok
        gc.collect()
        mx.clear_cache()

    # ---- Phase 3: AUROC per adapter (positives = own domain, negatives = other domains) ----
    log("\n=== Phase 3: AUROC ===")
    per_adapter = {}
    aurocs = []
    sign_correct = 0
    sign_total = 0
    for a in DOMAINS:
        pos = [scores[a][(a, i)] for i in range(len(prompts[a]))]
        neg = []
        for d in DOMAINS:
            if d == a:
                continue
            neg += [scores[a][(d, i)] for i in range(len(prompts[d]))]
        au = auroc(pos, neg)
        aurocs.append(au)
        mean_on = sum(pos) / len(pos)
        mean_off = sum(neg) / len(neg)
        # sign-at-0 accuracy: on-domain predicted positive (s>=0), off-domain negative (s<0)
        for s in pos:
            sign_correct += int(s >= 0)
            sign_total += 1
        for s in neg:
            sign_correct += int(s < 0)
            sign_total += 1
        per_adapter[a] = {
            "auroc": au, "mean_s_on": mean_on, "mean_s_off": mean_off,
            "n_pos": len(pos), "n_neg": len(neg),
        }
        log(f"  {a}: AUROC={au:.4f}  mean_s_on={mean_on:+.4f}  mean_s_off={mean_off:+.4f}")

    mean_auroc = sum(aurocs) / len(aurocs)
    sign_acc = sign_correct / sign_total if sign_total else float("nan")

    killed = mean_auroc < KILL_AUROC
    verdict = "killed" if killed else "supported"
    all_pass = not killed

    results = {
        "experiment_id": "exp_spark_interference_self_label",
        "config": {
            "base_model": MODEL_ID,
            "adapter_root": str(ADAPTER_ROOT),
            "domains": list(DOMAINS),
            "proj": MATH_PROJ,
            "lora_scale": LORA_SCALE,
            "k_tokens": K_TOKENS,
            "n_prompts": N_PROMPTS,
            "kill_auroc": KILL_AUROC,
            "seed": SEED,
        },
        "per_adapter": per_adapter,
        "mean_auroc": mean_auroc,
        "sign_at_zero_accuracy": sign_acc,
        "raw_scores": {a: {f"{d}:{i}": scores[a][(d, i)]
                           for d in DOMAINS for i in range(len(prompts[d]))}
                       for a in DOMAINS},
        "kill_criteria": {
            "2300": {
                "text": "mean AUROC of s_i (on-domain vs off-domain), averaged over 3 adapters < 0.70",
                "type": "target",
                "measured_mean_auroc": mean_auroc,
                "threshold": KILL_AUROC,
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
    log(f"per-adapter AUROC: " + "  ".join(f"{a}={per_adapter[a]['auroc']:.3f}" for a in DOMAINS))
    log(f"MEAN AUROC = {mean_auroc:.4f}  (kill if < {KILL_AUROC})")
    log(f"sign-at-0 accuracy = {sign_acc:.4f}")
    log(f"VERDICT: {verdict}  all_pass={all_pass}")
    log(f"Wrote {RESULTS_FILE}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
