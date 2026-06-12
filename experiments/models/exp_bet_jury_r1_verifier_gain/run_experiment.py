#!/usr/bin/env python3
"""exp_bet_jury_r1_verifier_gain — adapter-as-verifier BoN(8) vs self-consistency(8), GSM8K.

BET jury-decode R1 (ladder: .agents/bets/jury-decode.md). Frozen base gemma-4-e4b-it-4bit +
math LoRA (r=6 q_proj, scale 6) as GENERATOR; the SAME adapter prompted as a judge is the
VERIFIER (logP(Yes)-logP(No) on "is the final answer correct?").

Arms (per question, N>=200 GSM8K test, seed 42, no-thinking harness):
  greedy  : 1 greedy chain (floor reference, 1/8 budget)
  SC(8)   : majority vote over 8 sampled chains (temp 0.8)
  BoN(8)  : SAME 8 chains, argmax verifier score  -> generation budget identical to SC by
            construction; verifier prefill tokens reported separately.
Diagnostic (gates nothing): BoN by mean untempered chain logprob (likelihood ranking).

KILL K2315: verifier AUC <= 0.55 (correct-vs-wrong pooled candidates)  -> killed
KILL K2316: acc(BoN8) <= acc(SC8) at equal generation budget           -> killed
Supported : both clear AND acc(BoN8)-acc(SC8) >= +0.03 (pre-registered in MATH.md)
Else      : provisional (0 < gain < 3pp).

NO MOCKS. Real model, real adapter, real GSM8K. is_smoke=False. mlx-lm 0.31.2.
Wrapper attaches via subclass nn.Module + setattr (never __call__ override — F#831).
"""

import gc
import json
import os
import re
import time
from collections import Counter
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

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
ADAPTER_MATH = EXP_DIR.parent.parent.parent / "data" / "adapters" / "math" / "adapters.safetensors"

LORA_SCALE = 6.0          # <= 8 guard (F#627 recipe)
LORA_RANK = 6
N_LAYERS_EXPECTED = 42
N_GSM8K = int(os.environ.get("N_GSM8K", "200"))
N_SAMPLES = 8
TEMPERATURE = 0.8
MAX_NEW_TOKENS = 512
SEED = 42

# Pre-registered thresholds (MATH.md)
K2315_AUC_MIN = 0.55      # killed if AUC <= this
GATE_GAIN = 0.03          # supported needs BoN - SC >= +3pp


def log(msg):
    print(msg, flush=True)


def log_mem(label=""):
    log(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB "
        f"cache={mx.get_cache_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")


# ----------------------------------------------------------------------------
# Adapter attach: y = base(x) + s * (x @ A) @ B   (single adapter, q_proj only)
# ----------------------------------------------------------------------------

class LoRAQProj(nn.Module):
    def __init__(self, base_linear, a, b, scale):
        super().__init__()
        self.linear = base_linear
        self.lora_a = a
        self.lora_b = b
        self.scale = scale
        self.linear.freeze()

    def __call__(self, x):
        y = self.linear(x)
        delta = self.scale * ((x @ self.lora_a) @ self.lora_b)
        return y + delta.astype(x.dtype)


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def attach_math(model, ad):
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        ak = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
        bk = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
        if ak not in ad or bk not in ad:
            continue
        a = ad[ak].astype(mx.float32)
        b = ad[bk].astype(mx.float32)
        assert a.shape[1] == LORA_RANK, f"rank mismatch: {a.shape}"
        setattr(layer.self_attn, "q_proj", LoRAQProj(layer.self_attn.q_proj, a, b, LORA_SCALE))
        count += 1
    mx.eval(model.parameters())
    assert count == N_LAYERS_EXPECTED, f"expected {N_LAYERS_EXPECTED} wrapped q_proj, got {count}"
    log(f"  Attached math LoRA to {count} q_proj layers (scale={LORA_SCALE}, rank={LORA_RANK})")


# ----------------------------------------------------------------------------
# Generation (no-thinking harness; greedy or temperature sampling)
# Returns (text, n_generated_tokens, mean_untempered_logprob_of_chain)
# ----------------------------------------------------------------------------

def format_chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(model, tokenizer, prompt_ids, max_new, temperature, eos_ids):
    cache = make_prompt_cache(model)
    logits = model(prompt_ids[None], cache=cache)[:, -1, :]
    out, lp_sum = [], 0.0
    for _ in range(max_new):
        logits32 = logits.astype(mx.float32)
        if temperature <= 0.0:
            tok = mx.argmax(logits32, axis=-1)
        else:
            tok = mx.random.categorical(logits32 / temperature)
        logprobs = logits32 - mx.logsumexp(logits32, axis=-1, keepdims=True)
        lp = logprobs[0, tok[0]]
        mx.eval(tok, lp)
        tid = tok.item()
        out.append(tid)
        lp_sum += lp.item()           # untempered logprob (likelihood diagnostic)
        if tid in eos_ids:
            break
        logits = model(mx.array([[tid]]), cache=cache)[:, -1, :]
    del cache
    mean_lp = lp_sum / max(len(out), 1)
    return tokenizer.decode(out), len(out), mean_lp


# ----------------------------------------------------------------------------
# Verifier: single forward pass, score = logP(Yes) - logP(No) at last position
# ----------------------------------------------------------------------------

VERIFIER_TEMPLATE = (
    "Question:\n{q}\n\nProposed solution:\n{sol}\n\n"
    "Is the final answer of this proposed solution correct? "
    "Reply with exactly one word: Yes or No."
)


def verifier_score(model, tokenizer, question, solution, yes_id, no_id):
    prompt = format_chat(tokenizer, VERIFIER_TEMPLATE.format(q=question, sol=solution))
    ids = mx.array(tokenizer.encode(prompt))
    logits = model(ids[None])[:, -1, :].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    s = logprobs[0, yes_id] - logprobs[0, no_id]
    mx.eval(s)
    return float(s.item()), ids.shape[0]


# ----------------------------------------------------------------------------
# GSM8K
# ----------------------------------------------------------------------------

def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))
    items = [{"question": ds[i]["question"], "answer": ds[i]["answer"]} for i in range(len(ds))]
    log(f"  Loaded {len(items)} GSM8K problems")
    return items


def gsm8k_gt(answer):
    m = re.search(r"####\s*([\-\d,\.]+)", answer)
    return m.group(1).replace(",", "").strip() if m else None


def gsm8k_pred(text):
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


GSM8K_INSTR = "Solve this math problem step by step. End your answer with '#### <number>'.\n\n"


# ----------------------------------------------------------------------------
# AUC (rank-based Mann-Whitney, ties get average rank)
# ----------------------------------------------------------------------------

def rank_auc(pos_scores, neg_scores):
    if not pos_scores or not neg_scores:
        return float("nan")
    allv = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    allv.sort(key=lambda t: t[0])
    ranks, i = {}, 0
    n = len(allv)
    rank_sum_pos = 0.0
    while i < n:
        j = i
        while j < n and allv[j][0] == allv[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if allv[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 72)
    log("exp_bet_jury_r1_verifier_gain — adapter-as-verifier BoN(8) vs SC(8), GSM8K")
    log(f"Base: {MODEL_ID}  adapter: {ADAPTER_MATH}")
    log(f"N={N_GSM8K} samples={N_SAMPLES} temp={TEMPERATURE} max_new={MAX_NEW_TOKENS}")
    log("=" * 72)
    assert ADAPTER_MATH.exists(), f"missing {ADAPTER_MATH}"
    log_mem("start")

    model, tokenizer = load(MODEL_ID)
    math_ad = mx.load(str(ADAPTER_MATH))
    attach_math(model, math_ad)
    log_mem("model+adapter")

    eos = tokenizer.eos_token_id
    eot_enc = tokenizer.encode("<end_of_turn>")
    eos_ids = {eos, eot_enc[-1] if eot_enc else eos}
    yes_id = tokenizer.encode("Yes")[-1]
    no_id = tokenizer.encode("No")[-1]
    assert yes_id != no_id
    log(f"  yes_id={yes_id} no_id={no_id} eos_ids={eos_ids}")

    items = load_gsm8k(N_GSM8K)

    n_greedy_ok = n_sc_ok = n_bon_ok = n_bonlik_ok = n_pass8 = 0
    tokens_greedy = tokens_candidates = verifier_prefill_tokens = 0
    auc_pos, auc_neg = [], []          # verifier scores, pooled candidates
    lik_pos, lik_neg = [], []          # likelihood diagnostic
    per_q = []

    for qi, it in enumerate(items):
        gt = gsm8k_gt(it["answer"])
        prompt = format_chat(tokenizer, GSM8K_INSTR + it["question"])
        prompt_ids = mx.array(tokenizer.encode(prompt))

        # --- greedy floor (1/8 budget reference) ---
        mx.random.seed(SEED * 100003 + qi)
        g_text, g_ntok, _ = generate(model, tokenizer, prompt_ids, MAX_NEW_TOKENS, 0.0, eos_ids)
        tokens_greedy += g_ntok
        g_pred = gsm8k_pred(g_text)
        g_ok = (gt is not None and g_pred == gt)
        n_greedy_ok += int(g_ok)

        # --- 8 sampled candidates (shared by SC and BoN: equal generation budget) ---
        cands = []
        for si in range(N_SAMPLES):
            mx.random.seed(SEED * 100003 + qi * 1009 + si + 1)
            text, ntok, mean_lp = generate(model, tokenizer, prompt_ids,
                                           MAX_NEW_TOKENS, TEMPERATURE, eos_ids)
            tokens_candidates += ntok
            pred = gsm8k_pred(text)
            ok = (gt is not None and pred is not None and pred == gt)
            vs, vtok = verifier_score(model, tokenizer, it["question"], text.strip(),
                                      yes_id, no_id)
            verifier_prefill_tokens += vtok
            cands.append({"pred": pred, "ok": ok, "vscore": vs, "lik": mean_lp, "ntok": ntok})
            if pred is not None and gt is not None:
                (auc_pos if ok else auc_neg).append(vs)
                (lik_pos if ok else lik_neg).append(mean_lp)

        n_pass8 += int(any(c["ok"] for c in cands))

        # --- SC(8): majority vote (tie -> first sampled, Counter insertion order) ---
        votes = Counter(c["pred"] for c in cands if c["pred"] is not None)
        sc_pred = votes.most_common(1)[0][0] if votes else None
        sc_ok = (gt is not None and sc_pred == gt)
        n_sc_ok += int(sc_ok)

        # --- BoN(8) primary: argmax verifier score among parseable candidates ---
        parseable = [c for c in cands if c["pred"] is not None] or cands
        bon = max(parseable, key=lambda c: c["vscore"])
        bon_ok = bon["ok"]
        n_bon_ok += int(bon_ok)

        # --- BoN diagnostic: argmax mean chain logprob ---
        bonlik = max(parseable, key=lambda c: c["lik"])
        n_bonlik_ok += int(bonlik["ok"])

        per_q.append({"gt": gt, "greedy": {"pred": g_pred, "ok": g_ok},
                      "sc_pred": sc_pred, "sc_ok": sc_ok,
                      "bon_pred": bon["pred"], "bon_ok": bon_ok,
                      "cands": [{k: c[k] for k in ("pred", "ok", "vscore", "lik", "ntok")}
                                for c in cands]})

        if (qi + 1) % 10 == 0:
            n = qi + 1
            log(f"  [{n}/{len(items)}] greedy={n_greedy_ok/n:.3f} sc8={n_sc_ok/n:.3f} "
                f"bon8={n_bon_ok/n:.3f} bonlik8={n_bonlik_ok/n:.3f} pass@8={n_pass8/n:.3f} "
                f"({time.time()-t0:.0f}s)")
            mx.clear_cache()

    n = len(items)
    acc_greedy = n_greedy_ok / n
    acc_sc = n_sc_ok / n
    acc_bon = n_bon_ok / n
    acc_bonlik = n_bonlik_ok / n
    pass_at_8 = n_pass8 / n
    gain = acc_bon - acc_sc
    auc_verifier = rank_auc(auc_pos, auc_neg)
    auc_lik = rank_auc(lik_pos, lik_neg)

    k2315_kill = not (auc_verifier > K2315_AUC_MIN)   # AUC <= 0.55 (or nan) -> kill
    k2316_kill = acc_bon <= acc_sc
    if k2315_kill or k2316_kill:
        verdict, all_pass = "killed", False
    elif gain >= GATE_GAIN:
        verdict, all_pass = "supported", True
    else:
        verdict, all_pass = "provisional", False      # 0 < gain < 3pp (pre-registered)

    results = {
        "experiment_id": "exp_bet_jury_r1_verifier_gain",
        "config": {
            "base_model": MODEL_ID,
            "adapter_math": str(ADAPTER_MATH),
            "lora_scale": LORA_SCALE, "lora_rank": LORA_RANK,
            "n_gsm8k": n, "n_samples": N_SAMPLES, "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS, "seed": SEED,
            "no_thinking_harness": True,
            "verifier": "same math adapter, logP(Yes)-logP(No) judge probe",
            "sampling": "temperature only (no top-p), untempered logprob recorded",
            "mlx_lm": "0.31.2",
        },
        "accuracy": {
            "greedy": acc_greedy,
            "self_consistency_8": acc_sc,
            "bon_8_verifier": acc_bon,
            "bon_8_likelihood_diagnostic": acc_bonlik,
            "pass_at_8_ceiling": pass_at_8,
        },
        "gain_bon_minus_sc": gain,
        "verifier_auc": auc_verifier,
        "likelihood_auc_diagnostic": auc_lik,
        "n_auc_pos": len(auc_pos), "n_auc_neg": len(auc_neg),
        "token_budget": {
            "greedy_generated": tokens_greedy,
            "candidates_generated_shared_sc_and_bon": tokens_candidates,
            "verifier_prefill_tokens": verifier_prefill_tokens,
            "note": "SC(8) and BoN(8) share the identical 8 chains -> equal generation budget "
                    "by construction; verifier adds prefill-only cost, zero generated tokens.",
        },
        "kill_criteria": {
            "2315": {"text": "verifier AUC <= 0.55 (no better than random)",
                     "auc": auc_verifier, "threshold": K2315_AUC_MIN,
                     "result": "fail" if k2315_kill else "pass"},
            "2316": {"text": "verifier-BoN <= self-consistency at equal token budget",
                     "acc_bon": acc_bon, "acc_sc": acc_sc,
                     "result": "fail" if k2316_kill else "pass"},
        },
        "gate_gain_required": GATE_GAIN,
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "details": per_q,
        "total_wall_clock_sec": time.time() - t0,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    log("\n" + "=" * 72)
    log(f"greedy={acc_greedy:.3f}  SC(8)={acc_sc:.3f}  BoN(8)={acc_bon:.3f}  "
        f"BoN-lik(8)={acc_bonlik:.3f}  pass@8={pass_at_8:.3f}")
    log(f"gain BoN-SC = {gain:+.3f} (gate >= +{GATE_GAIN})  "
        f"verifier AUC = {auc_verifier:.3f} (kill <= {K2315_AUC_MIN})  lik AUC = {auc_lik:.3f}")
    log(f"K2315: {'KILL' if k2315_kill else 'pass'}  K2316: {'KILL' if k2316_kill else 'pass'}")
    log(f"VERDICT: {verdict}  all_pass={all_pass}")
    log(f"Wrote {RESULTS_FILE}  ({time.time()-t0:.0f}s)")

    del model
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
