#!/usr/bin/env python3
"""exp_spark_phantom_router — does an ADAPTER-FREE self-entropy-argmin control recover F#875's +18.33pp?

F#875 (exp_spark_base_wins_bridges): a per-token entropy-argmin router over {frozen-base, math, code}
scored EM 0.6167 vs base-alone 0.4333 on GSM8K-as-solve() (+18.33pp), base selected 76.6% of tokens.

HYPOTHESIS (frame-break): the lift is the argmin-entropy DECODE rule itself, not adapter knowledge. A
strict ADAPTER-FREE control — at each token, emit the argmax of whichever member of a K=3 self-ensemble
of the SAME frozen base (ZERO adapters) has the lowest next-token Shannon entropy — should recover most
of the gain.

The self-ensemble uses PROMPT-FRAMING diversity (NOT temperature: temperature is argmax-invariant and
entropy-monotone, so an all-base temperature ensemble argmin-entropy is a no-op == base-alone greedy; see
MATH.md). Three semantically-equivalent framings of the SAME problem, same frozen base, shared emitted
token stream, independent KV caches. Member F0 == the exact F#875 solve() prompt (so the arm contains
base-alone greedy as one member); F1/F2 add a neutral preamble / reworded instruction.

ARMS (re-measured IN THIS RUN, same items, per seed in {42,1,2}, n=60):
  (a) base-alone greedy                B
  (b) F#875 3-arm entropy-argmin router R  over {base, math, code}
  (c) adapter-FREE self-entropy-argmin  F  over base-only K=3 prompt-framing ensemble
recovery_frac = (F - B) / (R - B)  using the IN-RUN R-B gap (not F#875's stale 0.4333/0.6167).

KILL 2312 (verbatim, pre-registered): mean(F - B) < +9.0pp  => KILLED (F#875 routing REAL, frame-break FALSE).
                                      mean(F - B) >= +9.0pp => SUPPORTED (F#875 collapses to decode artifact).
Non-degeneracy guard: if adapter-free arm collapses to base-alone (af differs from base on 0 items) =>
provisional (degenerate control), not a clean verdict.

NO MOCKS. Real frozen mlx-community/gemma-4-e4b-it-4bit. is_smoke=False. mlx-lm == 0.31.2.
Harness == F#874/875 no-thinking high-headroom: thinking OFF, weak prompt, MAX_NEW=800, greedy.
"""

import gc
import json
import os
import re
import subprocess
import sys
import tempfile
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
REPO = EXP_DIR.parents[2]
RESULTS_FILE = EXP_DIR / "results.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
ADAPTERS = REPO / "experiments" / "models" / "exp_p1_t2_single_domain_training" / "adapters"
MATH_ADAPTER = ADAPTERS / "math" / "adapters.safetensors"
CODE_ADAPTER = ADAPTERS / "code" / "adapters.safetensors"

LORA_SCALE = 6.0
LORA_RANK = 6
MAX_NEW = 800
SEEDS = [42, 1, 2]
N_TWODOMAIN = 60

KILL_THRESHOLD_PP = 9.0   # pre-registered kill 2312 (half of F#875's +18.33pp)


def log(m):
    print(m, flush=True)


def log_mem(label=""):
    log(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB "
        f"cache={mx.get_cache_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")


# --------------------------------------------------------------------------- #
# LoRA wrapper (router arm only) — verbatim from F#875: out = base(x)+scale*(x@A)@B
# --------------------------------------------------------------------------- #

class LoRAQProj(nn.Module):
    def __init__(self, base, lora_a, lora_b, scale):
        super().__init__()
        self.base = base
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.scale = float(scale)
        self.base.freeze()

    def __call__(self, x):
        y = self.base(x)
        z = (x @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def install_adapter(model, adapter_path):
    weights = mx.load(str(adapter_path))
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        ka = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
        kb = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
        if ka not in weights or kb not in weights:
            continue
        A = weights[ka].astype(mx.float32)
        B = weights[kb].astype(mx.float32)
        wrapper = LoRAQProj(layer.self_attn.q_proj, A, B, LORA_SCALE)
        setattr(layer.self_attn, "q_proj", wrapper)
        count += 1
    mx.eval(model.parameters())
    assert count == 42, f"expected 42 wrapped q_proj, got {count}"
    log(f"  installed adapter {adapter_path.parent.name}: {count} q_proj wrappers @ scale {LORA_SCALE}")
    return count


# --------------------------------------------------------------------------- #
# Prompts (weak, no-thinking, no few-shot). F0 == exact F#875 solve() prompt.
# --------------------------------------------------------------------------- #

def twodomain_prompt(q):
    # F0: identical to F#875 single-BASE / router prompt
    return (
        "Write a Python function `solve()` that computes the answer to this problem and returns "
        "the final number. Put it in a ```python code block.\n\n" + q
    )


def twodomain_prompt_f1(q):
    # F1: same task, neutral preamble. ZERO adapters; only the conditioning context changes.
    return (
        "You are a careful Python programmer. Think about edge cases.\n"
        "Write a Python function `solve()` that computes the answer to this problem and returns "
        "the final number. Put it in a ```python code block.\n\n" + q
    )


def twodomain_prompt_f2(q):
    # F2: same task, reworded equivalent instruction.
    return (
        "Implement a function `solve()` that returns the final number answering the problem below. "
        "Provide it in a ```python code block.\n\n" + q
    )


def build_ids(tokenizer, user):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], add_generation_prompt=True
    )
    return mx.array(ids)


# --------------------------------------------------------------------------- #
# Single-expert greedy decode (base-alone arm)  — verbatim from F#875
# --------------------------------------------------------------------------- #

EOS_IDS = None


def greedy(model, tokenizer, prompt_ids, max_new):
    cache = make_prompt_cache(model)
    logits = model(prompt_ids[None], cache=cache)[:, -1, :]
    y = mx.argmax(logits, axis=-1)
    mx.eval(y)
    out = []
    for _ in range(max_new):
        tid = int(y.item())
        if tid in EOS_IDS:
            break
        out.append(tid)
        logits = model(y[None], cache=cache)[:, -1, :]
        y = mx.argmax(logits, axis=-1)
        mx.eval(y)
    del cache
    return tokenizer.decode(out)


# --------------------------------------------------------------------------- #
# entropy-argmin lockstep decode (generic over K members sharing the emitted
# stream but with independent prompt prefixes / KV caches).
#   - router arm:       K=3 models [base, math, code], one shared prompt prefix
#   - adapter-free arm: K=3 prompt-framings of the SAME base, one model reused
# Implemented to match F#875 router_decode semantics exactly.
# --------------------------------------------------------------------------- #

def entropy(logits):
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    p = mx.exp(lp)
    return float((-(p * lp).sum()).item())


def argmin_entropy_decode(members, tokenizer, max_new):
    """members = list of (model, prompt_ids). All share the emitted-token stream, independent caches.
    Emit argmax of whichever member has the lowest next-token Shannon entropy. Returns
    (text, win_counts)."""
    K = len(members)
    caches = [make_prompt_cache(m) for (m, _) in members]
    last_logits = []
    for (m, pids), c in zip(members, caches):
        lg = m(pids[None], cache=c)[:, -1, :]
        last_logits.append(lg)
    win_counts = [0] * K
    out = []
    for _ in range(max_new):
        ents = [entropy(lg) for lg in last_logits]
        w = int(min(range(K), key=lambda i: ents[i]))
        win_counts[w] += 1
        tok_arr = mx.argmax(last_logits[w], axis=-1)
        mx.eval(tok_arr)
        tid = int(tok_arr.item())
        if tid in EOS_IDS:
            break
        out.append(tid)
        y = mx.array([[tid]])
        last_logits = []
        for (m, _), c in zip(members, caches):
            lg = m(y, cache=c)[:, -1, :]
            last_logits.append(lg)
    for c in caches:
        del c
    return tokenizer.decode(out), win_counts


# --------------------------------------------------------------------------- #
# Data + scoring — verbatim from F#875
# --------------------------------------------------------------------------- #

def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i in range(min(n, len(ds))):
        it = ds[i]
        m = re.search(r"####\s*([\-\d,]+(?:\.\d+)?)", it["answer"])
        ans = float(m.group(1).replace(",", "")) if m else None
        out.append({"question": it["question"], "answer_num": ans})
    log(f"  loaded {len(out)} GSM8K problems")
    return out


def extract_python(text):
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    if blocks:
        for b in blocks:
            if "def solve" in b:
                return b
        return blocks[0]
    idx = text.find("def solve")
    if idx >= 0:
        return text[idx:]
    return None


def score_twodomain(text, gold):
    code = extract_python(text)
    if code is None or "def solve" not in code:
        return False, "no_solve"
    harness = (
        code
        + "\n\nimport json,sys\n"
        + "try:\n    _r = solve()\n    print('RESULT_VALUE='+repr(_r))\n"
        + "except Exception as _e:\n    print('RESULT_ERR='+repr(_e))\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=10)
        out = r.stdout
        m = re.search(r"RESULT_VALUE=(.*)", out)
        if not m:
            return False, "exec_fail"
        try:
            val = float(eval(m.group(1), {"__builtins__": {}}, {}))
        except Exception:
            return False, "nonnumeric"
        ok = gold is not None and abs(val - gold) < 1e-2
        return ok, "ok" if ok else "wrong_value"
    except Exception:
        return False, "timeout"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# One seed: run all three arms on the SAME 60 items.
# --------------------------------------------------------------------------- #

def run_seed(seed, tokenizer, gsm):
    log("\n" + "=" * 70)
    log(f"SEED {seed}")
    log("=" * 70)
    mx.random.seed(seed)

    prompts_f0 = [build_ids(tokenizer, twodomain_prompt(p["question"])) for p in gsm]
    prompts_f1 = [build_ids(tokenizer, twodomain_prompt_f1(p["question"])) for p in gsm]
    prompts_f2 = [build_ids(tokenizer, twodomain_prompt_f2(p["question"])) for p in gsm]

    # ---- arm (a): base-alone greedy ----
    log("  -- arm (a) base-alone greedy --")
    base, _ = load(MODEL_ID)
    base_texts = []
    correct = 0
    for pids, p in zip(prompts_f0, gsm):
        txt = greedy(base, tokenizer, pids, MAX_NEW)
        base_texts.append(txt)
        ok, _ = score_twodomain(txt, p["answer_num"])
        correct += int(ok)
    em_base = correct / len(gsm)
    log(f"     base-alone EM = {em_base:.4f} ({correct}/{len(gsm)})")

    # ---- arm (c): adapter-FREE self-entropy-argmin (K=3 prompt framings, SAME base) ----
    log("  -- arm (c) adapter-FREE self-entropy-argmin (K=3 base prompt-framing ensemble) --")
    af_correct = 0
    af_wins = [0, 0, 0]
    af_differ = 0
    for i, p in enumerate(gsm):
        members = [(base, prompts_f0[i]), (base, prompts_f1[i]), (base, prompts_f2[i])]
        txt, wins = argmin_entropy_decode(members, tokenizer, MAX_NEW)
        ok, _ = score_twodomain(txt, p["answer_num"])
        af_correct += int(ok)
        for k in range(3):
            af_wins[k] += wins[k]
        if txt != base_texts[i]:
            af_differ += 1
        mx.clear_cache()
    em_af = af_correct / len(gsm)
    af_tot = sum(af_wins)
    log(f"     adapter-free EM = {em_af:.4f} ({af_correct}/{len(gsm)})")
    log(f"     af win_counts F0/F1/F2 = {af_wins} (total {af_tot} tok); "
        f"F0_win_frac={af_wins[0]/max(af_tot,1):.4f}")
    log(f"     af differs from base-alone on {af_differ}/{len(gsm)} items "
        f"-> {'NON-degenerate' if af_differ > 0 else 'DEGENERATE (==base-alone)'}")
    del base
    gc.collect(); mx.clear_cache()

    # ---- arm (b): F#875 3-arm router over {base, math, code} ----
    log("  -- arm (b) F#875 3-arm entropy-argmin router {base, math, code} --")
    rbase, _ = load(MODEL_ID)
    rmath, _ = load(MODEL_ID); install_adapter(rmath, MATH_ADAPTER)
    rcode, _ = load(MODEL_ID); install_adapter(rcode, CODE_ADAPTER)
    log_mem("router-loaded")
    r_correct = 0
    r_wins = [0, 0, 0]
    r_differ = 0
    for i, p in enumerate(gsm):
        # all three share ONE prompt prefix (the F0 solve() prompt), as in F#875
        members = [(rbase, prompts_f0[i]), (rmath, prompts_f0[i]), (rcode, prompts_f0[i])]
        txt, wins = argmin_entropy_decode(members, tokenizer, MAX_NEW)
        ok, _ = score_twodomain(txt, p["answer_num"])
        r_correct += int(ok)
        for k in range(3):
            r_wins[k] += wins[k]
        if txt != base_texts[i]:
            r_differ += 1
        mx.clear_cache()
    em_router = r_correct / len(gsm)
    r_tot = sum(r_wins)
    log(f"     router EM = {em_router:.4f} ({r_correct}/{len(gsm)})")
    log(f"     router win_counts base/math/code = {r_wins} (total {r_tot} tok); "
        f"base_win_frac={r_wins[0]/max(r_tot,1):.4f}")
    log(f"     router differs from base-alone on {r_differ}/{len(gsm)} items")
    del rbase, rmath, rcode
    gc.collect(); mx.clear_cache()

    r_minus_b = 100 * (em_router - em_base)
    f_minus_b = 100 * (em_af - em_base)
    recovery = (f_minus_b / r_minus_b) if r_minus_b != 0 else None
    log(f"  >> seed {seed}: B={em_base:.4f} R={em_router:.4f} F={em_af:.4f} | "
        f"R-B={r_minus_b:+.2f}pp F-B={f_minus_b:+.2f}pp recovery_frac="
        f"{('%.3f' % recovery) if recovery is not None else 'undef'}")

    return {
        "seed": seed,
        "em_base_alone": em_base,
        "em_router": em_router,
        "em_adapter_free": em_af,
        "router_minus_base_pp": r_minus_b,
        "adapterfree_minus_base_pp": f_minus_b,
        "recovery_frac": recovery,
        "router_win_counts": {"base": r_wins[0], "math": r_wins[1], "code": r_wins[2]},
        "router_total_tokens": r_tot,
        "router_base_win_fraction": r_wins[0] / max(r_tot, 1),
        "router_items_differ_from_base": r_differ,
        "af_win_counts": {"F0": af_wins[0], "F1": af_wins[1], "F2": af_wins[2]},
        "af_total_tokens": af_tot,
        "af_F0_win_fraction": af_wins[0] / max(af_tot, 1),
        "af_items_differ_from_base": af_differ,
        "af_degenerate": bool(af_differ == 0),
    }


def main():
    global EOS_IDS
    t0 = time.time()
    log("=" * 70)
    log("exp_spark_phantom_router")
    log(f"Base: {MODEL_ID}  (frozen 4-bit)")
    log(f"scale={LORA_SCALE} rank={LORA_RANK} max_new={MAX_NEW} seeds={SEEDS} n={N_TWODOMAIN} thinking=OFF")
    log(f"kill 2312: mean(F-B) < {KILL_THRESHOLD_PP}pp => KILLED (routing REAL)")
    log("=" * 70)
    assert MATH_ADAPTER.exists(), f"missing {MATH_ADAPTER}"
    assert CODE_ADAPTER.exists(), f"missing {CODE_ADAPTER}"

    _m, tokenizer = load(MODEL_ID)
    del _m
    eos = tokenizer.eos_token_id
    eot = tokenizer.encode("<end_of_turn>")
    EOS_IDS = {eos}
    if eot:
        EOS_IDS.add(eot[-1])
    log(f"EOS_IDS = {EOS_IDS}")
    gc.collect(); mx.clear_cache()

    gsm = load_gsm8k(N_TWODOMAIN)  # deterministic first-60; same items every seed

    per_seed = [run_seed(s, tokenizer, gsm) for s in SEEDS]

    # ---- aggregate ----
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    mean_B = mean([r["em_base_alone"] for r in per_seed])
    mean_R = mean([r["em_router"] for r in per_seed])
    mean_F = mean([r["em_adapter_free"] for r in per_seed])
    mean_RmB = mean([r["router_minus_base_pp"] for r in per_seed])
    mean_FmB = mean([r["adapterfree_minus_base_pp"] for r in per_seed])
    mean_recovery = (mean_FmB / mean_RmB) if (mean_RmB not in (None, 0)) else None
    any_degenerate = any(r["af_degenerate"] for r in per_seed)
    router_reproduces = (mean_RmB is not None and mean_RmB > 0)

    # verdict
    if any_degenerate:
        verdict = "provisional"
        kill_reason = ("adapter-free arm degenerate (collapsed to base-alone on some seed): F0 framing "
                       "won every token -> control is a no-op, cannot cleanly kill/support")
    elif not router_reproduces:
        verdict = "provisional"
        kill_reason = ("F#875 router did not reproduce a positive in-run lift over base "
                       f"(mean R-B={mean_RmB:+.2f}pp <= 0); recovery fraction undefined")
    elif mean_FmB < KILL_THRESHOLD_PP:
        verdict = "killed"
        kill_reason = (f"adapter-free control recovered only mean(F-B)={mean_FmB:+.2f}pp "
                       f"(< {KILL_THRESHOLD_PP}pp) -> F#875 routing is REAL, frame-break FALSE")
    else:
        verdict = "supported"
        kill_reason = (f"adapter-free control recovered mean(F-B)={mean_FmB:+.2f}pp "
                       f"(>= {KILL_THRESHOLD_PP}pp) -> F#875 collapses to an argmin-entropy decode artifact")

    all_pass = (verdict == "supported")

    log("\n" + "=" * 70)
    log("AGGREGATE (mean across seeds {42,1,2})")
    log("=" * 70)
    for r in per_seed:
        log(f"  seed {r['seed']}: B={r['em_base_alone']:.4f} R={r['em_router']:.4f} "
            f"F={r['em_adapter_free']:.4f} | R-B={r['router_minus_base_pp']:+.2f}pp "
            f"F-B={r['adapterfree_minus_base_pp']:+.2f}pp "
            f"rec={('%.3f' % r['recovery_frac']) if r['recovery_frac'] is not None else 'undef'} "
            f"af_differ={r['af_items_differ_from_base']}/{N_TWODOMAIN}")
    log(f"  MEAN: B={mean_B:.4f} R={mean_R:.4f} F={mean_F:.4f}")
    log(f"  MEAN R-B = {mean_RmB:+.2f}pp ; MEAN F-B = {mean_FmB:+.2f}pp ; "
        f"recovery_frac(mean) = {('%.3f' % mean_recovery) if mean_recovery is not None else 'undef'}")
    log(f"  KILL 2312 threshold: F-B < {KILL_THRESHOLD_PP}pp ?  -> VERDICT: {verdict}")
    log(f"  reason: {kill_reason}")

    results = {
        "experiment": "exp_spark_phantom_router",
        "model": MODEL_ID,
        "lora_scale": LORA_SCALE, "lora_rank": LORA_RANK,
        "max_new_tokens": MAX_NEW, "seeds": SEEDS, "n_twodomain": N_TWODOMAIN,
        "thinking": False, "is_smoke": False,
        "harness": "F#874/875 no-thinking high-headroom GSM8K-solve(); greedy; executable solve() scorer",
        "adapter_free_ensemble": "K=3 frozen-base prompt-framings (F0=exact F#875 solve() prompt, F1=preamble, F2=reworded); ZERO adapters; argmin-entropy lockstep",
        "kill_threshold_pp": KILL_THRESHOLD_PP,
        "per_seed": per_seed,
        "mean_em_base_alone": mean_B,
        "mean_em_router": mean_R,
        "mean_em_adapter_free": mean_F,
        "mean_router_minus_base_pp": mean_RmB,
        "mean_adapterfree_minus_base_pp": mean_FmB,
        "mean_recovery_frac": mean_recovery,
        "any_seed_adapter_free_degenerate": bool(any_degenerate),
        "router_reproduces_positive_lift": bool(router_reproduces),
        "kill_2312": {
            "threshold_pp": KILL_THRESHOLD_PP,
            "mean_adapterfree_minus_base_pp": mean_FmB,
            "killed": bool(verdict == "killed"),
        },
        "all_pass": all_pass,
        "verdict": verdict,
        "kill_reason": kill_reason,
        "total_time_s": round(time.time() - t0, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults -> {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']}s")
    log(f"FINAL VERDICT: {verdict}")


if __name__ == "__main__":
    main()
