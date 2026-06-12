#!/usr/bin/env python3
"""exp_spark_base_wins_bridges — frozen base as a first-class 3rd router option that WINS bridge tokens.

Composition on the DECODE-STEP axis (not parameter-merge). At each greedy step we run THREE independent
forward passes that share the emitted-token stream but have their own KV caches:
  - base : frozen mlx-community/gemma-4-e4b-it-4bit, no adapter
  - math : base + q_proj r6 scale6 math adapter   (Σ_layers scale*(x@A)@B ; never (ΣB)(ΣA))
  - code : base + q_proj r6 scale6 code adapter
The router emits the argmax token from whichever expert has the LOWEST next-token Shannon entropy
(argmin-entropy = most-confident expert). The frozen base is a first-class emitter, not a gate.

Genuine two-domain task: GSM8K word problems answered by WRITING a Python solve() function we EXECUTE.
  - code domain load-bearing: must be runnable Python (def/return = bridge tokens) or execution fails.
  - math domain load-bearing: arithmetic inside must be correct or returned number is wrong.

Harness = F#874 no-thinking high-headroom: thinking OFF, weak prompt, ~800 new tokens, greedy, SEED=42.

PRE-GATE (mandatory, reported FIRST): math adapter net-positive on GSM8K-prose AND code adapter
net-positive on code-completion, both in THIS harness. If either not net-positive => KILLED (pre-gate).

best_single is now max(base_solve, math_solve, code_solve) — base-alone is a FIRST-CLASS arm on the
EXACT solve() task/scorer (REVIEW fix 1). If the router cannot beat base-alone by >=+3.0pp the adapters
add nothing and the verdict is KILLED.

Kill 2311 (all three clauses; SUPPORTED only if all pass; thresholds UNCHANGED/verbatim):
  C1 router_EM - best_single_EM >= +0.030   (best_single now includes base-alone on the solve() task)
  C2 base_win_fraction >= 0.15
  C3 math_adapter_EM_math > base_EM_math AND code_adapter_EM_code > base_EM_code  (pre-gate)

NO MOCKS. Real model, real adapters, real execution. is_smoke=False. mlx-lm == 0.31.2.
"""

import gc
import json
import math
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
CODE_DATA = REPO / "experiments" / "models" / "exp_p1_t2_single_domain_training" / "data" / "code" / "valid.jsonl"

LORA_SCALE = 6.0          # matches recipe; <= 8 guard. Same for both adapters (F#863 comparable magnitude).
LORA_RANK = 6
MAX_NEW = 800             # F#874 ~800 tok headroom, no thinking truncation
SEED = 42

N_PREGATE_MATH = 40       # GSM8K-prose pre-gate slice
N_PREGATE_CODE = 40       # code-completion pre-gate slice
N_TWODOMAIN = 60          # math∧code task

# bridge / transition tokens we expect the frozen base to win
BRIDGE_STRS = {"def", "return", "=", ":", "\n", "(", ")", ",", "#", "####", " ", "://"}


def log(m):
    print(m, flush=True)


def log_mem(label=""):
    log(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB "
        f"cache={mx.get_cache_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")


# ---------------------------------------------------------------------------
# LoRA wrapper: out = base(x) + scale*(x@A)@B  (per-adapter, never (ΣB)(ΣA))
# Installed via setattr on the parent module (never override __call__ on instance).
# ---------------------------------------------------------------------------

class LoRAQProj(nn.Module):
    def __init__(self, base, lora_a, lora_b, scale):
        super().__init__()
        self.base = base                 # frozen QuantizedLinear
        self.lora_a = lora_a             # (d_in, r)
        self.lora_b = lora_b             # (r, d_out)
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


# ---------------------------------------------------------------------------
# Prompts (weak, no-thinking, no few-shot)
# ---------------------------------------------------------------------------

def build_ids(tokenizer, user):
    # plain chat template => thinking OFF for gemma-4
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], add_generation_prompt=True
    )
    return mx.array(ids)


def pregate_math_prompt(q):
    # weak prompt, matches the math-adapter training style (#### N)
    return f"Solve the following math problem step by step.\n\n{q}"


def pregate_code_prompt(user):
    return user  # code adapter trained on bare instruction -> code


def twodomain_prompt(q):
    # genuine math AND code: must WRITE runnable python that RETURNS the answer
    return (
        "Write a Python function `solve()` that computes the answer to this problem and returns "
        "the final number. Put it in a ```python code block.\n\n" + q
    )


# ---------------------------------------------------------------------------
# Single-expert greedy decode (pre-gate + best-single conditions)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3-way entropy-argmin router (base | math | code) in lockstep
# ---------------------------------------------------------------------------

def entropy(logits):
    # logits: (1, V) -> scalar nats
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    p = mx.exp(lp)
    return float((-(p * lp).sum()).item())


def router_decode(models, tokenizer, prompt_ids, max_new, bridge_ids):
    """models = [base, math, code]. Returns (text, win_counts, base_bridge_wins, base_total_wins, n_tok)."""
    caches = [make_prompt_cache(m) for m in models]
    # prefill all three on the prompt
    last_logits = []
    for m, c in zip(models, caches):
        lg = m(prompt_ids[None], cache=c)[:, -1, :]
        last_logits.append(lg)
    win_counts = [0, 0, 0]
    base_bridge_wins = 0
    out = []
    for _ in range(max_new):
        ents = [entropy(lg) for lg in last_logits]
        w = int(min(range(3), key=lambda i: ents[i]))
        win_counts[w] += 1
        tok_arr = mx.argmax(last_logits[w], axis=-1)
        mx.eval(tok_arr)
        tid = int(tok_arr.item())
        if w == 0 and tid in bridge_ids:
            base_bridge_wins += 1
        if tid in EOS_IDS:
            break
        out.append(tid)
        # advance all three caches with the emitted token
        y = mx.array([[tid]])
        last_logits = []
        for m, c in zip(models, caches):
            lg = m(y, cache=c)[:, -1, :]
            last_logits.append(lg)
    for c in caches:
        del c
    n_tok = sum(win_counts)
    return tokenizer.decode(out), win_counts, base_bridge_wins, n_tok


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

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


def load_code_valid(n):
    items = []
    for line in CODE_DATA.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        msgs = rec["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        gold = next(m["content"] for m in msgs if m["role"] == "assistant")
        items.append({"user": user, "gold": gold})
        if len(items) >= n:
            break
    log(f"  loaded {len(items)} code-completion problems")
    return items


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def extract_number(text):
    m = re.findall(r"####\s*([\-\d,]+(?:\.\d+)?)", text)
    if m:
        try:
            return float(m[-1].replace(",", ""))
        except ValueError:
            pass
    m = re.findall(r"(?:answer\s*(?:is|:)?\s*\$?)([\-\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        try:
            return float(m[-1].replace(",", ""))
        except ValueError:
            pass
    nums = re.findall(r"[\-\d,]+(?:\.\d+)?", text)
    for v in reversed(nums):
        try:
            return float(v.replace(",", ""))
        except ValueError:
            continue
    return None


def score_math_prose(text, gold):
    pred = extract_number(text)
    return pred is not None and gold is not None and abs(pred - gold) < 1e-2


def _first_code_line(text):
    for ln in text.splitlines():
        s = ln.strip()
        if s:
            return s
    return None


def score_code_completion(text, gold):
    # same loose match the carrier-ablation used: first non-empty line matches
    g = _first_code_line(text)
    gt = _first_code_line(gold)
    return g is not None and gt is not None and g == gt


def extract_python(text):
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    if blocks:
        for b in blocks:
            if "def solve" in b:
                return b
        return blocks[0]
    # no fence: take from first def solve
    idx = text.find("def solve")
    if idx >= 0:
        return text[idx:]
    return None


def score_twodomain(text, gold):
    """Execute the generated solve() and compare its return to gold. Genuine math∧code."""
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


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_pregate(tokenizer):
    """Net-positivity of each adapter on its OWN domain vs base, in THIS harness."""
    log("\n=== PHASE 1: PRE-GATE (net-positivity in F#874 no-thinking harness) ===")
    gsm = load_gsm8k(N_PREGATE_MATH)
    code = load_code_valid(N_PREGATE_CODE)

    # base on both pre-gate domains
    base, _ = load(MODEL_ID)
    log("  -- base on math-prose --")
    base_math = sum(score_math_prose(greedy(base, tokenizer, build_ids(tokenizer, pregate_math_prompt(p["question"])), 512), p["answer_num"]) for p in gsm) / len(gsm)
    log(f"     base_EM_math = {base_math:.4f}")
    log("  -- base on code-completion --")
    base_code = sum(score_code_completion(greedy(base, tokenizer, build_ids(tokenizer, pregate_code_prompt(it["user"])), 128), it["gold"]) for it in code) / len(code)
    log(f"     base_EM_code = {base_code:.4f}")
    del base
    gc.collect(); mx.clear_cache()

    # math adapter on math
    m_model, _ = load(MODEL_ID)
    install_adapter(m_model, MATH_ADAPTER)
    math_math = sum(score_math_prose(greedy(m_model, tokenizer, build_ids(tokenizer, pregate_math_prompt(p["question"])), 512), p["answer_num"]) for p in gsm) / len(gsm)
    log(f"     math_adapter_EM_math = {math_math:.4f}  (delta {100*(math_math-base_math):+.1f}pp)")
    del m_model
    gc.collect(); mx.clear_cache()

    # code adapter on code
    c_model, _ = load(MODEL_ID)
    install_adapter(c_model, CODE_ADAPTER)
    code_code = sum(score_code_completion(greedy(c_model, tokenizer, build_ids(tokenizer, pregate_code_prompt(it["user"])), 128), it["gold"]) for it in code) / len(code)
    log(f"     code_adapter_EM_code = {code_code:.4f}  (delta {100*(code_code-base_code):+.1f}pp)")
    del c_model
    gc.collect(); mx.clear_cache()

    c3_pass = (math_math > base_math) and (code_code > base_code)
    log(f"  PRE-GATE (C3): math {math_math:.4f}>{base_math:.4f} AND code {code_code:.4f}>{base_code:.4f} -> {'PASS' if c3_pass else 'FAIL (KILLED)'}")
    return {
        "base_EM_math": base_math, "base_EM_code": base_code,
        "math_adapter_EM_math": math_math, "code_adapter_EM_code": code_code,
        "delta_math_pp": 100*(math_math-base_math), "delta_code_pp": 100*(code_code-base_code),
        "c3_pass": bool(c3_pass),
    }


def phase_twodomain(tokenizer, bridge_ids):
    log("\n=== PHASE 2: TWO-DOMAIN TASK (GSM8K via runnable solve()) ===")
    gsm = load_gsm8k(N_TWODOMAIN)
    prompts = [build_ids(tokenizer, twodomain_prompt(p["question"])) for p in gsm]

    # best-single: base-alone, math-only and code-only -- ALL on the SAME solve() task/scorer.
    # run_single returns (accuracy, per_item_texts) so we can check router != base collapse.
    def run_single(adapter_path, label):
        mdl, _ = load(MODEL_ID)
        if adapter_path is not None:
            install_adapter(mdl, adapter_path)
        correct = 0
        texts = []
        for pids, p in zip(prompts, gsm):
            txt = greedy(mdl, tokenizer, pids, MAX_NEW)
            texts.append(txt)
            ok, _ = score_twodomain(txt, p["answer_num"])
            correct += int(ok)
        acc = correct / len(gsm)
        log(f"  {label}: EM = {acc:.4f} ({correct}/{len(gsm)})")
        del mdl
        gc.collect(); mx.clear_cache()
        return acc, texts

    # REVIEW fix 1: base-alone is a first-class arm on the EXACT solve() task (not the prose pre-gate).
    em_base_single, base_solve_texts = run_single(None, "single-BASE")
    em_math_single, _ = run_single(MATH_ADAPTER, "single-MATH")
    em_code_single, _ = run_single(CODE_ADAPTER, "single-CODE")
    best_single = max(em_base_single, em_math_single, em_code_single)
    log(f"  best_single = max(base={em_base_single:.4f}, math={em_math_single:.4f}, "
        f"code={em_code_single:.4f}) = {best_single:.4f}")

    # router: base | math | code
    log("  -- building 3 expert models for router --")
    base, _ = load(MODEL_ID)
    m_model, _ = load(MODEL_ID); install_adapter(m_model, MATH_ADAPTER)
    c_model, _ = load(MODEL_ID); install_adapter(c_model, CODE_ADAPTER)
    models = [base, m_model, c_model]
    log_mem("router-loaded")

    correct = 0
    tot_wins = [0, 0, 0]
    tot_base_bridge = 0
    tot_tok = 0
    n_items_differ_from_base = 0   # REVIEW fix 1: router output genuinely != base-alone (not collapsed)
    for idx, (pids, p) in enumerate(zip(prompts, gsm)):
        txt, wins, base_bridge, ntok = router_decode(models, tokenizer, pids, MAX_NEW, bridge_ids)
        ok, _ = score_twodomain(txt, p["answer_num"])
        correct += int(ok)
        if txt != base_solve_texts[idx]:
            n_items_differ_from_base += 1
        for i in range(3):
            tot_wins[i] += wins[i]
        tot_base_bridge += base_bridge
        tot_tok += ntok
        mx.clear_cache()
    em_router = correct / len(gsm)
    frac_differ = n_items_differ_from_base / max(len(gsm), 1)
    log(f"  router-vs-base divergence: {n_items_differ_from_base}/{len(gsm)} items differ "
        f"({frac_differ:.4f}) -> {'NOT collapsed' if n_items_differ_from_base > 0 else 'COLLAPSED to base'}")
    base_win_frac = tot_wins[0] / max(tot_tok, 1)
    base_bridge_frac = tot_base_bridge / max(tot_wins[0], 1)
    log(f"  ROUTER: EM = {em_router:.4f} ({correct}/{len(gsm)})")
    log(f"  win_counts base/math/code = {tot_wins}  (total {tot_tok} tok)")
    log(f"  base_win_fraction = {base_win_frac:.4f}  (>= 0.15 ?)  of which bridge-token {base_bridge_frac:.4f}")
    del base, m_model, c_model, models
    gc.collect(); mx.clear_cache()

    return {
        "em_base_single": em_base_single,
        "em_math_single": em_math_single, "em_code_single": em_code_single,
        "best_single": best_single,
        "best_single_arm": ("base" if best_single == em_base_single else
                            ("math" if best_single == em_math_single else "code")),
        "em_router": em_router,
        "router_lift_pp": 100*(em_router - best_single),
        "router_lift_vs_base_pp": 100*(em_router - em_base_single),
        "win_counts": {"base": tot_wins[0], "math": tot_wins[1], "code": tot_wins[2]},
        "total_tokens": tot_tok,
        "base_win_fraction": base_win_frac,
        "base_bridge_fraction_of_base_wins": base_bridge_frac,
        "insufficiency_gap_pp": 100*(em_router - best_single),
        "router_items_differ_from_base": n_items_differ_from_base,
        "router_frac_differ_from_base": frac_differ,
        "router_collapsed_to_base": bool(n_items_differ_from_base == 0),
    }


def main():
    global EOS_IDS
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 70)
    log("exp_spark_base_wins_bridges")
    log(f"Base: {MODEL_ID}")
    log(f"math adapter: {MATH_ADAPTER}")
    log(f"code adapter: {CODE_ADAPTER}")
    log(f"scale={LORA_SCALE} rank={LORA_RANK} max_new={MAX_NEW} seed={SEED} thinking=OFF")
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

    # bridge token ids
    bridge_ids = set()
    for s in BRIDGE_STRS:
        for variant in (s, " " + s):
            try:
                ids = tokenizer.encode(variant, add_special_tokens=False)
            except TypeError:
                ids = tokenizer.encode(variant)
            if len(ids) == 1:
                bridge_ids.add(ids[0])
    log(f"bridge token ids: {len(bridge_ids)} single-token pivots")
    gc.collect(); mx.clear_cache()

    pre = phase_pregate(tokenizer)

    if not pre["c3_pass"]:
        # pre-gate fail => KILLED, do not run main test (still real, not smoke)
        results = {
            "experiment": "exp_spark_base_wins_bridges",
            "model": MODEL_ID, "lora_scale": LORA_SCALE, "lora_rank": LORA_RANK,
            "max_new_tokens": MAX_NEW, "seed": SEED, "thinking": False,
            "is_smoke": False,
            "pregate": pre,
            "kill_2311": {
                "C1_router_lift": None, "C2_base_win_fraction": None,
                "C3_pregate_pass": False,
            },
            "all_pass": False,
            "verdict": "killed",
            "kill_reason": "pre-gate (clause 3) failed: an adapter is not net-positive on its own domain in this harness",
            "total_time_s": round(time.time() - t0, 1),
        }
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        log("\nPRE-GATE FAILED -> verdict=killed (clause 3). Results written.")
        log(f"FINAL VERDICT: killed")
        return

    two = phase_twodomain(tokenizer, bridge_ids)

    # best_single now includes base-alone on the solve() task (REVIEW fix 1): router must beat it by >=3pp.
    c1_pass = two["router_lift_pp"] >= 3.0
    c2_pass = two["base_win_fraction"] >= 0.15
    c3_pass = pre["c3_pass"]
    not_collapsed = not two["router_collapsed_to_base"]  # router output must genuinely differ from base
    all_pass = bool(c1_pass and c2_pass and c3_pass and not_collapsed)
    verdict = "supported" if all_pass else "killed"

    log("\n" + "=" * 70)
    log("KILL 2311 (all three clauses; thresholds unchanged)")
    log("=" * 70)
    log(f"  arms: base={two['em_base_single']:.4f} math={two['em_math_single']:.4f} "
        f"code={two['em_code_single']:.4f} -> best_single={two['best_single']:.4f} ({two['best_single_arm']})")
    log(f"  router_EM = {two['em_router']:.4f}  (vs base-alone: {two['router_lift_vs_base_pp']:+.1f}pp)")
    log(f"  C1 router_lift vs best_single = {two['router_lift_pp']:+.1f}pp (>= +3.0) : {'PASS' if c1_pass else 'FAIL'}")
    log(f"  C2 base_win_fraction = {two['base_win_fraction']:.4f} (>= 0.15) : {'PASS' if c2_pass else 'FAIL'}")
    log(f"  C3 pre-gate net-positive : {'PASS' if c3_pass else 'FAIL'}")
    log(f"  non-collapse: router differs from base on {two['router_items_differ_from_base']}/{N_TWODOMAIN} "
        f"items : {'PASS' if not_collapsed else 'FAIL (collapsed)'}")
    log(f"  VERDICT: {verdict}")

    results = {
        "experiment": "exp_spark_base_wins_bridges",
        "model": MODEL_ID, "lora_scale": LORA_SCALE, "lora_rank": LORA_RANK,
        "max_new_tokens": MAX_NEW, "seed": SEED, "thinking": False,
        "n_pregate_math": N_PREGATE_MATH, "n_pregate_code": N_PREGATE_CODE, "n_twodomain": N_TWODOMAIN,
        "is_smoke": False,
        "pregate": pre,
        "twodomain": two,
        "kill_2311": {
            "C1_router_lift_pp": two["router_lift_pp"], "C1_pass": bool(c1_pass),
            "C1_best_single_includes_base": True,
            "C1_router_lift_vs_base_pp": two["router_lift_vs_base_pp"],
            "C2_base_win_fraction": two["base_win_fraction"], "C2_pass": bool(c2_pass),
            "C3_pregate_pass": bool(c3_pass),
            "non_collapse_pass": bool(not_collapsed),
        },
        "all_pass": all_pass,
        "verdict": verdict,
        "total_time_s": round(time.time() - t0, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults -> {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']}s")
    log(f"FINAL VERDICT: {verdict}")


if __name__ == "__main__":
    main()
