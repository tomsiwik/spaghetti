#!/usr/bin/env python3
"""exp_spark_base_safe_harbor — Off-domain interference is concentrated on rare
top-k-conflict tokens; the frozen base is the per-token safe harbor.

Frozen base gemma-4-e4b-it-4bit + F#627 solo math LoRA (q_proj, r=6, scale=6.0).

Per decode step t (lockstep base + composed on the SAME emitted context):
  S_B = top-k(base_logits), S_C = top-k(composed_logits), k=8
  J_t = |S_B ∩ S_C| / |S_B ∪ S_C|        (discrete Jaccard — NOT entropy/energy)
  gate@tau: emit argmax(base) if J_t < tau  (CONFLICT -> base safe harbor)
            else emit argmax(composed)       (AGREEMENT -> math token)

Policies on HumanEval pass@1 (off-domain) + GSM8K exact-match (on-domain), greedy:
  base   : no adapter
  fixed  : always composed (math adapter always on)
  gated@tau : the rule above, tau swept over {0.20,0.35,0.50,0.65,0.80}

K1 (kill 2292, off-domain HumanEval): interference_reduction = (drop_fixed-drop_gated)/drop_fixed >= 0.60
K2 (kill 2292, on-domain GSM8K):      retention = lift_gated/lift_fixed >= 0.80
Selected tau = max interference_reduction s.t. retention>=0.80 (else max int.red.).
SUPPORTED iff both K1 and K2 pass at selected tau, else KILLED.

NO MOCKS. Real model, real adapter, real benchmark execution. is_smoke=False.
Adapts loader/eval/benchmark infra from exp_spark_entropy_gated_lora; the gate is
a NEW discrete set-overlap base-fallback (no new MLX primitives beyond argmax/sort).
mlx-lm == 0.31.2.
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

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
# F#627 solo math adapter (q_proj, r=6, scale=6.0). Canonical location.
ADAPTER_PATH = EXPERIMENT_DIR.parent.parent.parent / "data" / "adapters" / "math" / "adapters.safetensors"

LORA_SCALE = 6.0          # adapter_config.json lora_parameters.scale; <= 8 guard OK
LORA_RANK = 6
TOPK = 8                  # top-k set size for Jaccard
TAUS = [0.20, 0.35, 0.50, 0.65, 0.80]
N_HUMANEVAL = 40
N_GSM8K = 40
MAX_NEW_TOKENS = 1024     # thinking-mode needs headroom
SEED = 42


def log(msg):
    print(msg, flush=True)


def log_mem(label=""):
    log(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB "
        f"cache={mx.get_cache_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")


# ----------------------------------------------------------------------------
# Composed LoRA wrapper (subclass nn.Module + setattr — NEVER override __call__
# on an instance; F#831). Single adapter (N=1) => delta = B @ A; scale constant.
# ----------------------------------------------------------------------------

class ComposedLoRALinear(nn.Module):
    def __init__(self, base_linear, lora_a, lora_b, scale):
        super().__init__()
        self.linear = base_linear        # frozen QuantizedLinear
        self.lora_a = lora_a             # (in, r)
        self.lora_b = lora_b             # (r, out)
        self.scale = scale
        self.linear.freeze()

    def __call__(self, x):
        y = self.linear(x)
        z = (x @ self.lora_a) @ self.lora_b          # Σ B_i @ A_i  (N=1)
        return y + (self.scale * z).astype(x.dtype)


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def attach_composed_lora(model, adapter):
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        a_key = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
        b_key = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
        if a_key not in adapter or b_key not in adapter:
            continue
        base_linear = layer.self_attn.q_proj
        a = adapter[a_key].astype(mx.float32)
        b = adapter[b_key].astype(mx.float32)
        wrapper = ComposedLoRALinear(base_linear, a, b, LORA_SCALE)
        setattr(layer.self_attn, "q_proj", wrapper)   # canonical setattr
        count += 1
    mx.eval(model.parameters())
    log(f"  Attached {count} ComposedLoRALinear wrappers on q_proj")
    assert count == 42, f"expected 42 wrapped layers, got {count}"
    return model


def topk_set(logits_row, k=TOPK):
    """Return a Python frozenset of the top-k token indices for a (V,) logits vector."""
    idx = mx.argpartition(-logits_row, k - 1)[:k]
    mx.eval(idx)
    return frozenset(int(i) for i in idx.tolist())


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def format_chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def eot_ids(tokenizer):
    eos = tokenizer.eos_token_id
    eot_enc = tokenizer.encode("<end_of_turn>")
    eot = eot_enc[-1] if eot_enc else eos
    return eos, eot


def generate_base(model, tokenizer, prompt, max_new=MAX_NEW_TOKENS):
    """Plain greedy decode, no adapter. Returns (text, n_tokens)."""
    ids = mx.array(tokenizer.encode(prompt))
    cache = make_prompt_cache(model)
    eos, eot = eot_ids(tokenizer)
    logits = model(ids[None], cache=cache)[:, -1, :]
    tok = mx.argmax(logits, axis=-1)
    mx.eval(tok)
    out = [tok.item()]
    for _ in range(max_new - 1):
        if out[-1] in (eos, eot):
            break
        logits = model(mx.array([[out[-1]]]), cache=cache)[:, -1, :]
        tok = mx.argmax(logits, axis=-1)
        mx.eval(tok)
        out.append(tok.item())
    del cache
    return tokenizer.decode(out), len(out)


def generate_fixed(comp_model, tokenizer, prompt, max_new=MAX_NEW_TOKENS):
    """Always-composed greedy decode (math adapter always on)."""
    return generate_base(comp_model, tokenizer, prompt, max_new)


def generate_gated(comp_model, base_model, tokenizer, prompt, tau,
                   max_new=MAX_NEW_TOKENS):
    """Lockstep base + composed decode. At each step compute discrete Jaccard of
    top-k sets; if J < tau emit base argmax (safe harbor), else composed argmax.
    Returns (text, n_tokens, conflict_rate). conflict_rate = fraction of steps that
    fell back to base."""
    ids = mx.array(tokenizer.encode(prompt))
    base_cache = make_prompt_cache(base_model)
    comp_cache = make_prompt_cache(comp_model)
    eos, eot = eot_ids(tokenizer)

    def step(cur):
        bl = base_model(cur, cache=base_cache)[:, -1, :][0]   # (V,)
        cl = comp_model(cur, cache=comp_cache)[:, -1, :][0]
        sb = topk_set(bl)
        sc = topk_set(cl)
        inter = len(sb & sc)
        union = len(sb | sc)
        j = inter / union if union else 1.0
        if j < tau:
            tok = int(mx.argmax(bl).item())   # CONFLICT -> base safe harbor
            conflict = 1
        else:
            tok = int(mx.argmax(cl).item())   # AGREEMENT -> composed
            conflict = 0
        return tok, conflict

    tok, conflict = step(ids[None])
    out = [tok]
    n_conf = conflict
    n_steps = 1
    for _ in range(max_new - 1):
        if out[-1] in (eos, eot):
            break
        tok, conflict = step(mx.array([[out[-1]]]))
        out.append(tok)
        n_conf += conflict
        n_steps += 1
    del base_cache, comp_cache
    return tokenizer.decode(out), len(out), n_conf / max(n_steps, 1)


# ----------------------------------------------------------------------------
# Benchmark data
# ----------------------------------------------------------------------------

def load_humaneval(n):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    probs = []
    for i in range(min(n, len(ds))):
        it = ds[i]
        probs.append({"task_id": it["task_id"], "prompt": it["prompt"],
                      "test": it["test"], "entry_point": it["entry_point"]})
    log(f"  Loaded {len(probs)} HumanEval problems")
    return probs


def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    probs = []
    for i in range(min(n, len(ds))):
        it = ds[i]
        m = re.search(r"####\s*([\-\d,]+(?:\.\d+)?)", it["answer"])
        ans = float(m.group(1).replace(",", "")) if m else None
        probs.append({"question": it["question"], "answer_num": ans})
    log(f"  Loaded {len(probs)} GSM8K problems")
    return probs


# ----------------------------------------------------------------------------
# Eval: prompts, extraction, scoring
# ----------------------------------------------------------------------------

def strip_thinking(text):
    if not text:
        return text
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def humaneval_prompt(p):
    return ("Complete this Python function. Return the full function in a "
            "```python code block.\n\n```python\n" + p["prompt"] + "\n```")


def extract_code(text, prompt_code, entry_point):
    text = strip_thinking(text)
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    for blk in blocks:
        if f"def {entry_point}" in blk:
            return blk.strip()
    if blocks:
        return prompt_code + "\n" + blocks[0].strip()
    if f"def {entry_point}" in text:
        return text
    return prompt_code + "\n" + text


def run_humaneval_test(code, test, entry_point, timeout=12):
    full = f"{code}\n\n{test}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def gsm8k_prompt(p):
    return ("Solve this math problem step by step. End with '#### ' followed by "
            "the final numeric answer.\n\n" + p["question"])


def extract_gsm8k(text):
    text = strip_thinking(text)
    for pat in (r"####\s*([\-\d,]+(?:\.\d+)?)",
                r"(?:answer\s*(?:is|:)?\s*\$?)([\-\d,]+(?:\.\d+)?)",
                r"([\-\d,]+(?:\.\d+)?)"):
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m[-1].replace(",", ""))
            except ValueError:
                continue
    return None


# ----------------------------------------------------------------------------
# Condition runners
# ----------------------------------------------------------------------------

def score_humaneval(gen_fn, tokenizer, problems):
    passed, details, confs = 0, [], []
    for p in problems:
        prompt = format_chat(tokenizer, humaneval_prompt(p))
        text, ntok, c = gen_fn(prompt)
        confs.append(c)
        code = extract_code(text, p["prompt"], p["entry_point"])
        ok = run_humaneval_test(code, p["test"], p["entry_point"])
        passed += int(ok)
        details.append({"task_id": p["task_id"], "passed": ok, "ntok": ntok, "conflict": round(c, 3)})
    acc = passed / len(problems)
    mc = sum(confs) / len(confs)
    log(f"    HumanEval pass@1 = {acc:.4f} ({passed}/{len(problems)})  mean_conflict={mc:.3f}")
    return acc, details, mc


def score_gsm8k(gen_fn, tokenizer, problems):
    correct, details, confs = 0, [], []
    for p in problems:
        prompt = format_chat(tokenizer, gsm8k_prompt(p))
        text, ntok, c = gen_fn(prompt)
        confs.append(c)
        pred = extract_gsm8k(text)
        exp = p["answer_num"]
        ok = pred is not None and exp is not None and abs(pred - exp) < 1e-2
        correct += int(ok)
        details.append({"pred": pred, "exp": exp, "passed": ok, "ntok": ntok, "conflict": round(c, 3)})
    acc = correct / len(problems)
    mc = sum(confs) / len(confs)
    log(f"    GSM8K exact-match = {acc:.4f} ({correct}/{len(problems)})  mean_conflict={mc:.3f}")
    return acc, details, mc


# ----------------------------------------------------------------------------
# Phases
# ----------------------------------------------------------------------------

def phase_base(humaneval, gsm8k):
    log("\n=== PHASE 1: BASE (no adapter) ===")
    model, tok = load(MODEL_ID)
    gen = lambda pr: (*generate_base(model, tok, pr), 0.0)
    he_acc, he_det, _ = score_humaneval(gen, tok, humaneval)
    gs_acc, gs_det, _ = score_gsm8k(gen, tok, gsm8k)
    log_mem("base-done")
    del model, tok
    gc.collect(); mx.clear_cache()
    return {"humaneval": he_acc, "gsm8k": gs_acc, "he_details": he_det, "gs_details": gs_det}


def phase_lora(humaneval, gsm8k):
    """Loads base + composed once; runs FIXED policy and all gated@tau policies."""
    log("\n=== PHASE 2: COMPOSED (fixed) + GATED tau-sweep ===")
    adapter = mx.load(str(ADAPTER_PATH))
    base_model, tok = load(MODEL_ID)
    comp_model, _ = load(MODEL_ID)
    attach_composed_lora(comp_model, adapter)
    del adapter
    gc.collect(); mx.clear_cache()

    # FIXED (always composed)
    log("  -- FIXED (always composed) --")
    fgen = lambda pr: (*generate_fixed(comp_model, tok, pr), 0.0)
    f_he, f_he_det, _ = score_humaneval(fgen, tok, humaneval)
    f_gs, f_gs_det, _ = score_gsm8k(fgen, tok, gsm8k)
    fixed = {"humaneval": f_he, "gsm8k": f_gs, "he_details": f_he_det, "gs_details": f_gs_det}

    gated = {}
    for tau in TAUS:
        log(f"  -- GATED tau={tau} --")
        ggen = lambda pr, t=tau: generate_gated(comp_model, base_model, tok, pr, t)
        g_he, g_he_det, g_he_c = score_humaneval(ggen, tok, humaneval)
        g_gs, g_gs_det, g_gs_c = score_gsm8k(ggen, tok, gsm8k)
        gated[str(tau)] = {"humaneval": g_he, "gsm8k": g_gs,
                           "he_conflict": g_he_c, "gs_conflict": g_gs_c,
                           "he_details": g_he_det, "gs_details": g_gs_det}
        log_mem(f"tau{tau}-done")

    del base_model, comp_model, tok
    gc.collect(); mx.clear_cache()
    return fixed, gated


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 70)
    log("exp_spark_base_safe_harbor")
    log(f"Reference: F#627, F#827, arxiv:2311.03099")
    log(f"Platform skills: adapting exp_spark_entropy_gated_lora infra (no new MLX primitives)")
    log(f"Base model: {MODEL_ID}")
    log(f"Adapter: {ADAPTER_PATH}")
    log(f"KC count: 2 (K1 HumanEval pass@1 target, K2 GSM8K exact target)")
    log(f"n_humaneval={N_HUMANEVAL} n_gsm8k={N_GSM8K} scale={LORA_SCALE} rank={LORA_RANK} topk={TOPK}")
    log(f"taus={TAUS}")
    log("=" * 70)
    assert ADAPTER_PATH.exists(), f"adapter missing: {ADAPTER_PATH}"
    log_mem("start")

    log("\n=== PHASE 0: load data ===")
    humaneval = load_humaneval(N_HUMANEVAL)
    gsm8k = load_gsm8k(N_GSM8K)

    base = phase_base(humaneval, gsm8k)
    fixed, gated = phase_lora(humaneval, gsm8k)

    drop_fixed = base["humaneval"] - fixed["humaneval"]     # >0 interference
    lift_fixed = fixed["gsm8k"] - base["gsm8k"]             # >0 on-domain lift

    # Per-tau KC math
    sweep = {}
    for tau in TAUS:
        g = gated[str(tau)]
        drop_gated = base["humaneval"] - g["humaneval"]
        lift_gated = g["gsm8k"] - base["gsm8k"]
        ir = (drop_fixed - drop_gated) / drop_fixed if drop_fixed > 1e-9 else None
        ret = lift_gated / lift_fixed if lift_fixed > 1e-9 else None
        sweep[str(tau)] = {
            "humaneval": g["humaneval"], "gsm8k": g["gsm8k"],
            "drop_gated_pp": drop_gated * 100, "lift_gated_pp": lift_gated * 100,
            "interference_reduction": ir, "retention": ret,
            "he_conflict_rate": g["he_conflict"], "gs_conflict_rate": g["gs_conflict"],
        }

    # Selection: max interference_reduction s.t. retention>=0.80; else max int.red.
    def ir_val(d):
        return d["interference_reduction"] if d["interference_reduction"] is not None else -1e9
    feasible = {t: d for t, d in sweep.items()
                if d["retention"] is not None and d["retention"] >= 0.80}
    pool = feasible if feasible else sweep
    sel_tau = max(pool, key=lambda t: ir_val(pool[t]))
    sel = sweep[sel_tau]

    ir = sel["interference_reduction"]
    ret = sel["retention"]
    k1_pass = ir is not None and ir >= 0.60
    k2_pass = ret is not None and ret >= 0.80
    all_pass = bool(k1_pass and k2_pass)
    verdict = "SUPPORTED" if all_pass else "KILLED"

    log("\n" + "=" * 70)
    log("KILL CRITERIA")
    log("=" * 70)
    log(f"  HumanEval pass@1: base={base['humaneval']:.4f} fixed={fixed['humaneval']:.4f}")
    log(f"  GSM8K exact:      base={base['gsm8k']:.4f} fixed={fixed['gsm8k']:.4f}")
    log(f"  drop_fixed={drop_fixed*100:+.1f}pp  lift_fixed={lift_fixed*100:+.1f}pp")
    for tau in TAUS:
        d = sweep[str(tau)]
        log(f"  tau={tau}: HE={d['humaneval']:.3f} GSM={d['gsm8k']:.3f} "
            f"IR={d['interference_reduction']} RET={d['retention']} "
            f"conf(he={d['he_conflict_rate']:.3f},gs={d['gs_conflict_rate']:.3f})")
    log(f"  SELECTED tau={sel_tau}")
    log(f"  K1 interference_reduction={ir} (>=0.60): {'PASS' if k1_pass else 'FAIL'}")
    log(f"  K2 retention={ret} (>=0.80): {'PASS' if k2_pass else 'FAIL'}")
    log(f"  VERDICT: {verdict}")

    results = {
        "experiment": "exp_spark_base_safe_harbor",
        "model": MODEL_ID,
        "adapter_path": str(ADAPTER_PATH),
        "adapter_note": "F#627 solo math adapter (q_proj r=6 scale=6.0), data/adapters/math",
        "lora_scale": LORA_SCALE, "lora_rank": LORA_RANK, "topk": TOPK,
        "n_humaneval": N_HUMANEVAL, "n_gsm8k": N_GSM8K,
        "enable_thinking": True, "greedy": True, "is_smoke": False,
        "taus": TAUS, "selected_tau": float(sel_tau),
        "metrics": {
            "humaneval_pass1": {"base": base["humaneval"], "fixed": fixed["humaneval"],
                                 "gated_selected": sel["humaneval"]},
            "gsm8k_exact": {"base": base["gsm8k"], "fixed": fixed["gsm8k"],
                            "gated_selected": sel["gsm8k"]},
            "drop_fixed_pp": drop_fixed * 100, "lift_fixed_pp": lift_fixed * 100,
            "interference_reduction": ir, "retention": ret,
            "he_conflict_rate_selected": sel["he_conflict_rate"],
            "gs_conflict_rate_selected": sel["gs_conflict_rate"],
        },
        "sweep": sweep,
        "kill_criteria": {
            "K1": {"id": 2292, "metric": "HumanEval pass@1",
                   "interference_reduction": ir, "threshold": 0.60, "pass": bool(k1_pass)},
            "K2": {"id": 2292, "metric": "GSM8K exact-match",
                   "retention": ret, "threshold": 0.80, "pass": bool(k2_pass)},
        },
        "all_pass": all_pass, "verdict": verdict,
        "total_time_s": round(time.time() - t0, 1),
        "details": {
            "base": {"he": base["he_details"], "gs": base["gs_details"]},
            "fixed": {"he": fixed["he_details"], "gs": fixed["gs_details"]},
            "gated_selected": {"he": gated[sel_tau]["he_details"],
                               "gs": gated[sel_tau]["gs_details"]},
        },
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults -> {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']}s")
    log(f"FINAL VERDICT: {verdict}")


if __name__ == "__main__":
    main()
