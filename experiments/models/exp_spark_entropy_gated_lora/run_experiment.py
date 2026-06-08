#!/usr/bin/env python3
"""exp_spark_entropy_gated_lora — Off-domain LoRA interference as a low-entropy-token artifact.

Frozen base gemma-4-e4b-it-4bit + math LoRA (q_proj, r=6, scale=6.0).
Three conditions on HumanEval pass@1 (off-domain) and GSM8K exact-match (on-domain):
  - base   : no adapter
  - fixed  : adapter at constant scale 6.0 (gate=1 every step)
  - gated  : adapter scale 6.0 * (1 - p_top1_base(t)) per decode step

Per-token gate uses the FROZEN base model's own p_top1 on the actual decoded context,
computed by a second base instance run in lockstep with the gated/lora model.

K1: interference_reduction = (drop_fixed - drop_gated)/drop_fixed >= 0.75  (HumanEval pass@1)
K2: retention = lift_gated / lift_fixed >= 0.80                            (GSM8K exact-match)

NO MOCKS. Real model, real adapter, real benchmark execution. is_smoke=False.
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
from mlx.utils import tree_flatten

device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 6 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
ADAPTER_PATH = (
    EXPERIMENT_DIR.parent
    / "exp_p1_t2_single_domain_training"
    / "adapters"
    / "math"
    / "adapters.safetensors"
)
LORA_SCALE = 6.0          # from adapter_config.json lora_parameters.scale; <= 8 guard OK
LORA_RANK = 6
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
# Gated LoRA wrapper (subclass nn.Module + setattr — NEVER override __call__ on instance)
# ----------------------------------------------------------------------------

class GateHolder:
    """Single mutable per-step gate shared by all wrappers."""
    def __init__(self):
        self.value = mx.array(1.0)


class GatedLoRALinear(nn.Module):
    def __init__(self, base_linear, lora_a, lora_b, scale, holder):
        super().__init__()
        self.linear = base_linear        # frozen QuantizedLinear
        self.lora_a = lora_a             # (in, r)
        self.lora_b = lora_b             # (r, out)
        self.scale = scale
        self._holder = holder
        self.linear.freeze()

    def __call__(self, x):
        y = self.linear(x)
        z = (x @ self.lora_a) @ self.lora_b          # (..., out)
        return y + (self._holder.value * self.scale * z).astype(x.dtype)


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def attach_gated_lora(model, adapter, holder):
    """Wrap q_proj on every layer with GatedLoRALinear using the trained A/B."""
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
        wrapper = GatedLoRALinear(base_linear, a, b, LORA_SCALE, holder)
        setattr(layer.self_attn, "q_proj", wrapper)   # canonical: setattr, not __call__ override
        count += 1
    mx.eval(model.parameters())
    log(f"  Attached {count} GatedLoRALinear wrappers on q_proj")
    assert count == 42, f"expected 42 wrapped layers, got {count}"
    return model


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


def generate_base(model, tokenizer, prompt, max_new=MAX_NEW_TOKENS):
    """Plain greedy decode, no adapter. Returns (text, n_tokens)."""
    ids = mx.array(tokenizer.encode(prompt))
    cache = make_prompt_cache(model)
    logits = model(ids[None], cache=cache)[:, -1, :]
    tok = mx.argmax(logits, axis=-1)
    out = [tok.item()]
    eos = tokenizer.eos_token_id
    eot = tokenizer.encode("<end_of_turn>")
    eot = eot[-1] if eot else eos
    for _ in range(max_new - 1):
        if out[-1] in (eos, eot):
            break
        logits = model(mx.array([[out[-1]]]), cache=cache)[:, -1, :]
        tok = mx.argmax(logits, axis=-1)
        mx.eval(tok)
        out.append(tok.item())
    del cache
    return tokenizer.decode(out), len(out)


def generate_gated(lora_model, base_model, tokenizer, holder, prompt,
                   gated, max_new=MAX_NEW_TOKENS):
    """Lockstep decode. base_model gives p_top1; lora_model generates.

    gated=False -> gate fixed at 1.0 (constant scale 6.0).
    gated=True  -> gate = 1 - p_top1_base(t) per step.
    Returns (text, n_tokens, mean_gate).
    """
    ids = mx.array(tokenizer.encode(prompt))
    base_cache = make_prompt_cache(base_model)
    lora_cache = make_prompt_cache(lora_model)

    eos = tokenizer.eos_token_id
    eot_enc = tokenizer.encode("<end_of_turn>")
    eot = eot_enc[-1] if eot_enc else eos

    # Prefill base on full prompt -> p_top1 for first generated token.
    base_logits = base_model(ids[None], cache=base_cache)[:, -1, :]
    if gated:
        p_top1 = mx.max(mx.softmax(base_logits, axis=-1), axis=-1)
        holder.value = (1.0 - p_top1).reshape(())
    else:
        holder.value = mx.array(1.0)
    gate_sum = float(holder.value.item())
    n_gate = 1

    # Prefill lora model with the (now-set) gate.
    lora_logits = lora_model(ids[None], cache=lora_cache)[:, -1, :]
    tok = mx.argmax(lora_logits, axis=-1)
    mx.eval(tok)
    out = [tok.item()]

    for _ in range(max_new - 1):
        if out[-1] in (eos, eot):
            break
        cur = mx.array([[out[-1]]])
        # base step -> p_top1 for THIS step's context
        base_logits = base_model(cur, cache=base_cache)[:, -1, :]
        if gated:
            p_top1 = mx.max(mx.softmax(base_logits, axis=-1), axis=-1)
            holder.value = (1.0 - p_top1).reshape(())
        else:
            holder.value = mx.array(1.0)
        mx.eval(holder.value)
        gate_sum += float(holder.value.item())
        n_gate += 1
        # lora step with that gate
        lora_logits = lora_model(cur, cache=lora_cache)[:, -1, :]
        tok = mx.argmax(lora_logits, axis=-1)
        mx.eval(tok)
        out.append(tok.item())

    del base_cache, lora_cache
    return tokenizer.decode(out), len(out), gate_sum / max(n_gate, 1)


# ----------------------------------------------------------------------------
# Benchmark data
# ----------------------------------------------------------------------------

def load_humaneval(n):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    probs = []
    for i in range(min(n, len(ds))):
        it = ds[i]
        probs.append({
            "task_id": it["task_id"], "prompt": it["prompt"],
            "test": it["test"], "entry_point": it["entry_point"],
        })
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
    return (
        "Complete this Python function. Return the full function in a "
        "```python code block.\n\n```python\n" + p["prompt"] + "\n```"
    )


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
    return (
        "Solve this math problem step by step. End with '#### ' followed by "
        "the final numeric answer.\n\n" + p["question"]
    )


def extract_gsm8k(text):
    text = strip_thinking(text)
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
    m = re.findall(r"([\-\d,]+(?:\.\d+)?)", text)
    if m:
        try:
            return float(m[-1].replace(",", ""))
        except ValueError:
            pass
    return None


# ----------------------------------------------------------------------------
# Condition runners
# ----------------------------------------------------------------------------

def eval_humaneval(gen_fn, tokenizer, problems):
    passed, details, gates = 0, [], []
    for p in problems:
        prompt = format_chat(tokenizer, humaneval_prompt(p))
        text, ntok, g = gen_fn(prompt)
        gates.append(g)
        code = extract_code(text, p["prompt"], p["entry_point"])
        ok = run_humaneval_test(code, p["test"], p["entry_point"])
        passed += int(ok)
        details.append({"task_id": p["task_id"], "passed": ok, "ntok": ntok, "gate": round(g, 3)})
    acc = passed / len(problems)
    mean_gate = sum(gates) / len(gates)
    log(f"    HumanEval pass@1 = {acc:.4f} ({passed}/{len(problems)})  mean_gate={mean_gate:.3f}")
    return acc, details, mean_gate


def eval_gsm8k(gen_fn, tokenizer, problems):
    correct, details, gates = 0, [], []
    for p in problems:
        prompt = format_chat(tokenizer, gsm8k_prompt(p))
        text, ntok, g = gen_fn(prompt)
        gates.append(g)
        pred = extract_gsm8k(text)
        exp = p["answer_num"]
        ok = pred is not None and exp is not None and abs(pred - exp) < 1e-2
        correct += int(ok)
        details.append({"pred": pred, "exp": exp, "passed": ok, "ntok": ntok, "gate": round(g, 3)})
    acc = correct / len(problems)
    mean_gate = sum(gates) / len(gates)
    log(f"    GSM8K exact-match = {acc:.4f} ({correct}/{len(problems)})  mean_gate={mean_gate:.3f}")
    return acc, details, mean_gate


# ----------------------------------------------------------------------------
# Phases
# ----------------------------------------------------------------------------

def phase_base(humaneval, gsm8k):
    log("\n=== PHASE 1: BASE (no adapter) ===")
    model, tok = load(MODEL_ID)
    gen = lambda pr: (*generate_base(model, tok, pr), 0.0)
    he_acc, he_det, _ = eval_humaneval(gen, tok, humaneval)
    gs_acc, gs_det, _ = eval_gsm8k(gen, tok, gsm8k)
    log_mem("base-done")
    del model, tok
    gc.collect(); mx.clear_cache()
    return {"humaneval": he_acc, "gsm8k": gs_acc,
            "he_details": he_det, "gs_details": gs_det}


def phase_lora(humaneval, gsm8k, gated):
    name = "GATED" if gated else "FIXED"
    log(f"\n=== PHASE {'3' if gated else '2'}: {name} adapter ===")
    holder = GateHolder()
    adapter = mx.load(str(ADAPTER_PATH))
    base_model, tok = load(MODEL_ID)      # provides p_top1
    lora_model, _ = load(MODEL_ID)        # carries the adapter
    attach_gated_lora(lora_model, adapter, holder)
    del adapter
    gc.collect(); mx.clear_cache()

    gen = lambda pr: generate_gated(lora_model, base_model, tok, holder, pr, gated=gated)
    he_acc, he_det, he_gate = eval_humaneval(gen, tok, humaneval)
    gs_acc, gs_det, gs_gate = eval_gsm8k(gen, tok, gsm8k)
    log_mem(f"{name}-done")
    del base_model, lora_model, tok, holder
    gc.collect(); mx.clear_cache()
    return {"humaneval": he_acc, "gsm8k": gs_acc,
            "he_details": he_det, "gs_details": gs_det,
            "he_mean_gate": he_gate, "gs_mean_gate": gs_gate}


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 70)
    log("exp_spark_entropy_gated_lora")
    log(f"Base: {MODEL_ID}  Adapter: {ADAPTER_PATH}")
    log(f"n_humaneval={N_HUMANEVAL} n_gsm8k={N_GSM8K} scale={LORA_SCALE} rank={LORA_RANK}")
    log("=" * 70)
    assert ADAPTER_PATH.exists(), f"adapter missing: {ADAPTER_PATH}"
    log_mem("start")

    log("\n=== PHASE 0: load data ===")
    humaneval = load_humaneval(N_HUMANEVAL)
    gsm8k = load_gsm8k(N_GSM8K)

    base = phase_base(humaneval, gsm8k)
    fixed = phase_lora(humaneval, gsm8k, gated=False)
    gated = phase_lora(humaneval, gsm8k, gated=True)

    # ---- Kill criteria ----
    drop_fixed = base["humaneval"] - fixed["humaneval"]      # >0 means interference
    drop_gated = base["humaneval"] - gated["humaneval"]
    lift_fixed = fixed["gsm8k"] - base["gsm8k"]              # >0 means on-domain lift
    lift_gated = gated["gsm8k"] - base["gsm8k"]

    interference_reduction = (
        (drop_fixed - drop_gated) / drop_fixed if drop_fixed > 1e-9 else None
    )
    retention = lift_gated / lift_fixed if lift_fixed > 1e-9 else None

    drop_gated_pp = drop_gated * 100.0
    k1_pass = (
        interference_reduction is not None
        and interference_reduction >= 0.75
        and drop_gated_pp <= 3.0
    )
    k2_pass = retention is not None and retention >= 0.80

    all_pass = bool(k1_pass and k2_pass)
    verdict = "SUPPORTED" if all_pass else "KILLED"

    log("\n" + "=" * 70)
    log("KILL CRITERIA")
    log("=" * 70)
    log(f"  HumanEval pass@1: base={base['humaneval']:.4f} fixed={fixed['humaneval']:.4f} gated={gated['humaneval']:.4f}")
    log(f"  GSM8K exact:      base={base['gsm8k']:.4f} fixed={fixed['gsm8k']:.4f} gated={gated['gsm8k']:.4f}")
    log(f"  drop_fixed={drop_fixed*100:+.1f}pp drop_gated={drop_gated*100:+.1f}pp")
    log(f"  lift_fixed={lift_fixed*100:+.1f}pp lift_gated={lift_gated*100:+.1f}pp")
    log(f"  K1 interference_reduction={interference_reduction} (>=0.75) AND drop_gated<=3pp ({drop_gated_pp:.1f}pp): {'PASS' if k1_pass else 'FAIL'}")
    log(f"  K2 retention={retention} (>=0.80): {'PASS' if k2_pass else 'FAIL'}")
    log(f"  VERDICT: {verdict}")

    results = {
        "experiment": "exp_spark_entropy_gated_lora",
        "model": MODEL_ID,
        "adapter_path": str(ADAPTER_PATH),
        "lora_scale": LORA_SCALE,
        "lora_rank": LORA_RANK,
        "n_humaneval": N_HUMANEVAL,
        "n_gsm8k": N_GSM8K,
        "enable_thinking": True,
        "greedy": True,
        "is_smoke": False,
        "metrics": {
            "humaneval_pass1": {"base": base["humaneval"], "fixed": fixed["humaneval"], "gated": gated["humaneval"]},
            "gsm8k_exact": {"base": base["gsm8k"], "fixed": fixed["gsm8k"], "gated": gated["gsm8k"]},
            "drop_fixed_pp": drop_fixed * 100,
            "drop_gated_pp": drop_gated * 100,
            "lift_fixed_pp": lift_fixed * 100,
            "lift_gated_pp": lift_gated * 100,
            "interference_reduction": interference_reduction,
            "retention": retention,
            "mean_gate_humaneval_gated": gated["he_mean_gate"],
            "mean_gate_gsm8k_gated": gated["gs_mean_gate"],
        },
        "kill_criteria": {
            "K1": {"id": 2291, "metric": "HumanEval pass@1",
                   "interference_reduction": interference_reduction,
                   "drop_gated_pp": drop_gated_pp, "pass": bool(k1_pass)},
            "K2": {"id": 2291, "metric": "GSM8K exact-match",
                   "retention": retention, "pass": bool(k2_pass)},
        },
        "all_pass": all_pass,
        "verdict": verdict,
        "total_time_s": round(time.time() - t0, 1),
        "details": {
            "base": {"he": base["he_details"], "gs": base["gs_details"]},
            "fixed": {"he": fixed["he_details"], "gs": fixed["gs_details"]},
            "gated": {"he": gated["he_details"], "gs": gated["gs_details"]},
        },
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults -> {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']}s")
    log(f"FINAL VERDICT: {verdict}")


if __name__ == "__main__":
    main()
