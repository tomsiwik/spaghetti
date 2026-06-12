#!/usr/bin/env python3
"""exp_spark_layer_self_gate — Off-domain LoRA damage is LAYER-LOCALIZED and prompt-predictable.

Frozen base mlx-community/gemma-4-e4b-it-4bit. Two q_proj LoRA adapters (r=6, scale=6.0, 42 layers):
  code  = on-task adapter (HumanEval),  math = off-task interferer.

Conditions on HumanEval pass@1 (off-domain) n=50:
  A base        : no adapter
  B code-solo   : code in all 42 layers (ceiling)
  C naive comp  : code all 42 + math all 42 (in-run interference baseline; the KILL ANCHOR)
  D self-gate   : code all 42 + math only in the top-k of 42 q_proj layers where the per-layer prompt
                  cosine gamma^l = mean_t cos(delta_math^l(x), q_codebase^l(x)) is most CONSTRUCTIVE.
                  gamma^l comes from ONE free prompt forward pass per prompt. Sweep k.

KILL (DB id 2296, anchored to IN-RUN C):
  best_D = max_k pass@1(D,k).
  SUPPORTED iff best_D - pass@1(C) >= +8pp  AND  best_D >= pass@1(B) - 6pp.
  KILLED otherwise.

Mask is a genuine per-prompt computation (no hardcoded layer indices). Composition is Sum_i (B_i @ A_i).
NO MOCKS. Real model, real adapters, real HumanEval unit-test execution. is_smoke=False.
mlx-lm == 0.31.x.
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
ADAPTER_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training" / "adapters"
MATH_ADAPTER = ADAPTER_DIR / "math" / "adapters.safetensors"
CODE_ADAPTER = ADAPTER_DIR / "code" / "adapters.safetensors"

LORA_SCALE = 6.0          # adapter_config lora_parameters.scale; <= 8 guard OK
LORA_RANK = 6
N_LAYERS = 42
N_HUMANEVAL = 50
N_GSM8K = 50
K_SWEEP = [6, 12, 18, 24, 30, 36]
MAX_NEW_TOKENS = 1024
SEED = 42


def log(msg):
    print(msg, flush=True)


def log_mem(label=""):
    log(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB "
        f"cache={mx.get_cache_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")


# ----------------------------------------------------------------------------
# Shared mutable state for the composed wrapper
# ----------------------------------------------------------------------------

class Ctrl:
    """Shared controller for all wrappers.

    math_mask : python list of 42 bools — whether math delta is active at layer l.
    probe     : when True, wrappers record per-layer gamma into self.gamma during the forward.
    """
    def __init__(self):
        self.math_mask = [True] * N_LAYERS   # default: math everywhere (condition C)
        self.probe = False
        self.gamma = {}                      # layer_idx -> float cosine (filled during probe pass)


class ComposedLoRALinear(nn.Module):
    """q_proj = W_q x + s*(B_c A_c) x + mask_l * s*(B_m A_m) x.

    setattr-replaced submodule (NOT __call__-on-instance override).
    During a probe pass, records gamma^l = mean_t cos(delta_math, q_codebase) into ctrl.gamma.
    """
    def __init__(self, base_linear, ac, bc, am, bm, scale, ctrl, layer_idx,
                 has_code, has_math):
        super().__init__()
        self.linear = base_linear            # frozen QuantizedLinear (W_q)
        self.ac = ac; self.bc = bc           # code A/B  (in,r),(r,out)  or None
        self.am = am; self.bm = bm           # math A/B               or None
        self.scale = scale
        self._ctrl = ctrl
        self._li = layer_idx
        self._has_code = has_code
        self._has_math = has_math
        self.linear.freeze()

    def __call__(self, x):
        y = self.linear(x)                                   # W_q x  (..., out)
        q_codebase = y
        if self._has_code:
            zc = (x @ self.ac) @ self.bc
            zc = (self.scale * zc).astype(x.dtype)
            q_codebase = y + zc
        else:
            zc = None

        delta_m = None
        if self._has_math:
            zm = (x @ self.am) @ self.bm
            delta_m = (self.scale * zm).astype(x.dtype)

        # Probe pass: record per-layer cosine(delta_math, q_codebase) over prompt tokens.
        if self._ctrl.probe and (delta_m is not None):
            d = delta_m.reshape(-1, delta_m.shape[-1]).astype(mx.float32)   # (T, out)
            q = q_codebase.reshape(-1, q_codebase.shape[-1]).astype(mx.float32)
            num = mx.sum(d * q, axis=-1)
            den = mx.sqrt(mx.sum(d * d, axis=-1)) * mx.sqrt(mx.sum(q * q, axis=-1)) + 1e-8
            cos_t = num / den
            self._ctrl.gamma[self._li] = float(mx.mean(cos_t).item())

        out = q_codebase
        if delta_m is not None and self._ctrl.math_mask[self._li]:
            out = q_codebase + delta_m
        return out


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def attach_composed(model, code_adapter, math_adapter, ctrl, has_code, has_math):
    """Replace q_proj on every layer with ComposedLoRALinear. Returns wrapped count."""
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        base_linear = layer.self_attn.q_proj
        ac = bc = am = bm = None
        if has_code:
            ka = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
            kb = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
            ac = code_adapter[ka].astype(mx.float32)
            bc = code_adapter[kb].astype(mx.float32)
        if has_math:
            ka = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
            kb = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
            am = math_adapter[ka].astype(mx.float32)
            bm = math_adapter[kb].astype(mx.float32)
        wrapper = ComposedLoRALinear(base_linear, ac, bc, am, bm, LORA_SCALE,
                                     ctrl, li, has_code, has_math)
        setattr(layer.self_attn, "q_proj", wrapper)
        count += 1
    mx.eval(model.parameters())
    log(f"  Attached {count} ComposedLoRALinear (code={has_code} math={has_math})")
    assert count == N_LAYERS, f"expected {N_LAYERS} layers, got {count}"
    return model


# ----------------------------------------------------------------------------
# Per-prompt free gamma + top-k mask
# ----------------------------------------------------------------------------

def compute_gamma(model, tokenizer, ctrl, prompt):
    """ONE free forward pass over the prompt tokens; records gamma^l for all 42 layers.

    Returns list of 42 floats (gamma per layer). Does not decode.
    """
    ids = mx.array(tokenizer.encode(prompt))
    ctrl.gamma = {}
    ctrl.probe = True
    cache = make_prompt_cache(model)
    _ = model(ids[None], cache=cache)     # prompt-only forward; wrappers fill ctrl.gamma
    mx.eval(_)
    ctrl.probe = False
    del cache
    g = [ctrl.gamma.get(li, 0.0) for li in range(N_LAYERS)]
    return g


def topk_constructive_mask(gamma, k):
    """Keep math active at the k layers with the largest (most constructive) gamma."""
    order = sorted(range(N_LAYERS), key=lambda i: gamma[i], reverse=True)
    keep = set(order[:k])
    return [i in keep for i in range(N_LAYERS)]


# ----------------------------------------------------------------------------
# Generation (greedy)
# ----------------------------------------------------------------------------

def format_chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
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
# Eval scoring
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
# Condition evaluation
# ----------------------------------------------------------------------------

def eval_humaneval_static(model, tokenizer, ctrl, problems, mask):
    """Static mask (same for all prompts): A/B/C."""
    ctrl.math_mask = list(mask)
    passed, details = 0, []
    for p in problems:
        prompt = format_chat(tokenizer, humaneval_prompt(p))
        text, ntok = generate(model, tokenizer, prompt)
        code = extract_code(text, p["prompt"], p["entry_point"])
        ok = run_humaneval_test(code, p["test"], p["entry_point"])
        passed += int(ok)
        details.append({"task_id": p["task_id"], "passed": ok, "ntok": ntok})
    acc = passed / len(problems)
    log(f"    HumanEval pass@1 = {acc:.4f} ({passed}/{len(problems)})")
    return acc, details


def eval_humaneval_selfgate(model, tokenizer, ctrl, problems, k):
    """Per-prompt: free gamma forward pass -> top-k constructive math mask -> decode."""
    passed, details = 0, []
    for p in problems:
        prompt = format_chat(tokenizer, humaneval_prompt(p))
        gamma = compute_gamma(model, tokenizer, ctrl, prompt)
        ctrl.math_mask = topk_constructive_mask(gamma, k)
        text, ntok = generate(model, tokenizer, prompt)
        code = extract_code(text, p["prompt"], p["entry_point"])
        ok = run_humaneval_test(code, p["test"], p["entry_point"])
        passed += int(ok)
        details.append({"task_id": p["task_id"], "passed": ok, "ntok": ntok,
                        "n_pos_gamma": int(sum(1 for g in gamma if g > 0)),
                        "mean_gamma": round(sum(gamma) / len(gamma), 4)})
    acc = passed / len(problems)
    log(f"    HumanEval pass@1 (D,k={k}) = {acc:.4f} ({passed}/{len(problems)})")
    return acc, details


def eval_gsm8k_static(model, tokenizer, ctrl, problems, mask):
    ctrl.math_mask = list(mask)
    correct, details = 0, []
    for p in problems:
        prompt = format_chat(tokenizer, gsm8k_prompt(p))
        text, ntok = generate(model, tokenizer, prompt)
        pred = extract_gsm8k(text)
        exp = p["answer_num"]
        ok = pred is not None and exp is not None and abs(pred - exp) < 1e-2
        correct += int(ok)
        details.append({"pred": pred, "exp": exp, "passed": ok, "ntok": ntok})
    acc = correct / len(problems)
    log(f"    GSM8K exact = {acc:.4f} ({correct}/{len(problems)})")
    return acc, details


def eval_gsm8k_selfgate(model, tokenizer, ctrl, problems, k):
    correct, details = 0, []
    for p in problems:
        prompt = format_chat(tokenizer, gsm8k_prompt(p))
        gamma = compute_gamma(model, tokenizer, ctrl, prompt)
        ctrl.math_mask = topk_constructive_mask(gamma, k)
        text, ntok = generate(model, tokenizer, prompt)
        pred = extract_gsm8k(text)
        exp = p["answer_num"]
        ok = pred is not None and exp is not None and abs(pred - exp) < 1e-2
        correct += int(ok)
        details.append({"pred": pred, "exp": exp, "passed": ok, "ntok": ntok})
    acc = correct / len(problems)
    log(f"    GSM8K exact (D,k={k}) = {acc:.4f} ({correct}/{len(problems)})")
    return acc, details


# ----------------------------------------------------------------------------
# Phases
# ----------------------------------------------------------------------------

ALL_OFF = [False] * N_LAYERS
ALL_ON = [True] * N_LAYERS


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 70)
    log("exp_spark_layer_self_gate")
    log(f"Base: {MODEL_ID}")
    log(f"math={MATH_ADAPTER}\ncode={CODE_ADAPTER}")
    log(f"n_he={N_HUMANEVAL} n_gsm={N_GSM8K} scale={LORA_SCALE} k_sweep={K_SWEEP}")
    log("=" * 70)
    assert MATH_ADAPTER.exists(), f"missing {MATH_ADAPTER}"
    assert CODE_ADAPTER.exists(), f"missing {CODE_ADAPTER}"
    log_mem("start")

    log("\n=== PHASE 0: data ===")
    humaneval = load_humaneval(N_HUMANEVAL)
    gsm8k = load_gsm8k(N_GSM8K)

    code_ad = mx.load(str(CODE_ADAPTER))
    math_ad = mx.load(str(MATH_ADAPTER))

    he = {}   # condition -> pass@1
    gs = {}
    det = {}

    # ---- A: base only (no adapters; build a model with no LoRA wrappers) ----
    log("\n=== PHASE A: BASE only ===")
    modelA, tok = load(MODEL_ID)
    ctrlA = Ctrl()
    # base = code adapter present but masked off + math off => wrap with has_code False, has_math False
    attach_composed(modelA, None, None, ctrlA, has_code=False, has_math=False)
    he["A"], det["A_he"] = eval_humaneval_static(modelA, tok, ctrlA, humaneval, ALL_OFF)
    gs["A"], det["A_gs"] = eval_gsm8k_static(modelA, tok, ctrlA, gsm8k, ALL_OFF)
    log_mem("A-done")
    del modelA, ctrlA
    gc.collect(); mx.clear_cache()

    # ---- B: code-solo (code all 42, no math) ----
    log("\n=== PHASE B: CODE-SOLO (ceiling) ===")
    modelB, tok = load(MODEL_ID)
    ctrlB = Ctrl()
    attach_composed(modelB, code_ad, None, ctrlB, has_code=True, has_math=False)
    he["B"], det["B_he"] = eval_humaneval_static(modelB, tok, ctrlB, humaneval, ALL_OFF)
    gs["B"], det["B_gs"] = eval_gsm8k_static(modelB, tok, ctrlB, gsm8k, ALL_OFF)
    log_mem("B-done")
    del modelB, ctrlB
    gc.collect(); mx.clear_cache()

    # ---- C and D share one model (code+math wrappers); switch via ctrl.math_mask ----
    log("\n=== PHASE C+D: CODE + MATH (composed) ===")
    modelCD, tok = load(MODEL_ID)
    ctrl = Ctrl()
    attach_composed(modelCD, code_ad, math_ad, ctrl, has_code=True, has_math=True)

    # C: math in all 42 layers (the in-run kill anchor)
    log("  -- C: naive full-layer composition --")
    he["C"], det["C_he"] = eval_humaneval_static(modelCD, tok, ctrl, humaneval, ALL_ON)
    gs["C"], det["C_gs"] = eval_gsm8k_static(modelCD, tok, ctrl, gsm8k, ALL_ON)

    # D: layer-self-gate, sweep k (per-prompt free gamma top-k constructive mask)
    he_D = {}
    det_D = {}
    log("  -- D: layer-self-gate sweep --")
    for k in K_SWEEP:
        acc, dt = eval_humaneval_selfgate(modelCD, tok, ctrl, humaneval, k)
        he_D[str(k)] = acc
        det_D[str(k)] = dt

    # best D and its k -> measure GSM8K at that k for on-domain characterization
    best_k = max(he_D, key=lambda kk: he_D[kk])
    best_D = he_D[best_k]
    log(f"  best D: k={best_k} pass@1={best_D:.4f}")
    gs_D_bestk, det_gsD = eval_gsm8k_selfgate(modelCD, tok, ctrl, gsm8k, int(best_k))

    log_mem("CD-done")
    del modelCD, ctrl, code_ad, math_ad
    gc.collect(); mx.clear_cache()

    # ---- Kill criteria (anchored to in-run C) ----
    recover_pp = (best_D - he["C"]) * 100.0
    floor_pp = (best_D - (he["B"] - 0.06)) * 100.0   # best_D - (B - 6pp)
    cond_recover = recover_pp >= 8.0
    cond_floor = best_D >= (he["B"] - 0.06)
    all_pass = bool(cond_recover and cond_floor)
    verdict = "SUPPORTED" if all_pass else "KILLED"

    log("\n" + "=" * 70)
    log("KILL CRITERIA (DB id 2296, anchored to in-run C)")
    log("=" * 70)
    log(f"  pass@1: A={he['A']:.4f} B={he['B']:.4f} C={he['C']:.4f} bestD(k={best_k})={best_D:.4f}")
    log(f"  recovery best_D - C = {recover_pp:+.1f}pp  (need >= +8pp): {'PASS' if cond_recover else 'FAIL'}")
    log(f"  floor  best_D vs B-6pp = {floor_pp:+.1f}pp  (need >= 0): {'PASS' if cond_floor else 'FAIL'}")
    log(f"  VERDICT: {verdict}")

    results = {
        "experiment": "exp_spark_layer_self_gate",
        "model": MODEL_ID,
        "math_adapter": str(MATH_ADAPTER),
        "code_adapter": str(CODE_ADAPTER),
        "lora_scale": LORA_SCALE,
        "lora_rank": LORA_RANK,
        "n_layers": N_LAYERS,
        "n_humaneval": N_HUMANEVAL,
        "n_gsm8k": N_GSM8K,
        "k_sweep": K_SWEEP,
        "enable_thinking": True,
        "greedy": True,
        "is_smoke": False,
        "metrics": {
            "humaneval_pass1": {"A": he["A"], "B": he["B"], "C": he["C"], "D": he_D},
            "gsm8k_exact": {"A": gs["A"], "B": gs["B"], "C": gs["C"],
                            "D_bestk": gs_D_bestk, "best_k": int(best_k)},
            "best_D_pass1": best_D,
            "best_D_k": int(best_k),
            "recovery_vs_C_pp": recover_pp,
            "floor_vs_B_minus6_pp": floor_pp,
        },
        "kill_criteria": {
            "id": 2296,
            "metric": "HumanEval pass@1 n=50",
            "anchor": "in-run C",
            "recovery_vs_C_pp": recover_pp,
            "recovery_threshold_pp": 8.0,
            "cond_recover_pass": bool(cond_recover),
            "cond_floor_pass": bool(cond_floor),
            "pass": bool(all_pass),
        },
        "all_pass": all_pass,
        "verdict": verdict,
        "total_time_s": round(time.time() - t0, 1),
        "details": {
            "A_he": det["A_he"], "B_he": det["B_he"], "C_he": det["C_he"],
            "D_he": det_D, "gsm_D_bestk": det_gsD,
            "A_gs": det["A_gs"], "B_gs": det["B_gs"], "C_gs": det["C_gs"],
        },
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults -> {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']}s")
    log(f"FINAL VERDICT: {verdict}")


if __name__ == "__main__":
    main()
