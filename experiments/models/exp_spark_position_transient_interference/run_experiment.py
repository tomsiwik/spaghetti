"""exp_spark_position_transient_interference — is off-domain LoRA interference a
decode-POSITION transient (front-loaded), not a static weight penalty?

Frozen mlx-community/gemma-4-e4b-it-4bit + real F#627 math LoRA (q_proj, r=6, scale 6.0).
Composition: y = base(x) + scale*(x@lora_a)@lora_b  (single adapter, Σ Bᵢ@Aᵢ form).
Gate = python scalar per q_proj wrapper, mutated per decode step / set for whole fwd pass.
NO mocks. Real teacher-forced per-position NLL diagnostic + real greedy generation.

KC 2293 (target, behavioral):
  target_score = min(off_recovery, on_retention) >= 0.70  AND frontload_ratio >= 0.60
  off_recovery = (acc_med_A16 - acc_med_A0)/(acc_med_B - acc_med_A0)
  on_retention = (acc_math_A16 - acc_math_B)/(acc_math_A0 - acc_math_B)
  Preconditions: acc_med_A0 <= acc_med_B-0.06 ; acc_math_A0 >= acc_math_B+0.06
See MATH.md §3-4.
"""
from __future__ import annotations

import gc
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parents[2]

BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
MATH_ADAPTER = REPO / "data" / "adapters" / "math" / "adapters.safetensors"
MED_EVAL = REPO / "data" / "corpora" / "distillation" / "medical" / "eval.jsonl"
MATH_EVAL = REPO / "data" / "corpora" / "distillation" / "math" / "eval.jsonl"

# exact training-prompt templates (so adapter activates as trained; answers stay short/clean)
MED_TEMPLATE = "Answer this medical multiple choice question. Respond with only the letter (A/B/C/D).\n\n{q}"
MATH_TEMPLATE = "Solve the following math problem step by step.\n\n{q}"

SCALE = 6.0          # adapter train scale, <= 8
LATE_FIRE_K = 16     # adapter off for first 16 generated tokens, then on
MAX_NEW = 256        # generation budget (MATH.md §5)
N_DIAG = 50          # held-out medical refs for teacher-forced NLL diagnostic
N_GEN = 50           # held-out items per domain for behavioral generation
SEED = 42

TARGET_THRESH = 0.70
FRONTLOAD_THRESH = 0.60
PRECOND_GAP = 0.06


# --------------------------------------------------------------------------- #
# Single-adapter gated q_proj wrapper.
# Canonical pattern: subclass nn.Module + setattr (mem-antipattern-call-override).
# Composition: y = base(x) + scale*gate*((x@A)@B). Gate is a python scalar.
# --------------------------------------------------------------------------- #
class GatedQProj(nn.Module):
    def __init__(self, base, A, B, scale):
        super().__init__()
        self.base = base
        self.A, self.B = A, B
        self.scale = scale
        self.gate = 1.0  # python scalar, mutable, not a param -> no grad

    def __call__(self, x):
        y = self.base(x)
        if self.gate != 0.0:
            xf = x.astype(self.A.dtype)
            y = y + (self.scale * self.gate) * ((xf @ self.A) @ self.B).astype(y.dtype)
        return y


def attach(model, w, scale):
    layers = model.language_model.model.layers
    wrappers = []
    for L in range(len(layers)):
        attn = layers[L].self_attn
        pre = f"language_model.model.layers.{L}.self_attn.q_proj."
        g = GatedQProj(attn.q_proj, w[pre + "lora_a"], w[pre + "lora_b"], scale)
        attn.q_proj = g
        wrappers.append(g)
    return wrappers


def set_gate(wrappers, val):
    for g in wrappers:
        g.gate = val


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path, n: int) -> List[Dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            m = ex["messages"]
            out.append({"q": m[0]["content"], "gold": m[1]["content"]})
            if len(out) >= n:
                break
    return out


def gold_letter(gold: str):
    m = re.search(r"\b([ABCD])\b", gold)
    return m.group(1) if m else None


def pred_letter(text: str):
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else None


def gold_num(gold: str):
    m = re.search(r"####\s*([\d,\-\.]+)", gold)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?\d+\.?\d*", gold.replace(",", ""))
    return nums[-1] if nums else None


def pred_num(text: str):
    m = re.search(r"####\s*([\d,\-\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


def num_eq(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-4
    except ValueError:
        return a == b


# --------------------------------------------------------------------------- #
# Teacher-forced per-position NLL — ONE forward pass per sequence per condition.
# Returns per-position NLL of the gold assistant tokens (aligned to gen position t).
# --------------------------------------------------------------------------- #
def teacher_forced_nll(model, prompt_ids: List[int], target_ids: List[int]) -> List[float]:
    """NLL_t = -log p(target_t | prompt, target_<t) for each target position t.
    Single forward pass over [prompt + target] (no decode)."""
    full = mx.array(prompt_ids + target_ids)[None]
    logits = model(full)[0]                       # (S, V)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    p = len(prompt_ids)
    tgt = mx.array(target_ids)
    # token at full position (p-1+t) predicts target_t
    idx = mx.arange(len(target_ids))
    sel = logp[p - 1 + idx, tgt]                   # (len(target),)
    mx.eval(sel)
    return [-float(v) for v in sel.tolist()]


# --------------------------------------------------------------------------- #
# Greedy decode with a position gate (adapter off for first late_fire_k gen tokens).
# --------------------------------------------------------------------------- #
def decode_gated(model, wrappers, eos_ids, prompt_ids: mx.array, max_new: int,
                 late_fire_k: int, base_mode: bool) -> List[int]:
    cache = make_prompt_cache(model)
    # prompt pass: base_mode => gate off; else gate on (prompt counts as t<0, always ON
    # for A0/A16 since policy only gates the FIRST 16 GENERATED tokens)
    set_gate(wrappers, 0.0 if base_mode else 1.0)
    logits = model(prompt_ids[None], cache=cache)[0, -1]
    out = []
    for step in range(max_new):
        if base_mode:
            set_gate(wrappers, 0.0)
        else:
            set_gate(wrappers, 0.0 if step < late_fire_k else 1.0)
        tok = int(mx.argmax(logits).item())
        if tok in eos_ids:
            break
        out.append(tok)
        logits = model(mx.array([[tok]]), cache=cache)[0, -1]
        mx.eval(logits)
    return out


# --------------------------------------------------------------------------- #
def build_prompt(tokenizer, template: str, q: str) -> List[int]:
    messages = [{"role": "user", "content": template.format(q=q)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )


def run_diagnostic(model, wrappers, tokenizer, med_items) -> Dict:
    """Front-load NLL diagnostic on medical refs. Per-position delta = compose - base."""
    sum_pos = [0.0] * MAX_NEW    # sum over seqs of max(delta_t,0)
    cnt_pos = [0] * MAX_NEW
    per_seq = []
    for i, ex in enumerate(med_items):
        prompt_ids = build_prompt(tokenizer, MED_TEMPLATE, ex["q"])
        target_ids = tokenizer.encode(ex["gold"], add_special_tokens=False)[:MAX_NEW]
        if len(target_ids) == 0:
            continue
        set_gate(wrappers, 0.0)
        nll_base = teacher_forced_nll(model, prompt_ids, target_ids)
        set_gate(wrappers, 1.0)
        nll_comp = teacher_forced_nll(model, prompt_ids, target_ids)
        deltas = [c - b for c, b in zip(nll_comp, nll_base)]
        pos_mass = [max(d, 0.0) for d in deltas]
        front = sum(pos_mass[:LATE_FIRE_K])
        total = sum(pos_mass)
        per_seq.append({"idx": i, "T": len(deltas), "front_mass": front, "total_mass": total})
        for t, m in enumerate(pos_mass):
            sum_pos[t] += m
            cnt_pos[t] += 1
        if (i + 1) % 10 == 0:
            mx.clear_cache()
            print(f"  diag [{i+1}/{len(med_items)}]", flush=True)
    set_gate(wrappers, 1.0)
    total_front = sum(s["front_mass"] for s in per_seq)
    total_all = sum(s["total_mass"] for s in per_seq)
    frontload_ratio = (total_front / total_all) if total_all > 1e-9 else float("nan")
    mean_pos = [(sum_pos[t] / cnt_pos[t]) if cnt_pos[t] else 0.0 for t in range(MAX_NEW)]
    return {
        "frontload_ratio": frontload_ratio,
        "total_front_mass": total_front,
        "total_all_mass": total_all,
        "mean_delta_curve_first64": mean_pos[:64],
        "n_seqs": len(per_seq),
    }


def run_generation(model, wrappers, tokenizer, eos_ids, items, template, is_math) -> Dict[str, float]:
    """3 conditions: B(base), A0(always on), A16(late-fire). Returns acc dict (fractions)."""
    correct = {"B": 0, "A0": 0, "A16": 0}
    n = 0
    for i, ex in enumerate(items):
        gold = gold_num(ex["gold"]) if is_math else gold_letter(ex["gold"])
        if gold is None:
            continue
        n += 1
        prompt_ids = mx.array(build_prompt(tokenizer, template, ex["q"]))
        # B: base
        gB = decode_gated(model, wrappers, eos_ids, prompt_ids, MAX_NEW, LATE_FIRE_K, base_mode=True)
        tB = tokenizer.decode(gB)
        # A0: always on
        gA0 = decode_gated(model, wrappers, eos_ids, prompt_ids, MAX_NEW, late_fire_k=0, base_mode=False)
        tA0 = tokenizer.decode(gA0)
        # A16: late-fire
        gA16 = decode_gated(model, wrappers, eos_ids, prompt_ids, MAX_NEW, LATE_FIRE_K, base_mode=False)
        tA16 = tokenizer.decode(gA16)
        chk = num_eq if is_math else (lambda t, g: pred_letter(t) == g)
        correct["B"] += int(chk(pred_num(tB) if is_math else tB, gold))
        correct["A0"] += int(chk(pred_num(tA0) if is_math else tA0, gold))
        correct["A16"] += int(chk(pred_num(tA16) if is_math else tA16, gold))
        if (i + 1) % 10 == 0:
            mx.clear_cache()
            dom = "math" if is_math else "med"
            print(f"  gen {dom} [{i+1}/{len(items)}] "
                  f"B={correct['B']/n:.2f} A0={correct['A0']/n:.2f} A16={correct['A16']/n:.2f}",
                  flush=True)
    return {k: correct[k] / n for k in correct} | {"n": n}


def main():
    t0 = time.time()
    print(f"Reference: F#627 / F#827 / F#837 / arXiv:2306.01708", flush=True)
    print(f"Platform skills: adapting verified exp_spark_temporal_interference composition "
          f"(no NEW MLX primitives) -> /mlx-dev,/fast-mlx token-saver skip", flush=True)
    print(f"Base model: {BASE_MODEL}  mlx_lm=0.31.2", flush=True)
    print(f"KC count: 1 (KC2293 target_score=min(off_recovery,on_retention)>=0.70 "
          f"+ frontload_ratio>=0.60 diagnostic)", flush=True)
    assert MATH_ADAPTER.exists(), f"adapter missing: {MATH_ADAPTER}"
    assert MED_EVAL.exists() and MATH_EVAL.exists(), "eval data missing"

    mx.random.seed(SEED)
    import random
    random.seed(SEED)

    w = mx.load(str(MATH_ADAPTER))
    model, tokenizer = load(BASE_MODEL)
    model.freeze()
    wrappers = attach(model, w, SCALE)
    model.eval()
    eos_ids = set(tokenizer.eos_token_ids) if hasattr(tokenizer, "eos_token_ids") else {tokenizer.eos_token_id}
    mx.clear_cache()

    med_items = load_jsonl(MED_EVAL, max(N_DIAG, N_GEN))
    math_items = load_jsonl(MATH_EVAL, N_GEN)

    # ---- Phase 1: teacher-forced front-load diagnostic (medical) ----
    print(f"\n[Phase 1] teacher-forced NLL diagnostic on {N_DIAG} medical refs", flush=True)
    diag = run_diagnostic(model, wrappers, tokenizer, med_items[:N_DIAG])
    print(f"  frontload_ratio = {diag['frontload_ratio']:.3f}  (n={diag['n_seqs']})", flush=True)
    mx.clear_cache(); gc.collect()

    # ---- Phase 2: behavioral generation ----
    print(f"\n[Phase 2] generation: medical MCQ ({N_GEN})", flush=True)
    med_acc = run_generation(model, wrappers, tokenizer, eos_ids, med_items[:N_GEN], MED_TEMPLATE, is_math=False)
    mx.clear_cache(); gc.collect()
    print(f"\n[Phase 2] generation: math ({N_GEN})", flush=True)
    math_acc = run_generation(model, wrappers, tokenizer, eos_ids, math_items[:N_GEN], MATH_TEMPLATE, is_math=True)
    mx.clear_cache(); gc.collect()

    # ---- Verdict ----
    acc_med_B, acc_med_A0, acc_med_A16 = med_acc["B"], med_acc["A0"], med_acc["A16"]
    acc_math_B, acc_math_A0, acc_math_A16 = math_acc["B"], math_acc["A0"], math_acc["A16"]

    med_denom = acc_med_B - acc_med_A0          # >0 if adapter hurt medical
    math_denom = acc_math_A0 - acc_math_B       # >0 if adapter lifted math
    off_recovery = (acc_med_A16 - acc_med_A0) / med_denom if abs(med_denom) > 1e-9 else float("nan")
    on_retention = (acc_math_A16 - acc_math_B) / math_denom if abs(math_denom) > 1e-9 else float("nan")

    precond_med = acc_med_A0 <= acc_med_B - PRECOND_GAP
    precond_math = acc_math_A0 >= acc_math_B + PRECOND_GAP
    preconditions_ok = precond_med and precond_math

    frontload_ratio = diag["frontload_ratio"]
    if not preconditions_ok:
        target_score = float("nan")
        verdict = "PROVISIONAL"
        all_pass = False
    else:
        target_score = min(off_recovery, on_retention)
        kc_pass = (target_score >= TARGET_THRESH) and (frontload_ratio >= FRONTLOAD_THRESH)
        all_pass = kc_pass
        verdict = "SUPPORTED" if kc_pass else "KILLED"

    is_smoke = (verdict == "PROVISIONAL")

    results = {
        "experiment_id": "exp_spark_position_transient_interference",
        "config": {
            "base_model": BASE_MODEL, "mlx_lm_version": "0.31.2",
            "math_adapter": str(MATH_ADAPTER), "targets": ["self_attn.q_proj"],
            "scale": SCALE, "late_fire_k": LATE_FIRE_K, "max_new": MAX_NEW,
            "n_diag": N_DIAG, "n_gen": N_GEN, "seed": SEED,
            "enable_thinking": False, "decode": "greedy_argmax",
        },
        "diagnostic": diag,
        "accuracy": {
            "medical": {"B": acc_med_B, "A0": acc_med_A0, "A16": acc_med_A16, "n": med_acc["n"]},
            "math": {"B": acc_math_B, "A0": acc_math_A0, "A16": acc_math_A16, "n": math_acc["n"]},
        },
        "off_recovery": off_recovery,
        "on_retention": on_retention,
        "target_score": target_score,
        "frontload_ratio": frontload_ratio,
        "preconditions": {
            "med_A0<=B-0.06": precond_med, "math_A0>=B+0.06": precond_math,
            "ok": preconditions_ok,
        },
        "kill_criteria": {
            "2293": {
                "text": "target_score=min(off_recovery,on_retention)>=0.70 AND frontload_ratio>=0.60",
                "target_score": target_score, "frontload_ratio": frontload_ratio,
                "thresh_target": TARGET_THRESH, "thresh_frontload": FRONTLOAD_THRESH,
                "type": "target_behavioral",
                "result": ("pass" if all_pass else ("untested" if not preconditions_ok else "fail")),
            }
        },
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": is_smoke,
        "total_wall_clock_sec": time.time() - t0,
    }

    out_path = EXP_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== RESULTS ===", flush=True)
    print(f"medical: B={acc_med_B:.2f} A0={acc_med_A0:.2f} A16={acc_med_A16:.2f}", flush=True)
    print(f"math:    B={acc_math_B:.2f} A0={acc_math_A0:.2f} A16={acc_math_A16:.2f}", flush=True)
    print(f"frontload_ratio={frontload_ratio:.3f}  off_recovery={off_recovery:.3f}  "
          f"on_retention={on_retention:.3f}  target_score={target_score}", flush=True)
    print(f"preconditions_ok={preconditions_ok}", flush=True)
    print(f"VERDICT: {verdict}  all_pass={all_pass}  is_smoke={is_smoke}", flush=True)
    print(f"Wrote {out_path}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
