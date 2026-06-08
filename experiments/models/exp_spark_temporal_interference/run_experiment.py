"""exp_spark_temporal_interference — is off-domain LoRA interference TEMPORAL?

Tests whether zeroing the off-domain (code) adapter only at the top-5% highest
BASE-entropy decode steps recovers most of the on-domain (GSM8K) accuracy lost to
composition. 5 arms, same prompts/seed/decoding. NO mocks. Real Gemma-4 + real adapters.

KCs (MATH.md §5):
  K1 (2288, target): recov(D) = (acc_D-acc_C)/(acc_B-acc_C) >= 0.50  -> PASS (hypothesis survives)
  K2 (2289, target, control): acc_D - acc_E >= 2pp                   -> PASS (not just dropout)
  K3 (2290, structural): realized gated fraction <= 0.05             -> PASS (concentrated)
  all_pass = K1 & K2 & K3.  SUPPORTED iff all_pass else KILLED.
"""
from __future__ import annotations

import gc
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parents[2]

BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
MATH_ADAPTER = REPO / "data" / "adapters" / "math" / "adapters.safetensors"
CODE_ADAPTER = REPO / "experiments" / "models" / "exp_composition_residual_analysis" / "adapter_code.safetensors"
GSM8K = REPO / "experiments" / "models" / "exp_p9_ttlora_polar_hybrid" / "data" / "gsm8k_test.jsonl"

SCALE = 6.0           # adapter training scale, <= 8
N_EVAL = 80           # GSM8K items (>=50); 5 arms incl double-forward arm C
MAX_TOKENS = 512      # decode budget per question (thinking mode needs room)
TOP_Q = 0.05          # top-5% base-entropy gate
SEED = 42

K1_THRESH = 0.50
K2_THRESH_PP = 2.0
K3_THRESH = 0.05


# ---------------------------------------------------------------------------
# Gated q_proj wrapper (canonical: subclass nn.Module + setattr, never override __call__)
# Composition: y = base(x) + scale*(x@mA)@mB + scale*code_gate*(x@cA)@cB   (Σ Bᵢ@Aᵢ form)
# ---------------------------------------------------------------------------
class GatedQProj(nn.Module):
    def __init__(self, base, mA, mB, cA, cB, scale):
        super().__init__()
        self.base = base
        self.mA, self.mB, self.cA, self.cB = mA, mB, cA, cB
        self.scale = scale
        self.math_gate = 1.0   # python scalars (mutable, not params -> no grad)
        self.code_gate = 1.0

    def __call__(self, x):
        y = self.base(x)
        xf = x.astype(self.mA.dtype)
        if self.math_gate != 0.0:
            y = y + (self.scale * self.math_gate) * ((xf @ self.mA) @ self.mB).astype(y.dtype)
        if self.code_gate != 0.0:
            y = y + (self.scale * self.code_gate) * ((xf @ self.cA) @ self.cB).astype(y.dtype)
        return y


def attach_gated(model, math_w, code_w, scale):
    layers = model.language_model.model.layers
    wrappers = []
    for L in range(len(layers)):
        attn = layers[L].self_attn
        pre = f"language_model.model.layers.{L}.self_attn.q_proj."
        g = GatedQProj(
            attn.q_proj,
            math_w[pre + "lora_a"], math_w[pre + "lora_b"],
            code_w[pre + "lora_a"], code_w[pre + "lora_b"],
            scale,
        )
        attn.q_proj = g
        wrappers.append(g)
    return wrappers


def set_gates(wrappers, math_gate, code_gate):
    for g in wrappers:
        g.math_gate = math_gate
        g.code_gate = code_gate


# ---------------------------------------------------------------------------
# Data + extraction
# ---------------------------------------------------------------------------
def load_gsm8k(n: int) -> List[Dict[str, str]]:
    out = []
    with open(GSM8K) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def gt_answer(ans: str) -> str | None:
    m = re.search(r"####\s*([\d,\-\.]+)", ans)
    return m.group(1).replace(",", "").strip() if m else None


def extract_pred(text: str) -> str | None:
    m = re.search(r"####\s*([\d,\-\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


def is_correct(text: str, gt: str) -> bool:
    p = extract_pred(text)
    if p is None:
        return False
    try:
        return abs(float(p) - float(gt)) < 1e-4
    except ValueError:
        return p == gt


# ---------------------------------------------------------------------------
# Decode primitives
# ---------------------------------------------------------------------------
def shannon_entropy_nats(logits_1v: mx.array) -> mx.array:
    # logits_1v shape (V,). entropy of softmax in nats.
    lse = mx.logsumexp(logits_1v)
    logp = logits_1v - lse
    p = mx.exp(logp)
    return -mx.sum(p * logp)


def decode_plain(model, eos_ids, prompt_ids: mx.array, max_tokens: int) -> List[int]:
    """Greedy argmax decode, no per-step gating control. Returns generated token ids."""
    cache = make_prompt_cache(model)
    logits = model(prompt_ids[None], cache=cache)[0, -1]
    out = []
    for _ in range(max_tokens):
        tok = int(mx.argmax(logits).item())
        if tok in eos_ids:
            break
        out.append(tok)
        logits = model(mx.array([[tok]]), cache=cache)[0, -1]
        mx.eval(logits)
    return out


def decode_compose_with_base_entropy(model, wrappers, base_model, base_eos, eos_ids,
                                     prompt_ids: mx.array, max_tokens: int
                                     ) -> Tuple[List[int], List[float]]:
    """Arm C: math+code all steps. At each generated step record the FROZEN BASE entropy
    over the same prefix (base_model fed the identical token stream, its own cache, no adapters).
    Returns (generated tokens, per-step base entropy list aligned to generated tokens)."""
    set_gates(wrappers, 1.0, 1.0)
    cache = make_prompt_cache(model)
    bcache = make_prompt_cache(base_model)

    logits = model(prompt_ids[None], cache=cache)[0, -1]
    blogits = base_model(prompt_ids[None], cache=bcache)[0, -1]
    out, ents = [], []
    for _ in range(max_tokens):
        # base entropy at the CURRENT prefix (decision point for the next token)
        ent = float(shannon_entropy_nats(blogits).item())
        tok = int(mx.argmax(logits).item())
        if tok in eos_ids:
            break
        out.append(tok)
        ents.append(ent)
        logits = model(mx.array([[tok]]), cache=cache)[0, -1]
        blogits = base_model(mx.array([[tok]]), cache=bcache)[0, -1]
        mx.eval(logits, blogits)
    return out, ents


def decode_compose_gated(model, wrappers, eos_ids, prompt_ids: mx.array,
                         max_tokens: int, gated_steps: set) -> List[int]:
    """Re-decode with math always on; code gated to 0 at step indices in gated_steps.
    Step index = position in the generated sequence (0-based), matching arm C's profile."""
    cache = make_prompt_cache(model)
    set_gates(wrappers, 1.0, 1.0)
    logits = model(prompt_ids[None], cache=cache)[0, -1]
    out = []
    step = 0
    for _ in range(max_tokens):
        # gate decision for THIS step (producing token at position `step`)
        for g in wrappers:
            g.math_gate = 1.0
            g.code_gate = 0.0 if step in gated_steps else 1.0
        tok = int(mx.argmax(logits).item())
        if tok in eos_ids:
            break
        out.append(tok)
        logits = model(mx.array([[tok]]), cache=cache)[0, -1]
        mx.eval(logits)
        step += 1
    return out


# ---------------------------------------------------------------------------
# Per-question pipeline
# ---------------------------------------------------------------------------
def topk_high_entropy_steps(ents: List[float], q: float) -> set:
    """Per-sequence top-q% by base entropy; ties resolved to stay <= q% (ceil count but capped)."""
    n = len(ents)
    if n == 0:
        return set()
    k = max(1, math.floor(q * n))  # floor keeps fraction <= q
    order = sorted(range(n), key=lambda i: ents[i], reverse=True)
    return set(order[:k])


def main():
    print(f"Base model: {BASE_MODEL}", flush=True)
    print(f"Platform skills invoked: [/mlx-dev, /fast-mlx]", flush=True)
    print(f"Reference: arXiv:2306.01708 (TIES), Finding #827/#666", flush=True)
    print(f"KC count: 3 (K1 target recov>=0.50, K2 target ΔDE>=2pp, K3 struct frac<=0.05)", flush=True)
    print(f"math adapter: {MATH_ADAPTER}", flush=True)
    print(f"code adapter: {CODE_ADAPTER}", flush=True)
    assert MATH_ADAPTER.exists() and CODE_ADAPTER.exists(), "adapter missing"
    mx.random.seed(SEED)
    import random
    random.seed(SEED)

    t_all = time.time()

    # Load weights
    math_w = mx.load(str(MATH_ADAPTER))
    code_w = mx.load(str(CODE_ADAPTER))

    # Adapted model (carries both adapters, gated) + clean base model (for entropy)
    model, tokenizer = load(BASE_MODEL)
    model.freeze()
    wrappers = attach_gated(model, math_w, code_w, SCALE)
    model.eval()

    base_model, _ = load(BASE_MODEL)
    base_model.freeze()
    base_model.eval()

    eos_ids = set(tokenizer.eos_token_ids) if hasattr(tokenizer, "eos_token_ids") else {tokenizer.eos_token_id}
    mx.clear_cache()

    ds = load_gsm8k(N_EVAL)
    # filter to items with a parseable gt
    ds = [ex for ex in ds if gt_answer(ex["answer"]) is not None]
    print(f"\nEval items: {len(ds)}  max_tokens={MAX_TOKENS}  top_q={TOP_Q}", flush=True)

    arms = ["A_base", "B_math", "C_compose", "D_entropy_gate", "E_random_gate"]
    correct = {a: 0 for a in arms}
    gated_fracs = []
    per_item = []

    for i, ex in enumerate(ds):
        gt = gt_answer(ex["answer"])
        messages = [{"role": "user", "content": f"Solve step by step.\n\n{ex['question']}"}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=True
        )
        prompt_ids = mx.array(prompt)

        # Arm A: base (no adapters) -- use the clean base_model
        gA = decode_plain(base_model, eos_ids, prompt_ids, MAX_TOKENS)
        tA = tokenizer.decode(gA)

        # Arm B: math-only
        set_gates(wrappers, 1.0, 0.0)
        gB = decode_plain(model, eos_ids, prompt_ids, MAX_TOKENS)
        tB = tokenizer.decode(gB)

        # Arm C: compose all steps + base-entropy profile
        gC, ents = decode_compose_with_base_entropy(
            model, wrappers, base_model, eos_ids, eos_ids, prompt_ids, MAX_TOKENS
        )
        tC = tokenizer.decode(gC)

        # gate sets
        S_hi = topk_high_entropy_steps(ents, TOP_Q)
        n_steps = len(ents)
        k = len(S_hi)
        # random equal-count gate (positions within the same length)
        S_rand = set(random.sample(range(n_steps), k)) if n_steps > 0 and k > 0 else set()
        frac = (k / n_steps) if n_steps > 0 else 0.0
        gated_fracs.append(frac)

        # Arm D: entropy-gate
        gD = decode_compose_gated(model, wrappers, eos_ids, prompt_ids, MAX_TOKENS, S_hi)
        tD = tokenizer.decode(gD)

        # Arm E: random-gate
        gE = decode_compose_gated(model, wrappers, eos_ids, prompt_ids, MAX_TOKENS, S_rand)
        tE = tokenizer.decode(gE)

        cA = is_correct(tA, gt); cB = is_correct(tB, gt); cC = is_correct(tC, gt)
        cD = is_correct(tD, gt); cE = is_correct(tE, gt)
        correct["A_base"] += cA; correct["B_math"] += cB; correct["C_compose"] += cC
        correct["D_entropy_gate"] += cD; correct["E_random_gate"] += cE

        per_item.append({
            "idx": i, "gt": gt, "n_steps": n_steps, "k_gated": k, "frac": frac,
            "A": cA, "B": cB, "C": cC, "D": cD, "E": cE,
        })

        if (i + 1) % 10 == 0:
            n = i + 1
            print(f"  [{n}/{len(ds)}] "
                  f"A={correct['A_base']/n*100:.1f} B={correct['B_math']/n*100:.1f} "
                  f"C={correct['C_compose']/n*100:.1f} D={correct['D_entropy_gate']/n*100:.1f} "
                  f"E={correct['E_random_gate']/n*100:.1f}  frac={sum(gated_fracs)/n*100:.2f}%",
                  flush=True)
            mx.clear_cache()

    n = len(ds)
    acc = {a: correct[a] / n * 100.0 for a in arms}
    mean_frac = sum(gated_fracs) / n if n else 0.0

    accA, accB, accC = acc["A_base"], acc["B_math"], acc["C_compose"]
    accD, accE = acc["D_entropy_gate"], acc["E_random_gate"]
    bc_drop = accB - accC
    denom = (accB - accC)
    recov_D = (accD - accC) / denom if abs(denom) > 1e-9 else float("nan")
    recov_E = (accE - accC) / denom if abs(denom) > 1e-9 else float("nan")
    delta_DE = accD - accE

    # KCs
    k1_pass = (not math.isnan(recov_D)) and recov_D >= K1_THRESH
    k2_pass = delta_DE >= K2_THRESH_PP
    k3_pass = mean_frac <= K3_THRESH
    all_pass = k1_pass and k2_pass and k3_pass
    premise_ok = bc_drop > 0  # composition actually hurt on-domain

    verdict = "SUPPORTED" if all_pass else "KILLED"

    results = {
        "experiment_id": "exp_spark_temporal_interference",
        "config": {
            "base_model": BASE_MODEL,
            "mlx_lm_version": "0.31.2",
            "math_adapter": str(MATH_ADAPTER),
            "code_adapter": str(CODE_ADAPTER),
            "scale": SCALE, "n_eval": n, "max_tokens": MAX_TOKENS,
            "top_q": TOP_Q, "seed": SEED,
            "targets": ["self_attn.q_proj"], "enable_thinking": True,
            "decode": "greedy_argmax",
        },
        "accuracy_pct": acc,
        "B_minus_C_drop_pp": bc_drop,
        "premise_reproduced_BC_drop_positive": premise_ok,
        "recovery_fraction_D": recov_D,
        "recovery_fraction_E": recov_E,
        "delta_D_minus_E_pp": delta_DE,
        "mean_gated_fraction": mean_frac,
        "kill_criteria": {
            "2288": {"text": "K1 recov(D)>=0.50 (entropy gate recovers >=50% lost acc)",
                     "value": recov_D, "thresh": K1_THRESH, "type": "target_behavioral",
                     "result": "pass" if k1_pass else "fail"},
            "2289": {"text": "K2 acc_D-acc_E>=2pp (beats equal-count random dropout)",
                     "value": delta_DE, "thresh": K2_THRESH_PP, "type": "target_behavioral",
                     "result": "pass" if k2_pass else "fail"},
            "2290": {"text": "K3 gated fraction<=0.05 (temporally concentrated)",
                     "value": mean_frac, "thresh": K3_THRESH, "type": "structural",
                     "result": "pass" if k3_pass else "fail"},
        },
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "per_item": per_item,
        "total_wall_clock_sec": time.time() - t_all,
    }

    out_path = EXP_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== RESULTS ===", flush=True)
    print(f"acc A(base)={accA:.1f} B(math)={accB:.1f} C(compose)={accC:.1f} "
          f"D(entropy)={accD:.1f} E(random)={accE:.1f}", flush=True)
    print(f"B-C drop = {bc_drop:.1f}pp  (premise_reproduced={premise_ok})", flush=True)
    print(f"recov(D)={recov_D:.3f}  recov(E)={recov_E:.3f}  ΔD-E={delta_DE:.1f}pp  frac={mean_frac*100:.2f}%", flush=True)
    print(f"K1 {'PASS' if k1_pass else 'FAIL'}  K2 {'PASS' if k2_pass else 'FAIL'}  K3 {'PASS' if k3_pass else 'FAIL'}", flush=True)
    print(f"VERDICT: {verdict}  all_pass={all_pass}", flush=True)
    print(f"Wrote {out_path}  ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
