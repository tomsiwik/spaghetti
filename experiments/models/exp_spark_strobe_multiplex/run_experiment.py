#!/usr/bin/env python3
"""exp_spark_strobe_multiplex — blind round-robin token-level strobing of 3 real
domain adapters across decode steps vs a MAGNITUDE-MATCHED static 1/N
((1/N) Σ B@A) composition.

K2299 (pre-registered, MATH.md): on a 51-item mixed math/code/medical eval,
content-BLIND round-robin strobing across {math,code,medical} adapters does NOT
beat the magnitude-matched static (1/N) Σ composition by >= +4pp aggregate
accuracy  =>  killed.

The gating comparison is STROBE vs STATIC_NORM (= base + SCALE*(1/N) Σ_i (xA_i)B_i).
STATIC_NORM injects, per step, a residual whose per-adapter contribution is run at
SCALE/N rather than SCALE, so STROBE (one adapter at full SCALE) and STATIC_NORM
(N adapters at SCALE/N) carry the SAME total residual budget per step. This
isolates SIMULTANEITY (deltas coexisting in one matmul) from total residual
MAGNITUDE. The raw-sum STATIC (all N at full SCALE) is also recorded as context
only — it is magnitude-confounded and does NOT gate the verdict.

Real: frozen mlx-community/gemma-4-e4b-it-4bit + the three trained LoRA
safetensors under data/adapters/{math,python,medical}. No mocks.

SMOKE_TEST=1 uses 2 items/domain for a wiring check (is_smoke:true => provisional).
"""

import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate
from safetensors import safe_open

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
ADAPTER_DIR = REPO_ROOT / "data" / "adapters"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
DOMAINS = ["math", "python", "medical"]   # adapter order == clock phase order
N_ADAPT = len(DOMAINS)
RANK = 6
SCALE = 6.0
SEED = 42
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_PER_DOMAIN = 2 if IS_SMOKE else 17       # 17*3 = 51 mixed items
DELTA_PP_THRESHOLD = 4.0                    # K2299 refutation threshold


def log(m):
    print(m, flush=True)


# ───────────────────────── global decode clock ──────────────────────────────
# A single mutable phase shared by all 42 q_proj wrappers. Advanced once per
# generated token by a logits-processor hook (content-BLIND: depends only on the
# step index, never on the token). MODE in {"off","static","strobe"}.
CLOCK = {"phase": 0, "mode": "off"}


def reset_clock(mode):
    CLOCK["phase"] = 0
    CLOCK["mode"] = mode


def make_advance_processor():
    """logits_processor: runs once per decode step (after the forward). It only
    advances the clock — does not touch logits."""
    def proc(tokens, logits):
        CLOCK["phase"] = (CLOCK["phase"] + 1) % N_ADAPT
        return logits
    return proc


# ───────────────────────── multiplexed q_proj wrapper ───────────────────────
class StrobeLinear(nn.Module):
    """Wraps a (quantized) q_proj. Holds all N adapters as (A_i, B_i).

      off         : base only
      static      : base + SCALE * Σ_i (x A_i) B_i              (raw sum, CONTEXT only)
      static_norm : base + SCALE * (1/N) Σ_i (x A_i) B_i        (magnitude-matched 1/N)
      strobe      : base + SCALE * (x A_k) B_k, k = CLOCK.phase  (one adapter/step)

    static_norm and strobe carry the SAME total residual budget per step
    (N adapters at SCALE/N vs one adapter at SCALE), so STROBE vs STATIC_NORM
    isolates simultaneity from magnitude. The raw `static` is 3x over-driven and
    only recorded for context.
    """

    def __init__(self, base, a_list, b_list):
        super().__init__()
        self.base = base
        # A_i: (d_in, r), B_i: (r, d_out)
        self.A = a_list   # list[mx.array]
        self.B = b_list

    def __call__(self, x):
        y = self.base(x)
        mode = CLOCK["mode"]
        if mode == "off":
            return y
        if mode == "static" or mode == "static_norm":
            acc = (x @ self.A[0]) @ self.B[0]
            for i in range(1, N_ADAPT):
                acc = acc + (x @ self.A[i]) @ self.B[i]
            scale = SCALE / N_ADAPT if mode == "static_norm" else SCALE
            return y + (scale * acc).astype(x.dtype)
        # strobe
        k = CLOCK["phase"]
        return y + (SCALE * ((x @ self.A[k]) @ self.B[k])).astype(x.dtype)


def _layers(model):
    return model.language_model.model.layers


def load_adapter_tensors():
    """Returns per-layer-index dict: layer -> (A_list, B_list) across domains."""
    per_domain = {}
    for d in DOMAINS:
        p = ADAPTER_DIR / d / "adapters.safetensors"
        t = {}
        with safe_open(str(p), framework="numpy") as f:
            for key in f.keys():
                if key.endswith(".lora_a") or key.endswith(".lora_b"):
                    t[key] = mx.array(f.get_tensor(key))
        per_domain[d] = t
    return per_domain


def inject_strobe(model):
    per_domain = load_adapter_tensors()
    n_wrapped = 0
    for li, layer in enumerate(_layers(model)):
        prefix = f"language_model.model.layers.{li}.self_attn.q_proj"
        a_list, b_list = [], []
        for d in DOMAINS:
            a = per_domain[d][f"{prefix}.lora_a"]
            b = per_domain[d][f"{prefix}.lora_b"]
            a_list.append(a)
            b_list.append(b)
        layer.self_attn.q_proj = StrobeLinear(layer.self_attn.q_proj, a_list, b_list)
        n_wrapped += 1
    # sanity: deltas are non-trivial and adapters differ
    p0 = "language_model.model.layers.0.self_attn.q_proj"
    norms = {d: float(mx.linalg.norm(per_domain[d][f"{p0}.lora_b"]).item()) for d in DOMAINS}
    log(f"[inject] wrapped {n_wrapped} q_proj layers; layer0 B-norms={norms}")
    assert n_wrapped == 42, f"expected 42 wraps, got {n_wrapped}"
    assert all(v > 1e-4 for v in norms.values()), "an adapter B is ~zero (untrained?)"
    return per_domain


# ───────────────────────── generation helper ────────────────────────────────
def gen(model, tokenizer, prompt, max_tokens, mode):
    reset_clock(mode)
    procs = [make_advance_processor()] if mode == "strobe" else None
    # phase starts at 0; for strobe each decode step uses phase then proc bumps it.
    return mlx_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
        verbose=False, logits_processors=procs,
    )


def fmt(tokenizer, user):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
    )


# ───────────────────────── per-domain eval slices ───────────────────────────
def gsm8k_items(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=SEED).select(range(n))
    out = []
    for ex in ds:
        gt = re.search(r"####\s*([\d,\-\.]+)", ex["answer"]).group(1).replace(",", "").strip()
        out.append((f"Solve step by step.\n\n{ex['question']}\n\nAnswer:", gt))
    return out


def score_gsm8k(response, gt):
    m = re.search(r"####\s*([\d,\-\.]+)", response)
    if m and m.group(1).replace(",", "").strip() == gt:
        return True
    nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
    return bool(nums) and nums[-1] == gt


def humaneval_items(n):
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test").select(range(n))
    return [(f"Complete this Python function:\n\n```python\n{ex['prompt']}\n```\n\n"
             f"Respond with only the function body.",
             (ex["prompt"], ex["test"], ex["entry_point"])) for ex in ds]


def score_humaneval(response, meta):
    prompt, test, entry = meta
    cm = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    completion = cm.group(1) if cm else response
    full = prompt + completion + "\n\n" + test + f"\n\ncheck({entry})\n"
    try:
        r = subprocess.run([sys.executable, "-c", full], timeout=10,
                           capture_output=True, text=True)
        return r.returncode == 0
    except Exception:
        return False


def medqa_items(n):
    from datasets import load_dataset
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test").shuffle(seed=SEED).select(range(n))
    out = []
    for ex in ds:
        o = ex["options"]
        q = (f"{ex['question']}\n(A) {o['A']}\n(B) {o['B']}\n(C) {o['C']}\n(D) {o['D']}")
        out.append((f"Answer with only the letter (A/B/C/D).\n\n{q}", ex["answer_idx"]))
    return out


def score_medqa(response, gt):
    pred = response.strip().upper()
    letter = next((L for L in "ABCD" if pred.startswith(L)), None)
    if not letter:
        m = re.search(r"\b([ABCD])\b", pred)
        letter = m.group(1) if m else None
    return letter == gt


DOMAIN_EVAL = {
    "math":    (gsm8k_items,     score_gsm8k,     1024),
    "python":  (humaneval_items, score_humaneval, 512),
    "medical": (medqa_items,     score_medqa,     20),
}


def build_eval():
    items = []  # (domain, prompt_user, gt, scorer, max_tokens)
    for d in DOMAINS:
        builder, scorer, mt = DOMAIN_EVAL[d]
        for user, gt in builder(N_PER_DOMAIN):
            items.append((d, user, gt, scorer, mt))
    return items


def run_condition(model, tokenizer, items, mode):
    per_domain_correct = {d: 0 for d in DOMAINS}
    per_domain_total = {d: 0 for d in DOMAINS}
    for d, user, gt, scorer, mt in items:
        prompt = fmt(tokenizer, user)
        resp = gen(model, tokenizer, prompt, mt, mode)
        ok = scorer(resp, gt)
        per_domain_total[d] += 1
        if ok:
            per_domain_correct[d] += 1
    total = sum(per_domain_total.values())
    correct = sum(per_domain_correct.values())
    agg = correct / total * 100
    per = {d: per_domain_correct[d] / per_domain_total[d] * 100 for d in DOMAINS}
    return {"aggregate_pp": agg, "correct": correct, "total": total, "per_domain_pp": per}


def main():
    t0 = time.time()
    log(f"[load] {MODEL_ID}  smoke={IS_SMOKE}  n_per_domain={N_PER_DOMAIN}")
    model, tokenizer = mlx_load(MODEL_ID)
    inject_strobe(model)
    mx.eval(model.parameters())

    items = build_eval()
    log(f"[eval] {len(items)} mixed items "
        f"({N_PER_DOMAIN}/domain x {N_ADAPT})")

    # Reference single-adapter ceilings (each domain's own adapter on its own slice):
    # implemented by forcing strobe phase constant — we run a 'single' mode per domain.
    # Magnitude-matched gating baseline (true 1/N): per-step residual budget == strobe.
    static_norm = run_condition(model, tokenizer, items, "static_norm")
    log(f"[static_norm] agg={static_norm['aggregate_pp']:.2f}pp per={static_norm['per_domain_pp']}")
    # Raw-sum static: magnitude-confounded, recorded for context only (does NOT gate).
    static = run_condition(model, tokenizer, items, "static")
    log(f"[static_raw] agg={static['aggregate_pp']:.2f}pp per={static['per_domain_pp']}")
    strobe = run_condition(model, tokenizer, items, "strobe")
    log(f"[strobe] agg={strobe['aggregate_pp']:.2f}pp per={strobe['per_domain_pp']}")

    # GATE: strobe vs MAGNITUDE-MATCHED static_norm — isolates simultaneity.
    delta = strobe["aggregate_pp"] - static_norm["aggregate_pp"]
    delta_vs_raw = strobe["aggregate_pp"] - static["aggregate_pp"]  # context only
    # K2299: strobe must beat magnitude-matched static_norm by >= +4pp, else killed.
    k2299_pass = delta >= DELTA_PP_THRESHOLD
    all_pass = bool(k2299_pass)
    verdict = "supported" if all_pass else "killed"
    if IS_SMOKE:
        verdict = "provisional"

    results = {
        "experiment": "exp_spark_strobe_multiplex",
        "model": MODEL_ID,
        "is_smoke": IS_SMOKE,
        "verdict": verdict,
        "all_pass": all_pass,
        "n_per_domain": N_PER_DOMAIN,
        "n_total_items": len(items),
        "domains": DOMAINS,
        "rank": RANK,
        "scale": SCALE,
        "delta_pp_threshold": DELTA_PP_THRESHOLD,
        "gating_baseline": "static_norm",
        "static_norm": static_norm,
        "static_raw": static,
        "strobe": strobe,
        "strobe_minus_static_norm_pp": delta,
        "strobe_minus_static_raw_pp": delta_vs_raw,
        "kill_criteria": {
            "K2299": {
                "text": "blind round-robin strobing does NOT beat magnitude-matched "
                        "static (1/N) Σ by >=+4pp => killed",
                "gating_baseline": "static_norm",
                "static_norm_agg_pp": static_norm["aggregate_pp"],
                "static_raw_agg_pp": static["aggregate_pp"],
                "strobe_agg_pp": strobe["aggregate_pp"],
                "delta_pp": delta,
                "delta_vs_raw_pp": delta_vs_raw,
                "threshold_pp": DELTA_PP_THRESHOLD,
                "pass": bool(k2299_pass),
            }
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"[done] verdict={verdict} delta_vs_static_norm={delta:+.2f}pp "
        f"(vs_raw={delta_vs_raw:+.2f}pp) elapsed={results['elapsed_sec']}s -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
