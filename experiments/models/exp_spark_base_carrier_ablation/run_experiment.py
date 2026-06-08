"""exp_spark_base_carrier_ablation — is the frozen 4-bit base q_proj a carrier wave or load-bearing?

Pre-registered in MATH.md:
  K1 (target, behavioral, DB kill-id 2294):
     R_mean(0) = mean_d[ A_d(0)/A_d(1) ] < 0.80  OR  any R_d(0) < 0.65  ==> KILLED.
  K2 (validity guard): adapter at alpha=1 must beat chance by >=0.10 on >=2/3 domains,
     else PROVISIONAL (R undefined against a dead reference) — NOT a kill.

Mechanism (★ in MATH.md §1):  q = alpha * W_q x + s * (x @ A) @ B^T   inside the 42 q_proj layers
only; everything else (k/v/o/MLP/embed/lm_head) is the untouched frozen base.

Real MLX, no mocks: real mlx-community/gemma-4-e4b-it-4bit, real safetensors q_proj r6 adapters
from exp_composition_residual_analysis, real greedy decoding, real held-out accuracy.
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

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parents[2]
DATA_ROOT = REPO / "experiments" / "models" / "exp_p1_t2_single_domain_training" / "data"
ADAPTER_DIR = REPO / "experiments" / "models" / "exp_composition_residual_analysis"

BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
LORA_SCALE = 6.0  # <= 8 (mem-antipattern-unsafe-scale); matches the recipe the adapters were trained with
ALPHAS = [1.0, 0.5, 0.25, 0.0]

# Held-out slice sizes from the pre-registered KC.
N_PER_DOMAIN = {"math": 50, "code": 25, "medical": 50}
MAX_NEW = {"math": 256, "code": 96, "medical": 8}
CHANCE = {"math": 0.0, "code": 0.0, "medical": 0.25}

DOMAIN_ADAPTER = {
    "math": ADAPTER_DIR / "adapter_math.safetensors",
    "code": ADAPTER_DIR / "adapter_code.safetensors",
    "medical": ADAPTER_DIR / "adapter_medical.safetensors",
}

K1_RMEAN_THRESH = 0.80
K1_PERDOMAIN_THRESH = 0.65
K2_MARGIN = 0.10

EOS_IDS = None  # filled after tokenizer load


# ---------------------------------------------------------------------------
# Base-attenuating LoRA wrapper.  Installed via setattr on the parent module
# (mem-antipattern-call-override-silent-bypass: NEVER patch __call__ on an instance).
# ---------------------------------------------------------------------------

class AttenuatedLoRAQProj(nn.Module):
    """Wraps a frozen QuantizedLinear q_proj:  out = alpha * base(x) + scale * (x @ A) @ B^T.

    `alpha` is a plain python float read at call time so a single attribute flip
    re-evaluates the whole model at the new attenuation with no reload.
    A, B are the trained LoRA factors (A: (d_in, r), B: (r, d_out)).
    """

    def __init__(self, base: nn.Module, lora_a: mx.array, lora_b: mx.array, scale: float):
        super().__init__()
        self.base = base                 # frozen 4-bit QuantizedLinear (registered submodule)
        self.lora_a = lora_a             # (d_in, r)
        self.lora_b = lora_b             # (r, d_out)
        self.scale = float(scale)
        self.alpha = 1.0                 # mutated externally per alpha-point

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)                              # (..., d_out)
        delta = (x @ self.lora_a) @ self.lora_b              # (..., d_out)
        return self.alpha * base_out + self.scale * delta


def install_adapter(model, adapter_path: Path) -> List[AttenuatedLoRAQProj]:
    """Load one domain adapter's q_proj r6 factors and wrap every q_proj. Returns wrappers."""
    weights = mx.load(str(adapter_path))
    layers = model.language_model.model.layers
    wrappers: List[AttenuatedLoRAQProj] = []
    for li, layer in enumerate(layers):
        attn = layer.self_attn
        ka = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
        kb = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
        if ka not in weights or kb not in weights:
            raise KeyError(f"missing adapter key for layer {li}: {ka}")
        A = weights[ka].astype(mx.float32)   # (2560, 6)
        B = weights[kb].astype(mx.float32)   # (6, 2048)
        wrapper = AttenuatedLoRAQProj(attn.q_proj, A, B, LORA_SCALE)
        setattr(attn, "q_proj", wrapper)     # canonical: replace submodule, not __call__
        wrappers.append(wrapper)
    mx.eval(model.parameters())
    return wrappers


def uninstall_adapter(model, wrappers: List[AttenuatedLoRAQProj]) -> None:
    """Restore raw q_proj (so a fresh adapter can be installed cleanly)."""
    layers = model.language_model.model.layers
    for li, layer in enumerate(layers):
        layer.self_attn.q_proj = wrappers[li].base
    gc.collect()
    mx.clear_cache()


def set_alpha(wrappers: List[AttenuatedLoRAQProj], alpha: float) -> None:
    for w in wrappers:
        w.alpha = float(alpha)


# ---------------------------------------------------------------------------
# Data + scoring
# ---------------------------------------------------------------------------

def load_slice(domain: str, n: int) -> List[Dict[str, str]]:
    path = DATA_ROOT / domain / "valid.jsonl"
    items: List[Dict[str, str]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        msgs = rec["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        gold = next(m["content"] for m in msgs if m["role"] == "assistant")
        items.append({"user": user, "gold": gold})
        if len(items) >= n:
            break
    return items


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _last_number(text: str) -> str | None:
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = _NUM.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def _mcq_letter(text: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else None


def _first_code_line(text: str) -> str | None:
    for ln in text.splitlines():
        s = ln.strip()
        if s:
            return s
    return None


def score(domain: str, gen: str, gold: str) -> bool:
    if domain == "math":
        g, gt = _last_number(gen), _last_number(gold)
        return g is not None and gt is not None and g == gt
    if domain == "medical":
        g, gt = _mcq_letter(gen), _mcq_letter(gold)
        return g is not None and gt is not None and g == gt
    if domain == "code":
        g, gt = _first_code_line(gen), _first_code_line(gold)
        return g is not None and gt is not None and g == gt
    raise ValueError(domain)


# ---------------------------------------------------------------------------
# Greedy decode (per-sample, real)
# ---------------------------------------------------------------------------

def greedy_generate(model, tokenizer, prompt_ids: mx.array, max_new: int) -> str:
    """Greedy decode from a 1-D prompt token array. Returns decoded continuation text."""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    tokens = prompt_ids[None]                      # (1, T)
    logits = model(tokens, cache=cache)            # (1, T, V)
    y = mx.argmax(logits[:, -1, :], axis=-1)       # (1,)
    mx.eval(y)
    out_ids: List[int] = []
    for _ in range(max_new):
        tid = int(y.item())
        if tid in EOS_IDS:
            break
        out_ids.append(tid)
        logits = model(y[None], cache=cache)       # (1,1,V)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
    del cache
    return tokenizer.decode(out_ids)


def build_prompt_ids(tokenizer, user: str) -> mx.array:
    msgs = [{"role": "user", "content": user}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
    return mx.array(ids)


# ---------------------------------------------------------------------------
# Per-domain evaluation across all alphas
# ---------------------------------------------------------------------------

def eval_domain(model, tokenizer, domain: str) -> Dict[float, Dict[str, Any]]:
    n = N_PER_DOMAIN[domain]
    items = load_slice(domain, n)
    print(f"\n=== domain={domain}: {len(items)} held-out items, adapter={DOMAIN_ADAPTER[domain].name} ===", flush=True)

    wrappers = install_adapter(model, DOMAIN_ADAPTER[domain])

    # Pre-tokenize prompts once (reused across alphas).
    prompts = [build_prompt_ids(tokenizer, it["user"]) for it in items]

    out: Dict[float, Dict[str, Any]] = {}
    for alpha in ALPHAS:
        set_alpha(wrappers, alpha)
        t0 = time.time()
        n_correct = 0
        for pi, (pids, it) in enumerate(zip(prompts, items)):
            gen = greedy_generate(model, tokenizer, pids, MAX_NEW[domain])
            if score(domain, gen, it["gold"]):
                n_correct += 1
            if pi % 10 == 9:
                mx.clear_cache()
        acc = n_correct / len(items)
        out[alpha] = {"accuracy": acc, "n_correct": n_correct, "n": len(items)}
        print(f"  alpha={alpha:<4}  acc={acc:.4f}  ({n_correct}/{len(items)})  t={time.time()-t0:.1f}s", flush=True)
        mx.clear_cache()

    uninstall_adapter(model, wrappers)
    del wrappers, prompts, items
    gc.collect()
    mx.clear_cache()
    return out


def main():
    global EOS_IDS
    t_all = time.time()
    print(f"Base model: {BASE_MODEL}", flush=True)
    print(f"Adapters dir: {ADAPTER_DIR}", flush=True)
    print(f"Alphas: {ALPHAS}  LORA_SCALE={LORA_SCALE}", flush=True)
    print(f"Slices: {N_PER_DOMAIN}", flush=True)

    for d, p in DOMAIN_ADAPTER.items():
        assert p.exists(), f"missing adapter {p}"

    model, tokenizer = load(BASE_MODEL)
    model.eval()
    EOS_IDS = set(tokenizer.eos_token_ids) if hasattr(tokenizer, "eos_token_ids") else {tokenizer.eos_token_id}
    print(f"EOS ids: {EOS_IDS}", flush=True)
    mx.eval(model.parameters())
    mx.clear_cache()

    per_domain: Dict[str, Dict[float, Dict[str, Any]]] = {}
    for domain in ["math", "code", "medical"]:
        per_domain[domain] = eval_domain(model, tokenizer, domain)

    # --- aggregate retention ratios ---
    A1 = {d: per_domain[d][1.0]["accuracy"] for d in per_domain}
    A0 = {d: per_domain[d][0.0]["accuracy"] for d in per_domain}

    def ratio(d):
        return (A0[d] / A1[d]) if A1[d] > 1e-9 else float("nan")

    R_d0 = {d: ratio(d) for d in per_domain}
    valid_ratios = [v for v in R_d0.values() if not math.isnan(v)]
    R_mean0 = sum(valid_ratios) / len(valid_ratios) if valid_ratios else float("nan")

    # --- K2 validity guard ---
    beats_chance = {d: (A1[d] - CHANCE[d]) >= K2_MARGIN for d in per_domain}
    n_valid_ref = sum(beats_chance.values())
    k2_pass = n_valid_ref >= 2

    # --- K1 evaluation ---
    any_domain_below = any((not math.isnan(R_d0[d])) and R_d0[d] < K1_PERDOMAIN_THRESH for d in per_domain)
    rmean_below = (not math.isnan(R_mean0)) and R_mean0 < K1_RMEAN_THRESH
    k1_criterion_met = rmean_below or any_domain_below  # criterion met == KILLED

    if not k2_pass:
        verdict = "PROVISIONAL"
        k1_result = "invalid"
    elif k1_criterion_met:
        verdict = "KILLED"
        k1_result = "fail"   # carrier hypothesis refuted
    else:
        verdict = "SUPPORTED"
        k1_result = "pass"   # carrier hypothesis supported

    results = {
        "experiment_id": "exp_spark_base_carrier_ablation",
        "config": {
            "base_model": BASE_MODEL,
            "lora_scale": LORA_SCALE,
            "alphas": ALPHAS,
            "n_per_domain": N_PER_DOMAIN,
            "max_new": MAX_NEW,
            "chance": CHANCE,
            "adapters": {d: str(p) for d, p in DOMAIN_ADAPTER.items()},
            "k1_rmean_thresh": K1_RMEAN_THRESH,
            "k1_perdomain_thresh": K1_PERDOMAIN_THRESH,
            "k2_margin": K2_MARGIN,
        },
        "accuracy_by_alpha": {
            d: {str(a): per_domain[d][a]["accuracy"] for a in ALPHAS} for d in per_domain
        },
        "raw_by_alpha": {
            d: {str(a): per_domain[d][a] for a in ALPHAS} for d in per_domain
        },
        "A_d_alpha1": A1,
        "A_d_alpha0": A0,
        "R_d_0": R_d0,
        "R_mean_0": R_mean0,
        "k2_beats_chance": beats_chance,
        "kill_criteria": {
            "2294": {
                "text": "R_mean(0) < 0.80 OR any R_d(0) < 0.65 on GSM8K(50)/HumanEval(25)/MedQA(50) on-domain accuracy",
                "R_mean_0": R_mean0,
                "R_d_0": R_d0,
                "rmean_below_0.80": rmean_below,
                "any_domain_below_0.65": any_domain_below,
                "result": k1_result,
                "type": "target_behavioral",
            },
            "K2_validity": {
                "text": "adapter at alpha=1 beats chance by >=0.10 on >=2/3 domains",
                "n_valid_ref": n_valid_ref,
                "result": "pass" if k2_pass else "fail",
                "type": "validity_guard",
            },
        },
        "verdict": verdict,
        "all_pass": verdict == "SUPPORTED",
        "is_smoke": False,
        "total_wall_clock_sec": time.time() - t_all,
    }

    out_path = EXP_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nWrote {out_path}", flush=True)
    print(f"A_d(1): {A1}", flush=True)
    print(f"A_d(0): {A0}", flush=True)
    print(f"R_d(0): {R_d0}", flush=True)
    print(f"R_mean(0): {R_mean0}", flush=True)
    print(f"K2 valid-ref domains: {n_valid_ref}/3  -> {'pass' if k2_pass else 'fail'}", flush=True)
    print(f"Verdict: {verdict}  all_pass={results['all_pass']}", flush=True)
    print(f"Total: {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
