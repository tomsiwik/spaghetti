#!/usr/bin/env python3
"""exp_spark_base_fragility_oracle — frozen-base random-noise fragility predicts per-token
composition damage with ZERO adapter information.

HYPOTHESIS: per-token composition interference is a property of the FROZEN BASE geometry, not the
adapters. At mid layer L* take base hidden hₜ, add a FIXED-NORM RANDOM Gaussian ε (NOT any adapter
delta, seeded EPS_SEED), measure base-fragilityₜ = mean_k KL(P_base(hₜ) ‖ P_base(hₜ+ε_k)) over
K seeded draws. Claim: this random-noise probe of the frozen base ALONE ranks which tokens the
(1/N) math+python+medical composed adapter damages most, where
    damageₜ = base_logprob(y_{t+1}) − composed_logprob(y_{t+1}).

PRE-REGISTERED KILL 2305 (verbatim): "Spearman rho < 0.30 between base-fragility and per-token
interference damage, OR top-decile-damage AUC < 0.62".

The predictor uses ZERO adapter bytes: ε is random Gaussian of fixed relative norm, seeded.
Composition is Σᵢ (1/N) sᵢ Bᵢ Aᵢ, never (ΣB)(ΣA). LORA_SCALE=6.0 ≤ 8.
NO MOCKS. Real frozen gemma-4-e4b-it-4bit + real math/python/medical adapters. is_smoke=False.
"""

import gc
import json
import math
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models import gemma4_text

device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 6 * 1024**3)

EXP_DIR = Path(__file__).resolve().parent
RESULTS_FILE = EXP_DIR / "results.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
ADAPTER_ROOT = Path("/Users/tom/Code/tomsiwik/llm/data/adapters")
DOMAINS = ["math", "python", "medical"]            # N = 3 (1/N) average merge

LORA_SCALE = 6.0          # <= 8 guard OK (matches adapter training scale)
N_LAYERS_EXPECTED = 42
MID_LAYER = 21            # L*, mid of 42-layer stack
N_ADAPTERS = len(DOMAINS)

# Fragility probe (ZERO adapter info)
EPS_SEED = 20250609       # FIXED seed for ε — no Math.random, no Date, no time-derived seed
N_NOISE_DRAWS = 8         # K: average KL over K seeded random draws (kills single-draw artifact)
NOISE_FRAC = 0.08         # ρ: ‖ε‖ = ρ · ‖hₜ‖  (fixed-norm relative perturbation)

# Held-out eval text
N_PROMPTS_PER_DOMAIN = 16
MAX_TOKENS_PER_PROMPT = 96  # cap tokens scored per prompt for memory
MIN_TOKENS = 200            # sanity: need enough tokens for a meaningful Spearman

# Pre-registered thresholds (kill 2305)
RHO_FLOOR = 0.30
AUC_FLOOR = 0.62


def log(msg):
    print(msg, flush=True)


def log_mem(label=""):
    log(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB "
        f"cache={mx.get_cache_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")


# ----------------------------------------------------------------------------
# Held-out domain text (real datasets the adapters trained on; disjoint items)
# ----------------------------------------------------------------------------

def load_eval_prompts():
    """Return list of (domain, text) plain strings to score. Real HF datasets."""
    from datasets import load_dataset
    items = []

    # math — gsm8k (use tail items, disjoint from the 2000 head used in training)
    ds = load_dataset("gsm8k", "main", split="test")
    for i in range(N_PROMPTS_PER_DOMAIN):
        it = ds[i]
        items.append(("math", it["question"].strip() + "\n" + it["answer"].strip()))

    # python — code_alpaca (instruction + output)
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    picked = 0
    for i in range(len(ds)):
        it = ds[i]
        out = (it.get("output") or "").strip()
        if "def " in out or "return" in out:
            txt = (it["instruction"].strip() + "\n" + (it.get("input") or "").strip()
                   + "\n" + out).strip()
            items.append(("python", txt))
            picked += 1
            if picked >= N_PROMPTS_PER_DOMAIN:
                break

    # medical — medmcqa (question + correct option)
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    opts = ["opa", "opb", "opc", "opd"]
    for i in range(N_PROMPTS_PER_DOMAIN):
        it = ds[i]
        ans = it[opts[int(it["cop"])]]
        txt = f"{it['question'].strip()}\nAnswer: {ans.strip()}. {(it.get('exp') or '').strip()}"
        items.append(("medical", txt.strip()))

    log(f"  Loaded {len(items)} held-out domain texts "
        f"({N_PROMPTS_PER_DOMAIN} each: math/python/medical)")
    return items


def encode_capped(tokenizer, text):
    ids = tokenizer.encode(text)
    if len(ids) > MAX_TOKENS_PER_PROMPT:
        ids = ids[:MAX_TOKENS_PER_PROMPT]
    return ids


# ----------------------------------------------------------------------------
# Tap/inject wrapper on the mid layer L* (drift-proof: subclass, captures real h,
# adds a controllable additive delta to the layer OUTPUT). Lets us realize both
# P_base(h) (delta=None) and P_base(h+ε) (delta=ε) with full forwards through the
# IDENTICAL downstream stack + norm + tied head + softcap.
# ----------------------------------------------------------------------------

class TapInjectLayer(gemma4_text.DecoderLayer):
    """Wraps a real DecoderLayer; passthrough except: caches output h (tap) and adds
    self._inject (broadcastable to h) to the returned hidden state when set."""

    @classmethod
    def wrap(cls, layer):
        layer.__class__ = cls          # rebind class in place; keeps all submodules/params
        layer._inject = None
        layer._captured = None
        return layer

    def __call__(self, x, mask=None, cache=None, per_layer_input=None,
                 shared_kv=None, offset=None):
        h, shared_kv, offset = gemma4_text.DecoderLayer.__call__(
            self, x, mask, cache, per_layer_input, shared_kv, offset)
        self._captured = h             # (1, seq, d) base output at L*
        if self._inject is not None:
            h = h + self._inject
        return h, shared_kv, offset


# ----------------------------------------------------------------------------
# (1/N) composed q_proj wrapper:  y = W h + (1/N) Σᵢ sᵢ (h@Aᵢ)@Bᵢ
# subclass nn.Module + setattr (never __call__ override on instance, F#831)
# Composition is Σᵢ Bᵢ Aᵢ, NEVER (ΣB)(ΣA).
# ----------------------------------------------------------------------------

class AvgComposedQProj(nn.Module):
    def __init__(self, base_linear, a_list, b_list, scale, inv_n):
        super().__init__()
        self.linear = base_linear
        self.a_list = a_list           # list of (in, r)
        self.b_list = b_list           # list of (r, out)
        self.scale = scale
        self.inv_n = inv_n
        self.linear.freeze()

    def __call__(self, x):
        y = self.linear(x)
        acc = None
        for a, b in zip(self.a_list, self.b_list):
            d = (x @ a) @ b            # independent delta_i = B_i A_i h
            acc = d if acc is None else (acc + d)
        y = y + (self.scale * self.inv_n * acc).astype(x.dtype)
        return y


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def attach_avg_composed(model, adapters, scale, inv_n):
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        ak = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
        bk = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
        if ak not in adapters[0]:
            continue
        a_list = [ad[ak].astype(mx.float32) for ad in adapters]
        b_list = [ad[bk].astype(mx.float32) for ad in adapters]
        wrapper = AvgComposedQProj(layer.self_attn.q_proj, a_list, b_list, scale, inv_n)
        setattr(layer.self_attn, "q_proj", wrapper)
        count += 1
    mx.eval(model.parameters())
    assert count == N_LAYERS_EXPECTED, f"expected {N_LAYERS_EXPECTED} wrapped, got {count}"
    log(f"  Attached {count} AvgComposedQProj (N={1/inv_n:.0f}, scale={scale}, 1/N merge)")
    return model


# ----------------------------------------------------------------------------
# Per-token log-prob of the TRUE next token, full-sequence single forward.
# ----------------------------------------------------------------------------

def token_logprobs(model, ids_row):
    """ids_row: (1, T) int. Returns logprob of true next token at positions 0..T-2 → (T-1,)."""
    logits = model(ids_row)                       # (1, T, V)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    targets = ids_row[0, 1:]                       # (T-1,)
    idx = mx.arange(targets.shape[0])
    lp = logp[0, :-1, :][idx, targets]             # (T-1,)
    mx.eval(lp)
    return lp


def kl_base_vs_perturbed(logits_base, logits_pert):
    """KL(P_base ‖ P_pert) per position. logits_*: (1, T, V) → (T,)."""
    lp_b = logits_base - mx.logsumexp(logits_base, axis=-1, keepdims=True)
    lp_p = logits_pert - mx.logsumexp(logits_pert, axis=-1, keepdims=True)
    p_b = mx.exp(lp_b)
    kl = (p_b * (lp_b - lp_p)).sum(axis=-1)[0]     # (T,)
    mx.eval(kl)
    return kl


# ----------------------------------------------------------------------------
# Stats: Spearman rho + top-decile AUC (no scipy/sklearn dependency)
# ----------------------------------------------------------------------------

def rankdata(x):
    n = len(x)
    order = sorted(range(n), key=lambda i: x[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    n = len(a)
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((rb[i] - mb) ** 2 for i in range(n)))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def auc_top_decile(scores, labels):
    """AUC of `scores` (fragility) ranking the binary `labels` (top-decile damage = 1).
    Mann-Whitney U / (n_pos n_neg). Higher score should rank positives higher."""
    ranks = rankdata(scores)
    pos = [ranks[i] for i in range(len(labels)) if labels[i] == 1]
    n_pos = len(pos)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n_pos, n_neg
    sum_ranks_pos = sum(pos)
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg), n_pos, n_neg


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    t0 = time.time()
    log("=" * 72)
    log("exp_spark_base_fragility_oracle")
    log(f"Base: {MODEL_ID}  | L*={MID_LAYER} | K={N_NOISE_DRAWS} draws | rho={NOISE_FRAC}")
    log(f"EPS_SEED={EPS_SEED}  (fixed; no Math.random/Date)")
    log(f"Composition: (1/{N_ADAPTERS}) Σ s·BᵢAᵢ  domains={DOMAINS}  scale={LORA_SCALE}")
    log("=" * 72)

    adapters = []
    for d in DOMAINS:
        p = ADAPTER_ROOT / d / "adapters.safetensors"
        assert p.exists(), f"missing {p}"
        adapters.append(mx.load(str(p)))
    log_mem("start")

    prompts = load_eval_prompts()

    # ---------- Phase 1: BASE model — fragility (zero adapter info) + base logprobs ----------
    log("\n=== Phase 1: base fragility + base logprobs ===")
    model, tok = load(MODEL_ID)
    target_layer = TapInjectLayer.wrap(get_lm(model).model.layers[MID_LAYER])
    mx.eval(model.parameters())

    per_tok = []   # dict per scored token: domain, base_lp, fragility
    for pi, (domain, text) in enumerate(prompts):
        ids = encode_capped(tok, text)
        if len(ids) < 4:
            continue
        ids_row = mx.array(ids)[None]              # (1, T)

        # base forward (delta=None): captures h at L*, gives base logits
        target_layer._inject = None
        logits_base = model(ids_row)
        mx.eval(logits_base)
        h = target_layer._captured                  # (1, T, d) base mid-layer output
        h_norm = mx.sqrt((h * h).sum(axis=-1))      # (1, T)
        d_model = h.shape[-1]

        # base per-token logprob of true next token
        logp_b = logits_base - mx.logsumexp(logits_base, axis=-1, keepdims=True)
        targets = ids_row[0, 1:]
        idx = mx.arange(targets.shape[0])
        base_lp = logp_b[0, :-1, :][idx, targets]   # (T-1,)
        mx.eval(base_lp)

        # fragility: mean over K seeded random fixed-norm noise draws
        kl_acc = mx.zeros((h.shape[1],))
        for k in range(N_NOISE_DRAWS):
            mx.random.seed(EPS_SEED + 1000 * pi + k)     # deterministic per (prompt, draw)
            u = mx.random.normal(h.shape)                 # (1, T, d)
            u_norm = mx.sqrt((u * u).sum(axis=-1, keepdims=True))
            u = u / mx.maximum(u_norm, 1e-8)              # unit direction per token
            eps = NOISE_FRAC * h_norm[..., None] * u      # ‖eps‖ = rho·‖h‖
            target_layer._inject = eps
            logits_pert = model(ids_row)
            mx.eval(logits_pert)
            kl_acc = kl_acc + kl_base_vs_perturbed(logits_base, logits_pert)
        target_layer._inject = None
        fragility = kl_acc / N_NOISE_DRAWS              # (T,)
        mx.eval(fragility)

        # align: damage/base_lp are for positions 0..T-2 (predict token t+1).
        # fragility at position t is the curvature of base output at t (predicts token t+1).
        T = ids_row.shape[1]
        for t in range(T - 1):
            per_tok.append({
                "prompt_idx": pi,
                "domain": domain,
                "base_lp": float(base_lp[t].item()),
                "fragility": float(fragility[t].item()),
            })
        if pi % 8 == 0:
            log(f"  [base] prompt {pi}/{len(prompts)} domain={domain} T={T} "
                f"tot_tok={len(per_tok)}")

    del model, tok
    gc.collect(); mx.clear_cache()
    log_mem("phase1-done")
    log(f"  collected {len(per_tok)} scored tokens")

    # ---------- Phase 2: COMPOSED model — composed logprobs → damage label ----------
    log("\n=== Phase 2: (1/N) composed logprobs → damage ===")
    model, tok = load(MODEL_ID)
    attach_avg_composed(model, adapters, LORA_SCALE, 1.0 / N_ADAPTERS)
    gc.collect(); mx.clear_cache()

    ptr = 0
    for pi, (domain, text) in enumerate(prompts):
        ids = encode_capped(tok, text)
        if len(ids) < 4:
            continue
        ids_row = mx.array(ids)[None]
        comp_lp = token_logprobs(model, ids_row)       # (T-1,)
        T = ids_row.shape[1]
        for t in range(T - 1):
            rec = per_tok[ptr]
            assert rec["prompt_idx"] == pi, f"alignment drift at {ptr}"
            rec["composed_lp"] = float(comp_lp[t].item())
            rec["damage"] = rec["base_lp"] - rec["composed_lp"]   # base − composed
            ptr += 1
        if pi % 8 == 0:
            log(f"  [composed] prompt {pi}/{len(prompts)} domain={domain}")
    assert ptr == len(per_tok), f"alignment: scored {ptr} of {len(per_tok)}"

    del model, tok
    gc.collect(); mx.clear_cache()
    log_mem("phase2-done")

    # ---------- Phase 3: correlate ----------
    log("\n=== Phase 3: correlate fragility vs damage ===")
    frag = [r["fragility"] for r in per_tok]
    dmg = [r["damage"] for r in per_tok]
    n = len(frag)
    assert n >= MIN_TOKENS, f"too few tokens ({n} < {MIN_TOKENS})"

    rho = spearman(frag, dmg)

    # top-decile-damage binary labels
    thr = sorted(dmg)[int(math.ceil(0.9 * n)) - 1]
    labels = [1 if d >= thr else 0 for d in dmg]
    auc, n_pos, n_neg = auc_top_decile(frag, labels)

    # diagnostics
    mean_dmg = sum(dmg) / n
    frac_damaged = sum(1 for d in dmg if d > 0) / n

    # ---------- Kill 2305 (verbatim) ----------
    clause_rho = rho < RHO_FLOOR
    clause_auc = (not math.isnan(auc)) and auc < AUC_FLOOR
    killed = clause_rho or clause_auc
    verdict = "killed" if killed else "supported"
    all_pass = not killed

    results = {
        "experiment_id": "exp_spark_base_fragility_oracle",
        "config": {
            "base_model": MODEL_ID,
            "adapters": DOMAINS,
            "adapter_paths": [str(ADAPTER_ROOT / d / "adapters.safetensors") for d in DOMAINS],
            "n_adapters": N_ADAPTERS,
            "lora_scale": LORA_SCALE,
            "merge": "(1/N) average  y = Wh + (1/N) Σ s·B_i A_i",
            "mid_layer": MID_LAYER,
            "n_noise_draws": N_NOISE_DRAWS,
            "noise_frac_rho": NOISE_FRAC,
            "eps_seed": EPS_SEED,
            "eps_is_random_noise_not_adapter_delta": True,
            "predictor_uses_zero_adapter_info": True,
            "n_prompts_per_domain": N_PROMPTS_PER_DOMAIN,
            "max_tokens_per_prompt": MAX_TOKENS_PER_PROMPT,
            "rho_floor": RHO_FLOOR,
            "auc_floor": AUC_FLOOR,
        },
        "n_tokens_scored": n,
        "spearman_rho_fragility_vs_damage": rho,
        "top_decile_damage_auc": auc,
        "top_decile_n_pos": n_pos,
        "top_decile_n_neg": n_neg,
        "mean_damage": mean_dmg,
        "frac_tokens_damaged": frac_damaged,
        "kill_criteria": {
            "2305": {
                "text": ("Spearman rho < 0.30 between base-fragility and per-token "
                         "interference damage, OR top-decile-damage AUC < 0.62"),
                "clause_rho_below_0_30": bool(clause_rho),
                "clause_auc_below_0_62": bool(clause_auc),
                "result": "fail" if killed else "pass",
            }
        },
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "total_wall_clock_sec": time.time() - t0,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    log("\n" + "=" * 72)
    log(f"n_tokens={n}  rho={rho:.4f}  AUC(top-decile)={auc:.4f}  "
        f"(pos={n_pos}/neg={n_neg})")
    log(f"mean_damage={mean_dmg:.4f}  frac_damaged={frac_damaged:.3f}")
    log(f"KILL 2305: rho<{RHO_FLOOR}={clause_rho}  AUC<{AUC_FLOOR}={clause_auc}")
    log(f"VERDICT: {verdict}  all_pass={all_pass}")
    log(f"Wrote {RESULTS_FILE}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
