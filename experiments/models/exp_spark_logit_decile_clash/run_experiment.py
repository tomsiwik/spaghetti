#!/usr/bin/env python3
"""exp_spark_logit_decile_clash — off-domain composition damage is an OUTPUT-DISTRIBUTION CLASH at
the logit head: the adapter resurrects base-pruned (bottom-decile p0) tokens, not any weight geometry.

clash_t = sum_{v in topK(relu(dz)) ∩ bottom-decile(p0)} relu(dz_t[v]) / sum_{v in topK(relu(dz))} relu(dz_t[v])
  where dz = composed_logits - base_logits.   p0 = softmax(base_logits).

LABEL (primary): kl_damage_t = KL(p0 || pA), pA = softmax(composed_logits).
LABEL (secondary): nll_damage_t = base_logprob(y_{t+1}) - composed_logprob(y_{t+1}).

IN-SCRIPT GEOMETRIC BASELINE (same tokens, same prompts): we compute the F#869 alignment-angle and
F#864 delta-magnitude predictors HERE and define
  best_geometric_predictor_rho = max(|rho(align_angle, kl)|, |rho(delta_mag, kl)|).
The clash signal must beat that by +0.15 on identical data.

PRE-REGISTERED KILL 2306 (verbatim):
  "Spearman(clash_signal, per-token KL damage) < 0.45 OR clash_rho < best_geometric_predictor_rho + 0.15"

Composition is Σᵢ (1/N) sᵢ Bᵢ Aᵢ, never (ΣB)(ΣA). LORA_SCALE=6.0 ≤ 8.
Off-domain prompts: wikitext-2, squad contexts, filtered alpaca (no math/code/medical).
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

LORA_SCALE = 6.0
N_LAYERS_EXPECTED = 42
MID_LAYER = 21
N_ADAPTERS = len(DOMAINS)

TOPK = 128                  # K for top-K positive logit-shift mass
N_PROMPTS = 48              # off-domain prompts (16 wikitext + 16 squad + 16 alpaca)
MAX_TOKENS_PER_PROMPT = 96
MIN_TOKENS = 300

# Pre-registered thresholds (kill 2306)
RHO_FLOOR = 0.45
MARGIN = 0.15


def log(msg):
    print(msg, flush=True)


def log_mem(label=""):
    log(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB "
        f"cache={mx.get_cache_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")


# ----------------------------------------------------------------------------
# Off-domain prompt set (explicit, pre-registered). NONE of math/python/medical.
# ----------------------------------------------------------------------------

_DOMAIN_TOKENS = ["def ", "return", "import ", " mg", "dose", "patient", "theorem",
                  "solve", "equation", " = ", "def(", "function"]


def _has_domain(text):
    low = text.lower()
    return any(tok.strip().lower() in low for tok in _DOMAIN_TOKENS)


def load_offdomain_prompts():
    from datasets import load_dataset
    items = []

    # general English prose — wikitext-2 test
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    picked = 0
    for i in range(len(ds)):
        t = ds[i]["text"].strip()
        if len(t) > 220 and not _has_domain(t):
            items.append(("wikitext", t))
            picked += 1
            if picked >= 16:
                break

    # open-domain QA prose — squad validation contexts
    ds = load_dataset("squad", split="validation")
    seen = set()
    picked = 0
    for i in range(len(ds)):
        ctx = ds[i]["context"].strip()
        if ctx in seen:
            continue
        seen.add(ctx)
        if len(ctx) > 220 and not _has_domain(ctx):
            items.append(("squad", ctx))
            picked += 1
            if picked >= 16:
                break

    # everyday instruction-following — alpaca (filtered, no code/math/medical)
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    picked = 0
    for i in range(len(ds)):
        it = ds[i]
        if (it.get("input") or "").strip():
            continue  # skip ones with structured input
        txt = (it["instruction"].strip() + "\n" + (it.get("output") or "").strip()).strip()
        if len(txt) > 180 and not _has_domain(txt):
            items.append(("alpaca", txt))
            picked += 1
            if picked >= 16:
                break

    log(f"  Loaded {len(items)} OFF-domain texts (wikitext/squad/alpaca, none math/code/medical)")
    return items


def encode_capped(tokenizer, text):
    ids = tokenizer.encode(text)
    if len(ids) > MAX_TOKENS_PER_PROMPT:
        ids = ids[:MAX_TOKENS_PER_PROMPT]
    return ids


# ----------------------------------------------------------------------------
# Tap layer at L*: captures the layer OUTPUT hidden state (drift-proof subclass).
# Used in BOTH base and composed passes -> gives h_base and h_composed at L* on
# identical tokens, hence Δh = h_composed - h_base for the geometric baselines.
# ----------------------------------------------------------------------------

class TapLayer(gemma4_text.DecoderLayer):
    @classmethod
    def wrap(cls, layer):
        layer.__class__ = cls
        layer._captured = None
        return layer

    def __call__(self, x, mask=None, cache=None, per_layer_input=None,
                 shared_kv=None, offset=None):
        h, shared_kv, offset = gemma4_text.DecoderLayer.__call__(
            self, x, mask, cache, per_layer_input, shared_kv, offset)
        self._captured = h
        return h, shared_kv, offset


# ----------------------------------------------------------------------------
# (1/N) composed q_proj: y = W h + (1/N) Σᵢ sᵢ (h@Aᵢ)@Bᵢ. Σ Bᵢ Aᵢ, NOT (ΣB)(ΣA).
# ----------------------------------------------------------------------------

class AvgComposedQProj(nn.Module):
    def __init__(self, base_linear, a_list, b_list, scale, inv_n):
        super().__init__()
        self.linear = base_linear
        self.a_list = a_list
        self.b_list = b_list
        self.scale = scale
        self.inv_n = inv_n
        self.linear.freeze()

    def __call__(self, x):
        y = self.linear(x)
        acc = None
        for a, b in zip(self.a_list, self.b_list):
            d = (x @ a) @ b
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
# Stats (no scipy): Spearman rho.
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


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    t0 = time.time()
    log("=" * 72)
    log("exp_spark_logit_decile_clash")
    log(f"Base: {MODEL_ID} | L*={MID_LAYER} | TOPK={TOPK}")
    log(f"Composition: (1/{N_ADAPTERS}) Σ s·BᵢAᵢ domains={DOMAINS} scale={LORA_SCALE}")
    log(f"KILL 2306: rho(clash,KL)<{RHO_FLOOR} OR clash_rho<geom_rho+{MARGIN}")
    log("=" * 72)

    adapters = []
    for d in DOMAINS:
        p = ADAPTER_ROOT / d / "adapters.safetensors"
        assert p.exists(), f"missing {p}"
        adapters.append(mx.load(str(p)))
    log_mem("start")

    prompts = load_offdomain_prompts()

    # ---------- Phase 1: BASE — base logits (p0), base hidden h at L*, base logprob ----------
    log("\n=== Phase 1: base forward (p0, h_base@L*, base_lp) ===")
    model, tok = load(MODEL_ID)
    base_tap = TapLayer.wrap(get_lm(model).model.layers[MID_LAYER])
    mx.eval(model.parameters())

    # store per prompt to align with phase 2; per token records built across phases
    per_tok = []   # one dict per scored token (positions 0..T-2)
    base_cache = []  # per prompt: (ids_row, z0 logits (1,T,V), h_base (1,T,d))
    for pi, (domain, text) in enumerate(prompts):
        ids = encode_capped(tok, text)
        if len(ids) < 4:
            base_cache.append(None)
            continue
        ids_row = mx.array(ids)[None]
        z0 = model(ids_row)                       # (1,T,V)
        mx.eval(z0)
        h_base = base_tap._captured               # (1,T,d)
        mx.eval(h_base)

        # base logprob of true next token at 0..T-2
        lp0_full = z0 - mx.logsumexp(z0, axis=-1, keepdims=True)
        targets = ids_row[0, 1:]
        idx = mx.arange(targets.shape[0])
        base_lp = lp0_full[0, :-1, :][idx, targets]   # (T-1,)
        mx.eval(base_lp)

        base_cache.append((ids_row, z0, h_base, base_lp))
        T = ids_row.shape[1]
        for t in range(T - 1):
            per_tok.append({"prompt_idx": pi, "domain": domain,
                            "base_lp": float(base_lp[t].item())})
        if pi % 12 == 0:
            log(f"  [base] {pi}/{len(prompts)} {domain} T={T} tot_tok={len(per_tok)}")
    log_mem("phase1-done")
    log(f"  collected {len(per_tok)} scored tokens")

    # ---------- Phase 2: COMPOSED — composed logits (pA), composed hidden h at L* ----------
    log("\n=== Phase 2: (1/N) composed forward (pA, h_comp@L*) ===")
    attach_avg_composed(model, adapters, LORA_SCALE, 1.0 / N_ADAPTERS)
    comp_tap = TapLayer.wrap(get_lm(model).model.layers[MID_LAYER])  # already TapLayer; re-wrap noop-safe
    gc.collect(); mx.clear_cache()

    ptr = 0
    n_resurrected = 0
    for pi, (domain, text) in enumerate(prompts):
        bc = base_cache[pi]
        if bc is None:
            continue
        ids_row, z0, h_base, base_lp = bc
        zA = model(ids_row)                       # (1,T,V) composed logits
        mx.eval(zA)
        h_comp = comp_tap._captured               # (1,T,d)
        mx.eval(h_comp)

        T = ids_row.shape[1]

        # --- distributions ---
        lp0 = z0 - mx.logsumexp(z0, axis=-1, keepdims=True)
        lpA = zA - mx.logsumexp(zA, axis=-1, keepdims=True)
        p0 = mx.exp(lp0)                          # (1,T,V) base dist
        # KL(p0 || pA) per position
        kl = (p0 * (lp0 - lpA)).sum(axis=-1)[0]   # (T,)
        mx.eval(kl)
        # composed logprob of true next token
        targets = ids_row[0, 1:]
        idx = mx.arange(targets.shape[0])
        comp_lp = lpA[0, :-1, :][idx, targets]    # (T-1,)
        mx.eval(comp_lp)

        # --- clash signal at the head ---
        dz = (zA - z0)[0]                          # (T,V) composed logit shift
        rdz = mx.maximum(dz, 0.0)                  # relu(dz)
        # bottom-decile threshold of p0 per position (10th percentile of probabilities)
        p0r = p0[0]                                # (T,V)
        V = p0r.shape[-1]
        k10 = max(1, int(0.10 * V))
        # bottom-decile mask: tokens with the k10 smallest p0 — use threshold via partition
        sorted_p0 = mx.sort(p0r, axis=-1)          # ascending (T,V)
        thr_dec = sorted_p0[:, k10 - 1][:, None]   # (T,1) 10th-percentile prob
        bottom_mask = (p0r <= thr_dec)             # (T,V) base-pruned tokens

        # topK over relu(dz): get indices of K largest per position
        topk_idx = mx.argpartition(-rdz, TOPK - 1, axis=-1)[:, :TOPK]   # (T,K)
        rows = mx.arange(T)[:, None]
        topk_vals = rdz[rows, topk_idx]                                 # (T,K)
        topk_inbottom = bottom_mask[rows, topk_idx]                     # (T,K) bool
        num = (topk_vals * topk_inbottom).sum(axis=-1)                  # (T,)
        den = topk_vals.sum(axis=-1)                                    # (T,)
        clash = num / mx.maximum(den, 1e-12)                           # (T,)
        mx.eval(clash, num, den)

        # --- geometric baselines at L* (same tokens) ---
        dh = (h_comp - h_base)[0]                  # (T,d)
        hb = h_base[0]                             # (T,d)
        dh_norm = mx.sqrt((dh * dh).sum(axis=-1))  # (T,)
        hb_norm = mx.sqrt((hb * hb).sum(axis=-1))  # (T,)
        delta_mag = dh_norm / mx.maximum(hb_norm, 1e-8)            # (T,) F#864
        cos_dh_h = (dh * hb).sum(axis=-1) / mx.maximum(dh_norm * hb_norm, 1e-12)  # (T,) F#869
        mx.eval(delta_mag, cos_dh_h)

        for t in range(T - 1):
            rec = per_tok[ptr]
            assert rec["prompt_idx"] == pi, f"alignment drift at {ptr}"
            rec["kl_damage"] = float(kl[t].item())
            rec["nll_damage"] = rec["base_lp"] - float(comp_lp[t].item())
            rec["clash"] = float(clash[t].item())
            rec["align_cos"] = float(cos_dh_h[t].item())
            rec["delta_mag"] = float(delta_mag[t].item())
            if rec["clash"] > 0.5:
                n_resurrected += 1
            ptr += 1
        if pi % 12 == 0:
            log(f"  [composed] {pi}/{len(prompts)} {domain} "
                f"mean_clash≈{float(clash.mean().item()):.3f}")
    assert ptr == len(per_tok), f"alignment: scored {ptr} of {len(per_tok)}"

    del model, tok
    gc.collect(); mx.clear_cache()
    log_mem("phase2-done")

    # ---------- Phase 3: correlate ----------
    log("\n=== Phase 3: correlate ===")
    n = len(per_tok)
    assert n >= MIN_TOKENS, f"too few tokens ({n} < {MIN_TOKENS})"

    clash = [r["clash"] for r in per_tok]
    kl = [r["kl_damage"] for r in per_tok]
    nll = [r["nll_damage"] for r in per_tok]
    align = [r["align_cos"] for r in per_tok]
    dmag = [r["delta_mag"] for r in per_tok]

    rho_clash_kl = spearman(clash, kl)
    rho_clash_nll = spearman(clash, nll)

    # geometric baselines — take magnitude of rho vs KL (sign-free predictor power)
    rho_align_kl = spearman(align, kl)
    rho_dmag_kl = spearman(dmag, kl)
    best_geom_rho = max(abs(rho_align_kl), abs(rho_dmag_kl))
    best_geom_name = "align_angle" if abs(rho_align_kl) >= abs(rho_dmag_kl) else "delta_mag"

    # diagnostics
    mean_clash = sum(clash) / n
    mean_kl = sum(kl) / n
    frac_resurrected = n_resurrected / n

    # ---------- Kill 2306 (verbatim) ----------
    clause_a = rho_clash_kl < RHO_FLOOR
    clause_b = rho_clash_kl < best_geom_rho + MARGIN
    killed = clause_a or clause_b
    verdict = "killed" if killed else "supported"
    all_pass = not killed

    results = {
        "experiment_id": "exp_spark_logit_decile_clash",
        "config": {
            "base_model": MODEL_ID,
            "adapters": DOMAINS,
            "adapter_paths": [str(ADAPTER_ROOT / d / "adapters.safetensors") for d in DOMAINS],
            "n_adapters": N_ADAPTERS,
            "lora_scale": LORA_SCALE,
            "merge": "(1/N) average  y = Wh + (1/N) Σ s·B_i A_i",
            "mid_layer": MID_LAYER,
            "topk": TOPK,
            "n_prompts": len(prompts),
            "max_tokens_per_prompt": MAX_TOKENS_PER_PROMPT,
            "offdomain_sources": ["wikitext-2-raw-v1/test", "squad/validation",
                                  "tatsu-lab/alpaca (filtered no math/code/medical)"],
            "label_primary": "kl_damage = KL(p0 || pA)",
            "label_secondary": "nll_damage = base_lp - composed_lp",
            "rho_floor": RHO_FLOOR,
            "margin": MARGIN,
        },
        "n_tokens_scored": n,
        "spearman_rho_clash_vs_kl": rho_clash_kl,
        "spearman_rho_clash_vs_nll": rho_clash_nll,
        "geometric_baselines": {
            "rho_align_angle_vs_kl": rho_align_kl,
            "rho_delta_mag_vs_kl": rho_dmag_kl,
            "best_geometric_predictor": best_geom_name,
            "best_geometric_predictor_rho": best_geom_rho,
            "threshold_to_beat": best_geom_rho + MARGIN,
        },
        "mean_clash": mean_clash,
        "mean_kl_damage": mean_kl,
        "frac_tokens_clash_gt_half": frac_resurrected,
        "kill_criteria": {
            "2306": {
                "text": ("Spearman(clash_signal, per-token KL damage) < 0.45 OR "
                         "clash_rho < best_geometric_predictor_rho + 0.15"),
                "clause_a_rho_below_0_45": bool(clause_a),
                "clause_b_below_geom_plus_margin": bool(clause_b),
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
    log(f"n_tokens={n}")
    log(f"rho(clash, KL)  = {rho_clash_kl:.4f}   rho(clash, NLL) = {rho_clash_nll:.4f}")
    log(f"GEOM: rho(align,KL)={rho_align_kl:.4f}  rho(dmag,KL)={rho_dmag_kl:.4f}  "
        f"best={best_geom_name}={best_geom_rho:.4f}")
    log(f"threshold_to_beat = {best_geom_rho + MARGIN:.4f}")
    log(f"mean_clash={mean_clash:.3f} mean_KL={mean_kl:.4f} frac_clash>0.5={frac_resurrected:.3f}")
    log(f"KILL 2306: A(rho<{RHO_FLOOR})={clause_a}  B(rho<geom+{MARGIN})={clause_b}")
    log(f"VERDICT: {verdict}  all_pass={all_pass}")
    log(f"Wrote {RESULTS_FILE}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
