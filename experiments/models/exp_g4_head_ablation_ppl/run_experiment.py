"""Behavioral PPL ablation of top-20% vs bottom-20% q_proj head slabs (F#742 follow-up).

For each domain in {code, math, medical}:
  (a) base       — no adapter
  (b) intact     — base + full q_proj LoRA adapter
  (c) top20      — base + adapter with the 67 highest-mass head slabs zeroed
  (d) bot20      — base + adapter with the 67 lowest-mass head slabs zeroed
Measures assistant-token perplexity on the held-out valid split (50 samples)
and evaluates K#1967 (degradation ratio) and K#1968 (adapter meaningfulness).

Head-slab mass ranking is REUSED verbatim from the parent experiment's
results.json (per_layer_head_mass) — not recomputed. See MATH.md.

Real MLX, real Gemma-4 E4B 4bit, real forward passes. is_smoke = False.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from safetensors import safe_open

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parents[2]
PARENT_DIR = REPO / "experiments" / "models" / "exp_g4_attention_head_importance_ranking"
ADAPTER_ROOT = REPO / "experiments" / "models" / "exp_p1_t2_single_domain_training" / "adapters"
DATA_ROOT = REPO / "experiments" / "models" / "exp_p1_t2_single_domain_training" / "data"

BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
DOMAINS = ["code", "math", "medical"]

# Architecture (matches parent + gemma4_text.py Attention)
NUM_HEADS = 8
SLIDING_HEAD_DIM = 256
GLOBAL_HEAD_DIM = 512
FULL_ATTENTION_LAYERS = {5, 11, 17, 23, 29, 35, 41}
HIDDEN = 2560
NUM_LAYERS = 42
RANK = 6
SCALE = 6.0

N_SAMPLES = 50
MAX_SEQ = 512
ABLATE_FRACTION = 0.20

# Pre-registered thresholds (MATH.md §6)
K1967_RATIO_THRESHOLD = 2.0     # R̄ > 2.0 = PASS (heads matter); ≤ 2.0 = FIRE/KILL
K1968_GAIN_THRESHOLD = 0.05     # g >= 5% = PASS (adapter meaningful); < 5% = FIRE/KILL


def head_dim_for(layer: int) -> int:
    return GLOBAL_HEAD_DIM if layer in FULL_ATTENTION_LAYERS else SLIDING_HEAD_DIM


# ----------------------------------------------------------------------------
# Adapter loading (matches parent conventions)
# ----------------------------------------------------------------------------
def load_adapter(domain: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Return {layer: (A (HIDDEN,RANK), B (RANK,q_out))} float32."""
    path = ADAPTER_ROOT / domain / "adapters.safetensors"
    assert path.exists(), f"adapter missing: {path}"
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    with safe_open(str(path), framework="numpy") as f:
        keys = set(f.keys())
        for layer in range(NUM_LAYERS):
            ak = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_a"
            bk = f"language_model.model.layers.{layer}.self_attn.q_proj.lora_b"
            if ak not in keys or bk not in keys:
                continue
            a = f.get_tensor(ak).astype(np.float32)  # (HIDDEN, RANK)
            b = f.get_tensor(bk).astype(np.float32)  # (RANK, q_out)
            q_out = NUM_HEADS * head_dim_for(layer)
            assert a.shape == (HIDDEN, RANK), f"L{layer} a {a.shape}"
            assert b.shape == (RANK, q_out), f"L{layer} b {b.shape} != ({RANK},{q_out})"
            out[layer] = (a, b)
    return out


def load_parent_mass() -> Dict[str, np.ndarray]:
    r = json.loads((PARENT_DIR / "results.json").read_text())
    assert r["base_model"] == BASE_MODEL, r["base_model"]
    return {d: np.array(r["per_layer_head_mass"][d], dtype=np.float64) for d in DOMAINS}


def ranked_slab_sets(mass: np.ndarray) -> Tuple[set, set, int]:
    """Return (top_set, bottom_set, k) of (layer,head) tuples by mass."""
    flat = mass.flatten()  # 336, layer-major: idx = layer*NUM_HEADS + head
    k = max(1, int(round(ABLATE_FRACTION * flat.size)))
    order = np.argsort(flat)  # ascending
    bottom = order[:k]
    top = order[-k:]
    to_lh = lambda idx: (int(idx // NUM_HEADS), int(idx % NUM_HEADS))
    return set(map(to_lh, top)), set(map(to_lh, bottom)), k


def build_masked_B(domain_adapter, ablate_set) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Copy adapter and zero the head-slab columns of B for (layer,head) in ablate_set."""
    masked: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    per_layer_heads: Dict[int, List[int]] = {}
    for (l, h) in ablate_set:
        per_layer_heads.setdefault(l, []).append(h)
    for layer, (a, b) in domain_adapter.items():
        b2 = b.copy()
        if layer in per_layer_heads:
            hd = head_dim_for(layer)
            for h in per_layer_heads[layer]:
                b2[:, h * hd:(h + 1) * hd] = 0.0
        masked[layer] = (a, b2)
    return masked


# ----------------------------------------------------------------------------
# LoRA q_proj wrapper: base(x) + scale * (x @ A) @ B   (subclass + setattr, F#831)
# ----------------------------------------------------------------------------
class LoRAQProj(nn.Module):
    def __init__(self, base: nn.Module, A: mx.array, B: mx.array, scale: float):
        super().__init__()
        self.base = base
        self.lora_a = A      # (HIDDEN, RANK)
        self.lora_b = B      # (RANK, q_out)
        self.scale = scale

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        delta = (x @ self.lora_a) @ self.lora_b
        return y + self.scale * delta.astype(y.dtype)


def attach_adapter(model, adapter: Dict[int, Tuple[np.ndarray, np.ndarray]]):
    """Install LoRAQProj wrappers on each layer's self_attn.q_proj. Returns originals."""
    lm = model.language_model
    originals = {}
    for layer, (a, b) in adapter.items():
        attn = lm.model.layers[layer].self_attn
        base_q = attn.q_proj
        originals[layer] = base_q
        wrapper = LoRAQProj(base_q, mx.array(a), mx.array(b), SCALE)
        setattr(attn, "q_proj", wrapper)
    return originals


def detach_adapter(model, originals):
    lm = model.language_model
    for layer, base_q in originals.items():
        setattr(lm.model.layers[layer].self_attn, "q_proj", base_q)


# ----------------------------------------------------------------------------
# Data + PPL
# ----------------------------------------------------------------------------
def load_samples(domain: str, n: int) -> List[List[dict]]:
    lines = (DATA_ROOT / domain / "valid.jsonl").read_text().splitlines()
    out = []
    for ln in lines:
        if not ln.strip():
            continue
        msgs = json.loads(ln)["messages"]
        out.append(msgs)
        if len(out) >= n:
            break
    return out


def build_scored_batch(tok, messages: List[dict]):
    """Return (input_ids (1,L), score_mask (L,)) where score_mask marks assistant tokens.

    Prompt (everything up to and including the assistant turn opener) is masked
    out (mask_prompt=True, matching training); only assistant content tokens are
    scored, replicating training-loss semantics.
    """
    user = [m for m in messages if m["role"] == "user"]
    asst = [m for m in messages if m["role"] == "assistant"]
    if not user or not asst:
        return None
    prompt_ids = tok.apply_chat_template(
        [user[0]], add_generation_prompt=True, tokenize=True
    )
    full_ids = tok.apply_chat_template(
        [user[0], asst[0]], add_generation_prompt=False, tokenize=True
    )
    if len(full_ids) <= len(prompt_ids):
        return None
    full_ids = full_ids[:MAX_SEQ]
    L = len(full_ids)
    mask = np.zeros(L, dtype=bool)
    start = min(len(prompt_ids), L)
    mask[start:] = True            # score assistant tokens only
    return np.array(full_ids, dtype=np.int32)[None, :], mask


def corpus_ppl(model, tok, samples) -> float:
    """Assistant-token perplexity over the fixed corpus (teacher forcing)."""
    total_nll = 0.0
    total_tok = 0
    for messages in samples:
        b = build_scored_batch(tok, messages)
        if b is None:
            continue
        ids, smask = b
        x = mx.array(ids)
        logits = model(x)                       # (1, L, V)
        logits = logits[:, :-1, :].astype(mx.float32)
        targets = x[:, 1:]                       # (1, L-1)
        tgt_mask = mx.array(smask[1:])           # align: predicting token t from t-1
        logp = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        )
        mx.eval(logp)
        logp_np = np.array(logp)
        m = smask[1:]
        total_nll += float(logp_np[m].sum())
        total_tok += int(m.sum())
        del logits, logp, x
    mx.clear_cache()
    if total_tok == 0:
        return float("nan")
    return float(np.exp(total_nll / total_tok))


# ----------------------------------------------------------------------------
# Phases
# ----------------------------------------------------------------------------
def measure_arm(model, tok, samples, adapter_or_none, label) -> float:
    if adapter_or_none is None:
        ppl = corpus_ppl(model, tok, samples)
    else:
        originals = attach_adapter(model, adapter_or_none)
        try:
            ppl = corpus_ppl(model, tok, samples)
        finally:
            detach_adapter(model, originals)
    print(f"    [{label}] PPL = {ppl:.4f}")
    return ppl


def main() -> int:
    print("=== exp_g4_head_ablation_ppl ===")
    print(f"Reference: arxiv:1905.10650 (Michel 2019) + Finding #742")
    print(f"Platform skills invoked: /mlx-dev, /fast-mlx")
    print(f"Base model: {BASE_MODEL}")
    print(f"KC count: 2 (K#1967 PPL degradation ratio [TARGET], K#1968 adapter gain [TARGET])")

    masses = load_parent_mass()
    print(f"Loaded parent per-head mass for {list(masses.keys())}")

    print("\nLoading base model...")
    model, tok = load(BASE_MODEL)
    mx.eval(model.parameters())
    mx.clear_cache()

    per_domain = {}
    for dom in DOMAINS:
        print(f"\n--- domain: {dom} ---")
        top_set, bot_set, k = ranked_slab_sets(masses[dom])
        overlap = len(top_set & bot_set)
        print(f"  slabs total={masses[dom].size}, k(20%)={k}, top/bot overlap={overlap}")
        assert overlap == 0, "top and bottom sets must be disjoint"

        adapter = load_adapter(dom)
        top_masked = build_masked_B(adapter, top_set)
        bot_masked = build_masked_B(adapter, bot_set)

        samples = load_samples(dom, N_SAMPLES)
        print(f"  held-out samples: {len(samples)}")

        ppl_base = measure_arm(model, tok, samples, None, "base")
        ppl_intact = measure_arm(model, tok, samples, adapter, "intact")
        ppl_top = measure_arm(model, tok, samples, top_masked, "top20_zeroed")
        ppl_bot = measure_arm(model, tok, samples, bot_masked, "bot20_zeroed")

        deg_top = ppl_top - ppl_intact
        deg_bot = ppl_bot - ppl_intact
        # ratio: positive degradations expected; guard tiny/neg bottom degradation
        if deg_bot > 1e-6:
            ratio = deg_top / deg_bot
        elif deg_top > 1e-6:
            ratio = float("inf")   # top hurts, bottom does not -> heads matter
        else:
            ratio = 1.0            # neither ablation hurts
        gain = (ppl_base - ppl_intact) / ppl_base if ppl_base > 0 else 0.0

        per_domain[dom] = {
            "ppl_base": ppl_base,
            "ppl_intact": ppl_intact,
            "ppl_top20_zeroed": ppl_top,
            "ppl_bot20_zeroed": ppl_bot,
            "deg_top": deg_top,
            "deg_bot": deg_bot,
            "degradation_ratio": ratio,
            "adapter_gain": gain,
            "k_slabs": k,
            "n_samples": len(samples),
        }
        print(f"  deg_top={deg_top:.4f} deg_bot={deg_bot:.4f} ratio={ratio:.4f} gain={gain*100:.2f}%")

        del adapter, top_masked, bot_masked, samples
        gc.collect()
        mx.clear_cache()

    # --- aggregate KCs ---
    finite_ratios = [per_domain[d]["degradation_ratio"] for d in DOMAINS
                     if np.isfinite(per_domain[d]["degradation_ratio"])]
    # mean ratio uses finite values; inf domains count as >2 (heads matter)
    inf_domains = [d for d in DOMAINS if not np.isfinite(per_domain[d]["degradation_ratio"])]
    if finite_ratios:
        r_bar = float(np.mean(finite_ratios))
    else:
        r_bar = float("inf")
    domains_ratio_gt2 = sum(
        1 for d in DOMAINS
        if (not np.isfinite(per_domain[d]["degradation_ratio"]))
        or per_domain[d]["degradation_ratio"] > 2.0
    )
    gains = [per_domain[d]["adapter_gain"] for d in DOMAINS]
    g_bar = float(np.mean(gains))

    # K#1967: PASS iff heads matter (mean ratio > 2 OR inf domains dominate)
    if inf_domains:
        # if any domain has inf ratio and finite mean also > 2, clearly pass;
        # treat overall as pass only if non-inf domains also lean that way
        k1967_pass = (r_bar > K1967_RATIO_THRESHOLD) if finite_ratios else True
        # but a single inf among small finite shouldn't auto-pass; require >=2/3 dom >2
        k1967_pass = domains_ratio_gt2 >= 2
    else:
        k1967_pass = r_bar > K1967_RATIO_THRESHOLD
    k1968_pass = g_bar >= K1968_GAIN_THRESHOLD

    all_pass = bool(k1967_pass and k1968_pass)
    verdict = "SUPPORTED" if all_pass else "KILLED"
    # success #114: top>2x bottom on >=2/3 domains AND gain>=5%
    success_114 = bool(domains_ratio_gt2 >= 2 and g_bar >= K1968_GAIN_THRESHOLD)

    print("\n=== Kill Criteria ===")
    print(f"K#1967 (TARGET): R̄={r_bar:.4f}, domains ratio>2={domains_ratio_gt2}/3, "
          f"inf_domains={inf_domains} -> {'PASS' if k1967_pass else 'FIRE/KILL'}")
    print(f"K#1968 (TARGET): mean adapter gain g={g_bar*100:.2f}% "
          f"(thr {K1968_GAIN_THRESHOLD*100:.0f}%) -> {'PASS' if k1968_pass else 'FIRE/KILL'}")
    print(f"Verdict: {verdict}")

    results = {
        "experiment": "exp_g4_head_ablation_ppl",
        "is_smoke": False,
        "base_model": BASE_MODEL,
        "adapter_source": "exp_p1_t2_single_domain_training",
        "adapter_target": "self_attn.q_proj",
        "ranking_source": "exp_g4_attention_head_importance_ranking/results.json::per_layer_head_mass",
        "rank": RANK,
        "scale": SCALE,
        "num_heads": NUM_HEADS,
        "n_samples_per_domain": N_SAMPLES,
        "max_seq_length": MAX_SEQ,
        "ablate_fraction": ABLATE_FRACTION,
        "held_out_corpus": "exp_p1_t2_single_domain_training/data/<domain>/valid.jsonl (first 50, assistant-token PPL)",
        "domains": DOMAINS,
        "per_domain": per_domain,
        "r_bar": r_bar,
        "domains_ratio_gt2": domains_ratio_gt2,
        "g_bar": g_bar,
        "k1967_threshold": K1967_RATIO_THRESHOLD,
        "k1968_threshold": K1968_GAIN_THRESHOLD,
        "k1967_pass": bool(k1967_pass),
        "k1968_pass": bool(k1968_pass),
        "k_results": {
            "1967": "pass" if k1967_pass else "fail",
            "1968": "pass" if k1968_pass else "fail",
        },
        "all_pass": all_pass,
        "success_114": success_114,
        "verdict": verdict,
    }
    out = EXP_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
