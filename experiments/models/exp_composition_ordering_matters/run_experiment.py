"""exp_composition_ordering_matters — FP summation ordering invariance test.

Phase 1 — pure numpy: per-layer Σ B_i A_i in 6 permutations of (medical, math, code).
         Max pairwise relative Frobenius / operator-norm gap across permutations (K1928 proxy).

Phase 2 — MLX: replace LoRALinear.__call__ with a permutation-order sum of 3 adapter pairs.
         Measure PPL on mixed-domain 100-row held-out corpus across all 6 permutations.
         Max pairwise relative PPL gap (K1975 target).

See MATH.md for theorem and predictions.
"""
from __future__ import annotations

import gc
import importlib.metadata as md
import itertools
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
from safetensors import safe_open

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parents[2]
UPSTREAM = REPO / "micro" / "models" / "exp_p1_t2_single_domain_training" / "adapters"
DATA = REPO / "micro" / "models" / "exp_p1_t2_single_domain_training" / "data"

BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
DOMAINS = ("medical", "math", "code")

# Eval params (MATH.md §Methodology).
N_EVAL_PER_DOMAIN = (33, 33, 34)  # medical / math / code — sums to 100
MAX_SEQ_LEN = 512
SEED = 42
ADAPTER_SCALE = 6.0  # per lora_config_*.yaml

# KC thresholds (MATH.md §Pre-registered Kill Criteria).
K1928_FROB_REL_THRESH = 1e-5
K1975_PPL_REL_THRESH = 0.01  # 1pp relative

mx.random.seed(SEED)
rng = np.random.default_rng(SEED)


# ---------- Phase 1 ----------

def load_adapter_weights_numpy(domain: str) -> Dict[str, np.ndarray]:
    """Return {full_key: numpy fp32} for a domain adapter.safetensors."""
    path = UPSTREAM / domain / "adapters.safetensors"
    assert path.exists(), f"missing: {path}"
    out: Dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k).astype(np.float32)
    return out


def group_keys_by_layer(keys) -> Dict[str, Tuple[str, str]]:
    """Map a layer prefix (e.g. 'language_model.model.layers.0.self_attn.q_proj')
    to (lora_a_key, lora_b_key)."""
    prefixes: Dict[str, Dict[str, str]] = {}
    for k in keys:
        if k.endswith(".lora_a"):
            prefix = k[: -len(".lora_a")]
            prefixes.setdefault(prefix, {})["a"] = k
        elif k.endswith(".lora_b"):
            prefix = k[: -len(".lora_b")]
            prefixes.setdefault(prefix, {})["b"] = k
    out: Dict[str, Tuple[str, str]] = {}
    for p, d in prefixes.items():
        if "a" in d and "b" in d:
            out[p] = (d["a"], d["b"])
    return out


def phase1_weight_space(results: dict) -> List[str]:
    """K1928 — per-layer relative Frobenius / operator-norm gap across 6 permutations.

    Returns the sorted list of layer prefixes (reused in Phase 2 to match module names).
    """
    print("\n=== PHASE 1: Weight-space ordering (K1928) ===", flush=True)
    t0 = time.time()

    weights = {dom: load_adapter_weights_numpy(dom) for dom in DOMAINS}
    # Reference layer prefixes from first domain; assume same across domains (sanity-checked).
    layers_med = group_keys_by_layer(weights["medical"].keys())
    for dom in DOMAINS[1:]:
        layers_other = group_keys_by_layer(weights[dom].keys())
        assert set(layers_med.keys()) == set(layers_other.keys()), f"layer mismatch {dom}"
    layer_prefixes = sorted(layers_med.keys())
    print(f"  layers: {len(layer_prefixes)}", flush=True)

    perms = list(itertools.permutations([0, 1, 2]))
    assert len(perms) == 6

    per_layer: List[dict] = []
    for prefix in layer_prefixes:
        a_key, b_key = layers_med[prefix]
        # Per-domain delta: ΔW = scale * (B @ A)^T  ? No: lora forward is y = x @ A @ B.
        # Effective weight delta on y = x @ W.T style linear: ΔW^T = scale * A @ B (shape d_in×d_out)
        # To keep it simple and in line with forward-pass pre-transpose:
        #   let Delta_fwd = scale * A @ B  ∈  R^(d_in × d_out)
        # We compare Delta_fwd across permutation sums.
        deltas = []
        for dom in DOMAINS:
            A = weights[dom][a_key]   # (d_in, r)
            B = weights[dom][b_key]   # (r, d_out)
            d_fwd = ADAPTER_SCALE * (A @ B)  # (d_in, d_out) fp32
            deltas.append(d_fwd)

        # Compute 6 permutation sums in explicit left-associative order.
        perm_sums = []
        for p in perms:
            # left-fold: ((d[p0] + d[p1]) + d[p2])
            s = deltas[p[0]] + deltas[p[1]]
            s = s + deltas[p[2]]
            perm_sums.append(s)

        # Reference norm (take perm 0's S).
        ref = perm_sums[0]
        ref_frob = float(np.linalg.norm(ref, ord="fro"))
        # Operator-norm bound is O(Frobenius); Frobenius is the canonical KC metric.
        # Operator norm (single SVD, ref only) as diagnostic.
        # Skip per-pair operator norm to keep Phase 1 fast (630 SVDs would dominate wall-clock).

        max_rel_frob_gap = 0.0
        max_abs_frob_gap = 0.0
        for i in range(6):
            for j in range(i + 1, 6):
                diff = perm_sums[i] - perm_sums[j]
                abs_f = float(np.linalg.norm(diff, ord="fro"))
                rel_f = abs_f / max(ref_frob, 1e-30)
                if abs_f > max_abs_frob_gap:
                    max_abs_frob_gap = abs_f
                if rel_f > max_rel_frob_gap:
                    max_rel_frob_gap = rel_f

        per_layer.append({
            "prefix": prefix,
            "ref_frob": ref_frob,
            "max_abs_frob_gap": max_abs_frob_gap,
            "max_rel_frob_gap": max_rel_frob_gap,
        })

    global_max_rel_frob = max(l["max_rel_frob_gap"] for l in per_layer)
    global_max_abs_frob = max(l["max_abs_frob_gap"] for l in per_layer)

    k1928_fire = global_max_rel_frob > K1928_FROB_REL_THRESH

    elapsed = time.time() - t0
    print(f"  phase1 done in {elapsed:.1f}s", flush=True)
    print(f"  global_max_rel_frob_gap: {global_max_rel_frob:.3e}  (threshold {K1928_FROB_REL_THRESH:.1e})")
    print(f"  global_max_abs_frob_gap: {global_max_abs_frob:.3e}")
    print(f"  K1928 (ordering matters at weight level) FIRES: {k1928_fire}")

    # Keep per-layer summary; no full arrays.
    results["phase1"] = {
        "n_layers": len(per_layer),
        "global_max_rel_frob_gap": global_max_rel_frob,
        "global_max_abs_frob_gap": global_max_abs_frob,
        "k1928_fire": k1928_fire,
        "per_layer": per_layer,
        "elapsed_sec": elapsed,
    }
    return layer_prefixes


# ---------- Phase 2 ----------

# Storage for per-instance adapter pairs and permutation order on LoRALinear.
# We monkey-patch the class-level __call__ to consume these per-instance attrs.
_ORIG_CALL = LoRALinear.__call__


def _perm_order_call(self: LoRALinear, x):
    """Custom forward for LoRALinear when perm-ordering enabled.

    Reads per-instance attrs:
        self.po_lora_as  -> tuple of (mx.array, mx.array, mx.array) for 3 adapters A_i
        self.po_lora_bs  -> tuple of (mx.array, mx.array, mx.array) for 3 adapters B_i
        self.po_order    -> tuple[int, int, int] = permutation of (0,1,2)
    """
    y = self.linear(x)
    dx = self.dropout(x)
    order = self.po_order
    # Left-fold sum in specified order
    term0 = (dx @ self.po_lora_as[order[0]]) @ self.po_lora_bs[order[0]]
    term1 = (dx @ self.po_lora_as[order[1]]) @ self.po_lora_bs[order[1]]
    term2 = (dx @ self.po_lora_as[order[2]]) @ self.po_lora_bs[order[2]]
    accum = term0 + term1
    accum = accum + term2
    return y + (self.scale * accum).astype(x.dtype)


def load_eval_jsonl(domain: str, split: str, limit: int) -> List[dict]:
    path = DATA / domain / f"{split}.jsonl"
    assert path.exists(), f"missing data: {path}"
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def format_chat(tok, messages: List[dict]) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def compute_ppl(model, tok, rows: List[dict], desc: str = "") -> float:
    """Teacher-forcing NLL per token across rows; returns PPL."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        text = format_chat(tok, row["messages"])
        ids = tok.encode(text)
        if len(ids) > MAX_SEQ_LEN:
            ids = ids[:MAX_SEQ_LEN]
        if len(ids) < 2:
            continue
        x = mx.array([ids[:-1]])
        y = mx.array([ids[1:]])
        logits = model(x)
        lse = mx.logsumexp(logits, axis=-1)
        tgt = mx.take_along_axis(logits, y[..., None], axis=-1).squeeze(-1)
        nll = (lse - tgt).sum()
        mx.eval(nll)
        total_nll += float(nll.item())
        total_tokens += int(y.shape[-1])
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"    [{desc}] {i+1}/{len(rows)}, elapsed {elapsed:.1f}s", flush=True)
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def phase2_behavioral(results: dict, layer_prefixes: List[str]) -> None:
    """K1975 — PPL invariance across 6 permutations of forward-pass LoRA summation."""
    print("\n=== PHASE 2: Behavioral ordering (K1975) ===", flush=True)

    # Sanity: install LoRALinear modules by loading with one of the adapters.
    print("  loading base + medical adapter (to install LoRALinear architecture)...", flush=True)
    t0 = time.time()
    model, tok = load(BASE_MODEL, adapter_path=str(UPSTREAM / "medical"))
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # Collect LoRALinear modules by full path name.
    modules_by_name: Dict[str, LoRALinear] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            modules_by_name[name] = mod
    if not modules_by_name:
        raise RuntimeError("no LoRALinear modules installed")
    print(f"  LoRALinear modules: {len(modules_by_name)}", flush=True)

    # Check coverage: layer_prefixes from safetensors should correspond to module names.
    # safetensors keys: 'language_model.model.layers.0.self_attn.q_proj'
    # module names via named_modules in mlx_lm:  typically same path structure.
    # If not matching we build a prefix-stripping helper.
    # First attempt: direct match.
    matched = 0
    for pref in layer_prefixes:
        if pref in modules_by_name:
            matched += 1
    if matched < len(layer_prefixes):
        print(f"  direct-match {matched}/{len(layer_prefixes)}, trying strip 'language_model.'", flush=True)
        # Try stripping common top-level prefix
        stripped = [p.split(".", 1)[-1] if p.startswith("language_model.") else p for p in layer_prefixes]
        matched2 = sum(1 for p in stripped if p in modules_by_name)
        if matched2 >= matched:
            name_map = {orig: stripped[i] for i, orig in enumerate(layer_prefixes)}
        else:
            name_map = {orig: orig for orig in layer_prefixes}
    else:
        name_map = {orig: orig for orig in layer_prefixes}

    # Load all 3 adapter weight dicts as numpy once, convert to mx lazily per layer
    adapter_np = {dom: load_adapter_weights_numpy(dom) for dom in DOMAINS}
    layer_key_groups = group_keys_by_layer(adapter_np["medical"].keys())

    # For each module, attach 3-adapter weights + order attr.
    installed = 0
    for orig_pref, mod_name in name_map.items():
        if mod_name not in modules_by_name:
            continue
        mod = modules_by_name[mod_name]
        a_key, b_key = layer_key_groups[orig_pref]
        # mx.array tuple; keep dtype matching existing mod.lora_a
        target_dtype = mod.lora_a.dtype
        a_mx = tuple(mx.array(adapter_np[dom][a_key], dtype=target_dtype) for dom in DOMAINS)
        b_mx = tuple(mx.array(adapter_np[dom][b_key], dtype=target_dtype) for dom in DOMAINS)
        mod.po_lora_as = a_mx
        mod.po_lora_bs = b_mx
        mod.po_order = (0, 1, 2)  # default
        installed += 1
    print(f"  installed 3-adapter pairs on {installed}/{len(modules_by_name)} modules", flush=True)
    assert installed == len(modules_by_name), "mismatch in module coverage"

    # Patch the class __call__ to the perm-order version (applies to all LoRALinear instances).
    LoRALinear.__call__ = _perm_order_call

    # Force re-evaluation of any graph state.
    mx.eval(model.parameters())

    # Build eval rows: 33 + 33 + 34 mixed.
    eval_rows: List[dict] = []
    for dom, n in zip(DOMAINS, N_EVAL_PER_DOMAIN):
        valid_path = DATA / dom / "valid.jsonl"
        rows = load_eval_jsonl(dom, "valid", n) if valid_path.exists() else []
        if len(rows) < n:
            extra = load_eval_jsonl(dom, "train", n - len(rows))
            rows = rows + extra
        eval_rows.extend(rows[:n])
    print(f"  eval rows: {len(eval_rows)} (target 100)", flush=True)

    # Iterate permutations.
    perms = list(itertools.permutations([0, 1, 2]))
    ppls: Dict[Tuple[int, int, int], float] = {}
    for order in perms:
        # Update per-module order
        for m in modules_by_name.values():
            m.po_order = tuple(order)
        mx.clear_cache()
        desc = f"perm={order}"
        print(f"\n  measuring PPL for {desc}...", flush=True)
        t0 = time.time()
        ppl = compute_ppl(model, tok, eval_rows, desc=desc)
        print(f"  {desc}: PPL={ppl:.6f}  ({time.time()-t0:.1f}s)", flush=True)
        ppls[order] = ppl

    # Restore original __call__ to avoid pollution.
    LoRALinear.__call__ = _ORIG_CALL

    # Compute pairwise PPL gap
    ppl_values = list(ppls.values())
    ppl_min = min(ppl_values)
    ppl_max = max(ppl_values)
    ppl_mean = sum(ppl_values) / len(ppl_values)
    max_abs_gap = ppl_max - ppl_min
    max_rel_gap = max_abs_gap / ppl_mean if ppl_mean > 0 else float("inf")
    k1929_fire = max_rel_gap > K1975_PPL_REL_THRESH

    print(f"\n  PPL values by permutation:")
    for order, ppl in ppls.items():
        print(f"    {order}: {ppl:.6f}")
    print(f"  PPL min={ppl_min:.6f}, max={ppl_max:.6f}, mean={ppl_mean:.6f}")
    print(f"  max_abs_gap={max_abs_gap:.3e}, max_rel_gap={max_rel_gap:.3e}  (threshold {K1975_PPL_REL_THRESH})")
    print(f"  K1975 (ordering matters behaviorally) FIRES: {k1929_fire}")

    results["phase2"] = {
        "ppl_per_permutation": {str(k): v for k, v in ppls.items()},
        "ppl_min": ppl_min,
        "ppl_max": ppl_max,
        "ppl_mean": ppl_mean,
        "max_abs_gap": max_abs_gap,
        "max_rel_gap": max_rel_gap,
        "k1929_fire": k1929_fire,
        "n_eval_rows": len(eval_rows),
    }

    del model, tok, modules_by_name
    gc.collect()
    mx.clear_cache()


# ---------- Main ----------

def main() -> int:
    try:
        mlx_v = md.version("mlx")
    except Exception:
        mlx_v = "unknown"
    mlxlm_v = md.version("mlx_lm")
    print(f"exp_composition_ordering_matters")
    print(f"mlx={mlx_v}, mlx_lm={mlxlm_v}, base={BASE_MODEL}")
    print(f"domains: {DOMAINS}, n_eval_per_domain={N_EVAL_PER_DOMAIN}, seed={SEED}")

    results: dict = {
        "experiment": "exp_composition_ordering_matters",
        "is_smoke": False,
        "base_model": BASE_MODEL,
        "mlx_version": mlx_v,
        "mlx_lm_version": mlxlm_v,
        "adapter_source": "exp_p1_t2_single_domain_training (q_proj r=6 scale=6.0)",
        "domains": list(DOMAINS),
        "n_eval_per_domain": list(N_EVAL_PER_DOMAIN),
        "seed": SEED,
        "adapter_scale": ADAPTER_SCALE,
    }

    layer_prefixes = phase1_weight_space(results)

    # Only run Phase 2 if Phase 1 doesn't already kill; PPL test is robustness cross-check.
    # (We always run it to get data for the target KC per F#666, even if K1928 doesn't fire.)
    phase2_behavioral(results, layer_prefixes)

    # F#666 target-gated kill rule
    p1 = results["phase1"]
    p2 = results["phase2"]

    k1928_fire = p1["k1928_fire"]
    k1929_fire = p2["k1929_fire"]

    # KILL iff both fire (ordering matters); SUPPORTED iff both don't fire.
    both_fire = k1928_fire and k1929_fire
    both_pass = (not k1928_fire) and (not k1929_fire)

    if both_fire:
        verdict = "KILLED"
        all_pass = False
    elif both_pass:
        verdict = "SUPPORTED"
        all_pass = True
    else:
        # proxy/target mismatch: F#666 provisional / finding-about-proxy
        verdict = "PROVISIONAL_F666_MISMATCH"
        all_pass = False

    results["k1928_fire"] = k1928_fire
    results["k1929_fire"] = k1929_fire
    results["verdict"] = verdict
    results["all_pass"] = all_pass

    print("\n=== FINAL ===")
    print(f"K1928 (weight-space frob gap > {K1928_FROB_REL_THRESH:.0e}): {'FIRE' if k1928_fire else 'no'}")
    print(f"K1975 (behavioral PPL gap > {K1975_PPL_REL_THRESH}): {'FIRE' if k1929_fire else 'no'}")
    print(f"Verdict: {verdict}")
    print(f"all_pass: {all_pass}")

    with open(EXP_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {EXP_DIR/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
