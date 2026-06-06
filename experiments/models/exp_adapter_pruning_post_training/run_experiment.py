"""exp_adapter_pruning_post_training — magnitude pruning of trained LoRA adapter weights.

Phase 1 (K1922 — single-adapter): medical adapter, full vs 50%-magnitude-pruned, PPL on
        first 100 medical/valid.jsonl rows. Pass if ΔPPL ≤ 0.1.

Phase 2 (K1923 — composition): medical + math (additive sum), full vs both-pruned-50%,
        PPL on 50 med + 50 math rows. Pass if ΔPPL_compose ≤ 3.0.

See MATH.md for theorem and predictions.
"""
from __future__ import annotations

import gc
import importlib.metadata as md
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from safetensors import safe_open

import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parents[2]
UPSTREAM = REPO / "micro" / "models" / "exp_p1_t2_single_domain_training" / "adapters"
DATA = REPO / "micro" / "models" / "exp_p1_t2_single_domain_training" / "data"

BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
ADAPTER_SCALE = 6.0  # per lora_config_*.yaml — matches trained config; not unsafe 20

# Eval params
N_EVAL_PHASE1 = 100  # medical valid
N_EVAL_PHASE2_PER_DOMAIN = 50  # 50 med + 50 math = 100 mixed
MAX_SEQ_LEN = 512
SEED = 42

# Pruning params
PRUNE_KEEP_FRACTION = 0.5  # keep top-50% by magnitude per matrix

# KC thresholds (locked pre-run; see MATH.md §Pre-registered Kill Criteria)
K1922_PPL_DELTA_THRESH = 0.10  # single-adapter behavioral
K1923_PPL_DELTA_THRESH = 3.00  # composition behavioral

mx.random.seed(SEED)


# ---------- adapter I/O ----------

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
    """Map layer prefix -> (lora_a_key, lora_b_key)."""
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


# ---------- pruning ----------

def magnitude_prune_np(w: np.ndarray, keep_frac: float) -> Tuple[np.ndarray, float]:
    """Zero out smallest (1-keep_frac) entries by |w|.
    Returns (pruned, retained_energy_fraction).
    """
    flat = np.abs(w).ravel()
    n = flat.size
    k = max(1, int(round(keep_frac * n)))  # keep top-k
    if k >= n:
        return w.copy(), 1.0
    # threshold = k-th largest |w|
    thresh = np.partition(flat, n - k)[n - k]
    mask = (np.abs(w) >= thresh).astype(w.dtype)
    pruned = w * mask
    e_full = float(np.sum(w.astype(np.float64) ** 2))
    e_pruned = float(np.sum(pruned.astype(np.float64) ** 2))
    f_retained = e_pruned / max(e_full, 1e-30)
    return pruned, f_retained


def prune_adapter_dict(weights: Dict[str, np.ndarray], keep_frac: float) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Return pruned copy and per-key retained-energy fractions."""
    pruned: Dict[str, np.ndarray] = {}
    f_map: Dict[str, float] = {}
    for k, w in weights.items():
        p, f = magnitude_prune_np(w, keep_frac)
        pruned[k] = p
        f_map[k] = f
    return pruned, f_map


# ---------- forward-pass monkey-patches ----------

# Phase 1: single adapter (medical) — read per-instance lora_a_p1 / lora_b_p1
def _phase1_call(self: LoRALinear, x):
    y = self.linear(x)
    dx = self.dropout(x)
    delta = (dx @ self.p1_lora_a) @ self.p1_lora_b
    return y + (self.scale * delta).astype(x.dtype)


# Phase 2: composition (medical + math) — read per-instance lora_a/b_med_p2, _math_p2
def _phase2_call(self: LoRALinear, x):
    y = self.linear(x)
    dx = self.dropout(x)
    term_med = (dx @ self.p2_med_a) @ self.p2_med_b
    term_math = (dx @ self.p2_math_a) @ self.p2_math_b
    return y + (self.scale * (term_med + term_math)).astype(x.dtype)


# ---------- eval ----------

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


# ---------- module helpers ----------

def collect_lora_modules(model) -> Dict[str, LoRALinear]:
    out: Dict[str, LoRALinear] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            out[name] = mod
    return out


def build_name_map(safetensors_keys: List[str], modules_by_name: Dict[str, LoRALinear]) -> Dict[str, str]:
    """Map a safetensors layer prefix to a module name in the loaded model.
    safetensors keys: 'language_model.model.layers.0.self_attn.q_proj.lora_a'
    module names: typically 'language_model.model.layers.0.self_attn.q_proj' (direct match).
    """
    layer_prefixes = sorted({k.rsplit(".lora_", 1)[0] for k in safetensors_keys})
    direct = sum(1 for p in layer_prefixes if p in modules_by_name)
    if direct == len(layer_prefixes):
        return {p: p for p in layer_prefixes}
    # try strip top-level
    stripped = [p.split(".", 1)[-1] if p.startswith("language_model.") else p for p in layer_prefixes]
    direct2 = sum(1 for p in stripped if p in modules_by_name)
    if direct2 >= direct:
        return {orig: stripped[i] for i, orig in enumerate(layer_prefixes)}
    return {p: p for p in layer_prefixes}


# ---------- Phase 1 ----------

def phase1_single_adapter(results: dict) -> None:
    print("\n=== PHASE 1: single-adapter pruning (K1922) ===", flush=True)
    t0 = time.time()

    print("  loading base + medical adapter...", flush=True)
    model, tok = load(BASE_MODEL, adapter_path=str(UPSTREAM / "medical"))
    modules_by_name = collect_lora_modules(model)
    if not modules_by_name:
        raise RuntimeError("no LoRALinear modules installed")
    print(f"  LoRALinear modules: {len(modules_by_name)}  (loaded in {time.time()-t0:.1f}s)", flush=True)

    med_np = load_adapter_weights_numpy("medical")
    name_map = build_name_map(list(med_np.keys()), modules_by_name)
    layer_keys = group_keys_by_layer(med_np.keys())

    # FULL: install full medical adapter via _phase1_call
    target_dtype = next(iter(modules_by_name.values())).lora_a.dtype
    med_full_mx: Dict[str, Tuple[mx.array, mx.array]] = {}
    for orig_pref in name_map:
        a_key, b_key = layer_keys[orig_pref]
        med_full_mx[orig_pref] = (
            mx.array(med_np[a_key], dtype=target_dtype),
            mx.array(med_np[b_key], dtype=target_dtype),
        )

    def _install(weights_mx: Dict[str, Tuple[mx.array, mx.array]]):
        installed = 0
        for orig_pref, mod_name in name_map.items():
            if mod_name not in modules_by_name:
                continue
            mod = modules_by_name[mod_name]
            a, b = weights_mx[orig_pref]
            mod.p1_lora_a = a
            mod.p1_lora_b = b
            installed += 1
        return installed

    n_full = _install(med_full_mx)
    print(f"  installed FULL medical on {n_full} modules", flush=True)
    LoRALinear.__call__ = _phase1_call
    mx.eval(model.parameters())

    eval_rows = load_eval_jsonl("medical", "valid", N_EVAL_PHASE1)
    print(f"  eval rows: {len(eval_rows)}", flush=True)

    print("  measuring FULL PPL...", flush=True)
    t = time.time()
    ppl_full = compute_ppl(model, tok, eval_rows, desc="full")
    print(f"  ppl_full = {ppl_full:.6f}  ({time.time()-t:.1f}s)", flush=True)

    # PRUNED: prune medical and reinstall
    print(f"  pruning medical adapter at keep_frac={PRUNE_KEEP_FRACTION}...", flush=True)
    med_pruned_np, f_retained_med = prune_adapter_dict(med_np, PRUNE_KEEP_FRACTION)
    f_retained_a = np.mean([f for k, f in f_retained_med.items() if k.endswith(".lora_a")])
    f_retained_b = np.mean([f for k, f in f_retained_med.items() if k.endswith(".lora_b")])
    print(f"    mean retained energy: A={f_retained_a:.4f}, B={f_retained_b:.4f}", flush=True)

    med_pruned_mx: Dict[str, Tuple[mx.array, mx.array]] = {}
    for orig_pref in name_map:
        a_key, b_key = layer_keys[orig_pref]
        med_pruned_mx[orig_pref] = (
            mx.array(med_pruned_np[a_key], dtype=target_dtype),
            mx.array(med_pruned_np[b_key], dtype=target_dtype),
        )

    n_pruned = _install(med_pruned_mx)
    print(f"  installed PRUNED medical on {n_pruned} modules", flush=True)
    mx.clear_cache()
    mx.eval(model.parameters())

    print("  measuring PRUNED PPL...", flush=True)
    t = time.time()
    ppl_pruned = compute_ppl(model, tok, eval_rows, desc="pruned")
    print(f"  ppl_pruned = {ppl_pruned:.6f}  ({time.time()-t:.1f}s)", flush=True)

    delta = ppl_pruned - ppl_full
    k1922_fire = delta > K1922_PPL_DELTA_THRESH

    print(f"\n  ΔPPL = {delta:.6f}  (threshold {K1922_PPL_DELTA_THRESH})")
    print(f"  K1922 (single-adapter pruning too aggressive) FIRES: {k1922_fire}")

    # Keep the predicted weight-space gap as a diagnostic (one layer per module class).
    sample_pref = next(iter(name_map))
    a_key, b_key = layer_keys[sample_pref]
    A = med_np[a_key]
    B = med_np[b_key]
    Ap = med_pruned_np[a_key]
    Bp = med_pruned_np[b_key]
    dW_full = ADAPTER_SCALE * (A @ B)
    dW_pruned = ADAPTER_SCALE * (Ap @ Bp)
    rel_dw = float(np.linalg.norm(dW_full - dW_pruned, ord="fro") / max(np.linalg.norm(dW_full, ord="fro"), 1e-30))
    print(f"  diagnostic: ‖ΔW − ΔW'‖_F / ‖ΔW‖_F (sample layer {sample_pref}) = {rel_dw:.4f}")

    results["phase1"] = {
        "n_modules": len(modules_by_name),
        "n_eval_rows": len(eval_rows),
        "ppl_full": ppl_full,
        "ppl_pruned": ppl_pruned,
        "ppl_delta": delta,
        "k1922_fire": k1922_fire,
        "f_retained_mean_A": float(f_retained_a),
        "f_retained_mean_B": float(f_retained_b),
        "sample_layer_relative_dw_gap": rel_dw,
        "elapsed_sec": time.time() - t0,
    }

    # Cleanup before Phase 2
    LoRALinear.__call__ = _ORIG_CALL
    del model, tok, modules_by_name, med_full_mx, med_pruned_mx
    gc.collect()
    mx.clear_cache()


# ---------- Phase 2 ----------

def phase2_composition(results: dict) -> None:
    print("\n=== PHASE 2: composition pruning (K1923) ===", flush=True)
    t0 = time.time()

    print("  loading base + medical adapter (architecture)...", flush=True)
    model, tok = load(BASE_MODEL, adapter_path=str(UPSTREAM / "medical"))
    modules_by_name = collect_lora_modules(model)
    print(f"  LoRALinear modules: {len(modules_by_name)}  (loaded in {time.time()-t0:.1f}s)", flush=True)

    med_np = load_adapter_weights_numpy("medical")
    math_np = load_adapter_weights_numpy("math")
    # Sanity: same keys
    assert set(med_np.keys()) == set(math_np.keys()), "medical/math adapter key mismatch"

    name_map = build_name_map(list(med_np.keys()), modules_by_name)
    layer_keys = group_keys_by_layer(med_np.keys())

    target_dtype = next(iter(modules_by_name.values())).lora_a.dtype

    def _to_mx(adapter_np: Dict[str, np.ndarray]) -> Dict[str, Tuple[mx.array, mx.array]]:
        out: Dict[str, Tuple[mx.array, mx.array]] = {}
        for orig_pref in name_map:
            a_key, b_key = layer_keys[orig_pref]
            out[orig_pref] = (
                mx.array(adapter_np[a_key], dtype=target_dtype),
                mx.array(adapter_np[b_key], dtype=target_dtype),
            )
        return out

    def _install_compose(med_mx, math_mx):
        installed = 0
        for orig_pref, mod_name in name_map.items():
            if mod_name not in modules_by_name:
                continue
            mod = modules_by_name[mod_name]
            mod.p2_med_a, mod.p2_med_b = med_mx[orig_pref]
            mod.p2_math_a, mod.p2_math_b = math_mx[orig_pref]
            installed += 1
        return installed

    # FULL composition
    med_full_mx = _to_mx(med_np)
    math_full_mx = _to_mx(math_np)
    n_full = _install_compose(med_full_mx, math_full_mx)
    print(f"  installed FULL med+math composition on {n_full} modules", flush=True)

    LoRALinear.__call__ = _phase2_call
    mx.eval(model.parameters())

    # Eval rows: 50 med + 50 math
    eval_rows: List[dict] = []
    eval_rows.extend(load_eval_jsonl("medical", "valid", N_EVAL_PHASE2_PER_DOMAIN))
    eval_rows.extend(load_eval_jsonl("math", "valid", N_EVAL_PHASE2_PER_DOMAIN))
    print(f"  eval rows: {len(eval_rows)} (50 med + 50 math)", flush=True)

    print("  measuring FULL composition PPL...", flush=True)
    t = time.time()
    ppl_compose_full = compute_ppl(model, tok, eval_rows, desc="compose-full")
    print(f"  ppl_compose_full = {ppl_compose_full:.6f}  ({time.time()-t:.1f}s)", flush=True)

    # PRUNED composition
    print(f"  pruning both adapters at keep_frac={PRUNE_KEEP_FRACTION}...", flush=True)
    med_pruned_np, f_med = prune_adapter_dict(med_np, PRUNE_KEEP_FRACTION)
    math_pruned_np, f_math = prune_adapter_dict(math_np, PRUNE_KEEP_FRACTION)

    f_med_a = float(np.mean([v for k, v in f_med.items() if k.endswith(".lora_a")]))
    f_med_b = float(np.mean([v for k, v in f_med.items() if k.endswith(".lora_b")]))
    f_math_a = float(np.mean([v for k, v in f_math.items() if k.endswith(".lora_a")]))
    f_math_b = float(np.mean([v for k, v in f_math.items() if k.endswith(".lora_b")]))
    print(f"    med retained: A={f_med_a:.4f}, B={f_med_b:.4f}", flush=True)
    print(f"    math retained: A={f_math_a:.4f}, B={f_math_b:.4f}", flush=True)

    med_pruned_mx = _to_mx(med_pruned_np)
    math_pruned_mx = _to_mx(math_pruned_np)
    n_pruned = _install_compose(med_pruned_mx, math_pruned_mx)
    print(f"  installed PRUNED med+math on {n_pruned} modules", flush=True)
    mx.clear_cache()
    mx.eval(model.parameters())

    print("  measuring PRUNED composition PPL...", flush=True)
    t = time.time()
    ppl_compose_pruned = compute_ppl(model, tok, eval_rows, desc="compose-pruned")
    print(f"  ppl_compose_pruned = {ppl_compose_pruned:.6f}  ({time.time()-t:.1f}s)", flush=True)

    delta_compose = ppl_compose_pruned - ppl_compose_full
    k1923_fire = delta_compose > K1923_PPL_DELTA_THRESH

    print(f"\n  ΔPPL_compose = {delta_compose:.6f}  (threshold {K1923_PPL_DELTA_THRESH})")
    print(f"  K1923 (composition pruning > 3pp degradation) FIRES: {k1923_fire}")

    LoRALinear.__call__ = _ORIG_CALL

    results["phase2"] = {
        "n_modules": len(modules_by_name),
        "n_eval_rows": len(eval_rows),
        "ppl_compose_full": ppl_compose_full,
        "ppl_compose_pruned": ppl_compose_pruned,
        "ppl_compose_delta": delta_compose,
        "k1923_fire": k1923_fire,
        "f_retained_med_A": f_med_a,
        "f_retained_med_B": f_med_b,
        "f_retained_math_A": f_math_a,
        "f_retained_math_B": f_math_b,
        "elapsed_sec": time.time() - t0,
    }

    del model, tok, modules_by_name
    gc.collect()
    mx.clear_cache()


# ---------- Main ----------

_ORIG_CALL = LoRALinear.__call__


def main() -> int:
    try:
        mlx_v = md.version("mlx")
    except Exception:
        mlx_v = "unknown"
    mlxlm_v = md.version("mlx_lm")
    print("exp_adapter_pruning_post_training")
    print(f"mlx={mlx_v}, mlx_lm={mlxlm_v}, base={BASE_MODEL}")
    print(f"prune_keep_fraction={PRUNE_KEEP_FRACTION}, scale={ADAPTER_SCALE}, seed={SEED}")

    results: dict = {
        "experiment": "exp_adapter_pruning_post_training",
        "is_smoke": False,
        "base_model": BASE_MODEL,
        "mlx_version": mlx_v,
        "mlx_lm_version": mlxlm_v,
        "adapter_source": "exp_p1_t2_single_domain_training (q_proj r=6 scale=6.0)",
        "prune_keep_fraction": PRUNE_KEEP_FRACTION,
        "adapter_scale": ADAPTER_SCALE,
        "seed": SEED,
        "n_eval_phase1": N_EVAL_PHASE1,
        "n_eval_phase2_per_domain": N_EVAL_PHASE2_PER_DOMAIN,
        "k1922_threshold": K1922_PPL_DELTA_THRESH,
        "k1923_threshold": K1923_PPL_DELTA_THRESH,
    }

    phase1_single_adapter(results)
    phase2_composition(results)

    p1 = results["phase1"]
    p2 = results["phase2"]

    k1922_fire = p1["k1922_fire"]
    k1923_fire = p2["k1923_fire"]

    # Verdict per MATH.md §Pre-registered Kill Criteria
    if k1922_fire:
        verdict = "KILLED"
        all_pass = False
    elif (not k1922_fire) and (not k1923_fire):
        verdict = "SUPPORTED"
        all_pass = True
    else:
        # k1922 pass, k1923 fire — partial result, structural finding
        verdict = "PROVISIONAL"
        all_pass = False

    results["k1922_fire"] = k1922_fire
    results["k1923_fire"] = k1923_fire
    results["verdict"] = verdict
    results["all_pass"] = all_pass

    print("\n=== FINAL ===")
    print(f"K1922 (ΔPPL_single > {K1922_PPL_DELTA_THRESH}): {'FIRE' if k1922_fire else 'pass'}")
    print(f"K1923 (ΔPPL_compose > {K1923_PPL_DELTA_THRESH}): {'FIRE' if k1923_fire else 'pass'}")
    print(f"Verdict: {verdict}")
    print(f"all_pass: {all_pass}")

    with open(EXP_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {EXP_DIR/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
