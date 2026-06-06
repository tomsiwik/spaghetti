"""exp_g4_adapter_magnitude_distribution — structure and importance of LoRA weights on Gemma 4 E4B.

Phased execution (memory-safe, mlx-dev pattern):
  Phase 1 — pure numpy analysis of safetensor weights (K1917 gaussianity, SVD, sparsity).
  Phase 2 — per-domain, with model loaded:
       baseline PPL, Fisher-proxy gradients (K1918), top-20% / random-20% ablation PPL (K1971/K1972).
       mx.clear_cache() between domains; del model before final eval.

See MATH.md for theorem, predictions, kill criteria.
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
from scipy import stats

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parents[2]
UPSTREAM = REPO / "micro" / "models" / "exp_p1_t2_single_domain_training" / "adapters"
DATA = REPO / "micro" / "models" / "exp_p1_t2_single_domain_training" / "data"

BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
DOMAINS = ["medical", "math", "code"]

# Eval + ablation parameters (MATH.md §Methodology).
N_EVAL = 100            # held-out samples for PPL
GRAD_BATCH = 8          # batch size for Fisher-proxy gradient pass
MAX_SEQ_LEN = 512
SEED = 42
PRUNE_FRAC = 0.20
NUM_LAYERS_GEMMA4_E4B = 42  # sanity check against adapter

# KC thresholds (MATH.md §Pre-registered Kill Criteria).
K1917_GAUSSIAN_FRAC = 0.80
K1917_SHAPIRO_P = 0.01
K1917_MAX_SKEW = 0.5
K1917_MAX_KURT = 1.0
K1918_R_THRESH = 0.3
K1971_RATIO = 0.5

mx.random.seed(SEED)
rng = np.random.default_rng(SEED)


# ---------- Phase 1 helpers ----------

def load_adapter_weights_numpy(domain: str) -> Dict[str, np.ndarray]:
    """Load safetensor adapter as dict {full_key: numpy array}."""
    path = UPSTREAM / domain / "adapters.safetensors"
    assert path.exists(), f"missing: {path}"
    out: Dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k).astype(np.float64)
    return out


def normality_stats(w: np.ndarray) -> Tuple[float, float, float]:
    """Return (shapiro_p, skew, excess_kurtosis) on flattened w; subsample to 5000 if needed."""
    flat = w.ravel()
    if flat.size > 5000:
        idx = rng.choice(flat.size, size=5000, replace=False)
        sample = flat[idx]
    else:
        sample = flat
    try:
        _, p = stats.shapiro(sample)
    except Exception:
        p = 0.0
    skew = float(stats.skew(flat))
    kurt = float(stats.kurtosis(flat))  # fisher definition (excess)
    return float(p), skew, kurt


def effective_rank_99(A: np.ndarray) -> int:
    """Number of singular values needed to reach 99% cumulative variance."""
    s = np.linalg.svd(A, compute_uv=False)
    s2 = s * s
    cum = np.cumsum(s2) / s2.sum()
    return int(np.searchsorted(cum, 0.99) + 1)


def phase1_structural(results: dict) -> None:
    """K1917 — per-matrix normality, SVD spectrum, sparsity. Pure numpy."""
    print("\n=== PHASE 1: Structural analysis (K1917) ===")
    per_matrix = []
    for dom in DOMAINS:
        print(f"  {dom}...", flush=True)
        weights = load_adapter_weights_numpy(dom)
        for key, w in weights.items():
            p, skew, kurt = normality_stats(w)
            is_lora_a = key.endswith(".lora_a")
            effrank = effective_rank_99(w) if is_lora_a else None
            sparsity = float(np.mean(np.abs(w) < 1e-4))
            per_matrix.append({
                "domain": dom,
                "key": key,
                "kind": "lora_a" if is_lora_a else "lora_b",
                "shape": list(w.shape),
                "shapiro_p": p,
                "skew": skew,
                "excess_kurtosis": kurt,
                "effective_rank_99": effrank,
                "sparsity_lt_1e4": sparsity,
                "mean": float(np.mean(w)),
                "std": float(np.std(w)),
            })
    # Aggregate K1917
    n = len(per_matrix)
    shapiro_pass_frac = sum(1 for m in per_matrix if m["shapiro_p"] > K1917_SHAPIRO_P) / n
    skew_pass_frac = sum(1 for m in per_matrix if abs(m["skew"]) < K1917_MAX_SKEW) / n
    kurt_pass_frac = sum(1 for m in per_matrix if abs(m["excess_kurtosis"]) < K1917_MAX_KURT) / n
    all_three_pass_frac = sum(
        1 for m in per_matrix
        if m["shapiro_p"] > K1917_SHAPIRO_P
        and abs(m["skew"]) < K1917_MAX_SKEW
        and abs(m["excess_kurtosis"]) < K1917_MAX_KURT
    ) / n
    # Effective rank across lora_a matrices (sanity P2)
    effranks = [m["effective_rank_99"] for m in per_matrix if m["effective_rank_99"] is not None]
    mean_effrank = float(np.mean(effranks)) if effranks else 0.0
    # Sparsity
    mean_sparsity = float(np.mean([m["sparsity_lt_1e4"] for m in per_matrix]))
    # K1917 fires iff all_three_pass_frac >= K1917_GAUSSIAN_FRAC
    k1917_fire = all_three_pass_frac >= K1917_GAUSSIAN_FRAC

    print(f"\n  n_matrices: {n}")
    print(f"  shapiro p>{K1917_SHAPIRO_P} frac: {shapiro_pass_frac:.3f}")
    print(f"  |skew|<{K1917_MAX_SKEW} frac: {skew_pass_frac:.3f}")
    print(f"  |excess kurt|<{K1917_MAX_KURT} frac: {kurt_pass_frac:.3f}")
    print(f"  all-three pass frac: {all_three_pass_frac:.3f}")
    print(f"  mean effective_rank_99 (lora_a, expected ~6): {mean_effrank:.3f}")
    print(f"  mean sparsity (|w|<1e-4): {mean_sparsity:.4f}")
    print(f"  K1917 (Gaussian structure) FIRES: {k1917_fire}")

    results["phase1"] = {
        "n_matrices": n,
        "shapiro_pass_frac": shapiro_pass_frac,
        "skew_pass_frac": skew_pass_frac,
        "kurt_pass_frac": kurt_pass_frac,
        "all_three_pass_frac": all_three_pass_frac,
        "mean_effective_rank_99": mean_effrank,
        "mean_sparsity_lt_1e4": mean_sparsity,
        "k1917_fire": k1917_fire,
        "per_matrix": per_matrix,
    }


# ---------- Phase 2 helpers ----------

def collect_lora_modules(model: nn.Module) -> List[Tuple[str, LoRALinear]]:
    """Return list of (dotted_path, module) for each LoRALinear via named_modules()."""
    found: List[Tuple[str, LoRALinear]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            found.append((name, mod))
    return found


def snapshot_lora(modules: List[Tuple[str, LoRALinear]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Capture a numpy snapshot of lora_a and lora_b for restoration."""
    snap = {}
    for path, mod in modules:
        a = np.array(mod.lora_a, copy=True)
        b = np.array(mod.lora_b, copy=True)
        snap[path] = (a, b)
    return snap


def restore_lora(modules: List[Tuple[str, LoRALinear]], snap: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    for path, mod in modules:
        a, b = snap[path]
        mod.lora_a = mx.array(a, dtype=mod.lora_a.dtype)
        mod.lora_b = mx.array(b, dtype=mod.lora_b.dtype)


def apply_topk_mask(modules: List[Tuple[str, LoRALinear]], frac: float) -> None:
    """In-place: zero the top-frac magnitude entries per matrix (both lora_a and lora_b)."""
    for path, mod in modules:
        for attr in ("lora_a", "lora_b"):
            arr_mx = getattr(mod, attr)
            arr_np = np.array(arr_mx)
            flat = np.abs(arr_np).ravel()
            k = max(1, int(round(frac * flat.size)))
            # threshold = k-th largest
            thresh = np.partition(flat, -k)[-k]
            mask = np.abs(arr_np) < thresh  # keep only bottom entries
            # tie-break: when equal to threshold, zero up to k total
            eq = (np.abs(arr_np) == thresh)
            # randomize tie breaking deterministically
            eq_flat = eq.ravel()
            # we already zero entries >= thresh; count how many we need to zero
            zero_so_far = int((~mask).sum())
            if zero_so_far > k:
                # need to un-zero some equal-to-thresh entries
                eq_positions = np.where(eq_flat)[0]
                to_keep = zero_so_far - k
                if to_keep > 0 and len(eq_positions) > 0:
                    chosen = rng.choice(eq_positions, size=min(to_keep, len(eq_positions)), replace=False)
                    flat_mask = mask.ravel().copy()
                    flat_mask[chosen] = True
                    mask = flat_mask.reshape(arr_np.shape)
            new = np.where(mask, arr_np, 0.0).astype(arr_np.dtype)
            setattr(mod, attr, mx.array(new))


def apply_random_mask(modules: List[Tuple[str, LoRALinear]], frac: float, seed_offset: int = 0) -> None:
    """In-place: zero a random frac of entries per matrix (deterministic from SEED+offset)."""
    local_rng = np.random.default_rng(SEED + seed_offset)
    for path, mod in modules:
        for attr in ("lora_a", "lora_b"):
            arr_mx = getattr(mod, attr)
            arr_np = np.array(arr_mx)
            flat_size = arr_np.size
            k = max(1, int(round(frac * flat_size)))
            idx = local_rng.choice(flat_size, size=k, replace=False)
            flat = arr_np.ravel().copy()
            flat[idx] = 0.0
            new = flat.reshape(arr_np.shape).astype(arr_np.dtype)
            setattr(mod, attr, mx.array(new))


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


def compute_ppl(model: nn.Module, tok, rows: List[dict], desc: str = "") -> Tuple[float, int]:
    """Teacher-forcing NLL / token across rows; returns (ppl, total_tokens)."""
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
        x = mx.array([ids[:-1]])   # (1, T-1)
        y = mx.array([ids[1:]])    # (1, T-1)
        logits = model(x)           # (1, T-1, V)
        # cross-entropy per token; reduction sum then track token count
        lse = mx.logsumexp(logits, axis=-1)  # (1, T-1)
        # gather y
        tgt = mx.take_along_axis(logits, y[..., None], axis=-1).squeeze(-1)  # (1, T-1)
        nll = (lse - tgt).sum()
        mx.eval(nll)
        total_nll += float(nll.item())
        total_tokens += int(y.shape[-1])
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"    [{desc}] {i+1}/{len(rows)} rows, elapsed {elapsed:.1f}s, nll/tok={total_nll/max(1,total_tokens):.4f}", flush=True)
    if total_tokens == 0:
        return float("inf"), 0
    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return ppl, total_tokens


def compute_fisher_correlation(
    model: nn.Module,
    tok,
    modules: List[Tuple[str, LoRALinear]],
    rows: List[dict],
    batch_size: int,
) -> Tuple[float, List[float]]:
    """Single-batch Fisher proxy: g(w)² for LoRA params. Returns (mean |r|, per-matrix |r|)."""
    # Build a batched tokenization, right-pad to common length up to MAX_SEQ_LEN.
    token_seqs = []
    for row in rows[:batch_size]:
        text = format_chat(tok, row["messages"])
        ids = tok.encode(text)
        if len(ids) > MAX_SEQ_LEN:
            ids = ids[:MAX_SEQ_LEN]
        if len(ids) >= 2:
            token_seqs.append(ids)
    if not token_seqs:
        return 0.0, []
    T = max(len(s) for s in token_seqs)
    pad_id = tok.pad_token_id if getattr(tok, "pad_token_id", None) is not None else 0
    x_np = np.full((len(token_seqs), T - 1), pad_id, dtype=np.int32)
    y_np = np.full((len(token_seqs), T - 1), pad_id, dtype=np.int32)
    mask_np = np.zeros((len(token_seqs), T - 1), dtype=np.float32)
    for i, s in enumerate(token_seqs):
        x_np[i, : len(s) - 1] = s[:-1]
        y_np[i, : len(s) - 1] = s[1:]
        mask_np[i, : len(s) - 1] = 1.0
    x = mx.array(x_np)
    y = mx.array(y_np)
    mask = mx.array(mask_np)

    # Set training mode for gradient-enabled forward
    model.train()
    # Freeze all non-LoRA parameters so grads are only through lora_a/lora_b.
    # mlx_lm pattern: call model.freeze() then unfreeze LoRALinear modules.
    if hasattr(model, "freeze"):
        model.freeze()
    for _, mod in modules:
        if hasattr(mod, "unfreeze"):
            mod.unfreeze()

    def loss_fn(m, x_in, y_in, mask_in):
        logits = m(x_in)
        lse = mx.logsumexp(logits, axis=-1)
        tgt = mx.take_along_axis(logits, y_in[..., None], axis=-1).squeeze(-1)
        per_tok = (lse - tgt) * mask_in
        return per_tok.sum() / mask_in.sum()

    grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = grad_fn(model, x, y, mask)
    mx.eval(loss, grads)

    # Walk the grads tree, find entries corresponding to lora_a and lora_b.
    # grads mirrors model.parameters() structure.
    correlations: List[float] = []

    for path, mod in modules:
        # Get weight values.
        w_a = np.array(mod.lora_a).ravel()
        w_b = np.array(mod.lora_b).ravel()
        # Navigate grads tree by path.
        g_node = grads
        for p in path.split("."):
            if p.isdigit():
                g_node = g_node[int(p)]
            else:
                g_node = g_node[p] if isinstance(g_node, dict) else getattr(g_node, p, None)
            if g_node is None:
                break
        if g_node is None or "lora_a" not in g_node or "lora_b" not in g_node:
            continue
        ga = np.array(g_node["lora_a"]).ravel()
        gb = np.array(g_node["lora_b"]).ravel()
        # Fisher proxy = g²; correlate |w| with g² per matrix.
        for w_flat, g_flat in [(w_a, ga), (w_b, gb)]:
            if w_flat.size != g_flat.size or w_flat.size < 2:
                continue
            ws = np.abs(w_flat)
            gs = g_flat * g_flat
            if ws.std() < 1e-12 or gs.std() < 1e-12:
                correlations.append(0.0)
                continue
            r = float(np.corrcoef(ws, gs)[0, 1])
            if not np.isfinite(r):
                r = 0.0
            correlations.append(abs(r))

    mean_r = float(np.mean(correlations)) if correlations else 0.0
    return mean_r, correlations


def phase2_behavioral(results: dict) -> None:
    """K1918 (Fisher correlation) + K1971/K1972 (ablation PPL)."""
    print("\n=== PHASE 2: Behavioral analysis (K1918, K1971, K1972) ===")
    per_domain: Dict[str, dict] = {}
    for dom in DOMAINS:
        print(f"\n--- domain: {dom} ---", flush=True)
        mx.clear_cache()
        gc.collect()

        adapter_path = str(UPSTREAM / dom)
        print(f"  loading base + adapter from {adapter_path}", flush=True)
        t0 = time.time()
        model, tok = load(BASE_MODEL, adapter_path=adapter_path)
        print(f"  model loaded in {time.time()-t0:.1f}s", flush=True)

        modules = collect_lora_modules(model)
        print(f"  LoRA modules found: {len(modules)}")
        assert len(modules) > 0, f"no LoRALinear modules on {dom}"

        snap = snapshot_lora(modules)

        # Load eval data (valid.jsonl is small; fall back to train.jsonl if needed)
        valid_path = DATA / dom / "valid.jsonl"
        eval_rows = load_eval_jsonl(dom, "valid", N_EVAL) if valid_path.exists() else []
        if len(eval_rows) < N_EVAL:
            extra = load_eval_jsonl(dom, "train", N_EVAL - len(eval_rows))
            eval_rows = eval_rows + extra
        eval_rows = eval_rows[:N_EVAL]
        print(f"  eval rows: {len(eval_rows)}")

        # Baseline PPL
        print("  measuring baseline PPL...", flush=True)
        ppl_base, _ = compute_ppl(model, tok, eval_rows, desc=f"{dom}/base")
        print(f"  baseline PPL: {ppl_base:.4f}")

        # K1918: Fisher correlation (before any ablation; uses snapshot state)
        print("  computing Fisher-proxy correlation...", flush=True)
        try:
            fisher_mean_r, fisher_per_mat = compute_fisher_correlation(
                model, tok, modules, eval_rows, GRAD_BATCH
            )
        except Exception as e:
            print(f"  WARN Fisher failed: {e}", flush=True)
            fisher_mean_r, fisher_per_mat = float("nan"), []
        print(f"  Fisher-proxy |r| mean: {fisher_mean_r:.4f} (n={len(fisher_per_mat)})")

        # Restore (Fisher pass may have left model in train mode; reset)
        model.eval()
        restore_lora(modules, snap)
        mx.eval(model.parameters())

        # Top-k magnitude pruning
        print(f"  applying top-{int(PRUNE_FRAC*100)}% magnitude mask...", flush=True)
        apply_topk_mask(modules, PRUNE_FRAC)
        mx.eval(model.parameters())
        ppl_top, _ = compute_ppl(model, tok, eval_rows, desc=f"{dom}/top")
        print(f"  top-pruned PPL: {ppl_top:.4f}")

        # Restore + random-k
        restore_lora(modules, snap)
        mx.eval(model.parameters())
        print(f"  applying random-{int(PRUNE_FRAC*100)}% mask...", flush=True)
        apply_random_mask(modules, PRUNE_FRAC, seed_offset=hash(dom) % 1024)
        mx.eval(model.parameters())
        ppl_rand, _ = compute_ppl(model, tok, eval_rows, desc=f"{dom}/rand")
        print(f"  random-pruned PPL: {ppl_rand:.4f}")

        # Restore for cleanliness
        restore_lora(modules, snap)

        dppl_top = ppl_top - ppl_base
        dppl_rand = ppl_rand - ppl_base
        denom = max(abs(dppl_top), abs(dppl_rand), 1e-6)
        R = abs(dppl_top - dppl_rand) / denom
        per_domain[dom] = {
            "ppl_baseline": ppl_base,
            "ppl_topk_pruned": ppl_top,
            "ppl_random_pruned": ppl_rand,
            "dPPL_top": dppl_top,
            "dPPL_rand": dppl_rand,
            "ratio_R": R,
            "fisher_mean_abs_r": fisher_mean_r,
            "fisher_per_matrix_count": len(fisher_per_mat),
        }
        print(f"  ΔPPL_top={dppl_top:+.4f}, ΔPPL_rand={dppl_rand:+.4f}, R={R:.4f}")

        # Cleanup: delete model, clear cache
        del model, tok, modules, snap
        gc.collect()
        mx.clear_cache()

    # K1918
    fisher_rs = [d["fisher_mean_abs_r"] for d in per_domain.values() if np.isfinite(d["fisher_mean_abs_r"])]
    mean_fisher_r = float(np.mean(fisher_rs)) if fisher_rs else float("nan")
    k1918_fire = (np.isfinite(mean_fisher_r)) and (mean_fisher_r < K1918_R_THRESH)

    # K1971 / K1972 (shared measurement): mean R across domains
    rs = [d["ratio_R"] for d in per_domain.values()]
    mean_R = float(np.mean(rs)) if rs else float("nan")
    k1971_fire = (np.isfinite(mean_R)) and (mean_R < K1971_RATIO)
    k1972_fire = k1971_fire  # same test, same threshold per MATH.md §Methodology

    results["phase2"] = {
        "per_domain": per_domain,
        "mean_fisher_abs_r": mean_fisher_r,
        "mean_R_ratio": mean_R,
        "k1918_fire": k1918_fire,
        "k1971_fire": k1971_fire,
        "k1972_fire": k1972_fire,
    }


# ---------- Main ----------

def main() -> int:
    mlx_v = md.version("mlx")
    mlxlm_v = md.version("mlx_lm")
    print(f"exp_g4_adapter_magnitude_distribution")
    print(f"mlx={mlx_v}, mlx_lm={mlxlm_v}, base={BASE_MODEL}")
    print(f"domains: {DOMAINS}, N_EVAL={N_EVAL}, prune_frac={PRUNE_FRAC}, seed={SEED}")

    results: dict = {
        "experiment": "exp_g4_adapter_magnitude_distribution",
        "is_smoke": False,
        "base_model": BASE_MODEL,
        "mlx_version": mlx_v,
        "mlx_lm_version": mlxlm_v,
        "adapter_source": "exp_p1_t2_single_domain_training (q_proj r=6 scale=6.0)",
        "domains": DOMAINS,
        "N_EVAL": N_EVAL,
        "prune_frac": PRUNE_FRAC,
        "seed": SEED,
    }

    phase1_structural(results)
    phase2_behavioral(results)

    # Apply F#666 target-gated kill rule
    p1 = results["phase1"]
    p2 = results["phase2"]

    # K1 (structure): K1917 proxy AND K1971 target must both fire for kill
    k1_proxy = p1["k1917_fire"]
    k1_target = p2["k1971_fire"]
    k1_killed = k1_proxy and k1_target

    # K2 (magnitude=importance): K1918 proxy AND K1972 target must both fire for kill
    k2_proxy = p2["k1918_fire"]
    k2_target = p2["k1972_fire"]
    k2_killed = k2_proxy and k2_target

    all_fired = k1_killed and k2_killed
    all_supported = (not k1_proxy) and (not k1_target) and (not k2_proxy) and (not k2_target)

    if all_fired:
        verdict = "KILLED"
        all_pass = False
    elif all_supported:
        verdict = "SUPPORTED"
        all_pass = True
    elif (k1_proxy != k1_target) or (k2_proxy != k2_target):
        # proxy/target mismatch within a KC → F#666 provisional
        verdict = "PROVISIONAL_F666_MISMATCH"
        all_pass = False
    else:
        verdict = "MIXED"
        all_pass = False

    results["k1917_fire"] = p1["k1917_fire"]
    results["k1918_fire"] = p2["k1918_fire"]
    results["k1971_fire"] = p2["k1971_fire"]
    results["k1972_fire"] = p2["k1972_fire"]
    results["k1_killed"] = k1_killed
    results["k2_killed"] = k2_killed
    results["verdict"] = verdict
    results["all_pass"] = all_pass

    print("\n=== FINAL ===")
    print(f"K1917 (proxy:Gaussian)           : {'FIRE' if k1_proxy else 'no'}")
    print(f"K1971 (target:compression beh.) : {'FIRE' if k1_target else 'no'}")
    print(f"K1918 (proxy:|r|<0.3)             : {'FIRE' if k2_proxy else 'no'}")
    print(f"K1972 (target:ablation ratio)    : {'FIRE' if k2_target else 'no'}")
    print(f"K1 killed (both K1917+K1971 fire): {k1_killed}")
    print(f"K2 killed (both K1918+K1972 fire): {k2_killed}")
    print(f"Verdict: {verdict}")

    with open(EXP_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {EXP_DIR/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
