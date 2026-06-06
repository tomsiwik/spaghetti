"""exp_g4_layerwise_adapter_sensitivity — per-layer perturbation sensitivity scan on Gemma 4 E4B.

For each of the 42 decoder layers in mlx-community/gemma-4-e4b-it-4bit, inject a
relative Gaussian perturbation (eps=0.10, scaled by per-token contribution norm) into
the layer output and measure the resulting PPL on a fixed 30-row medical-MCQ batch.

KCs (per MATH.md §5, F#666 target-paired):
  K1919 (proxy):    CV(s_l) > 0.30
  K1976 (target):   top-7 sensitive layers cluster into <=3 contiguous bands with a
                     largest contiguous block of >=3 layers
"""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

import mlx.core as mx
from mlx_lm import load

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parents[2]
DATA = REPO / "micro" / "models" / "exp_p1_t2_single_domain_training" / "data"

BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"

N_EVAL = 30
MAX_SEQ_LEN = 512
SEED = 42
EPS = 0.10  # relative perturbation magnitude

# KC thresholds (locked pre-run; see MATH.md §5)
K1919_CV_THRESH = 0.30
K1976_MAX_BANDS = 3
K1976_MIN_LARGEST_BAND = 3

mx.random.seed(SEED)


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
    if total_tokens == 0:
        return float("inf")
    elapsed = time.time() - t0
    print(f"    [{desc}] {len(rows)} rows in {elapsed:.1f}s -> ppl={math.exp(total_nll/total_tokens):.4f}", flush=True)
    return math.exp(total_nll / total_tokens)


# ---------- per-layer perturbation patch ----------
#
# Strategy: class-level monkey-patch of DecoderLayer.__call__ once at startup,
# routing through a small dispatcher that reads `self._perturb_eps` (default
# None). Only the target layer has the attribute set; all others fall through
# to the original call. This avoids the Python type-level lookup of __call__
# (instance-level monkey-patching does not override `obj()` for nn.Module
# subclasses, since `type(obj).__call__` is what Python invokes).

_ORIGINAL_DECODER_CALL = None


def _install_decoder_dispatch(decoder_cls):
    """Install class-level dispatch that respects per-instance _perturb_eps."""
    global _ORIGINAL_DECODER_CALL
    if _ORIGINAL_DECODER_CALL is not None:
        return  # already installed
    _ORIGINAL_DECODER_CALL = decoder_cls.__call__

    def dispatched(self, x, *args, **kwargs):
        result = _ORIGINAL_DECODER_CALL(self, x, *args, **kwargs)
        eps = getattr(self, "_perturb_eps", None)
        if eps is None:
            return result
        # Result is (h, shared_kv, offset) for Gemma 4
        if isinstance(result, tuple):
            h = result[0]
            rest = result[1:]
        else:
            h = result
            rest = ()
        delta = h - x
        d = h.shape[-1]
        # per-token RMS contribution norm: shape (B, T, 1)
        delta_norm = mx.sqrt(mx.sum(delta * delta, axis=-1, keepdims=True) / d)
        seed = int(getattr(self, "_perturb_seed", 1000))
        key = mx.random.key(seed)
        noise = mx.random.normal(shape=h.shape, key=key, dtype=h.dtype)
        h_pert = h + (eps * delta_norm * noise).astype(h.dtype)
        if rest:
            return (h_pert,) + rest
        return h_pert

    decoder_cls.__call__ = dispatched


class layer_perturbation:
    """Context manager: enable perturbation on ONE layer, disable on exit."""

    def __init__(self, layer, eps: float, layer_idx: int):
        self.layer = layer
        self.eps = eps
        self.layer_idx = layer_idx

    def __enter__(self):
        self.layer._perturb_eps = self.eps
        self.layer._perturb_seed = 1000 + self.layer_idx
        return self

    def __exit__(self, *exc):
        try:
            del self.layer._perturb_eps
        except AttributeError:
            pass
        try:
            del self.layer._perturb_seed
        except AttributeError:
            pass


def patch_layer(model, layer_idx: int, eps: float):
    return layer_perturbation(model.layers[layer_idx], eps, layer_idx)


# ---------- band/contiguity analysis ----------

def find_contiguous_bands(indices: List[int]) -> List[List[int]]:
    """Group sorted indices into contiguous-integer bands.
    [3,4,5,8,9,15] -> [[3,4,5],[8,9],[15]]
    """
    if not indices:
        return []
    sorted_idx = sorted(indices)
    bands = [[sorted_idx[0]]]
    for i in sorted_idx[1:]:
        if i == bands[-1][-1] + 1:
            bands[-1].append(i)
        else:
            bands.append([i])
    return bands


# ---------- main ----------

def main():
    results: dict = {
        "experiment_id": "exp_g4_layerwise_adapter_sensitivity",
        "config": {
            "base_model": BASE_MODEL,
            "n_eval": N_EVAL,
            "max_seq_len": MAX_SEQ_LEN,
            "seed": SEED,
            "eps": EPS,
            "k1919_cv_thresh": K1919_CV_THRESH,
            "k1976_max_bands": K1976_MAX_BANDS,
            "k1976_min_largest_band": K1976_MIN_LARGEST_BAND,
        },
        "phase1_per_layer_ppl": {},
        "kill_criteria": {},
        "verdict": None,
        "all_pass": None,
    }

    print("=== Loading base model ===", flush=True)
    t0 = time.time()
    model, tok = load(BASE_MODEL)
    n_layers = len(model.layers)
    print(f"  loaded in {time.time()-t0:.1f}s; n_layers={n_layers}", flush=True)
    assert n_layers == 42, f"expected 42 layers, got {n_layers}"

    # Install class-level dispatch on the actual DecoderLayer class
    decoder_cls = type(model.layers[0])
    _install_decoder_dispatch(decoder_cls)
    print(f"  installed dispatch on {decoder_cls.__name__}", flush=True)

    # 30 rows medical valid
    rows = load_eval_jsonl("medical", "valid", N_EVAL)
    print(f"  {len(rows)} eval rows loaded", flush=True)

    print("\n=== Phase 0: baseline PPL (no perturbation) ===", flush=True)
    ppl_base = compute_ppl(model, tok, rows, desc="baseline")
    results["baseline_ppl"] = ppl_base
    mx.clear_cache()

    print("\n=== Phase 1: per-layer perturbation scan ===", flush=True)
    print(f"  eps={EPS}, n_layers={n_layers}, n_rows={len(rows)}", flush=True)
    per_layer_ppl = {}
    t_phase = time.time()
    for li in range(n_layers):
        with patch_layer(model, li, EPS):
            ppl_l = compute_ppl(model, tok, rows, desc=f"L{li:02d}")
        per_layer_ppl[li] = ppl_l
        results["phase1_per_layer_ppl"][li] = ppl_l
        # Periodic flush of intermediate results
        if (li + 1) % 5 == 0:
            elapsed = time.time() - t_phase
            print(f"  progress {li+1}/{n_layers} layers, {elapsed:.0f}s elapsed", flush=True)
            with open(EXP_DIR / "results.json", "w") as fh:
                json.dump(results, fh, indent=2)
        mx.clear_cache()

    print("\n=== Phase 2: KC evaluation ===", flush=True)
    s = np.array([per_layer_ppl[li] - ppl_base for li in range(n_layers)], dtype=np.float64)

    # Avoid degenerate CV when mean is near zero or negative-mean (rare)
    mean_s = float(np.mean(s))
    std_s = float(np.std(s, ddof=0))
    cv = std_s / abs(mean_s) if abs(mean_s) > 1e-9 else 0.0
    range_ratio = float(np.max(s) / np.min(s)) if np.min(s) > 0 else float("inf")

    results["s_per_layer"] = {int(i): float(v) for i, v in enumerate(s)}
    results["mean_s"] = mean_s
    results["std_s"] = std_s
    results["cv_s"] = cv
    results["range_ratio"] = range_ratio
    results["max_layer"] = int(np.argmax(s))
    results["min_layer"] = int(np.argmin(s))

    # K1919: CV > 0.30 PASS
    k1919_pass = bool(cv > K1919_CV_THRESH)
    results["kill_criteria"]["1919"] = {
        "text": "All layers equally sensitive — CV(s_l) > 0.30 differentiates",
        "value": cv,
        "thresh": K1919_CV_THRESH,
        "result": "pass" if k1919_pass else "fail",
        "type": "proxy_structural",
    }

    # K1976: top-7 sensitivity layers contiguity
    top7 = list(np.argsort(s)[::-1][:7])
    bands = find_contiguous_bands([int(x) for x in top7])
    largest_band_size = max((len(b) for b in bands), default=0)
    n_bands = len(bands)
    k1976_pass = bool(n_bands <= K1976_MAX_BANDS and largest_band_size >= K1976_MIN_LARGEST_BAND)
    results["top7_layers"] = [int(x) for x in top7]
    results["bands"] = [[int(x) for x in b] for b in bands]
    results["n_bands"] = n_bands
    results["largest_band_size"] = largest_band_size
    results["kill_criteria"]["1976"] = {
        "text": "Top-7 most-sensitive layers form actionable contiguous band(s)",
        "n_bands": n_bands,
        "largest_band": largest_band_size,
        "max_bands_thresh": K1976_MAX_BANDS,
        "min_largest_thresh": K1976_MIN_LARGEST_BAND,
        "result": "pass" if k1976_pass else "fail",
        "type": "target_actionable",
    }

    all_pass = k1919_pass and k1976_pass
    all_fail = (not k1919_pass) and (not k1976_pass)
    if all_pass:
        verdict = "SUPPORTED"
    elif all_fail:
        verdict = "KILLED"
    else:
        verdict = "PROVISIONAL"
    results["verdict"] = verdict
    results["all_pass"] = all_pass

    # Predictions vs. measurements
    predictions = {
        "P1_cv_gt_0_30": {"predicted": "> 0.30", "measured": cv, "pass": k1919_pass},
        "P2_range_ratio_gt_3": {"predicted": "> 3.0", "measured": range_ratio, "pass": bool(range_ratio > 3.0)},
        "P3_top7_le_3_bands": {"predicted": "<= 3 bands", "measured": n_bands, "pass": bool(n_bands <= 3)},
        "P4_band_intersects_16_31": {
            "predicted": "top band overlaps [16,31]",
            "measured": [int(x) for x in top7],
            "pass": bool(any(16 <= li <= 31 for li in top7)),
        },
    }
    results["predictions"] = predictions

    print(f"\n=== Verdict: {verdict} ===", flush=True)
    print(f"  CV(s_l) = {cv:.4f} (thresh > {K1919_CV_THRESH})  -> K1919 {'PASS' if k1919_pass else 'FAIL'}", flush=True)
    print(f"  top-7 layers = {[int(x) for x in top7]}", flush=True)
    print(f"  bands = {[[int(x) for x in b] for b in bands]} (n={n_bands}, largest={largest_band_size})", flush=True)
    print(f"  K1976 {'PASS' if k1976_pass else 'FAIL'}", flush=True)

    with open(EXP_DIR / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to {EXP_DIR/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
