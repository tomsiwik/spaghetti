"""Validate the .pierre v1 adapter file format.

Kill criteria (from MATH.md):
  K1637: bitwise-lossless round-trip for 10 random adapters.
  K1638: every file contains magic, version=1, 64-byte sig slot, and
         manifest with the six required keys.

Runs `save → load` across the configured adapter matrix and writes
results.json. Deterministic via mx.random.seed and Python's random.seed.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import random
import struct
import sys
import tempfile
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adapter_format_v1 import (  # noqa: E402
    HEADER_LEN,
    MAGIC,
    REQUIRED_MANIFEST_KEYS,
    SIG_SLOT_LEN,
    VERSION,
    load,
    save,
)

SEED = 20260418

# PoLAR-like shapes: r ∈ {4,6,8,16}; projection dim 3072 (Gemma 4 E4B hidden).
# Target counts ∈ {1, 2, 4} expand the adapter to that many (A, B) pairs.
HIDDEN = 3072
ADAPTER_MATRIX = [
    # (rank, dtype_str, n_targets, domain)
    (4, "float32", 1, "math"),
    (6, "float32", 2, "math"),
    (6, "float16", 2, "code"),
    (6, "bfloat16", 2, "code"),
    (8, "float32", 2, "medical"),
    (8, "float16", 4, "medical"),
    (8, "bfloat16", 1, "retail"),
    (16, "float32", 2, "law"),
    (16, "float16", 4, "law"),
    (16, "bfloat16", 4, "finance"),
]


DTYPE_MAP = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}

TARGET_NAMES = ["v_proj", "o_proj", "q_proj", "k_proj"]


def _random_adapter(rank: int, dtype_str: str, n_targets: int) -> dict[str, mx.array]:
    dtype = DTYPE_MAP[dtype_str]
    weights: dict[str, mx.array] = {}
    for i in range(n_targets):
        target = TARGET_NAMES[i]
        a = mx.random.normal(shape=(rank, HIDDEN)).astype(dtype)
        b = mx.random.normal(shape=(HIDDEN, rank)).astype(dtype)
        weights[f"layer0.{target}.A"] = a
        weights[f"layer0.{target}.B"] = b
    mx.eval(*weights.values())
    return weights


def _tensor_fingerprint(weights: dict[str, mx.array]) -> dict[str, str]:
    """SHA-256 per tensor (view bf16 as uint16 for hashing)."""
    out: dict[str, str] = {}
    for name, arr in weights.items():
        mx.eval(arr)
        if arr.dtype == mx.bfloat16:
            raw = arr.view(mx.uint16)
            mx.eval(raw)
            import numpy as np

            b = np.asarray(raw).tobytes()
        else:
            import numpy as np

            b = np.asarray(arr).tobytes()
        out[name] = hashlib.sha256(b).hexdigest()
    return out


def _check_kc2_header_fields(path: str) -> tuple[bool, dict]:
    with open(path, "rb") as f:
        header = f.read(HEADER_LEN)
    checks = {}
    checks["magic_ok"] = header[0:8] == MAGIC
    (ver,) = struct.unpack("<I", header[8:12])
    checks["version_ok"] = ver == VERSION
    checks["sig_slot_zero"] = header[12:76] == bytes(SIG_SLOT_LEN)
    (mlen,) = struct.unpack("<Q", header[76:84])
    with open(path, "rb") as f:
        f.seek(HEADER_LEN)
        manifest = json.loads(f.read(mlen).decode("utf-8"))
    checks["manifest_keys_ok"] = REQUIRED_MANIFEST_KEYS.issubset(manifest.keys())
    all_ok = all(checks.values())
    return all_ok, checks


def main() -> int:
    mx.random.seed(SEED)
    random.seed(SEED)

    results = {
        "experiment": "exp_prod_adapter_format_spec_v1",
        "seed": SEED,
        "adapter_count": len(ADAPTER_MATRIX),
        "mlx_version": mx.__version__,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "per_adapter": [],
    }

    k1_passes = 0
    k2_passes = 0

    with tempfile.TemporaryDirectory() as td:
        for idx, (rank, dtype_str, n_targets, domain) in enumerate(ADAPTER_MATRIX):
            weights = _random_adapter(rank, dtype_str, n_targets)
            original_fp = _tensor_fingerprint(weights)

            manifest = {
                "spec_version": 1,
                "base_model_id": "mlx-community/gemma-4-e4b-it-4bit",
                "base_model_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "created_at": "2026-04-18T12:00:00Z",
                "signed": False,
                "signer_pubkey": "",
                "training_metadata": {
                    "adapter_type": "polar",
                    "rank": rank,
                    "targets": TARGET_NAMES[:n_targets],
                    "steps": 1000,
                    "domain": domain,
                    "loss": 0.012 + idx * 0.001,
                    "dataset": "synthetic",
                    "lora_scale": 4.0,
                    "dtype": dtype_str,
                },
            }

            path = os.path.join(td, f"adapter_{idx:02d}.pierre")
            save(path, weights, manifest)

            # Load back and compare
            loaded_weights, loaded_manifest, loaded_sig = load(path)
            reloaded_fp = _tensor_fingerprint(loaded_weights)

            # P1 + P2: tensor equality (hash identical + dtype preserved)
            fp_equal = original_fp == reloaded_fp
            dtype_preserved = all(
                loaded_weights[k].dtype == weights[k].dtype for k in weights
            )
            shapes_preserved = all(
                loaded_weights[k].shape == weights[k].shape for k in weights
            )
            # P3: manifest equality
            manifest_equal = loaded_manifest == manifest
            # KC1 (all three together)
            kc1_ok = fp_equal and dtype_preserved and shapes_preserved and manifest_equal
            if kc1_ok:
                k1_passes += 1

            # KC2: header + required fields
            kc2_ok, kc2_checks = _check_kc2_header_fields(path)
            if kc2_ok:
                k2_passes += 1

            results["per_adapter"].append(
                {
                    "idx": idx,
                    "rank": rank,
                    "dtype": dtype_str,
                    "n_targets": n_targets,
                    "domain": domain,
                    "file_bytes": os.path.getsize(path),
                    "fp_equal": fp_equal,
                    "dtype_preserved": dtype_preserved,
                    "shapes_preserved": shapes_preserved,
                    "manifest_equal": manifest_equal,
                    "sig_slot_zero": loaded_sig == bytes(SIG_SLOT_LEN),
                    "kc1_ok": kc1_ok,
                    "kc2_ok": kc2_ok,
                    "kc2_checks": kc2_checks,
                }
            )

            # Free
            del weights
            del loaded_weights
            mx.clear_cache()

    k1_pass = k1_passes == len(ADAPTER_MATRIX)
    k2_pass = k2_passes == len(ADAPTER_MATRIX)
    all_pass = k1_pass and k2_pass

    results["k1_passes"] = k1_passes
    results["k2_passes"] = k2_passes
    results["k1_pass"] = k1_pass
    results["k2_pass"] = k2_pass
    results["all_pass"] = all_pass
    results["verdict"] = "SUPPORTED" if all_pass else "KILLED"
    results["is_smoke"] = False

    out_path = HERE / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "per_adapter"}, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
