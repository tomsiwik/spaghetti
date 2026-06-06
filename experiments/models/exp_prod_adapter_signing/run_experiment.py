"""Validate Ed25519 signing + verification for the .pierre v1 format.

Kill criteria (from MATH.md):
  K1639: Tampered adapter (1 byte flipped outside sig slot) rejected
         100/100.
  K1640: Unsigned adapter rejected by default; allow_unsigned=True
         loads it; signed paths behave correctly across 5 cases.
  K1641: verify_file adds <100 ms median overhead on a 400 KB adapter
         over 50 paired loads.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import random
import struct
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adapter_signing import (  # noqa: E402
    SignatureMismatch,
    SignatureRequired,
    UnsignedRejected,
    load_verified,
    save_signed,
    verify_file,
)

# Reach v1 helpers too (save unsigned, header layout).
V1_DIR = HERE.parent / "exp_prod_adapter_format_spec_v1"
sys.path.insert(0, str(V1_DIR))
from adapter_format_v1 import HEADER_LEN, SIG_SLOT_LEN, save  # noqa: E402

SEED = 20260418

HIDDEN = 3072
DTYPE_MAP = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
TARGET_NAMES = ["v_proj", "o_proj", "q_proj", "k_proj"]

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


def _manifest(idx: int, rank: int, n_targets: int, domain: str) -> dict:
    return {
        "spec_version": 1,
        "base_model_id": "mlx-community/gemma-4-e4b-it-4bit",
        "base_model_hash": "sha256:" + "0" * 64,
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
            "dataset": f"{domain}_v1",
            "lora_scale": 4.0,
        },
    }


def _random_adapter(rank: int, dtype_str: str, n_targets: int) -> dict[str, mx.array]:
    dtype = DTYPE_MAP[dtype_str]
    out: dict[str, mx.array] = {}
    for i in range(n_targets):
        target = TARGET_NAMES[i]
        a = mx.random.normal(shape=(rank, HIDDEN)).astype(dtype)
        b = mx.random.normal(shape=(HIDDEN, rank)).astype(dtype)
        out[f"layer0.{target}.A"] = a
        out[f"layer0.{target}.B"] = b
    mx.eval(*out.values())
    return out


def _flip_one_byte_outside_sigslot(path: str, rng: random.Random) -> int:
    """Flip a random byte at an offset >= 84 (inside manifest or
    safetensors blob), outside the sig slot [12:76] and outside the
    magic/version/manifest_len header.

    Returns the offset flipped.
    """
    size = os.path.getsize(path)
    # Manifest starts at HEADER_LEN (84). Pick uniformly in [HEADER_LEN, size).
    offset = rng.randrange(HEADER_LEN, size)
    with open(path, "r+b") as f:
        f.seek(offset)
        b = f.read(1)
        flipped = bytes([b[0] ^ 0xFF])
        f.seek(offset)
        f.write(flipped)
    return offset


# ---------------------------------------------------------------------------
# K1639 — tamper rejection across 100 trials
# ---------------------------------------------------------------------------
def run_k1639(rng: random.Random, tmp: Path) -> dict:
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()

    reject_count = 0
    accept_count = 0
    trials = 100
    per_trial = []

    for i in range(trials):
        rank, dtype_str, n_targets, domain = ADAPTER_MATRIX[i % len(ADAPTER_MATRIX)]
        w = _random_adapter(rank, dtype_str, n_targets)
        m = _manifest(i, rank, n_targets, domain)
        path = tmp / f"tamper_{i:03d}.pierre"
        save_signed(str(path), w, m, priv)

        # sanity: original verifies before tamper
        pre_tamper = verify_file(str(path), pub)
        if not pre_tamper:
            # A pre-tamper failure is itself a KC1639 failure signal —
            # record and treat as accepted (worst case for caller).
            per_trial.append({
                "trial": i, "offset": None,
                "pre_tamper_verify": False, "post_tamper_verify": None,
                "rejected": False,
            })
            accept_count += 1
            continue

        offset = _flip_one_byte_outside_sigslot(str(path), rng)
        post_tamper = verify_file(str(path), pub)
        rejected = (post_tamper is False)
        if rejected:
            reject_count += 1
        else:
            accept_count += 1
        per_trial.append({
            "trial": i, "offset": offset,
            "pre_tamper_verify": True,
            "post_tamper_verify": post_tamper,
            "rejected": rejected,
        })

    passed = (reject_count == trials)
    return {
        "kc_id": 1639,
        "trials": trials,
        "reject_count": reject_count,
        "accept_count": accept_count,
        "passed": passed,
        "per_trial": per_trial,
    }


# ---------------------------------------------------------------------------
# K1640 — unsigned rejected by default; five-case spec conformance
# ---------------------------------------------------------------------------
def run_k1640(tmp: Path) -> dict:
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    wrong_priv = Ed25519PrivateKey.generate()
    wrong_pub = wrong_priv.public_key()

    # Prepare one unsigned and one signed adapter from the same weights.
    rank, dtype_str, n_targets, domain = ADAPTER_MATRIX[4]
    w = _random_adapter(rank, dtype_str, n_targets)

    unsigned_manifest = _manifest(0, rank, n_targets, domain)  # signed=False
    unsigned_path = tmp / "kc1640_unsigned.pierre"
    save(str(unsigned_path), w, unsigned_manifest, signature=None)

    signed_path = tmp / "kc1640_signed.pierre"
    save_signed(str(signed_path), w, _manifest(0, rank, n_targets, domain), priv)

    cases: list[dict] = []

    # Case 1: unsigned without allow_unsigned raises UnsignedRejected.
    try:
        load_verified(str(unsigned_path))
        case1_ok = False
        case1_note = "no exception raised"
    except UnsignedRejected:
        case1_ok = True
        case1_note = "raised UnsignedRejected"
    except Exception as exc:
        case1_ok = False
        case1_note = f"wrong exception: {type(exc).__name__}"
    cases.append({
        "case": 1,
        "desc": "unsigned, no allow_unsigned -> UnsignedRejected",
        "ok": case1_ok, "note": case1_note,
    })

    # Case 2: unsigned with allow_unsigned=True loads.
    try:
        wts, mani, sig = load_verified(str(unsigned_path), allow_unsigned=True)
        case2_ok = (isinstance(wts, dict) and isinstance(mani, dict)
                    and mani.get("signed") is False)
        case2_note = "loaded"
    except Exception as exc:
        case2_ok = False
        case2_note = f"exception: {type(exc).__name__}: {exc}"
    cases.append({
        "case": 2,
        "desc": "unsigned, allow_unsigned=True -> loads",
        "ok": case2_ok, "note": case2_note,
    })

    # Case 3: signed with correct pubkey loads.
    try:
        wts, mani, sig = load_verified(str(signed_path), public_key=pub)
        case3_ok = (isinstance(wts, dict) and mani.get("signed") is True
                    and sig != bytes(SIG_SLOT_LEN))
        case3_note = "loaded"
    except Exception as exc:
        case3_ok = False
        case3_note = f"exception: {type(exc).__name__}: {exc}"
    cases.append({
        "case": 3,
        "desc": "signed, correct public_key -> loads",
        "ok": case3_ok, "note": case3_note,
    })

    # Case 4: signed without public_key raises SignatureRequired.
    try:
        load_verified(str(signed_path))
        case4_ok = False
        case4_note = "no exception raised"
    except SignatureRequired:
        case4_ok = True
        case4_note = "raised SignatureRequired"
    except Exception as exc:
        case4_ok = False
        case4_note = f"wrong exception: {type(exc).__name__}"
    cases.append({
        "case": 4,
        "desc": "signed, no public_key -> SignatureRequired",
        "ok": case4_ok, "note": case4_note,
    })

    # Case 5: signed with wrong public_key raises SignatureMismatch.
    try:
        load_verified(str(signed_path), public_key=wrong_pub)
        case5_ok = False
        case5_note = "no exception raised"
    except SignatureMismatch:
        case5_ok = True
        case5_note = "raised SignatureMismatch"
    except Exception as exc:
        case5_ok = False
        case5_note = f"wrong exception: {type(exc).__name__}"
    cases.append({
        "case": 5,
        "desc": "signed, wrong public_key -> SignatureMismatch",
        "ok": case5_ok, "note": case5_note,
    })

    passed = all(c["ok"] for c in cases)
    return {
        "kc_id": 1640,
        "cases": cases,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# K1641 — verify overhead < 100 ms median
# ---------------------------------------------------------------------------
def run_k1641(rng: random.Random, tmp: Path) -> dict:
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()

    # Representative: r=8, fp32, 2 targets ⇒ ~400 KB file.
    rank, dtype_str, n_targets, domain = 8, "float32", 2, "medical"
    w = _random_adapter(rank, dtype_str, n_targets)
    m_unsigned = _manifest(0, rank, n_targets, domain)
    m_signed = _manifest(0, rank, n_targets, domain)

    unsigned_path = tmp / "perf_unsigned.pierre"
    signed_path = tmp / "perf_signed.pierre"

    save(str(unsigned_path), w, m_unsigned, signature=None)
    save_signed(str(signed_path), w, m_signed, priv)

    size_unsigned = os.path.getsize(unsigned_path)
    size_signed = os.path.getsize(signed_path)

    # Warm the FS cache by reading each file once.
    _ = open(unsigned_path, "rb").read()
    _ = open(signed_path, "rb").read()

    n_per_mode = 50
    # Interleave to reduce drift; shuffle inside the interleave.
    schedule = (["unsigned"] * n_per_mode) + (["signed"] * n_per_mode)
    rng.shuffle(schedule)

    signed_ms: list[float] = []
    unsigned_ms: list[float] = []
    for mode in schedule:
        if mode == "signed":
            t0 = time.perf_counter()
            load_verified(str(signed_path), public_key=pub)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            signed_ms.append(dt_ms)
        else:
            t0 = time.perf_counter()
            load_verified(str(unsigned_path), allow_unsigned=True)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            unsigned_ms.append(dt_ms)

    signed_sorted = sorted(signed_ms)
    unsigned_sorted = sorted(unsigned_ms)
    signed_median = signed_sorted[n_per_mode // 2]
    unsigned_median = unsigned_sorted[n_per_mode // 2]
    overhead_ms = signed_median - unsigned_median

    passed = overhead_ms < 100.0
    return {
        "kc_id": 1641,
        "n_per_mode": n_per_mode,
        "size_unsigned_bytes": size_unsigned,
        "size_signed_bytes": size_signed,
        "signed_median_ms": signed_median,
        "unsigned_median_ms": unsigned_median,
        "overhead_ms": overhead_ms,
        "threshold_ms": 100.0,
        "passed": passed,
        "signed_p90_ms": signed_sorted[int(n_per_mode * 0.9)],
        "unsigned_p90_ms": unsigned_sorted[int(n_per_mode * 0.9)],
    }


def main() -> int:
    mx.random.seed(SEED)
    random.seed(SEED)
    rng = random.Random(SEED)

    results = {
        "experiment": "exp_prod_adapter_signing",
        "seed": SEED,
        "mlx_version": mx.__version__,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "is_smoke": False,
    }

    with tempfile.TemporaryDirectory() as td_str:
        tmp = Path(td_str)
        results["k1639"] = run_k1639(rng, tmp)
        results["k1640"] = run_k1640(tmp)
        results["k1641"] = run_k1641(rng, tmp)

    all_pass = bool(
        results["k1639"]["passed"]
        and results["k1640"]["passed"]
        and results["k1641"]["passed"]
    )
    results["all_pass"] = all_pass
    results["verdict"] = "SUPPORTED" if all_pass else "KILLED"

    out_path = HERE / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"K1639 rejected {results['k1639']['reject_count']}/100 -> "
          f"{'PASS' if results['k1639']['passed'] else 'FAIL'}")
    print(f"K1640 cases {sum(c['ok'] for c in results['k1640']['cases'])}/5 -> "
          f"{'PASS' if results['k1640']['passed'] else 'FAIL'}")
    print(f"K1641 overhead {results['k1641']['overhead_ms']:.3f} ms "
          f"(signed {results['k1641']['signed_median_ms']:.3f} vs "
          f"unsigned {results['k1641']['unsigned_median_ms']:.3f}) -> "
          f"{'PASS' if results['k1641']['passed'] else 'FAIL'}")
    print(f"verdict: {results['verdict']}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
