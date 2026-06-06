"""Reference encoder/decoder for the `.pierre` adapter file format v1.

Layout (all multi-byte ints little-endian):
    0:8    magic            b"\\x89PIERRE\\n"
    8:12   version          u32 = 1
    12:76  signature_slot   64 bytes (zero if unsigned)
    76:84  manifest_length  u64
    84:84+M manifest        UTF-8 JSON
    ...    safetensors_blob output of mx.save_safetensors(weights)

See MATH.md for the proof that save/load is lossless (Theorem 1).
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from typing import Any

import mlx.core as mx

MAGIC = b"\x89PIERRE\n"
VERSION = 1
SIG_SLOT_LEN = 64
HEADER_LEN = 8 + 4 + SIG_SLOT_LEN + 8  # = 84


REQUIRED_MANIFEST_KEYS = {
    "spec_version",
    "base_model_id",
    "base_model_hash",
    "created_at",
    "signed",
    "training_metadata",
}


class PierreFormatError(ValueError):
    pass


def _safetensors_bytes(weights: dict[str, mx.array]) -> bytes:
    """Serialize an {name: mx.array} dict to safetensors bytes via a tmpfile.

    mx.save_safetensors writes to a path; we round-trip via tmp to get bytes.
    """
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        mx.save_safetensors(path, weights)
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _safetensors_load(blob: bytes) -> dict[str, mx.array]:
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(blob)
        loaded = mx.load(path)
        mx.eval(*loaded.values())
        return dict(loaded)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def save(
    path: str,
    weights: dict[str, mx.array],
    manifest: dict[str, Any],
    *,
    signature: bytes | None = None,
) -> None:
    """Write a .pierre v1 file.

    signature: 64-byte Ed25519 sig, or None for an unsigned file (slot
    zero-filled).
    """
    if signature is None:
        sig = bytes(SIG_SLOT_LEN)
    else:
        if len(signature) != SIG_SLOT_LEN:
            raise PierreFormatError(
                f"signature must be {SIG_SLOT_LEN} bytes, got {len(signature)}"
            )
        sig = signature

    missing = REQUIRED_MANIFEST_KEYS - set(manifest.keys())
    if missing:
        raise PierreFormatError(f"manifest missing required keys: {sorted(missing)}")

    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    st_blob = _safetensors_bytes(weights)

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(sig)
        f.write(struct.pack("<Q", len(manifest_bytes)))
        f.write(manifest_bytes)
        f.write(st_blob)


def load(path: str) -> tuple[dict[str, mx.array], dict[str, Any], bytes]:
    """Read a .pierre v1 file. Returns (weights, manifest, signature_slot)."""
    with open(path, "rb") as f:
        header = f.read(HEADER_LEN)
        if len(header) < HEADER_LEN:
            raise PierreFormatError("file truncated before header complete")
        if header[0:8] != MAGIC:
            raise PierreFormatError(
                f"bad magic: {header[0:8]!r} (expected {MAGIC!r})"
            )
        (version,) = struct.unpack("<I", header[8:12])
        if version != VERSION:
            raise PierreFormatError(f"unsupported version: {version}")
        signature = header[12:76]
        (manifest_length,) = struct.unpack("<Q", header[76:84])

        manifest_bytes = f.read(manifest_length)
        if len(manifest_bytes) != manifest_length:
            raise PierreFormatError("truncated manifest")
        manifest = json.loads(manifest_bytes.decode("utf-8"))

        missing = REQUIRED_MANIFEST_KEYS - set(manifest.keys())
        if missing:
            raise PierreFormatError(
                f"manifest missing required keys: {sorted(missing)}"
            )

        st_blob = f.read()

    weights = _safetensors_load(st_blob)
    return weights, manifest, signature


def canonical_body(path: str) -> bytes:
    """Return the file bytes with the signature slot zeroed.

    This is the byte sequence that Ed25519 signing/verification operates
    on (exp_prod_adapter_signing will consume this).
    """
    with open(path, "rb") as f:
        buf = bytearray(f.read())
    for i in range(12, 12 + SIG_SLOT_LEN):
        buf[i] = 0
    return bytes(buf)
