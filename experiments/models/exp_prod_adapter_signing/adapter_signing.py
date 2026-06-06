"""Ed25519 signing + verification layer for the `.pierre` v1 format.

Builds on `adapter_format_v1` from `exp_prod_adapter_format_spec_v1`:
signature slot at bytes [12:76], helper `canonical_body(path)` that
returns file bytes with the slot zeroed. See MATH.md (Theorem 1) for
the EUF-CMA reduction.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

# Reference the v1 encoder/decoder living in the sibling experiment.
_HERE = Path(__file__).resolve().parent
_V1_DIR = _HERE.parent / "exp_prod_adapter_format_spec_v1"
sys.path.insert(0, str(_V1_DIR))
from adapter_format_v1 import (  # noqa: E402
    HEADER_LEN,
    MAGIC,
    REQUIRED_MANIFEST_KEYS,
    SIG_SLOT_LEN,
    VERSION,
    PierreFormatError,
    canonical_body,
    load,
    save,
)


class SignatureRequired(PierreFormatError):
    """Manifest declares signed=True but no public_key was supplied."""


class SignatureMismatch(PierreFormatError):
    """Signature present but does not verify under the supplied pubkey."""


class UnsignedRejected(PierreFormatError):
    """Manifest declares signed=False and allow_unsigned is False."""


def save_signed(
    path: str,
    weights: dict[str, mx.array],
    manifest: dict[str, Any],
    private_key: Ed25519PrivateKey,
) -> None:
    """Write a signed `.pierre` v1 file.

    Protocol (see MATH.md §SIGN):
      1. Flip manifest.signed=True and write the signer pubkey hex
         *before* signing — the canonical body must reflect the final
         manifest.
      2. Write a v1 file with zero-filled signature slot using
         adapter_format_v1.save().
      3. Read the canonical body (slot zeroed, by construction).
      4. Sign canonical body with Ed25519.
      5. Patch the signature bytes into the file's slot.
    """
    pub_bytes = private_key.public_key().public_bytes_raw()
    pub_hex = pub_bytes.hex()

    manifest_signed = dict(manifest)
    manifest_signed["signed"] = True
    manifest_signed["signer_pubkey"] = pub_hex

    # Step 2: v1 file with zero sig slot (signature=None ⇒ all-zero).
    save(path, weights, manifest_signed, signature=None)

    # Step 3: canonical body (slot already zeroed from step 2).
    body = canonical_body(path)

    # Step 4: Ed25519 sign (64 bytes).
    sig = private_key.sign(body)
    assert len(sig) == SIG_SLOT_LEN, f"Ed25519 sig must be 64B, got {len(sig)}"

    # Step 5: patch sig into file.
    with open(path, "r+b") as f:
        f.seek(12)
        f.write(sig)


def verify_file(path: str, expected_public_key: Ed25519PublicKey) -> bool:
    """Return True iff file's signature verifies under `expected_public_key`.

    Fails closed: any parse error, unsigned manifest, pubkey-binding
    mismatch, or InvalidSignature returns False.
    """
    try:
        with open(path, "rb") as f:
            header = f.read(HEADER_LEN)
        if len(header) < HEADER_LEN or header[0:8] != MAGIC:
            return False
        (ver,) = struct.unpack("<I", header[8:12])
        if ver != VERSION:
            return False
        sig = header[12:76]
        if sig == bytes(SIG_SLOT_LEN):
            return False  # all-zero slot = unsigned

        (mlen,) = struct.unpack("<Q", header[76:84])
        with open(path, "rb") as f:
            f.seek(HEADER_LEN)
            manifest = json.loads(f.read(mlen).decode("utf-8"))

        if not manifest.get("signed", False):
            return False

        # Strict pubkey-binding: the manifest must name the same pubkey
        # the caller is presenting. Prevents a signed file being
        # "laundered" by swapping the signer_pubkey field.
        expected_pub_hex = expected_public_key.public_bytes_raw().hex()
        if manifest.get("signer_pubkey", "") != expected_pub_hex:
            return False

        body = canonical_body(path)
        expected_public_key.verify(sig, body)
        return True
    except (InvalidSignature, ValueError, OSError, KeyError):
        return False


def load_verified(
    path: str,
    public_key: Ed25519PublicKey | None = None,
    *,
    allow_unsigned: bool = False,
) -> tuple[dict[str, mx.array], dict[str, Any], bytes]:
    """Signed-by-default loader. See MATH.md §LOADER POLICY.

    Policy:
      - If manifest.signed == True:
          - public_key is None                 -> SignatureRequired
          - public_key set but verify fails    -> SignatureMismatch
          - verify succeeds                    -> return load(path)
      - If manifest.signed == False:
          - allow_unsigned == False            -> UnsignedRejected
          - allow_unsigned == True             -> return load(path)
    """
    # Read manifest cheaply without decoding safetensors.
    with open(path, "rb") as f:
        header = f.read(HEADER_LEN)
        if len(header) < HEADER_LEN:
            raise PierreFormatError("file truncated before header complete")
        if header[0:8] != MAGIC:
            raise PierreFormatError(f"bad magic: {header[0:8]!r}")
        (mlen,) = struct.unpack("<Q", header[76:84])
        manifest_bytes = f.read(mlen)

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except ValueError as exc:
        raise PierreFormatError(f"manifest json decode failed: {exc}") from exc

    missing = REQUIRED_MANIFEST_KEYS - set(manifest.keys())
    if missing:
        raise PierreFormatError(f"manifest missing required keys: {sorted(missing)}")

    signed = bool(manifest.get("signed", False))

    if signed:
        if public_key is None:
            raise SignatureRequired(
                "adapter manifest declares signed=True; public_key required"
            )
        if not verify_file(path, public_key):
            raise SignatureMismatch(
                "adapter signature invalid under supplied public_key"
            )
    else:
        if not allow_unsigned:
            raise UnsignedRejected(
                "adapter is unsigned; set allow_unsigned=True to load anyway"
            )

    return load(path)
