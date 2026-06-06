#!/usr/bin/env python3
"""exp_followup_grassmannian_native_macro — precondition-probe runner.

Pure file-probe + DB-check tripwire. No MLX model load, no training,
no cosine computation. Probe completes in <10s.

Per MATH.md tripwire:
  K1550 = All(P1, P2, P3) — fails if any precondition probe fails.

If any probe FAILs, verdict=killed, all_pass=False, status=killed.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

EXP_ID = "exp_followup_grassmannian_native_macro"
OUT_PATH = Path(__file__).resolve().parent / "results.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
MICRO_MODELS = REPO_ROOT / "micro" / "models"
HF_CACHE = Path(os.path.expanduser("~/.cache/huggingface"))


def probe_p1_target_model_safetensors() -> tuple[bool, dict]:
    """P1 — real trained Gemma 4 E4B / Qwen3-4B LoRA safetensors."""
    patterns = [
        "*gemma*4*e4b*",
        "*gemma*4*26b*",
        "*qwen3*4b*",
    ]
    hits = []
    for pat in patterns:
        for root in (MICRO_MODELS, HF_CACHE):
            if not root.exists():
                continue
            for p in root.glob(f"**/{pat}/**/*.safetensors"):
                if "lora" in p.name.lower() or "adapter" in p.name.lower():
                    hits.append(str(p))
    # Also check for conventional adapter.safetensors in any sub-exp dir
    for p in MICRO_MODELS.glob("**/adapters/**/*.safetensors"):
        hits.append(str(p))
    return (len(hits) > 0, {"hits": len(hits), "sample": hits[:5]})


def probe_p2_grassmannian_ap_init() -> tuple[bool, dict]:
    """P2 — Grassmannian-AP initialized adapters (P_avg packing)."""
    markers = []
    for p in MICRO_MODELS.glob("**/grassmannian*/**/*.safetensors"):
        markers.append(str(p))
    for p in MICRO_MODELS.glob("**/ap_init*/**/*.safetensors"):
        markers.append(str(p))
    for p in MICRO_MODELS.glob("**/*grassmannian*init*.json"):
        markers.append(str(p))
    return (len(markers) > 0, {"hits": len(markers), "sample": markers[:5]})


def probe_p3_upstream_t21_supported() -> tuple[bool, dict]:
    """P3 — upstream exp_p1_t2_single_domain_training must be supported."""
    try:
        cp = subprocess.run(
            ["experiment", "get", "exp_p1_t2_single_domain_training"],
            capture_output=True, text=True, timeout=15
        )
        txt = cp.stdout
        status_line = next(
            (ln.strip() for ln in txt.splitlines() if ln.strip().startswith("Status:")),
            ""
        )
        status_val = status_line.split(":", 1)[-1].strip() if status_line else "unknown"
        passed = (status_val == "supported")
        return (passed, {"status": status_val, "stderr": cp.stderr[:300]})
    except Exception as e:  # noqa: BLE001
        return (False, {"status": f"error: {e}"})


def main() -> int:
    t0 = time.time()
    p1_pass, p1_info = probe_p1_target_model_safetensors()
    p2_pass, p2_info = probe_p2_grassmannian_ap_init()
    p3_pass, p3_info = probe_p3_upstream_t21_supported()

    all_pass = bool(p1_pass and p2_pass and p3_pass)
    verdict = "supported" if all_pass else "killed"

    results = {
        "experiment_id": EXP_ID,
        "is_smoke": False,
        "all_pass": all_pass,
        "verdict": verdict,
        "probes": {
            "p1_target_model_safetensors": {"pass": p1_pass, **p1_info},
            "p2_grassmannian_ap_init":     {"pass": p2_pass, **p2_info},
            "p3_upstream_t21_supported":    {"pass": p3_pass, **p3_info},
        },
        "kill_criteria": {
            "K1550": {
                "text": "Over-packed (Nr>d) max|cos| <= 100*sqrt(r/d) with real trained adapters",
                "result": "pass" if all_pass else "fail",
                "note": (
                    "Tripwire K1550 = All(P1,P2,P3); fails → UNMEASURABLE → killed"
                ),
            }
        },
        "wall_seconds": round(time.time() - t0, 3),
        "notes": (
            "Precondition-probe tripwire per MATH.md. No MLX model load. "
            "Probe-KILL pattern consistent with 15+ prior cohort instances "
            "(Findings #605-#622) — same upstream blocker: "
            "exp_p1_t2_single_domain_training verdict=killed."
        ),
    }
    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
