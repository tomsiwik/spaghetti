"""
exp_model_knowledge_gap_26b_base — F#478 retest at Gemma 4 26B-A4B.

Status: BLOCKED on resource (base model not cached).

This script is a structural scaffold that would execute the full experiment if
the base model were present. It does NOT run in this iteration — see
MATH.md §7 (Blockers) for rationale (F#478 monotonic prior + 26B model not
cached + single-iteration compute budget).

When run: the script must
  1. Verify the base model is cached (raise RuntimeError if not).
  2. Load `mlx-community/gemma-4-26b-a4b-it-4bit` via mlx_lm.load.
  3. Train rank-6 LoRA on v_proj+o_proj for {code, math, medical}
     (500 steps, effective batch 8, enable_thinking=True).
  4. Eval MMLU-Pro per domain (structural, K1702).
  5. Run N=30 held-out behavioral prompts with adversarial-judge rating
     (target, K1703 / K1816).
  6. Emit results.json with {K1702, K1703, K1816} pass/fail and verdict.

The current invocation writes a results.json with verdict=BLOCKED and does not
attempt to touch the model path; this avoids silent proxy-model substitution
(researcher hat antipattern 'm' / reviewer antipattern (t)).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
BASE_MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
# mlx-lm / huggingface cache convention: models--org--repo
BASE_MODEL_CACHE_DIR = HF_CACHE / BASE_MODEL_ID.replace("/", "--").replace(
    "mlx-community--", "models--mlx-community--"
)


def base_model_cached() -> bool:
    """Check whether the Gemma 4 26B-A4B 4-bit model is present in the HF cache."""
    if not HF_CACHE.exists():
        return False
    # The real cache directory uses the `models--mlx-community--<name>` prefix.
    candidates = [
        HF_CACHE / f"models--mlx-community--gemma-4-26b-a4b-it-4bit",
    ]
    return any(c.exists() for c in candidates)


def run_blocked() -> dict:
    """Emit a BLOCKED result without touching the model or the GPU."""
    return {
        "experiment_id": "exp_model_knowledge_gap_26b_base",
        "status": "BLOCKED",
        "verdict": "PROVISIONAL",
        "is_smoke": False,
        "all_pass": False,
        "blocker": {
            "reason": "BASE_MODEL_NOT_CACHED",
            "model_id": BASE_MODEL_ID,
            "expected_cache_path": str(BASE_MODEL_CACHE_DIR),
            "approx_download_size_gb": 14,
            "approx_training_time_hours": 2.5,
            "proof_first_prior": (
                "F#478 monotonic extension via scaling-law monotonicity "
                "(Kaplan 2020, Hoffmann 2022) predicts KILL on K1702; "
                "running before deriving the MoE-niche measurement wastes "
                "compute without resolving the claim."
            ),
        },
        "kill_criteria": {
            "K1702_structural_proxy": "untested",
            "K1703_target_behavioral": "untested",
            "K1816_target_win_rate": "untested",
        },
        "paired_proxy_target": {
            "proxy": "K1702",
            "target": "K1703 (and K1816)",
            "both_required_fail_to_kill": True,
            "both_required_pass_to_support": True,
        },
        "notes": (
            "This scaffold refuses to silently proxy to Gemma 4 4B "
            "(researcher hat antipattern 'm'); no run performed."
        ),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    cached = base_model_cached()
    if not cached:
        result = run_blocked()
        out = EXPERIMENT_DIR / "results.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"[BLOCKED] Base model '{BASE_MODEL_ID}' not in HF cache "
            f"({HF_CACHE}); wrote provisional results to {out}",
            file=sys.stderr,
        )
        return 0

    # Live path would dispatch to train + eval here. Guarded by the cache
    # check so we never silently run with the wrong model.
    raise NotImplementedError(
        "Live path requires 26B-A4B cached; implement per MATH.md §6 when "
        "compute budget is explicitly authorized."
    )


if __name__ == "__main__":
    raise SystemExit(main())
