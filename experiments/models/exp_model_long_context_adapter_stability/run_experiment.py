"""
exp_model_long_context_adapter_stability — F#263 frontier-extension to long context.

Status: BLOCKED on compute budget (4-8 h NIAH + RULER sweep exceeds 30 min
single-iteration researcher budget per guardrail 1009).

This script is a structural scaffold that would execute the full experiment if
a dedicated long-running compute window were authorized. It does NOT run in the
drain iteration — see MATH.md §7 (Blockers) for rationale (compute budget +
RULER baseline measurement also expensive).

When run (in dedicated >=4h session, after `experiment update --priority 2`):
the script must
  1. Verify required adapters are on disk:
     - code/math/medical from exp_p1_t2_single_domain_training (F#627)
     - legal/finance from exp_p1_t2_multi_domain_5
  2. Load `mlx-community/gemma-4-e4b-it-4bit` via mlx_lm.load (cached).
  3. Measure base NIAH at {8k, 32k, 128k} with needle at {10,25,50,75,90}%
     depth — this alone is ~25 samples * 3 lengths * 10 min/128k-sample.
  4. Compose N=5 adapters via correct math `Σ_i B_i @ A_i` (NOT (ΣB)(ΣA);
     mem-antipattern-001), LORA_SCALE <= 8 (F#328/#330).
  5. Re-measure NIAH under composition; compute K1706 (within 5pp of base
     across ALL three lengths).
  6. Run RULER 13-subtask suite (Hsieh 2024, arxiv:2404.06654) base + composed,
     all three lengths; compute K1707 (within 3pp on every subtask).
  7. Emit results.json with {K1706, K1707} pass/fail and verdict per F#666.

The current invocation writes a results.json with verdict=BLOCKED on compute
budget and does not load the model; this avoids silent proxy substitution
(researcher hat antipattern 'm' / reviewer antipattern (t)) such as running
only 8k and claiming 128k by extrapolation.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
BASE_MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
BASE_MODEL_CACHE_DIR = (
    HF_CACHE / f"models--mlx-community--gemma-4-e4b-it-4bit"
)

# Estimated wall-clock budget per MATH.md §7 (Blockers B1):
#   128k prefill on M5 Pro 48GB ~ 10 min/sample
#   NIAH grid: 5 depths * 3 lengths * 5 samples ~ 1-2 h
#   RULER: 13 subtasks * 3 lengths ~ 3-5 h
#   Total: 4-8 h (well over 30 min single-iter budget per guardrail 1009)
COMPUTE_BUDGET_MINUTES = 240  # lower bound, conservative
SINGLE_ITER_BUDGET_MINUTES = 30  # researcher hat guardrail 1009


def base_model_cached() -> bool:
    return BASE_MODEL_CACHE_DIR.exists()


def required_adapters_present() -> dict[str, bool]:
    """Check the 5 PoLAR adapters required for N=5 composition exist on disk."""
    repo_root = EXPERIMENT_DIR.parent.parent.parent
    expected = {
        "code": repo_root / "experiments/models/exp_p1_t2_single_domain_training/adapters/code",
        "math": repo_root / "experiments/models/exp_p1_t2_single_domain_training/adapters/math",
        "medical": repo_root / "experiments/models/exp_p1_t2_single_domain_training/adapters/medical",
        "legal": repo_root / "experiments/models/exp_p1_t2_multi_domain_5/adapters/legal",
        "finance": repo_root / "experiments/models/exp_p1_t2_multi_domain_5/adapters/finance",
    }
    return {k: v.exists() for k, v in expected.items()}


def run_blocked(reason: str) -> dict:
    """Emit a BLOCKED result without touching the model or the GPU."""
    return {
        "experiment_id": "exp_model_long_context_adapter_stability",
        "status": "BLOCKED",
        "verdict": "PROVISIONAL",
        "is_smoke": False,
        "all_pass": False,
        "blocker": {
            "reason": reason,
            "model_id": BASE_MODEL_ID,
            "expected_cache_path": str(BASE_MODEL_CACHE_DIR),
            "approx_compute_budget_minutes": COMPUTE_BUDGET_MINUTES,
            "single_iter_budget_minutes": SINGLE_ITER_BUDGET_MINUTES,
            "ratio": COMPUTE_BUDGET_MINUTES / SINGLE_ITER_BUDGET_MINUTES,
            "proof_first_prior": (
                "F#263 short-context degradation extends partially: §3.2 "
                "V/O structural protection predicts 8k PASS, mild 32k drift; "
                "§3.3 range extrapolation makes 128k empirically open. "
                "Running only the proof-predicted lengths (8k/32k) and "
                "claiming 128k by extrapolation would violate "
                "antipattern 'm' (proxy-model-substitution at the "
                "context-length axis). Full sweep is required."
            ),
        },
        "kill_criteria": {
            "K1706_structural_proxy_NIAH_within_5pp": "untested",
            "K1707_target_behavioral_RULER_within_3pp": "untested",
        },
        "paired_proxy_target": {
            "proxy": "K1706 (NIAH retrieval)",
            "target": "K1707 (RULER 13-subtask)",
            "both_required_fail_to_kill": True,
            "both_required_pass_to_support": True,
        },
        "reclaim_path": [
            "schedule dedicated >=4h compute session",
            "experiment update --priority 2",
            "invoke /mlx-dev + /fast-mlx skills",
            "implement NIAH harness first (1-2 h)",
            "run RULER baseline + composed (3-5 h)",
            "complete with --status supported|killed per F#666 verdict rule",
        ],
        "notes": (
            "Design-only iteration. Refuses to silently proxy by measuring "
            "only short context and extrapolating to 128k (researcher hat "
            "antipattern 'm'); no run performed. Filing follows F#768 "
            "BLOCKED-on-resource pattern (compute-budget variant)."
        ),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    # The base model IS cached (Gemma 4 E4B is the dev base per PLAN.md Part 2)
    # but the 4-8 h sweep exceeds single-iteration budget. Refuse on budget,
    # not on cache.
    cached = base_model_cached()
    adapters = required_adapters_present()

    if not cached:
        reason = "BASE_MODEL_NOT_CACHED"
    elif not all(adapters.values()):
        missing = [k for k, v in adapters.items() if not v]
        reason = f"REQUIRED_ADAPTERS_MISSING: {missing}"
    else:
        reason = "COMPUTE_BUDGET_EXCEEDS_DRAIN_ITERATION"

    result = run_blocked(reason)
    out = EXPERIMENT_DIR / "results.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"[BLOCKED:{reason}] wrote provisional results to {out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
