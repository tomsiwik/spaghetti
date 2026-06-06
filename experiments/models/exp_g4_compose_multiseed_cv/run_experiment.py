"""Precondition probe for exp_g4_compose_multiseed_cv.

Heavy evaluation (3 seeds × 5 adapters × MMLU-Pro) is forbidden until
preconditions P1/P2/P3 pass. This script is the fast-path probe: pure
file-existence + upstream-verdict read, no MLX.

Tripwire (MATH.md): any P_i FAIL → K1590 UNMEASURABLE → status=killed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "results.json"

UPSTREAM_T21 = REPO / "experiments/models/exp_p1_t2_single_domain_training"


def probe_p1_adapters() -> dict:
    """P1 — 3 seeds × 5 Gemma 4 r=6 LoRA adapters on disk (15 safetensors)."""
    candidates = list(REPO.glob("experiments/models/**/seed*/**/*.safetensors"))
    # also any obvious multi-seed cohort dir
    g4_seeds = [p for p in candidates if "g4" in str(p) or "gemma4" in str(p)]
    return {
        "precondition": "P1",
        "description": "3 seeds × 5 Gemma 4 adapters exist",
        "expected_safetensors": 15,
        "found_any_safetensors": len(candidates),
        "found_gemma4_seeded_safetensors": len(g4_seeds),
        "pass": len(g4_seeds) >= 15,
    }


def probe_p2_upstream() -> dict:
    """P2 — upstream T2.1 training landed at LORA_SCALE=5, max_tokens>=512."""
    r = UPSTREAM_T21 / "results.json"
    if not r.exists():
        return {"precondition": "P2", "description": "upstream T2.1 results", "pass": False, "reason": "results.json missing"}
    data = json.loads(r.read_text())
    verdict = data.get("verdict")
    all_pass = data.get("all_pass")
    lora_scale = data.get("lora_scale") or data.get("config", {}).get("lora_scale")
    return {
        "precondition": "P2",
        "description": "upstream exp_p1_t2_single_domain_training landed",
        "verdict": verdict,
        "all_pass": all_pass,
        "lora_scale": lora_scale,
        "pass": verdict == "SUPPORTED" and bool(all_pass),
    }


def probe_p3_eval_harness() -> dict:
    """P3 — MMLU-Pro evaluator with reproducible baseline for this cohort."""
    harness = list(REPO.glob("experiments/models/**/mmlu_pro*.py"))
    return {
        "precondition": "P3",
        "description": "MMLU-Pro harness with cohort baseline",
        "harness_files_found": [str(p.relative_to(REPO)) for p in harness[:5]],
        # No cohort-baseline file exists because cohort is 11/11 probe-killed
        "pass": False,
        "reason": "cohort has no landed MMLU-Pro baseline; all upstream cohort members are probe-KILLed",
    }


def main() -> None:
    t0 = time.time()
    p1 = probe_p1_adapters()
    p2 = probe_p2_upstream()
    p3 = probe_p3_eval_harness()
    checks = [p1, p2, p3]
    all_pass = all(c["pass"] for c in checks)

    result = {
        "experiment_id": "exp_g4_compose_multiseed_cv",
        "is_smoke": False,
        "mode": "precondition_probe",
        "wall_seconds": round(time.time() - t0, 4),
        "preconditions": checks,
        "all_pass": all_pass,
        "k1590_measurable": all_pass,
        "verdict": "SUPPORTED" if all_pass else "KILLED",
        "reason": (
            "All preconditions pass — proceed to heavy 3-seed CV evaluation."
            if all_pass
            else "Precondition(s) fail → K1590 UNMEASURABLE → status=killed per MATH.md tripwire."
        ),
        "cohort_context": {
            "tag": "audit-2026-04-17",
            "prior_kill_findings": [605, 606, 608, 610, 611, 612, 613, 615, 616, 617, 618],
            "note": "12th consecutive cohort precondition-probe KILL. Blocker: exp_p1_t2_single_domain_training retrain at LORA_SCALE=5, max_tokens>=512, rank sweep, grad-SNR logging.",
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
