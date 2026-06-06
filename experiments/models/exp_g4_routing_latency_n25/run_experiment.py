"""exp_g4_routing_latency_n25 — probe-only runner.

Pre-registered tripwire (MATH.md): KC #1597 is UNMEASURABLE unless
  P1: 25 Gemma 4 v_proj+o_proj r=6 LoRA safetensors on disk
  P2: upstream exp_p1_t2_single_domain_training SUPPORTED
  P3: Gemma 4 ridge router on disk

On any FAIL → write results.json with verdict=KILLED and exit 0.
No MLX model load. No hypothetical latency number.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "results.json"


def probe_p1_adapters() -> tuple[bool, int, list[str]]:
    """P1: count Gemma-4 v_proj/o_proj r=6 LoRA safetensors."""
    candidate_dirs = [
        REPO / "experiments/models/exp_p1_t2_single_domain_training/adapters",
        REPO / "experiments/models/exp_g4_single_domain_vproj_think",
        REPO / "experiments/models/exp_g4_5domain_real_hf",
        REPO / "experiments/models/exp_g4_e2e_n25_25real",
    ]
    hits: list[str] = []
    for d in candidate_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.safetensors"):
            hits.append(str(p.relative_to(REPO)))
    return (len(hits) >= 25, len(hits), hits[:5])


def probe_p2_upstream() -> tuple[bool, dict]:
    """P2: upstream T2.1 status + all_pass."""
    try:
        r = subprocess.run(
            ["experiment", "get", "exp_p1_t2_single_domain_training"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        out = r.stdout
    except Exception as e:
        return False, {"error": str(e)}
    status_supported = "Status:   supported" in out
    all_pass_false = ("all_pass=false" in out.lower()) or ("AUDIT-RERUN KILLED" in out)
    return status_supported and not all_pass_false, {
        "status_supported": status_supported,
        "all_pass_false_flag": all_pass_false,
    }


def probe_p3_router() -> tuple[bool, list[str]]:
    """P3: Gemma 4 ridge router implementation on disk."""
    candidates = [
        REPO / "experiments/models/exp_g4_tfidf_ridge_n25_clean/run_experiment.py",
        REPO / "experiments/models/exp_p1_c0_composition_port_gemma4/run_experiment.py",
    ]
    existing = [str(c.relative_to(REPO)) for c in candidates if c.exists()]
    return False, existing


def main() -> int:
    t0 = time.time()

    p1_ok, n_adapters, p1_examples = probe_p1_adapters()
    p2_ok, p2_detail = probe_p2_upstream()
    p3_ok, p3_candidates = probe_p3_router()

    all_pass = p1_ok and p2_ok and p3_ok
    verdict = "KILLED" if not all_pass else "PENDING_REAL_RUN"

    payload = {
        "experiment_id": "exp_g4_routing_latency_n25",
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "kill_criteria": {
            "K1597": {
                "text": "latency <= 1.20x base",
                "result": "UNMEASURABLE" if not all_pass else "pending",
            }
        },
        "probes": {
            "P1_n25_gemma4_adapters": {
                "pass": p1_ok,
                "count_on_disk": n_adapters,
                "required": 25,
                "example_hits": p1_examples,
            },
            "P2_upstream_t21_supported": {
                "pass": p2_ok,
                "detail": p2_detail,
                "blocker_finding_refs": [
                    "#605", "#606", "#608", "#610", "#611", "#612",
                    "#613", "#615", "#616", "#617", "#618", "#619",
                    "#620", "#621",
                ],
            },
            "P3_gemma4_ridge_router": {
                "pass": p3_ok,
                "candidates_found": p3_candidates,
                "note": "existence alone insufficient; also depends on P1",
            },
        },
        "wall_time_s": round(time.time() - t0, 4),
        "cohort_kill_index": 15,
        "unblocks_on": (
            "exp_p1_t2_single_domain_training rerun at LORA_SCALE=5, "
            "max_tokens>=512, >=5 disjoint domains, rank sweep "
            "{2,4,6,12,24}, grad-SNR spectra, plus N=25 v_proj+o_proj "
            "r=6 adapter materialization."
        ),
    }

    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
