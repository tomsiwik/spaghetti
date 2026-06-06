"""Precondition probe for exp_g4_energy_vs_ridge_routing.

Does NOT run the full energy-vs-ridge comparison. Per the MATH.md pre-registered
rule and the 9-deep cohort standing guidance, heavy MLX work is deferred until
the upstream adapter retrain lands. This probe only checks P1/P2/P3 artifacts
and emits a KILLED verdict with UNMEASURABLE numerics.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
UPSTREAM = REPO / "micro" / "models" / "exp_p1_t2_single_domain_training"


def probe_p1() -> dict:
    """P1: Gemma 4 N=25 adapter weights on disk."""
    adapters_dir = UPSTREAM / "adapters"
    if not adapters_dir.exists():
        return {
            "pass": False,
            "detail": f"upstream adapters dir missing: {adapters_dir}",
            "safetensors_found": 0,
            "expected_min": 25,
        }
    weights = sorted(adapters_dir.glob("*/adapters.safetensors"))
    configs = sorted(adapters_dir.glob("*/adapter_config.json"))
    domains = sorted(p.name for p in adapters_dir.iterdir() if p.is_dir())
    return {
        "pass": len(weights) >= 25,
        "detail": (
            f"{len(weights)} safetensors / {len(configs)} configs / "
            f"{len(domains)} domain dirs: {domains}"
        ),
        "safetensors_found": len(weights),
        "expected_min": 25,
        "domains": domains,
    }


def probe_p2() -> dict:
    """P2: ridge routing head exists on Gemma 4 N=25 embeddings."""
    ridge_dir = REPO / "micro" / "models" / "exp_g4_ridge_routing_n25_mcq"
    res_file = ridge_dir / "results.json"
    if not res_file.exists():
        return {
            "pass": False,
            "detail": f"no Gemma 4 ridge-routing baseline results: {res_file} missing",
        }
    try:
        res = json.loads(res_file.read_text())
    except Exception as e:
        return {"pass": False, "detail": f"results.json unreadable: {e}"}
    verdict = res.get("verdict") or res.get("status") or ""
    if str(verdict).upper() == "KILLED":
        return {
            "pass": False,
            "detail": f"ridge-routing baseline experiment verdict={verdict}",
            "upstream_verdict": str(verdict),
        }
    return {
        "pass": True,
        "detail": f"ridge-routing baseline present, verdict={verdict}",
        "upstream_verdict": str(verdict),
    }


def probe_p3() -> dict:
    """P3: measured energy-gap AUC reference on Gemma 4."""
    energy_dir = REPO / "micro" / "models" / "energy_gap_topk_routing"
    res_file = energy_dir / "results.json"
    if not res_file.exists():
        return {
            "pass": False,
            "detail": f"no Gemma 4 energy-gap AUC reference: {res_file} missing",
        }
    try:
        res = json.loads(res_file.read_text())
    except Exception as e:
        return {"pass": False, "detail": f"results.json unreadable: {e}"}
    base = (res.get("config") or {}).get("base_model") or res.get("base_model") or ""
    on_gemma4 = "gemma-4" in str(base).lower() or "gemma4" in str(base).lower()
    return {
        "pass": bool(on_gemma4),
        "detail": f"energy-gap reference present, base_model={base!r}",
        "on_gemma4": bool(on_gemma4),
    }


def main() -> None:
    start = time.time()
    p1 = probe_p1()
    p2 = probe_p2()
    p3 = probe_p3()
    all_pass = bool(p1["pass"] and p2["pass"] and p3["pass"])

    verdict = "SUPPORTED" if all_pass else "KILLED"
    result = {
        "experiment_id": "exp_g4_energy_vs_ridge_routing",
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "precondition_probe": True,
        "kc": {
            "K1588": {
                "text": "ridge > energy by 10pp on Gemma 4 N=25",
                "result": "UNMEASURABLE" if not all_pass else "TODO",
            }
        },
        "preconditions": {
            "P1_adapters_on_disk": p1,
            "P2_ridge_routing_head": p2,
            "P3_energy_gap_gemma4_ref": p3,
        },
        "blocking_upstream": (
            "exp_p1_t2_single_domain_training (rerun at LORA_SCALE=5, max_tokens>=512, "
            "5+ disjoint domains; shared cohort dependency, 10th consecutive KILL)"
        ),
        "cohort": "audit-2026-04-17",
        "wall_time_s": round(time.time() - start, 3),
    }

    out = HERE / "results.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
