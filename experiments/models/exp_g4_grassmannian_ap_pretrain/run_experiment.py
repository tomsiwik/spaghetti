"""exp_g4_grassmannian_ap_pretrain — precondition probe.

Per MATH.md pre-registered tripwire, verify P1/P2/P3 before any heavy MLX
run. If any fail, write results.json with status=killed and exit.

This matches the audit-2026-04-17 cohort standing rule established across
Findings #605/#606/#608/#610/#611/#612/#613/#615/#616/#617.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "results.json"


def probe_p1_n25_adapters() -> tuple[bool, str]:
    """P1: Gemma 4 E4B N=25 disjoint-domain q_proj+v_proj 42-layer adapters."""
    candidates = [
        ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters",
        ROOT / "experiments/models/exp_p1_t3_n25_composition",
        ROOT / "experiments/models/exp_p0_n25_vproj_composition",
    ]
    found = []
    for c in candidates:
        if c.exists():
            found.extend(list(c.rglob("*.safetensors")))
    if len(found) >= 25:
        return True, f"P1 PASS: {len(found)} safetensors across N=25 adapter dirs"
    return False, (
        f"P1 FAIL: expected >=25 q_proj+v_proj safetensors, found {len(found)}. "
        f"Searched: {[str(c.relative_to(ROOT)) for c in candidates]}"
    )


def probe_p2_grassmannian_skeleton() -> tuple[bool, str]:
    """P2: Gemma 4 port of Finding #132 AP skeleton (runnable code)."""
    port_dir = ROOT / "experiments/models/exp_p1_t0_grassmannian_gemma4"
    if not port_dir.exists():
        return False, f"P2 FAIL: {port_dir.relative_to(ROOT)} does not exist"
    safetensors = list(port_dir.rglob("*.safetensors"))
    runnable = (port_dir / "run_experiment.py").exists()
    if safetensors and runnable:
        return True, f"P2 PASS: {len(safetensors)} safetensors + runnable"
    return False, (
        f"P2 FAIL: {port_dir.relative_to(ROOT)} has runnable={runnable} "
        f"safetensors={len(safetensors)} — port not materialized"
    )


def probe_p3_t2_upstream() -> tuple[bool, str]:
    """P3: upstream T2.1 rerun at LORA_SCALE=5 with all_pass=true."""
    t2 = ROOT / "experiments/models/exp_p1_t2_single_domain_training/results.json"
    if not t2.exists():
        return False, "P3 FAIL: upstream T2.1 results.json missing"
    try:
        data = json.loads(t2.read_text())
    except Exception as exc:
        return False, f"P3 FAIL: could not parse T2.1 results.json ({exc})"
    all_pass = data.get("all_pass", False)
    verdict = data.get("verdict", "unknown")
    lora_scale = data.get("lora_scale") or data.get("config", {}).get("lora_scale")
    if all_pass and verdict.lower() == "supported":
        return True, f"P3 PASS: T2.1 all_pass=true verdict={verdict} lora_scale={lora_scale}"
    return False, (
        f"P3 FAIL: T2.1 all_pass={all_pass} verdict={verdict} lora_scale={lora_scale} — "
        "upstream rebuild (LORA_SCALE=5, max_tokens>=512, rank sweep, grad-SNR logging) not landed"
    )


def main() -> int:
    t0 = time.time()
    probes = [probe_p1_n25_adapters(), probe_p2_grassmannian_skeleton(), probe_p3_t2_upstream()]
    passes = sum(1 for ok, _ in probes if ok)
    messages = [msg for _, msg in probes]
    killed = passes < 3

    result = {
        "experiment": "exp_g4_grassmannian_ap_pretrain",
        "status": "killed" if killed else "ready_for_heavy_run",
        "verdict": "killed" if killed else "pending_heavy_run",
        "all_pass": not killed,
        "wall_time_s": round(time.time() - t0, 4),
        "is_smoke": True,
        "kc_results": {
            "K1589": {
                "text": "interference ratio <= 0.67",
                "result": "fail" if killed else "pending",
                "measurement": "UNMEASURABLE" if killed else "TBD",
            }
        },
        "preconditions": {
            "P1_n25_adapters": probes[0][0],
            "P2_grassmannian_skeleton_port": probes[1][0],
            "P3_t2_upstream_rebuilt": probes[2][0],
        },
        "evidence": messages,
        "cohort": "audit-2026-04-17",
        "cohort_probe_kill_count": 11,
        "upstream_blocker": (
            "exp_p1_t2_single_domain_training rerun at LORA_SCALE=5, max_tokens>=512, "
            "rank sweep {2,4,6,12,24}, grad-SNR logging, 5+ disjoint domains"
        ),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if killed else 1  # killed path is the expected path this cohort


if __name__ == "__main__":
    raise SystemExit(main())
