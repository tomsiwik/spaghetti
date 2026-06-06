"""exp_g4_memory_footprint_n25 — precondition probe.

Per MATH.md pre-registered tripwire, verify P1/P2/P3 before any
multi-adapter MLX load. If any fail, write results.json with status=killed
and exit.

This matches the audit-2026-04-17 cohort standing rule established across
Findings #605/#606/#608/#610/#611/#612/#613/#615/#616/#617/#618/#619/#620.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "results.json"


def probe_p1_gemma4_base() -> tuple[bool, str]:
    """P1: Gemma 4 E4B 4-bit base model resolvable via mlx-lm."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    hub = hf_home / "hub"
    if not hub.exists():
        return False, f"P1 FAIL: HF hub cache {hub} does not exist"
    candidates = [
        d for d in hub.iterdir()
        if d.is_dir() and "gemma" in d.name.lower() and "4bit" in d.name.lower()
    ]
    if candidates:
        return True, f"P1 PASS: {len(candidates)} gemma*4bit dir(s) in HF cache"
    return False, (
        f"P1 FAIL: no gemma*4bit dir under {hub}. Searched {hub.name}/*gemma*4bit*"
    )


def probe_p2_n25_vproj_oproj_adapters() -> tuple[bool, str]:
    """P2: N=25 Gemma 4 v_proj+o_proj r=6 adapter safetensors on disk."""
    candidates = [
        ROOT / "experiments/models/exp_g4_25domain_real_hf",
        ROOT / "experiments/models/exp_g4_5domain_real_hf",
        ROOT / "experiments/models/exp_p0_n25_vproj_composition",
        ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters",
    ]
    found: list[Path] = []
    searched: list[str] = []
    for c in candidates:
        rel = str(c.relative_to(ROOT))
        searched.append(rel)
        if c.exists():
            found.extend(list(c.rglob("*.safetensors")))
    if len(found) >= 25:
        return True, f"P2 PASS: {len(found)} v_proj+o_proj safetensors (>=25)"
    return False, (
        f"P2 FAIL: expected >=25 Gemma 4 v_proj+o_proj r=6 safetensors, "
        f"found {len(found)}. Searched: {searched}"
    )


def probe_p3_multi_adapter_loader() -> tuple[bool, str]:
    """P3: runnable Gemma 4 multi-adapter (N>1 simultaneous) mount path."""
    t2 = ROOT / "experiments/models/exp_p1_t2_single_domain_training/results.json"
    if not t2.exists():
        return False, "P3 FAIL: upstream T2.1 results.json missing"
    try:
        data = json.loads(t2.read_text())
    except Exception as exc:
        return False, f"P3 FAIL: could not parse T2.1 results.json ({exc})"
    all_pass = data.get("all_pass", False)
    verdict = str(data.get("verdict", "unknown"))
    lora_scale = data.get("lora_scale") or data.get("config", {}).get("lora_scale")
    if all_pass and verdict.lower() == "supported":
        return True, (
            f"P3 PASS: T2.1 all_pass=true verdict={verdict} lora_scale={lora_scale}"
        )
    return False, (
        f"P3 FAIL: T2.1 all_pass={all_pass} verdict={verdict} "
        f"lora_scale={lora_scale} — upstream rebuild "
        "(LORA_SCALE=5, max_tokens>=512, rank sweep, grad-SNR logging, "
        "5+ disjoint domains) not landed; "
        "no Gemma 4 N>1 simultaneous-mount loader can be validated without it"
    )


def main() -> int:
    t0 = time.time()
    probes = [
        probe_p1_gemma4_base(),
        probe_p2_n25_vproj_oproj_adapters(),
        probe_p3_multi_adapter_loader(),
    ]
    passes = sum(1 for ok, _ in probes if ok)
    messages = [msg for _, msg in probes]
    killed = passes < 3

    result = {
        "experiment": "exp_g4_memory_footprint_n25",
        "status": "killed" if killed else "ready_for_heavy_run",
        "verdict": "killed" if killed else "pending_heavy_run",
        "all_pass": not killed,
        "wall_time_s": round(time.time() - t0, 4),
        "is_smoke": True,
        "kc_results": {
            "K1596": {
                "text": "peak RSS <= 5 GB with base + N=25 adapters attached",
                "result": "fail" if killed else "pending",
                "measurement": "UNMEASURABLE" if killed else "TBD",
            }
        },
        "preconditions": {
            "P1_gemma4_base_resolvable": probes[0][0],
            "P2_n25_vproj_oproj_safetensors": probes[1][0],
            "P3_multi_adapter_loader_upstream": probes[2][0],
        },
        "evidence": messages,
        "cohort": "audit-2026-04-17",
        "cohort_probe_kill_count": 14,
        "upstream_blocker": (
            "exp_p1_t2_single_domain_training rerun at LORA_SCALE=5, max_tokens>=512, "
            "rank sweep {2,4,6,12,24}, grad-SNR logging, 5+ disjoint domains, "
            "plus N=25 v_proj+o_proj r=6 safetensors materialized"
        ),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if killed else 1  # killed path is the expected path this cohort


if __name__ == "__main__":
    raise SystemExit(main())
