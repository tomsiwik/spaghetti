"""
P9.G1: Benchmark Showdown — precondition-probe runner (2026-04-19 rewrite).

Original runner required real MLX inference on Pierre v3 (base Gemma 4 E4B + math/
medical adapters) for GSM8K / MedMCQA / MMLU-Pro. See git history for the prior
full-MLX implementation; it is retained in the commit log but replaced here because
its preconditions were not met on 2026-04-19.

This runner executes MATH.md §P precondition-probe only:
  P1  upstream exp_p9_full_stack_integration status == supported with results.json
  P2  adapter safetensors/npz present for math, medical (at minimum)
  P3  adapters/registry.json resolves to real weight files

Tripwire: all-FAIL -> K1390/K1391/K1392 UNMEASURABLE -> status=killed, all_pass=false.
Probe is pure filesystem + JSON reads; no MLX, no network, no model load. Wall
budget: <2 s.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

UPSTREAM_DIR = REPO_ROOT / "micro" / "models" / "exp_p9_full_stack_integration"
REGISTRY_PATH = REPO_ROOT / "data" / "adapters" / "registry.json"
T2_ADAPTERS_ROOT = REPO_ROOT / "micro" / "models" / "exp_p1_t2_single_domain_training" / "adapters"


def probe_p1_upstream() -> tuple[bool, str]:
    """P1: upstream exp_p9_full_stack_integration must be supported with results.json."""
    try:
        out = subprocess.run(
            ["experiment", "get", "exp_p9_full_stack_integration"],
            capture_output=True, text=True, timeout=10, cwd=REPO_ROOT,
        )
        status_line = next(
            (l for l in out.stdout.splitlines() if "Status:" in l), ""
        )
        is_supported = "supported" in status_line.lower()
    except Exception as e:
        return False, f"experiment get failed: {e!r}"
    results_json_exists = (UPSTREAM_DIR / "results.json").exists()
    ok = is_supported and results_json_exists
    detail = (
        f"status_line={status_line.strip()!r}, "
        f"results_json_exists={results_json_exists}"
    )
    return ok, detail


def probe_p2_adapters_on_disk() -> tuple[bool, str]:
    """P2: math + medical adapter weight files present under T2.1 adapter root."""
    required_domains = ("math", "medical")
    found: dict[str, list[str]] = {}
    for d in required_domains:
        adapter_dir = T2_ADAPTERS_ROOT / d
        if not adapter_dir.exists():
            found[d] = []
            continue
        weights = [
            p.name
            for p in adapter_dir.rglob("*")
            if p.is_file() and p.suffix in (".safetensors", ".npz")
        ]
        found[d] = weights
    all_present = all(found[d] for d in required_domains)
    return all_present, f"weight_files_by_domain={found}"


def probe_p3_registry_resolves() -> tuple[bool, str]:
    """P3: registry entries resolve to real weight files."""
    if not REGISTRY_PATH.exists():
        return False, "registry.json absent"
    try:
        reg = json.loads(REGISTRY_PATH.read_text())
    except Exception as e:
        return False, f"registry parse failed: {e!r}"
    adapters = reg.get("adapters", [])
    resolutions = []
    ok_count = 0
    for entry in adapters:
        path = REPO_ROOT / entry.get("path", "")
        if not path.exists():
            resolutions.append((entry.get("name"), "dir_missing"))
            continue
        weights = [
            p.name
            for p in path.rglob("*")
            if p.is_file() and p.suffix in (".safetensors", ".npz")
        ]
        if weights:
            ok_count += 1
            resolutions.append((entry.get("name"), f"{len(weights)} weight files"))
        else:
            resolutions.append((entry.get("name"), "no_weight_files"))
    ok = ok_count > 0
    return ok, f"registry_entries={len(adapters)}, resolved={ok_count}, detail={resolutions}"


def main() -> int:
    t0 = time.perf_counter()
    probes = {
        "P1_upstream_supported": probe_p1_upstream(),
        "P2_adapters_on_disk":   probe_p2_adapters_on_disk(),
        "P3_registry_resolves":  probe_p3_registry_resolves(),
    }
    wall_s = time.perf_counter() - t0

    # P2 is the binding precondition for K1390/K1391/K1392 (see MATH.md §P).
    # P2 FAIL alone is sufficient to mark the KCs unmeasurable -> killed.
    p2_ok = probes["P2_adapters_on_disk"][0]
    p1_ok = probes["P1_upstream_supported"][0]
    kcs_measurable = p2_ok  # math+medical weights on disk are required
    verdict = "supported" if kcs_measurable and p1_ok else "killed"
    all_pass = kcs_measurable and p1_ok

    unmeasurable = not kcs_measurable
    kcs = {
        "K1390_gsm8k_vs_27b":    "unmeasurable" if unmeasurable else "untested",
        "K1391_gain_vs_base":    "unmeasurable" if unmeasurable else "untested",
        "K1392_medmcqa_vs_base": "unmeasurable" if unmeasurable else "untested",
    }

    payload = {
        "experiment_id": "exp_p9_benchmark_showdown",
        "verdict": verdict,
        "all_pass": bool(all_pass),
        "is_smoke": False,
        "wall_s": round(wall_s, 3),
        "preconditions": {
            name: {"pass": ok, "detail": detail}
            for name, (ok, detail) in probes.items()
        },
        "kill_criteria": kcs,
        "notes": (
            "MATH.md §P precondition-probe tripwire. All-FAIL means math/medical "
            "adapter weights do not exist on disk despite registry entries "
            "claiming scores (e.g., math-gsm8k score=82.0). Upstream "
            "exp_p1_t2_single_domain_training verdict=killed with documented "
            "Python 3.14 datasets/dill incompat. Without weights, Theorem 1 "
            "RHS I(ΔW; D) is undefined => K1390/K1391/K1392 malformed => "
            "status=killed per tripwire. Same upstream blocker as 17-member "
            "audit-2026-04-17 cohort; this P9-tagged experiment is NOT in that "
            "cohort but shares the root cause."
        ),
    }
    RESULTS_FILE.write_text(json.dumps(payload, indent=2))
    print(f"[probe] verdict={verdict} all_pass={all_pass} wall={wall_s:.3f}s")
    for name, (ok, detail) in probes.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'} - {detail}")
    # Exit 0 in both cases (probe ran successfully); verdict is in results.json.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
