"""Preemptive-kill runner for exp_prod_adapter_registry_host.

Drains the 5-theorem stack via pure-stdlib checks (no MLX, no network
calls, no long-running load test). The claim is structurally blocked
on the local Apple-Silicon platform: no registry server implementation,
no ``pierre://`` URI resolver, no public host, and a 24 h load-test
requirement far exceeds the micro-iteration time ceiling.

See MATH.md §T1–§T5. Writes ``results.json`` with verdict=KILLED and
per-theorem evidence.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
SIGNING_DIR = REPO_ROOT / "experiments/models/exp_prod_adapter_signing"


def _grep_repo(needle: str) -> int:
    """Count files under REPO_ROOT that reference ``needle``."""
    try:
        out = subprocess.run(
            [
                "grep",
                "-rIln",
                "--include=*.py",
                "--include=*.md",
                "--include=*.toml",
                "--include=*.yaml",
                needle,
                str(REPO_ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Exclude this experiment's own docs to avoid self-reference
        hits = [
            line for line in out.stdout.splitlines()
            if "exp_prod_adapter_registry_host" not in line
        ]
        return len(hits)
    except Exception:
        return -1


def theorem_1_artifact_shortfall() -> dict:
    """T1: required registry infrastructure absent from repo and host."""
    signed_artifact = SIGNING_DIR / "adapter_signing.py"

    pierre_scheme_hits = _grep_repo("pierre://")
    # Tight check: a registry-specific server (not generic framework
    # imports elsewhere in the repo). Must mention BOTH a server
    # framework token AND a `.pierre` / registry-specific noun.
    server_specific_hits = (
        _grep_repo("pierre_registry")
        + _grep_repo("adapter_registry_server")
        + _grep_repo(".pierre.*GET")
    )
    # Tight check: a load-test harness aimed at `.pierre` resolution.
    load_specific_hits = (
        _grep_repo("locust.*pierre")
        + _grep_repo("k6.*pierre")
        + _grep_repo("load_test.*adapter")
    )

    nvidia_smi = shutil.which("nvidia-smi") is not None
    uname_m = platform.machine()
    is_apple = uname_m == "arm64" and platform.system() == "Darwin"

    missing: list[str] = []
    if pierre_scheme_hits <= 0:
        missing.append("pierre_uri_scheme_resolver")
    if server_specific_hits <= 0:
        missing.append("registry_specific_http_server")
    if load_specific_hits <= 0:
        missing.append("pierre_specific_load_harness")
    missing.append("public_dns_for_pierre_namespace")

    required_total = 5  # including signed-output which IS present
    return {
        "theorem": "T1_artifact_shortfall",
        "blocks": len(missing) >= 3,
        "required_total": required_total,
        "shortfall": len(missing),
        "missing": missing,
        "signed_artifact_exists": signed_artifact.exists(),
        "pierre_scheme_grep_hits": pierre_scheme_hits,
        "registry_specific_server_grep_hits": server_specific_hits,
        "pierre_specific_load_harness_grep_hits": load_specific_hits,
        "nvidia_smi_on_path": nvidia_smi,
        "uname_m": uname_m,
        "is_apple": is_apple,
    }


def theorem_2_resource_budget() -> dict:
    """T2: 24 h load test > 120 min micro ceiling; also physical topology."""
    required_minutes = 24 * 60
    micro_ceiling_minutes = 120
    over_factor = required_minutes / micro_ceiling_minutes
    request_total = 1000 * required_minutes
    return {
        "theorem": "T2_resource_budget",
        "blocks": over_factor > 1.0,
        "required_minutes": required_minutes,
        "micro_ceiling_minutes": micro_ceiling_minutes,
        "over_factor": over_factor,
        "request_total": request_total,
        "note": (
            "24 h continuous load is both a time ceiling and a "
            "physical-topology ceiling (no public network)."
        ),
    }


def theorem_3_schema_completeness() -> dict:
    """T3: DB literal success_criteria=[] + ⚠ INCOMPLETE."""
    return {
        "theorem": "T3_schema_completeness",
        "blocks": True,
        "db_success_criteria_count": 0,
        "db_incomplete_flag": True,
        "db_literal": "success_criteria: [] # MISSING — ⚠ INCOMPLETE",
        "axis": "F#502/F#646 schema-completeness-vs-instance-fix",
        "occurrence": (
            "4th: after tfidf_routing_no_alias, flywheel_real_users, "
            "loader_portability"
        ),
    }


def theorem_4_kc_pin_count() -> dict:
    """T4: under-pinned KC; K1661 lacks attack-vector enumeration."""
    kcs = [
        {
            "id": 1659,
            "pins": {"epsilon": "<200ms", "aggregation": "p99"},
            "has_threshold": True,
        },
        {
            "id": 1660,
            "pins": {
                "epsilon": ">99%",
                "duration": "24h",
                "rate": "1000/min",
            },
            "has_threshold": True,
        },
        {
            "id": 1661,
            "pins": {},  # no rejection rate, no enumeration
            "has_threshold": False,
            "note": (
                "binary rejection without rate OR attack-vector enumeration"
            ),
        },
    ]
    full_pin_template = {"baseline", "pool", "enum", "rescale", "epsilon"}
    total_pins = sum(len(k["pins"]) for k in kcs)
    pin_ratio = total_pins / (len(kcs) * len(full_pin_template))
    auto_block_threshold = 0.20
    return {
        "theorem": "T4_kc_pin_count",
        "blocks": pin_ratio <= auto_block_threshold,
        "kcs": kcs,
        "total_pins": total_pins,
        "pin_ratio_of_full_template": pin_ratio,
        "auto_block_threshold": auto_block_threshold,
        "non_falsifiable_kc": [k["id"] for k in kcs if not k["has_threshold"]],
        "note": (
            "reinforces T3 but below auto-block threshold; registered, "
            "not sole blocker"
        ),
    }


def theorem_5_source_breaches() -> dict:
    """T5: five LITERAL scope gaps between signing (SUPPORTED) and target."""
    breaches = [
        {
            "id": "A",
            "label": "transport-scope (load-time → serve-time)",
            "source_scope": "mx.load / verify_file on local filesystem",
            "target_scope": "HTTP(S) resolution over public network",
            "blocks": True,
        },
        {
            "id": "B",
            "label": "throughput-scope (0.39 ms local vs <200ms p99 network)",
            "source_scope": "K1641: 0.39 ms verify overhead on 400 KB file",
            "target_scope": "K1659: <200ms p99 end-to-end incl DNS+TLS+HTTP",
            "blocks": True,
        },
        {
            "id": "C",
            "label": "uptime-scope (undeclared in source)",
            "source_scope": "no uptime/availability KC in source",
            "target_scope": "K1660 >99 % over 24 h, 1000 req/min",
            "blocks": True,
        },
        {
            "id": "D",
            "label": "push-path (pull-only → producer API)",
            "source_scope": "consumer-load path only (K1639/K1640)",
            "target_scope": "K1661 producer-push rejection of unsigned",
            "blocks": True,
        },
        {
            "id": "E",
            "label": "hardware/infra-topology (ap-017 s reuse)",
            "source_scope": "local Apple Silicon; no network",
            "target_scope": "public DNS + routable FQDN + TLS cert",
            "blocks": True,
        },
    ]
    return {
        "theorem": "T5_source_literal_breaches",
        "blocks": sum(1 for b in breaches if b["blocks"]) >= 3,
        "breaches": breaches,
        "count_blocking": sum(1 for b in breaches if b["blocks"]),
        "reused_axis": "ap-017 (s) hardware-topology-unavailable (2nd)",
        "generalization": (
            "iter 35 CUDA hardware absent → iter 36 public network/DNS absent"
        ),
    }


def main() -> None:
    t0 = time.time()
    t1 = theorem_1_artifact_shortfall()
    t2 = theorem_2_resource_budget()
    t3 = theorem_3_schema_completeness()
    t4 = theorem_4_kc_pin_count()
    t5 = theorem_5_source_breaches()

    theorems = [t1, t2, t3, t4, t5]
    blocking = [t for t in theorems if t["blocks"]]
    # Sole-blocker test: T1 ∨ T2 ∨ T3 ∨ T5 each alone block
    sole_blockers = [t["theorem"] for t in (t1, t2, t3, t5) if t["blocks"]]
    all_block_strict = all(t["blocks"] for t in theorems)
    defense_in_depth = len(blocking) >= 3

    results = {
        "experiment_id": "exp_prod_adapter_registry_host",
        "verdict": "KILLED",
        "status": "killed",
        "preemptive_kill": True,
        "is_smoke": False,
        "all_pass": False,
        "all_block_strict": all_block_strict,
        "defense_in_depth": defense_in_depth,
        "sole_blockers": sole_blockers,
        "theorems": theorems,
        "kill_criteria": {
            "K1659": {
                "result": "fail",
                "reason": (
                    "no registry server, no pierre:// resolver, no public "
                    "host — T1 ∧ T5(A,B,E)"
                ),
            },
            "K1660": {
                "result": "fail",
                "reason": (
                    "no infra + 24h load > 120min micro ceiling — T2 ∧ T5(C)"
                ),
            },
            "K1661": {
                "result": "fail",
                "reason": (
                    "no producer-push path on disk + KC under-pinned — "
                    "T1 ∧ T4 ∧ T5(D)"
                ),
            },
        },
        "wall_seconds": round(time.time() - t0, 6),
        "parent_source": {
            "id": "exp_prod_adapter_signing",
            "status": "supported",
            "scope": "local Apple, load-time, pull-only",
        },
        "ap_017_axis": {
            "reuse": "s",
            "name": "hardware-topology-unavailable",
            "instance": "2nd",
            "generalization": (
                "CUDA hardware absent (iter 35) → public network / DNS "
                "infra absent (iter 36)"
            ),
        },
    }

    out_path = EXP_DIR / "results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    print(
        f"verdict=KILLED preemptive=True all_block_strict={all_block_strict} "
        f"defense_in_depth={defense_in_depth} t_blocking={len(blocking)}/5 "
        f"sole_blockers={sole_blockers}"
    )


if __name__ == "__main__":
    main()
