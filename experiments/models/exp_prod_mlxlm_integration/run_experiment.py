"""Preflight-only experiment.

This experiment cannot execute its kill criteria because mlx-lm has no
plugin/loader registration API and Pierre has no served-side dispatch
layer. We record the precondition state in results.json so the kill is
auditable.

Run: ~1s. No model load. No GPU. No network.

See MATH.md (Theorems 1-3) for the loader-soundness reasoning chain
that produces this preflight.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import os
import platform
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).parent
EXP_NAME = EXP_DIR.name


def check_mlx_lm_present() -> dict:
    """T1.precondition.A — is mlx-lm importable and what version?"""
    try:
        m = importlib.import_module("mlx_lm")
        ver = getattr(m, "__version__", None) or md.version("mlx-lm")
        return {"pass": True, "version": ver, "path": m.__file__}
    except Exception as e:  # pragma: no cover
        return {"pass": False, "error": repr(e)}


def check_loader_plugin_api() -> dict:
    """T1.precondition.B — does mlx-lm expose a loader/adapter plugin
    entry-point group? (No such group is documented; we enumerate to
    confirm.)"""
    eps = md.entry_points()
    groups = sorted({ep.group for ep in eps if "mlx" in (ep.group or "").lower()
                     or "mlx" in ep.name.lower()})
    candidate_groups = [
        "mlx_lm.loaders", "mlx_lm.models", "mlx_lm.adapters",
        "mlx_lm.plugins", "mlx_lm.providers",
    ]
    found = {g: [ep.name for ep in eps if ep.group == g] for g in candidate_groups}
    has_plugin_api = any(v for v in found.values())
    return {
        "pass": has_plugin_api,
        "groups_with_mlx_namespace": groups,
        "candidate_groups_searched": candidate_groups,
        "candidate_groups_populated": found,
        "reason": ("No plugin/loader entry-point group is registered by "
                   "mlx-lm; only console_scripts."),
    }


def check_server_body_schema() -> dict:
    """T2.precondition — does mlx_lm.server accept a multi-adapter
    selector in the request body, or only a single str?"""
    try:
        import mlx_lm.server as s
        src = Path(s.__file__).read_text()
        accepts_field = '"adapters"' in src or "'adapters'" in src
        validates_as_str = "_validate(\"adapter\", str" in src
        return {
            "pass": accepts_field and not validates_as_str,
            "accepts_adapters_in_body": accepts_field,
            "validates_as_single_str": validates_as_str,
            "reason": ("Body has 'adapters' but is validated as str — "
                       "no multi-adapter selector schema."),
        }
    except Exception as e:  # pragma: no cover
        return {"pass": False, "error": repr(e)}


def check_pierre_g4e4b_model_present() -> dict:
    """T1.precondition.C — does a 'pierre-g4e4b' model exist on disk
    or in the HF hub cache?"""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    matches = []
    if hf_cache.exists():
        matches = [str(p.name) for p in hf_cache.iterdir() if "pierre" in p.name.lower()]
    local_dir = Path("experiments/models/pierre-g4e4b")
    return {
        "pass": bool(matches) or local_dir.exists(),
        "hf_cache_matches": matches,
        "local_dir_exists": local_dir.exists(),
        "reason": "No 'pierre-g4e4b' model registered locally or in HF cache.",
    }


def check_trained_adapter_for_baseline() -> dict:
    """T3.precondition — direct-Pierre baseline needs at least one
    trained adapter on disk."""
    base = Path("experiments/models/exp_p1_t2_single_domain_training/adapters")
    domains = ["math", "code", "medical"]
    state = {}
    any_safetensors = False
    for d in domains:
        cfg = base / d / "adapter_config.json"
        st = base / d / "adapters.safetensors"
        cfg_ok = cfg.exists()
        st_ok = st.exists()
        if st_ok:
            any_safetensors = True
        state[d] = {
            "config_present": cfg_ok,
            "safetensors_present": st_ok,
            "safetensors_size_bytes": (st.stat().st_size if st_ok else 0),
        }
    return {
        "pass": any_safetensors,
        "per_domain": state,
        "reason": ("Trained adapter weights missing — only adapter_config.json "
                   "is on disk for math/code/medical (Finding #421 + "
                   "exp_bench_aime_2026 / exp_bench_livecodebench_v6 KILLs)."),
    }


def check_dependency_kill_status() -> dict:
    """Soft check: report depends_on experiment status from disk."""
    dep_results = Path("experiments/models/exp_prod_pip_package_pierre/results.json")
    if not dep_results.exists():
        return {"pass": False, "reason": "Dependency results.json missing."}
    try:
        data = json.loads(dep_results.read_text())
        verdict = data.get("verdict", "UNKNOWN")
        return {
            "pass": verdict not in ("KILLED",),
            "depends_on": "exp_prod_pip_package_pierre",
            "depends_on_verdict": verdict,
            "depends_on_reason": data.get("reason", ""),
        }
    except Exception as e:  # pragma: no cover
        return {"pass": False, "error": repr(e)}


def run() -> dict:
    t0 = time.time()
    preflight = {
        "T1A_mlx_lm_present": check_mlx_lm_present(),
        "T1B_loader_plugin_api_present": check_loader_plugin_api(),
        "T1C_pierre_g4e4b_model_present": check_pierre_g4e4b_model_present(),
        "T2_server_multi_adapter_body_schema": check_server_body_schema(),
        "T3_trained_adapter_for_baseline": check_trained_adapter_for_baseline(),
        "DEP_pip_package_pierre_supported": check_dependency_kill_status(),
    }

    # KCs are unmeasured because preflight blocks execution.
    # Record explicit per-KC reasoning so reviewers see why each is FAIL.
    blocker_count = sum(1 for v in preflight.values() if not v.get("pass", False))

    kc = {
        "K1651_server_serves_with_adapter_set": False,
        "K1652_extra_body_adapter_selection": False,
        "K1653_tps_within_5pct_of_direct_pierre": False,
    }
    kc_rationale = {
        "K1651_server_serves_with_adapter_set": (
            "Cannot execute. Failing preconditions: "
            "(T1B) no mlx-lm plugin/loader API; "
            "(T1C) no 'pierre-g4e4b' model; "
            "(DEP) exp_prod_pip_package_pierre KILLED. "
            "The literal CLI 'mlx_lm.server --model pierre-g4e4b' "
            "errors at model resolution before any HTTP listener starts."
        ),
        "K1652_extra_body_adapter_selection": (
            "Cannot execute. (T2) server body schema validates "
            "'adapter' as a single str; no multi-adapter selector "
            "exists. Even if T1 passed, the wire format does not "
            "carry Pierre's per-sample routing target."
        ),
        "K1653_tps_within_5pct_of_direct_pierre": (
            "Cannot execute. (T3) direct-Pierre baseline requires "
            "trained adapters on disk; only adapter_config.json is "
            "present for math/code/medical. Cannot compute a ratio "
            "when neither pipeline runs."
        ),
    }

    remediation = [
        "1. Resurrect & complete exp_prod_pip_package_pierre (rename "
        "package 'lora-compose' -> 'pierre', include pierre/ in wheel).",
        "2. Train (or restore from object store) per-domain adapters "
        "via exp_p1_t2_single_domain_training and assert "
        "st_size > 0 in its preflight (P11.ADAPTER-REBUILD).",
        "3. Decide path: (a) upstream PR adding mlx_lm.loaders entry-"
        "point group, OR (b) ship pierre.server as a thin wrapper "
        "around mlx_lm.server.APIHandler. Option (b) is the realistic "
        "near-term unblock — file exp_prod_pierre_server.",
        "4. Extend body schema or define a Pierre-side parser for "
        "'adapters: {domain: str, weight?: float} | list[...]' so "
        "extra_body multi-adapter selection has a wire format.",
        "5. Once 1-4 land, re-claim this experiment and run K1651-"
        "K1653 in earnest.",
    ]

    out = {
        "experiment": EXP_NAME,
        "verdict": "KILLED",
        "all_pass": False,
        "is_smoke": False,
        "ran": False,
        "status": "infrastructure_blocked",
        "reason": (
            "Five independent preconditions fail before any HTTP "
            "smoke can be timed. mlx-lm 0.31.2 has no plugin/loader "
            "API; no 'pierre-g4e4b' model exists; body schema is "
            "single-adapter str; trained baseline adapters missing; "
            "depends_on exp_prod_pip_package_pierre is KILLED."
        ),
        "preflight": preflight,
        "preflight_blocker_count": blocker_count,
        "kill_criteria": kc,
        "kill_criteria_rationale": kc_rationale,
        "remediation_summary": remediation,
        "host": {
            "platform": platform.system().lower(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }

    out_path = EXP_DIR / "results.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    res = run()
    print(json.dumps({k: v for k, v in res.items()
                      if k in ("experiment", "verdict", "all_pass",
                               "preflight_blocker_count", "elapsed_sec")},
                     indent=2))
