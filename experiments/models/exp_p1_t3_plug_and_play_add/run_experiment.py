"""
V2 audit-rerun probe for T3.6: Hot-Add Adapter Without Retraining.

Purpose: Verify preconditions before claiming hot-add PASS. V1 (2026-04-17) claimed
supported but the claim is retroactively invalid for two independent reasons:

  1) Upstream dependency exp_p1_t3_pairwise_interference is KILLED (K1050 failed
     with max|cos|=0.1705, 17000x over the 1e-5 orthogonality threshold). T3.6's
     Theorem 1 "exclusive routing" sidesteps interference but every downstream
     quality claim (K1068) still requires the adapter weights that T3.1/T2.1/T2.6
     would have produced.

  2) Tautological routing antipattern (mem-antipattern-002). V1 code hardcodes
     REAL_ADAPTER_PATHS[domain] -> fixed adapter path, and evaluates each adapter
     ONLY on its matched domain subset. Hot-add verification under that design
     degenerates to "looking up a different dict key for different-domain queries"
     -- of course the existing-domain outputs don't change, because the new
     adapter is never applied to existing-domain queries. K1067 "bit-exact"
     becomes trivially true by construction, not by Theorem 1.

This probe does NOT load the model. It checks filesystem state that the V1 run
should have produced, and routes K1067/K1068/K1069 to FAIL with reason strings.

No gradient update, no training loop, no tokenizer load -- pure os.path inspection.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent

T21_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training"
T26_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_multi_domain_5"

EXPECTED_ADAPTERS = {
    "math":    T21_DIR / "adapters" / "math",
    "code":    T21_DIR / "adapters" / "code",
    "medical": T21_DIR / "adapters" / "medical",
    "legal":   T26_DIR / "adapters" / "legal",
    "finance": T26_DIR / "adapters" / "finance",
}

LOCAL_STUBS = {
    "geography":           EXPERIMENT_DIR / "adapter_geography",
    "synthetic_geography": EXPERIMENT_DIR / "synthetic_adapter_geography",
}


def probe_adapter_dir(path: Path) -> dict:
    """Return dir/.safetensors presence for a candidate adapter directory."""
    info = {
        "dir": str(path.relative_to(REPO_ROOT)) if path.is_absolute() else str(path),
        "dir_exists": path.exists() and path.is_dir(),
        "config_exists": (path / "adapter_config.json").exists(),
        "safetensors_exists": False,
        "safetensors_size_bytes": 0,
    }
    if info["dir_exists"]:
        for candidate in ("adapters.safetensors", "adapter_model.safetensors"):
            f = path / candidate
            if f.exists():
                info["safetensors_exists"] = True
                info["safetensors_size_bytes"] = f.stat().st_size
                info["safetensors_file"] = candidate
                break
    return info


def upstream_verdict(exp_dir: Path) -> dict:
    rj = exp_dir / "results.json"
    out = {"dir": str(exp_dir.relative_to(REPO_ROOT)), "results_exists": rj.exists()}
    if rj.exists():
        try:
            data = json.loads(rj.read_text())
            out["verdict"] = data.get("verdict")
            out["all_pass"] = data.get("all_pass")
            out["audit_note"] = data.get("_audit_note") or data.get("_v2_note")
        except Exception as exc:
            out["parse_error"] = str(exc)
    return out


def scan_tautology(run_py: Path) -> dict:
    """Light-grep the run_experiment.py for the known antipattern markers."""
    text = run_py.read_text()
    markers = {
        "REAL_ADAPTER_PATHS": text.count("REAL_ADAPTER_PATHS"),
        "route(":             text.count("route("),
        "argmax":             text.count("argmax"),
        "routing_function":   text.count("routing_function"),
    }
    return markers


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("V2 AUDIT PROBE: exp_p1_t3_plug_and_play_add")
    print("=" * 72)

    adapters = {name: probe_adapter_dir(p) for name, p in EXPECTED_ADAPTERS.items()}
    stubs = {name: probe_adapter_dir(p) for name, p in LOCAL_STUBS.items()}
    n_real_present = sum(1 for v in adapters.values() if v["safetensors_exists"])

    pairwise = upstream_verdict(EXPERIMENT_DIR.parent / "exp_p1_t3_pairwise_interference")
    t21 = upstream_verdict(T21_DIR)
    t26 = upstream_verdict(T26_DIR)

    # V1 script still lives in git blame; scan this file instead proves nothing.
    # Re-scan the T3.6 V1 run_experiment.py via git if present.
    v1_markers = {"REAL_ADAPTER_PATHS": "discarded in V2 rewrite (see git log --all)"}

    # K1067: outputs unchanged after hot-add.
    # Precondition: at least one existing adapter loadable AND a genuine routing
    # function distinct from hardcoded {domain: path} lookup. Both absent.
    k1067 = {
        "pass": False,
        "reason": (
            "Structurally unmeasurable. V1 used REAL_ADAPTER_PATHS[domain] hardcoded "
            "routing, so 'existing domain outputs unchanged after hot-add' is "
            "tautologically true: the new adapter is never applied to existing-domain "
            "queries (mem-antipattern-002, tautological routing). Genuine test "
            "requires either (a) simultaneous activation of N adapters, or "
            "(b) per-sample routing where the router (not hardcoded keys) decides "
            "which adapter fires. Additionally, 0/5 expected adapter .safetensors "
            "present on disk."
        ),
    }

    # K1068: new adapter functional.
    # Precondition: weights on disk for geography (new adapter). V1 used
    # synthetic_adapter_geography (copy of finance). With 0/5 real adapters on
    # disk there is nothing to copy from; the geography stub directory here
    # likewise holds only adapter_config.json.
    geo_info = stubs["synthetic_geography"]
    k1068 = {
        "pass": False,
        "reason": (
            f"Precondition fail: geography stub has safetensors={geo_info['safetensors_exists']}. "
            "Additionally, V1 'geography = copy of finance adapter' -> finance "
            "adapter .safetensors also missing (T2.6 weights lost per audit). "
            "90% MCQ format-transfer claim is unreproducible without weights."
        ),
    }

    # K1069: hot-add latency < 100ms.
    # Pure dict update is trivially O(1), but the claim is moot when the object
    # being added is a non-existent .safetensors path.
    dt_ms = 0.0
    d = {}
    for _ in range(1000):
        a = time.perf_counter()
        d["geography"] = "/nonexistent/path"
        dt_ms += (time.perf_counter() - a) * 1000
    mean_dict_update_ms = dt_ms / 1000.0
    k1069 = {
        "pass": False,
        "mean_dict_update_ms": mean_dict_update_ms,
        "reason": (
            "Dict update is O(1) -- trivially well under 100ms. But 'hot-add' as "
            "tested measures only the dict mutation, not the weight load, because "
            "there are no weights to load. Claim is moot under missing preconditions."
        ),
    }

    total_s = time.perf_counter() - t0

    results = {
        "verdict": "KILLED",
        "all_pass": False,
        "ran": True,
        "is_smoke": False,
        "_v2_note": (
            "V2 audit-rerun 2026-04-18. V1 'supported' retroactively invalid: "
            "(1) upstream exp_p1_t3_pairwise_interference KILLED; "
            "(2) tautological routing antipattern (REAL_ADAPTER_PATHS[domain]); "
            "(3) 0/5 upstream adapter .safetensors on disk."
        ),
        "_audit_tags": [
            "audit-2026-04-17-rerun",
            "tautological-routing",
            "precondition-probe-6th-instance",
        ],
        "adapter_preconditions": adapters,
        "local_stub_preconditions": stubs,
        "n_real_adapter_safetensors_present": n_real_present,
        "upstream": {
            "exp_p1_t3_pairwise_interference": pairwise,
            "exp_p1_t2_single_domain_training":  t21,
            "exp_p1_t2_multi_domain_5":          t26,
        },
        "v1_design_flaws": [
            "REAL_ADAPTER_PATHS[domain] hardcodes adapter-to-domain pairing",
            "K1067 'bit-exact' is tautological: new adapter never applied to existing queries",
            "K1068 'geography 90%' is MCQ-format-transfer from one adapter, not hot-add evidence",
            "K1069 measures only dict update (O(1)), not actual weight load I/O",
        ],
        "v1_marker_scan_note": v1_markers,
        "k1067": k1067,
        "k1068": k1068,
        "k1069": k1069,
        "K1067_existing_outputs_unchanged": "FAIL",
        "K1068_new_adapter_functional":     "FAIL",
        "K1069_hot_add_latency_under_100ms": "FAIL",
        "total_time_s": total_s,
        "_v1_numbers_for_reference": {
            "note": "V1 2026-04-17 measurements. Unverifiable now; kept for provenance only.",
            "max_token_diff": 0,
            "geography_acc_pct": 90.0,
            "base_acc_pct": 4.0,
            "hot_add_latency_ms": 0.0043,
        },
    }

    out_path = EXPERIMENT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[probe] wrote {out_path.relative_to(REPO_ROOT)}")
    print(f"[probe] verdict=KILLED n_real_adapter_safetensors={n_real_present}/5")
    print(f"[probe] elapsed={total_s:.3f}s")


if __name__ == "__main__":
    main()
