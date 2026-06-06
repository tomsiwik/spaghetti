"""
V2 audit-rerun probe for T3.7: Hot-Remove Adapter Without Affecting Remaining.

Purpose: verify preconditions before claiming hot-remove PASS. V1 (2026-04-10)
claimed supported, but the claim is retroactively invalid for THREE independent
reasons -- each sufficient to kill on its own:

  1) Tautological routing (mem-antipattern-002). V1 code hardcodes
     REAL_ADAPTER_PATHS[domain] -> fixed adapter path and evaluates each adapter
     only on its matched-domain subset. Under that design K1070 "bit-exact
     remaining outputs after removing adapter k" degenerates to:
     "looking up a different dict key for a different-domain query."  The
     remaining domains (math/medical/legal/finance) were never routed to the
     removed "geography" key in the first place, so removing that key is
     algebraically a no-op on paths -- not evidence of remove-invariance.

  2) Adapter copy forgery (mem-antipattern-009). V1 synthesises "geography"
     and "history" adapters via shutil.copy from the finance adapter directory.
     The geography adapter IS the finance adapter byte-for-byte. K1071
     "freed slot reusable, history=100% vs base=4%" measures finance-weights
     on high_school_european_history MCQs, not a genuinely new adapter.
     This is (a) a format-transfer artefact (MCQ letter prediction), not
     domain-competence evidence, and (b) depends on finance weights being on
     disk -- which they are not (T2.6 adapter weights lost per audit).

  3) Dict mutation vs weight unload (mem-antipattern-011 specialisation).
     K1072 "hot-remove latency < 10ms" measures `del d[k]` over a Python dict
     (O(1) amortised by hash-table semantics -- trivially sub-microsecond).
     The Theorem 3 object should be the unload of the adapter *weights* from
     GPU memory, not the dict key removal. V1's 0.0009ms p99 under a 10ms
     threshold is a 10,000x margin on the wrong benchmark.

Additionally: 0/5 upstream adapter .safetensors are on disk. T2.1 upstream
KILLED 2026-04-18 (metric-swap + format-artefact). T2.6 adapter weights
absent from disk (configs only). Even a correctly-designed V3 cannot run
until those artefacts exist.

This probe does NOT load the model. It checks filesystem state that the V1
run should have produced, and routes K1070/K1071/K1072 to FAIL with reason
strings. Pure os.path inspection + micro-dict benchmark.
"""

from __future__ import annotations

import json
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
    "geography": EXPERIMENT_DIR / "adapter_geography",
    "history":   EXPERIMENT_DIR / "adapter_history",
}


def probe_adapter_dir(path: Path) -> dict:
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


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("V2 AUDIT PROBE: exp_p1_t3_plug_and_play_remove")
    print("=" * 72)

    adapters = {name: probe_adapter_dir(p) for name, p in EXPECTED_ADAPTERS.items()}
    stubs = {name: probe_adapter_dir(p) for name, p in LOCAL_STUBS.items()}
    n_real_present = sum(1 for v in adapters.values() if v["safetensors_exists"])
    n_stub_present = sum(1 for v in stubs.values() if v["safetensors_exists"])

    t21 = upstream_verdict(T21_DIR)
    t26 = upstream_verdict(T26_DIR)
    t36 = upstream_verdict(EXPERIMENT_DIR.parent / "exp_p1_t3_plug_and_play_add")

    # K1070: existing outputs bit-exact after removing adapter k.
    # Unmeasurable by design: V1 REAL_ADAPTER_PATHS[domain] routing never sends
    # a remaining-domain query to the removed adapter, so "no change after remove"
    # is algebraically forced regardless of weights. Additionally, 0/5 real
    # safetensors present means no weights could be loaded for the baseline itself.
    k1070 = {
        "pass": False,
        "reason": (
            "Structurally unmeasurable. V1 routing is tautological "
            "(mem-antipattern-002): REAL_ADAPTER_PATHS[domain] hardcodes "
            "adapter-to-domain pairing, and each remaining domain is queried "
            "only against its matched adapter. Removing 'geography' changes "
            "the dict key 'geography' only -- the dict keys 'math', 'medical', "
            "'legal', 'finance' are untouched by Python dict semantics, not "
            "by Theorem 1. Genuine K1070 requires either (a) simultaneous "
            "activation of N>=2 adapters where removing k could interact with "
            "the remainder, or (b) per-sample routing where the router chooses "
            "among the live set and may route differently after a removal. "
            f"Additionally, 0/5 expected adapter .safetensors present on disk: "
            f"math={adapters['math']['safetensors_exists']}, "
            f"code={adapters['code']['safetensors_exists']}, "
            f"medical={adapters['medical']['safetensors_exists']}, "
            f"legal={adapters['legal']['safetensors_exists']}, "
            f"finance={adapters['finance']['safetensors_exists']}."
        ),
    }

    # K1071: freed slot reusable. V1 synthesises 'geography' and 'history' via
    # shutil.copy from the finance adapter directory (adapter copy forgery).
    # 'history' is finance byte-for-byte, so history=100% on high_school_european_history
    # measures format-transfer of finance weights, not a new adapter.
    geo_info = stubs["geography"]
    hist_info = stubs["history"]
    k1071 = {
        "pass": False,
        "reason": (
            "Adapter copy forgery (mem-antipattern-009). V1 constructs both "
            "the removed adapter (geography = shutil.copy(finance)) and the "
            "freed-slot replacement (history = shutil.copy(finance)). They "
            "share weights byte-for-byte; K1071's 'history=100% vs base=4%' "
            "is finance-weights answering MCQ letters, not a novel adapter "
            "occupying a freed slot. Additionally, finance source adapter "
            "has no .safetensors on disk, so even the copy cannot be produced "
            "in V2. Probe: "
            f"geography_stub_has_safetensors={geo_info['safetensors_exists']}, "
            f"history_stub_has_safetensors={hist_info['safetensors_exists']}, "
            f"finance_source_has_safetensors={adapters['finance']['safetensors_exists']}."
        ),
    }

    # K1072: hot-remove latency < 10ms. Theorem 3 object should be weight
    # deallocation / GPU memory release, but V1 measured a bare dict.__delitem__.
    dt_ms = 0.0
    d = {f"key_{i}": f"/nonexistent/{i}" for i in range(1000)}
    for k in list(d.keys()):
        a = time.perf_counter()
        del d[k]
        dt_ms += (time.perf_counter() - a) * 1000.0
    mean_dict_del_ms = dt_ms / 1000.0
    k1072 = {
        "pass": False,
        "mean_dict_del_ms": mean_dict_del_ms,
        "reason": (
            "Measures the wrong object (mem-antipattern-011 specialisation). "
            "Python `del d[k]` is O(1) amortised and trivially sub-microsecond "
            "regardless of adapter semantics. The Theorem 3 object is adapter "
            "*weight* unload (release GPU memory, close mmap, drop model "
            "references) -- not a hash-table deletion. V1 reported "
            "p99=0.0009ms against a 10ms threshold -- a 10,800x margin on a "
            "benchmark that could not have failed under any adapter "
            f"implementation. Probe reproduces: mean dict del={mean_dict_del_ms:.6f}ms."
        ),
    }

    total_s = time.perf_counter() - t0

    results = {
        "verdict": "KILLED",
        "all_pass": False,
        "ran": True,
        "is_smoke": False,
        "_v2_note": (
            "V2 audit-rerun 2026-04-18. V1 'supported' (2026-04-10) retroactively "
            "invalid for THREE independent reasons: (a) tautological routing "
            "REAL_ADAPTER_PATHS[domain] makes K1070 algebraically forced, not "
            "Theorem 1 evidence; (b) adapter copy forgery -- both removed and "
            "replacement adapters are shutil.copy(finance) so K1071 measures "
            "finance-weights under two labels, not a freed-slot test; "
            "(c) K1072 times dict deletion rather than weight unload. "
            "Additionally 0/5 upstream adapter .safetensors on disk "
            "(T2.1 KILLED 2026-04-18, T2.6 weights lost)."
        ),
        "_audit_tags": [
            "audit-2026-04-17-rerun",
            "tautological-routing",
            "adapter-copy-forgery",
            "latency-wrong-object",
            "precondition-probe-7th-instance",
        ],
        "adapter_preconditions": adapters,
        "local_stub_preconditions": stubs,
        "n_real_adapter_safetensors_present": n_real_present,
        "n_local_stub_safetensors_present": n_stub_present,
        "upstream": {
            "exp_p1_t2_single_domain_training": t21,
            "exp_p1_t2_multi_domain_5":         t26,
            "exp_p1_t3_plug_and_play_add":      t36,
        },
        "v1_design_flaws": [
            "REAL_ADAPTER_PATHS[domain] hardcodes adapter-to-domain pairing",
            "K1070 'bit-exact' is tautological: remaining-domain queries never routed to removed key",
            "K1071 'freed slot' is adapter copy forgery: geography==history==finance (shutil.copy)",
            "K1072 times dict.__delitem__ (O(1)) not weight unload",
            "shutil.copy(finance, history_dir) mint adapters from existing weights",
            "Phase 4 'high_school_european_history' MCQ measures format transfer of finance weights, not novelty",
        ],
        "k1070": k1070,
        "k1071": k1071,
        "k1072": k1072,
        "K1070_remaining_outputs_unchanged": "FAIL",
        "K1071_freed_slot_reusable":         "FAIL",
        "K1072_remove_latency":              "FAIL",
        "total_time_s": total_s,
        "_v1_numbers_for_reference": {
            "note": "V1 2026-04-10 measurements. Unverifiable now; kept for provenance only.",
            "total_diffs_post_remove": 0,
            "history_acc_pct": 100.0,
            "base_acc_pct": 4.0,
            "remove_p99_ms": 0.000922,
            "remove_mean_ms": 0.000205,
        },
    }

    out_path = EXPERIMENT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[probe] wrote {out_path.relative_to(REPO_ROOT)}")
    print(
        f"[probe] verdict=KILLED n_real_adapter_safetensors={n_real_present}/5 "
        f"n_local_stub_safetensors={n_stub_present}/2"
    )
    print(f"[probe] elapsed={total_s:.3f}s")


if __name__ == "__main__":
    main()
