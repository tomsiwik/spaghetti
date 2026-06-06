#!/usr/bin/env python3
"""exp_g4_per_token_top2_routing

Port Finding #58 (per-token top-2 routing, BitNet) to Gemma 4 E4B 4-bit at N=25.
KC K1578 demands `routed_PPL < 0.95 * exclusive_PPL` on 5 Gemma 4 domains.

Routed through a pre-registered precondition probe (MATH.md Preconditions).
If any of P1/P2/P3 fails, K1578 is FAIL (unmeasurable) and verdict = KILLED.
Heavy training (~4h MLX) is skipped on probe failure — the 7th instance in the
audit-2026-04-17 precondition-probe KILL cohort.
"""

import json
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent
MODELS_DIR = EXPERIMENT_DIR.parent

T21_DIR = MODELS_DIR / "exp_p1_t2_single_domain_training"
T21_ADAPTERS = T21_DIR / "adapters"
T21_RESULTS = T21_DIR / "results.json"
N25_STUB_DIR = MODELS_DIR / "exp_p0_n25_vproj_composition"

HIDDEN_PROBE_DIR = MODELS_DIR / "exp_g4_hidden_state_probe"
TFIDF_CLEAN_DIR = MODELS_DIR / "exp_g4_tfidf_ridge_n25_clean"
TFIDF_NOALIAS_DIR = MODELS_DIR / "exp_g4_tfidf_routing_no_alias"
TFIDF_GEMMA4_DIR = MODELS_DIR / "exp_p1_t4_tfidf_routing_gemma4"

EXPECTED_DOMAINS = ["math", "code", "medical", "finance", "legal"]


def _adapter_weights_count(root: Path) -> int:
    if not root.exists():
        return 0
    return len(list(root.rglob("adapters.safetensors")))


def probe_p1_adapter_weights() -> dict:
    present_domains = []
    for dom in EXPECTED_DOMAINS:
        weight_path = T21_ADAPTERS / dom / "adapters.safetensors"
        if weight_path.exists() and weight_path.stat().st_size > 0:
            present_domains.append(dom)
    n25_weights = _adapter_weights_count(N25_STUB_DIR)
    total = len(present_domains) + n25_weights
    return {
        "precondition": "P1_gemma4_adapter_weights",
        "target_upstream": str(T21_DIR),
        "expected_domains": EXPECTED_DOMAINS,
        "t21_domain_weights_present": present_domains,
        "n25_stub_dir_weights_count": n25_weights,
        "total_weights_available": total,
        "min_required": 5,
        "status": "PASS" if total >= 5 else "FAIL",
        "note": (
            "Upstream T2.1 (exp_p1_t2_single_domain_training) is KILLED "
            "(metric-swap + format-artifact); adapters/<dom>/adapters.safetensors "
            "are missing from disk (only adapter_config.json stubs remain). "
            "exp_p0_n25_vproj_composition has 20 lora_config_*.yaml but no "
            "trained weights. Cannot measure exclusive_PPL or routed_PPL "
            "without Gemma 4 adapter .safetensors."
        ) if total < 5 else (
            f"{total} adapter weight files present; K1578 measurable if P2∧P3."
        ),
    }


def probe_p2_per_token_router() -> dict:
    candidate_paths = [
        HIDDEN_PROBE_DIR / "router.safetensors",
        HIDDEN_PROBE_DIR / "ridge_router.safetensors",
        HIDDEN_PROBE_DIR / "results.json",
    ]
    found = [str(p) for p in candidate_paths if p.exists()]
    probe_dir_exists = HIDDEN_PROBE_DIR.exists()
    return {
        "precondition": "P2_per_token_router",
        "candidate_router_artifacts": [str(p) for p in candidate_paths],
        "found": found,
        "probe_dir_exists": probe_dir_exists,
        "status": "PASS" if found else "FAIL",
        "note": (
            "No Gemma 4 per-token router exists. Finding #310 established the "
            "mechanism only on Qwen hidden states; Gemma 4 replication has not "
            "been run. Falling back to TF-IDF per-sequence routing as the "
            "per-token mechanism violates Finding #305 (Theorem C, shared-KV "
            "null result) — the test collapses to exclusive vs exclusive."
        ) if not found else "per-token router artifact present",
    }


def probe_p3_exclusive_tfidf_baseline() -> dict:
    candidate_upstream = [
        TFIDF_CLEAN_DIR / "results.json",
        TFIDF_NOALIAS_DIR / "results.json",
        TFIDF_GEMMA4_DIR / "results.json",
    ]
    valid_upstream = []
    for up in candidate_upstream:
        if not up.exists():
            continue
        try:
            data = json.loads(up.read_text())
        except Exception:
            continue
        verdict = str(data.get("verdict", "UNKNOWN")).upper()
        if verdict in ("SUPPORTED", "PROVEN", "PROVISIONAL"):
            valid_upstream.append({"path": str(up), "verdict": verdict})
        else:
            valid_upstream.append({"path": str(up), "verdict": verdict,
                                   "excluded": "verdict not supported/proven"})
    # We require at least one upstream with SUPPORTED verdict AND measured
    # exclusive_PPL field — neither exists.
    has_measured_exclusive_ppl = any(
        u.get("verdict") in ("SUPPORTED", "PROVEN")
        and (
            Path(u["path"]).exists()
            and "exclusive_ppl" in Path(u["path"]).read_text().lower()
        )
        for u in valid_upstream
    )
    return {
        "precondition": "P3_exclusive_tfidf_baseline",
        "candidates_checked": [str(p) for p in candidate_upstream],
        "upstream_verdicts": valid_upstream,
        "has_measured_exclusive_ppl_field": has_measured_exclusive_ppl,
        "status": "PASS" if has_measured_exclusive_ppl else "FAIL",
        "note": (
            "No upstream TF-IDF N=25 experiment on Gemma 4 measured "
            "exclusive_PPL. Finding #583 KILLED at 88% accuracy threshold; "
            "adapters needed to compute PPL never existed (P1 failure "
            "propagates). Finding #431 is routing-accuracy only (86.1%), not PPL."
        ) if not has_measured_exclusive_ppl else "exclusive_PPL baseline present",
    }


def main() -> int:
    probes = [
        probe_p1_adapter_weights(),
        probe_p2_per_token_router(),
        probe_p3_exclusive_tfidf_baseline(),
    ]
    all_pass = all(p["status"] == "PASS" for p in probes)

    if all_pass:
        results = {
            "verdict": "PROVISIONAL",
            "all_pass": False,
            "ran": False,
            "is_smoke": False,
            "probes": probes,
            "kcs": {
                "K1578": {
                    "status": "FAIL",
                    "reason": "probe-only path (PPL measurement loop not implemented)",
                    "threshold": "routed_PPL < 0.95 * exclusive_PPL on 5 domains",
                    "measured": None,
                },
            },
            "note": (
                "All preconditions PASS — full per-token top-2 routing loop "
                "would run here. Not implemented in probe-only mode; upgrade "
                "to --status provisional with TODO to implement the "
                "exclusive_PPL + routed_PPL measurement when unblocked."
            ),
        }
    else:
        failing = [p["precondition"] for p in probes if p["status"] == "FAIL"]
        results = {
            "verdict": "KILLED",
            "all_pass": False,
            "ran": False,
            "is_smoke": False,
            "probes": probes,
            "failing_preconditions": failing,
            "kcs": {
                "K1578": {
                    "status": "FAIL",
                    "reason": (
                        "unmeasurable: per-token top-2 routing vs exclusive "
                        "TF-IDF on Gemma 4 N=25 requires Gemma 4 adapter "
                        "weights (P1) + per-token router (P2) + exclusive "
                        f"TF-IDF PPL baseline (P3); missing → {', '.join(failing)}"
                    ),
                    "threshold": "routed_PPL < 0.95 * exclusive_PPL on 5 domains",
                    "measured": None,
                },
            },
            "note": (
                "Precondition-probe KILL (7th instance in audit-2026-04-17 "
                "cohort). Pre-registered routing per MATH.md: K1578 FAIL "
                "unmeasurable when any of P1/P2/P3 fail. No KC threshold "
                "was relaxed; no KC text modified post-hoc. Class-level "
                "standing rule: skip heavy Gemma 4 training (~4h MLX) when "
                "upstream adapter training is blocked by KILLED parents."
            ),
            "standing_rules_invoked": [
                "precondition-probe-before-macro-rerun",
                "adapter-registry-not-artefact",
                "downstream-inherits-upstream-audit",
                "shared-kv-null-routing-antipattern",
            ],
            "unblock_path": (
                "1. Rerun exp_p1_t2_single_domain_training at LORA_SCALE=5 "
                "(Finding #586 scale-safety) to regenerate math/code/medical "
                "adapters. 2. Train 2 more Gemma 4 adapters (finance, legal) "
                "on disjoint corpora at matched recipe. 3. Train a per-token "
                "ridge router on Gemma 4 hidden states (Finding #310 recipe, "
                "target ≥95% token accuracy). 4. Re-run this probe; if all "
                "PASS, implement the per-token top-2 PPL measurement loop."
            ),
        }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
