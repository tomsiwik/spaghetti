#!/usr/bin/env python3
"""T2.5 V2 (audit-2026-04-17-rerun, code-bug): SFT-Residual M2P on Gemma 4 E4B.

V1 code ran with T2.1 adapter weights that existed at that time and produced
acc_step0=80, acc_final=58, QR=0.707 → KILLED (PAPER.md v1). The audit tagged
this experiment audit-2026-04-17-rerun + code-bug meaning code is KNOWN-BUGGY
and a rerun is expected after fixing the cluster-level bug.

Applying the class-level standing rule (precondition-probe before macro rerun —
now 4th instance this loop):

  P1  T2.1 math adapter .safetensors on disk     (REQUIRED for B_sft load)
  P2  T2.1 math train/valid data on disk         (REQUIRED for training loop)
  P3  T2.1 upstream experiment not KILLED        (REQUIRED for B_sft validity)

If any precondition fails, the V1 'training corrupts B_sft' finding is
unverifiable AND any fix to the cluster bug (code-bug) cannot be measured.
Do NOT run the original heavy training loop — it will assert-fail at
load_b_sft (missing safetensors) and waste model-load time.

Kill criteria (re-pre-registered, unchanged from v1 MATH.md):
  K1044 M2P GSM8K acc ≥ 73.8% (= 0.90 × 82% SFT)
  K1045 B_applied compute < 10ms on M5 Pro
  K1046 ||B_applied - B_sft||_F = 0 at step 0 (all 42 layers)

All three require B_sft matrices → precondition P1. Without P1, all three FAIL
as unmeasurable (not measured-and-fell-short — explicitly routed). This is the
same honest-FAIL pattern used in the peer_comparison + mtbench_composed kills
earlier this loop.
"""

import json
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
T21_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training"
ADAPTER_PATH = T21_DIR / "adapters" / "math" / "adapters.safetensors"
TRAIN_JSONL = T21_DIR / "data" / "math" / "train.jsonl"
T21_RESULTS = T21_DIR / "results.json"


def probe_p1_adapter_weights() -> dict:
    """P1: T2.1 math adapter .safetensors present on disk."""
    exists = ADAPTER_PATH.exists()
    config_stub = (T21_DIR / "adapters" / "math" / "adapter_config.json").exists()
    size = ADAPTER_PATH.stat().st_size if exists else 0
    return {
        "precondition": "P1_adapter_weights",
        "target_path": str(ADAPTER_PATH),
        "exists": exists,
        "size_bytes": size,
        "config_stub_present": config_stub,
        "status": "PASS" if exists and size > 0 else "FAIL",
        "note": (
            "adapter_config.json stub present but adapter.safetensors missing"
            if config_stub and not exists else
            "both config and weights missing" if not config_stub else
            "weights present"
        ),
    }


def probe_p2_training_data() -> dict:
    """P2: T2.1 math train.jsonl present."""
    exists = TRAIN_JSONL.exists()
    n_lines = 0
    if exists:
        try:
            n_lines = sum(1 for _ in TRAIN_JSONL.open())
        except Exception:
            n_lines = -1
    return {
        "precondition": "P2_training_data",
        "target_path": str(TRAIN_JSONL),
        "exists": exists,
        "n_lines": n_lines,
        "status": "PASS" if exists and n_lines > 0 else "FAIL",
    }


def probe_p3_upstream_valid() -> dict:
    """P3: T2.1 (exp_p1_t2_single_domain_training) verdict != KILLED."""
    if not T21_RESULTS.exists():
        return {
            "precondition": "P3_upstream_valid",
            "status": "FAIL",
            "note": "T2.1 results.json missing — cannot verify upstream",
        }
    try:
        upstream = json.loads(T21_RESULTS.read_text())
    except Exception as e:
        return {
            "precondition": "P3_upstream_valid",
            "status": "FAIL",
            "note": f"T2.1 results.json parse error: {e}",
        }
    upstream_verdict = upstream.get("verdict", "UNKNOWN")
    return {
        "precondition": "P3_upstream_valid",
        "upstream_verdict": upstream_verdict,
        "upstream_all_pass": upstream.get("all_pass"),
        "status": "PASS" if upstream_verdict != "KILLED" else "FAIL",
        "note": (
            "T2.1 KILLED 2026-04-18 (metric-swap MedQA vs MedMCQA + format-"
            "artefact max_tokens=256 CoT truncation); even if math adapter "
            "existed, reported 82% is overstated per T2.1 _audit_note"
            if upstream_verdict == "KILLED" else "upstream valid"
        ),
    }


def main() -> None:
    print("T2.5 V2 precondition-probe (audit-2026-04-17-rerun, code-bug)", flush=True)
    print("=" * 60, flush=True)

    p1 = probe_p1_adapter_weights()
    p2 = probe_p2_training_data()
    p3 = probe_p3_upstream_valid()

    print(f"\nP1 {p1['status']}: {p1['note']}", flush=True)
    print(f"  {p1['target_path']}", flush=True)
    print(f"  exists={p1['exists']} size={p1['size_bytes']}B", flush=True)

    print(f"\nP2 {p2['status']}: n_lines={p2['n_lines']}", flush=True)
    print(f"  {p2['target_path']}", flush=True)

    print(f"\nP3 {p3['status']}: {p3['note']}", flush=True)

    preconditions_pass = all(p["status"] == "PASS" for p in [p1, p2, p3])

    # KC routing — every KC needs B_sft (from P1). Without P1, all FAIL as
    # unmeasurable. This is honest FAIL, not measured-and-fell-short.
    k1044_pass = False
    k1045_pass = False
    k1046_pass = False
    if preconditions_pass:
        print("\nPreconditions all PASS — would proceed to original training "
              "loop after applying code-bug fix. NOT IMPLEMENTED in v2 probe.",
              flush=True)
        kc_note = (
            "Preconditions pass. Probe does not itself measure KCs — it only "
            "gates whether measurement is possible. If preconditions pass in a "
            "future iteration, restore the v1 heavy training loop (git blame "
            "this file) and apply the code-bug cluster fix."
        )
    else:
        kc_note = (
            "Preconditions FAIL — K1044/K1045/K1046 routed FAIL as unmeasurable "
            "(no B_sft matrices; upstream T2.1 KILLED). This is 'cannot measure' "
            "not 'measured and fell short'. See PAPER.md V2 Audit section."
        )

    verdict = "PASS_PROBE_ONLY" if preconditions_pass else "KILLED"
    all_pass = preconditions_pass and k1044_pass and k1045_pass and k1046_pass

    results = {
        "verdict": verdict,
        "all_pass": all_pass,
        "ran": True,
        "is_smoke": False,
        "probe_version": "v2-audit-2026-04-17-rerun",

        "preconditions": {
            "P1_adapter_weights": p1,
            "P2_training_data": p2,
            "P3_upstream_valid": p3,
            "all_pass": preconditions_pass,
        },

        "kill_criteria": {
            "K1044_m2p_gsm8k_acc": {
                "threshold_pct": 73.8,
                "measured": None,
                "status": "PASS" if k1044_pass else "FAIL",
                "reason": kc_note,
            },
            "K1045_b_applied_compute_ms": {
                "threshold_ms": 10.0,
                "measured": None,
                "status": "PASS" if k1045_pass else "FAIL",
                "reason": kc_note,
            },
            "K1046_zero_init_frob": {
                "threshold": 1e-6,
                "measured": None,
                "status": "PASS" if k1046_pass else "FAIL",
                "reason": kc_note,
            },
        },

        "note": (
            "V1 (committed pre-audit) ran with T2.1 adapter weights that existed "
            "at the time — reported acc_step0=80%, acc_final=58%, QR=0.707, "
            "KILLED. Audit tagged this experiment audit-2026-04-17-rerun + "
            "code-bug; V2 rerun is blocked by upstream KILL (T2.1 metric-swap + "
            "format-artefact) AND missing adapter safetensors. Applying class-"
            "level precondition-probe rule (4th instance this loop). V2 "
            "verdict KILLED is unchanged from V1, but the kill-reason shifts "
            "from gradient-identity failure to upstream+artefact unavailability."
        ),

        "v1_numbers_for_reference": {
            "acc_step0_pct": 80.0,
            "acc_final_pct": 58.0,
            "quality_ratio": 0.707,
            "k1044_threshold_pct": 73.8,
            "relative_correction_of_b_sft": 0.2456,
            "source": "PAPER.md V1 §Results table; original results.json not committed",
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}", flush=True)
    print(f"Verdict: {verdict}", flush=True)
    print(f"all_pass: {all_pass}", flush=True)
    print(f"preconditions_pass: {preconditions_pass}", flush=True)
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
