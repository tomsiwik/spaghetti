#!/usr/bin/env python3
"""exp_followup_ss_rn_path_valid_sft

Replicate Finding #403 at Gemma 4 scale with SFT-residual M2P on personalization
data (NOT SFT replay). KC K1572 demands |acc_final_Gemma4 - 74.4%| <= 5pp.

Routed through a pre-registered precondition probe (MATH.md Preconditions).
If any of P1/P2/P3 fails, K1572 is FAIL (unmeasurable) and verdict = KILLED.
Heavy training is skipped on probe failure to avoid a 30+ minute Gemma 4
model-load wall on a run that cannot produce measurable evidence. This is the
same honest-fail path used by 5 prior 2026-04-18 audit-rerun experiments.
"""

import json
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
T21_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training"
ADAPTER_PATH = T21_DIR / "adapters" / "math" / "adapters.safetensors"
ADAPTER_CONFIG_STUB = T21_DIR / "adapters" / "math" / "adapter_config.json"
SFT_TRAIN_JSONL = T21_DIR / "data" / "math" / "train.jsonl"
T21_RESULTS = T21_DIR / "results.json"


def probe_p1_adapter_weights() -> dict:
    exists = ADAPTER_PATH.exists()
    size = ADAPTER_PATH.stat().st_size if exists else 0
    return {
        "precondition": "P1_adapter_weights",
        "target_path": str(ADAPTER_PATH),
        "exists": exists,
        "size_bytes": size,
        "config_stub_present": ADAPTER_CONFIG_STUB.exists(),
        "status": "PASS" if exists and size > 0 else "FAIL",
        "note": (
            "B_sft (.safetensors) missing; adapter_config.json stub present "
            "but MLX LoRA weights gitignored and never committed. Synthesizing "
            "random B_sft reclassifies the run as T2.4 (random-init ΔB, already "
            "KILLED at QR=-5.89)."
            if not exists else "B_sft present"
        ),
    }


def probe_p2_personalization_data() -> dict:
    sft_present = SFT_TRAIN_JSONL.exists()
    n_sft_lines = 0
    if sft_present:
        try:
            with SFT_TRAIN_JSONL.open() as fh:
                n_sft_lines = sum(1 for _ in fh)
        except Exception:
            n_sft_lines = -1
    persona_dir_candidates = [
        EXPERIMENT_DIR / "data" / "personalization",
        EXPERIMENT_DIR / "data" / "persona",
        Path.cwd() / "data" / "personalization",
    ]
    persona_found = [str(p) for p in persona_dir_candidates if p.exists()]
    return {
        "precondition": "P2_personalization_data",
        "sft_data_path": str(SFT_TRAIN_JSONL),
        "sft_data_present": sft_present,
        "sft_data_n_lines": n_sft_lines,
        "persona_candidate_paths_found": persona_found,
        "status": "PASS" if persona_found else "FAIL",
        "note": (
            "No personalization corpus staged. Theorem C (EWC data separation) "
            "requires P ≠ GSM8K; substituting GSM8K re-creates the parent "
            "failure mode (QR=0.707)."
            if not persona_found else "persona corpus candidate present"
        ),
    }


def probe_p3_upstream_valid() -> dict:
    if not T21_RESULTS.exists():
        return {
            "precondition": "P3_upstream_valid",
            "status": "FAIL",
            "note": "T2.1 results.json missing — cannot verify upstream verdict",
        }
    try:
        upstream = json.loads(T21_RESULTS.read_text())
    except Exception as e:
        return {
            "precondition": "P3_upstream_valid",
            "status": "FAIL",
            "note": f"T2.1 results.json parse error: {e}",
        }
    verdict = upstream.get("verdict", "UNKNOWN")
    return {
        "precondition": "P3_upstream_valid",
        "upstream_verdict": verdict,
        "status": "PASS" if verdict != "KILLED" else "FAIL",
        "note": (
            "T2.1 KILLED 2026-04-18 (metric-swap MedMCQA↔MedQA + "
            "format-artefact max_tokens=256 truncating CoT). B_sft validity "
            "inherits the KILL by standing rule #3."
            if verdict == "KILLED" else "T2.1 verdict valid"
        ),
    }


def main() -> int:
    probes = [
        probe_p1_adapter_weights(),
        probe_p2_personalization_data(),
        probe_p3_upstream_valid(),
    ]
    all_preconditions_pass = all(p["status"] == "PASS" for p in probes)

    if all_preconditions_pass:
        results = {
            "verdict": "PROVISIONAL",
            "all_pass": False,
            "ran": False,
            "is_smoke": False,
            "probes": probes,
            "note": (
                "All preconditions PASS — Gemma 4 training loop would run "
                "here. Not implemented in probe-only mode; mark PROVISIONAL "
                "with TODO when preconditions are unblocked."
            ),
            "kcs": {
                "K1572": {
                    "status": "FAIL",
                    "reason": "probe-only path (training loop not implemented)",
                },
            },
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
                "K1572": {
                    "status": "FAIL",
                    "reason": (
                        "unmeasurable: Finding #403 Gemma 4 replication "
                        "(within-5pp band) requires B_sft (P1) + "
                        "personalization corpus (P2) + valid upstream (P3); "
                        f"missing → {', '.join(failing)}"
                    ),
                    "threshold": "|acc_final - 74.4%| <= 5pp",
                    "measured": None,
                },
            },
            "note": (
                "Precondition-probe KILL (6th instance this loop). "
                "Pre-registered routing per MATH.md: K1572 FAIL unmeasurable "
                "when any of P1/P2/P3 fail. No KC threshold was relaxed; no "
                "KC text modified post-hoc. Class-level standing rule: skip "
                "heavy training when preconditions are blocked."
            ),
            "standing_rules_invoked": [
                "precondition-probe-before-macro-rerun",
                "adapter-registry-not-artefact",
                "downstream-inherits-upstream-audit",
                "code-bug-tag-decoy-on-mathematical-failure",
            ],
            "unblock_path": (
                "1. Rerun exp_p1_t2_single_domain_training at LORA_SCALE=5 "
                "to regenerate adapters/math/adapters.safetensors. "
                "2. Stage a persona-tagged corpus disjoint from GSM8K "
                "(e.g. persona-prefixed math queries). "
                "3. Re-run this probe; if all PASS, implement the "
                "output_scale·head(z) training loop and measure K1572."
            ),
        }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0 if all_preconditions_pass else 1


if __name__ == "__main__":
    sys.exit(main())
