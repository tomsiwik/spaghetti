#!/usr/bin/env python3
"""exp_followup_sft_behavioral_lora_scale_5

Replicate Finding #403 at Gemma 4 scale: B_applied = B_sft + output_scale * head(z)
with personalization data (NOT SFT replay). KC K1565 routes through a
precondition probe per MATH.md (class-level standing rule; 5th instance).

Preconditions (pre-registered in MATH.md):
  P1 T2.1 math adapter .safetensors on disk   (required for B_sft)
  P2 personalization data corpus              (required for data-distribution separation)
  P3 T2.1 upstream verdict != KILLED          (required for B_sft validity)

If any of P1/P2/P3 fails, K1565 is FAIL (unmeasurable) and verdict = KILLED.
Heavy training is skipped — this avoids a 30+ minute model-load wall on a
run that cannot produce measurable evidence. Same honest-fail routing as
exp_p1_t2_sft_residual_gemma4 V2, peer_comparison_llama31_8b,
peer_comparison_qwen3_4b, mtbench_composed, sequential_activation_compose_real.
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
            "B_sft (.safetensors) missing; adapter_config.json stub present but "
            "MLX LoRA weights were gitignored and never committed"
            if not exists else "B_sft present"
        ),
    }


def probe_p2_personalization_data() -> dict:
    """P2: personalization corpus (disjoint from SFT data).

    KC MATH.md requires data separation from GSM8K. No personalization corpus
    is staged in this repo for Gemma 4 scale; the SFT train.jsonl is the only
    on-disk data relative to T2.1. Status = FAIL (missing corpus, required by
    Theorem C data-separation clause).
    """
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
        EXPERIMENT_DIR.parent / "exp_followup_ss_rn_path_valid_sft" / "data",
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
            "No personalization corpus staged. Theorem C (data separation) "
            "is a required precondition: SFT replay is the known failure "
            "mode (parent QR=0.707). Cannot substitute GSM8K for persona data "
            "without reclassifying the experiment."
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
            "T2.1 KILLED 2026-04-18 via metric-swap (MedMCQA vs MedQA) + "
            "format-artefact (max_tokens=256 truncating Gemma 4 CoT before "
            "'#### answer'). B_sft validity inherits the KILL."
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
                "All preconditions PASS — heavy training would be launched "
                "here. This branch is not yet implemented because preconditions "
                "have never been met since 2026-04-17 (parent adapter missing). "
                "Mark PROVISIONAL with a TODO to implement the Finding #403 "
                "replication training loop when P1/P2/P3 are unblocked."
            ),
            "kcs": {
                "K1565": {
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
                "K1565": {
                    "status": "FAIL",
                    "reason": (
                        "unmeasurable: Finding #403 replication requires "
                        "B_sft (P1) + personalization data (P2) + valid "
                        "upstream (P3); missing -> "
                        f"{', '.join(failing)}"
                    ),
                    "threshold": "QR >= 0.90",
                    "measured": None,
                },
            },
            "note": (
                "Precondition-probe KILL (5th instance this loop). Pre-registered "
                "routing in MATH.md: K1565 FAIL unmeasurable when any of "
                "P1/P2/P3 fail. No KC threshold was relaxed; no KC text was "
                "modified after data collection. Class-level standing rule "
                "applies: skip heavy training when preconditions are blocked."
            ),
            "standing_rules_invoked": [
                "precondition-probe-before-macro-rerun",
                "adapter-registry-not-artefact",
                "downstream-inherits-upstream-audit",
                "code-bug-tag-decoy-on-mathematical-failure",
            ],
            "unblock_path": (
                "1. Rerun exp_p1_t2_single_domain_training at LORA_SCALE=5 "
                "to regenerate adapters.safetensors. "
                "2. Stage a personalization corpus disjoint from GSM8K "
                "(e.g. persona-tagged math queries). "
                "3. Re-run this probe; if all PASS, implement the training "
                "loop and measure K1565."
            ),
        }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0 if all_preconditions_pass else 1


if __name__ == "__main__":
    sys.exit(main())
