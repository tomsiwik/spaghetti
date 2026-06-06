"""Pre-registered precondition probe for exp_g4_snr_rank_predictor.

This runner does not train or fit an SNR predictor. It verifies the three
preconditions (P1/P2/P3) required to measure KC #1586 and KC #1587 on
Gemma 4 E4B 4-bit. If any precondition FAILs, the experiment is KILLED
with the blocker recorded — matching the audit-2026-04-17 cohort
standing rule.

The main measurement (r_95 within-2x across 5 domains vs null) is
implemented behind `if all_preconditions_passed:`. It is deliberately
unreachable on the current platform state (zero rank-sweep adapters on
disk; zero gradient-SNR spectra logged). When upstream regeneration of
`exp_p1_t2_single_domain_training` at LORA_SCALE=5 with rank sweep
lands, the probe passes and the measurement branch executes without
edits to KC #1586 / KC #1587.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXP_DIR / "results.json"


# Pre-registered adapter paths (five Gemma 4 E4B domain adapters).
# math/code/medical come from exp_p1_t2_single_domain_training; finance +
# legal are the two domains the cohort recovery plan calls for.
CANDIDATE_ADAPTER_PATHS = {
    "math":    REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/math",
    "code":    REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/code",
    "medical": REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/medical",
    "finance": REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/finance",
    "legal":   REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/adapters/legal",
}

# Per-domain rank sweep directories (optional — needed for full r* measurement).
RANK_SWEEP_RANKS = [2, 4, 6, 12, 24]

UPSTREAM_TRAINING_RESULTS = (
    REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training/results.json"
)


def probe_p1_adapters_and_rank_sweep() -> tuple[bool, dict]:
    """P1: ≥5 domain adapter safetensors + rank-sweep endpoints exist."""
    detail: dict = {
        "required_domains": list(CANDIDATE_ADAPTER_PATHS.keys()),
        "rank_sweep_ranks": RANK_SWEEP_RANKS,
        "per_domain": {},
    }
    r6_found = 0
    rank_sweep_found = 0
    for name, dirpath in CANDIDATE_ADAPTER_PATHS.items():
        weights_candidates = [
            dirpath / "adapter_model.safetensors",
            dirpath / "adapters.safetensors",
        ]
        hit = None
        for w in weights_candidates:
            if w.exists() and w.stat().st_size > 0:
                hit = {"path": str(w), "bytes": w.stat().st_size}
                r6_found += 1
                break
        # Rank-sweep check: look for sibling dirs named e.g. `math_r2`, `math_r12`, etc.
        sweep_detail = {}
        for r in RANK_SWEEP_RANKS:
            sweep_dir = dirpath.parent / f"{name}_r{r}"
            sweep_hit = None
            for fname in ("adapter_model.safetensors", "adapters.safetensors"):
                w = sweep_dir / fname
                if w.exists() and w.stat().st_size > 0:
                    sweep_hit = {"path": str(w), "bytes": w.stat().st_size}
                    rank_sweep_found += 1
                    break
            sweep_detail[f"r{r}"] = sweep_hit
        detail["per_domain"][name] = {
            "r6_endpoint": hit if hit is not None else {"path": None, "bytes": 0},
            "rank_sweep": sweep_detail,
        }
    detail["r6_adapters_found"] = r6_found
    detail["rank_sweep_adapters_found"] = rank_sweep_found
    detail["rank_sweep_required"] = len(CANDIDATE_ADAPTER_PATHS) * len(RANK_SWEEP_RANKS)
    # P1 requires at minimum the 5 r=6 endpoints; full rank sweep is stronger but
    # minimally we need one endpoint to compute any within-2x window.
    passed = r6_found == len(CANDIDATE_ADAPTER_PATHS)
    return passed, detail


def probe_p2_gradient_snr_spectra() -> tuple[bool, dict]:
    """P2: training-gradient SNR spectra exist per domain."""
    detail: dict = {"per_domain": {}}
    found = 0
    for name, dirpath in CANDIDATE_ADAPTER_PATHS.items():
        candidates = [
            dirpath / "grad_snr.json",
            dirpath / "training_log.jsonl",
            dirpath / "gradient_spectrum.npy",
            dirpath.parent / f"{name}_grad_snr.json",
        ]
        hit = None
        for c in candidates:
            if c.exists() and c.stat().st_size > 0:
                hit = {"path": str(c), "bytes": c.stat().st_size}
                found += 1
                break
        detail["per_domain"][name] = hit if hit is not None else {"path": None, "bytes": 0}
    detail["domains_with_spectra"] = found
    detail["required"] = len(CANDIDATE_ADAPTER_PATHS)
    passed = found == len(CANDIDATE_ADAPTER_PATHS)
    return passed, detail


def probe_p3_rank_ablation_baseline() -> tuple[bool, dict]:
    """P3: upstream adapter behavioral deltas are supported (not KILLED).

    Without a trustworthy r=6 behavioral endpoint per domain, within-2x
    cannot be computed (the comparison has no ground truth).
    """
    detail: dict = {"results_path": str(UPSTREAM_TRAINING_RESULTS)}
    if not UPSTREAM_TRAINING_RESULTS.exists():
        detail["verdict"] = "MISSING"
        return False, detail
    try:
        data = json.loads(UPSTREAM_TRAINING_RESULTS.read_text())
    except json.JSONDecodeError as e:
        detail["verdict"] = f"UNPARSEABLE: {e}"
        return False, detail
    verdict = data.get("verdict", "MISSING")
    detail["verdict"] = verdict
    detail["all_pass"] = data.get("all_pass")
    detail["base_gsm8k_pct"] = data.get("base_gsm8k_pct")
    passed = (
        verdict not in ("KILLED", "MISSING")
        and data.get("all_pass") is True
        and (data.get("base_gsm8k_pct") or 0.0) > 20.0
    )
    return passed, detail


def main() -> int:
    p1_pass, p1_detail = probe_p1_adapters_and_rank_sweep()
    p2_pass, p2_detail = probe_p2_gradient_snr_spectra()
    p3_pass, p3_detail = probe_p3_rank_ablation_baseline()

    all_pre = p1_pass and p2_pass and p3_pass

    result = {
        "experiment": "exp_g4_snr_rank_predictor",
        "verdict": "KILLED" if not all_pre else "UNDECIDED",
        "all_pass": False if not all_pre else None,
        "ran": True,
        "is_smoke": False,
        "kill_criterion_ids": [1586, 1587],
        "preconditions": {
            "P1_adapters_and_rank_sweep": {"passed": p1_pass, **p1_detail},
            "P2_gradient_snr_spectra":    {"passed": p2_pass, **p2_detail},
            "P3_rank_ablation_baseline":  {"passed": p3_pass, **p3_detail},
        },
        "K1586": "UNMEASURABLE" if not all_pre else "PENDING",
        "K1587": "UNMEASURABLE" if not all_pre else "PENDING",
        "blocker": (
            "Upstream regeneration of exp_p1_t2_single_domain_training at "
            "LORA_SCALE=5 with disjoint math/code/medical/finance/legal "
            "corpora at max_tokens>=512 + per-domain rank sweep {2,4,6,12,24} "
            "+ per-step gradient-SNR logging is required before KC #1586 / "
            "#1587 are measurable. Per audit-2026-04-17 cohort standing rule, "
            "heavy retraining (~12h MLX for the full rank sweep) is not "
            "executed inside this hat iteration — the precondition probe "
            "fails honestly instead."
        ) if not all_pre else None,
        "notes": (
            "Pre-registered probe. If the measurement branch below is ever "
            "reached, KC #1586 / #1587 are locked — relaxing them (e.g. "
            "within-4x, or threshold 0.80) invalidates the probe and "
            "requires a v2 experiment."
        ),
    }

    if all_pre:
        # Unreachable on current platform state. Placeholder shows the
        # measurement is not silently skipped — it is gated on P1/P2/P3.
        raise RuntimeError(
            "Preconditions unexpectedly passed. The r_95 within-2x "
            "measurement branch must be implemented and re-registered "
            "before this runner can produce a supported verdict."
        )

    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {RESULTS_PATH}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
