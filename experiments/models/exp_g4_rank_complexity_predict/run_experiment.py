"""Pre-registered precondition probe for exp_g4_rank_complexity_predict.

KC #1629 (Spearman rho >= 0.85 between domain complexity c(D) and
optimal rank r*) and its paired target KC #1629-T (behavioral gap at
predicted rank <= 2.0pp vs r=12 oracle) are locked in MATH.md and can
only be measured if three upstream artifacts exist on disk:

  P1: 25 rank-sweep adapter safetensors (5 domains x 5 ranks)
  P2: five-domain training corpora (.jsonl) available for c(D)
  P3: upstream training `exp_p1_t2_single_domain_training` is supported
      with a plausible, non-format-artifact baseline (`base_gsm8k_pct > 20`).

If any precondition FAILs, the experiment is KILLED on UNMEASURABLE per
the audit-2026-04-17 cohort standing rule — heavy retraining (~12h MLX
for the 25-adapter sweep) is out of scope for a single researcher
iteration. This runner does not fit the predictor; the measurement
branch is deliberately gated on `all_preconditions_passed` and is
unreachable on the current platform state.

Structurally identical to `exp_g4_snr_rank_predictor/run_experiment.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXP_DIR / "results.json"

UPSTREAM_DIR = REPO_ROOT / "experiments/models/exp_p1_t2_single_domain_training"
UPSTREAM_RESULTS = UPSTREAM_DIR / "results.json"

DOMAINS = ["math", "code", "medical", "finance", "legal"]
RANK_SWEEP = [2, 4, 6, 8, 12]  # from MATH.md


def probe_p1_rank_sweep_adapters() -> tuple[bool, dict]:
    """P1: for each domain x rank, a non-zero safetensors exists."""
    detail: dict = {
        "required_domains": DOMAINS,
        "rank_sweep": RANK_SWEEP,
        "per_domain": {},
    }
    total_found = 0
    total_required = len(DOMAINS) * len(RANK_SWEEP)
    for d in DOMAINS:
        sweep_detail: dict = {}
        for r in RANK_SWEEP:
            candidate_dirs = [
                UPSTREAM_DIR / "adapters" / f"{d}_r{r}",
            ]
            # The r=6 central point is allowed to live at adapters/{d}/
            if r == 6:
                candidate_dirs.append(UPSTREAM_DIR / "adapters" / d)
            hit = None
            for sd in candidate_dirs:
                for fname in ("adapter_model.safetensors", "adapters.safetensors"):
                    w = sd / fname
                    if w.exists() and w.stat().st_size > 0:
                        hit = {"path": str(w), "bytes": w.stat().st_size}
                        total_found += 1
                        break
                if hit is not None:
                    break
            sweep_detail[f"r{r}"] = hit
        detail["per_domain"][d] = sweep_detail
    detail["adapters_found"] = total_found
    detail["adapters_required"] = total_required
    passed = total_found == total_required
    return passed, detail


def probe_p2_corpora_for_complexity() -> tuple[bool, dict]:
    """P2: per-domain train.jsonl with >= 1k rows on disk."""
    detail: dict = {"per_domain": {}, "min_rows": 1000}
    found = 0
    for d in DOMAINS:
        path = UPSTREAM_DIR / "data" / d / "train.jsonl"
        row_count = 0
        exists = path.exists() and path.stat().st_size > 0
        if exists:
            # Cheap line count; we do not parse each row.
            with path.open("rb") as f:
                for row_count, _ in enumerate(f, start=1):
                    if row_count > 1000:
                        break
        ok = exists and row_count >= 1000
        if ok:
            found += 1
        detail["per_domain"][d] = {
            "path": str(path),
            "exists": exists,
            "rows_seen": row_count,
            "ok": ok,
        }
    detail["domains_with_corpus"] = found
    detail["required"] = len(DOMAINS)
    passed = found == len(DOMAINS)
    return passed, detail


def probe_p3_rank_ablation_baseline() -> tuple[bool, dict]:
    """P3: upstream training is supported with a plausible baseline."""
    detail: dict = {"results_path": str(UPSTREAM_RESULTS)}
    if not UPSTREAM_RESULTS.exists():
        detail["verdict"] = "MISSING"
        return False, detail
    try:
        data = json.loads(UPSTREAM_RESULTS.read_text())
    except json.JSONDecodeError as e:
        detail["verdict"] = f"UNPARSEABLE: {e}"
        return False, detail
    verdict = data.get("verdict", "MISSING")
    detail["verdict"] = verdict
    detail["all_pass"] = data.get("all_pass")
    detail["base_gsm8k_pct"] = data.get("base_gsm8k_pct")
    passed = (
        verdict in ("supported", "proven", "provisional")
        and data.get("all_pass") is True
        and (data.get("base_gsm8k_pct") or 0.0) > 20.0
    )
    return passed, detail


def main() -> int:
    p1_pass, p1_detail = probe_p1_rank_sweep_adapters()
    p2_pass, p2_detail = probe_p2_corpora_for_complexity()
    p3_pass, p3_detail = probe_p3_rank_ablation_baseline()

    all_pre = p1_pass and p2_pass and p3_pass

    result = {
        "experiment": "exp_g4_rank_complexity_predict",
        "verdict": "KILLED" if not all_pre else "UNDECIDED",
        "all_pass": False if not all_pre else None,
        "ran": True,
        "is_smoke": False,
        "kill_criterion_ids": [1629],
        "preconditions": {
            "P1_rank_sweep_adapters": {"passed": p1_pass, **p1_detail},
            "P2_corpora_for_complexity": {"passed": p2_pass, **p2_detail},
            "P3_rank_ablation_baseline": {"passed": p3_pass, **p3_detail},
        },
        "K1629": "UNMEASURABLE" if not all_pre else "PENDING",
        "K1629_T": "UNMEASURABLE" if not all_pre else "PENDING",
        "blocker": (
            "Rank-sweep retraining at LORA_SCALE=5 + max_tokens>=512 + "
            "thinking mode preserved, across {math, code, medical, finance, "
            "legal} x r in {2,4,6,8,12} (25 adapters, ~12h MLX), is required "
            "before KC #1629 / #1629-T are measurable. Upstream "
            "exp_p1_t2_single_domain_training currently supplies only r=6 "
            "adapters for math/code/medical (3/25). Per audit-2026-04-17 "
            "cohort standing rule, heavy retraining is out of scope for a "
            "single researcher iteration — the precondition probe fails "
            "honestly instead."
        ) if not all_pre else None,
        "notes": (
            "Pre-registered probe. KC #1629 and KC #1629-T are locked; "
            "relaxing either (within-4x, threshold 0.80, fewer domains) "
            "invalidates the probe and requires a v2 experiment. "
            "Sibling exp_g4_snr_rank_predictor LEARNINGS warned Ralph "
            "against claiming a 10th cohort-downstream probe; this "
            "precondition-KILL result serves as reinforcement of that "
            "standing-down rule and routes to upstream rebuild as the fix."
        ),
    }

    if all_pre:
        # Unreachable on current platform state. KC #1629 / #1629-T
        # measurements must be implemented before the probe can pass.
        raise RuntimeError(
            "Preconditions unexpectedly passed. The rank-complexity "
            "correlation + target-gap measurement branch must be "
            "implemented and re-registered before this runner can produce "
            "a supported verdict."
        )

    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {RESULTS_PATH}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
