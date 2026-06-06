"""Pierre (Gemma 4 E4B + 5 adapters) vs Qwen 3 4B — precondition probe.

This macro experiment has three preconditions (see MATH.md §4). When any
precondition fails, we DO NOT run partial benchmarks — partial results would
conflate "not measured" with "measured and weak", which is the class of
metric-swap / silent-downgrade antipattern the audit has been cleaning up.

Probe behaviour:
  1. Check all 5 Pierre adapter `.safetensors` files exist on disk.
  2. Check lm-eval-harness is importable under current platform Python.
  3. Check upstream T2.1 status and scan PAPER.md for metric-swap flag.

Emit `results.json` with explicit precondition state and verdict=KILLED
when any precondition fails. If all three pass, transition to the real
5-benchmark sweep (stub at bottom).

No MLX / no heavy load at probe time — this is a pure filesystem + import
probe, safe to run on any env. Intentional: the heavy comparison belongs
behind the preconditions, not in front of them.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
REGISTRY = REPO_ROOT / "data" / "adapters" / "registry.json"
T2_1_DIR = REPO_ROOT / "micro" / "models" / "exp_p1_t2_single_domain_training"

PIERRE_DOMAINS = ["math", "code", "medical", "sql", "bash"]


def check_p1_adapters() -> dict:
    """P1: every Pierre adapter has weights on disk."""
    adapter_root = REPO_ROOT / "data" / "adapters"
    missing, present = [], []
    for d in PIERRE_DOMAINS:
        domain_dir = adapter_root / d
        weights = list(domain_dir.glob("*.safetensors")) if domain_dir.exists() else []
        if weights:
            present.append({"domain": d, "files": [p.name for p in weights]})
        else:
            missing.append(d)
    return {
        "precondition": "P1_adapter_weights_on_disk",
        "passed": len(missing) == 0,
        "present": present,
        "missing_domains": missing,
        "registry_path": str(REGISTRY.relative_to(REPO_ROOT)),
    }


def check_p2_harness() -> dict:
    """P2: lm-eval-harness importable (via `uv run`)."""
    try:
        out = subprocess.run(
            ["uv", "run", "python", "-c", "import lm_eval; print(lm_eval.__version__)"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        ok = out.returncode == 0
        return {
            "precondition": "P2_harness_importable",
            "passed": ok,
            "stdout": out.stdout.strip()[:200],
            "stderr": out.stderr.strip()[:400],
        }
    except Exception as e:
        return {"precondition": "P2_harness_importable", "passed": False, "error": str(e)}


def check_p3_upstream() -> dict:
    """P3: T2.1 verdict=supported, no metric-swap flag."""
    results_path = T2_1_DIR / "results.json"
    paper_path = T2_1_DIR / "PAPER.md"
    if not results_path.exists():
        return {"precondition": "P3_upstream_valid", "passed": False, "reason": "T2.1 results.json missing"}
    try:
        data = json.loads(results_path.read_text())
    except Exception as e:
        return {"precondition": "P3_upstream_valid", "passed": False, "reason": f"T2.1 results.json unparseable: {e}"}
    verdict = str(data.get("verdict", "")).upper()
    paper = paper_path.read_text() if paper_path.exists() else ""
    paper_l = paper.lower()
    metric_swap = ("metric-swap" in paper_l) or ("medqa" in paper_l and "medmcqa" in paper_l)
    return {
        "precondition": "P3_upstream_valid",
        "passed": verdict == "SUPPORTED" and not metric_swap,
        "upstream_verdict": verdict,
        "metric_swap_flagged": metric_swap,
    }


def main() -> int:
    p1 = check_p1_adapters()
    p2 = check_p2_harness()
    p3 = check_p3_upstream()
    preconditions = [p1, p2, p3]
    all_ok = all(p["passed"] for p in preconditions)

    # KC results — blocked-by-prereq ⇒ fail against positive requirement.
    # K1694: Pierre wins >=3/5 benchmarks. Cannot PASS without measurement.
    k_1694_pass = False
    # K1695: thinking-mode + matched harness. Requires P1 AND P2. P3 is about
    # benchmark-identity validity, not the thinking/matched-config leg itself.
    k_1695_pass = bool(p1["passed"] and p2["passed"])

    if all_ok:
        verdict = "TODO_RUN_FULL"
        evidence = "All preconditions pass; proceed to 5-benchmark sweep (not implemented at probe stage)."
    else:
        verdict = "KILLED"
        blockers = [p["precondition"] for p in preconditions if not p["passed"]]
        evidence = f"Blocked by preconditions: {', '.join(blockers)}"

    out = {
        "experiment": "exp_model_peer_comparison_qwen3_4b",
        "verdict": verdict,
        "all_pass": all_ok and k_1694_pass and k_1695_pass,
        "is_smoke": False,
        "ran": True,
        "preconditions": preconditions,
        "kill_criteria": {
            "K1694_pierre_>=_qwen3_4b_on_3_of_5": k_1694_pass,
            "K1695_thinking_mode_and_matched_harness": k_1695_pass,
        },
        "evidence": evidence,
    }
    (EXP_DIR / "results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0  # probe always exits 0; verdict communicates blocker


if __name__ == "__main__":
    sys.exit(main())
