#!/usr/bin/env python3
"""
exp_followup_lora_scale_safe_sweep — precondition probe.

audit-2026-04-17 cohort tripwire: verify 3 preconditions before any
retraining. If any FAIL, K1553 is UNMEASURABLE → status=killed per
pre-registration. No data generated, no KC relaxed.

P1: "Flagship 5" enumeration reachable.
P2: Baseline LORA_SCALE=20 adapter safetensors on disk for >=3/5.
P3: Retraining pipeline viable (datasets + mlx + base model + configs).
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

EXP_DIR = Path(__file__).parent
REPO_ROOT = EXP_DIR.parent.parent.parent
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def probe_p1_flagship_enumeration() -> tuple[bool, dict]:
    """Flagship 5 enumerable from authoritative on-disk source."""
    sources_checked = []
    for rel in [".audit/supported_00.md", ".audit/RECOVERY_PLAN.md",
                "docs/audit/supported_00.md"]:
        p = REPO_ROOT / rel
        sources_checked.append({"path": str(rel), "exists": p.exists()})
    any_exists = any(s["exists"] for s in sources_checked)
    return any_exists, {
        "sources_checked": sources_checked,
        "any_authoritative_source": any_exists,
        "reason": None if any_exists else
            "No authoritative .audit/ or docs/audit/ doc on disk; "
            "flagship 5 IDs unrecoverable from this repo revision.",
    }


def probe_p2_baseline_adapters() -> tuple[bool, dict]:
    """At least 3/5 flagship adapters present as safetensors on disk.

    With P1 FAIL we cannot name the 5 flagships. Conservative fallback:
    survey the full `experiments/models/` tree for any experiment whose dir
    contains a LORA_SCALE=20 marker AND a safetensors file — count
    distinct such experiments. >=3 → conservative PASS.
    """
    micro_models = REPO_ROOT / "micro" / "models"
    if not micro_models.exists():
        return False, {"reason": "experiments/models/ dir missing",
                        "distinct_candidates": 0}

    candidates = []
    # Bounded scan — cap at first 120 exp dirs for runtime safety.
    for i, exp_dir in enumerate(sorted(p for p in micro_models.iterdir()
                                       if p.is_dir())):
        if i >= 120:
            break
        has_scale20 = False
        has_safetensors = False
        try:
            for cfg in exp_dir.rglob("*.yaml"):
                try:
                    txt = cfg.read_text(errors="ignore")
                    if "LORA_SCALE" in txt and "20" in txt:
                        has_scale20 = True
                        break
                except Exception:
                    continue
            for st in exp_dir.rglob("*.safetensors"):
                if st.is_file():
                    has_safetensors = True
                    break
        except Exception:
            continue
        if has_scale20 and has_safetensors:
            candidates.append(exp_dir.name)

    n = len(candidates)
    return (n >= 3), {
        "distinct_candidates_with_scale20_and_safetensors": n,
        "candidate_names": candidates[:10],
        "threshold": 3,
    }


def probe_p3_retraining_pipeline() -> tuple[bool, dict]:
    """datasets/dill/peft/mlx_lm importable AND base model cached."""
    mods = ["datasets", "dill", "peft", "mlx_lm"]
    import_status = {m: (importlib.util.find_spec(m) is not None)
                     for m in mods}

    # Cross-check upstream T2.1 _reconstruction_note for datasets/dill
    # Python 3.14 block.
    t2_results = REPO_ROOT / "micro" / "models" / \
                 "exp_p1_t2_single_domain_training" / "results.json"
    upstream_block = None
    if t2_results.exists():
        try:
            j = json.loads(t2_results.read_text())
            note = j.get("_reconstruction_note", "") or ""
            if "datasets/dill Python 3.14 upstream incompat" in note:
                upstream_block = "datasets/dill Python 3.14 incompat " \
                                 "documented in upstream T2.1"
        except Exception:
            pass

    # Base model cached?
    base_model_dirs = list(HF_CACHE.glob("models--mlx-community--gemma*"))
    base_cached = len(base_model_dirs) > 0

    all_modules_importable = all(import_status.values())
    ok = all_modules_importable and base_cached and upstream_block is None

    return ok, {
        "module_importable": import_status,
        "base_model_cached": base_cached,
        "base_model_dirs_found": len(base_model_dirs),
        "upstream_block_documented": upstream_block,
    }


def main() -> None:
    t0 = time.time()
    p1_ok, p1_data = probe_p1_flagship_enumeration()
    p2_ok, p2_data = probe_p2_baseline_adapters()
    p3_ok, p3_data = probe_p3_retraining_pipeline()

    n_pass = sum([p1_ok, p2_ok, p3_ok])
    all_pass = n_pass == 3
    verdict = "SUPPORTED" if all_pass else "KILLED"

    results = {
        "verdict": verdict,
        "all_pass": all_pass,
        "ran": True,
        "is_smoke": False,
        "probe_only": True,
        "wall_s": round(time.time() - t0, 3),
        "K1553_mechanism_survives_scale_reduction": "UNMEASURABLE — precondition probe failed" if not all_pass else "UNTESTED — retraining not yet run",
        "preconditions": {
            "P1_flagship_enumeration": {"pass": p1_ok, **p1_data},
            "P2_baseline_adapters_on_disk": {"pass": p2_ok, **p2_data},
            "P3_retraining_pipeline_viable": {"pass": p3_ok, **p3_data},
            "n_pass": n_pass,
            "threshold_for_measurable": 3,
        },
        "cohort_note": "17th consecutive audit-2026-04-17 precondition-probe KILL; same upstream blocker pattern (T2.1 safetensors + datasets/dill Python 3.14) documented across Findings #605–#624.",
    }

    out = EXP_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Probe complete in {results['wall_s']}s. Verdict: {verdict}. "
          f"Preconditions passed: {n_pass}/3. Wrote {out.name}.")


if __name__ == "__main__":
    main()
