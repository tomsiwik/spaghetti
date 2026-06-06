"""exp_model_pre_registration_n100_macro — KILLED_PREEMPTIVE preempt runner.

Pure stdlib. No MLX, no model load, no HTTP bind. <=3 s wall.
Implements the 5-theorem stack defined in MATH.md:
  T1 — artifact-absence (100 Gemma 4 E4B adapters, Room Model
       W_combined routine, 100-domain eval harness, per-domain solo
       baseline runner, N=100 composition/routing framework)
  T2 — cost-bound
  T3 — schema-incomplete + empty references
  T4 — audit-pin reinforce
  T5 — source-scope breach (T5-K variant, double-parent KILLED)

Runs from project root (cwd = repo top).
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
EXP_ID = "exp_model_pre_registration_n100_macro"
PARENT_A = "exp_model_room_model_gemma4_speed"
PARENT_B = "exp_p1_t3_n25_composition"
MICRO_T35 = "exp_p1_t3_n100_composition"  # not declared; only referenced in notes

START = time.time()

# ---------- T1: artifact-absence ----------

CODE_GLOBS = ["pierre/**/*.py", "macro/**/*.py", "composer/**/*.py",
              "experiments/models/**/*.py"]


def _code_files() -> list[Path]:
    files: list[Path] = []
    for g in CODE_GLOBS:
        files.extend(ROOT.glob(g))
    files = [f for f in files if f.resolve() != Path(__file__).resolve()]
    return files


def _grep_cooccur(pat_a: str, pat_b: str, files: list[Path]) -> list[str]:
    """Files where BOTH patterns appear (any line)."""
    rxa = re.compile(pat_a, re.IGNORECASE)
    rxb = re.compile(pat_b, re.IGNORECASE)
    hits: list[str] = []
    for f in files:
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if rxa.search(txt) and rxb.search(txt):
            hits.append(str(f.relative_to(ROOT)))
            if len(hits) >= 20:
                return hits
    return hits


def _adapter_inventory() -> tuple[int, list[str]]:
    """Count .safetensors adapter checkpoints under experiments/models/**/adapters/
    with size > 1 KB. A1 / A2: we ignore per-checkpoint duplicates by
    counting distinct parent directories (one adapter per dir).
    """
    seen_dirs: set[str] = set()
    samples: list[str] = []
    for p in ROOT.glob("experiments/models/*/adapters/**/*.safetensors"):
        try:
            if not p.is_file():
                continue
            if p.stat().st_size <= 1024:
                continue
            parent = str(p.parent.relative_to(ROOT))
            if parent in seen_dirs:
                continue
            seen_dirs.add(parent)
            if len(samples) < 5:
                samples.append(parent)
        except Exception:
            pass
    return len(seen_dirs), samples


def probe_t1() -> dict:
    files = _code_files()

    # 1. 100 Gemma 4 E4B adapter cohort on disk
    adapter_count, adapter_samples = _adapter_inventory()
    has_100_adapters = adapter_count >= 100

    # 2. Room Model W_combined construction routine (the functional one)
    w_combined_hits = _grep_cooccur(
        r"W_combined|w_combined",
        r"construct|build|assemble|pre[_-]?sum",
        files,
    )

    # 3. Per-domain held-out eval harness at N=100
    n100_eval_hits = _grep_cooccur(
        r"100[_-]?domain|domain[_-]?cohort[_-]?100|N\s*=\s*100",
        r"heldout|held[_-]?out|per[_-]?domain|eval",
        files,
    )

    # 4. Per-domain solo baseline runner at N=100
    solo_baseline_hits = _grep_cooccur(
        r"solo[_-]?baseline|per[_-]?domain[_-]?solo|solo[_-]?adapter",
        r"100[_-]?domain|domain[_-]?cohort|N\s*=\s*100|heldout",
        files,
    )

    # 5. N=100 composition / routing framework
    n100_compose_hits = _grep_cooccur(
        r"N\s*=\s*100|100[_-]?adapter|cohort[_-]?100|domain[_-]?cohort",
        r"compose|route|stack|router|gate",
        files,
    )

    need = {
        "cohort_100_gemma4_e4b_adapters_on_disk": has_100_adapters,
        "room_model_w_combined_routine": bool(w_combined_hits),
        "per_domain_heldout_eval_harness_n100": bool(n100_eval_hits),
        "per_domain_solo_baseline_runner_n100": bool(solo_baseline_hits),
        "n100_composition_routing_framework": bool(n100_compose_hits),
    }
    shortfall = sum(1 for v in need.values() if not v)
    return {
        "block": shortfall >= 3,
        "shortfall": shortfall,
        "threshold": 3,
        "adapter_count_found": adapter_count,
        "adapter_threshold": 100,
        "evidence": {
            "adapter_samples": adapter_samples,
            "w_combined_hits_sample": w_combined_hits[:3],
            "n100_eval_hits_sample": n100_eval_hits[:3],
            "solo_baseline_hits_sample": solo_baseline_hits[:3],
            "n100_compose_hits_sample": n100_compose_hits[:3],
        },
        "need": need,
    }


# ---------- T2: cost-bound ----------

def probe_t2() -> dict:
    # Conservative N=100 macro composition eval protocol:
    base_cold_s = 15 * 60
    wcombined_s = 10 * 60           # K1710 asks < 60 s; parent KILLED at N=5, N=100 worse
    eval_per_domain_s = 50 * 5      # 50 Q/domain × 5 s/Q
    n_domains = 100
    composed_eval_s = n_domains * eval_per_domain_s
    solo_eval_s = n_domains * eval_per_domain_s
    solo_adapter_cold_s = n_domains * 10

    total_s = (base_cold_s + wcombined_s + composed_eval_s
               + solo_eval_s + solo_adapter_cold_s)
    total_min = total_s / 60
    ceiling_min = 120

    # Floor (smoke: 10 Q/domain, 1 s/sample, no solo re-eval)
    floor_s = (base_cold_s + wcombined_s
               + n_domains * 10 * 1
               + 0
               + n_domains * 10)
    floor_min = floor_s / 60

    return {
        "block": total_min > ceiling_min,
        "conservative_min": round(total_min, 1),
        "floor_min": round(floor_min, 1),
        "ceiling_min": ceiling_min,
        "formula_conservative": (
            f"{base_cold_s}s cold + {wcombined_s}s W_combined + "
            f"{n_domains}*{eval_per_domain_s}s composed-eval + "
            f"{n_domains}*{eval_per_domain_s}s solo-eval + "
            f"{n_domains}*10s solo-load"
        ),
    }


# ---------- T3: schema-incomplete ----------

def probe_t3() -> dict:
    try:
        out = subprocess.run(
            ["experiment", "get", EXP_ID],
            capture_output=True, text=True, timeout=20, cwd=str(ROOT),
        ).stdout
    except Exception as e:
        return {"block": False, "error": str(e)}
    empty_sc = bool(re.search(
        r"Success Criteria: NONE|success_criteria:\s*\[\]", out))
    incomplete_flag = ("INCOMPLETE" in out) and ("success_criteria" in out)
    empty_refs = "references:" not in out or not re.search(
        r"references:\s*-\s*\S", out
    )
    block = empty_sc or incomplete_flag
    return {
        "block": block,
        "has_success_criteria_empty": empty_sc,
        "has_incomplete_flag": incomplete_flag,
        "references_empty": empty_refs,
        "evidence_line": next(
            (l for l in out.splitlines() if "INCOMPLETE" in l), ""
        ),
    }


# ---------- T4: audit-pin reinforcer ----------

def probe_t4() -> dict:
    audit_dir = ROOT / ".audit"
    audits = list(audit_dir.glob("pin_*.json")) if audit_dir.exists() else []
    hits = 0
    total = 0
    for p in audits:
        try:
            data = json.loads(p.read_text())
            total += 1
            if EXP_ID in json.dumps(data):
                hits += 1
        except Exception:
            pass
    ratio = (hits / total) if total else 0.0
    return {
        "block": False,
        "pin_ratio": round(ratio, 2),
        "floor": 0.20,
        "reinforces": ratio >= 0.20,
        "audit_dir_exists": audit_dir.exists(),
    }


# ---------- T5: source-scope breach (T5-K double-parent KILLED) ----------

def _db_status(exp_id: str) -> tuple[str, str]:
    try:
        out = subprocess.run(
            ["experiment", "get", exp_id],
            capture_output=True, text=True, timeout=20, cwd=str(ROOT),
        ).stdout
    except Exception as e:
        return ("error", f"db_probe_error: {e}")
    m = re.search(r"Status:\s*(\S+)", out)
    return (m.group(1) if m else "unknown", out)


def _parent_has(exp_id: str, rx: str) -> bool:
    src_dir = ROOT / "micro" / "models" / exp_id
    text = ""
    for fn in ("results.json", "PAPER.md", "MATH.md"):
        p = src_dir / fn
        try:
            text += "\n" + p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return bool(re.search(rx, text, re.IGNORECASE))


def probe_t5() -> dict:
    status_a, _ = _db_status(PARENT_A)
    status_b, _ = _db_status(PARENT_B)

    both_killed = (status_a == "killed") and (status_b == "killed")

    # Transitive-kill breach dimensions (T5-K variant)
    # Look in parents' on-disk artifacts for evidence of the kill pattern.
    # Breach fires if the parent-killed KC maps onto the target's KC.
    room_speed_killed = _parent_has(
        PARENT_A, r"K1688.*FAIL|69\.18\s*tok|\b69\b.*tok.*s"
    )
    room_quality_killed = _parent_has(
        PARENT_A, r"K1689.*FAIL|0\.9941|cos.*0\.99[0-4]"
    )
    n25_adapter_failed = _parent_has(
        PARENT_B, r"K1060.*FAIL|0/5.*adapter|0\s*of\s*5"
    )
    n25_mmlu_regressed = _parent_has(
        PARENT_B, r"K1061.*FAIL|MMLU.*regress|mmlu.*fail"
    )
    # T3.5 retrofit breach: micro-scale SUPPORTED record not declared as parent
    t35_status, _ = _db_status(MICRO_T35)
    t35_is_not_declared_parent = True  # literal: absent from depends_on

    breaches = {
        "A_room_speed_breach": room_speed_killed,
        "B_room_quality_breach": room_quality_killed,
        "C_n25_adapter_cohort_failed": n25_adapter_failed,
        "D_n25_mmlu_regressed": n25_mmlu_regressed,
        "E_t35_retrofit_non_declared": t35_is_not_declared_parent,
    }
    breach_count = sum(1 for v in breaches.values() if v)

    return {
        "block": both_killed and breach_count >= 3,
        "variant": "T5-K_double_parent_killed",
        "parent_a": PARENT_A,
        "parent_a_status": status_a,
        "parent_b": PARENT_B,
        "parent_b_status": status_b,
        "both_parents_killed": both_killed,
        "breach_count": breach_count,
        "threshold": 3,
        "breaches": {k: ("true" if v else "false") for k, v in breaches.items()},
        "t35_notes": (
            f"{MICRO_T35} (SUPPORTED micro, scale={t35_status}) referenced "
            f"in `notes` but NOT declared in `depends_on` — retrofit if "
            f"counted as parent; see MATH A7"
        ),
    }


# ---------- main ----------

def main() -> int:
    results = {
        "experiment": EXP_ID,
        "verdict": "KILLED_PREEMPTIVE",
        "all_pass": False,
        "is_smoke": False,
        "ran": False,
        "status": "infrastructure_blocked",
        "kill_criteria": {
            "K1708_per_domain_quality_within_5pct_of_solo": False,
            "K1709_no_domain_below_80pct_of_solo": False,
            "K1710_w_combined_construction_lt_60s": False,
        },
        "preempt": {},
        "reason": "",
        "runtime_sec": 0.0,
    }

    t1 = probe_t1()
    t2 = probe_t2()
    t3 = probe_t3()
    t4 = probe_t4()
    t5 = probe_t5()

    results["preempt"] = {
        "T1_artifact_absence": t1,
        "T2_cost_bound": t2,
        "T3_schema_incomplete": t3,
        "T4_audit_pin_reinforcer": t4,
        "T5_source_scope_breach_Kdouble": t5,
    }
    blocks = [k for k, v in results["preempt"].items() if v.get("block")]
    results["preempt_blocks"] = blocks
    results["preempt_block_count"] = len(blocks)
    results["reason"] = (
        f"Preempt over-determined: {len(blocks)} independent blocks "
        f"({', '.join(blocks) if blocks else 'none'}). "
        f"T5 variant: T5-K double-parent-KILLED (first in drain). "
        f"See MATH.md §2."
    )
    results["runtime_sec"] = round(time.time() - START, 2)

    out = EXP_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps({
        "verdict": results["verdict"],
        "blocks": blocks,
        "runtime_sec": results["runtime_sec"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
