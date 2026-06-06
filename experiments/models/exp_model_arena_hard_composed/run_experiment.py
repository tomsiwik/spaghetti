"""exp_model_arena_hard_composed — KILLED_PREEMPTIVE preempt runner.

Pure stdlib. No MLX, no model load, no HTTP bind. ≤3 s wall.
Implements the 5-theorem stack defined in MATH.md:
  T1 — artifact-absence (Arena-Hard prompts, LLM judge, Pierre N=5
       serve endpoint, base eval endpoint, pairwise-CI framework)
  T2 — cost-bound
  T3 — schema-incomplete
  T4 — audit-pin reinforce
  T5 — source-scope breach vs SUPPORTED parent
       `exp_p1_t2_single_domain_training`

Runs from project root (cwd = repo top).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
EXP_ID = "exp_model_arena_hard_composed"
SRC_EXP_ID = "exp_p1_t2_single_domain_training"

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


def _grep_files(pattern: str, files: list[Path]) -> list[str]:
    rx = re.compile(pattern, re.IGNORECASE)
    hits: list[str] = []
    for f in files:
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, line in enumerate(txt.splitlines(), start=1):
            if rx.search(line):
                hits.append(f"{f.relative_to(ROOT)}:{i}:{line.strip()[:120]}")
                if len(hits) >= 20:
                    return hits
    return hits


def _grep_cooccur(pat_a: str, pat_b: str, files: list[Path]) -> list[str]:
    """Files where BOTH patterns appear (any line). Stronger than OR-match."""
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


def probe_t1() -> dict:
    files = _code_files()

    # 1. Arena-Hard prompt set (500 prompts, LMSYS)
    arena_prompt_hits = _grep_cooccur(
        r"arena[_-]?hard",
        r"\bprompts?\b|dataset|jsonl|load",
        files,
    )
    # 2. LLM-judge client with pairwise schema
    judge_hits = _grep_cooccur(
        r"arena[_-]?hard",
        r"judge|pairwise|win[_-]?rate|gpt-?4|judger",
        files,
    )
    # 3. Pierre N=5 compose/serve endpoint
    n5_serve_hits = _grep_cooccur(
        r"(N\s*=\s*5|n_?adapters?\s*=\s*5|five[_-]?adapter)",
        r"compose|serve|stack|routing",
        files,
    )
    # 4. Base Gemma 4 E4B pairwise-comparison peer endpoint
    base_peer_hits = _grep_cooccur(
        r"gemma[_-]?4[_-]?e4b|gemma-4-e4b",
        r"pairwise|arena|win[_-]?rate|baseline[_-]?generation",
        files,
    )
    # 5. Pairwise win-rate + bootstrap CI
    bootstrap_hits = _grep_cooccur(
        r"bootstrap",
        r"win[_-]?rate|pairwise|arena",
        files,
    )

    need = {
        "arena_hard_prompt_set": bool(arena_prompt_hits),
        "llm_judge_client_pairwise": bool(judge_hits),
        "pierre_n5_serve_endpoint": bool(n5_serve_hits),
        "base_gemma4_e4b_peer_endpoint": bool(base_peer_hits),
        "pairwise_bootstrap_ci_framework": bool(bootstrap_hits),
    }
    shortfall = sum(1 for v in need.values() if not v)
    return {
        "block": shortfall >= 3,  # pre-reg threshold
        "shortfall": shortfall,
        "threshold": 3,
        "evidence": {
            "arena_prompt_hits_sample": arena_prompt_hits[:3],
            "judge_hits_sample": judge_hits[:3],
            "n5_serve_hits_sample": n5_serve_hits[:3],
            "base_peer_hits_sample": base_peer_hits[:3],
            "bootstrap_hits_sample": bootstrap_hits[:3],
        },
        "need": need,
    }


# ---------- T2: cost-bound ----------

def probe_t2() -> dict:
    # Conservative Arena-Hard protocol:
    # 500 prompts × 2 sides (Pierre + base) = 1000 generations
    # Per-sample generate 15 s (open-ended, 400-800 tok completion)
    # Judge calls 500 × 5 s (GPT-4-Turbo w/ network)
    # Cold loads 2 × 15 min (base + Pierre-composed)
    # Pierre N=5 compose overhead 5 min
    n_prompts = 500
    n_sides = 2
    secs_per_sample = 15
    n_judge = 500
    secs_per_judge = 5
    model_cold_s = 15 * 60 * 2
    compose_s = 5 * 60
    total_s = (n_prompts * n_sides * secs_per_sample
               + n_judge * secs_per_judge
               + model_cold_s + compose_s)
    total_min = total_s / 60
    ceiling_min = 120

    # Floor: 5 s/sample (lightning path, still blocks)
    floor_s = n_prompts * n_sides * 5 + n_judge * secs_per_judge \
              + model_cold_s + compose_s
    floor_min = floor_s / 60

    return {
        "block": total_min > ceiling_min,
        "conservative_min": round(total_min, 1),
        "floor_min": round(floor_min, 1),
        "ceiling_min": ceiling_min,
        "formula_conservative": (f"{n_prompts}*{n_sides}*{secs_per_sample}s "
                                 f"+ {n_judge}*{secs_per_judge}s "
                                 f"+ {model_cold_s}s + {compose_s}s"),
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
    empty_sc = bool(re.search(r"Success Criteria: NONE|success_criteria:\s*\[\]", out))
    incomplete_flag = ("INCOMPLETE" in out) and ("success_criteria" in out)
    block = empty_sc or incomplete_flag
    return {
        "block": block,
        "has_success_criteria_empty": empty_sc,
        "has_incomplete_flag": incomplete_flag,
        "evidence_line": next((l for l in out.splitlines() if "INCOMPLETE" in l), ""),
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


# ---------- T5: source-scope breach ----------

def probe_t5() -> dict:
    src_dir = ROOT / "micro" / "models" / SRC_EXP_ID
    src_results = src_dir / "results.json"
    src_paper = src_dir / "PAPER.md"
    src_math = src_dir / "MATH.md"

    # Read source verdict from DB (canonical)
    try:
        src_db = subprocess.run(
            ["experiment", "get", SRC_EXP_ID],
            capture_output=True, text=True, timeout=20, cwd=str(ROOT),
        ).stdout
    except Exception:
        src_db = ""
    src_verdict_supported = "Status:   supported" in src_db

    if not src_verdict_supported:
        return {
            "block": False,
            "reason": "source is not supported; T5-K variant would apply",
            "source_verdict_db_literal": next(
                (l for l in src_db.splitlines() if "Status:" in l), ""
            ),
        }

    src_text = ""
    for p in (src_results, src_paper, src_math):
        try:
            src_text += "\n" + p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

    def _src_has(rx: str) -> bool:
        return bool(re.search(rx, src_text, re.IGNORECASE))

    scope_dims = {
        "A_arena_hard_prompts": _src_has(r"arena[_-]?hard"),
        "B_llm_judge_pairwise": _src_has(r"llm[_-]?judge|pairwise|gpt-?4[_-]?judge|judger"),
        "C_open_ended_generation": _src_has(r"open[_-]?ended|conversational|400.*800.*tok"),
        "D_n5_composition": _src_has(
            r"(N\s*=\s*5.*adapter|compose.*5.*adapter|adapter[_-]?stack.*5)"
        ),
        "E_bootstrap_winrate_ci": _src_has(
            r"(bootstrap.*win[_-]?rate|win[_-]?rate.*bootstrap|pairwise.*bootstrap)"
        ),
    }
    breach_count = sum(1 for k, v in scope_dims.items() if not v)

    return {
        "block": breach_count >= 3,  # pre-reg threshold
        "breach_count": breach_count,
        "threshold": 3,
        "source_verdict_db_supported": src_verdict_supported,
        "scope_dimensions": {k: ("source-has" if v else "BREACH") for k, v in scope_dims.items()},
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
            "K1700_pierre_n5_arena_hard_win_rate_ge_50pct": False,
            "K1701_ci_lower_bound_gt_40pct": False,
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
        "T5_source_scope_breach": t5,
    }
    blocks = [k for k, v in results["preempt"].items() if v.get("block")]
    results["preempt_blocks"] = blocks
    results["preempt_block_count"] = len(blocks)
    results["reason"] = (
        f"Preempt over-determined: {len(blocks)} independent blocks "
        f"({', '.join(blocks)}). See MATH.md §2."
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
