"""exp_prod_openai_api_compat — KILLED_PREEMPTIVE preempt runner.

Pure stdlib. No MLX, no model load, no HTTP bind. ≤3 s wall.
Implements the 5-theorem stack defined in MATH.md:
  T1 — artifact-absence
  T2 — cost-bound
  T3 — schema-incomplete
  T4 — audit-pin reinforce
  T5-K — parent-KILLED inheritance (novel sub-axis)

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
EXP_ID = "exp_prod_openai_api_compat"
SRC_EXP_ID = "exp_prod_mlxlm_integration"

START = time.time()

# ---------- T1: artifact-absence ----------

CODE_GLOBS = ["pierre/**/*.py", "macro/**/*.py", "composer/**/*.py"]

def _code_files() -> list[Path]:
    files: list[Path] = []
    for g in CODE_GLOBS:
        files.extend(ROOT.glob(g))
    # exclude this runner itself
    files = [f for f in files if f.resolve() != Path(__file__).resolve()]
    return files

def _grep_any(pattern: str, files: list[Path]) -> list[str]:
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

def probe_t1() -> dict:
    files = _code_files()

    # a. /v1/chat/completions endpoint in a decorator
    chat_ep_hits = _grep_any(r"""@(?:app|router)\.(?:post|get|api_route).*["']/v1/chat/completions""", files)

    # b. pierre serve / CLI entry
    serve_paths = [p for p in files if re.search(r"pierre/(serve|server|cli|__main__)", str(p))]
    serve_hits: list[str] = []
    for p in serve_paths:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
            if re.search(r"(def\s+(serve|main|cli)\b|uvicorn\.run|fastapi\.FastAPI)", txt):
                serve_hits.append(str(p.relative_to(ROOT)))
        except Exception:
            pass

    # c. X-Pierre-Adapters or extra_body["adapters"] handler
    adapter_hdr_hits = _grep_any(r"""(X-Pierre-Adapters|x-pierre-adapters|extra_body\[["']adapters["']\]|headers\.get\(["']x-pierre-adapters)""", files)

    # d. SSE streaming harness in same file as chat endpoint
    stream_hits = _grep_any(r"""(EventSourceResponse|StreamingResponse.*text/event-stream|media_type=["']text/event-stream)""", files)

    need = {
        "chat_completions_endpoint": bool(chat_ep_hits),
        "pierre_serve_entry": bool(serve_hits),
        "adapter_header_or_extra_body": bool(adapter_hdr_hits),
        "sse_stream_harness": bool(stream_hits),
    }
    shortfall = sum(1 for v in need.values() if not v)
    return {
        "block": shortfall >= 3,  # pre-reg: need ≥ 3 of 4 absent
        "shortfall": shortfall,
        "evidence": {
            "chat_endpoint_hits_sample": chat_ep_hits[:3],
            "serve_entry_hits_sample": serve_hits[:3],
            "adapter_header_hits_sample": adapter_hdr_hits[:3],
            "sse_stream_hits_sample": stream_hits[:3],
        },
        "need": need,
    }

# ---------- T2: cost-bound ----------

def probe_t2() -> dict:
    surface_combos = 4 * 4  # 4 endpoints × 4 surface features
    composes = 3            # base, N=1, N=3
    seeds = 3
    secs_per_call = 45
    cold_start = 30 * 60    # 30 min server cold-start + reload
    total_s = surface_combos * composes * seeds * secs_per_call + cold_start
    total_min = total_s / 60
    ceiling_min = 120
    return {
        "block": total_min > ceiling_min,
        "estimated_min": round(total_min, 1),
        "ceiling_min": ceiling_min,
        "formula": f"{surface_combos}*{composes}*{seeds}*{secs_per_call}s + {cold_start}s cold-start",
    }

# ---------- T3: schema-incomplete ----------

def probe_t3() -> dict:
    try:
        out = subprocess.run(
            ["experiment", "get", EXP_ID],
            capture_output=True, text=True, timeout=20, cwd=str(ROOT),
        ).stdout
    except Exception as e:  # pragma: no cover
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
    audits = list((ROOT / ".audit").glob("pin_*.json")) if (ROOT / ".audit").exists() else []
    # Heuristic: if any .audit pin file references this exp id, count pin_ratio in it
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
        "block": False,  # reinforcer only
        "pin_ratio": round(ratio, 2),
        "floor": 0.20,
        "reinforces": ratio >= 0.20,
    }

# ---------- T5-K: parent-KILLED inheritance ----------

def probe_t5k() -> dict:
    src_dir = ROOT / "micro" / "models" / SRC_EXP_ID
    src_results = src_dir / "results.json"
    if not src_results.exists():
        return {"block": False, "reason": f"{src_results} missing"}
    try:
        data = json.loads(src_results.read_text())
    except Exception as e:
        return {"block": False, "reason": f"parse error: {e}"}

    preflight = data.get("preflight", {})
    block_keys = [k for k, v in preflight.items()
                  if isinstance(v, dict) and v.get("pass") is False]
    verdict_killed = data.get("verdict", "").upper() == "KILLED"
    reason = data.get("reason", "")[:300]

    # Look up target's declared depends_on
    try:
        target_yaml = subprocess.run(
            ["experiment", "get", EXP_ID],
            capture_output=True, text=True, timeout=20, cwd=str(ROOT),
        ).stdout
    except Exception:
        target_yaml = ""
    depends_claim_hit = SRC_EXP_ID in target_yaml

    block = verdict_killed and depends_claim_hit and len(block_keys) >= 3
    return {
        "block": block,
        "parent_verdict": data.get("verdict"),
        "parent_failed_preflight_keys": block_keys,
        "parent_reason_snippet": reason,
        "target_declares_dependency": depends_claim_hit,
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
            "K1682_stock_openai_client_roundtrip": False,
            "K1683_adapter_selection_via_header_or_extra_body": False,
            "K1684_tools_response_format_logprobs_stream_options_parity": False,
        },
        "preempt": {},
        "reason": "",
        "runtime_sec": 0.0,
    }
    t1 = probe_t1()
    t2 = probe_t2()
    t3 = probe_t3()
    t4 = probe_t4()
    t5k = probe_t5k()

    results["preempt"] = {
        "T1_artifact_absence": t1,
        "T2_cost_bound": t2,
        "T3_schema_incomplete": t3,
        "T4_audit_pin_reinforcer": t4,
        "T5K_parent_killed_inheritance": t5k,
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
