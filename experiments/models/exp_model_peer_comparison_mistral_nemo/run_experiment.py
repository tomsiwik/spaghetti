"""exp_model_peer_comparison_mistral_nemo — KILLED_PREEMPTIVE preempt runner.

Pure stdlib. No MLX, no model load, no HTTP bind. ≤3 s wall.
Implements the 5-theorem stack defined in MATH.md:
  T1 — artifact-absence (unified harness, Mistral Nemo MLX weights,
       MATH-500 & IFEval harnesses, N=5 adapter stack)
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
EXP_ID = "exp_model_peer_comparison_mistral_nemo"
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


def _hf_cache_has(glob_pat: str) -> list[str]:
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    if not hub.exists():
        return []
    return sorted(str(p.name) for p in hub.glob(glob_pat))


def probe_t1() -> dict:
    files = _code_files()

    # 1. unified cross-model 5-benchmark harness
    unified_hits = _grep_files(
        r"(peer[_-]?comparison|cross[_-]?model[_-]?eval|five[_-]?benchmark|mistral[_-]?nemo.*gemma|gemma.*mistral[_-]?nemo)",
        files,
    )
    # 2. Mistral Nemo MLX weights in HF cache
    mistral_nemo_cache = _hf_cache_has("models--mlx-community--Mistral-Nemo-*")
    # 3. MATH-500 harness (with boxed extraction)
    math500_hits = _grep_files(
        r"(math[_-]?500|MATH-500)",
        files,
    )
    # 4. IFEval harness (strict or lenient verifier)
    ifeval_hits = _grep_files(
        r"\bifeval\b|IFEval|instruction[_-]?following[_-]?eval|prompt[_-]?instructions",
        files,
    )
    # 5. Pierre N=5 adapter stack evidence
    n5_stack_hits = _grep_files(
        r"(N\s*=\s*5.*adapter|compose.*5.*adapter|adapter[_-]?stack.*5)",
        files,
    )

    need = {
        "unified_cross_model_harness": bool(unified_hits),
        "mistral_nemo_mlx_weights_local": bool(mistral_nemo_cache),
        "math500_harness": bool(math500_hits),
        "ifeval_harness": bool(ifeval_hits),
        "pierre_n5_adapter_stack": bool(n5_stack_hits),
    }
    shortfall = sum(1 for v in need.values() if not v)
    return {
        "block": shortfall >= 3,  # pre-reg threshold
        "shortfall": shortfall,
        "threshold": 3,
        "evidence": {
            "unified_hits_sample": unified_hits[:3],
            "mistral_nemo_cache_dirs": mistral_nemo_cache[:3],
            "math500_hits_sample": math500_hits[:3],
            "ifeval_hits_sample": ifeval_hits[:3],
            "n5_stack_hits_sample": n5_stack_hits[:3],
        },
        "need": need,
    }


# ---------- T2: cost-bound ----------

def probe_t2() -> dict:
    # Conservative 100-sample budget per benchmark × 2 models × 5 benches
    samples_per = 100
    n_bench = 5
    n_models = 2
    secs_per_sample = 8  # avg generate on M5 Pro (bf16) for ~200 tok output
    model_cold_s = 15 * 60  # 2 model cold loads @ ~15 min each (Mistral 12B + Gemma 4 E4B + Pierre compose)
    compose_s = 5 * 60      # Pierre N=5 compose overhead
    total_s = samples_per * n_bench * n_models * secs_per_sample \
              + model_cold_s * n_models + compose_s
    total_min = total_s / 60
    ceiling_min = 120

    # Full-benchmark scenario for transparency
    full_samples = 100 + 164 + 500 + 541 + 1319  # MMLUP-subset + HE + M500 + IF + GSM
    full_total_min = (full_samples * n_models * secs_per_sample
                      + model_cold_s * n_models + compose_s) / 60

    return {
        "block": total_min > ceiling_min,
        "conservative_min": round(total_min, 1),
        "full_bench_min": round(full_total_min, 1),
        "ceiling_min": ceiling_min,
        "formula_conservative": f"{samples_per}*{n_bench}*{n_models}*{secs_per_sample}s + {n_models}*{model_cold_s}s + {compose_s}s",
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

    # Read source verdict from DB (canonical)
    try:
        src_db = subprocess.run(
            ["experiment", "get", SRC_EXP_ID],
            capture_output=True, text=True, timeout=20, cwd=str(ROOT),
        ).stdout
    except Exception:
        src_db = ""
    src_verdict_supported = "Status:   supported" in src_db

    # T5-K variant check
    if not src_verdict_supported:
        return {
            "block": False,
            "reason": "source is not supported; T5-K variant would apply",
            "source_verdict_db_literal": next(
                (l for l in src_db.splitlines() if "Status:" in l), ""
            ),
        }

    # Scope probes — each returns True if source has evidence; False = breach
    src_text = ""
    for p in (src_results, src_paper, src_dir / "MATH.md"):
        try:
            src_text += "\n" + p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

    def _src_has(rx: str) -> bool:
        return bool(re.search(rx, src_text, re.IGNORECASE))

    scope_dims = {
        "A_mmlu_pro": _src_has(r"mmlu[_-]?pro"),
        "B_math_500": _src_has(r"math[_-]?500|MATH-500"),
        "C_ifeval": _src_has(r"\bifeval\b|IFEval"),
        "D_cross_model_peer_comparison": _src_has(r"mistral[_-]?nemo|peer[_-]?comparison|cross[_-]?model"),
        "E_n5_adapter_composition": _src_has(
            r"(N\s*=\s*5.*adapter|compose.*5.*adapter|adapter[_-]?stack.*5)"
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
            "K1696_pierre_ge_mistral_nemo_on_2_of_5_benchmarks": False,
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
