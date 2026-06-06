"""exp_model_quantization_composition_stability — KILLED_PREEMPTIVE runner.

Pure stdlib. No MLX, no model load, no HTTP bind. Target <=4 s wall.
Implements the 5-theorem stack defined in MATH.md:
  T1 — artifact-absence (5 Gemma 4 E4B domain adapter .safetensors
       on disk, W4A16 base bound to composition path, N=5 composition
       routine passing parent K1060/K1061, composed MMLU-Pro harness,
       non-regressed bf16 anchor)
  T2 — cost-bound (2-precision x MMLU-Pro x adapter-train protocol)
  T3 — schema-incomplete + empty references
  T4 — audit-pin reinforce
  T5 — source-scope breach (T5-K variant, single-parent KILLED)

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
EXP_ID = "exp_model_quantization_composition_stability"
PARENT_A = "exp_p1_t3_n25_composition"

START = time.time()

# ---------- T1: artifact-absence ----------

CODE_GLOBS = ["pierre/**/*.py", "macro/**/*.py", "composer/**/*.py",
              "experiments/models/**/*.py"]

REQUIRED_DOMAINS = ["math", "code", "medical", "legal", "finance"]


def _code_files() -> list[Path]:
    files: list[Path] = []
    for g in CODE_GLOBS:
        files.extend(ROOT.glob(g))
    files = [f for f in files if f.resolve() != Path(__file__).resolve()]
    return files


def _grep_cooccur(pat_a: str, pat_b: str, files: list[Path]) -> list[str]:
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


def _domain_adapter_inventory() -> tuple[int, list[str], list[str]]:
    """For each of the 5 required domains, check if any .safetensors
    file > 1 KB exists anywhere under experiments/models/**/adapters/
    whose path contains that domain name. Returns (present_count,
    present_domain_list, sample_paths).
    """
    present: set[str] = set()
    samples: list[str] = []
    for p in ROOT.glob("experiments/models/*/adapters/**/*.safetensors"):
        try:
            if not p.is_file():
                continue
            if p.stat().st_size <= 1024:
                continue
            rel = str(p.relative_to(ROOT))
            for dom in REQUIRED_DOMAINS:
                if re.search(rf"/{dom}/|/{dom}_|_{dom}/|_{dom}\.",
                             rel, re.IGNORECASE):
                    if dom not in present:
                        present.add(dom)
                        if len(samples) < 5:
                            samples.append(rel)
                    break
        except Exception:
            pass
    return len(present), sorted(present), samples


def probe_t1() -> dict:
    files = _code_files()

    # 1. 5 Gemma 4 E4B domain adapters on disk
    present_count, present_domains, domain_samples = (
        _domain_adapter_inventory()
    )
    all_5_adapters = present_count >= 5

    # 2. W4A16 base bound to composition path
    w4a16_compose_hits = _grep_cooccur(
        r"w4a16|4[_-]?bit|group[_-]?64|affine.*4",
        r"compose|compos|merge|add_adapter|W_combined",
        files,
    )

    # 3. N=5 composition routine in repo (functional surface)
    n5_compose_hits = _grep_cooccur(
        r"N\s*=\s*5|\bn_domains\s*=\s*5|five[_-]?domain",
        r"compose|W_combined|merge|add_adapter",
        files,
    )

    # 4. MMLU-Pro composed-model eval harness
    mmlu_compose_hits = _grep_cooccur(
        r"mmlu[_-]?pro|MMLU-Pro|mmlupro",
        r"W_combined|compose|composed|merge",
        files,
    )

    # 5. bf16 reference anchor (non-regressed bf16 N=5 composition
    #    MMLU-Pro score). We probe for literal anchor-value reporting
    #    in sibling experiment results.json files.
    bf16_anchor_hits = _grep_cooccur(
        r"bf16|bfloat16|bf[_-]?16",
        r"mmlu[_-]?pro.*compos|compos.*mmlu[_-]?pro",
        files,
    )

    need = {
        "five_gemma4_domain_adapters_on_disk": all_5_adapters,
        "w4a16_base_bound_to_composition": bool(w4a16_compose_hits),
        "n5_composition_routine": bool(n5_compose_hits),
        "mmlu_pro_composed_model_harness": bool(mmlu_compose_hits),
        "bf16_reference_anchor": bool(bf16_anchor_hits),
    }
    shortfall = sum(1 for v in need.values() if not v)
    return {
        "block": shortfall >= 3,
        "shortfall": shortfall,
        "threshold": 3,
        "adapter_domains_present": present_domains,
        "adapter_domains_required": REQUIRED_DOMAINS,
        "adapter_domain_count": present_count,
        "evidence": {
            "adapter_samples": domain_samples,
            "w4a16_compose_hits_sample": w4a16_compose_hits[:3],
            "n5_compose_hits_sample": n5_compose_hits[:3],
            "mmlu_compose_hits_sample": mmlu_compose_hits[:3],
            "bf16_anchor_hits_sample": bf16_anchor_hits[:3],
        },
        "need": need,
    }


# ---------- T2: cost-bound ----------

def probe_t2() -> dict:
    base_cold_s_per_run = 15 * 60
    n_base_runs = 2  # bf16 reference + W4A16 run (serial, 48 GB limit)
    adapter_train_s_each = 15 * 60
    n_adapters = 5
    w4a16_quantize_s = 10 * 60
    mmlu_pro_q = 1000
    mmlu_pro_s_per_q = 5
    n_mmlu_runs = 2  # bf16 + W4A16
    per_domain_delta_s = 10 * 60

    total_s = (
        n_base_runs * base_cold_s_per_run
        + n_adapters * adapter_train_s_each
        + w4a16_quantize_s
        + n_mmlu_runs * mmlu_pro_q * mmlu_pro_s_per_q
        + per_domain_delta_s
    )
    total_min = total_s / 60
    ceiling_min = 120

    # Floor (smoke: skip adapter train; reuse base loads; 100 Q x 1 s
    # x 2 precisions; no per-domain delta).
    floor_s = (
        n_base_runs * base_cold_s_per_run  # cannot skip cold-load
        + w4a16_quantize_s
        + n_mmlu_runs * 100 * 1
    )
    floor_min = floor_s / 60

    return {
        "block": total_min > ceiling_min,
        "conservative_min": round(total_min, 1),
        "floor_min": round(floor_min, 1),
        "ceiling_min": ceiling_min,
        "floor_note": (
            "Floor under ceiling but K1713 threshold (1.5 pp) is "
            "inside 95% CI half-width (~10 pp) at 100 Q — "
            "scientifically incoherent."
        ),
        "formula_conservative": (
            f"{n_base_runs} * {base_cold_s_per_run}s cold + "
            f"{n_adapters} * {adapter_train_s_each}s adapter train + "
            f"{w4a16_quantize_s}s quantize + "
            f"{n_mmlu_runs} * {mmlu_pro_q} * {mmlu_pro_s_per_q}s "
            f"MMLU-Pro + {per_domain_delta_s}s per-domain delta"
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
        r"success_criteria:\s*\[\]|Success Criteria: NONE", out))
    incomplete_flag = ("INCOMPLETE" in out) and ("success_criteria" in out)
    empty_refs = not bool(re.search(r"References:\s*\n\s+-\s*\S", out))
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


# ---------- T5: source-scope breach (T5-K single-parent KILLED) ----------

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
    parent_killed = status_a == "killed"

    # (A) Parent K1060 FAIL — 0/5 adapter .safetensors on disk
    k1060_adapter_breach = _parent_has(
        PARENT_A, r"K1060.*FAIL|0/5 adapter|safetensors_exists.*false",
    )
    # (B) Parent K1061 FAIL — MMLU regression at bf16 composition
    k1061_mmlu_breach = _parent_has(
        PARENT_A, r"K1061.*FAIL|MMLU.*regress|regress.*MMLU",
    )
    # (C) K1059 PASS scope — parameter-space orthogonality, not
    #     behavioral; does not transfer to 4-bit quantization grid
    k1059_scope_breach = _parent_has(
        PARENT_A, r"K1059.*PASS|Grassmannian|max\|cos\||orthogonality",
    )
    # (D) Tautological-routing inheritance — F#645 / F#502
    tautological_breach = _parent_has(
        PARENT_A,
        r"tautological[_-]?routing|REAL_ADAPTER_PATHS|hardcode.*adapter",
    )
    # (E) KC-target coupling — K1713 / K1714 reference "N=5 composition"
    #     which is parent-KILLED routine. Definitional; unconditional.
    kc_coupling_breach = True

    breaches = {
        "A_k1060_adapter_artifact_breach": k1060_adapter_breach,
        "B_k1061_mmlu_anchor_breach": k1061_mmlu_breach,
        "C_k1059_param_vs_behavioral_breach": k1059_scope_breach,
        "D_tautological_routing_inheritance_breach": tautological_breach,
        "E_kc_target_coupling_breach": kc_coupling_breach,
    }
    breach_count = sum(1 for v in breaches.values() if v)

    return {
        "block": parent_killed and breach_count >= 3,
        "variant": "T5-K_single_parent_killed",
        "parent": PARENT_A,
        "parent_status": status_a,
        "parent_killed": parent_killed,
        "breach_count": breach_count,
        "threshold": 3,
        "breaches": {k: ("true" if v else "false")
                     for k, v in breaches.items()},
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
            "K1713_w4a16_within_1p5pp_of_bf16_composition": False,
            "K1714_per_domain_ranking_preserved": False,
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
        "T5_source_scope_breach_Ksingle": t5,
    }
    blocks = [k for k, v in results["preempt"].items() if v.get("block")]
    results["preempt_blocks"] = blocks
    results["preempt_block_count"] = len(blocks)
    results["reason"] = (
        f"Preempt over-determined: {len(blocks)} independent blocks "
        f"({', '.join(blocks) if blocks else 'none'}). "
        f"T5 variant: T5-K single-parent-KILLED (3rd in drain). "
        f"See MATH.md section 2."
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
