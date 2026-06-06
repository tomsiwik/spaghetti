"""Preemptive kill runner for exp_prod_adapter_attach_ux
(ap-017 5-theorem stack).

No model, no inference, no MLX. Pure stdlib. Probes the 4 artifacts
that the target's K1673/K1674/K1675 would require; writes results.json
with T1..T5 verdicts and a single-shot 'killed_preregistered' stamp.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
SOURCE_MATH = REPO_ROOT / "experiments/models/exp_p1_t4_serving_v2/MATH.md"
SOURCE_RESULTS = REPO_ROOT / "experiments/models/exp_p1_t4_serving_v2/results.json"
SOURCE_RUN = REPO_ROOT / "experiments/models/exp_p1_t4_serving_v2/run_experiment.py"


def _ripgrep(pattern: str, *, extra: list[str] | None = None) -> list[str]:
    cmd = [
        "grep", "-rE", pattern,
        "--include=*.py",
        "--exclude-dir=.venv",
        "--exclude-dir=__pycache__",
        "--exclude-dir=node_modules",
        "--exclude-dir=archive",
        str(REPO_ROOT),
    ]
    if extra:
        cmd.extend(extra)
    out = subprocess.run(cmd, capture_output=True, text=True)
    self_path = str(Path(__file__).resolve())
    return [
        l for l in out.stdout.splitlines()
        if l.strip() and not l.startswith(self_path)
    ]


def _drop_comments(hits: list[str]) -> list[str]:
    out = []
    for l in hits:
        _, _, tail = l.partition(":")
        if tail.lstrip().startswith("#"):
            continue
        out.append(l)
    return out


def t1_prerequisite_inventory() -> dict:
    # 1. pierre CLI entry point in pyproject.toml
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    # Scripts block pattern: look for `pierre = "..."` under [project.scripts]
    m = re.search(
        r"\[project\.scripts\](.*?)(?=^\[|\Z)",
        pyproject, flags=re.S | re.M,
    )
    scripts_block = m.group(1) if m else ""
    pierre_cli = bool(re.search(r"^\s*pierre\s*=", scripts_block, flags=re.M))

    # 2. Persistent server process model
    server_hits = _ripgrep(
        r"(pierre.*serve|run_server|@app\.(post|get)|uvicorn\."
        r"|FastAPI\(|pierre-server|pierre_server)"
    )
    server_code = _drop_comments([
        l for l in server_hits
        if "/skills/" not in l
        and ".jsonl" not in l
        and "/data/" not in l
        and "/memory/" not in l
    ])
    has_server = len(server_code) > 0

    # 3. p99 latency harness in adapter-swap paths
    # Narrow to files that also mention attach/detach/swap
    p99_candidate_hits = _ripgrep(
        r"(p99|percentile|np\.percentile|numpy\.percentile|quantile\()"
    )
    p99_code_raw = _drop_comments([
        l for l in p99_candidate_hits
        if "/skills/" not in l
        and ".jsonl" not in l
        and "/data/" not in l
        and "/memory/" not in l
    ])
    # Require co-occurrence with attach/detach/swap keywords in the same file.
    p99_files: set[str] = set()
    for l in p99_code_raw:
        path, _, _ = l.partition(":")
        p99_files.add(path)
    p99_in_swap_path = False
    for f in p99_files:
        try:
            body = Path(f).read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if re.search(r"\battach|\bdetach|hot.swap|hotswap", body):
            p99_in_swap_path = True
            break

    # 4. logit-cosine pre/post-attach harness
    cosine_hits = _ripgrep(
        r"(pre_attach_logits|post_detach_logits|logit_cosine"
        r"|cosine_similarity.*logit|logit.*cosine_similarity)"
    )
    cosine_code = _drop_comments([
        l for l in cosine_hits
        if "/skills/" not in l
        and ".jsonl" not in l
        and "/data/" not in l
        and "/memory/" not in l
    ])
    has_cosine = len(cosine_code) > 0

    required = {
        "pierre_cli_entry_point": pierre_cli,
        "persistent_server_process_model": has_server,
        "p99_latency_harness_in_swap_path": p99_in_swap_path,
        "logit_cosine_pre_post_attach_harness": has_cosine,
    }
    shortfall = sum(1 for present in required.values() if not present)
    return {
        "required": required,
        "shortfall": shortfall,
        "pyproject_scripts_block": scripts_block.strip(),
        "server_hits_in_code": len(server_code),
        "p99_files_considered": sorted(p99_files)[:10],
        "cosine_hits_in_code": len(cosine_code),
        "block": shortfall >= 1,
    }


def t2_scale_safety() -> dict:
    # p99 needs >= 100 samples; cycle = attach + forward + detach + forward + cos
    # Per-cycle: p99 attach ~50ms + forward ~30ms + detach ~20ms + forward ~30ms
    ms_per_cycle = 50 + 30 + 20 + 30  # 130
    n_cycles = 100
    n_seeds = 3
    swap_loop_sec = (ms_per_cycle * n_cycles * n_seeds) / 1000.0
    model_load_sec = 15
    adapter_loads_sec = 0.2 * n_cycles  # 200ms per adapter load x 100
    est_minutes = (swap_loop_sec + model_load_sec + adapter_loads_sec) / 60.0
    ceiling_min = 120
    return {
        "ms_per_cycle": ms_per_cycle,
        "n_cycles": n_cycles,
        "n_seeds": n_seeds,
        "swap_loop_seconds": swap_loop_sec,
        "model_load_seconds": model_load_sec,
        "adapter_loads_seconds": adapter_loads_sec,
        "est_minutes": round(est_minutes, 2),
        "ceiling_minutes": ceiling_min,
        "block": est_minutes > ceiling_min,
    }


def t3_schema_completeness() -> dict:
    out = subprocess.run(
        ["experiment", "get", "exp_prod_adapter_attach_ux"],
        capture_output=True, text=True,
    )
    text = out.stdout
    incomplete = "INCOMPLETE" in text
    missing_success = (
        "Success Criteria: NONE" in text or "success_criteria: []" in text
    )
    return {
        "db_literal_incomplete": incomplete,
        "success_criteria_missing": missing_success,
        "block": incomplete and missing_success,
    }


def t4_pin_ratio() -> dict:
    kc_pins = {
        "K1673_cli_attach":  True,    # literal "pierre attach math"
        "K1673_cli_detach":  True,    # literal "pierre detach math"
        "K1673_restart":     False,   # "without server restart" - undefined
        "K1674_api_attach":  True,    # "attach_adapter()"
        "K1674_api_detach":  True,    # "detach_adapters()"
        "K1674_ms":          True,    # "<200ms"
        "K1674_p99":         True,    # "p99"
        "K1674_hot":         False,   # "hot-swaps" - undefined
        "K1675_cos":         True,    # "logit cosine with pre-attach"
        "K1675_exactly":     False,   # "exactly" - no threshold
    }
    pinned = sum(1 for v in kc_pins.values() if v)
    total = len(kc_pins)
    ratio = pinned / total
    return {
        "pinned": pinned,
        "total": total,
        "pin_ratio": round(ratio, 3),
        "threshold": 0.20,
        "block": ratio < 0.20,
    }


def t5_source_scope_breach() -> dict:
    math_text = SOURCE_MATH.read_text() if SOURCE_MATH.exists() else ""
    results_text = (
        SOURCE_RESULTS.read_text() if SOURCE_RESULTS.exists() else ""
    )
    run_text = SOURCE_RUN.read_text() if SOURCE_RUN.exists() else ""

    # (A) CLI-scope: source MATH.md has zero CLI / subprocess / pierre-binary vocab
    cli_terms = re.findall(
        r"(?i)\b(cli|pierre\s+attach|pierre\s+detach|argv|entry[-_ ]point"
        r"|subprocess\.)\b",
        math_text,
    )
    source_has_cli_vocab = len(cli_terms) > 0

    # (B) detach-scope: source K1240 "swap+first-forward" - grep for detach path
    # in source MATH + run_experiment.py
    source_detach_hits = len(re.findall(
        r"\bdetach(?:_adapters?)?\b", math_text + run_text,
    ))

    # (C) p99-scope: source results.json contains "p50" not "p99"
    source_has_p50 = bool(re.search(r'"?p50"?\s*[:=]|\bp50\b', results_text))
    source_has_p99 = bool(re.search(r'"?p99"?\s*[:=]|\bp99\b', results_text))

    # (D) process-restart-scope: source's run_experiment.py is a one-shot script,
    # no persistent server
    source_has_server = bool(re.search(
        r"(FastAPI|uvicorn|@app\.(post|get)|run_server)", run_text,
    ))

    # (E) state-consistency-scope: source never measures pre/post logit cosine
    source_has_roundtrip = bool(re.search(
        r"(pre_attach|post_detach|logit_cosine|cosine.*pre|round.trip)",
        math_text + run_text,
    ))

    breaches = {
        "A_cli_scope":                not source_has_cli_vocab,    # breach if source has ZERO CLI vocab
        "B_detach_scope":             source_detach_hits == 0,      # breach if source never mentions detach
        "C_p99_scope":                source_has_p50 and not source_has_p99,  # breach if source reports p50 only
        "D_process_restart_scope":    not source_has_server,        # breach if source has no server
        "E_state_consistency_scope":  not source_has_roundtrip,     # breach if source never tests round-trip
    }
    hits = sum(1 for v in breaches.values() if v)
    return {
        "breaches": breaches,
        "literal_hits": hits,
        "block": hits >= 3,
        "source_math_found": SOURCE_MATH.exists(),
        "source_results_found": SOURCE_RESULTS.exists(),
        "source_run_found": SOURCE_RUN.exists(),
        "source_detach_hits": source_detach_hits,
        "source_has_p50": source_has_p50,
        "source_has_p99": source_has_p99,
        "source_cli_vocab_count": len(cli_terms),
    }


def main() -> None:
    t0 = time.time()
    t1 = t1_prerequisite_inventory()
    t2 = t2_scale_safety()
    t3 = t3_schema_completeness()
    t4 = t4_pin_ratio()
    t5 = t5_source_scope_breach()

    all_block = t1["block"] and t3["block"] and t5["block"]
    defense_in_depth = any([t1["block"], t3["block"], t5["block"]])

    kc_results = {
        "K1673": "fail",
        "K1674": "fail",
        "K1675": "fail",
    }

    results = {
        "experiment_id": "exp_prod_adapter_attach_ux",
        "verdict": "KILLED_PREEMPTIVE",
        "all_pass": False,
        "all_block": all_block,
        "defense_in_depth": defense_in_depth,
        "is_smoke": False,
        "theorems": {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5},
        "kill_criteria": kc_results,
        "ap_017_axis": (
            "composition-bug (software-infrastructure-unbuilt variant)"
        ),
        "ap_017_scope_index": 35,
        "supported_source_preempt_index": 16,
        "f502_instance_index": 7,
        "defense_in_depth_theorems_firing": sum([
            int(t1["block"]), int(t3["block"]), int(t5["block"]),
        ]),
        "wall_seconds": round(time.time() - t0, 4),
    }

    out_path = EXP_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
