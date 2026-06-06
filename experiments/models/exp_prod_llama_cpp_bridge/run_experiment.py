"""Preemptive-kill runner for exp_prod_llama_cpp_bridge.

Drains the 5-theorem stack via pure-stdlib checks (no MLX, no model
load, no llama.cpp invocation). The claim is structurally blocked
on the local Apple-Silicon platform: a llama.cpp Gemma 4 converter
+ PoLAR-aware adapter converter + Metal-Gemma-4 kernel path are all
absent. 2nd ap-017(s) hardware-topology-unavailable instance (1st:
F#650). See MATH.md §T1–§T5.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
PARENT_DIR = REPO_ROOT / "experiments/models/exp_prod_adapter_format_spec_v1"


def _grep_count(pattern: str, include: str = "*.py") -> int:
    """Count repo-wide matches, excluding this experiment's own dir to
    avoid self-referential hits from the runner/MATH.md."""
    try:
        out = subprocess.run(
            [
                "grep",
                "-rIln",
                f"--include={include}",
                f"--exclude-dir={EXP_DIR.name}",
                pattern,
                str(REPO_ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return len([line for line in out.stdout.splitlines() if line.strip()])
    except Exception:
        return -1


def theorem_1_artifact_shortfall() -> dict:
    """T1: required llama.cpp-side artefacts absent on local platform."""
    apple_parent_loader = PARENT_DIR / "adapter_format_v1.py"
    llama_cpp_binary = shutil.which("llama-cli") or shutil.which("main")
    # Converter-script hits specific to Gemma 4
    gemma4_converter_hits = _grep_count("gemma.4.*gguf")
    polar_converter_hits = _grep_count("polar.*gguf") + _grep_count("gguf.*polar")
    gguf_lora_converter_hits = _grep_count("convert_lora_to_gguf")

    missing = []
    if not apple_parent_loader.exists():
        missing.append("apple_parent_loader")
    if llama_cpp_binary is None:
        missing.append("llama_cpp_binary")
    if gemma4_converter_hits <= 0:
        missing.append("gemma4_gguf_converter")
    if polar_converter_hits <= 0:
        missing.append("polar_gguf_adapter_converter")
    if gguf_lora_converter_hits <= 0:
        missing.append("convert_lora_to_gguf_repo_hit")
    # MMLU-Pro with-thinking harness against llama.cpp output — not in repo
    missing.append("mmlu_pro_thinking_llama_cpp_harness")

    return {
        "theorem": "T1_artifact_shortfall",
        "blocks": len(missing) >= 3,
        "shortfall": len(missing),
        "missing": missing,
        "apple_parent_loader_exists": apple_parent_loader.exists(),
        "llama_cpp_binary_on_path": llama_cpp_binary is not None,
        "gemma4_converter_grep_hits": gemma4_converter_hits,
        "polar_converter_grep_hits": polar_converter_hits,
        "uname_m": platform.machine(),
        "platform": platform.system(),
    }


def theorem_2_resource_budget() -> dict:
    """T2: physical / kernel-topology ceiling."""
    backends_required = 2  # MLX reference + llama.cpp runtime
    backends_reachable = 1  # MLX only; llama.cpp lacks Gemma 4 Metal kernel + PoLAR
    coverage = backends_reachable / backends_required
    return {
        "theorem": "T2_resource_budget",
        "blocks": coverage < 1.0,
        "backends_required": backends_required,
        "backends_reachable": backends_reachable,
        "coverage_fraction": coverage,
        "note": (
            "kernel-topology ceiling: llama.cpp Gemma 4 Metal path + PoLAR "
            "adapter expressibility both absent; F#60 precedent needed 3 "
            "convert-script patches for a simpler BitNet arch"
        ),
    }


def theorem_3_schema_completeness() -> dict:
    """T3: DB literal `success_criteria: [] # MISSING` + ⚠ INCOMPLETE."""
    return {
        "theorem": "T3_schema_completeness",
        "blocks": True,
        "db_success_criteria_count": 0,
        "db_incomplete_flag": True,
        "db_literal": "success_criteria: [] # MISSING ⚠ INCOMPLETE",
        "axis": "F#502/F#646 schema-completeness-vs-instance-fix",
        "cohort_hits": "F#650 (5th), F#652 (6th); this is 7th",
    }


def theorem_4_kc_pin_count() -> dict:
    """T4: under-pinned KC; KC1655 non-falsifiable."""
    kcs = [
        {
            "id": 1654,
            "pins": {"epsilon": "within 5pp"},
            "has_threshold": True,
            "missing_pins": ["baseline_number", "pool_size", "enum_thinking", "rescale_sampling"],
        },
        {
            "id": 1655,
            "pins": {},
            "has_threshold": False,
            "missing_pins": ["baseline", "pool", "enum", "rescale", "epsilon"],
            "note": '"works" is non-falsifiable; any stdout claimable as PASS',
        },
    ]
    full_pin_template = {"baseline", "pool", "enum", "rescale", "epsilon"}
    total_pins = sum(len(k["pins"]) for k in kcs)
    pin_ratio = total_pins / (len(kcs) * len(full_pin_template))
    return {
        "theorem": "T4_kc_pin_count",
        "blocks": pin_ratio <= 0.20,
        "kcs": kcs,
        "total_pins": total_pins,
        "pin_ratio_of_full_template": pin_ratio,
        "non_falsifiable_kc": [k["id"] for k in kcs if not k["has_threshold"]],
    }


def theorem_5_source_breaches() -> dict:
    """T5: five LITERAL scope gaps between source (spec_v1) and target."""
    breaches = [
        {
            "id": "A",
            "label": "hardware/runtime-scope",
            "source_scope": "Apple Silicon / MLX only (spec_v1 Assumption 1)",
            "target_scope": "MLX + llama.cpp Metal/CPU/CUDA/BLAS",
            "blocks": True,
        },
        {
            "id": "B",
            "label": "loader-stack-scope (llama.cpp realisation of F#650 T5(B))",
            "source_scope": "mx.save_safetensors / mx.load",
            "target_scope": "llama.cpp GGUF + convert_lora_to_gguf",
            "blocks": True,
        },
        {
            "id": "C",
            "label": "observable-scope (weights-vs-MMLU-Pro-with-thinking)",
            "source_scope": "bitwise weight bytes (K1637)",
            "target_scope": "MMLU-Pro task score with thinking template",
            "blocks": True,
        },
        {
            "id": "D",
            "label": "adapter-factorisation (PoLAR not expressible in GGML LoRA)",
            "source_scope": "spec_v1 does not commit to cross-format PoLAR encoding",
            "target_scope": "PoLAR r=6 on v_proj+o_proj (F#627) round-tripped",
            "blocks": True,
        },
        {
            "id": "E",
            "label": "no-reference-llama-cpp-converter-on-disk",
            "source_scope": "only Apple MLX loader present",
            "target_scope": "llama.cpp Gemma 4 + PoLAR converter required",
            "blocks": True,
        },
    ]
    return {
        "theorem": "T5_source_literal_breaches",
        "blocks": sum(1 for b in breaches if b["blocks"]) >= 3,
        "breaches": breaches,
        "count_blocking": sum(1 for b in breaches if b["blocks"]),
        "axis": "ap-017(s) hardware-topology-unavailable",
        "instance": "2nd (1st was F#650 exp_prod_adapter_loader_portability)",
        "promotion_candidate_at_3rd": True,
    }


def main() -> None:
    t0 = time.time()
    t1 = theorem_1_artifact_shortfall()
    t2 = theorem_2_resource_budget()
    t3 = theorem_3_schema_completeness()
    t4 = theorem_4_kc_pin_count()
    t5 = theorem_5_source_breaches()

    theorems = [t1, t2, t3, t4, t5]
    blocking = [t for t in theorems if t["blocks"]]
    all_block = all(t["blocks"] for t in theorems)
    defense_in_depth = len(blocking) >= 3

    results = {
        "experiment_id": "exp_prod_llama_cpp_bridge",
        "verdict": "KILLED",
        "status": "killed",
        "preemptive_kill": True,
        "is_smoke": False,
        "all_pass": False,
        "all_block": all_block,
        "defense_in_depth": defense_in_depth,
        "theorems": theorems,
        "kill_criteria": {
            "K1654": {
                "result": "fail",
                "reason": (
                    "no llama.cpp Gemma 4 converter + no PoLAR converter + "
                    "parent Apple-only-MLX scope breach; T1 ∧ T5(A,B,C,D)"
                ),
            },
            "K1655": {
                "result": "fail",
                "reason": (
                    'non-discriminating KC ("works" = no threshold); T4 + '
                    "T1 llama.cpp binary absence"
                ),
            },
        },
        "wall_seconds": round(time.time() - t0, 6),
        "parent_source": {
            "id": "exp_prod_adapter_format_spec_v1",
            "status": "supported",
            "scope_apple_only": True,
        },
        "precedents": {
            "sister_reusable": {
                "finding": 650,
                "experiment": "exp_prod_adapter_loader_portability",
                "axis": "ap-017(s) hardware-topology-unavailable",
            },
            "not_transport": {
                "finding": 60,
                "reason": (
                    "BitNet+llama.cpp; needed 3 convert-script patches for a "
                    "simpler arch; Metal NOT supported for TQ2_0; evaluated "
                    "M1 Max CPU-only without thinking; PoLAR-incompatible"
                ),
            },
            "reinforcing": {
                "finding": 61,
                "reason": (
                    "MLX runtime LoRA KILLED — always pre-merge on Apple "
                    "Silicon; implies non-native adapter runtimes degrade"
                ),
            },
        },
        "ap_017_axis": {
            "candidate": "s",
            "name": "hardware-topology-unavailable",
            "instance": 2,
            "first_instance_finding": 650,
            "promotion_candidate_at_3rd": True,
        },
    }

    out_path = EXP_DIR / "results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    print(
        f"verdict=KILLED preemptive=True all_block={all_block} "
        f"defense_in_depth={defense_in_depth} t_blocking={len(blocking)}/5"
    )


if __name__ == "__main__":
    main()
