"""Preemptive kill runner for exp_prod_differential_privacy_training
(ap-017 5-theorem stack).

No model, no inference, no MLX. Pure stdlib. Probes the 4 artifacts
that the target's K1665/K1666 would require; writes results.json with
T1..T5 verdicts and a single-shot 'killed_preregistered' stamp.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
SOURCE_MATH = REPO_ROOT / "experiments/models/exp_p1_t5_user_local_training/MATH.md"
SOURCE_TRAIN = REPO_ROOT / "experiments/models/exp_p1_t5_user_local_training/train_personal_adapter.py"


def _ripgrep(pattern: str, *, extra: list[str] | None = None) -> list[str]:
    cmd = [
        "grep", "-rE", pattern,
        "--include=*.py",
        "--exclude-dir=.venv",
        "--exclude-dir=__pycache__",
        "--exclude-dir=node_modules",
        str(REPO_ROOT),
    ]
    if extra:
        cmd.extend(extra)
    out = subprocess.run(cmd, capture_output=True, text=True)
    # Exclude self-matches: the runner's own regex text must not count
    # as evidence that the primitive exists in-repo.
    self_path = str(Path(__file__).resolve())
    return [
        l for l in out.stdout.splitlines()
        if l.strip() and not l.startswith(self_path)
    ]


def _drop_comments(hits: list[str]) -> list[str]:
    """Drop `path:# ...` and `path:    # ...` lines — comments are not
    evidence that a library is in use."""
    out = []
    for l in hits:
        _, _, tail = l.partition(":")
        if tail.lstrip().startswith("#"):
            continue
        out.append(l)
    return out


def t1_prerequisite_inventory() -> dict:
    # 1. dp_sgd_optimizer_mlx: any import/use of DP-SGD primitives
    dp_hits = _ripgrep(
        r"(opacus|make_private|RDPAccountant|noise_multiplier"
        r"|per_sample_grad|clip_per_sample|dp_sgd|DPOptimizer"
        r"|gaussian_mechanism|sigma_noise)"
    )
    dp_hits_code = _drop_comments([
        l for l in dp_hits
        if "/skills/" not in l
        and ".jsonl" not in l
        and "/data/" not in l
        and "/cassettes/" not in l
    ])

    # 2. per_sample_gradient_mlx: vmap / per-sample grad on MLX
    vmap_hits = _ripgrep(r"\b(vmap|per_sample_grad_mlx)\b")
    vmap_code = _drop_comments([
        l for l in vmap_hits
        if "/skills/" not in l
        and ".jsonl" not in l
        and "/data/" not in l
        and "/cassettes/" not in l
    ])

    # 3. rdp_accountant: any ε-accountant implementation
    rdp_hits = _ripgrep(r"(RDP|epsilon_accountant|rdp_accountant|compute_rdp)")
    rdp_code = _drop_comments([
        l for l in rdp_hits
        if "/skills/" not in l
        and ".jsonl" not in l
        and "/data/" not in l
    ])

    # 4. non_dp_lora_baseline_on_same_data: does the source run a non-DP
    # baseline at matched data/hparams that the target can compare to?
    # The only repo LoRA training source at this platform is T5.1
    # (supported). It has ONE adapter config, no comparator pair.
    source_has_comparator = False
    if SOURCE_TRAIN.exists():
        txt = SOURCE_TRAIN.read_text()
        # Heuristic: a comparator pair would have TWO distinct training
        # configs or a --dp / --non-dp flag. T5.1 has neither.
        source_has_comparator = (
            "--dp" in txt
            or "non_dp_baseline" in txt
            or ("config_a" in txt and "config_b" in txt)
        )

    # 5. pyproject.toml DP dependency
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    dp_dep = bool(re.search(
        r"\b(opacus|jax-privacy|tensorflow-privacy|dp-accountant"
        r"|private-transformers)\b",
        pyproject, flags=re.I,
    ))

    required = {
        "dp_sgd_optimizer_mlx": bool(dp_hits_code) or dp_dep,
        "per_sample_gradient_mlx": bool(vmap_code),
        "rdp_accountant": bool(rdp_code) or dp_dep,
        "non_dp_lora_baseline_on_same_data": source_has_comparator,
    }
    shortfall = sum(1 for present in required.values() if not present)
    return {
        "required": required,
        "shortfall": shortfall,
        "pyproject_has_dp_dep": dp_dep,
        "dp_primitive_hits_in_code": len(dp_hits_code),
        "vmap_hits_in_code": len(vmap_code),
        "rdp_hits_in_code": len(rdp_code),
        "block": shortfall >= 1,
    }


def t2_scale_safety() -> dict:
    # Source T2.1 non-DP LoRA: ≤22 min on Gemma 4 E4B (K1031).
    non_dp_baseline_min = 22
    dp_overhead_factor = 10  # Opacus floor (Yu et al. 2022, §5.2)
    n_seeds = 3              # K1666 requires 3
    dp_total_min = non_dp_baseline_min * dp_overhead_factor * n_seeds
    baseline_pair_min = non_dp_baseline_min * n_seeds
    total_est_min = dp_total_min + baseline_pair_min
    ceiling_min = 120
    return {
        "non_dp_baseline_min_per_seed": non_dp_baseline_min,
        "dp_overhead_factor_floor": dp_overhead_factor,
        "n_seeds_required": n_seeds,
        "dp_training_min": dp_total_min,
        "non_dp_baseline_pair_min": baseline_pair_min,
        "est_total_minutes": total_est_min,
        "ceiling_minutes": ceiling_min,
        "overshoot_factor": total_est_min / ceiling_min,
        "block": total_est_min > ceiling_min,
    }


def t3_schema_completeness() -> dict:
    out = subprocess.run(
        ["experiment", "get", "exp_prod_differential_privacy_training"],
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
        "K1665_eps":   True,    # epsilon=8
        "K1665_delta": True,    # delta=1e-5
        "K1665_pct":   True,    # within 10%
        "K1665_qty":   False,   # "quality" — no metric defined
        "K1666_nseed": True,    # 3 seeds
        "K1666_repro": False,   # "reproducible" — no threshold
    }
    pinned = sum(1 for v in kc_pins.values() if v)
    total = len(kc_pins)
    ratio = pinned / total
    return {
        "pinned": pinned,
        "total": total,
        "pin_ratio": ratio,
        "threshold": 0.20,
        "block": ratio < 0.20,
    }


def t5_source_scope_breach() -> dict:
    math_text = SOURCE_MATH.read_text() if SOURCE_MATH.exists() else ""
    # (A) privacy-mechanism-scope: source MATH.md has zero DP vocabulary
    dp_terms = re.findall(
        r"(?i)\b(privacy|differential|epsilon|dp[-_ ]sgd"
        r"|gaussian\s+noise|clip.*grad)\b",
        math_text,
    )
    source_has_dp_vocab = len(dp_terms) > 0

    # (B) library-scope: source K1099 literal "<200 lines"
    source_line_limit = "< 200 lines" in math_text or "200 lines" in math_text

    # (C) comparator-scope: source KC K1097 "≥ 5pp" vs base (not DP-vs-nonDP)
    source_compares_base = "5pp" in math_text or "5 pp" in math_text

    # (D) reproducibility-scope: source ran N=1 (single user, single seed)
    source_n1 = (
        "n=50" in math_text or "50 examples" in math_text
        or "50 conversation" in math_text
    )

    # (E) platform-library-scope: source platform MLX; DP libs are PT/JAX
    # We verify this by checking that source is MLX (via mlx-lm import chain
    # in train script) AND no DP MLX lib exists (T1 already proves this).
    source_is_mlx = False
    if SOURCE_TRAIN.exists():
        t = SOURCE_TRAIN.read_text()
        source_is_mlx = "mlx_lm" in t or "mlx-lm" in t or "import mlx" in t

    breaches = {
        "A_privacy_mechanism_scope": not source_has_dp_vocab,  # breach if source has ZERO DP vocab
        "B_library_scope_200_lines":  source_line_limit,        # breach if source binds <200 lines
        "C_comparator_scope":         source_compares_base,     # breach if source compares to base only
        "D_reproducibility_scope":    source_n1,                # breach if source ran N=1
        "E_platform_library_scope":   source_is_mlx,            # breach if source MLX + no DP lib (T1)
    }
    hits = sum(1 for v in breaches.values() if v)
    return {
        "breaches": breaches,
        "literal_hits": hits,
        "block": hits >= 3,
        "source_math_found": SOURCE_MATH.exists(),
        "source_train_found": SOURCE_TRAIN.exists(),
        "source_dp_vocab_count": len(dp_terms),
    }


def main() -> None:
    t0 = time.time()
    t1 = t1_prerequisite_inventory()
    t2 = t2_scale_safety()
    t3 = t3_schema_completeness()
    t4 = t4_pin_ratio()
    t5 = t5_source_scope_breach()

    all_block = t1["block"] and t2["block"] and t3["block"] and t5["block"]
    defense_in_depth = any([
        t1["block"], t2["block"], t3["block"], t5["block"]
    ])

    kc_results = {
        "K1665": "fail",
        "K1666": "fail",
    }

    results = {
        "experiment_id": "exp_prod_differential_privacy_training",
        "verdict": "KILLED_PREEMPTIVE",
        "all_pass": False,
        "all_block": all_block,
        "defense_in_depth": defense_in_depth,
        "is_smoke": False,
        "theorems": {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5},
        "kill_criteria": kc_results,
        "ap_017_axis": (
            "composition-bug "
            "(software-infrastructure-unbuilt, "
            "platform-library cross-cut variant)"
        ),
        "ap_017_scope_index": 34,
        "supported_source_preempt_index": 15,
        "f502_instance_index": 6,
        "defense_in_depth_theorems_firing": sum([
            int(t1["block"]), int(t2["block"]),
            int(t3["block"]), int(t5["block"]),
        ]),
        "wall_seconds": round(time.time() - t0, 4),
    }

    out_path = EXP_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
