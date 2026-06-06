#!/usr/bin/env python3
"""
T3.4 V2 — N=25 Domain Composition (audit-2026-04-17-rerun, tautological-routing)

MATH: experiments/models/exp_p1_t3_n25_composition/MATH.md (see V2 Audit Section)

V1 verdict retroactively invalid for 2 independent reasons:
  (1) All 5 adapter .safetensors missing on disk (T2.1 KILLED 2026-04-18;
      T2.6 adapter weights lost).
  (2) V1 design hardcodes REAL_ADAPTER_PATHS[domain] — tautological routing
      under mem-antipattern-002. K1060/K1061 do not test composition, they
      test single-adapter-on-matched-domain.

V2 probe structure:
  Phase 1: Grassmannian QR check (K1059) — pure numpy, genuine measurement
  Phase 2: Adapter precondition probe (K1060) — filesystem check, FAIL
  Phase 3: Tautological-routing design analysis (K1061) — FAIL by design,
           no eval performed
  Phase 4: Theoretical size calc (K1062) — formula-only, moot without real
           weights

Outcome: verdict=KILLED, all_pass=false.
"""

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Constants (match V1 MATH.md Theorems 1-3)
N_LAYERS = 42
RANK = 6
D_IN = 2560
D_OUT = 2048
N_DOMAINS = 25
SEED = 42

# Adapter paths (match V1)
T21_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training"
T26_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_multi_domain_5"

REAL_ADAPTER_PATHS = {
    "math":    T21_DIR / "adapters" / "math",
    "code":    T21_DIR / "adapters" / "code",
    "medical": T21_DIR / "adapters" / "medical",
    "legal":   T26_DIR / "adapters" / "legal",
    "finance": T26_DIR / "adapters" / "finance",
}


def phase1_grassmannian_check() -> dict:
    """K1059: QR orthogonality on random numpy matrices. Pure math, reproducible."""
    print("\n=== Phase 1: Grassmannian Orthogonality (K1059) ===", flush=True)
    t0 = time.time()

    rng = np.random.default_rng(SEED)
    max_cos_global = 0.0
    layer_maxes = []

    for layer in range(N_LAYERS):
        W = rng.standard_normal((D_IN, RANK * N_DOMAINS)).astype(np.float64)
        Q, _ = np.linalg.qr(W)
        Q = Q.astype(np.float32)
        A_list = [Q[:, i * RANK:(i + 1) * RANK] for i in range(N_DOMAINS)]

        max_cos_layer = 0.0
        for i, j in combinations(range(N_DOMAINS), 2):
            Ai, Aj = A_list[i], A_list[j]
            AiAj = Ai.T @ Aj
            fro_ij = float(np.linalg.norm(AiAj, "fro"))
            fro_i = float(np.linalg.norm(Ai, "fro"))
            fro_j = float(np.linalg.norm(Aj, "fro"))
            cos = fro_ij / (fro_i * fro_j + 1e-30)
            if cos > max_cos_layer:
                max_cos_layer = cos

        layer_maxes.append(max_cos_layer)
        if max_cos_layer > max_cos_global:
            max_cos_global = max_cos_layer

    elapsed = time.time() - t0
    k1059_pass = max_cos_global < 1e-5

    print(f"  Layers tested: {N_LAYERS}", flush=True)
    print(f"  Global max|cos| = {max_cos_global:.3e}", flush=True)
    print(f"  K1059 (<1e-5): {'PASS' if k1059_pass else 'FAIL'}", flush=True)
    print(f"  Phase 1 time: {elapsed:.1f}s", flush=True)

    return {
        "max_cos_grassmannian": float(max_cos_global),
        "cos_mean_over_layers": float(np.mean(layer_maxes)),
        "n_layers_tested": N_LAYERS,
        "k1059_pass": k1059_pass,
        "phase1_time_s": round(elapsed, 2),
    }


def phase2_adapter_precondition_probe() -> dict:
    """K1060: Probe adapter .safetensors existence. 5th precondition-probe case."""
    print("\n=== Phase 2: Adapter Precondition Probe (K1060) ===", flush=True)

    per_domain = {}
    n_present = 0
    for domain, path in REAL_ADAPTER_PATHS.items():
        st_path = path / "adapters.safetensors"
        exists = st_path.exists()
        size_b = st_path.stat().st_size if exists else 0
        per_domain[domain] = {
            "dir": str(path.relative_to(EXPERIMENT_DIR.parent.parent.parent))
                   if path.is_absolute() else str(path),
            "safetensors_exists": exists,
            "safetensors_size_bytes": size_b,
        }
        if exists:
            n_present += 1
        print(f"  {domain}: safetensors={'PRESENT' if exists else 'MISSING'} "
              f"({size_b / 1e6:.2f} MB)" if exists else
              f"  {domain}: safetensors=MISSING (0 B, only config stub)",
              flush=True)

    # K1060 structurally requires adapter weights AND genuine composition
    # (simultaneous activation or per-sample routing). V1 had neither.
    k1060_pass = False
    reason = (
        "Adapter preconditions fail: 0/5 .safetensors on disk. Additionally, "
        "even if weights existed, V1 design hardcodes REAL_ADAPTER_PATHS[domain] "
        "which is tautological routing (mem-antipattern-002) — tests each adapter "
        "only on its matched domain subset, not composition."
    )
    print(f"  Adapters present: {n_present}/5", flush=True)
    print(f"  K1060: FAIL — {reason}", flush=True)

    return {
        "adapter_preconditions": per_domain,
        "n_adapter_safetensors_present": n_present,
        "k1060_pass": k1060_pass,
        "k1060_fail_reason": reason,
    }


def phase3_tautological_routing_analysis() -> dict:
    """K1061: design analysis — MMLU preservation under composition is
    structurally unmeasurable because V1 design tests per-domain-adapter
    on neutral subjects, not composition."""
    print("\n=== Phase 3: Tautological Routing Design Analysis (K1061) ===", flush=True)

    # Read v1 run_experiment.py design flaws from source (this file is the v2 rewrite).
    # V1 had: for domain in [math, medical, legal, finance]:
    #             for subj in neutral_subjects:
    #                 eval_mmlu(subj, REAL_ADAPTER_PATHS[domain], ...)
    # This loads ONE adapter per call — not composition, not routing.
    design_flaws = [
        "V1 Phase 3 loops over domain × neutral_subject, loading ONE adapter per eval",
        "No N=25 composition happens — adapter activation is per-call single-domain",
        "MCQ format transfer (56-88% on neutral) is single-adapter behavior, not composition behavior",
        "Theorem 3 'exclusive routing → zero interference' is never exercised — routing function is absent",
    ]
    for f in design_flaws:
        print(f"  - {f}", flush=True)

    k1061_pass = False
    reason = (
        "Structurally unmeasurable: V1 design tests single-adapter-on-neutral-"
        "subject, not composition. True composition claim requires simultaneous "
        "N=25 activation or per-sample routing (see MATH.md V2 Audit Section)."
    )
    print(f"  K1061: FAIL — {reason}", flush=True)

    return {
        "design_flaws": design_flaws,
        "k1061_pass": k1061_pass,
        "k1061_fail_reason": reason,
    }


def phase4_theoretical_size() -> dict:
    """K1062: theoretical size by formula. Moot without real weights."""
    print("\n=== Phase 4: Theoretical Size Calculation (K1062) ===", flush=True)

    # Per-domain: 42 layers × (D_IN × r + r × D_OUT) × 4 B (float32)
    per_domain_bytes = N_LAYERS * (D_IN * RANK + RANK * D_OUT) * 4
    per_domain_mb = per_domain_bytes / (1024**2)
    total_mb = 25 * per_domain_mb
    limit_mb = 1024.0
    k1062_pass = total_mb < limit_mb

    print(f"  Per-adapter theoretical: {per_domain_mb:.2f} MB (float32)", flush=True)
    print(f"  25 adapters theoretical: {total_mb:.2f} MB", flush=True)
    print(f"  Limit: {limit_mb:.0f} MB", flush=True)
    print(f"  K1062 (<1GB theoretical): {'PASS' if k1062_pass else 'FAIL'} (MOOT — no real adapters)", flush=True)

    return {
        "per_domain_theoretical_mb": round(per_domain_mb, 2),
        "total_theoretical_mb": round(total_mb, 2),
        "limit_mb": limit_mb,
        "k1062_pass": k1062_pass,
        "k1062_moot_note": "Formula holds but no real adapter weights on disk to measure",
    }


def main():
    print("=" * 60, flush=True)
    print("T3.4 V2 — N=25 Grassmannian Composition (precondition-probe)", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    results = {
        "verdict": "KILLED",
        "all_pass": False,
        "ran": True,
        "is_smoke": False,
        "_v2_note": (
            "V2 audit-rerun 2026-04-18. V1 PAPER.md 'supported' retroactively "
            "invalid for 2 independent reasons: (1) adapters .safetensors missing "
            "on disk (T2.1 KILLED 2026-04-18; T2.6 weights lost); (2) V1 design "
            "uses tautological routing — REAL_ADAPTER_PATHS[domain] hardcodes "
            "adapter-to-domain pairing; no genuine composition exercised."
        ),
        "_audit_tags": ["audit-2026-04-17-rerun", "tautological-routing",
                        "precondition-probe-5th-instance"],
    }

    results.update(phase1_grassmannian_check())
    results.update(phase2_adapter_precondition_probe())
    results.update(phase3_tautological_routing_analysis())
    results.update(phase4_theoretical_size())

    # KC routing summary
    results["K1059_grassmannian_orthogonality"] = "PASS" if results["k1059_pass"] else "FAIL"
    results["K1060_no_domain_degraded"] = "PASS" if results["k1060_pass"] else "FAIL"
    results["K1061_mmlu_preserved"] = "PASS" if results["k1061_pass"] else "FAIL"
    results["K1062_size_under_1gb"] = "PASS" if results["k1062_pass"] else "FAIL"

    results["total_time_s"] = round(time.time() - t0, 2)

    # V1 numbers kept for reference (unverifiable — no .safetensors)
    results["_v1_numbers_for_reference"] = {
        "note": "V1 2026-04-10 measurements. Unverifiable now; kept for provenance only.",
        "max_cos_grassmannian": 2.165e-8,
        "math_gsm8k_pct": 44.0,
        "medical_medmcqa_pct": 36.0,
        "legal_mmlu_prof_law_pct": 64.0,
        "finance_mmlu_macro_pct": 56.0,
        "code_mmlu_cs_pct": 72.0,
        "mmlu_neutral_range_pct": [56.0, 88.0],
        "total_size_mb": 48.45,
    }

    print("\n" + "=" * 60, flush=True)
    print("KC SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  K1059 (Grassmannian):        {results['K1059_grassmannian_orthogonality']} (genuine)", flush=True)
    print(f"  K1060 (0/25 degraded):       {results['K1060_no_domain_degraded']} (adapters missing + tautological)", flush=True)
    print(f"  K1061 (MMLU preserved):      {results['K1061_mmlu_preserved']} (design analysis)", flush=True)
    print(f"  K1062 (<1GB theoretical):    {results['K1062_size_under_1gb']} (moot)", flush=True)
    print(f"  Verdict: {results['verdict']}, all_pass={results['all_pass']}", flush=True)
    print(f"  Total time: {results['total_time_s']:.1f}s", flush=True)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {RESULTS_FILE}", flush=True)

    return results


if __name__ == "__main__":
    main()
