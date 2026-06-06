"""
V2 audit-rerun probe for T4.3: MLX-Native Adapter Hot-Swap Serving.

Purpose: verify preconditions + detect design flaws before claiming PASS.
V1 (2026-04-17) reported "supported" but the claim is retroactively invalid
for FOUR independent reasons, each sufficient to kill on its own:

  C1) Upstream artefacts absent. V1 needs 5 adapter `.safetensors` files from
      T2.1 + T2.6. 0/5 are on disk as of 2026-04-18 (T2.1 KILLED 2026-04-18
      for MCQ metric-swap + format-artefact; T2.6 adapter weights lost —
      only `adapter_config.json` remains in each dir). K1081 "5/5 load+generate"
      therefore cannot be verified. Precondition-probe-8th-instance.

  C2) Hidden MLX graph compilation overhead (mem-antipattern-011 specialisation).
      V1's `swap_adapter(model, path)` times only:
          t0 = time.perf_counter()
          model.load_weights(str(weights_file), strict=False)
          mx.eval(model.parameters())
          t1 = time.perf_counter()
      The MLX compute graph is recompiled / re-traced on the FIRST forward pass
      after a parameter mutation — not at `mx.eval(model.parameters())` (which
      only materialises the new tensors on device). V1's Phase 2 loops 20
      swaps back-to-back without a single forward pass between, so the real
      post-swap cost never enters the clock. Theorem 3 "swap time bounded by
      S_adapter / B_mem + T_eval" omits the recompile term. A correct
      measurement times `load_weights` → `mx.eval` → first-token generation
      minus the baseline prefill time of the same prompt.

  C3) Prefill/decode conflation + out-of-domain prompt bias (new: throughput-
      conflation antipattern). V1's K1083 "adapter throughput ≥ 80% of base":
        base_tok_s  = total_tokens / (prefill + decode) on prompt P1 (no adapter)
        lora_tok_s  = total_tokens / (prefill + decode) on prompt P1 (math adapter)
      where P1 = "Explain the concept of machine learning in simple terms."
      Two flaws:
        (a) Prefill is compute-bound, decode is memory-bound. Conflating them
            into a single tok/s hides which phase the adapter actually slows.
            The correct metric is `N_decoded / (t_end - t_first_token)`, which
            strips prefill entirely.
        (b) The math adapter is evaluated on a non-math prompt. Out-of-domain
            generation commonly early-EOS or emits unusual token distributions,
            skewing tok/s in either direction. Paper's Phase 1 table itself
            flags the medical adapter outlier (3.7 tok/s vs 26–28) as a
            short-answer artefact — same failure mode, applied to K1083 by
            construction. Ratio 90.8% is therefore not a characterisation of
            LoRA overhead on Apple Silicon.

  C4) Tautological routing (mem-antipattern-002). V1's "routing registry":
          routing_registry = {d: p for d, p in ADAPTER_PATHS.items()}
          for domain, adapter_path in ADAPTER_PATHS.items():
              selected_path = routing_registry[domain]
      This is `dict[k] == dict[k]`. `selected_path == adapter_path` is True
      by set-theoretic identity, regardless of any routing logic. The
      experiment's routing header in the title ("via routing header") and
      the Theorem 3 connection to T4.1 TF-IDF both require routing from the
      PROMPT TEXT to a domain — which is never invoked. K1084 "correct adapter
      selected per request" is forced, not measured. Measured latency ~0.7µs
      is a dict-hash microbench, not TF-IDF routing cost (which would include
      tokenisation + sparse matmul + argmax, per T4.1's own math).

Additionally, the V1 code writes `results.json` under this file's directory,
but no `results.json` exists on disk here — V1 run outputs were apparently
lost. The PAPER.md "SUPPORTED" verdict therefore rests on an unverifiable
claim even before the C1–C4 analysis.

This probe does NOT load the model. It checks filesystem state that V1 should
have produced, identifies the 4 design flaws by reference, and routes K1081–
K1084 to FAIL with explicit reason strings. Pure os.path inspection + trivial
benchmarks. Fast (<1s) and side-effect-free.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent

T21_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training"
T26_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_multi_domain_5"

EXPECTED_ADAPTERS = {
    "math":    T21_DIR / "adapters" / "math",
    "code":    T21_DIR / "adapters" / "code",
    "medical": T21_DIR / "adapters" / "medical",
    "legal":   T26_DIR / "adapters" / "legal",
    "finance": T26_DIR / "adapters" / "finance",
}


def probe_adapter_dir(path: Path) -> dict:
    info = {
        "dir": str(path.relative_to(REPO_ROOT)) if path.is_absolute() else str(path),
        "dir_exists": path.exists() and path.is_dir(),
        "config_exists": (path / "adapter_config.json").exists(),
        "safetensors_exists": False,
        "safetensors_size_bytes": 0,
    }
    if info["dir_exists"]:
        for candidate in ("adapters.safetensors", "adapter_model.safetensors"):
            f = path / candidate
            if f.exists():
                info["safetensors_exists"] = True
                info["safetensors_size_bytes"] = f.stat().st_size
                info["safetensors_file"] = candidate
                break
    return info


def upstream_verdict(exp_dir: Path) -> dict:
    rj = exp_dir / "results.json"
    out = {"dir": str(exp_dir.relative_to(REPO_ROOT)), "results_exists": rj.exists()}
    if rj.exists():
        try:
            data = json.loads(rj.read_text())
            out["verdict"] = data.get("verdict")
            out["all_pass"] = data.get("all_pass")
            out["audit_note"] = data.get("_audit_note") or data.get("_v2_note")
        except Exception as exc:
            out["parse_error"] = str(exc)
    return out


def microbench_dict_lookup(n: int = 1000) -> float:
    """Reproduce V1's K1084 measurement to expose its tautology."""
    d = {f"domain_{i}": f"/nonexistent/adapter_{i}" for i in range(n)}
    total = 0.0
    keys = list(d.keys())
    for k in keys:
        t0 = time.perf_counter()
        _ = d[k]
        total += (time.perf_counter() - t0) * 1e6  # microseconds
    return total / n


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("V2 AUDIT PROBE: exp_p1_t4_vllm_adapter_serving")
    print("=" * 72)

    adapters = {name: probe_adapter_dir(p) for name, p in EXPECTED_ADAPTERS.items()}
    n_real_present = sum(1 for v in adapters.values() if v["safetensors_exists"])
    n_config_only = sum(
        1 for v in adapters.values()
        if v["dir_exists"] and v["config_exists"] and not v["safetensors_exists"]
    )

    t21 = upstream_verdict(T21_DIR)
    t26 = upstream_verdict(T26_DIR)

    v1_results_on_disk = (EXPERIMENT_DIR / "results.json").exists()

    # K1081: MLX loads Gemma 4 E4B + 5 LoRA adapters, generates valid output.
    # Precondition failure: 0/5 adapter .safetensors present.
    k1081 = {
        "pass": False,
        "reason": (
            f"Upstream adapter artefacts absent (precondition failure). "
            f"{n_real_present}/5 expected .safetensors present on disk; "
            f"{n_config_only}/5 dirs are config-only. T2.1 upstream "
            f"status=killed (2026-04-18, MCQ metric-swap + format-artefact). "
            f"T2.6 adapter .safetensors lost. `load_adapters(model, path)` "
            f"requires a weights file in each adapter dir — V1 "
            f"ADAPTER_PATHS would raise FileNotFoundError on the first "
            f"swap if re-run today. V1 results.json also absent from disk "
            f"(unverifiable claim). "
            f"Probe: math={adapters['math']['safetensors_exists']}, "
            f"code={adapters['code']['safetensors_exists']}, "
            f"medical={adapters['medical']['safetensors_exists']}, "
            f"legal={adapters['legal']['safetensors_exists']}, "
            f"finance={adapters['finance']['safetensors_exists']}."
        ),
    }

    # K1082: adapter swap p99 < 50ms.
    # Structurally wrong object: V1 times load_weights + mx.eval(parameters)
    # in a tight loop. MLX recompiles the forward graph on the FIRST forward
    # pass after parameter mutation, not at mx.eval. Theorem 1's
    # S_adapter / B_mem + T_eval omits the recompile term entirely.
    k1082 = {
        "pass": False,
        "reason": (
            "Wrong object measured (mem-antipattern-011 specialisation for "
            "MLX graph compilation). V1's swap_adapter: "
            "    t0; model.load_weights(...); mx.eval(model.parameters()); t1 "
            "times only parameter materialisation. MLX recompiles the forward "
            "compute graph on the first forward pass following a parameter "
            "mutation. V1's Phase 2 loops 20 back-to-back swaps without any "
            "generation between them, so the recompile cost never enters the "
            "clock. Theorem 1 T_swap ≤ S_adapter / B_mem + T_eval omits the "
            "recompile term. A correct measurement would be "
            "`t_first_token(after_swap) - t_first_token(baseline_same_prompt)` — "
            "i.e. swap latency ≡ incremental time-to-first-token after the swap, "
            "not load_weights() wall time alone. V1 4.77ms under 50ms threshold "
            "is a ~10.5× margin on a benchmark that omits the real post-swap "
            "cost. Additionally cannot be re-run now: preconditions (C1) absent."
        ),
    }

    # K1083: adapter throughput >= 80% of base.
    # Two flaws:
    #   (a) prefill + decode conflated into single tok/s
    #   (b) math adapter evaluated on non-math prompt (out-of-domain bias)
    k1083 = {
        "pass": False,
        "reason": (
            "Throughput metric conflates prefill (compute-bound) and decode "
            "(memory-bound) phases and applies an out-of-domain prompt to the "
            "math adapter. Specifically: "
            "(a) V1's generate_tokens includes both the prefill (prompt "
            "processing, ~O(L_prompt) in compute) and the decode (~O(1/tok), "
            "memory-bandwidth-dominated on Apple Silicon). LoRA overhead is a "
            "per-token memory-traffic addition; the adapter's effect on the "
            "memory-bound regime is exactly what the 80%-ratio KC wants to "
            "characterise. Mixing prefill into the denominator hides that "
            "effect. Correct metric: decode-only "
            "tok/s = N_decoded / (t_end - t_first_token). "
            "(b) V1 evaluates the math adapter on P1 = 'Explain the concept "
            "of machine learning in simple terms.' — a non-math prompt. "
            "Out-of-domain prompting commonly early-EOSes or shifts the token "
            "distribution. V1's own Phase 1 notes the medical adapter hit 3.7 "
            "tok/s 'because model answered briefly (denominator small)', the "
            "same failure mode as this K1083 measurement. The 90.8% ratio is "
            "not a characterisation of LoRA overhead. Additionally cannot be "
            "re-run now: preconditions (C1) absent."
        ),
    }

    # K1084: correct adapter per request via routing registry.
    # routing_registry = {d: p for d, p in ADAPTER_PATHS.items()}
    # selected = routing_registry[domain]  # domain provided by test
    # selected == adapter_path is a set-theoretic identity.
    dict_lookup_us = microbench_dict_lookup(n=1000)
    k1084 = {
        "pass": False,
        "measured_dict_lookup_us": dict_lookup_us,
        "reason": (
            "Tautological routing (mem-antipattern-002). V1 code: "
            "    routing_registry = {d: p for d, p in ADAPTER_PATHS.items()} "
            "    selected_path = routing_registry[domain] "
            "where `domain` is the iteration key from ADAPTER_PATHS. Testing "
            "`selected_path == adapter_path` is `dict[k] == dict[k]` — True "
            "by set-theoretic identity, not by routing logic. Zero TF-IDF, "
            "zero text input, zero classifier. K1084 requires a router that "
            "takes raw prompt text as input and predicts a domain label; "
            "T4.1's pipeline (tokenise → sparse matmul → argmax) was never "
            "invoked. Reported latency (~0.7µs) is a Python dict-hash "
            f"microbench (this probe reproduces: {dict_lookup_us:.3f}µs mean "
            f"over 1000 lookups), not TF-IDF routing cost. "
            "Additionally cannot be re-run now: preconditions (C1) absent."
        ),
    }

    total_s = time.perf_counter() - t0

    results = {
        "verdict": "KILLED",
        "all_pass": False,
        "ran": True,
        "is_smoke": False,
        "_v2_note": (
            "V2 audit-rerun 2026-04-18. V1 'supported' (2026-04-17) retroactively "
            "invalid for FOUR independent reasons: (C1) 0/5 upstream adapter "
            ".safetensors on disk (T2.1 KILLED, T2.6 weights lost); "
            "(C2) swap_adapter times load_weights + mx.eval without forward pass, "
            "omitting MLX graph-recompile cost (Theorem 3 object mis-specified); "
            "(C3) throughput conflates prefill/decode and uses OOD prompt on math "
            "adapter; (C4) routing_registry = identity-dict on ADAPTER_PATHS, "
            "selected_path == adapter_path by set-theoretic identity. V1 "
            "results.json also missing from disk — finding unverifiable."
        ),
        "_audit_tags": [
            "audit-2026-04-17-rerun",
            "code-bug",
            "tautological-routing",
            "prefill-decode-conflation",
            "graph-compile-omitted",
            "ood-prompt-bias",
            "precondition-probe-8th-instance",
        ],
        "adapter_preconditions": adapters,
        "n_real_adapter_safetensors_present": n_real_present,
        "n_config_only_dirs": n_config_only,
        "v1_results_json_on_disk": v1_results_on_disk,
        "upstream": {
            "exp_p1_t2_single_domain_training": t21,
            "exp_p1_t2_multi_domain_5":         t26,
        },
        "v1_design_flaws": [
            "0/5 upstream adapter .safetensors present on disk",
            "swap_adapter times load_weights + mx.eval, omits MLX graph recompile",
            "Phase 2 loops 20 back-to-back swaps without any forward pass between",
            "throughput metric mixes prefill (compute-bound) and decode (memory-bound)",
            "math adapter evaluated on non-math prompt (OOD generation bias)",
            "routing_registry = {d: p for d, p in ADAPTER_PATHS.items()} is identity dict",
            "K1084 tests dict[domain] == ADAPTER_PATHS[domain] — True by construction",
            "reported <1µs routing is Python dict hash, not TF-IDF pipeline cost",
            "V1 results.json missing from disk (claim unverifiable even on provenance)",
        ],
        "k1081": k1081,
        "k1082": k1082,
        "k1083": k1083,
        "k1084": k1084,
        "K1081_loads_and_generates":     "FAIL",
        "K1082_swap_under_50ms":         "FAIL",
        "K1083_throughput_ratio_80pct":  "FAIL",
        "K1084_routing_correct":         "FAIL",
        "total_time_s": total_s,
        "_v1_numbers_for_reference": {
            "note": (
                "V1 2026-04-17 PAPER.md measurements. Unverifiable (results.json "
                "missing); kept for provenance only."
            ),
            "swap_p50_ms": 3.62,
            "swap_p99_ms": 4.77,
            "swap_max_ms": 4.79,
            "base_tok_s": 41.5,
            "lora_tok_s": 37.6,
            "throughput_ratio": 0.908,
            "routing_latency_us": 0.7,
        },
    }

    out_path = EXPERIMENT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[probe] wrote {out_path.relative_to(REPO_ROOT)}")
    print(
        f"[probe] verdict=KILLED n_real_adapter_safetensors={n_real_present}/5 "
        f"n_config_only={n_config_only}/5 dict_lookup_us={dict_lookup_us:.3f}"
    )
    print(f"[probe] elapsed={total_s:.3f}s")


if __name__ == "__main__":
    main()
