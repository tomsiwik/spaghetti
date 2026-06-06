"""
exp_pierre_adapter_hotswap_latency — design-lock PROVISIONAL scaffold.

This scaffold does NOT load MLX, does NOT run the benchmark, and always writes
a valid results.json with verdict=PROVISIONAL and KCs as "untested". It enumerates
the structural reasons actual execution is deferred to `_impl` and returns 0.

Rationale (see MATH.md §7):
  B1. Pre-reg hygiene (K1910 "token glitch") needed operational definition in
      MATH.md first; running before the definition would have violated KC-lock.
  B2. PLAN.md §1012 requires `/mlx-dev` + `/fast-mlx` invocation before MLX
      platform code. This iteration does NOT invoke them — deferred to `_impl`.
  B3. Prior art (experiments/models/adapter_hotswap_latency/) already verified
      Theorems 1+2 on Qwen3-0.6B. Gemma 4 E4B transfer is an `_impl` deliverable.

No MLX, no numpy, no model loads. json + pathlib only.
"""

import json
from pathlib import Path


KILL_CRITERIA = [
    {
        "id": "K1909",
        "text": "Adapter hot-swap latency > 100ms (noticeable to user)",
        "result": "untested",
        "note": (
            "operational: t_attach_median over 20 runs on Gemma 4 E4B. "
            "Predicted [0.6, 2.7] ms per Theorem 1 (MATH.md §3). "
            "Blocker: B1/B2/B3 — execution deferred to _impl."
        ),
    },
    {
        "id": "K1910",
        "text": "Hot-swap during generation produces > 1 token glitch",
        "result": "untested",
        "note": (
            "operational: glitch-count = Σ_k |{i : T_0[i] ≠ T_swap(k)[i]}| "
            "for k in {1,2,4,8}, same-adapter detach/re-attach. Predicted 0 "
            "per Theorem 2 (MATH.md §4). Blocker: B1/B2/B3 — execution deferred."
        ),
    },
]


def main() -> int:
    out = Path(__file__).parent / "results.json"
    payload = {
        "experiment_id": "exp_pierre_adapter_hotswap_latency",
        "verdict": "PROVISIONAL",
        "status_note": (
            "design-lock scaffold; execution deferred to "
            "exp_pierre_adapter_hotswap_latency_impl"
        ),
        "all_pass": None,
        "is_smoke": False,
        "kill_criteria": KILL_CRITERIA,
        "blockers": [
            "B1: K1910 'token glitch' needed operational definition in MATH.md "
            "before first run (PLAN.md §1 KC-lock).",
            "B2: /mlx-dev + /fast-mlx not invoked this iteration; required by "
            "PLAN.md §1012 before MLX platform code.",
            "B3: Prior art adapter_hotswap_latency already verified Theorems 1+2 "
            "on Qwen3-0.6B; Gemma 4 E4B transfer is the _impl deliverable.",
        ],
        "theorem_reuse": {
            "T1_source": "experiments/models/adapter_hotswap_latency/MATH.md §Theorem 1",
            "T2_source": "experiments/models/adapter_hotswap_latency/MATH.md §Theorem 2",
            "T1_predicted_attach_ms": [0.6, 2.7],
            "T2_predicted_glitch_count": 0,
        },
        "hygiene_fixes_applied": {
            "references": [
                "adapter_hotswap_latency (in-repo prior art)",
                "Finding #388", "Finding #275", "Finding #627", "Finding #562",
                "Finding #666", "Hu et al. arxiv:2106.09685",
            ],
            "success_criteria": (
                "t_attach_median < 100ms AND glitch-count = 0 → SUPPORTED; "
                "else KILLED."
            ),
            "platform": "mlx",
            "experiment_dir": "experiments/models/exp_pierre_adapter_hotswap_latency/",
        },
        "f666_routing": (
            "Both KCs are target-metrics (user-perceived latency + behavioral "
            "output-equivalence). F#666 applies as structural check, not "
            "preempt-kill reason. This is NOT an F#666-pure preempt-kill case."
        ),
    }
    out.write_text(json.dumps(payload, indent=2))
    print("[PROVISIONAL] design-lock scaffold: see MATH.md §7, results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
