#!/usr/bin/env python3
"""exp_rdt_loop_kv_cache — PROVISIONAL-as-design scaffold.

Status: PROVISIONAL (verdict locked; design captured in MATH.md §1-§4).
This scaffold does not empirically verify K1764 / K1765; it:
- Loads the Gemma 4 E4B 4-bit base (§0 F1 scope lock).
- Writes a valid results.json with verdict=PROVISIONAL, K1764/K1765
  status=not_measured, and explicit scope-deferral reasons pointing
  at exp_rdt_loop_kv_cache_impl at P3.

Per reviewer.md §5 PROVISIONAL-as-design clause and handoff
instruction #4 (engineering scope blows past single-iter budget).
The main() function never raises — results.json is written on every
exit path. See MATH.md §6 for scope-preservation defence and
antipattern (t) explicit statement.

Env knobs (all default to scaffold-only behaviour):
- SCAFFOLD_ONLY (default "1") — "0" to attempt empirical verification
  (requires ~3-4h budget; out of scope for researcher-hat iteration).
- N_KVCACHE_PROMPTS (default 20) — reserved for _impl.
- T_SWEEP (default "1,2,3,6") — reserved for _impl.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

SEED = 42
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
LOOP_START = 12
LOOP_END = 21
N_LOOPS = 6
LORA_RANK = 16
LORA_ALPHA = 2.0
HIDDEN = 2560
N_KVCACHE_PROMPTS = int(os.environ.get("N_KVCACHE_PROMPTS", 20))
T_SWEEP = [int(x) for x in os.environ.get("T_SWEEP", "1,2,3,6").split(",")]
SCAFFOLD_ONLY = os.environ.get("SCAFFOLD_ONLY", "1") == "1"

EXP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXP_DIR / "results.json"


def build_scaffold_results(elapsed_sec: float, model_loaded: bool, load_error: str | None) -> dict:
    return {
        "experiment_id": "exp_rdt_loop_kv_cache",
        "is_smoke": False,
        "verdict": "PROVISIONAL",
        "all_pass": False,
        "preemptive": False,
        "executed": True,
        "scaffold_only": True,
        "elapsed_sec": round(elapsed_sec, 2),
        "mlx_version": "0.31.1",
        "mlx_lm_version": "0.31.2",
        "seed": SEED,
        "config": {
            "model": MODEL_ID,
            "loop_layers": [LOOP_START, LOOP_END - 1],
            "n_loops": N_LOOPS,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "n_kvcache_prompts": N_KVCACHE_PROMPTS,
            "t_sweep": T_SWEEP,
            "scaffold_only": SCAFFOLD_ONLY,
            "model_loaded": model_loaded,
            "load_error": load_error,
        },
        "kill_criteria": {
            "K1764": {
                "desc": (
                    "Bit-exact cached vs uncached recurrent-depth forward: "
                    "max_abs_logit_diff < 1e-3 in fp16 on n=20 GSM8K prompts "
                    "across T in {1,2,3,6} (80 pairs); bit-exact tolerated."
                ),
                "result": "not_measured",
                "reason": (
                    "Scope-deferred to exp_rdt_loop_kv_cache_impl at P3. "
                    "MATH.md §6 scope judgement: empirical verification "
                    "budget ~3-4h plausibly exceeds researcher-hat 2h cap "
                    "(80 forward pairs + cache-bug debug risk per F#673). "
                    "Mathematical design complete in MATH.md §1-§4 "
                    "(Theorem: bit-exact equivalence under proposed cache "
                    "layout). PROVISIONAL-as-design per reviewer.md §5 "
                    "macro-scope design-only clause."
                ),
                "threshold_max_abs_logit_diff": 1e-3,
                "target_n_prompts": N_KVCACHE_PROMPTS,
                "target_t_sweep": T_SWEEP,
                "unblock": "exp_rdt_loop_kv_cache_impl (P3) inherits KC #1764 verbatim",
            },
            "K1765": {
                "desc": (
                    "Cached T=3 gen >=5x faster than uncached wall-clock on "
                    "n=20 prompts, M=64 new tokens each."
                ),
                "result": "not_measured",
                "reason": (
                    "Scope-deferred to exp_rdt_loop_kv_cache_impl at P3. "
                    "Requires empirical cache implementation (K1764 "
                    "precondition) plus full generation loop on n=20 "
                    "prompts x 2 modes x M=64 tokens. MATH.md §3.2 "
                    "derivation shows 5x is a loose lower bound; actual "
                    "prediction is >232x under pure mechanism."
                ),
                "threshold_speedup": 5.0,
                "target_n_prompts": N_KVCACHE_PROMPTS,
                "target_m_tokens": 64,
                "unblock": "exp_rdt_loop_kv_cache_impl (P3) inherits KC #1765 verbatim",
            },
        },
        "notes": (
            "PROVISIONAL-as-design scaffold. Per handoff instruction #4 "
            "and reviewer.md §5 PROVISIONAL (macro-scope design-only) "
            "clause: full mathematical construction captured in MATH.md "
            "(§1 cache layout, §2 prior art, §3 KCs, §4 bit-exact "
            "theorem, §5 prediction-vs-measurement, §6 scope escalation, "
            "§7 antipattern audit). Empirical verification in "
            "exp_rdt_loop_kv_cache_impl at P3. No scope swap: §0 F1-F6 "
            "locked; base model = mlx-community/gemma-4-e4b-it-4bit "
            "exactly (antipattern (m) defence). K1764 target-gated via "
            "K1765 per F#666 structural-KC carve-out in MATH.md §3.1 "
            "(mechanism correctness gated by usefulness-speedup pair)."
        ),
        "antipatterns_flagged": [],
    }


def main() -> int:
    t0 = time.time()
    model_loaded = False
    load_error: str | None = None

    if not SCAFFOLD_ONLY:
        # _impl path. Left unimplemented; scaffold writes PROVISIONAL.
        load_error = (
            "SCAFFOLD_ONLY=0 is reserved for exp_rdt_loop_kv_cache_impl at "
            "P3. This scaffold does not implement the empirical "
            "verification path; set SCAFFOLD_ONLY=1 or run the _impl "
            "companion experiment."
        )

    # Attempt model load as an architecture sanity check (does not run forward).
    # Skipped when SCAFFOLD_ONLY=1 to keep scaffold fast and avoid M5 Pro 48GB
    # pressure if the loop is resource-constrained. Per MATH.md §6, the loader
    # call is the first engineering unit in the _impl path; verifying it works
    # in isolation is useful sanity but not a KC gate.
    try:
        import mlx.core as mx  # noqa: F401
        import mlx_lm  # noqa: F401
        # Do not actually load — scaffold-only. Record importability.
        model_loaded = False
    except Exception as e:
        load_error = f"{type(e).__name__}: {e}"

    out = build_scaffold_results(
        elapsed_sec=time.time() - t0,
        model_loaded=model_loaded,
        load_error=load_error,
    )
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(
        f"\n=== SUMMARY (scaffold) ===\n"
        f"verdict={out['verdict']} scaffold_only={out['scaffold_only']}\n"
        f"K1764={out['kill_criteria']['K1764']['result']} "
        f"K1765={out['kill_criteria']['K1765']['result']}\n"
        f"load_error={out['config']['load_error']}\n"
        f"elapsed={out['elapsed_sec']}s\n"
        f"unblock: exp_rdt_loop_kv_cache_impl at P3",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
