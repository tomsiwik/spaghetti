"""K=2 strategy × domain composition — Pierre's headline product config.

Tests three pairs sequentially:
  - strategy_full + domain_math    (eval emphasis: GSM8K)
  - strategy_full + domain_code    (eval emphasis: HumanEval)
  - strategy_full + domain_medical (eval emphasis: MedQA)

Each pair runs the standard 4-method matrix at K=2.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))

from _pierre_shared.eval_runner import (  # type: ignore  # noqa: E402
    MethodSpec, run_pierre_compose_experiment, compose_fisher_rao,
)


def main():
    pairs = [
        ("strategy_full", "domain_math"),
        ("strategy_full", "domain_code"),
        ("strategy_full", "domain_medical"),
    ]
    aggregated = {"per_pair": {}, "config": {"pairs": pairs, "K": 2}}

    for s, d in pairs:
        pair_label = f"{s}+{d}"
        out_path = EXP_DIR / f"results_{s}_{d}.json"
        print(f"\n{'='*60}\n  PAIR: {pair_label}\n{'='*60}")
        run_pierre_compose_experiment(
            method=MethodSpec(
                name=f"fisher_rao_K2_{pair_label}",
                kind="b_only",
                fn=compose_fisher_rao,
                fn_kwargs={},
            ),
            kc_thresholds={
                "k1_min_delta_over_fisher_rao": 0.0,  # this IS Fisher-Rao; K1 trivially passes
                "k2_max_delta_under_full_delta_dare": 4.0,
                "k3_max_preprocess_seconds": 5.0,
                "k4_label": f"K=2 pair {pair_label}",
                "k4_value": None,
                "k4_threshold": None,
            },
            out_path=out_path,
            extra_config={"pair": pair_label, "K": 2},
            adapter_names_override=[s, d],
        )
        # Capture per-pair result
        if out_path.exists():
            aggregated["per_pair"][pair_label] = json.loads(out_path.read_text())

    # Aggregate across pairs
    fr_avgs = []
    for pair_label, res in aggregated["per_pair"].items():
        if "methods" in res and "fisher_rao" in res["methods"]:
            fr_avgs.append(res["methods"]["fisher_rao"]["avg"])
    if fr_avgs:
        aggregated["fisher_rao_K2_avg_across_pairs"] = sum(fr_avgs) / len(fr_avgs)
        aggregated["fisher_rao_K2_avgs"] = fr_avgs

    (EXP_DIR / "results.json").write_text(json.dumps(aggregated, indent=2, default=str))
    print(f"\n=== K=2 strategy×domain composition complete ===")
    print(f"  Aggregate Fisher-Rao avg across {len(fr_avgs)} pairs: "
          f"{aggregated.get('fisher_rao_K2_avg_across_pairs', 'N/A')}")


if __name__ == "__main__":
    main()
