"""Uniform-routing ablation: compare learned router vs w_g = 1/G.

Tests whether Level 1 (learned group routing) adds value over simple uniform
weighting at micro scale. If uniform matches learned, the router contributes
nothing and should be acknowledged.

Runs 3-seed comparison on both single-domain and multi-domain settings.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.arena import run_single, run_multidomain, leaderboard


def run_ablation():
    seeds = [42, 123, 7]
    steps = 500
    md_steps = 300

    results = {"single": {}, "multidomain": {}}

    for mode_name, runner, step_key, step_val in [
        ("single", run_single, "steps", steps),
        ("multidomain", run_multidomain, "steps_per_domain", md_steps),
    ]:
        for model_name in ["capsule_moe", "capsule_moe_uniform"]:
            seed_results = []
            for seed in seeds:
                print(f"\n{'='*60}")
                print(f"  {mode_name} | {model_name} | seed={seed}")
                print(f"{'='*60}")
                r = runner(model_name, seed=seed, **{step_key: step_val})
                seed_results.append(r)
                print(f"  -> val_loss={r.val_loss}, final_loss={r.final_loss}")

            avg_val = sum(r.val_loss or r.final_loss for r in seed_results) / len(seed_results)
            avg_final = sum(r.final_loss for r in seed_results) / len(seed_results)

            results[mode_name][model_name] = {
                "seeds": [r.to_dict() for r in seed_results],
                "avg_val_loss": avg_val,
                "avg_final_loss": avg_final,
            }
            print(f"\n  {model_name} avg: val={avg_val:.4f}, final={avg_final:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("UNIFORM-ROUTING ABLATION RESULTS")
    print("=" * 60)

    for mode in ["single", "multidomain"]:
        print(f"\n--- {mode} ---")
        learned = results[mode]["capsule_moe"]
        uniform = results[mode]["capsule_moe_uniform"]
        delta = uniform["avg_val_loss"] - learned["avg_val_loss"]
        pct = delta / learned["avg_val_loss"] * 100
        print(f"  capsule_moe (learned): val={learned['avg_val_loss']:.4f}")
        print(f"  capsule_moe_uniform:   val={uniform['avg_val_loss']:.4f}")
        print(f"  delta: {delta:+.4f} ({pct:+.1f}%)")
        if abs(pct) < 2.0:
            print(f"  -> CONCLUSION: Level 1 routing adds NO significant value at this scale")
        elif pct > 2.0:
            print(f"  -> CONCLUSION: Learned routing HELPS (uniform degrades by {pct:.1f}%)")
        else:
            print(f"  -> CONCLUSION: Uniform routing is BETTER by {-pct:.1f}% (router overhead)")

    # Save raw results
    out_path = Path(__file__).parent / "ablation_uniform_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    run_ablation()
