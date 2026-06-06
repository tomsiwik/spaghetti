#!/usr/bin/env python3
"""Re-run ONLY the routing evaluation phase with train/val split.

This script loads the existing results.json, re-runs phase_routing_heads
with the updated train/val split code, and patches the routing results
back into results.json without touching any other fields.

Platform: Apple M5 Pro 48GB, MLX
"""

import json
import sys
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Import the full experiment module to reuse its functions
sys.path.insert(0, str(EXPERIMENT_DIR))
from run_experiment import (
    phase_routing_heads,
    DATA_DIR,
    log,
)


def main():
    t0 = time.time()

    # Load existing results
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    active_domains = results["active_domains"]
    log(f"Re-running routing with train/val split for {len(active_domains)} domains...")

    # Build domain_data mapping (same as main experiment)
    domain_data = {}
    for d in active_domains:
        ddir = DATA_DIR / d
        if ddir.exists():
            domain_data[d] = ddir

    # Run updated routing phase
    head_results = phase_routing_heads(domain_data, active_domains)

    # Patch into results
    results["routing_heads"] = head_results

    # Recompute S2
    avg_routing_acc = (
        sum(h["accuracy"] for h in head_results.values()) / len(head_results)
        if head_results else 0
    )
    n_routing_pass = sum(1 for h in head_results.values() if h["accuracy"] > 0.7)
    s2_pass = avg_routing_acc > 0.7

    # Also compute train-only average for comparison
    avg_train_acc = (
        sum(h["train_accuracy"] for h in head_results.values()) / len(head_results)
        if head_results else 0
    )
    avg_val_acc = (
        sum(h.get("val_accuracy", h["train_accuracy"]) for h in head_results.values())
        / len(head_results)
        if head_results else 0
    )

    results["s2_pass"] = s2_pass
    results["avg_routing_accuracy"] = round(avg_routing_acc, 4)
    results["avg_routing_train_accuracy"] = round(avg_train_acc, 4)
    results["avg_routing_val_accuracy"] = round(avg_val_acc, 4)
    results["n_routing_above_70"] = n_routing_pass

    # Re-check overall verdict (should not change)
    all_kills_pass = results["k1_pass"] and results["k2_pass"] and results["k3_pass"]
    results["all_kill_criteria_pass"] = all_kills_pass
    if not all_kills_pass:
        results["verdict"] = "KILLED"
    elif results["s1_pass"] and s2_pass and results["s3_pass"]:
        results["verdict"] = "SUPPORTED"
    else:
        results["verdict"] = "SUPPORTED (partial)"

    # Save
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    elapsed = time.time() - t0
    log(f"\nRouting re-run complete in {elapsed:.1f}s")
    log(f"  Train accuracy avg: {avg_train_acc:.3f}")
    log(f"  Val accuracy avg:   {avg_val_acc:.3f}")
    log(f"  S2 pass: {s2_pass}")
    log(f"  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
