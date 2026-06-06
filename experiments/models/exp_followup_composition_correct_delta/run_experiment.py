"""
Pre-flight NOT satisfied. Preemptive kill — see PAPER.md.

This module is intentionally a no-op kill-reporter. The experiment's
K1548 requires 5 trained LoRA adapters (math/code/medical/legal/finance on
Gemma 4 E4B 4bit per adapters/registry.json). Filesystem audit confirms all
5 referenced adapter directories contain only adapter_config.json — no
adapters.safetensors weight file in any of them. This is antipattern-017
(weight-less stub adapters), now at 3 confirmed instances in 2 days.

Any MLX load of a stub directory either errors on the missing weight file
or silently runs the base model, in which case the composition under test
would be identically the base model — K1548 would measure nothing.

The CORRECT unblock is to retrain the 5 domain adapters
(`P11.ADAPTER-REBUILD`), then run a v2 of this experiment as a ~15-minute
PPL measurement over held-out text.

Reviewer pre-flight grep (reproduces the audit):

    find experiments/models/exp_p1_t2_single_domain_training/adapters \
         experiments/models/exp_p1_t2_multi_domain_5/adapters \
         -maxdepth 2 -type f

Expected 5 `adapters.safetensors`; observed 0.
"""

import json
import os
import time

RESULT = {
    "verdict": "KILLED",
    "verdict_reason": "preemptive: antipattern-017 — all 5 required adapter weight files missing",
    "all_pass": False,
    "kill_criteria": {
        "K1548": {
            "text": (
                "At N=5 with explicit Sum(B_i @ A_i) composition, held-out PPL "
                "stays within 2x of solo baseline (no catastrophe)"
            ),
            "result": "fail",
            "reason": "unmeasurable — no trained LoRA weights for any of 5 registry adapters",
        }
    },
    "required_adapters": [
        "experiments/models/exp_p1_t2_single_domain_training/adapters/math/",
        "experiments/models/exp_p1_t2_single_domain_training/adapters/code/",
        "experiments/models/exp_p1_t2_single_domain_training/adapters/medical/",
        "experiments/models/exp_p1_t2_multi_domain_5/adapters/legal/",
        "experiments/models/exp_p1_t2_multi_domain_5/adapters/finance/",
    ],
    "unblock_path": "P11.ADAPTER-REBUILD",
    "antipatterns_triggered": ["antipattern-017"],
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}


def main():
    here = os.path.dirname(__file__)
    out = os.path.join(here, "results.json")
    with open(out, "w") as f:
        json.dump(RESULT, f, indent=2)
    print(json.dumps(RESULT, indent=2))


if __name__ == "__main__":
    main()
