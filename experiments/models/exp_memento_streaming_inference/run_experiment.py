"""run_experiment.py — exp_memento_streaming_inference (PREEMPT-KILL placeholder)

This experiment is preempt-killed per Finding #669 before any code path executes.
Parent exp_memento_gemma4_replication is PROVISIONAL (F#685) — no Gemma-4-MEMENTO
checkpoint exists, MEMENTO 2-stage SFT + block-mask attention not executable via
mlx_lm.lora CLI. Both KCs require a callable Gemma-4-MEMENTO forward pass under a
streaming-during-inference regime that does not exist on this platform.

This file exists for filesystem-conformance only (reviewer.md §1 required-artifacts
checklist). It MUST NOT be executed; running it raises SystemExit immediately.

See MATH.md §1 for the preempt theorem, §4 for the unblock condition.
"""

import sys


def main() -> int:
    print(
        "PREEMPT-KILL: exp_memento_streaming_inference is structurally blocked per F#669. "
        "Parent exp_memento_gemma4_replication is PROVISIONAL (F#685). "
        "K1939 (streaming-vs-batch task-accuracy parity) and K1940 (per-block streaming "
        "latency on M5 Pro) require a callable Gemma-4-MEMENTO forward pass with an "
        "inline streaming-mode path that does not exist. See MATH.md §4 for the unblock "
        "condition.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
