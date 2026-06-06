"""PREEMPT-KILL stub for exp_routing_adapter_signature.

Structural-KILL per MATH.md Theorem 1 (F#666-pure 24th drain-window instance,
triple-fire with F#715 bucket 6th + F#706/F#707/F#710-lineage routing-accuracy-as-proxy).
No code runs; experiment is de-registered at proof layer, not the compute layer.
"""
import json
import pathlib

OUT = pathlib.Path(__file__).parent / "results.json"


def main() -> None:
    OUT.write_text(
        json.dumps(
            {
                "verdict": "killed",
                "reason": (
                    "F#666-pure 24th drain-window structural-KILL: both KCs "
                    "(K1902 routing-accuracy, K1903 wall-clock per-adapter) are "
                    "proxy-only; zero target behavioral KC. Triple-fire: "
                    "F#666-pure + F#715 infrastructure-benchmark bucket 6th "
                    "(wall-clock sub-flavor) + F#706/F#707/F#710-lineage "
                    "routing-accuracy-as-proxy 3rd explicit instance."
                ),
                "kill_criteria": {
                    "1902": "fail (structural — proxy-only, no target KC)",
                    "1903": "fail (structural — F#715 wall-clock sub-flavor)",
                },
                "all_pass": False,
                "is_smoke": False,
                "platform": "n/a (preempt-structural, no compute)",
            },
            indent=2,
        )
    )
    print("preempt-KILL: F#666-pure 24th + F#715 bucket 6th + F#706/F#707/F#710-lineage")


if __name__ == "__main__":
    main()
