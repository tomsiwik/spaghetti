"""Graceful-failure stub for preempt-structural KILL (F#666-pure triple-fire).

Per MATH.md L1-L5, compute is inadmissible:
- K1864 and K1865 are both PPL-only (proxy-only) with no target-metric pair.
- §5 tautological-inter-variant-delta fires on both KCs (no per-variant base-anchor).
- Hygiene-multi-defect (3 defects) fires but F#702 patch path unavailable (0 target KCs).

No MLX load. No adapter mount. No PPL measurement. Stub writes results.json
recording the preempt verdict and exits cleanly.
"""

import json
from pathlib import Path

EXPERIMENT_ID = "exp_g4_adapter_svd_denoise"
OUT_DIR = Path(__file__).parent


def main() -> None:
    results = {
        "experiment_id": EXPERIMENT_ID,
        "verdict": "KILLED",
        "is_smoke": False,
        "preempt_structural": True,
        "all_pass": False,
        "kill_criteria": [
            {
                "id": 1864,
                "text": (
                    "SVD-truncated adapter PPL > original adapter PPL + 0.05 "
                    "(truncation removes signal)"
                ),
                "result": "preempted",
                "reason": (
                    "F#666-pure: PPL-only proxy with no target-metric pair; "
                    "§5 inter-variant delta without per-variant base-anchor."
                ),
            },
            {
                "id": 1865,
                "text": (
                    "Truncated adapter composition PPL not improved vs untruncated "
                    "composition"
                ),
                "result": "preempted",
                "reason": (
                    "F#666-pure: PPL-only proxy with no target-metric pair; "
                    "§5 inter-composition delta without per-variant base-anchor; "
                    "degenerate-equivalence under F#477 regime trivially satisfies."
                ),
            },
        ],
        "antipatterns_fired": [
            {
                "pattern": "F#666-pure-standalone",
                "role": "primary",
                "instance_index": 12,
                "bucket": "PPL",
                "bucket_instance_index": 3,
            },
            {
                "pattern": "tautological-inter-variant-delta (§5, F#709)",
                "role": "secondary",
                "instance_index": 6,
                "sub_variant": "intra-adapter-rank-delta",
                "sub_variant_instance_index": 2,
            },
            {
                "pattern": "hygiene-multi-defect (F#703)",
                "role": "tertiary",
                "defect_count": 3,
                "defects": ["success_criteria=[]", "platform=~", "references=[]"],
                "f702_patch_available": False,
                "f702_unavailability_instance_index": 3,
            },
        ],
        "fire_mode": "triple",
        "precedent": "2nd triple-fire (1st=F#714); PPL bucket saturates at 3-instance",
        "no_model_loaded": True,
        "no_adapter_mounted": True,
        "no_ppl_measured": True,
    }

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[{EXPERIMENT_ID}] preempt-structural KILL stub written.")


if __name__ == "__main__":
    main()
